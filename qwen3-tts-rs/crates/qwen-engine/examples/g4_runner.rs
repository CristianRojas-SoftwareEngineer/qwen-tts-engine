use std::env;
use std::fs;
use std::path::Path;

use qwen_core::config::{lookup_language, lookup_speaker, SAMPLE_RATE};
use qwen_core::{MarkupPlanner, QVoiceFile};
use qwen_engine::{
    apply_emotion_direction, attach_emo_ref, build_voice_from_ref_audio, encode_wav_bytes,
    load_voice_from_file, prepare_markup_06b, render_spans_buffer, resolve_emovoice_06b,
    synthesize_to_samples, validate_wav_path_or_header, EmoVoice06BAction, Model06B,
    ModelLoadOptions, RequestConfig, RequestExecution, SaveVoiceOptions, SessionScratch,
    VoiceBindings, VoiceLoadOptions,
};
use qwen_kernels::ComputeRuntime;

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut model_dir = String::new();
    let mut text: Option<String> = None;
    let mut out_path = "output.wav".to_string();
    let mut speaker = "ryan".to_string();
    let mut language = "English".to_string();
    let mut threads: Option<usize> = None;
    let mut user_seed: Option<u32> = None;
    let mut user_temp: Option<f32> = None;
    let mut opts = ModelLoadOptions::default();
    let mut req = RequestConfig::default();

    let mut ref_audio: Option<String> = None;
    let mut ref_text: Option<String> = None;
    let mut save_voice: Option<String> = None;
    let mut load_voice: Option<String> = None;
    let mut target_cv: Option<String> = None;
    let mut xvector_only = false;
    let mut icl_only = false;
    let mut graft = false;
    let mut icl_frames = 0usize;
    let mut voice_strength = 1.0f32;
    let mut emo_ref: Option<String> = None;
    let mut emo_ref_text: Option<String> = None;
    let mut emotion: Option<String> = None;
    let mut emotion_strength = 0.25f32;
    let mut no_compose = false;
    let mut compose_pause = 0.12f32;

    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "-d" | "--model-dir" => {
                i += 1;
                model_dir = args[i].clone();
            }
            "-t" | "--text" => {
                i += 1;
                text = Some(args[i].clone());
            }
            "-o" | "--output" => {
                i += 1;
                out_path = args[i].clone();
            }
            "-s" | "--speaker" => {
                i += 1;
                speaker = args[i].clone();
            }
            "-l" | "--language" => {
                i += 1;
                language = args[i].clone();
            }
            "-j" | "--threads" => {
                i += 1;
                let t = args[i].parse::<usize>().unwrap_or(0);
                if t > 0 {
                    threads = Some(t);
                }
            }
            "--seed" => {
                i += 1;
                user_seed = Some(args[i].parse::<u32>().unwrap_or(42));
            }
            "-T" | "--temperature" => {
                i += 1;
                user_temp = Some(args[i].parse::<f32>().unwrap_or(0.9));
            }
            "--int8" => opts.use_int8 = true,
            "--int4" => opts.use_int4 = true,
            "--quant-mixed" => opts.quant_mixed = true,
            "--silent" | "-S" => {
                opts.silent = true;
                req.silent = true;
            }
            "--debug" | "-D" => {
                opts.debug = true;
                req.debug = true;
            }
            "--ref-audio" => {
                i += 1;
                ref_audio = Some(args[i].clone());
            }
            "--ref-text" => {
                i += 1;
                ref_text = Some(args[i].clone());
            }
            "--save-voice" => {
                i += 1;
                save_voice = Some(args[i].clone());
            }
            "--load-voice" => {
                i += 1;
                load_voice = Some(args[i].clone());
            }
            "--target-cv" => {
                i += 1;
                target_cv = Some(args[i].clone());
            }
            "--xvector-only" => xvector_only = true,
            "--icl-only" => icl_only = true,
            "--graft" => {
                graft = true;
                icl_only = true;
            }
            "--icl-frames" => {
                i += 1;
                icl_frames = args[i].parse::<usize>().unwrap_or(0);
            }
            "--voice-strength" => {
                i += 1;
                voice_strength = args[i].parse::<f32>().unwrap_or(1.0);
            }
            "--emo-ref" => {
                i += 1;
                emo_ref = Some(args[i].clone());
            }
            "--emo-ref-text" => {
                i += 1;
                emo_ref_text = Some(args[i].clone());
            }
            "--emotion" => {
                i += 1;
                emotion = Some(args[i].clone());
            }
            "--emotion-strength" => {
                i += 1;
                emotion_strength = args[i].parse::<f32>().unwrap_or(0.25);
            }
            "--roughness" => {
                i += 1;
                req.cp_roughness = args[i].parse::<f32>().unwrap_or(0.0).clamp(0.0, 1.0);
            }
            "--no-compose" => no_compose = true,
            "--compose-pause" => {
                i += 1;
                compose_pause = args[i].parse::<f32>().unwrap_or(0.12);
            }
            other => {
                eprintln!("Unknown arg: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    if model_dir.is_empty() {
        eprintln!("Missing -d");
        std::process::exit(1);
    }
    if let Some(ref p) = ref_audio {
        if let Err(e) = validate_wav_path_or_header(Path::new(p)) {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }

    let spk = lookup_speaker(&speaker).unwrap_or_else(|| {
        eprintln!("Unknown speaker: {speaker}");
        std::process::exit(1);
    });
    let lang_id = lookup_language(&language).unwrap_or_else(|| {
        eprintln!("Unknown language: {language}");
        std::process::exit(1);
    });
    req.speaker_id = spk.id;
    req.language_id = Some(lang_id);

    let runtime = ComputeRuntime::new(threads);
    let mut model = Model06B::load(Path::new(&model_dir), &runtime, opts).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });

    let mut dir_to_apply = None;
    if let Some(ref emo_spec) = emotion {
        match resolve_emovoice_06b(
            emo_spec,
            load_voice.as_deref().map(Path::new),
            load_voice.is_some(),
            Some(&speaker),
            &[],
        ) {
            Ok(EmoVoice06BAction::LoadQVoiceIclOnly(p)) => {
                load_voice = Some(p.to_string_lossy().into_owned());
                icl_only = true;
            }
            Ok(EmoVoice06BAction::LoadBinXVector(p)) => {
                load_voice = Some(p.to_string_lossy().into_owned());
                xvector_only = true;
            }
            Ok(EmoVoice06BAction::ApplyDirection { dir_path, .. }) => {
                dir_to_apply = Some(dir_path);
            }
            Err(e) => {
                eprintln!("{e}");
                std::process::exit(1);
            }
        }
    }

    let mut voice = if let Some(ref lv) = load_voice {
        let vopts = VoiceLoadOptions {
            icl_only,
            xvector_only,
            graft_mode: graft,
            icl_frames_cap: icl_frames,
            voice_strength,
            vs_l0: 11,
            vs_l1: 27,
            silent: opts.silent,
        };
        let (v, meta) = load_voice_from_file(
            &runtime,
            &mut model,
            Path::new(lv),
            ref_text.as_deref(),
            &vopts,
        )
        .unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(1);
        });
        let _ = meta;
        let _ = QVoiceFile::validate_header;
        v
    } else if let Some(ref ra) = ref_audio {
        let sopts = SaveVoiceOptions {
            ref_text: ref_text.as_deref(),
            xvector_only,
            graft_mode: graft,
            icl_frames_cap: icl_frames,
            max_ref_seconds: 30.0,
            language: Some(&language),
            voice_name: None,
            target_cv_dir: target_cv.as_deref().map(Path::new),
            silent: opts.silent,
            debug: opts.debug,
        };
        build_voice_from_ref_audio(
            &runtime,
            &model,
            Path::new(ra),
            save_voice.as_deref().map(Path::new),
            &sopts,
        )
        .unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(1);
        })
    } else {
        VoiceBindings::preset(spk.id)
    };

    if let Some(dir_path) = dir_to_apply {
        match &mut voice.identity {
            qwen_engine::VoiceIdentity::XVector(emb)
            | qwen_engine::VoiceIdentity::IclGraft {
                speaker_embedding: emb,
                ..
            } => {
                apply_emotion_direction(emb, &dir_path, emotion_strength).unwrap_or_else(|e| {
                    eprintln!("{e}");
                    std::process::exit(1);
                });
            }
            _ => {}
        }
    }

    if let Some(ref er) = emo_ref {
        let ert = emo_ref_text.as_deref().unwrap_or("");
        attach_emo_ref(
            &runtime,
            &model,
            &mut voice,
            Path::new(er),
            ert,
            opts.silent,
            opts.debug,
        )
        .unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(1);
        });
    }

    let Some(raw_text) = text else {
        return;
    };

    let prep = prepare_markup_06b(&raw_text, &voice, no_compose, user_seed, user_temp, 0.9);
    req.seed = prep.seed;
    req.temperature = prep.temperature;

    let mut scratch = SessionScratch::new();
    let exec = RequestExecution::new(req);

    let samples = if prep.use_compose {
        let spans = MarkupPlanner::parse_spans(&prep.text);
        let (s, _) = render_spans_buffer(
            &runtime,
            &model,
            &voice,
            &mut scratch,
            &exec,
            &spans,
            compose_pause,
        )
        .unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(1);
        });
        s
    } else {
        let (s, _) =
            synthesize_to_samples(&runtime, &model, &voice, &mut scratch, &exec, &prep.text)
                .unwrap_or_else(|e| {
                    eprintln!("{e}");
                    std::process::exit(1);
                });
        s
    };

    let wav = encode_wav_bytes(&samples, SAMPLE_RATE as u32);
    fs::write(&out_path, wav).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
}
