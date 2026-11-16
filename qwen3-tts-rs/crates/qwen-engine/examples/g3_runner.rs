use std::env;
use std::fs;
use std::path::Path;

use qwen_core::config::{lookup_language, lookup_speaker, NUM_CODEBOOKS, SAMPLE_RATE};
use qwen_engine::{
    encode_wav_bytes, speech_decoder_decode, synthesize_to_samples, synthesize_to_sink,
    write_wav, Model06B, ModelLoadOptions, RequestConfig, RequestExecution, SessionScratch,
    VoiceBindings, WavStreamSink,
};
use qwen_kernels::ComputeRuntime;

fn run_decode_dump(codes_file: &str, model_dir: &str, out_wav: &str) -> Result<(), String> {
    let raw = fs::read_to_string(codes_file).map_err(|e| format!("read {codes_file}: {e}"))?;
    let mut codes = Vec::new();
    let mut n_frames = 0usize;
    for line in raw.lines() {
        let vals: Vec<i32> = line
            .split_whitespace()
            .filter_map(|s| s.parse::<i32>().ok())
            .collect();
        if vals.len() >= NUM_CODEBOOKS {
            codes.extend_from_slice(&vals[..NUM_CODEBOOKS]);
            n_frames += 1;
        }
    }
    if n_frames == 0 {
        return Err("No frames found in codes file".to_string());
    }
    let runtime = ComputeRuntime::new(None);
    let model = Model06B::load(
        Path::new(model_dir),
        &runtime,
        ModelLoadOptions {
            silent: true,
            ..Default::default()
        },
    )
    .map_err(|e| format!("load model: {e}"))?;

    let samples = speech_decoder_decode(&runtime, &model.decoder, &codes, n_frames)
        .map_err(|e| format!("decode: {e}"))?;
    write_wav(Path::new(out_wav), &samples, SAMPLE_RATE as u32)
        .map_err(|e| format!("write wav: {e}"))?;
    eprintln!("decode-dump: {n_frames} frames -> {} samples -> {out_wav}", samples.len());
    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.get(1).map(|s| s.as_str()) == Some("decode-dump") {
        if args.len() < 5 {
            eprintln!("Usage: g3_runner decode-dump <codes.txt> <model_dir> <out.wav>");
            std::process::exit(1);
        }
        if let Err(e) = run_decode_dump(&args[2], &args[3], &args[4]) {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
        return;
    }

    let mut model_dir = String::new();
    let mut text = String::new();
    let mut out_path = "output.wav".to_string();
    let mut speaker = "ryan".to_string();
    let mut language = "English".to_string();
    let mut threads: Option<usize> = None;
    let mut seed: u32 = 42;
    let mut opts = ModelLoadOptions::default();
    let mut req = RequestConfig::default();

    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "-d" | "--model-dir" => {
                i += 1;
                model_dir = args[i].clone();
            }
            "-t" | "--text" => {
                i += 1;
                text = args[i].clone();
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
                seed = args[i].parse::<u32>().unwrap_or(42);
            }
            "-T" | "--temperature" => {
                i += 1;
                req.temperature = args[i].parse::<f32>().unwrap_or(0.9);
            }
            "-k" | "--top-k" => {
                i += 1;
                req.top_k = args[i].parse::<i32>().unwrap_or(50);
            }
            "-p" | "--top-p" => {
                i += 1;
                req.top_p = args[i].parse::<f32>().unwrap_or(1.0);
            }
            "-r" | "--rep-penalty" => {
                i += 1;
                req.rep_penalty = args[i].parse::<f32>().unwrap_or(1.05);
            }
            "--int8" => opts.use_int8 = true,
            "--int4" => opts.use_int4 = true,
            "--quant-mixed" => opts.quant_mixed = true,
            "--stream" => req.stream = true,
            "--stream-chunk" => {
                i += 1;
                req.stream_chunk_frames = args[i].parse::<usize>().unwrap_or(10);
            }
            "--silent" | "-S" => {
                opts.silent = true;
                req.silent = true;
            }
            "--debug" | "-D" => {
                opts.debug = true;
                req.debug = true;
            }
            other => {
                eprintln!("Unknown arg: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    if model_dir.is_empty() || text.is_empty() {
        eprintln!("Missing -d or --text");
        std::process::exit(1);
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
    req.seed = seed;

    let runtime = ComputeRuntime::new(threads);
    let model = Model06B::load(Path::new(&model_dir), &runtime, opts).unwrap_or_else(|e| {
        eprintln!("Load error: {e}");
        std::process::exit(1);
    });
    let voice = VoiceBindings::preset(spk.id);
    let mut scratch = SessionScratch::new();
    let exec = RequestExecution::new(req.clone());

    if req.stream {
        let mut sink =
            WavStreamSink::create(Path::new(&out_path), SAMPLE_RATE as u32, req.volume)
                .unwrap_or_else(|e| {
                    eprintln!("Sink error: {e}");
                    std::process::exit(1);
                });
        if let Err(e) = synthesize_to_sink(
            &runtime,
            &model,
            &voice,
            &mut scratch,
            &exec,
            &text,
            &mut sink,
        ) {
            eprintln!("Synth error: {e}");
            std::process::exit(1);
        }
    } else {
        let (samples, _) =
            synthesize_to_samples(&runtime, &model, &voice, &mut scratch, &exec, &text)
                .unwrap_or_else(|e| {
                    eprintln!("Synth error: {e}");
                    std::process::exit(1);
                });
        let wav = encode_wav_bytes(&samples, SAMPLE_RATE as u32);
        fs::write(&out_path, wav).unwrap_or_else(|e| {
            eprintln!("Write error: {e}");
            std::process::exit(1);
        });
    }
}
