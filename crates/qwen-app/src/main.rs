// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

mod affinity;
mod cli;
mod server;

use std::fs;
use std::io::Read;
use std::path::Path;
use std::process;
use std::sync::Arc;
use std::thread;

use qwen_core::config::{lookup_language, lookup_speaker, FRAME_RATE, SAMPLE_RATE};
use qwen_core::{emotion_name_to_tok, MarkupPlanner, QVoiceFile};
use qwen_engine::{
    apply_emotion_direction, attach_emo_ref, batch_bench, batch_self_test,
    build_voice_from_ref_audio, generate_batch_longform, generate_batch_multi,
    load_voice_from_file, onset_fade, prepare_markup_06b, render_spans_buffer,
    resolve_emovoice_06b, split_text_for_batch, synthesize_to_samples, synthesize_to_sink,
    tail_glitch_score, tail_trim, time_stretch, validate_wav_path_or_header, write_wav, BatchReq,
    EmoVoice06BAction, Model06B, ModelLoadOptions, RequestConfig, RequestExecution,
    SaveVoiceOptions, SessionScratch, StdoutRawSink, VoiceBindings, VoiceIdentity,
    VoiceLoadOptions, WavStreamSink,
};
use qwen_kernels::bench::run_matmat_bench;
use qwen_kernels::selftest::run_kernel_selftest;
use qwen_kernels::ComputeRuntime;

use cli::{parse_cli, print_usage, CliArgs, ParseOutcome};
use server::validation::time_based_seed;
use server::{serve, ServeConfig};

fn spawn_stdin_watcher() -> Result<(), String> {
    thread::Builder::new()
        .name("qwen-watch-stdin".to_string())
        .spawn(|| {
            let mut stdin = std::io::stdin().lock();
            let mut buf = [0u8; 256];
            loop {
                match stdin.read(&mut buf) {
                    Ok(0) => process::exit(0),
                    Ok(_) => continue,
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(_) => process::exit(0),
                }
            }
        })
        .map(|_| ())
        .map_err(|e| format!("Error: --watch-stdin could not start the stdin watcher thread: {e}"))
}

fn print_qvoice_info(path: &Path) -> Result<(), String> {
    let meta_size = fs::metadata(path)
        .map_err(|e| format!("stat failed: {e}"))?
        .len();
    let qv = QVoiceFile::read_from_file(path, true).map_err(|e| e.to_string())?;
    let n_ref_frames = qv.ref_codes.len() / 16;
    let basename = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");

    print!(
        "  {:<30}  v{}  {:3} frames ({:.1}s ref)  {:5.1} KB",
        basename,
        qv.version,
        n_ref_frames,
        n_ref_frames as f32 / FRAME_RATE,
        meta_size as f32 / 1024.0
    );
    if let Some(m) = &qv.meta {
        if !m.voice_name.is_empty() {
            print!("  [{}]", m.voice_name);
        }
        if !m.lang_name.is_empty() {
            print!("  lang={}", m.lang_name);
        }
        let msz = if m.model_size >= 2048 {
            "1.7B"
        } else if m.model_size > 0 {
            "0.6B"
        } else {
            "?"
        };
        print!("  model={msz}");
    }
    if !qv.ref_text.is_empty() {
        let preview: String = qv.ref_text.chars().take(255).collect();
        print!("  \"{preview}\"");
    }
    println!();
    Ok(())
}

fn has_qvoice_ext(path: &Path) -> bool {
    path.extension()
        .and_then(|s| s.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("qvoice"))
}

fn list_voices(dir_str: &str) -> i32 {
    let p = Path::new(dir_str);
    if p.is_file() && has_qvoice_ext(p) {
        println!("Voice profiles:");
        if let Err(e) = print_qvoice_info(p) {
            eprintln!("Error: {dir_str} is not a valid .qvoice file ({e})");
            return 1;
        }
        return 0;
    }
    let Ok(entries) = fs::read_dir(p) else {
        eprintln!("Error: cannot open directory {dir_str}");
        return 1;
    };
    println!("Voice profiles in {dir_str}:\n");
    let mut paths: Vec<_> = entries
        .filter_map(|e| e.ok().map(|x| x.path()))
        .filter(|path| has_qvoice_ext(path))
        .collect();
    paths.sort();

    let mut count = 0usize;
    for path in paths {
        if print_qvoice_info(&path).is_ok() {
            count += 1;
        }
    }
    if count == 0 {
        println!("  (no .qvoice files found)");
    } else {
        println!("\n  {count} voice profile(s)");
    }
    0
}

fn delete_voice(path_str: &str) -> i32 {
    let path = Path::new(path_str);
    if !has_qvoice_ext(path) {
        eprintln!("Error: --delete-voice only works with .qvoice files");
        return 1;
    }
    if !path.is_file() {
        eprintln!("Error: file not found: {path_str}");
        return 1;
    }
    let mut magic = [0u8; 4];
    let valid = fs::File::open(path)
        .and_then(|mut f| f.read_exact(&mut magic))
        .is_ok()
        && &magic == b"QVCE";
    if !valid {
        eprintln!("Error: {path_str} is not a valid .qvoice file");
        return 1;
    }
    if let Err(e) = fs::remove_file(path) {
        eprintln!("Error: failed to delete {path_str}: {e}");
        return 1;
    }
    println!("Deleted {path_str}");
    0
}

fn try_run_auxiliary_cmd(cfg: &CliArgs) -> Option<i32> {
    if cfg.show_caps {
        let rt = ComputeRuntime::new(cfg.threads);
        print!("{}", rt.caps_report());
        return Some(0);
    }
    if cfg.self_test {
        let rt = ComputeRuntime::new(cfg.threads);
        let (failures, report) = run_kernel_selftest(&rt);
        print!("{report}");
        return Some(if failures == 0 { 0 } else { 1 });
    }
    if cfg.matmat_bench {
        let rt = ComputeRuntime::new(cfg.threads);
        let report = run_matmat_bench(&rt);
        print!("{report}");
        return Some(0);
    }
    if let Some(ref dir) = cfg.list_voices_dir {
        return Some(list_voices(dir));
    }
    if let Some(ref path) = cfg.delete_voice_path {
        return Some(delete_voice(path));
    }
    None
}

fn resolve_voice_and_emotion(
    cfg: &CliArgs,
    runtime: &ComputeRuntime,
    model: &mut Model06B,
    speaker_id: i32,
    cli_language_id: Option<i32>,
) -> Result<(VoiceBindings, Option<i32>), String> {
    // Resolver emoción (`0.6B emovoice`) fuera del modo compose
    let mut emo06b_dir_path: Option<std::path::PathBuf> = None;
    let mut effective_load_voice = cfg.load_voice.clone();
    let mut effective_xvector_only = cfg.xvector_only;
    let mut effective_icl_only = cfg.icl_only;

    if let Some(ref emo_spec) = cfg.emotion {
        if cfg.compose_spec.is_none() {
            if emotion_name_to_tok(emo_spec).is_none() {
                return Err(format!("Error: '{emo_spec}' is not a known emotion"));
            }
            let lv_path = cfg.load_voice.as_deref().map(Path::new);
            let has_clone = cfg.load_voice.is_some() || cfg.ref_audio.is_some();
            match resolve_emovoice_06b(
                emo_spec,
                lv_path,
                has_clone,
                cfg.speaker_name.as_deref(),
                &[],
            ) {
                Ok(EmoVoice06BAction::LoadQVoiceIclOnly(p)) => {
                    if !cfg.silent {
                        eprintln!("Emotion '{emo_spec}' on 0.6B: voice asset {}", p.display());
                    }
                    effective_load_voice = Some(p.to_string_lossy().to_string());
                    effective_icl_only = true;
                }
                Ok(EmoVoice06BAction::LoadBinXVector(p)) => {
                    if !cfg.silent {
                        eprintln!("Emotion '{emo_spec}' on 0.6B: voice asset {}", p.display());
                    }
                    effective_load_voice = Some(p.to_string_lossy().to_string());
                    effective_xvector_only = true;
                }
                Ok(EmoVoice06BAction::ApplyDirection { tok: _, dir_path }) => {
                    emo06b_dir_path = Some(dir_path);
                }
                Err(e) => return Err(e.to_string()),
            }
        }
    }

    let mut voice_lang_override: Option<i32> = None;
    let mut voice = if let Some(ref ref_audio_path) = cfg.ref_audio {
        let save_opts = SaveVoiceOptions {
            ref_text: cfg.ref_text.as_deref(),
            xvector_only: effective_xvector_only || cfg.ref_text.is_none(),
            graft_mode: cfg.graft,
            icl_frames_cap: cfg.icl_frames,
            max_ref_seconds: cfg.max_ref_duration,
            language: cfg.language.as_deref().or(Some("English")),
            voice_name: cfg.voice_name.as_deref(),
            target_cv_dir: cfg.target_cv.as_deref().map(Path::new),
            silent: cfg.silent,
            debug: cfg.debug,
        };
        build_voice_from_ref_audio(
            runtime,
            model,
            Path::new(ref_audio_path),
            cfg.save_voice.as_deref().map(Path::new),
            &save_opts,
        )
        .map_err(|e| e.to_string())?
    } else if let Some(ref lv_path) = effective_load_voice {
        let lv_opts = VoiceLoadOptions {
            icl_only: effective_icl_only,
            graft_mode: cfg.graft,
            xvector_only: effective_xvector_only,
            icl_frames_cap: cfg.icl_frames,
            voice_strength: cfg.voice_strength,
            vs_l0: cfg.vs_l0,
            vs_l1: cfg.vs_l1,
            silent: cfg.silent,
        };
        let (v, meta_opt) = load_voice_from_file(
            runtime,
            model,
            Path::new(lv_path),
            cfg.ref_text.as_deref(),
            &lv_opts,
        )
        .map_err(|e| e.to_string())?;
        if let Some(meta) = meta_opt {
            if meta.lang_id > 0 {
                if cli_language_id.is_none() {
                    voice_lang_override = Some(meta.lang_id as i32);
                    if !cfg.silent && !meta.lang_name.is_empty() {
                        eprintln!("  Auto-set language from voice: {}", meta.lang_name);
                    }
                } else if let Some(cl_id) = cli_language_id {
                    if cl_id != meta.lang_id as i32 && !cfg.silent {
                        eprintln!(
                            "WARNING: voice was created with language '{}' but you specified '{}'",
                            meta.lang_name,
                            cfg.language.as_deref().unwrap_or("")
                        );
                    }
                }
            }
        }
        v
    } else {
        VoiceBindings::preset(speaker_id)
    };

    if let Some(dir_path) = emo06b_dir_path {
        let applied = match &mut voice.identity {
            VoiceIdentity::XVector(emb) | VoiceIdentity::IclGraft { speaker_embedding: emb, .. } => {
                apply_emotion_direction(emb, &dir_path, cfg.emotion_strength)
                    .map_err(|e| e.to_string())?;
                true
            }
            VoiceIdentity::Preset(_) => false,
        };
        if !applied {
            return Err("Error: generic emotion direction requires a cloned voice".to_string());
        }
        if !cfg.silent {
            eprintln!(
                "Emotion on 0.6B: generic direction @ {:.2}",
                cfg.emotion_strength
            );
        }
    }

    if let (Some(ref emo_ref_path), Some(ref emo_ref_text)) = (&cfg.emo_ref, &cfg.emo_ref_text) {
        attach_emo_ref(
            runtime,
            model,
            &mut voice,
            Path::new(emo_ref_path),
            emo_ref_text,
            cfg.silent,
            cfg.debug,
        )
        .map_err(|e| e.to_string())?;
    }

    Ok((voice, voice_lang_override))
}

fn postprocess_and_write_wav(cfg: &CliArgs, audio: &mut Vec<f32>) -> Result<(), String> {
    if cfg.tail_trim {
        let cut = tail_trim(audio, SAMPLE_RATE, 0.30);
        if cut > 0 && !cfg.silent {
            eprintln!(
                "Tail-trim: cut {:.2}s degenerate tail ({cut} samples)",
                cut as f64 / SAMPLE_RATE as f64
            );
        }
    }
    if cfg.onset_fade_ms > 0 {
        onset_fade(audio, SAMPLE_RATE, cfg.onset_fade_ms as usize);
        if !cfg.silent {
            eprintln!("Onset-fade: {} ms over the real attack", cfg.onset_fade_ms);
        }
    }
    write_wav(Path::new(&cfg.output), audio, SAMPLE_RATE as u32).map_err(|e| e.to_string())?;
    if !cfg.silent {
        eprintln!(
            "Wrote {} ({} samples, {:.2}s)",
            cfg.output,
            audio.len(),
            audio.len() as f64 / SAMPLE_RATE as f64
        );
    }
    Ok(())
}

fn run_seed_audition(
    cfg: &CliArgs,
    runtime: &ComputeRuntime,
    model: &Model06B,
    voice: &VoiceBindings,
    scratch: &mut SessionScratch,
    req_cfg: &RequestConfig,
    active_text: &str,
) -> Result<Vec<f32>, String> {
    let n_takes = cfg.seed_audition.min(64);
    let mut candidates: Vec<(Vec<f32>, u32, f32)> = Vec::with_capacity(n_takes);
    for i in 0..n_takes {
        let mut ccfg = req_cfg.clone();
        ccfg.seed = req_cfg.seed.wrapping_add(i as u32);
        let exec = RequestExecution::new(ccfg.clone());
        if let Ok((a, _)) =
            synthesize_to_samples(runtime, model, voice, scratch, &exec, active_text)
        {
            if !a.is_empty() {
                let (g, _) = tail_glitch_score(&a, SAMPLE_RATE);
                if !cfg.silent {
                    eprintln!(
                        "  audition seed {}: {:.2}s glitch={g:.2}",
                        ccfg.seed,
                        a.len() as f32 / SAMPLE_RATE as f32
                    );
                }
                if cfg.audition_keep {
                    let out_p = Path::new(&cfg.output);
                    let stem = out_p
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("output");
                    let keep_name = match out_p.parent() {
                        Some(parent) if !parent.as_os_str().is_empty() => {
                            parent.join(format!("{stem}.seed{}.wav", ccfg.seed))
                        }
                        _ => Path::new(&format!("{stem}.seed{}.wav", ccfg.seed)).to_path_buf(),
                    };
                    if write_wav(&keep_name, &a, SAMPLE_RATE as u32).is_ok() && !cfg.silent {
                        eprintln!("    kept -> {}", keep_name.display());
                    }
                }
                candidates.push((a, ccfg.seed, g));
            }
        }
    }
    if candidates.is_empty() {
        return Err("Generation failed".to_string());
    }
    let mut lengths: Vec<usize> = candidates.iter().map(|(a, _, _)| a.len()).collect();
    lengths.sort_unstable();
    let med_n = lengths[lengths.len() / 2];
    let mut best_idx = 0usize;
    let mut best_cost = f32::MAX;
    let mut n_rej = 0usize;
    for (idx, (a, _, g)) in candidates.iter().enumerate() {
        let r = if med_n > 0 {
            a.len() as f32 / med_n as f32
        } else {
            1.0
        };
        let dd = (r - 1.0).abs();
        let bad = !(0.55..=2.50).contains(&r) || *g >= 0.5;
        if bad {
            n_rej += 1;
        }
        let cost = g * 10.0 + dd + if bad { 100.0 } else { 0.0 };
        if cost < best_cost {
            best_cost = cost;
            best_idx = idx;
        }
    }
    let best_seed = candidates[best_idx].1;
    let best_g = candidates[best_idx].2;
    let best_len = candidates[best_idx].0.len();
    if !cfg.silent {
        eprintln!(
            "  audition -> picked seed {best_seed} (glitch={best_g:.2}, {:.2}s of {} takes; {n_rej} rejected as broken)",
            best_len as f32 / SAMPLE_RATE as f32,
            candidates.len()
        );
    }
    Ok(candidates.swap_remove(best_idx).0)
}

fn run_app(mut cfg: CliArgs) -> Result<i32, String> {
    let physical_cores = affinity::pin_one_thread_per_core();

    if cfg.stdout_pcm {
        cfg.silent = true;
    }

    if let Some(code) = try_run_auxiliary_cmd(&cfg) {
        return Ok(code);
    }

    let model_dir_str = cfg
        .model_dir
        .clone()
        .ok_or_else(|| "Error: --model-dir is required".to_string())?;

    let create_voice_only =
        cfg.save_voice.is_some() && cfg.text.is_none() && cfg.serve_port.is_none();
    if cfg.text.is_none()
        && cfg.compose_spec.is_none()
        && cfg.serve_port.is_none()
        && !create_voice_only
        && !cfg.batch_test
        && !cfg.batch_bench
    {
        return Err("Error: --text, --compose or --serve is required".to_string());
    }

    let speaker_id = if let Some(ref spk_name) = cfg.speaker_name {
        lookup_speaker(spk_name)
            .ok_or_else(|| format!("Error: unknown speaker '{spk_name}'"))?
            .id
    } else {
        3061 // ryan
    };

    let cli_language_id = if let Some(ref lang_name) = cfg.language {
        Some(
            lookup_language(lang_name)
                .ok_or_else(|| format!("Error: unknown language '{lang_name}'"))?,
        )
    } else {
        None
    };

    if cfg.stream && (cfg.rate - 1.0).abs() > 1e-4 {
        return Err("Error: --rate != 1.0 cannot be combined with --stream".to_string());
    }

    // Validación temprana de WAV ANTES de cargar el modelo
    if let Some(ref ref_audio) = cfg.ref_audio {
        validate_wav_path_or_header(Path::new(ref_audio)).map_err(|e| e.to_string())?;
    }
    if let Some(ref emo_ref) = cfg.emo_ref {
        if cfg.emo_ref_text.as_deref().unwrap_or("").is_empty() {
            return Err(
                "Error: --emo-ref requires --emo-ref-text (the reference's transcript)"
                    .to_string(),
            );
        }
        validate_wav_path_or_header(Path::new(emo_ref)).map_err(|e| e.to_string())?;
    }

    if cfg.watch_stdin {
        spawn_stdin_watcher()?;
    }

    if !cfg.silent {
        eprintln!("Model dir: {model_dir_str}");
        if let Some(ref t) = cfg.text {
            eprintln!("Text: \"{t}\"");
        }
        eprintln!("Output: {}", cfg.output);
    }

    let runtime = Arc::new(ComputeRuntime::new(cfg.threads));
    let load_opts = ModelLoadOptions {
        use_int8: cfg.use_int8,
        use_int4: cfg.use_int4,
        quant_mixed: cfg.quant_mixed,
        silent: cfg.silent,
    };

    let mut model =
        Model06B::load(Path::new(&model_dir_str), &runtime, load_opts).map_err(|e| e.to_string())?;

    if cfg.ref_audio.is_some() && !model.config.is_base_model {
        return Err("Error: --ref-audio requires a Base model (not CustomVoice)".to_string());
    }
    if create_voice_only && cfg.ref_audio.is_none() {
        return Err("Error: --ref-audio is required to create a voice profile".to_string());
    }

    let (voice, voice_lang_override) =
        resolve_voice_and_emotion(&cfg, &runtime, &mut model, speaker_id, cli_language_id)?;

    if create_voice_only {
        if !cfg.silent {
            eprintln!("Voice profile created. Use --load-voice to generate speech.");
        }
        return Ok(0);
    }

    let model = Arc::new(model);
    let voice = Arc::new(voice);

    // Comandos de diagnóstico por lotes
    if cfg.batch_test {
        let ok = batch_self_test(&runtime, &model).map_err(|e| e.to_string())?;
        return Ok(if ok { 0 } else { 1 });
    }
    if cfg.batch_bench {
        batch_bench(&runtime, &model).map_err(|e| e.to_string())?;
        return Ok(0);
    }

    let language_id = if matches!(voice.identity, VoiceIdentity::Preset(_)) {
        Some(cli_language_id.unwrap_or(2050))
    } else {
        cli_language_id.or(voice_lang_override).or(Some(2050))
    };

    let max_tokens = if let Some(dur) = cfg.max_duration {
        if dur > 0.0 {
            ((dur * FRAME_RATE) as usize).max(1)
        } else {
            cfg.max_tokens
        }
    } else {
        cfg.max_tokens
    };

    if let Some(n_multi) = cfg.batch_multi_test {
        let text = cfg
            .text
            .as_deref()
            .ok_or_else(|| "Error: --text is required for --batch-multi-test".to_string())?;
        let base_seed = cfg.seed.unwrap_or_else(time_based_seed);
        let reqs: Vec<BatchReq> = (0..n_multi)
            .map(|b| BatchReq {
                text: text.to_string(),
                speaker_id,
                language_id,
                temperature: cfg.temperature,
                top_k: cfg.top_k,
                top_p: cfg.top_p,
                rep_penalty: cfg.rep_penalty,
                greedy_warmup: cfg.greedy_warmup,
                seed: base_seed.wrapping_add(b as u32),
                want_stream: false,
                chunk_frames: cfg.stream_chunk,
                volume: cfg.volume,
            })
            .collect();
        let outs =
            generate_batch_multi(&runtime, &model, &voice, &reqs, false).map_err(|e| e.to_string())?;
        for (b, aud) in outs.iter().enumerate() {
            let fn_out = format!("bm_{b}.wav");
            if !aud.is_empty() {
                write_wav(Path::new(&fn_out), aud, SAMPLE_RATE as u32)
                    .map_err(|e| e.to_string())?;
                eprintln!("  slot {b} (seed {}): {} samples -> {fn_out}", reqs[b].seed, aud.len());
            } else {
                eprintln!("  slot {b} (seed {}): EMPTY", reqs[b].seed);
            }
        }
        return Ok(0);
    }

    // Modo servidor HTTP
    if let Some(port) = cfg.serve_port {
        serve(ServeConfig {
            runtime: Arc::clone(&runtime),
            model: Arc::clone(&model),
            voice: Arc::clone(&voice),
            host: cfg.serve_host,
            port,
            workers: cfg.workers,
            batch_size: cfg.batch_size,
            physical_cores,
        })?;
        return Ok(0);
    }

    // Modo long-form por lotes (--batch / --batch-dry)
    if cfg.batch_mode {
        let text = cfg
            .text
            .as_deref()
            .ok_or_else(|| "Error: --text is required for --batch".to_string())?;
        let min_words = (cfg.batch_words / 3).max(6);
        let chunks = split_text_for_batch(text, cfg.batch_words, min_words);
        if chunks.is_empty() {
            return Err("--batch: text split produced 0 chunks".to_string());
        }
        if !cfg.silent || cfg.batch_dry {
            eprintln!(
                "--batch: {} chunk(s) (target ~{} words, min {}):",
                chunks.len(),
                cfg.batch_words,
                min_words
            );
            for (idx, ch) in chunks.iter().enumerate() {
                eprintln!("  [chunk {idx}, {} words] {ch}", ch.split_whitespace().count());
            }
        }
        if cfg.batch_dry {
            return Ok(0);
        }

        let seed = cfg.seed.unwrap_or_else(time_based_seed);
        if !cfg.silent {
            eprintln!(
                "seed: {seed}{}",
                if cfg.seed.is_some() { "" } else { " (auto/time-based)" }
            );
        }
        let req_cfg = RequestConfig {
            speaker_id,
            language_id,
            temperature: cfg.temperature,
            top_k: cfg.top_k,
            top_p: cfg.top_p,
            rep_penalty: cfg.rep_penalty,
            max_tokens,
            greedy_warmup: cfg.greedy_warmup,
            seed,
            stream: false,
            stream_chunk_frames: cfg.stream_chunk,
            volume: cfg.volume,
            rate: cfg.rate,
            silent: cfg.silent,
            debug: cfg.debug,
            cp_roughness: cfg.roughness,
        };

        let force_seq = std::env::var("QWEN_BATCH_SEQ")
            .ok()
            .map(|v| !v.trim().is_empty() && v.trim() != "0")
            .unwrap_or(false);
        let audio = if chunks.len() >= 2 && !force_seq {
            let mut buf = generate_batch_longform(
                &runtime,
                &model,
                &voice,
                &chunks,
                cfg.compose_pause,
                &req_cfg,
                speaker_id,
            )
            .map_err(|e| e.to_string())?;
            if (cfg.rate - 1.0).abs() > 1e-4 {
                buf = time_stretch(&buf, cfg.rate);
            }
            buf
        } else {
            let mut scratch = SessionScratch::new();
            let spans: Vec<_> = chunks
                .iter()
                .map(|c| qwen_core::SpanSpec::Utterance {
                    text: c.clone(),
                    mood: None,
                    is_filler: false,
                    rate: 0.0,
                    volume: 0.0,
                })
                .collect();
            let exec = RequestExecution::new(req_cfg);
            let (aud, _) = render_spans_buffer(
                &runtime,
                &model,
                &voice,
                &mut scratch,
                &exec,
                &spans,
                cfg.compose_pause,
            )
            .map_err(|e| e.to_string())?;
            aud
        };

        write_wav(Path::new(&cfg.output), &audio, SAMPLE_RATE as u32)
            .map_err(|e| e.to_string())?;
        if !cfg.silent {
            eprintln!(
                "Wrote {} ({} samples, {:.2}s)",
                cfg.output,
                audio.len(),
                audio.len() as f64 / SAMPLE_RATE as f64
            );
        }
        return Ok(0);
    }

    // Detección de paralingüística inline + compose
    let mut active_text = cfg.text.clone().unwrap_or_default();
    let mut compose_spec = cfg.compose_spec.clone();
    let mut resolved_seed = cfg.seed;
    let mut resolved_temp = cfg.temperature;

    if compose_spec.is_none() && !active_text.is_empty() {
        let prep = prepare_markup_06b(
            &active_text,
            &voice,
            cfg.no_compose,
            if cfg.seed_set { cfg.seed } else { None },
            if cfg.temp_set {
                Some(cfg.temperature)
            } else {
                None
            },
            cfg.temperature,
        );
        active_text = prep.text;
        if !cfg.seed_set && prep.did_para_substitute {
            resolved_seed = Some(prep.seed);
        }
        resolved_temp = prep.temperature;
        if prep.did_para_substitute && !cfg.silent {
            eprintln!(
                "Paralinguistics: inline [tag]->onomatopoeia (seed {}, T{:.1}): \"{}\"",
                resolved_seed.unwrap_or(prep.seed),
                resolved_temp,
                active_text
            );
        }
        if prep.use_compose {
            compose_spec = Some(active_text.clone());
            if !cfg.silent {
                eprintln!("Inline markup detected in --text -> compose mode");
            }
        }
    }

    let final_seed = resolved_seed.unwrap_or_else(time_based_seed);
    if !cfg.silent {
        eprintln!(
            "seed: {final_seed}{}",
            if resolved_seed.is_some() { "" } else { " (auto/time-based)" }
        );
    }

    let req_cfg = RequestConfig {
        speaker_id,
        language_id,
        temperature: resolved_temp,
        top_k: cfg.top_k,
        top_p: cfg.top_p,
        rep_penalty: cfg.rep_penalty,
        max_tokens,
        greedy_warmup: cfg.greedy_warmup,
        seed: final_seed,
        stream: cfg.stream,
        stream_chunk_frames: cfg.stream_chunk.clamp(2, 250),
        volume: cfg.volume,
        rate: cfg.rate,
        silent: cfg.silent,
        debug: cfg.debug,
        cp_roughness: cfg.roughness,
    };

    let mut scratch = SessionScratch::new();

    // Modo compose (sin streaming)
    if let Some(ref spec) = compose_spec {
        if !cfg.silent {
            eprintln!("Compose mode: rendering spans...");
        }
        let spans = MarkupPlanner::parse_spans(spec);
        let exec = RequestExecution::new(req_cfg);
        let (mut audio, _) = render_spans_buffer(
            &runtime,
            &model,
            &voice,
            &mut scratch,
            &exec,
            &spans,
            cfg.compose_pause,
        )
        .map_err(|e| e.to_string())?;
        postprocess_and_write_wav(&cfg, &mut audio)?;
        return Ok(0);
    }

    // Modo CLI en streaming (--stream / --stdout)
    if cfg.stream {
        if !cfg.silent {
            eprintln!(
                "Streaming: chunk={} frames ({:.1}s), {}",
                req_cfg.stream_chunk_frames,
                req_cfg.stream_chunk_frames as f32 / FRAME_RATE,
                if cfg.stdout_pcm {
                    "raw PCM to stdout"
                } else {
                    &cfg.output
                }
            );
        }
        let exec = RequestExecution::new(req_cfg);
        if cfg.stdout_pcm {
            let mut sink = StdoutRawSink::new(cfg.volume);
            synthesize_to_sink(&runtime, &model, &voice, &mut scratch, &exec, &active_text, &mut sink)
                .map_err(|e| e.to_string())?;
        } else {
            let mut sink = WavStreamSink::create(
                Path::new(&cfg.output),
                SAMPLE_RATE as u32,
                cfg.volume,
            )
            .map_err(|e| e.to_string())?;
            let timing = synthesize_to_sink(
                &runtime,
                &model,
                &voice,
                &mut scratch,
                &exec,
                &active_text,
                &mut sink,
            )
            .map_err(|e| e.to_string())?;
            if !cfg.silent {
                eprintln!(
                    "Wrote {} ({} samples, {:.2}s) [streamed]",
                    cfg.output,
                    timing.samples_produced,
                    timing.samples_produced as f64 / SAMPLE_RATE as f64
                );
            }
        }
        return Ok(0);
    }

    // Síntesis sin streaming (toma única o --seed-audition N)
    if cfg.audition_keep && cfg.seed_audition <= 1 && !cfg.silent {
        eprintln!("Warning: --audition-keep has no effect without --seed-audition N (N>1, non-stream)");
    }

    let mut audio = if cfg.seed_audition > 1 {
        run_seed_audition(
            &cfg,
            &runtime,
            &model,
            &voice,
            &mut scratch,
            &req_cfg,
            &active_text,
        )?
    } else {
        let exec = RequestExecution::new(req_cfg);
        let (a, _) =
            synthesize_to_samples(&runtime, &model, &voice, &mut scratch, &exec, &active_text)
                .map_err(|e| e.to_string())?;
        a
    };

    postprocess_and_write_wav(&cfg, &mut audio)?;
    Ok(0)
}

fn run_decode_dump(codes_path: &str, model_dir: &str, out_wav: &str) -> Result<i32, String> {
    let raw = fs::read_to_string(codes_path)
        .map_err(|e| format!("Failed to read codes file {codes_path}: {e}"))?;
    let mut codes = Vec::new();
    for tok in raw.split_whitespace() {
        let v = tok
            .parse::<i32>()
            .map_err(|_| format!("Invalid integer token '{tok}' in {codes_path}"))?;
        codes.push(v);
    }
    if codes.is_empty() || codes.len() % 16 != 0 {
        return Err(format!(
            "Invalid code count {} in {codes_path} (expected multiple of 16)",
            codes.len()
        ));
    }
    let n_frames = codes.len() / 16;
    let rt = ComputeRuntime::new(None);
    let opts = ModelLoadOptions {
        silent: true,
        ..Default::default()
    };
    let model = Model06B::load(Path::new(model_dir), &rt, opts).map_err(|e| e.to_string())?;
    let audio = qwen_engine::speech_decoder_decode(&rt, &model.decoder, &codes, n_frames)
        .map_err(|e| e.to_string())?;
    write_wav(Path::new(out_wav), &audio, SAMPLE_RATE as u32).map_err(|e| e.to_string())?;
    eprintln!(
        "decode-dump: {n_frames} frames -> {} samples -> {out_wav}",
        audio.len()
    );
    Ok(0)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("decode-dump") {
        if args.len() != 5 {
            eprintln!("Usage: {} decode-dump <codes.txt> <model_dir> <out.wav>", args[0]);
            process::exit(1);
        }
        match run_decode_dump(&args[2], &args[3], &args[4]) {
            Ok(code) => process::exit(code),
            Err(msg) => {
                eprintln!("{msg}");
                process::exit(1);
            }
        }
    }
    let prog = args.first().map(String::as_str).unwrap_or("qwen_tts");
    let cfg = match parse_cli(&args) {
        Ok(ParseOutcome::Help) => {
            print_usage(prog);
            process::exit(0);
        }
        Ok(ParseOutcome::Run(c)) => *c,
        Err(msg) => {
            eprintln!("{msg}");
            process::exit(1);
        }
    };

    match run_app(cfg) {
        Ok(code) => process::exit(code),
        Err(msg) => {
            eprintln!("{msg}");
            process::exit(1);
        }
    }
}
