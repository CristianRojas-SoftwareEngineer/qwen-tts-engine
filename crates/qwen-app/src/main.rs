// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

mod affinity;
mod cli;
mod request;
mod server;

use std::fs;
use std::io::Read;
use std::path::Path;
use std::process;
use std::sync::Arc;
use std::thread;

use qwen_core::config::{language_name, lookup_language, FRAME_RATE, SAMPLE_RATE};
use qwen_core::{QVoiceFile, Script};
use qwen_engine::{
    clone_voice, ensure_qvoice_path, finish_full_delivery, synthesize, tail_glitch_score,
    validate_wav_path_or_header, write_wav, AbortHandle, CloneOptions, Delivery, EdgeRampSink,
    Model06B, ModelLoadOptions, SessionScratch, StdoutWriter, SynthesisRequest, Voice,
    VoiceLoadOptions, WavFileWriter,
};
use qwen_kernels::bench::run_matmat_bench;
use qwen_kernels::selftest::run_kernel_selftest;
use qwen_kernels::ComputeRuntime;

use cli::{parse_cli, print_usage, CliArgs, ParseOutcome};
use server::{serve, ServerContext};

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

/// Imprime una línea por voz: archivo, nombre, idioma, ICL y tamaño. Un archivo que el lector
/// rechaza se muestra con `unsupported: <mensaje del lector>`.
fn print_qvoice_info(path: &Path) {
    let basename = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");
    let size = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    match QVoiceFile::read_from_file(path) {
        Ok(qv) => {
            let name = if qv.voice_name.is_empty() {
                "-"
            } else {
                qv.voice_name.as_str()
            };
            let lang = i32::try_from(qv.lang_id)
                .ok()
                .and_then(language_name)
                .unwrap_or("-");
            let icl = if qv.has_icl() {
                let n = qv.n_ref_frames();
                format!("{n} frames ({:.1}s)", n as f32 / FRAME_RATE)
            } else {
                "no".to_string()
            };
            println!(
                "  {basename:<30}  name={name}  lang={lang}  icl={icl}  {:.1} KB",
                size as f32 / 1024.0
            );
        }
        Err(e) => println!("  {basename:<30}  unsupported: {e}"),
    }
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
        print_qvoice_info(p);
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
        print_qvoice_info(&path);
        count += 1;
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

fn resolve_voice(
    cfg: &CliArgs,
    runtime: &ComputeRuntime,
    model: &Model06B,
    cli_language_id: Option<i32>,
) -> Result<Voice, String> {
    let voice = if let Some(ref ref_audio_path) = cfg.ref_audio {
        if let Some(ref save_path) = cfg.save_voice {
            ensure_qvoice_path(Path::new(save_path)).map_err(|e| e.to_string())?;
        }
        let clone_opts = CloneOptions {
            ref_text: if cfg.xvector_only {
                None
            } else {
                cfg.ref_text.clone()
            },
            max_ref_seconds: cfg.max_ref_duration,
            name: cfg.voice_name.clone(),
            language_id: cli_language_id,
            icl_frames_cap: cfg.icl_frames,
            log: request::log_level(cfg),
        };
        let (qv, voice) = clone_voice(runtime, model, Path::new(ref_audio_path), &clone_opts)
            .map_err(|e| e.to_string())?;
        if let Some(ref save_path) = cfg.save_voice {
            qv.write_to_file(Path::new(save_path))
                .map_err(|e| e.to_string())?;
            if !cfg.silent {
                eprintln!(
                    "Saved .qvoice v4 to {} (embedding + {} ICL frames)",
                    save_path,
                    qv.n_ref_frames()
                );
            }
        }
        voice
    } else if let Some(ref lv_path) = cfg.load_voice {
        let lv_opts = VoiceLoadOptions {
            ref_text_override: cfg.ref_text.clone(),
            use_icl: !cfg.xvector_only,
            icl_frames_cap: cfg.icl_frames,
            log: request::log_level(cfg),
        };
        let v = Voice::load(Path::new(lv_path), &lv_opts).map_err(|e| e.to_string())?;
        if let Some(vl_id) = v.language_id {
            let vl_name = language_name(vl_id).unwrap_or("");
            match cli_language_id {
                None => {
                    if !cfg.silent && !vl_name.is_empty() {
                        eprintln!("  Auto-set language from voice: {vl_name}");
                    }
                }
                Some(cl_id) => {
                    if cl_id != vl_id && !cfg.silent {
                        eprintln!(
                            "WARNING: voice was created with language '{vl_name}' but you specified '{}'",
                            cfg.language.as_deref().unwrap_or("")
                        );
                    }
                }
            }
        }
        v
    } else {
        Voice::builtin_default()
    };

    if !cfg.silent {
        eprintln!("Voice: {}", voice.name);
    }

    Ok(voice)
}

fn write_output_wav(cfg: &CliArgs, audio: &[f32]) -> Result<(), String> {
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
    voice: &Voice,
    scratch: &mut SessionScratch,
    req: &SynthesisRequest,
    script: &Script,
) -> Result<Vec<f32>, String> {
    let n_takes = cfg.seed_audition.min(64);
    let mut candidates: Vec<(Vec<f32>, u32, f32)> = Vec::with_capacity(n_takes);
    let abort = AbortHandle::new();
    for i in 0..n_takes {
        // Las tomas se sintetizan sin `tail_trim` ni `onset_fade`: el acabado va solo a la elegida.
        let mut ccfg = req.clone();
        ccfg.sampling.seed = req.sampling.seed.wrapping_add(i as u32);
        ccfg.tail_trim = false;
        ccfg.onset_fade_ms = 0.0;
        if let Ok(out) = synthesize(
            runtime,
            model,
            voice,
            scratch,
            &ccfg,
            &abort,
            script,
            Delivery::Buffer,
        ) {
            let a = out.samples.unwrap_or_default();
            if !a.is_empty() {
                let (g, _) = tail_glitch_score(&a, SAMPLE_RATE);
                if !cfg.silent {
                    eprintln!(
                        "  audition seed {}: {:.2}s glitch={g:.2}",
                        ccfg.sampling.seed,
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
                            parent.join(format!("{stem}.seed{}.wav", ccfg.sampling.seed))
                        }
                        _ => Path::new(&format!("{stem}.seed{}.wav", ccfg.sampling.seed)).to_path_buf(),
                    };
                    if write_wav(&keep_name, &a, SAMPLE_RATE as u32).is_ok() && !cfg.silent {
                        eprintln!("    kept -> {}", keep_name.display());
                    }
                }
                candidates.push((a, ccfg.sampling.seed, g));
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
    {
        return Err("Error: --text, --compose or --serve is required".to_string());
    }

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

    let runtime = Arc::new(ComputeRuntime::for_cores(cfg.threads, physical_cores));
    let load_opts = ModelLoadOptions {
        use_int8: cfg.use_int8,
        use_int4: cfg.use_int4,
        quant_mixed: cfg.quant_mixed,
        silent: cfg.silent,
    };

    let model =
        Model06B::load(Path::new(&model_dir_str), &runtime, load_opts).map_err(|e| e.to_string())?;

    if create_voice_only && cfg.ref_audio.is_none() {
        return Err("Error: --ref-audio is required to create a voice profile".to_string());
    }

    let voice = resolve_voice(&cfg, &runtime, &model, cli_language_id)?;

    if create_voice_only {
        if !cfg.silent {
            eprintln!("Voice profile created. Use --load-voice to generate speech.");
        }
        return Ok(0);
    }

    let model = Arc::new(model);
    let voice = Arc::new(voice);

    // Modo servidor HTTP
    if let Some(port) = cfg.serve_port {
        serve(
            ServerContext {
                runtime: Arc::clone(&runtime),
                model: Arc::clone(&model),
                voice: Arc::clone(&voice),
            },
            cfg.serve_host,
            port,
        )?;
        return Ok(0);
    }

    let (req, planned) = request::from_cli(&cfg, &voice);
    let abort = AbortHandle::new();

    let mut scratch = SessionScratch::new();

    let script = planned.script;

    // Modo CLI en streaming (--stream / --stdout)
    if cfg.stream {
        if !cfg.silent {
            eprintln!(
                "Streaming: chunk={} frames ({:.1}s), {}",
                req.chunk_frames,
                req.chunk_frames as f32 / FRAME_RATE,
                if cfg.stdout_pcm {
                    "raw PCM to stdout"
                } else {
                    &cfg.output
                }
            );
        }
        if cfg.stdout_pcm {
            let mut sink = EdgeRampSink::new(StdoutWriter, SAMPLE_RATE as u32);
            synthesize(
                &runtime,
                &model,
                &voice,
                &mut scratch,
                &req,
                &abort,
                &script,
                Delivery::Stream(&mut sink),
            )
            .map_err(|e| e.to_string())?;
        } else {
            let writer = WavFileWriter::create(Path::new(&cfg.output), SAMPLE_RATE as u32)
                .map_err(|e| e.to_string())?;
            let mut sink = EdgeRampSink::new(writer, SAMPLE_RATE as u32);
            let out = synthesize(
                &runtime,
                &model,
                &voice,
                &mut scratch,
                &req,
                &abort,
                &script,
                Delivery::Stream(&mut sink),
            )
            .map_err(|e| e.to_string())?;
            if !cfg.silent {
                eprintln!(
                    "Wrote {} ({} samples, {:.2}s) [streamed]",
                    cfg.output,
                    out.timing.samples_produced,
                    out.timing.samples_produced as f64 / SAMPLE_RATE as f64
                );
            }
        }
        return Ok(0);
    }

    // Síntesis sin streaming (toma única o --seed-audition N)
    if cfg.audition_keep && cfg.seed_audition <= 1 && !cfg.silent {
        eprintln!("Warning: --audition-keep has no effect without --seed-audition N (N>1, non-stream)");
    }

    let audio = if cfg.seed_audition > 1 {
        let mut chosen =
            run_seed_audition(&cfg, &runtime, &model, &voice, &mut scratch, &req, &script)?;
        // Las tomas no llevan acabado: `tail_trim` y `onset_fade` van solo a la elegida.
        finish_full_delivery(&mut chosen, &req);
        chosen
    } else {
        // `synthesize` ya aplica el acabado tras `time_stretch`.
        synthesize(
            &runtime,
            &model,
            &voice,
            &mut scratch,
            &req,
            &abort,
            &script,
            Delivery::Buffer,
        )
        .map_err(|e| e.to_string())?
        .samples
        .unwrap_or_default()
    };

    write_output_wav(&cfg, &audio)?;
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

/// Fija los hilos iniciales de OpenBLAS con el `-j` efectivo antes de que se inicialice.
///
/// Corre en `.CRT$XCT`: dentro de `_initterm`, con la CRT ya iniciada y antes de los
/// constructores (`.ctors`) de OpenBLAS y libgomp. Ver [`qwen_kernels::set_initial_openblas_threads`].
/// Si los argumentos no se pueden leer o no son válidos, no hace nada: `main` informa del error.
#[cfg(all(windows, target_env = "gnu"))]
mod openblas_preinit {
    extern "C" fn set_initial_threads() {
        let Ok(args) = std::env::args_os()
            .map(std::ffi::OsString::into_string)
            .collect::<Result<Vec<String>, _>>()
        else {
            return;
        };
        if let Ok(crate::cli::ParseOutcome::Run(cfg)) = crate::cli::parse_cli(&args) {
            qwen_kernels::set_initial_openblas_threads(qwen_kernels::resolve_num_threads(cfg.threads));
        }
    }

    #[used]
    #[link_section = ".CRT$XCT"]
    static SET_INITIAL_THREADS: extern "C" fn() = set_initial_threads;
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
