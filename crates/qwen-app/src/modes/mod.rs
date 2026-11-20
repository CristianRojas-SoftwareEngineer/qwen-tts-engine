// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Modos del binario `qwen_tts`. `diag` y `voice_cmd` no cargan el modelo; los modos de
//! síntesis comparten la preparación de [`run_with_model`].

pub mod diag;
pub mod serve;
pub mod single;
pub mod voice_cmd;

use std::io::Read;
use std::path::Path;
use std::process;
use std::sync::Arc;
use std::thread;

use qwen_core::config::lookup_language;
use qwen_engine::{validate_wav_path_or_header, Model06B, ModelLoadOptions, Synthesizer};
use qwen_kernels::ComputeRuntime;

use crate::cli::{CliArgs, CliMode};

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

/// Modos con modelo: crear voz, servidor y una síntesis. Valida, carga el modelo, resuelve la
/// voz y delega en el modo.
pub fn run_with_model(cfg: CliArgs, physical_cores: usize) -> Result<i32, String> {
    let model_dir_str = cfg
        .model_dir
        .clone()
        .ok_or_else(|| "Error: --model-dir is required".to_string())?;

    if cfg.mode == CliMode::NoJob {
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

    if cfg.mode == CliMode::CreateVoice && cfg.ref_audio.is_none() {
        return Err("Error: --ref-audio is required to create a voice profile".to_string());
    }

    let voice = voice_cmd::resolve_voice(&cfg, &runtime, &model, cli_language_id)?;

    match cfg.mode {
        CliMode::CreateVoice => {
            if !cfg.silent {
                eprintln!("Voice profile created. Use --load-voice to generate speech.");
            }
            Ok(0)
        }
        CliMode::Serve(port) => serve::run(&cfg, runtime, model, voice, port),
        _ => single::run(
            &cfg,
            &Synthesizer {
                runtime: &runtime,
                model: &model,
                voice: &voice,
            },
        ),
    }
}
