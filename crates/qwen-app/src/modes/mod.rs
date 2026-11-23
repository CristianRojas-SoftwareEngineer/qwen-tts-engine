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
use std::thread;

use qwen_core::config::lookup_language;
use qwen_engine::{read_wav_24khz, Model06B, Synthesizer};
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
        .map_err(|e| format!("Error: --watch-stdin no pudo iniciar el hilo que vigila stdin: {e}"))
}

/// Modos con modelo: crear voz, servidor y una síntesis. Valida, carga el modelo, resuelve la
/// voz y delega en el modo.
pub fn run_with_model(cfg: CliArgs, physical_cores: usize) -> Result<i32, String> {
    let model_dir_str = cfg
        .model_dir
        .clone()
        .ok_or_else(|| "Error: falta --model-dir".to_string())?;

    if cfg.mode == CliMode::NoJob {
        return Err("Error: falta --text o --serve".to_string());
    }

    let cli_language_id = if let Some(ref lang_name) = cfg.language {
        Some(
            lookup_language(lang_name)
                .ok_or_else(|| format!("Error: idioma desconocido '{lang_name}'"))?,
        )
    } else {
        None
    };

    if cfg.stdout_pcm && (cfg.rate - 1.0).abs() > 1e-4 {
        return Err("Error: --rate != 1.0 no se puede combinar con --stdout".to_string());
    }

    // Validación temprana de la referencia antes de cargar el modelo
    if let Some(ref ref_audio) = cfg.ref_audio {
        read_wav_24khz(Path::new(ref_audio)).map_err(|e| e.to_string())?;
    }
    if cfg.watch_stdin {
        spawn_stdin_watcher()?;
    }

    if !cfg.silent {
        eprintln!("Directorio del modelo: {model_dir_str}");
        if let Some(ref t) = cfg.text {
            eprintln!("Texto: \"{t}\"");
        }
        eprintln!("Salida: {}", cfg.output);
    }

    let runtime = ComputeRuntime::for_cores(cfg.threads, physical_cores);
    let model = Model06B::load(Path::new(&model_dir_str), &runtime, cfg.silent)
        .map_err(|e| e.to_string())?;

    let voice = voice_cmd::resolve_voice(&cfg, &runtime, &model, cli_language_id)?;

    match cfg.mode {
        CliMode::CreateVoice => {
            if !cfg.silent {
                eprintln!("Perfil de voz creado. Usa --load-voice para generar voz.");
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
