// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Gestión de voces: `--list-voices`, `--delete-voice` y la resolución de la voz de la sesión
//! (`--ref-audio` clona, `--load-voice` carga, si no la voz `default`).

use std::fs;
use std::path::Path;

use qwen_core::config::{language_name, FRAME_RATE};
use qwen_core::QVoiceFile;
use qwen_engine::{
    clone_voice, ensure_qvoice_path, CloneOptions, Model06B, Voice, VoiceLoadOptions,
};
use qwen_kernels::ComputeRuntime;

use crate::cli::CliArgs;
use crate::request;

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

/// `--list-voices`: un archivo `.qvoice` o todos los de una carpeta.
pub fn list_voices(dir_str: &str) -> i32 {
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

/// `--delete-voice`: borra un archivo `.qvoice`, de cualquier versión mientras lleve la firma
/// `QVCE`.
pub fn delete_voice(path_str: &str) -> i32 {
    let path = Path::new(path_str);
    if !has_qvoice_ext(path) {
        eprintln!("Error: --delete-voice only works with .qvoice files");
        return 1;
    }
    if !path.is_file() {
        eprintln!("Error: file not found: {path_str}");
        return 1;
    }
    if !QVoiceFile::has_magic(path) {
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

/// La voz de la sesión: clona `--ref-audio` (y la guarda con `--save-voice`), carga
/// `--load-voice` o usa la voz `default`.
pub fn resolve_voice(
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
