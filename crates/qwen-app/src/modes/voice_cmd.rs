// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Gestión de voces: `--list-voices`, `--delete-voice` y la resolución de la voz de la sesión
//! (`--ref-audio` clona, `--load-voice` carga, si no la voz `default`).

use std::fs;
use std::path::Path;

use qwen_core::config::language_name;
use qwen_core::QVoiceFile;
use qwen_engine::{
    clone_voice, ensure_qvoice_path, CloneOptions, Model06B, Voice, VoiceLoadOptions,
};
use qwen_kernels::ComputeRuntime;

use crate::cli::CliArgs;
use crate::request;

/// Imprime una línea por voz: archivo, idioma y tamaño. Un archivo que el lector
/// rechaza se muestra con `no compatible: <mensaje del lector>`.
fn print_qvoice_info(path: &Path) {
    let basename = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("desconocido");
    let size = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    match QVoiceFile::read_from_file(path) {
        Ok(qv) => {
            let lang = i32::try_from(qv.lang_id)
                .ok()
                .and_then(language_name)
                .unwrap_or("-");
            println!(
                "  {basename:<30}  lang={lang}  {:.1} KB",
                size as f32 / 1024.0
            );
        }
        Err(e) => println!("  {basename:<30}  no compatible: {e}"),
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
        println!("Perfiles de voz:");
        print_qvoice_info(p);
        return 0;
    }
    let Ok(entries) = fs::read_dir(p) else {
        eprintln!("Error: no se puede abrir el directorio {dir_str}");
        return 1;
    };
    println!("Perfiles de voz en {dir_str}:\n");
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
        println!("  (no se encontraron archivos .qvoice)");
    } else {
        println!("\n  {count} perfil(es) de voz");
    }
    0
}

/// `--delete-voice`: borra un archivo `.qvoice`, de cualquier versión mientras lleve la firma
/// `QVCE`.
pub fn delete_voice(path_str: &str) -> i32 {
    let path = Path::new(path_str);
    if !has_qvoice_ext(path) {
        eprintln!("Error: --delete-voice solo funciona con archivos .qvoice");
        return 1;
    }
    if !path.is_file() {
        eprintln!("Error: archivo no encontrado: {path_str}");
        return 1;
    }
    if !QVoiceFile::has_magic(path) {
        eprintln!("Error: {path_str} no es un archivo .qvoice válido");
        return 1;
    }
    if let Err(e) = fs::remove_file(path) {
        eprintln!("Error: no se pudo borrar {path_str}: {e}");
        return 1;
    }
    println!("Borrado {path_str}");
    0
}

/// Nombre de la voz clonada: el archivo de `--save-voice` sin extensión si se guarda; si no,
/// el WAV de `--ref-audio` sin extensión.
fn cloned_voice_name(ref_audio: &str, save_voice: Option<&str>) -> Result<String, String> {
    let source = save_voice.unwrap_or(ref_audio);
    Path::new(source)
        .file_stem()
        .and_then(|s| s.to_str())
        .map(str::to_string)
        .ok_or_else(|| format!("Error: no se puede obtener el nombre de la voz de '{source}'"))
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
        let name = cloned_voice_name(ref_audio_path, cfg.save_voice.as_deref())?;
        let clone_opts = CloneOptions {
            language_id: cli_language_id,
            log: request::log_level(cfg),
        };
        let qv = clone_voice(runtime, model, Path::new(ref_audio_path), &clone_opts)
            .map_err(|e| e.to_string())?;
        if let Some(ref save_path) = cfg.save_voice {
            qv.write_to_file(Path::new(save_path))
                .map_err(|e| e.to_string())?;
            if !cfg.silent {
                eprintln!("Guardado .qvoice v6 en {save_path}");
            }
        }
        Voice::from_qvoice(qv, name)
    } else if let Some(ref lv_path) = cfg.load_voice {
        let lv_opts = VoiceLoadOptions {
            log: request::log_level(cfg),
        };
        let v = Voice::load(Path::new(lv_path), &lv_opts).map_err(|e| e.to_string())?;
        if let Some(vl_id) = v.language_id {
            let vl_name = language_name(vl_id).unwrap_or("");
            match cli_language_id {
                None => {
                    if !cfg.silent && !vl_name.is_empty() {
                        eprintln!("  Idioma tomado de la voz: {vl_name}");
                    }
                }
                Some(cl_id) => {
                    if cl_id != vl_id && !cfg.silent {
                        eprintln!(
                            "Aviso: la voz se creó con el idioma '{vl_name}' pero indicaste '{}'",
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
        eprintln!("Voz: {}", voice.name);
    }

    Ok(voice)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cloned_voice_name() {
        assert_eq!(
            cloned_voice_name("dir/ref.wav", Some("voces/mia.qvoice")),
            Ok("mia".to_string())
        );
        assert_eq!(cloned_voice_name("dir/ref.wav", None), Ok("ref".to_string()));
        assert!(cloned_voice_name("", None).is_err());
    }
}
