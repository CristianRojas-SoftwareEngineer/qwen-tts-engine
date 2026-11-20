// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! La voz de una síntesis: x-vector del locutor y, opcionalmente, el contexto ICL (texto y
//! códigos del audio de referencia) ya recortado al tope de fotogramas.

use std::path::Path;

use qwen_core::config::{DEFAULT_LANGUAGE_ID, MEL_DIM, NUM_CODEBOOKS, SAMPLE_RATE};
use qwen_core::QVoiceFile;
use qwen_kernels::ComputeRuntime;

use crate::error::EngineError;
use crate::model::Model06B;
use crate::request::LogLevel;
use crate::speech_encoder::speech_encoder_encode;
use crate::voice_clone::{load_and_trim_ref_audio, mel_spectrogram, speaker_encoder_forward};

/// x-vector de la voz `default` incorporada (1024 `f32` little-endian).
const DEFAULT_VOICE_XVEC: &[u8] = include_bytes!("../assets/default_voice.xvec");
const _: () = assert!(DEFAULT_VOICE_XVEC.len() == 4096);

/// Nombre de la voz incorporada.
pub const DEFAULT_VOICE_NAME: &str = "default";

/// Contexto ICL de una voz: transcripción y códigos del audio de referencia.
///
/// Nunca está vacío y `ref_codes` ya está recortado al tope de fotogramas con que se construyó
/// la voz (`icl_frames_cap` o, si está definida, `QWEN_ICL_FRAMES`).
#[derive(Debug, Clone)]
pub struct IclPrompt {
    pub ref_text: String,
    /// `n_frames × NUM_CODEBOOKS` códigos.
    pub ref_codes: Vec<i32>,
}

/// Voz de una síntesis. Es inmutable una vez construida.
#[derive(Debug, Clone)]
pub struct Voice {
    pub name: String,
    /// Idioma propio de la voz (`None` si no declara ninguno).
    pub language_id: Option<i32>,
    /// x-vector del locutor.
    pub xvector: Vec<f32>,
    pub icl: Option<IclPrompt>,
}

/// Opciones de carga de un perfil `.qvoice`.
#[derive(Debug, Clone, Default)]
pub struct VoiceLoadOptions {
    /// Sustituye la transcripción ICL del archivo.
    pub ref_text_override: Option<String>,
    /// Usa el contexto ICL si el archivo lo trae; `false` deja solo el x-vector.
    pub use_icl: bool,
    /// Tope de fotogramas ICL (0 = sin tope). `QWEN_ICL_FRAMES`, si está definida, manda.
    pub icl_frames_cap: usize,
    pub log: LogLevel,
}

/// Opciones para clonar una voz desde un audio de referencia.
#[derive(Debug, Clone, Default)]
pub struct CloneOptions {
    /// Transcripción del audio; sin ella (o vacía) la voz es solo x-vector.
    pub ref_text: Option<String>,
    pub max_ref_seconds: f32,
    /// Nombre de la voz; si falta, el nombre del WAV sin extensión.
    pub name: Option<String>,
    pub language_id: Option<i32>,
    pub icl_frames_cap: usize,
    pub log: LogLevel,
}

/// Error si `path` no tiene extensión `.qvoice`.
pub fn ensure_qvoice_path(path: &Path) -> Result<(), EngineError> {
    let is_qvoice = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("qvoice"));
    if is_qvoice {
        Ok(())
    } else {
        Err(EngineError::InvalidArg(format!(
            "voice files must be .qvoice (got {})",
            path.display()
        )))
    }
}

/// Tope efectivo de fotogramas ICL: `QWEN_ICL_FRAMES` si es un entero válido, si no `cap`.
fn effective_icl_cap(cap: usize) -> usize {
    if let Ok(v) = std::env::var("QWEN_ICL_FRAMES") {
        if let Ok(n) = v.trim().parse::<usize>() {
            return n;
        }
    }
    cap
}

/// Recorta `ref_codes` a los primeros fotogramas permitidos por el tope efectivo.
fn cap_ref_codes(ref_codes: &mut Vec<i32>, cap: usize) {
    let cap = effective_icl_cap(cap);
    let n_frames = ref_codes.len() / NUM_CODEBOOKS;
    if cap > 0 && cap < n_frames {
        ref_codes.truncate(cap * NUM_CODEBOOKS);
    }
}

impl Voice {
    /// Voz `default` incorporada en el binario; es la voz por defecto del motor.
    pub fn builtin_default() -> Self {
        let xvector: Vec<f32> = DEFAULT_VOICE_XVEC
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Self {
            name: DEFAULT_VOICE_NAME.to_string(),
            language_id: Some(DEFAULT_LANGUAGE_ID),
            xvector,
            icl: None,
        }
    }

    /// Carga un perfil `.qvoice` v4. Sin nombre en el archivo, la voz se llama como el archivo
    /// (sin extensión).
    pub fn load(path: &Path, opts: &VoiceLoadOptions) -> Result<Self, EngineError> {
        ensure_qvoice_path(path)?;
        let mut qv = QVoiceFile::read_from_file(path)?;
        if qv.voice_name.is_empty() {
            qv.voice_name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_string();
        }
        // Fotogramas ICL del archivo antes del recorte, solo para el log.
        let ref_text_empty = opts
            .ref_text_override
            .as_deref()
            .map_or(qv.ref_text.is_empty(), str::is_empty);
        let file_icl_frames = (opts.use_icl && !ref_text_empty && !qv.ref_codes.is_empty())
            .then_some(qv.ref_codes.len() / NUM_CODEBOOKS);
        let voice = Self::from_qvoice(qv, opts);
        if !opts.log.is_silent() {
            eprintln!(
                "Voice clone: loaded .qvoice v4 from {}{}",
                path.display(),
                match file_icl_frames {
                    Some(n) => format!(" ({n} ICL frames)"),
                    None => String::new(),
                }
            );
        }
        Ok(voice)
    }

    /// Construye la voz desde un perfil ya leído. `voice_name` vacío da una voz sin nombre.
    pub fn from_qvoice(qv: QVoiceFile, opts: &VoiceLoadOptions) -> Self {
        let ref_text = opts
            .ref_text_override
            .clone()
            .unwrap_or_else(|| qv.ref_text.clone());
        let icl = if opts.use_icl && !qv.ref_codes.is_empty() && !ref_text.is_empty() {
            let mut ref_codes = qv.ref_codes;
            cap_ref_codes(&mut ref_codes, opts.icl_frames_cap);
            Some(IclPrompt {
                ref_text,
                ref_codes,
            })
        } else {
            None
        };
        let language_id = if qv.lang_id == 0 {
            None
        } else {
            Some(qv.lang_id as i32)
        };
        Self {
            name: qv.voice_name,
            language_id,
            xvector: qv.speaker_embedding,
            icl,
        }
    }
}

/// Clona una voz desde un audio de referencia.
///
/// Devuelve el perfil completo (`QVoiceFile`, `ref_codes` sin recortar, `voice_name` vacío si no
/// se dio nombre y `lang_id` 0 si no se dio idioma), que es lo que se guarda, y la voz ya
/// construida (ICL recortado al tope y nombre resuelto: el dado o el del WAV).
pub fn clone_voice(
    runtime: &ComputeRuntime,
    model: &Model06B,
    ref_wav: &Path,
    opts: &CloneOptions,
) -> Result<(QVoiceFile, Voice), EngineError> {
    let audio = load_and_trim_ref_audio(ref_wav, opts.max_ref_seconds, opts.log.is_silent())?;
    let (mel, n_frames) = mel_spectrogram(&audio, SAMPLE_RATE)?;
    if !opts.log.is_silent() {
        eprintln!("Mel spectrogram: {n_frames} frames x {MEL_DIM} mels");
    }
    let speaker_embedding =
        speaker_encoder_forward(runtime, &model.speaker_encoder, &mel, n_frames);

    let need_icl = opts.ref_text.as_deref().is_some_and(|t| !t.is_empty());
    let mut ref_codes = Vec::new();
    if need_icl {
        let (codes, frames) = speech_encoder_encode(
            runtime,
            &model.speech_encoder,
            &audio,
            opts.log.is_debug(),
            opts.log.is_silent(),
        )?;
        ref_codes = codes;
        if !opts.log.is_silent() {
            eprintln!("Encoded ref audio: {frames} ICL frames");
        }
    }

    let ref_text = if ref_codes.is_empty() {
        String::new()
    } else {
        opts.ref_text.clone().unwrap_or_default()
    };

    let qv = QVoiceFile {
        speaker_embedding,
        voice_name: opts.name.clone().unwrap_or_default(),
        lang_id: opts.language_id.map_or(0, |l| l as u32),
        ref_text,
        ref_codes,
    };

    let name = match opts.name.as_deref() {
        Some(n) if !n.is_empty() => n.to_string(),
        _ => ref_wav
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string(),
    };
    let icl = if qv.ref_codes.is_empty() {
        None
    } else {
        let mut ref_codes = qv.ref_codes.clone();
        cap_ref_codes(&mut ref_codes, opts.icl_frames_cap);
        Some(IclPrompt {
            ref_text: qv.ref_text.clone(),
            ref_codes,
        })
    };
    let voice = Voice {
        name,
        language_id: opts.language_id,
        xvector: qv.speaker_embedding.clone(),
        icl,
    };
    Ok((qv, voice))
}
