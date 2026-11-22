// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! La voz de una síntesis: x-vector del locutor y, opcionalmente, el contexto ICL (texto y
//! códigos del audio de referencia).

use std::path::Path;

use qwen_core::config::{DEFAULT_LANGUAGE_ID, MEL_DIM, SAMPLE_RATE};
use qwen_core::QVoiceFile;
use qwen_kernels::ComputeRuntime;

use crate::error::EngineError;
use crate::model::Model06B;
use crate::request::LogLevel;
use crate::speech_encoder::speech_encoder_encode;
use crate::voice_clone::{load_ref_audio, mel_spectrogram, speaker_encoder_forward};

/// x-vector de la voz `default` incorporada (1024 `f32` little-endian).
const DEFAULT_VOICE_XVEC: &[u8] = include_bytes!("../assets/default_voice.xvec");
const _: () = assert!(DEFAULT_VOICE_XVEC.len() == 4096);

/// Nombre de la voz incorporada.
pub const DEFAULT_VOICE_NAME: &str = "default";

/// Contexto ICL de una voz: transcripción y códigos del audio de referencia.
///
/// Nunca está vacío.
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
    /// Usa el contexto ICL si el archivo lo trae; `false` deja solo el x-vector.
    pub use_icl: bool,
    pub log: LogLevel,
}

/// Opciones para clonar una voz desde un audio de referencia.
#[derive(Debug, Clone, Default)]
pub struct CloneOptions {
    /// Transcripción del audio; sin ella la voz es solo x-vector.
    pub ref_text: Option<String>,
    pub language_id: Option<i32>,
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
            "los archivos de voz deben ser .qvoice (se obtuvo {})",
            path.display()
        )))
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

    /// Carga un perfil `.qvoice` v5. La voz se llama como el archivo (sin extensión).
    pub fn load(path: &Path, opts: &VoiceLoadOptions) -> Result<Self, EngineError> {
        ensure_qvoice_path(path)?;
        let qv = QVoiceFile::read_from_file(path)?;
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| {
                EngineError::InvalidArg(format!(
                    "no se puede obtener el nombre de la voz de {}",
                    path.display()
                ))
            })?
            .to_string();
        let voice = Self::from_qvoice(qv, name, opts);
        if !opts.log.is_silent() {
            eprintln!(
                "Clonación de voz: .qvoice v5 cargado desde {}",
                path.display()
            );
        }
        Ok(voice)
    }

    /// Construye la voz desde un perfil ya leído, con el nombre dado. Es el único camino que
    /// construye una voz desde un perfil, al cargar y al clonar.
    pub fn from_qvoice(qv: QVoiceFile, name: String, opts: &VoiceLoadOptions) -> Self {
        let icl = if opts.use_icl && !qv.ref_codes.is_empty() && !qv.ref_text.is_empty() {
            Some(IclPrompt {
                ref_text: qv.ref_text,
                ref_codes: qv.ref_codes,
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
            name,
            language_id,
            xvector: qv.speaker_embedding,
            icl,
        }
    }
}

/// Clona una voz desde un audio de referencia.
///
/// Devuelve solo el perfil (`QVoiceFile`, con `lang_id` 0 si no se dio idioma): es lo que se
/// guarda y de lo que se construye la voz con [`Voice::from_qvoice`]. Hay ICL si y solo si
/// `opts.ref_text` es `Some`; sin él, el perfil es solo x-vector.
pub fn clone_voice(
    runtime: &ComputeRuntime,
    model: &Model06B,
    ref_wav: &Path,
    opts: &CloneOptions,
) -> Result<QVoiceFile, EngineError> {
    let need_icl = opts.ref_text.is_some();
    if opts.ref_text.as_deref().is_some_and(|t| t.trim().is_empty()) {
        return Err(EngineError::InvalidArg(
            "la transcripción de la referencia (ref_text) no puede estar vacía".to_string(),
        ));
    }

    let audio = load_ref_audio(ref_wav, opts.log.is_silent())?;
    let (mel, n_frames) = mel_spectrogram(&audio, SAMPLE_RATE)?;
    if !opts.log.is_silent() {
        eprintln!("Espectrograma mel: {n_frames} frames x {MEL_DIM} mels");
    }
    let speaker_embedding =
        speaker_encoder_forward(runtime, &model.speaker_encoder, &mel, n_frames);

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
            eprintln!("Audio de referencia codificado: {frames} frames ICL");
        }
    }

    Ok(QVoiceFile {
        speaker_embedding,
        lang_id: opts.language_id.map_or(0, |l| l as u32),
        ref_text: opts.ref_text.clone().unwrap_or_default(),
        ref_codes,
    })
}
