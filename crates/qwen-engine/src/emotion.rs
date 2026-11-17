// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::path::{Path, PathBuf};

use qwen_core::config::{ENC_DIM, SAMPLE_RATE};
use qwen_core::{emotion_name_to_tok, read_bin_embedding};
use qwen_kernels::ComputeRuntime;

use crate::audio::{read_wav_24khz, trim_trailing_silence};
use crate::error::EngineError;
use crate::model::{Model06B, VoiceBindings};
use crate::speech_encoder::speech_encoder_encode;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmoVoice06BAction {
    LoadQVoiceIclOnly(PathBuf),
    LoadBinXVector(PathBuf),
    ApplyDirection {
        tok: &'static str,
        dir_path: PathBuf,
    },
}

pub fn emotion_voice_key(load_voice: Option<&Path>, speaker_name: Option<&str>) -> String {
    if let Some(p) = load_voice {
        let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        let trimmed = stem
            .split("_graft")
            .next()
            .unwrap_or(stem)
            .split("_06b")
            .next()
            .unwrap_or(stem);
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    speaker_name
        .filter(|s| !s.is_empty())
        .unwrap_or("ryan")
        .to_string()
}

/// Resuelve `--emotion <name>` en `0.6B` (`SPEC_06B.md` §3.2).
pub fn resolve_emovoice_06b(
    emotion_spec: &str,
    load_voice: Option<&Path>,
    has_clone_voice: bool,
    speaker_name: Option<&str>,
    preset_dirs: &[&Path],
) -> Result<EmoVoice06BAction, EngineError> {
    let tok = emotion_name_to_tok(emotion_spec).ok_or_else(|| {
        EngineError::InvalidArg(format!("Error: '{emotion_spec}' is not a known emotion"))
    })?;
    let key = emotion_voice_key(load_voice, speaker_name);

    let mut search_dirs: Vec<PathBuf> = Vec::new();
    if let Some(parent) = load_voice.and_then(|p| p.parent()) {
        if !parent.as_os_str().is_empty() {
            search_dirs.push(parent.to_path_buf());
        }
    }
    if preset_dirs.is_empty() {
        search_dirs.push(PathBuf::from("presets/emovoice"));
    } else {
        for d in preset_dirs {
            search_dirs.push(d.to_path_buf());
        }
    }

    for dir in &search_dirs {
        let qv = dir.join(format!("{key}_{tok}.qvoice"));
        if qv.is_file() {
            return Ok(EmoVoice06BAction::LoadQVoiceIclOnly(qv));
        }
        let bin = dir.join(format!("{key}_{tok}.bin"));
        if bin.is_file() {
            return Ok(EmoVoice06BAction::LoadBinXVector(bin));
        }
    }

    if has_clone_voice || load_voice.is_some() {
        let fallback_dirs: Vec<PathBuf> = if preset_dirs.is_empty() {
            vec![PathBuf::from("presets/emovoice")]
        } else {
            preset_dirs.iter().map(|d| d.to_path_buf()).collect()
        };
        for dir in &fallback_dirs {
            let dpath = dir.join(format!("dir_{tok}.bin"));
            if dpath.is_file() {
                return Ok(EmoVoice06BAction::ApplyDirection {
                    tok,
                    dir_path: dpath,
                });
            }
        }
    }

    Err(EngineError::InvalidArg(format!(
        "Error: no '{tok}' voice asset for '{key}' on the 0.6B (missing {key}_{tok}.{{qvoice,bin}} or dir_{tok}.bin on clone)."
    )))
}

/// Aplica el desplazamiento `dir_<tok>.bin` sobre un x-vector ECAPA preservando su norma L2 original `n0`.
pub fn apply_emotion_direction(
    speaker_embedding: &mut [f32],
    dir_path: &Path,
    strength: f32,
) -> Result<(), EngineError> {
    if speaker_embedding.len() != ENC_DIM {
        return Err(EngineError::InvalidArg(format!(
            "Expected speaker_embedding of length {ENC_DIM}, got {}",
            speaker_embedding.len()
        )));
    }
    let dir = read_bin_embedding(dir_path)?;
    let n0 = speaker_embedding
        .iter()
        .map(|&v| v * v)
        .sum::<f32>()
        .sqrt();
    for i in 0..ENC_DIM {
        speaker_embedding[i] += strength * n0 * dir[i];
    }
    let n1 = speaker_embedding
        .iter()
        .map(|&v| v * v)
        .sum::<f32>()
        .sqrt();
    if n1 > 1e-6f32 {
        let scale = n0 / n1;
        for v in speaker_embedding.iter_mut() {
            *v *= scale;
        }
    }
    Ok(())
}

/// Codifica `--emo-ref <wav>` con `--emo-ref-text <str>` y lo adjunta como ancla de prosodia ICL
/// sin alterar la identidad del hablante (`VoiceBindings.identity`).
pub fn attach_emo_ref(
    runtime: &ComputeRuntime,
    model: &Model06B,
    voice: &mut VoiceBindings,
    emo_ref_path: &Path,
    emo_ref_text: &str,
    silent: bool,
    debug: bool,
) -> Result<(), EngineError> {
    if emo_ref_text.trim().is_empty() {
        return Err(EngineError::InvalidArg(
            "Error: --emo-ref requires --emo-ref-text (the reference's transcript)".to_string(),
        ));
    }
    let enc = model.speech_encoder.as_ref().ok_or_else(|| {
        EngineError::Model(
            "Error: --emo-ref needs the speech encoder (speech_tokenizer/) and it failed to load"
                .to_string(),
        )
    })?;

    let mut audio = read_wav_24khz(emo_ref_path)?;
    trim_trailing_silence(&mut audio, SAMPLE_RATE, silent);
    let (codes, n_frames) = speech_encoder_encode(runtime, enc, &audio, debug, silent)?;
    if !silent {
        eprintln!(
            "Emotion-by-example: {n_frames} ref frames from {} (identity unchanged)",
            emo_ref_path.display()
        );
    }
    voice.emo_ref_text = Some(emo_ref_text.to_string());
    voice.emo_ref_codes = Some(codes);
    Ok(())
}
