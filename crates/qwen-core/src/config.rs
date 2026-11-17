// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

use crate::error::CoreError;

// Geometría fija de Qwen3-TTS 0.6B (`SPEC_06B.md`, `ADR-001`)
pub const SAMPLE_RATE: usize = 24000;
pub const FRAME_RATE: f32 = 12.5;
pub const HOP_SAMPLES: usize = 1920;
/// Temperatura de muestreo por defecto aplicada cuando no se especifica explícitamente en CLI, HTTP o Rust API.
pub const DEFAULT_TEMPERATURE: f32 = 0.325;

pub const TEXT_VOCAB_SIZE: usize = 151_936;
pub const CODEC_VOCAB_SIZE: usize = 3072;
pub const NUM_CODEBOOKS: usize = 16;
pub const CODEBOOK_SIZE: usize = 2048;
pub const CODEBOOK_DIM: usize = 256;

// Talker 0.6B
pub const TEXT_HIDDEN_SIZE: usize = 2048;
pub const HIDDEN_SIZE: usize = 1024;
pub const NUM_LAYERS: usize = 28;
pub const NUM_HEADS: usize = 16;
pub const NUM_KV_HEADS: usize = 8;
pub const HEAD_DIM: usize = 128;
pub const Q_DIM: usize = NUM_HEADS * HEAD_DIM; // 2048
pub const KV_DIM: usize = NUM_KV_HEADS * HEAD_DIM; // 1024
pub const INTERMEDIATE_SIZE: usize = 3072;
pub const RMS_NORM_EPS: f32 = 1e-6;
pub const ROPE_THETA: f32 = 1_000_000.0;

// Code Predictor 0.6B (sin `small_to_mtp_projection`)
pub const CP_HIDDEN_SIZE: usize = 1024;
pub const CP_NUM_LAYERS: usize = 5;
pub const CP_NUM_HEADS: usize = 16;
pub const CP_NUM_KV_HEADS: usize = 8;
pub const CP_HEAD_DIM: usize = 128;
pub const CP_Q_DIM: usize = CP_NUM_HEADS * CP_HEAD_DIM; // 2048
pub const CP_KV_DIM: usize = CP_NUM_KV_HEADS * CP_HEAD_DIM; // 1024
pub const CP_INTERMEDIATE_SIZE: usize = 3072;

// Speech Decoder 0.6B
pub const DEC_NUM_LAYERS: usize = 8;
pub const DEC_HIDDEN_SIZE: usize = 512;
pub const DEC_LATENT_DIM: usize = 1024;
pub const DEC_CODEBOOK_DIM: usize = 512;
pub const DEC_DECODER_DIM: usize = 1536;
pub const DEC_INTERMEDIATE_SIZE: usize = 1024;
pub const DEC_NUM_HEADS: usize = 16;
pub const DEC_HEAD_DIM: usize = 64;
pub const DEC_SLIDING_WINDOW: usize = 72;
pub const DEC_RMS_NORM_EPS: f32 = 1e-5;
pub const DEC_ROPE_THETA: f32 = 10_000.0;
pub const DEC_UPSAMPLE_RATES: [usize; 4] = [8, 5, 4, 3];
pub const DEC_UPSAMPLE_CHANNELS: [usize; 4] = [768, 384, 192, 96];
pub const DEC_CONVNEXT_RATIOS: [usize; 2] = [2, 2];

// Speaker Encoder (0.6B-Base)
pub const ENC_DIM: usize = 1024;
pub const MEL_DIM: usize = 128;

// IDs de tokens especiales de texto y códec
pub const TOKEN_ENDOFTEXT: i32 = 151_643;
pub const TOKEN_IM_START: i32 = 151_644;
pub const TOKEN_IM_END: i32 = 151_645;
pub const TOKEN_ASSISTANT: i32 = 77_091;
pub const TOKEN_NEWLINE: i32 = 198;
pub const TOKEN_TTS_PAD: i32 = 151_671;
pub const TOKEN_TTS_BOS: i32 = 151_672;
pub const TOKEN_TTS_EOS: i32 = 151_673;

pub const CODEC_PAD: i32 = 2148;
pub const CODEC_BOS: i32 = 2149;
pub const CODEC_EOS: i32 = 2150;
pub const CODEC_THINK: i32 = 2154;
pub const CODEC_NO_THINK: i32 = 2155;
pub const CODEC_THINK_BOS: i32 = 2156;
pub const CODEC_THINK_EOS: i32 = 2157;

#[derive(Debug, Clone, Copy)]
pub struct SpeakerPreset {
    pub name: &'static str,
    pub id: i32,
    pub language: &'static str,
    pub gender: &'static str,
}

pub const PRESET_SPEAKERS: &[SpeakerPreset] = &[
    SpeakerPreset { name: "ryan", id: 3061, language: "English", gender: "male" },
    SpeakerPreset { name: "aiden", id: 2861, language: "English", gender: "male" },
    SpeakerPreset { name: "vivian", id: 3065, language: "Chinese", gender: "female" },
    SpeakerPreset { name: "serena", id: 3066, language: "Chinese", gender: "female" },
    SpeakerPreset { name: "uncle_fu", id: 3010, language: "Chinese", gender: "male" },
    SpeakerPreset { name: "dylan", id: 2878, language: "Chinese", gender: "male" },
    SpeakerPreset { name: "eric", id: 2875, language: "Chinese", gender: "male" },
    SpeakerPreset { name: "ono_anna", id: 2873, language: "Japanese", gender: "female" },
    SpeakerPreset { name: "sohee", id: 2864, language: "Korean", gender: "female" },
];

pub fn lookup_speaker(name: &str) -> Option<SpeakerPreset> {
    let lower = name.trim().to_ascii_lowercase();
    PRESET_SPEAKERS.iter().copied().find(|s| s.name == lower)
}

pub fn lookup_language(name: &str) -> Option<i32> {
    match name.trim().to_ascii_lowercase().as_str() {
        "english" => Some(2050),
        "german" => Some(2053),
        "spanish" => Some(2054),
        "chinese" => Some(2055),
        "japanese" => Some(2058),
        "french" => Some(2061),
        "sichuan_dialect" => Some(2062),
        "korean" => Some(2064),
        "russian" => Some(2069),
        "italian" => Some(2070),
        "portuguese" => Some(2071),
        "beijing_dialect" => Some(2074),
        _ => None,
    }
}

#[derive(Debug, Clone, Deserialize)]
struct RawConfig {
    model_type: Option<String>,
    tts_model_size: Option<String>,
    tts_model_type: Option<String>,
    speaker_encoder_config: Option<RawSpeakerEncoderConfig>,
    talker_config: RawTalkerConfig,
}

#[derive(Debug, Clone, Deserialize)]
struct RawSpeakerEncoderConfig {
    enc_dim: Option<usize>,
    sample_rate: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
struct RawTalkerConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    text_hidden_size: Option<usize>,
    text_vocab_size: Option<usize>,
    vocab_size: Option<usize>,
    rms_norm_eps: Option<f32>,
    rope_theta: Option<f32>,
    #[serde(default)]
    spk_id: HashMap<String, i32>,
    code_predictor_config: RawCpConfig,
}

#[derive(Debug, Clone, Deserialize)]
struct RawCpConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
}

#[derive(Debug, Clone)]
pub struct ModelConfig06B {
    pub is_base_model: bool,
    pub tts_model_type: String,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub spk_ids: HashMap<String, i32>,
}

impl ModelConfig06B {
    pub fn load_and_validate(model_dir: &Path) -> Result<Self, CoreError> {
        let cfg_path = model_dir.join("config.json");
        let raw_str = std::fs::read_to_string(&cfg_path).map_err(|e| {
            CoreError::InvalidConfig(format!(
                "Cannot read {}: {e}",
                cfg_path.display()
            ))
        })?;
        let raw: RawConfig = serde_json::from_str(&raw_str)?;

        if let Some(mt) = raw.model_type.as_deref() {
            if mt != "qwen3_tts" {
                return Err(CoreError::InvalidConfig(format!(
                    "Unsupported model_type '{mt}' (expected 'qwen3_tts')"
                )));
            }
        }

        let tts_model_type = raw
            .tts_model_type
            .clone()
            .unwrap_or_else(|| "custom_voice".to_string());

        if tts_model_type.eq_ignore_ascii_case("voice_design") {
            return Err(CoreError::InvalidConfig(
                "VoiceDesign (1.7B) models are not supported by this 0.6B-only engine".to_string(),
            ));
        }

        if let Some(sz) = raw.tts_model_size.as_deref() {
            if sz != "0b6" {
                return Err(CoreError::InvalidConfig(format!(
                    "Unsupported tts_model_size '{sz}' (only 0.6B / '0b6' is supported)"
                )));
            }
        }

        let tc = &raw.talker_config;
        if tc.hidden_size != HIDDEN_SIZE || tc.intermediate_size != INTERMEDIATE_SIZE {
            return Err(CoreError::InvalidConfig(format!(
                "Unsupported Talker geometry hidden_size={} intermediate_size={} (expected 0.6B: {HIDDEN_SIZE}/{INTERMEDIATE_SIZE}; 1.7B is not supported)",
                tc.hidden_size, tc.intermediate_size
            )));
        }
        if tc.num_hidden_layers != NUM_LAYERS
            || tc.num_attention_heads != NUM_HEADS
            || tc.num_key_value_heads != NUM_KV_HEADS
            || tc.head_dim != HEAD_DIM
        {
            return Err(CoreError::InvalidConfig(format!(
                "Unsupported Talker attention/layer geometry: layers={} heads={} kv={} head_dim={}",
                tc.num_hidden_layers, tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim
            )));
        }
        if let Some(th) = tc.text_hidden_size {
            if th != TEXT_HIDDEN_SIZE {
                return Err(CoreError::InvalidConfig(format!(
                    "Unsupported text_hidden_size={th} (expected {TEXT_HIDDEN_SIZE})"
                )));
            }
        }
        if let Some(tv) = tc.text_vocab_size {
            if tv != TEXT_VOCAB_SIZE {
                return Err(CoreError::InvalidConfig(format!(
                    "Unsupported text_vocab_size={tv} (expected {TEXT_VOCAB_SIZE})"
                )));
            }
        }
        if let Some(cv) = tc.vocab_size {
            if cv != CODEC_VOCAB_SIZE {
                return Err(CoreError::InvalidConfig(format!(
                    "Unsupported codec vocab_size={cv} (expected {CODEC_VOCAB_SIZE})"
                )));
            }
        }

        let cp = &tc.code_predictor_config;
        if cp.hidden_size != CP_HIDDEN_SIZE
            || cp.intermediate_size != CP_INTERMEDIATE_SIZE
            || cp.num_hidden_layers != CP_NUM_LAYERS
            || cp.num_attention_heads != CP_NUM_HEADS
            || cp.num_key_value_heads != CP_NUM_KV_HEADS
            || cp.head_dim != CP_HEAD_DIM
        {
            return Err(CoreError::InvalidConfig(format!(
                "Unsupported CodePredictor geometry: h={} inter={} layers={} heads={} kv={} hd={}",
                cp.hidden_size,
                cp.intermediate_size,
                cp.num_hidden_layers,
                cp.num_attention_heads,
                cp.num_key_value_heads,
                cp.head_dim
            )));
        }

        let is_base_model =
            tts_model_type == "base" || raw.speaker_encoder_config.is_some();
        if let Some(sec) = &raw.speaker_encoder_config {
            if let Some(ed) = sec.enc_dim {
                if ed != ENC_DIM {
                    return Err(CoreError::InvalidConfig(format!(
                        "Unsupported speaker_encoder enc_dim={ed} (expected {ENC_DIM})"
                    )));
                }
            }
            if let Some(sr) = sec.sample_rate {
                if sr != SAMPLE_RATE {
                    return Err(CoreError::InvalidConfig(format!(
                        "Unsupported speaker_encoder sample_rate={sr} (expected {SAMPLE_RATE})"
                    )));
                }
            }
        }

        Ok(Self {
            is_base_model,
            tts_model_type,
            rms_norm_eps: tc.rms_norm_eps.unwrap_or(RMS_NORM_EPS),
            rope_theta: tc.rope_theta.unwrap_or(ROPE_THETA),
            spk_ids: tc.spk_id.clone(),
        })
    }
}
