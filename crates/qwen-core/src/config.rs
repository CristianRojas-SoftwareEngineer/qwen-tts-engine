// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::path::Path;

use serde::Deserialize;

use crate::error::CoreError;

// Geometría fija de Qwen3-TTS 0.6B Base: el motor solo carga este modelo, así que las dimensiones son constantes de compilación
pub const SAMPLE_RATE: usize = 24000;
pub const FRAME_RATE: f32 = 12.5;
pub const HOP_SAMPLES: usize = 1920;
/// Temperatura de muestreo por defecto aplicada cuando no se especifica explícitamente en CLI, HTTP o Rust API.
pub const DEFAULT_TEMPERATURE: f32 = 0.325;
/// Semilla pseudoaleatoria por defecto aplicada cuando no se especifica explícitamente en CLI, HTTP o Rust API.
pub const DEFAULT_SEED: u32 = 4;

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
pub const DEC_INTERMEDIATE_SIZE: usize = 1024;
pub const DEC_NUM_HEADS: usize = 16;
pub const DEC_HEAD_DIM: usize = 64;
pub const DEC_SLIDING_WINDOW: usize = 72;
pub const DEC_RMS_NORM_EPS: f32 = 1e-5;
pub const DEC_ROPE_THETA: f32 = 10_000.0;
pub const DEC_UPSAMPLE_RATES: [usize; 4] = [8, 5, 4, 3];
pub const DEC_UPSAMPLE_CHANNELS: [usize; 4] = [768, 384, 192, 96];

// Speaker Encoder
pub const ENC_DIM: usize = 1024;
pub const MEL_DIM: usize = 128;

// IDs de tokens especiales de texto y códec
pub const TOKEN_IM_START: i32 = 151_644;
pub const TOKEN_ASSISTANT: i32 = 77_091;
pub const TOKEN_NEWLINE: i32 = 198;
pub const TOKEN_TTS_PAD: i32 = 151_671;
pub const TOKEN_TTS_BOS: i32 = 151_672;
pub const TOKEN_TTS_EOS: i32 = 151_673;

pub const CODEC_PAD: i32 = 2148;
pub const CODEC_BOS: i32 = 2149;
pub const CODEC_EOS: i32 = 2150;
pub const CODEC_THINK: i32 = 2154;
pub const CODEC_THINK_BOS: i32 = 2156;
pub const CODEC_THINK_EOS: i32 = 2157;

/// Idioma por defecto cuando ni el pedido ni la voz declaran uno: español.
pub const DEFAULT_LANGUAGE_ID: i32 = 2054;

/// Única regla de idioma: el pedido (`-l`/`language`); si no hay, el de la voz; si tampoco,
/// [`DEFAULT_LANGUAGE_ID`].
pub fn effective_language(requested: Option<i32>, voice: Option<i32>) -> i32 {
    requested.or(voice).unwrap_or(DEFAULT_LANGUAGE_ID)
}

pub fn lookup_language(name: &str) -> Option<i32> {
    match name.trim().to_ascii_lowercase().as_str() {
        "english" => Some(2050),
        "german" => Some(2053),
        "spanish" => Some(2054),
        "chinese" => Some(2055),
        "japanese" => Some(2058),
        "french" => Some(2061),
        "korean" => Some(2064),
        "russian" => Some(2069),
        "italian" => Some(2070),
        "portuguese" => Some(2071),
        _ => None,
    }
}

/// Inversa de [`lookup_language`]: nombre capitalizado del id de idioma.
pub fn language_name(id: i32) -> Option<&'static str> {
    match id {
        2050 => Some("English"),
        2053 => Some("German"),
        2054 => Some("Spanish"),
        2055 => Some("Chinese"),
        2058 => Some("Japanese"),
        2061 => Some("French"),
        2064 => Some("Korean"),
        2069 => Some("Russian"),
        2070 => Some("Italian"),
        2071 => Some("Portuguese"),
        _ => None,
    }
}

#[derive(Debug, Clone, Deserialize)]
struct RawConfig {
    model_type: Option<String>,
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
    pub rms_norm_eps: f32,
}

/// Error de validación con el prefijo común de los modelos no soportados.
fn unsupported(detail: String) -> CoreError {
    CoreError::InvalidConfig(format!("Este motor solo carga Qwen3-TTS 0.6B Base: {detail}"))
}

impl ModelConfig06B {
    pub fn load_and_validate(model_dir: &Path) -> Result<Self, CoreError> {
        let cfg_path = model_dir.join("config.json");
        let raw_str = std::fs::read_to_string(&cfg_path).map_err(|e| {
            CoreError::InvalidConfig(format!("No se puede leer {}: {e}", cfg_path.display()))
        })?;
        let raw: RawConfig = serde_json::from_str(&raw_str)?;

        if let Some(mt) = raw.model_type.as_deref() {
            if mt != "qwen3_tts" {
                return Err(unsupported(format!(
                    "model_type '{mt}' no soportado (se esperaba 'qwen3_tts')"
                )));
            }
        }

        let tts_model_type = raw.tts_model_type.clone().unwrap_or_default();

        let tc = &raw.talker_config;
        if tc.hidden_size != HIDDEN_SIZE || tc.intermediate_size != INTERMEDIATE_SIZE {
            return Err(unsupported(format!(
                "Geometría del Talker no soportada: hidden_size={} intermediate_size={} (se esperaba {HIDDEN_SIZE}/{INTERMEDIATE_SIZE})",
                tc.hidden_size, tc.intermediate_size
            )));
        }
        if tc.num_hidden_layers != NUM_LAYERS
            || tc.num_attention_heads != NUM_HEADS
            || tc.num_key_value_heads != NUM_KV_HEADS
            || tc.head_dim != HEAD_DIM
        {
            return Err(unsupported(format!(
                "Geometría de atención y capas del Talker no soportada: layers={} heads={} kv={} head_dim={}",
                tc.num_hidden_layers, tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim
            )));
        }
        if let Some(th) = tc.text_hidden_size {
            if th != TEXT_HIDDEN_SIZE {
                return Err(unsupported(format!(
                    "text_hidden_size={th} no soportado (se esperaba {TEXT_HIDDEN_SIZE})"
                )));
            }
        }
        if let Some(tv) = tc.text_vocab_size {
            if tv != TEXT_VOCAB_SIZE {
                return Err(unsupported(format!(
                    "text_vocab_size={tv} no soportado (se esperaba {TEXT_VOCAB_SIZE})"
                )));
            }
        }
        if let Some(cv) = tc.vocab_size {
            if cv != CODEC_VOCAB_SIZE {
                return Err(unsupported(format!(
                    "vocab_size={cv} del códec no soportado (se esperaba {CODEC_VOCAB_SIZE})"
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
            return Err(unsupported(format!(
                "Geometría del Code Predictor no soportada: h={} inter={} layers={} heads={} kv={} hd={}",
                cp.hidden_size,
                cp.intermediate_size,
                cp.num_hidden_layers,
                cp.num_attention_heads,
                cp.num_key_value_heads,
                cp.head_dim
            )));
        }

        if !tts_model_type.eq_ignore_ascii_case("base") {
            return Err(unsupported(format!(
                "tts_model_type '{tts_model_type}' no soportado (se esperaba 'base')"
            )));
        }
        let Some(sec) = raw.speaker_encoder_config.as_ref() else {
            return Err(unsupported("falta speaker_encoder_config".to_string()));
        };
        if let Some(ed) = sec.enc_dim {
            if ed != ENC_DIM {
                return Err(unsupported(format!(
                    "enc_dim={ed} del speaker_encoder no soportado (se esperaba {ENC_DIM})"
                )));
            }
        }
        if let Some(sr) = sec.sample_rate {
            if sr != SAMPLE_RATE {
                return Err(unsupported(format!(
                    "sample_rate={sr} del speaker_encoder no soportado (se esperaba {SAMPLE_RATE})"
                )));
            }
        }

        Ok(Self {
            rms_norm_eps: tc.rms_norm_eps.unwrap_or(RMS_NORM_EPS),
        })
    }
}
