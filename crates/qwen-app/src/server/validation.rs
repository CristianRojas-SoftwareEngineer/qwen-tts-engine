// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::ops::RangeInclusive;

use qwen_core::config::{lookup_language, CODEC_VOCAB_SIZE};
use qwen_core::{SamplingParams, SeedChoice};
use qwen_engine::{SynthesisRequest, DEFAULT_CHUNK_FRAMES, MAX_CHUNK_FRAMES, MIN_CHUNK_FRAMES};
use serde_json::Value;

use crate::request::{
    RATE_RANGE, REP_PENALTY_RANGE, TEMPERATURE_RANGE, TOP_P_RANGE, VOLUME_RANGE,
};

pub const MAX_TTS_TEXT: usize = 8192;

/// Campos que admiten ambas rutas de síntesis.
const COMMON_FIELDS: [&str; 9] = [
    "text",
    "language",
    "voice",
    "temperature",
    "top_k",
    "top_p",
    "rep_penalty",
    "seed",
    "volume",
];

#[derive(Debug, Clone)]
pub struct ValidatedTtsRequest {
    pub text: String,
    pub language_id: Option<i32>,
    /// Temperatura pedida; `None` deja el valor por defecto (o el de la paralingüística).
    pub temperature: Option<f32>,
    pub top_k: i32,
    pub top_p: f32,
    pub rep_penalty: f32,
    /// Semilla pedida (`Clock` cuando el cliente envía un valor `< 0`).
    pub seed: SeedChoice,
    pub volume: f32,
    pub rate: f32,
    pub chunk_frames: usize,
}

fn is_only_whitespace_or_escaped(s: &str) -> bool {
    let bytes = s.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'\\' && i + 1 < bytes.len() && matches!(bytes[i + 1], b'n' | b't' | b'r') {
            i += 2;
            continue;
        }
        if !matches!(bytes[i], b' ' | b'\t' | b'\n' | b'\r') {
            return false;
        }
        i += 1;
    }
    true
}

fn extract_opt_str<'a>(
    obj: &'a serde_json::Map<String, Value>,
    key: &str,
) -> Result<Option<&'a str>, String> {
    match obj.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.as_str())),
        Some(_) => Err(format!("'{key}' inválido (se esperaba una cadena)")),
    }
}

fn extract_opt_f64(obj: &serde_json::Map<String, Value>, key: &str) -> Result<Option<f64>, String> {
    match obj.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(n)) => n
            .as_f64()
            .filter(|v| v.is_finite())
            .map(Some)
            .ok_or_else(|| format!("'{key}' inválido")),
        Some(_) => Err(format!("'{key}' inválido (se esperaba un número)")),
    }
}

/// Lee `key` como número dentro de `range` (`None` si falta). Compara en `f64` para no aceptar
/// un valor que solo entraría en el rango al redondearlo a `f32`.
fn extract_opt_in_range(
    obj: &serde_json::Map<String, Value>,
    key: &str,
    range: &RangeInclusive<f32>,
) -> Result<Option<f32>, String> {
    match extract_opt_f64(obj, key)? {
        Some(v) if (f64::from(*range.start())..=f64::from(*range.end())).contains(&v) => {
            Ok(Some(v as f32))
        }
        Some(_) => Err(format!(
            "'{key}' fuera de rango [{}, {}]",
            range.start(),
            range.end()
        )),
        None => Ok(None),
    }
}

fn extract_opt_i64(obj: &serde_json::Map<String, Value>, key: &str) -> Result<Option<i64>, String> {
    match obj.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(n)) => {
            if let Some(v) = n.as_i64() {
                Ok(Some(v))
            } else if let Some(f) = n.as_f64() {
                if f.is_finite() {
                    Ok(Some(f as i64))
                } else {
                    Err(format!("'{key}' inválido"))
                }
            } else {
                Err(format!("'{key}' inválido"))
            }
        }
        Some(_) => Err(format!("'{key}' inválido (se esperaba un entero)")),
    }
}

impl ValidatedTtsRequest {
    pub fn parse(body: &str, is_stream: bool, loaded_voice_name: &str) -> Result<Self, String> {
        let val: Value =
            serde_json::from_str(body).map_err(|_| "cuerpo JSON inválido".to_string())?;
        let obj = val
            .as_object()
            .ok_or_else(|| "el cuerpo JSON debe ser un objeto".to_string())?;

        // Lista estricta de campos: `rate` solo en `/v1/tts` y `chunk_frames` solo en streaming.
        for key in obj.keys() {
            let known = COMMON_FIELDS.contains(&key.as_str())
                || (!is_stream && key == "rate")
                || (is_stream && key == "chunk_frames");
            if !known {
                return Err(format!("campo desconocido '{key}'"));
            }
        }

        // text (obligatorio, no vacío, no solo espacios, <= 8192 bytes)
        let Some(text) = extract_opt_str(obj, "text")? else {
            return Err("falta 'text' o está vacío".to_string());
        };
        if text.is_empty() {
            return Err("falta 'text' o está vacío".to_string());
        }
        if is_only_whitespace_or_escaped(text) {
            return Err("'text' solo contiene espacios en blanco".to_string());
        }
        if text.len() > MAX_TTS_TEXT {
            return Err(format!("'text' excede {MAX_TTS_TEXT} bytes"));
        }

        // idioma
        let language_name = extract_opt_str(obj, "language")?;
        let language_id = match language_name {
            Some(lang) => {
                Some(lookup_language(lang).ok_or_else(|| format!("idioma desconocido '{lang}'"))?)
            }
            None => None,
        };

        // voice (opcional): debe coincidir con la voz cargada
        if let Some(v) = extract_opt_str(obj, "voice")? {
            if !v.eq_ignore_ascii_case(loaded_voice_name) {
                return Err(format!(
                    "la voz '{v}' no está cargada (cargada: '{loaded_voice_name}')"
                ));
            }
        }

        let defaults = SamplingParams::default();
        let req_defaults = SynthesisRequest::default();

        // Sin valor, cada parámetro toma el de `SamplingParams::default()` o el de
        // `SynthesisRequest::default()`; la temperatura queda en `None` (la resuelve el marcado).
        let temperature = extract_opt_in_range(obj, "temperature", &TEMPERATURE_RANGE)?;

        // top_k en [0, CODEC_VOCAB_SIZE]
        let top_k = match extract_opt_i64(obj, "top_k")? {
            Some(v) => {
                if v < 0 || v > CODEC_VOCAB_SIZE as i64 {
                    return Err(format!("'top_k' fuera de rango [0, {CODEC_VOCAB_SIZE}]"));
                }
                v as i32
            }
            None => defaults.top_k as i32,
        };

        let top_p = extract_opt_in_range(obj, "top_p", &TOP_P_RANGE)?.unwrap_or(defaults.top_p);
        let rep_penalty = extract_opt_in_range(obj, "rep_penalty", &REP_PENALTY_RANGE)?
            .unwrap_or(defaults.rep_penalty);

        // semilla: sin valor manda el defecto de `SamplingParams`; valores negativos activan reloj
        let seed = match extract_opt_i64(obj, "seed")? {
            Some(v) if v < 0 => SeedChoice::Clock,
            Some(v) if v <= u32::MAX as i64 => SeedChoice::Fixed(v as u32),
            Some(_) => return Err("'seed' excede u32::MAX".to_string()),
            None => SeedChoice::Default,
        };

        let volume =
            extract_opt_in_range(obj, "volume", &VOLUME_RANGE)?.unwrap_or(req_defaults.volume);
        // En streaming `rate` es siempre 1: la ruta no admite el campo.
        let rate = if is_stream {
            1.0
        } else {
            extract_opt_in_range(obj, "rate", &RATE_RANGE)?.unwrap_or(req_defaults.rate)
        };

        // chunk_frames (acotado a MIN_CHUNK_FRAMES..=MAX_CHUNK_FRAMES; sin valor, DEFAULT_CHUNK_FRAMES)
        let chunk_frames = if is_stream {
            let cf = extract_opt_i64(obj, "chunk_frames")?.unwrap_or(DEFAULT_CHUNK_FRAMES as i64);
            cf.clamp(MIN_CHUNK_FRAMES as i64, MAX_CHUNK_FRAMES as i64) as usize
        } else {
            DEFAULT_CHUNK_FRAMES
        };

        Ok(Self {
            text: text.to_string(),
            language_id,
            temperature,
            top_k,
            top_p,
            rep_penalty,
            seed,
            volume,
            rate,
            chunk_frames,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(body: &str, is_stream: bool) -> Result<ValidatedTtsRequest, String> {
        ValidatedTtsRequest::parse(body, is_stream, "default")
    }

    #[test]
    fn test_http_400_validations() {
        assert!(parse(r#"{"text":"   \n\t "}"#, false).is_err());
        assert!(parse(r#"{"text":"   \\n\\t "}"#, false).is_err());
        assert!(parse(r#"{"text":""}"#, false).is_err());
        assert!(parse(r#"{"text":"Hi.","language":"Klingon"}"#, false).is_err());
        assert!(parse(r#"{"text":"Hi.","temperature":5}"#, false).is_err());
        assert!(parse(r#"{"text":"Hi.","top_k":4000}"#, false).is_err());
        assert!(parse(r#"{"text":"Hi.","top_p":1.5}"#, false).is_err());
        assert!(parse(r#"{"text":"Hi.","rep_penalty":0.2}"#, false).is_err());
        assert!(parse(r#"{"text":"Hi.","rate":0.01}"#, false).is_err());
        assert!(parse(r#"{"text":"Hi.","volume":-1.0}"#, false).is_err());
        assert!(parse(r#"{"text":"Hi.","seed":99999999999}"#, false).is_err());

        let ok = parse(r#"{"text":"Hello!","rate":1.2}"#, false).unwrap();
        assert_eq!(ok.text, "Hello!");
        assert_eq!(ok.temperature, None);
        assert_eq!(ok.seed, SeedChoice::Default);
        assert!((ok.rate - 1.2).abs() < 1e-6);

        let neg_seed = parse(r#"{"text":"Hi.","seed":-1}"#, false).unwrap();
        assert_eq!(neg_seed.seed, SeedChoice::Clock);

        // Sin `language` queda `None`; la voz o el defecto de la interfaz lo resuelven después.
        let clone_ok = parse(r#"{"text":"Cloned.","chunk_frames":999}"#, true).unwrap();
        assert_eq!(clone_ok.language_id, None);
        assert_eq!(clone_ok.chunk_frames, 250);
        let clone_min = parse(r#"{"text":"Cloned.","chunk_frames":0}"#, true).unwrap();
        assert_eq!(clone_min.chunk_frames, 2);
    }

    #[test]
    fn test_voice_field_must_match_loaded_voice() {
        assert!(parse(r#"{"text":"Hi."}"#, false).is_ok());
        assert!(parse(r#"{"text":"Hi.","voice":"default"}"#, false).is_ok());
        assert!(parse(r#"{"text":"Hi.","voice":"DEFAULT"}"#, false).is_ok());
        let err = parse(r#"{"text":"Hi.","voice":"otra"}"#, false).unwrap_err();
        assert_eq!(err, "la voz 'otra' no está cargada (cargada: 'default')");
        assert!(parse(r#"{"text":"Hi.","voice":5}"#, false).is_err());
    }

    #[test]
    fn test_unknown_fields_are_rejected() {
        let err = parse(r#"{"text":"Hi.","foo":"x","bar":1}"#, false).unwrap_err();
        assert!(err.starts_with("campo desconocido '"), "{err}");
        assert!(parse(r#"{"foo":"Hi."}"#, false).is_err());
        // `null` cuenta como ausente
        assert!(parse(r#"{"text":"Hi.","seed":null}"#, false).is_ok());
    }

    #[test]
    fn test_route_specific_fields() {
        assert!(parse(r#"{"text":"Hi.","rate":1.2}"#, true).is_err());
        assert!(parse(r#"{"text":"Hi.","chunk_frames":4}"#, false).is_err());
        assert!(parse(r#"{"text":"Hi.","rate":1.2}"#, false).is_ok());
        assert!(parse(r#"{"text":"Hi.","chunk_frames":4}"#, true).is_ok());
    }
}
