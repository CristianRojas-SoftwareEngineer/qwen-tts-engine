// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use qwen_core::config::{lookup_language, CODEC_VOCAB_SIZE};
use qwen_core::SeedChoice;
use qwen_engine::DEFAULT_CHUNK_FRAMES;
use serde_json::Value;

pub const MAX_TTS_TEXT: usize = 8192;

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

        // voice_design es exclusivo de 1.7B -> rechazar con 400
        if let Some(vd) = obj.get("voice_design") {
            let enabled = match vd {
                Value::Bool(b) => *b,
                Value::String(s) => s.eq_ignore_ascii_case("true") || s == "1",
                Value::Number(n) => n.as_i64() == Some(1),
                _ => false,
            };
            if enabled {
                return Err("voice_design requiere el modelo VoiceDesign de 1.7B".to_string());
            }
        }

        // text o input (obligatorio, no vacío, no solo espacios, <= 8192 bytes)
        let text_raw = match extract_opt_str(obj, "text")? {
            Some(t) => Some(t),
            None => extract_opt_str(obj, "input")?,
        };
        let Some(text) = text_raw else {
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

        // temperature en [0.0, 2.0]; sin valor manda el defecto de `SamplingParams`
        let temperature = match extract_opt_f64(obj, "temperature")? {
            Some(v) => {
                if !(0.0..=2.0).contains(&v) {
                    return Err("'temperature' fuera de rango [0, 2]".to_string());
                }
                Some(v as f32)
            }
            None => None,
        };

        // top_k en [0, codec_vocab_size (3072)], por defecto 50
        let top_k = match extract_opt_i64(obj, "top_k")? {
            Some(v) => {
                if v < 0 || v > CODEC_VOCAB_SIZE as i64 {
                    return Err(format!("'top_k' fuera de rango [0, {CODEC_VOCAB_SIZE}]"));
                }
                v as i32
            }
            None => 50,
        };

        // top_p en [0.0, 1.0], por defecto 1.0
        let top_p = match extract_opt_f64(obj, "top_p")? {
            Some(v) => {
                if !(0.0..=1.0).contains(&v) {
                    return Err("'top_p' fuera de rango [0, 1]".to_string());
                }
                v as f32
            }
            None => 1.0,
        };

        // rep_penalty en [0.5, 2.0], por defecto 1.05
        let rep_penalty = match extract_opt_f64(obj, "rep_penalty")? {
            Some(v) => {
                if !(0.5..=2.0).contains(&v) {
                    return Err("'rep_penalty' fuera de rango [0.5, 2]".to_string());
                }
                v as f32
            }
            None => 1.05,
        };

        // semilla: sin valor manda el defecto de `SamplingParams`; valores negativos activan reloj
        let seed = match extract_opt_i64(obj, "seed")? {
            Some(v) if v < 0 => SeedChoice::Clock,
            Some(v) if v <= u32::MAX as i64 => SeedChoice::Fixed(v as u32),
            Some(_) => return Err("'seed' excede u32::MAX".to_string()),
            None => SeedChoice::Default,
        };

        // volumen / velocidad
        let volume = match extract_opt_f64(obj, "volume")? {
            Some(v) => {
                if !(0.0..=10.0).contains(&v) {
                    return Err("'volume' fuera de rango [0, 10]".to_string());
                }
                v as f32
            }
            None => 1.0,
        };
        let rate = match extract_opt_f64(obj, "rate")? {
            Some(v) => {
                if !(0.25..=4.0).contains(&v) {
                    return Err("'rate' fuera de rango [0.25, 4.0]".to_string());
                }
                v as f32
            }
            None => 1.0,
        };
        if is_stream && (rate - 1.0).abs() > 1e-6 {
            return Err("'rate' no está soportado en /v1/tts/stream".to_string());
        }

        // chunk_frames (acotado a 2..=250, por defecto 10)
        let chunk_frames = if is_stream {
            let cf = extract_opt_i64(obj, "chunk_frames")?.unwrap_or(DEFAULT_CHUNK_FRAMES as i64);
            cf.clamp(2, 250) as usize
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
        assert!(parse(r#"{"text":"Hi.","voice_design":"true"}"#, false).is_err());
        assert!(parse(r#"{"text":"Hi.","rate":1.2}"#, true).is_err());
        assert!(parse(r#"{"text":"Hi.","rate":0.01}"#, false).is_err());
        assert!(parse(r#"{"text":"Hi.","volume":-1.0}"#, false).is_err());
        assert!(parse(r#"{"text":"Hi.","seed":99999999999}"#, false).is_err());
        assert!(parse(r#"{"text":"Hi.","voice_design":"TRUE"}"#, false).is_err());

        let ok = parse(r#"{"input":"Hello!","rate":1.2}"#, false).unwrap();
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
        let err = parse(r#"{"text":"Hi.","voice":"ryan"}"#, false).unwrap_err();
        assert_eq!(err, "la voz 'ryan' no está cargada (cargada: 'default')");
        assert!(parse(r#"{"text":"Hi.","voice":5}"#, false).is_err());
    }

    #[test]
    fn test_removed_fields_are_ignored() {
        let r = parse(r#"{"text":"Hi.","speaker":"ryan","emotion":"sad"}"#, false).unwrap();
        assert_eq!(r.text, "Hi.");
        assert!(parse(r#"{"text":"Hi.","speaker":"nadie","emotion":7}"#, false).is_ok());
    }
}
