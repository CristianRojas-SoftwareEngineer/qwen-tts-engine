// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::time::{SystemTime, UNIX_EPOCH};

use qwen_core::config::{lookup_language, lookup_speaker, CODEC_VOCAB_SIZE};
use serde_json::Value;

pub const MAX_TTS_TEXT: usize = 8192;
pub const DEFAULT_CHUNK_FRAMES: usize = 10;

#[derive(Debug, Clone)]
pub struct ValidatedTtsRequest {
    pub text: String,
    pub speaker_id: i32,
    pub language_id: Option<i32>,
    pub temperature: f32,
    pub temp_set: bool,
    pub top_k: i32,
    pub top_p: f32,
    pub rep_penalty: f32,
    pub seed: Option<u32>,
    pub volume: f32,
    pub rate: f32,
    pub chunk_frames: usize,
}

fn time_based_seed() -> u32 {
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    ((d.as_secs() ^ (d.subsec_micros() as u64)) & 0xFFFF_FFFF) as u32
}

fn is_only_whitespace_or_escaped(s: &str) -> bool {
    let bytes = s.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'\\'
            && i + 1 < bytes.len()
            && matches!(bytes[i + 1], b'n' | b't' | b'r')
        {
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

fn extract_opt_str<'a>(obj: &'a serde_json::Map<String, Value>, key: &str) -> Result<Option<&'a str>, String> {
    match obj.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.as_str())),
        Some(_) => Err(format!("invalid '{key}' (expected string)")),
    }
}

fn extract_opt_f64(obj: &serde_json::Map<String, Value>, key: &str) -> Result<Option<f64>, String> {
    match obj.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(n)) => n
            .as_f64()
            .filter(|v| v.is_finite())
            .map(Some)
            .ok_or_else(|| format!("invalid '{key}'")),
        Some(_) => Err(format!("invalid '{key}' (expected number)")),
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
                    Err(format!("invalid '{key}'"))
                }
            } else {
                Err(format!("invalid '{key}'"))
            }
        }
        Some(_) => Err(format!("invalid '{key}' (expected integer)")),
    }
}

impl ValidatedTtsRequest {
    pub fn parse(
        body: &str,
        is_stream: bool,
        has_clone_voice: bool,
        default_speaker_id: i32,
        default_language_id: Option<i32>,
    ) -> Result<Self, String> {
        let val: Value = serde_json::from_str(body)
            .map_err(|_| "invalid JSON body".to_string())?;
        let obj = val
            .as_object()
            .ok_or_else(|| "JSON body must be an object".to_string())?;

        // voice_design is 1.7B-only -> reject with 400
        if let Some(vd) = obj.get("voice_design") {
            let enabled = match vd {
                Value::Bool(b) => *b,
                Value::String(s) => s == "true" || s == "1",
                Value::Number(n) => n.as_i64() == Some(1),
                _ => false,
            };
            if enabled {
                return Err("voice_design requires the 1.7B VoiceDesign model".to_string());
            }
        }

        // text or input (mandatory, non-empty, non-whitespace, <= 8192 bytes)
        let text_raw = match extract_opt_str(obj, "text")? {
            Some(t) => Some(t),
            None => extract_opt_str(obj, "input")?,
        };
        let Some(text) = text_raw else {
            return Err("missing or empty 'text'".to_string());
        };
        if text.is_empty() {
            return Err("missing or empty 'text'".to_string());
        }
        if is_only_whitespace_or_escaped(text) {
            return Err("'text' contains only whitespace".to_string());
        }
        if text.len() > MAX_TTS_TEXT {
            return Err(format!("'text' exceeds {MAX_TTS_TEXT} bytes"));
        }

        // language
        let language_name = extract_opt_str(obj, "language")?;
        let language_id = if let Some(lang) = language_name {
            let id = lookup_language(lang)
                .ok_or_else(|| format!("unknown language '{lang}'"))?;
            Some(id)
        } else if has_clone_voice {
            default_language_id
        } else {
            Some(2050) // English
        };

        // speaker or voice
        let spk_raw = match extract_opt_str(obj, "speaker")? {
            Some(s) => Some(s),
            None => extract_opt_str(obj, "voice")?,
        };
        let speaker_id = if let Some(spk) = spk_raw {
            let preset = lookup_speaker(spk)
                .ok_or_else(|| format!("unknown speaker '{spk}'"))?;
            preset.id
        } else if has_clone_voice {
            default_speaker_id
        } else {
            3061
        };

        // temperature in [0.0, 2.0], server default = 0.5
        let temp_opt = extract_opt_f64(obj, "temperature")?;
        let (temperature, temp_set) = match temp_opt {
            Some(v) => {
                if !(0.0..=2.0).contains(&v) {
                    return Err("'temperature' out of range [0, 2]".to_string());
                }
                (v as f32, true)
            }
            None => (0.5, false),
        };

        // top_k in [0, codec_vocab_size (3072)], default = 50
        let top_k = match extract_opt_i64(obj, "top_k")? {
            Some(v) => {
                if v < 0 || v > CODEC_VOCAB_SIZE as i64 {
                    return Err(format!("'top_k' out of range [0, {CODEC_VOCAB_SIZE}]"));
                }
                v as i32
            }
            None => 50,
        };

        // top_p in [0.0, 1.0], default = 1.0
        let top_p = match extract_opt_f64(obj, "top_p")? {
            Some(v) => {
                if !(0.0..=1.0).contains(&v) {
                    return Err("'top_p' out of range [0, 1]".to_string());
                }
                v as f32
            }
            None => 1.0,
        };

        // rep_penalty in [0.5, 2.0], default = 1.05
        let rep_penalty = match extract_opt_f64(obj, "rep_penalty")? {
            Some(v) => {
                if !(0.5..=2.0).contains(&v) {
                    return Err("'rep_penalty' out of range [0.5, 2]".to_string());
                }
                v as f32
            }
            None => 1.05,
        };

        // seed
        let seed = match extract_opt_i64(obj, "seed")? {
            Some(v) if v >= 0 => Some(v as u32),
            _ => None,
        };

        // emotion (optional, validated as string if present)
        let _emotion = extract_opt_str(obj, "emotion")?;

        // volume / rate
        let volume = extract_opt_f64(obj, "volume")?.unwrap_or(1.0) as f32;
        let rate = extract_opt_f64(obj, "rate")?.unwrap_or(1.0) as f32;
        if is_stream && (rate - 1.0).abs() > 1e-6 {
            return Err("'rate' is not supported on /v1/tts/stream".to_string());
        }

        // chunk_frames (2..=250, default 10)
        let chunk_frames = if is_stream {
            let cf = extract_opt_i64(obj, "chunk_frames")?.unwrap_or(DEFAULT_CHUNK_FRAMES as i64);
            cf.clamp(2, 250) as usize
        } else {
            DEFAULT_CHUNK_FRAMES
        };

        Ok(Self {
            text: text.to_string(),
            speaker_id,
            language_id,
            temperature,
            temp_set,
            top_k,
            top_p,
            rep_penalty,
            seed,
            volume,
            rate,
            chunk_frames,
        })
    }

    pub fn resolve_seed_or_time(&self) -> u32 {
        self.seed.unwrap_or_else(time_based_seed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_http_400_validations() {
        assert!(ValidatedTtsRequest::parse(r#"{"text":"   \n\t "}"#, false, false, 3061, Some(2050)).is_err());
        assert!(ValidatedTtsRequest::parse(r#"{"text":"   \\n\\t "}"#, false, false, 3061, Some(2050)).is_err());
        assert!(ValidatedTtsRequest::parse(r#"{"text":""}"#, false, false, 3061, Some(2050)).is_err());
        assert!(ValidatedTtsRequest::parse(r#"{"text":"Hi.","language":"Klingon"}"#, false, false, 3061, Some(2050)).is_err());
        assert!(ValidatedTtsRequest::parse(r#"{"text":"Hi.","speaker":"nadie"}"#, false, false, 3061, Some(2050)).is_err());
        assert!(ValidatedTtsRequest::parse(r#"{"text":"Hi.","temperature":5}"#, false, false, 3061, Some(2050)).is_err());
        assert!(ValidatedTtsRequest::parse(r#"{"text":"Hi.","top_k":4000}"#, false, false, 3061, Some(2050)).is_err());
        assert!(ValidatedTtsRequest::parse(r#"{"text":"Hi.","top_p":1.5}"#, false, false, 3061, Some(2050)).is_err());
        assert!(ValidatedTtsRequest::parse(r#"{"text":"Hi.","rep_penalty":0.2}"#, false, false, 3061, Some(2050)).is_err());
        assert!(ValidatedTtsRequest::parse(r#"{"text":"Hi.","voice_design":"true"}"#, false, false, 3061, Some(2050)).is_err());
        assert!(ValidatedTtsRequest::parse(r#"{"text":"Hi.","speaker":"ryan","rate":1.2}"#, true, false, 3061, Some(2050)).is_err());

        let ok = ValidatedTtsRequest::parse(r#"{"input":"Hello!","voice":"ryan","rate":1.2}"#, false, false, 3061, Some(2050)).unwrap();
        assert_eq!(ok.text, "Hello!");
        assert_eq!(ok.speaker_id, 3061);
        assert_eq!(ok.temperature, 0.5);
        assert!((ok.rate - 1.2).abs() < 1e-6);
    }
}
