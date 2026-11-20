// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::fmt;

/// Errores del crate núcleo (`qwen-core`) durante lectura de configuración, `safetensors`, tokenizador o `.qvoice`.
#[derive(Debug)]
pub enum CoreError {
    Io(std::io::Error),
    Json(serde_json::Error),
    InvalidConfig(String),
    Safetensors(String),
    Tokenizer(String),
    QVoice(String),
}

impl fmt::Display for CoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CoreError::Io(e) => write!(f, "Error de E/S: {e}"),
            CoreError::Json(e) => write!(f, "Error de JSON: {e}"),
            CoreError::InvalidConfig(s) => write!(f, "{s}"),
            CoreError::Safetensors(s) => write!(f, "Error de safetensors: {s}"),
            CoreError::Tokenizer(s) => write!(f, "Error del tokenizer: {s}"),
            CoreError::QVoice(s) => write!(f, "{s}"),
        }
    }
}

impl std::error::Error for CoreError {}

impl From<std::io::Error> for CoreError {
    fn from(e: std::io::Error) -> Self {
        CoreError::Io(e)
    }
}

impl From<serde_json::Error> for CoreError {
    fn from(e: serde_json::Error) -> Self {
        CoreError::Json(e)
    }
}
