// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Petición de síntesis: todo lo que el motor necesita para una síntesis, ya resuelto.

use qwen_core::SamplingParams;

/// Tope de fotogramas por síntesis cuando el llamador no fija otro.
pub const DEFAULT_MAX_FRAMES: usize = 8192;
/// Tamaño por defecto del fragmento de decodificación, en fotogramas.
pub const DEFAULT_CHUNK_FRAMES: usize = 10;
/// Pausa por defecto entre tramos hablados de una composición, en segundos.
pub const DEFAULT_COMPOSE_PAUSE_S: f32 = 0.12;

/// Nivel de registro por `stderr`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LogLevel {
    /// Sin registro.
    Silent,
    /// Registro normal.
    #[default]
    Normal,
    /// Registro normal y diagnósticos internos.
    Debug,
}

impl LogLevel {
    /// `true` si no debe escribirse nada por `stderr`.
    #[inline]
    pub fn is_silent(self) -> bool {
        self == LogLevel::Silent
    }

    /// `true` si se piden los diagnósticos internos.
    #[inline]
    pub fn is_debug(self) -> bool {
        self == LogLevel::Debug
    }
}

/// Parámetros de muestreo, audio y registro de una síntesis.
#[derive(Debug, Clone)]
pub struct SynthesisRequest {
    /// Idioma del prompt; `None` no envía token de idioma.
    pub language_id: Option<i32>,
    pub sampling: SamplingParams,
    /// Tope de fotogramas generados.
    pub max_frames: usize,
    /// Tamaño del fragmento de decodificación (`0` = `DEFAULT_CHUNK_FRAMES`; se acota a `2..=250`).
    pub chunk_frames: usize,
    pub cp_roughness: f32,
    pub volume: f32,
    pub rate: f32,
    /// Pausa entre tramos hablados de una composición, en segundos.
    pub compose_pause_s: f32,
    /// Recorte de la cola degenerada; solo en la entrega completa.
    pub tail_trim: bool,
    /// Fundido del ataque en milisegundos (`0` = sin fundido); solo en la entrega completa.
    pub onset_fade_ms: f32,
    pub log: LogLevel,
}

impl Default for SynthesisRequest {
    fn default() -> Self {
        Self {
            language_id: None,
            sampling: SamplingParams::default(),
            max_frames: DEFAULT_MAX_FRAMES,
            chunk_frames: DEFAULT_CHUNK_FRAMES,
            cp_roughness: 0.0,
            volume: 1.0,
            rate: 1.0,
            compose_pause_s: DEFAULT_COMPOSE_PAUSE_S,
            tail_trim: false,
            onset_fade_ms: 0.0,
            log: LogLevel::Normal,
        }
    }
}
