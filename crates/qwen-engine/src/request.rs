// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Petición de síntesis: todo lo que el motor necesita para una síntesis, ya resuelto.

use qwen_core::config::DEFAULT_LANGUAGE_ID;
use qwen_core::SamplingParams;

/// Máximo de frames por síntesis cuando el llamador no fija otro.
pub const DEFAULT_MAX_FRAMES: usize = 8192;
/// Tamaño por defecto del chunk de decodificación, en frames.
pub const DEFAULT_CHUNK_FRAMES: usize = 10;
/// Menor tamaño de chunk de decodificación admitido, en frames.
pub const MIN_CHUNK_FRAMES: usize = 2;
/// Mayor tamaño de chunk de decodificación admitido, en frames.
pub const MAX_CHUNK_FRAMES: usize = 250;
/// Pausa por defecto entre tramos hablados de una composición, en segundos.
pub const DEFAULT_COMPOSE_PAUSE_S: f32 = 0.12;

/// Nivel de log por `stderr`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LogLevel {
    /// Sin log.
    Silent,
    /// Log normal.
    #[default]
    Normal,
    /// Log normal y diagnósticos internos.
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

/// Parámetros de muestreo, audio y log de una síntesis.
#[derive(Debug, Clone)]
pub struct SynthesisRequest {
    /// Idioma del prompt, ya resuelto con `effective_language`.
    pub language_id: i32,
    pub sampling: SamplingParams,
    /// Máximo de frames generados.
    pub max_frames: usize,
    /// Tamaño del chunk de decodificación; se acota a `MIN_CHUNK_FRAMES..=MAX_CHUNK_FRAMES`.
    pub chunk_frames: usize,
    pub volume: f32,
    pub rate: f32,
    /// Pausa entre tramos hablados de una composición, en segundos.
    pub compose_pause_s: f32,
    pub log: LogLevel,
}

impl Default for SynthesisRequest {
    fn default() -> Self {
        Self {
            language_id: DEFAULT_LANGUAGE_ID,
            sampling: SamplingParams::default(),
            max_frames: DEFAULT_MAX_FRAMES,
            chunk_frames: DEFAULT_CHUNK_FRAMES,
            volume: 1.0,
            rate: 1.0,
            compose_pause_s: DEFAULT_COMPOSE_PAUSE_S,
            log: LogLevel::Normal,
        }
    }
}
