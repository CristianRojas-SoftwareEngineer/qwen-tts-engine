// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Preparación única de la petición: marcado, semilla, temperatura e idioma.
//!
//! Es el único sitio que aplica `plan_script`, `resolve_sampling` y la regla de idioma de cada
//! interfaz, tanto para la CLI como para el servidor HTTP.

use std::ops::RangeInclusive;

use qwen_core::config::{effective_language, lookup_language, FRAME_RATE};
use qwen_core::{
    plan_script, resolve_sampling, MarkupMode, SamplingParams, Script, SeedChoice,
};
use qwen_engine::{
    SynthesisRequest, Voice, DEFAULT_COMPOSE_PAUSE_S, DEFAULT_MAX_FRAMES,
    MAX_CHUNK_FRAMES, MIN_CHUNK_FRAMES,
};

use crate::cli::CliArgs;
use crate::server::validation::ValidatedTtsRequest;

/// Rangos admitidos de los parámetros numéricos, iguales en la CLI y en HTTP.
pub const TEMPERATURE_RANGE: RangeInclusive<f32> = 0.0..=2.0;
pub const TOP_P_RANGE: RangeInclusive<f32> = 0.0..=1.0;
pub const REP_PENALTY_RANGE: RangeInclusive<f32> = 0.5..=2.0;
pub const VOLUME_RANGE: RangeInclusive<f32> = 0.0..=10.0;
pub const RATE_RANGE: RangeInclusive<f32> = 0.25..=4.0;

/// Prepara la petición de la CLI. El idioma de `-l` ya fue validado antes de cargar el modelo.
///
/// Idioma: `-l`, si no el de la voz, si no español (`effective_language`).
pub fn from_cli(cli: &CliArgs, voice: &Voice) -> (SynthesisRequest, Script) {
    let cli_language_id = cli.language.as_deref().and_then(lookup_language);
    let language_id = effective_language(cli_language_id, voice.language_id);

    let max_frames = match cli.max_duration {
        Some(dur) => ((dur * FRAME_RATE) as usize).max(1),
        None => DEFAULT_MAX_FRAMES,
    };

    let text = cli.text.as_deref().unwrap_or_default();
    let mode = if cli.no_compose {
        MarkupMode::Literal
    } else {
        MarkupMode::Auto
    };
    let script = plan_script(text, mode);

    let base = SamplingParams {
        top_k: cli.top_k as usize,
        top_p: cli.top_p,
        rep_penalty: cli.rep_penalty,
        ..SamplingParams::default()
    };
    let sampling = resolve_sampling(base, cli.seed, cli.temperature);

    if !cli.silent {
        if mode == MarkupMode::Auto && matches!(script, Script::Spans(_)) {
            eprintln!("Marcado en línea detectado en --text -> modo compose");
        }
        eprintln!(
            "semilla: {}{}",
            sampling.seed,
            if cli.seed == SeedChoice::Clock {
                " (automática/temporal)"
            } else {
                ""
            }
        );
    }

    let req = SynthesisRequest {
        language_id,
        sampling,
        max_frames,
        chunk_frames: cli.stream_chunk.clamp(MIN_CHUNK_FRAMES, MAX_CHUNK_FRAMES),
        volume: cli.volume,
        rate: cli.rate,
        compose_pause_s: cli.compose_pause,
        silent: cli.silent,
    };
    (req, script)
}

/// Prepara la petición HTTP.
///
/// `chunk_frames` y `rate` ya llevan el valor efectivo de la ruta (la validación rechaza el
/// campo que la ruta no admite). Idioma: `effective_language`.
pub fn from_http(vreq: &ValidatedTtsRequest, voice: &Voice) -> (SynthesisRequest, Script) {
    let script = plan_script(&vreq.text, MarkupMode::Auto);

    let base = SamplingParams {
        top_k: vreq.top_k as usize,
        top_p: vreq.top_p,
        rep_penalty: vreq.rep_penalty,
        ..SamplingParams::default()
    };
    let sampling = resolve_sampling(base, vreq.seed, vreq.temperature);

    let req = SynthesisRequest {
        language_id: effective_language(vreq.language_id, voice.language_id),
        sampling,
        max_frames: DEFAULT_MAX_FRAMES,
        chunk_frames: vreq.chunk_frames,
        volume: vreq.volume,
        rate: vreq.rate,
        compose_pause_s: DEFAULT_COMPOSE_PAUSE_S,
        silent: true,
    };
    (req, script)
}

#[cfg(test)]
mod tests {
    use super::*;
    use qwen_core::config::{DEFAULT_SEED, DEFAULT_TEMPERATURE};
    use qwen_engine::DEFAULT_CHUNK_FRAMES;

    fn voice(language_id: Option<i32>) -> Voice {
        Voice {
            name: "v".to_string(),
            language_id,
            xvector: Vec::new(),
        }
    }

    fn parse(body: &str, stream: bool) -> ValidatedTtsRequest {
        ValidatedTtsRequest::parse(body, stream, "v").unwrap()
    }

    #[test]
    fn http_language_request_then_voice_then_spanish() {
        let v = parse(r#"{"text":"Hola."}"#, false);
        assert_eq!(from_http(&v, &voice(Some(2050))).0.language_id, 2050);
        assert_eq!(from_http(&v, &voice(None)).0.language_id, 2054);
        let e = parse(r#"{"text":"Hola.","language":"English"}"#, false);
        assert_eq!(from_http(&e, &voice(Some(2054))).0.language_id, 2050);
    }

    #[test]
    fn http_per_delivery_rules() {
        let body = r#"{"text":"Hola.","rate":1.2}"#;
        let (full, _) = from_http(&parse(body, false), &voice(None));
        assert_eq!(full.chunk_frames, DEFAULT_CHUNK_FRAMES);
        assert!((full.rate - 1.2).abs() < 1e-6);
        let body = r#"{"text":"Hola.","chunk_frames":4}"#;
        let (stream, _) = from_http(&parse(body, true), &voice(None));
        assert_eq!(stream.chunk_frames, 4);
        assert_eq!(stream.rate, 1.0);
        assert_eq!(stream.max_frames, DEFAULT_MAX_FRAMES);
        assert!(stream.silent);
    }

    #[test]
    fn http_seed_and_temperature_precedence() {
        let (default, _) = from_http(&parse(r#"{"text":"Hola."}"#, false), &voice(None));
        assert_eq!(
            (default.sampling.seed, default.sampling.temperature),
            (DEFAULT_SEED, DEFAULT_TEMPERATURE)
        );
        let body = r#"{"text":"Hola.","seed":9,"temperature":0.5}"#;
        let (user, _) = from_http(&parse(body, false), &voice(None));
        assert_eq!((user.sampling.seed, user.sampling.temperature), (9, 0.5));
    }
}
