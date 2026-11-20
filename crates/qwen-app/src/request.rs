// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Preparación única de la petición: marcado, semilla, temperatura e idioma.
//!
//! Es el único sitio que aplica `plan_script`, `resolve_sampling` y la regla de idioma de cada
//! interfaz, tanto para la CLI como para el servidor HTTP.

use qwen_core::config::{lookup_language, FRAME_RATE};
use qwen_core::{
    plan_script, resolve_sampling, MarkupMode, PlannedScript, SamplingParams, Script, SeedChoice,
};
use qwen_engine::{RequestConfig, Voice};

use crate::cli::CliArgs;
use crate::server::validation::{ValidatedTtsRequest, DEFAULT_CHUNK_FRAMES};

/// Prepara la petición de la CLI. El idioma de `-l` ya fue validado antes de cargar el modelo.
///
/// Regla de idioma vigente hasta la regla única: `-l`, si no el de la voz, si no inglés (2050).
pub fn from_cli(cli: &CliArgs, voice: &Voice) -> (RequestConfig, PlannedScript) {
    let cli_language_id = cli.language.as_deref().and_then(lookup_language);
    let language_id = cli_language_id.or(voice.language_id).or(Some(2050));

    let max_tokens = match cli.max_duration {
        Some(dur) if dur > 0.0 => ((dur * FRAME_RATE) as usize).max(1),
        _ => cli.max_tokens,
    };

    let (text, mode) = match &cli.compose_spec {
        Some(spec) => (spec.as_str(), MarkupMode::Compose),
        None if cli.no_compose => (cli.text.as_deref().unwrap_or_default(), MarkupMode::Literal),
        None => (cli.text.as_deref().unwrap_or_default(), MarkupMode::Auto),
    };
    let planned = plan_script(text, mode);

    let base = SamplingParams {
        top_k: cli.top_k as usize,
        top_p: cli.top_p,
        rep_penalty: cli.rep_penalty,
        greedy_warmup: cli.greedy_warmup,
        ..SamplingParams::default()
    };
    let sampling = resolve_sampling(base, cli.seed, cli.temperature, planned.para);

    if !cli.silent {
        if let Some((para_seed, _)) = planned.para {
            let shown_seed = match cli.seed {
                SeedChoice::Fixed(s) => s,
                _ => para_seed,
            };
            eprintln!(
                "Paralinguistics: inline [tag]->onomatopoeia (seed {}, T{:.1}): \"{}\"",
                shown_seed, sampling.temperature, planned.text
            );
        }
        if mode == MarkupMode::Auto && matches!(planned.script, Script::Spans(_)) {
            eprintln!("Inline markup detected in --text -> compose mode");
        }
        eprintln!(
            "seed: {}{}",
            sampling.seed,
            if cli.seed == SeedChoice::Clock {
                " (auto/time-based)"
            } else {
                ""
            }
        );
    }

    let req = RequestConfig {
        language_id,
        temperature: sampling.temperature,
        top_k: sampling.top_k as i32,
        top_p: sampling.top_p,
        rep_penalty: sampling.rep_penalty,
        max_tokens,
        greedy_warmup: sampling.greedy_warmup,
        seed: sampling.seed,
        stream_chunk_frames: cli.stream_chunk.clamp(2, 250),
        volume: cli.volume,
        rate: cli.rate,
        silent: cli.silent,
        debug: cli.debug,
        cp_roughness: cli.roughness,
    };
    (req, planned)
}

/// Prepara la petición HTTP. `stream` es `true` en `/v1/tts/stream`.
///
/// Reglas por entrega: `chunk_frames` solo cuenta en streaming (en `/v1/tts` es el valor por
/// defecto) y `rate` solo en la entrega completa. Idioma vigente hasta la regla única: el de la
/// petición, si no el de la voz, si no ninguno.
pub fn from_http(
    vreq: &ValidatedTtsRequest,
    voice: &Voice,
    stream: bool,
) -> (RequestConfig, PlannedScript) {
    let planned = plan_script(&vreq.text, MarkupMode::Auto);

    let base = SamplingParams {
        top_k: vreq.top_k as usize,
        top_p: vreq.top_p,
        rep_penalty: vreq.rep_penalty,
        ..SamplingParams::default()
    };
    let sampling = resolve_sampling(base, vreq.seed, vreq.temperature, planned.para);

    let req = RequestConfig {
        language_id: vreq.language_id.or(voice.language_id),
        temperature: sampling.temperature,
        top_k: sampling.top_k as i32,
        top_p: sampling.top_p,
        rep_penalty: sampling.rep_penalty,
        max_tokens: 8192,
        greedy_warmup: sampling.greedy_warmup,
        seed: sampling.seed,
        stream_chunk_frames: if stream {
            vreq.chunk_frames
        } else {
            DEFAULT_CHUNK_FRAMES
        },
        volume: vreq.volume,
        rate: if stream { 1.0 } else { vreq.rate },
        silent: true,
        debug: false,
        cp_roughness: 0.0,
    };
    (req, planned)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn voice(language_id: Option<i32>) -> Voice {
        Voice {
            name: "v".to_string(),
            language_id,
            xvector: Vec::new(),
            icl: None,
        }
    }

    fn parse(body: &str, stream: bool) -> ValidatedTtsRequest {
        ValidatedTtsRequest::parse(body, stream, "v").unwrap()
    }

    #[test]
    fn http_language_request_then_voice_then_none() {
        let v = parse(r#"{"text":"Hola."}"#, false);
        assert_eq!(
            from_http(&v, &voice(Some(2054)), false).0.language_id,
            Some(2054)
        );
        assert_eq!(from_http(&v, &voice(None), false).0.language_id, None);
        let e = parse(r#"{"text":"Hola.","language":"English"}"#, false);
        assert_eq!(
            from_http(&e, &voice(Some(2054)), false).0.language_id,
            Some(2050)
        );
    }

    #[test]
    fn http_per_delivery_rules() {
        let body = r#"{"text":"Hola.","chunk_frames":4,"rate":1.2}"#;
        let (full, _) = from_http(&parse(body, false), &voice(None), false);
        assert_eq!(full.stream_chunk_frames, DEFAULT_CHUNK_FRAMES);
        assert!((full.rate - 1.2).abs() < 1e-6);
        let body = r#"{"text":"Hola.","chunk_frames":4}"#;
        let (stream, _) = from_http(&parse(body, true), &voice(None), true);
        assert_eq!(stream.stream_chunk_frames, 4);
        assert_eq!(stream.rate, 1.0);
        assert_eq!((stream.max_tokens, stream.greedy_warmup), (8192, 0));
        assert!(stream.silent && stream.cp_roughness == 0.0);
    }

    #[test]
    fn http_seed_and_temperature_precedence() {
        let (para, _) = from_http(
            &parse(r#"{"text":"Hola [laugh] mundo."}"#, false),
            &voice(None),
            false,
        );
        assert_eq!((para.seed, para.temperature), (2024, 1.1));
        let body = r#"{"text":"Hola [laugh] mundo.","seed":4,"temperature":0.5}"#;
        let (user, _) = from_http(&parse(body, false), &voice(None), false);
        assert_eq!((user.seed, user.temperature), (4, 0.5));
    }
}
