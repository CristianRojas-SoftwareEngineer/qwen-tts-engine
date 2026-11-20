// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Preparación única de la petición: marcado, semilla, temperatura e idioma.
//!
//! Es el único sitio que aplica `plan_script`, `resolve_sampling` y la regla de idioma de cada
//! interfaz, tanto para la CLI como para el servidor HTTP.

use qwen_core::config::{effective_language, lookup_language, FRAME_RATE};
use qwen_core::{
    plan_script, resolve_sampling, MarkupMode, PlannedScript, SamplingParams, Script, SeedChoice,
};
use qwen_engine::{
    LogLevel, SynthesisRequest, Voice, DEFAULT_CHUNK_FRAMES, DEFAULT_COMPOSE_PAUSE_S,
    DEFAULT_MAX_FRAMES,
};

use crate::cli::CliArgs;
use crate::server::validation::ValidatedTtsRequest;

/// Nivel de log de la CLI: `--silent` manda sobre `--debug`.
pub fn log_level(cli: &CliArgs) -> LogLevel {
    if cli.silent {
        LogLevel::Silent
    } else if cli.debug {
        LogLevel::Debug
    } else {
        LogLevel::Normal
    }
}

/// Prepara la petición de la CLI. El idioma de `-l` ya fue validado antes de cargar el modelo.
///
/// Idioma: `-l`, si no el de la voz, si no español (`effective_language`).
pub fn from_cli(cli: &CliArgs, voice: &Voice) -> (SynthesisRequest, PlannedScript) {
    let cli_language_id = cli.language.as_deref().and_then(lookup_language);
    let language_id = effective_language(cli_language_id, voice.language_id);

    let max_frames = match cli.max_duration {
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
    let sampling = resolve_sampling(base, cli.seed, cli.temperature, planned.paralinguistic);

    if !cli.silent {
        if let Some((paralinguistic_seed, _)) = planned.paralinguistic {
            let shown_seed = match cli.seed {
                SeedChoice::Fixed(s) => s,
                _ => paralinguistic_seed,
            };
            eprintln!(
                "Paralingüística: [tag] en línea -> onomatopeya (semilla {}, T{:.1}): \"{}\"",
                shown_seed, sampling.temperature, planned.text
            );
        }
        if mode == MarkupMode::Auto && matches!(planned.script, Script::Spans(_)) {
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
        chunk_frames: cli.stream_chunk.clamp(2, 250),
        cp_roughness: cli.roughness,
        volume: cli.volume,
        rate: cli.rate,
        compose_pause_s: cli.compose_pause,
        tail_trim: cli.tail_trim,
        onset_fade_ms: cli.onset_fade_ms as f32,
        log: log_level(cli),
    };
    (req, planned)
}

/// Prepara la petición HTTP. `stream` es `true` en `/v1/tts/stream`.
///
/// Reglas por entrega: `chunk_frames` solo cuenta en streaming (en `/v1/tts` es el valor por
/// defecto) y `rate` solo en la entrega completa. Idioma: `effective_language`.
pub fn from_http(
    vreq: &ValidatedTtsRequest,
    voice: &Voice,
    stream: bool,
) -> (SynthesisRequest, PlannedScript) {
    let planned = plan_script(&vreq.text, MarkupMode::Auto);

    let base = SamplingParams {
        top_k: vreq.top_k as usize,
        top_p: vreq.top_p,
        rep_penalty: vreq.rep_penalty,
        ..SamplingParams::default()
    };
    let sampling = resolve_sampling(base, vreq.seed, vreq.temperature, planned.paralinguistic);

    let req = SynthesisRequest {
        language_id: effective_language(vreq.language_id, voice.language_id),
        sampling,
        max_frames: DEFAULT_MAX_FRAMES,
        chunk_frames: if stream {
            vreq.chunk_frames
        } else {
            DEFAULT_CHUNK_FRAMES
        },
        cp_roughness: 0.0,
        volume: vreq.volume,
        rate: if stream { 1.0 } else { vreq.rate },
        compose_pause_s: DEFAULT_COMPOSE_PAUSE_S,
        tail_trim: false,
        onset_fade_ms: 0.0,
        log: LogLevel::Silent,
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
    fn http_language_request_then_voice_then_spanish() {
        let v = parse(r#"{"text":"Hola."}"#, false);
        assert_eq!(from_http(&v, &voice(Some(2050)), false).0.language_id, 2050);
        assert_eq!(from_http(&v, &voice(None), false).0.language_id, 2054);
        let e = parse(r#"{"text":"Hola.","language":"English"}"#, false);
        assert_eq!(from_http(&e, &voice(Some(2054)), false).0.language_id, 2050);
    }

    #[test]
    fn http_per_delivery_rules() {
        let body = r#"{"text":"Hola.","chunk_frames":4,"rate":1.2}"#;
        let (full, _) = from_http(&parse(body, false), &voice(None), false);
        assert_eq!(full.chunk_frames, DEFAULT_CHUNK_FRAMES);
        assert!((full.rate - 1.2).abs() < 1e-6);
        let body = r#"{"text":"Hola.","chunk_frames":4}"#;
        let (stream, _) = from_http(&parse(body, true), &voice(None), true);
        assert_eq!(stream.chunk_frames, 4);
        assert_eq!(stream.rate, 1.0);
        assert_eq!(
            (stream.max_frames, stream.sampling.greedy_warmup),
            (DEFAULT_MAX_FRAMES, 0)
        );
        assert!(stream.log == LogLevel::Silent && stream.cp_roughness == 0.0);
    }

    #[test]
    fn http_seed_and_temperature_precedence() {
        let (tagged, _) = from_http(
            &parse(r#"{"text":"Hola [laugh] mundo."}"#, false),
            &voice(None),
            false,
        );
        assert_eq!(
            (tagged.sampling.seed, tagged.sampling.temperature),
            (2024, 1.1)
        );
        let body = r#"{"text":"Hola [laugh] mundo.","seed":4,"temperature":0.5}"#;
        let (user, _) = from_http(&parse(body, false), &voice(None), false);
        assert_eq!((user.sampling.seed, user.sampling.temperature), (4, 0.5));
    }
}
