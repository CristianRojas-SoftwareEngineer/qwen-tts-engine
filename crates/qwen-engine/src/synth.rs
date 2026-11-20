// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Única entrada pública de síntesis: [`synthesize`].
//!
//! Un guion (`Script`) se entrega completo (`Delivery::Buffer`) o progresivamente a un destino
//! (`Delivery::Stream`).

use qwen_core::config::HOP_SAMPLES;
use qwen_core::Script;
use qwen_kernels::ComputeRuntime;

use crate::audio::{time_stretch, AudioSink, MemorySink};
use crate::compose::compose_spans;
use crate::error::EngineError;
use crate::model::{AbortHandle, Model06B, SessionScratch, SynthesisTiming};
use crate::pipeline::{finish_full_delivery, generate_to_sink, Run};
use crate::request::SynthesisRequest;
use crate::voice::Voice;

/// Cómo se entrega el audio.
pub enum Delivery<'a> {
    /// Todo el audio en memoria, con el acabado de la entrega completa.
    Buffer,
    /// Fragmentos PCM hacia el destino, sin acabado y sin `rate` distinto de 1.
    Stream(&'a mut (dyn AudioSink + Send)),
}

/// Resultado de una síntesis.
pub struct SynthesisOutput {
    /// El audio mono `f32` a 24 kHz; solo con `Delivery::Buffer`.
    pub samples: Option<Vec<f32>>,
    /// Métricas de la generación; vacías (por defecto) en un guion de tramos.
    pub timing: SynthesisTiming,
}

/// Sintetiza `script` con `voice` y lo entrega según `delivery`.
///
/// - `Script::Plain` con `Buffer`: ganancia, `time_stretch` y, al final, `tail_trim` y
///   `onset_fade`.
/// - `Script::Plain` con `Stream`: fragmentos con la ganancia aplicada justo antes del destino.
/// - `Script::Spans`: una sola composición para las dos entregas (ganancia `x·(v·o)` por tramo,
///   pausas y fundido cruzado de 45 ms); con `Buffer` además el acabado final.
///
/// `rate` distinto de 1 con `Stream` es un error.
#[allow(clippy::too_many_arguments)]
pub fn synthesize(
    runtime: &ComputeRuntime,
    model: &Model06B,
    voice: &Voice,
    scratch: &mut SessionScratch,
    req: &SynthesisRequest,
    abort: &AbortHandle,
    script: &Script,
    delivery: Delivery<'_>,
) -> Result<SynthesisOutput, EngineError> {
    if matches!(delivery, Delivery::Stream(_)) && (req.rate - 1.0).abs() > 1e-4 {
        return Err(EngineError::InvalidArg(
            "rate != 1.0 cannot be combined with streaming delivery".to_string(),
        ));
    }
    let job = Run {
        runtime,
        model,
        voice,
        req,
        abort,
    };
    match (script, delivery) {
        (Script::Plain(text), Delivery::Buffer) => {
            let (samples, timing) = plain_buffer(&job, scratch, text)?;
            Ok(SynthesisOutput {
                samples: Some(samples),
                timing,
            })
        }
        (Script::Plain(text), Delivery::Stream(sink)) => {
            let timing = generate_to_sink(&job, scratch, text, sink)?;
            Ok(SynthesisOutput {
                samples: None,
                timing,
            })
        }
        (Script::Spans(spans), Delivery::Buffer) => {
            let mut sink = MemorySink::new(0);
            compose_spans(&job, scratch, spans, &mut sink)?;
            let mut samples = sink.samples;
            finish_full_delivery(&mut samples, req);
            Ok(SynthesisOutput {
                samples: Some(samples),
                timing: SynthesisTiming::default(),
            })
        }
        (Script::Spans(spans), Delivery::Stream(sink)) => {
            compose_spans(&job, scratch, spans, sink)?;
            Ok(SynthesisOutput {
                samples: None,
                timing: SynthesisTiming::default(),
            })
        }
    }
}

/// Texto plano hacia un buffer mono `f32` a 24 kHz con las métricas.
///
/// Orden del buffer: ganancia, `time_stretch` y, al final, `tail_trim` y `onset_fade`.
pub(crate) fn plain_buffer(
    job: &Run<'_>,
    scratch: &mut SessionScratch,
    text: &str,
) -> Result<(Vec<f32>, SynthesisTiming), EngineError> {
    let req = job.req;
    let mut sink = MemorySink::new(req.max_frames.min(1024) * HOP_SAMPLES);
    let mut timing = generate_to_sink(job, scratch, text, &mut sink)?;
    // `generate_to_sink` ya aplicó la ganancia, antes de `time_stretch`.
    let mut samples = sink.samples;
    if (req.rate - 1.0).abs() > 1e-4 {
        samples = time_stretch(&samples, req.rate);
    }
    finish_full_delivery(&mut samples, req);
    timing.samples_produced = samples.len();
    Ok((samples, timing))
}
