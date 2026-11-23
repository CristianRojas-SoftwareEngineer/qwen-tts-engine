// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Única entrada pública de síntesis: [`Synthesizer::synthesize`].
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
use crate::pipeline::{generate_to_sink, Run};
use crate::request::SynthesisRequest;
use crate::voice::Voice;

/// Cómo se entrega el audio.
pub enum Delivery<'a> {
    /// Todo el audio en memoria.
    Buffer,
    /// Chunks PCM hacia el destino, sin `rate` distinto de 1.
    Stream(&'a mut (dyn AudioSink + Send)),
}

/// Resultado de una síntesis.
pub struct SynthesisOutput {
    /// El audio mono `f32` a 24 kHz; solo con `Delivery::Buffer`.
    pub samples: Option<Vec<f32>>,
    /// Métricas de la generación; en un guion de tramos solo lleva `samples_produced`.
    pub timing: SynthesisTiming,
}

/// Contexto fijo de una sesión de síntesis: el runtime, el modelo y la voz.
///
/// Es la única entrada pública de síntesis, mediante [`Synthesizer::synthesize`].
#[derive(Clone, Copy)]
pub struct Synthesizer<'a> {
    pub runtime: &'a ComputeRuntime,
    pub model: &'a Model06B,
    pub voice: &'a Voice,
}

impl Synthesizer<'_> {
    /// Sintetiza `script` con la voz del contexto y lo entrega según `delivery`.
    ///
    /// - `Script::Plain` con `Buffer`: ganancia y `time_stretch`.
    /// - `Script::Plain` con `Stream`: chunks con la ganancia aplicada justo antes del
    ///   destino.
    /// - `Script::Spans`: una sola composición para las dos entregas (ganancia `x·(v·o)` por
    ///   tramo, pausas y fundido cruzado de 45 ms).
    ///
    /// `rate` distinto de 1 con `Stream` es un error.
    pub fn synthesize(
        &self,
        scratch: &mut SessionScratch,
        req: &SynthesisRequest,
        abort: &AbortHandle,
        script: &Script,
        delivery: Delivery<'_>,
    ) -> Result<SynthesisOutput, EngineError> {
        if matches!(delivery, Delivery::Stream(_)) && (req.rate - 1.0).abs() > 1e-4 {
            return Err(EngineError::InvalidArg(
                "rate != 1.0 no se puede combinar con la entrega en streaming".to_string(),
            ));
        }
        let job = Run {
            runtime: self.runtime,
            model: self.model,
            voice: self.voice,
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
                let (_, total) = compose_spans(&job, scratch, spans, &mut sink)?;
                Ok(SynthesisOutput {
                    samples: Some(sink.samples),
                    timing: SynthesisTiming {
                        samples_produced: total,
                        ..SynthesisTiming::default()
                    },
                })
            }
            (Script::Spans(spans), Delivery::Stream(sink)) => {
                let (_, total) = compose_spans(&job, scratch, spans, sink)?;
                Ok(SynthesisOutput {
                    samples: None,
                    timing: SynthesisTiming {
                        samples_produced: total,
                        ..SynthesisTiming::default()
                    },
                })
            }
        }
    }
}

/// Texto plano hacia un buffer mono `f32` a 24 kHz con las métricas.
///
/// Orden del buffer: ganancia y `time_stretch`.
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
    timing.samples_produced = samples.len();
    Ok((samples, timing))
}
