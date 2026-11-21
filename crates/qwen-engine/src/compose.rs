// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Composición por tramos: una sola implementación para la entrega completa y el streaming.
//!
//! Cada tramo hablado se sintetiza con ganancia `x·(v·o)` (`v` el volumen de la petición, `o` el
//! del tramo) y se une al anterior con una pausa o, junto a una muletilla, con un fundido
//! cruzado de 45 ms. Para poder fundir, se retienen las últimas 45 ms emitidas.

use qwen_core::config::SAMPLE_RATE;
use qwen_core::SpanSpec;

use crate::audio::AudioSink;
use crate::error::EngineError;
use crate::model::SessionScratch;
use crate::pipeline::Run;
use crate::synth::plain_buffer;

/// Duración del fundido cruzado junto a una muletilla, en muestras.
fn xfade_samples() -> usize {
    (0.045f32 * SAMPLE_RATE as f32) as usize
}

#[inline]
fn pause_sample_count(seconds: f32) -> usize {
    if seconds.is_finite() && seconds > 0.0 {
        (seconds.min(60.0) * SAMPLE_RATE as f32) as usize
    } else {
        0
    }
}

/// Entrega al destino reteniendo la cola que aún puede fundirse con el tramo siguiente.
struct SeamWriter<'a> {
    dest: &'a mut (dyn AudioSink + Send),
    held: Vec<f32>,
    cap: usize,
    total: usize,
}

impl<'a> SeamWriter<'a> {
    fn new(dest: &'a mut (dyn AudioSink + Send)) -> Self {
        Self {
            dest,
            held: Vec::new(),
            cap: xfade_samples(),
            total: 0,
        }
    }

    fn push(&mut self, samples: &[f32]) -> Result<(), EngineError> {
        self.held.extend_from_slice(samples);
        self.total += samples.len();
        if self.held.len() > self.cap {
            let n = self.held.len() - self.cap;
            self.dest.write_samples(&self.held[..n])?;
            self.held.drain(..n);
        }
        Ok(())
    }

    fn push_silence(&mut self, n: usize) -> Result<(), EngineError> {
        self.push(&vec![0.0f32; n])
    }

    /// Funde el principio de `seg` con la cola retenida y añade el resto.
    fn crossfade_in(&mut self, seg: &[f32]) -> Result<(), EngineError> {
        let xf = self.cap.min(self.held.len()).min(seg.len());
        let base = self.held.len() - xf;
        for (k, (h, &s)) in self.held[base..].iter_mut().zip(seg).enumerate() {
            let a = (xf - k) as f32 / (xf as f32);
            *h = *h * a + s * (1.0 - a);
        }
        self.push(&seg[xf..])
    }

    /// Vacía la cola retenida y cierra el destino.
    fn finish(self) -> Result<usize, EngineError> {
        if !self.held.is_empty() {
            self.dest.write_samples(&self.held)?;
        }
        self.dest.finish()?;
        Ok(self.total)
    }
}

/// Sintetiza un tramo hablado con prefill frío hacia memoria, con la velocidad y el volumen del
/// tramo aplicados sobre los de la petición.
fn synth_span(
    job: &Run<'_>,
    scratch: &mut SessionScratch,
    text: &str,
    rate: f32,
    volume: f32,
    idx: usize,
) -> Result<Vec<f32>, EngineError> {
    if !job.req.log.is_silent() {
        eprintln!("Tramo {idx}: \"{text}\"");
    }
    let mut span_req = job.req.clone();
    if rate > 0.0 {
        span_req.rate = (span_req.rate * rate).clamp(0.25, 4.0);
    }
    if volume > 0.0 {
        span_req.volume *= volume;
    }
    let span_job = Run {
        req: &span_req,
        ..*job
    };
    let (samples, _) = plain_buffer(&span_job, scratch, text)?;
    if samples.is_empty() {
        return Err(EngineError::Runtime(format!(
            "Composición: falló la síntesis del tramo {idx}"
        )));
    }
    Ok(samples)
}

/// Compone los tramos (`spans`) y los entrega a `dest` (que se cierra con `finish`). Devuelve el número de
/// tramos hablados y el de muestras entregadas.
pub(crate) fn compose_spans(
    job: &Run<'_>,
    scratch: &mut SessionScratch,
    spans: &[SpanSpec],
    dest: &mut (dyn AudioSink + Send),
) -> Result<(usize, usize), EngineError> {
    let silent = job.req.log.is_silent();
    let mut out = SeamWriter::new(dest);
    let mut spoken = 0usize;
    let mut last_spoken = false;
    let mut prev_filler = false;

    for span in spans {
        if job.abort.is_cancelled() {
            return Err(EngineError::Aborted);
        }
        match span {
            SpanSpec::Pause { seconds } => {
                let pn = pause_sample_count(*seconds);
                if pn > 0 {
                    out.push_silence(pn)?;
                }
                if !silent {
                    eprintln!("  [pausa {:.2}s]", pn as f32 / SAMPLE_RATE as f32);
                }
                last_spoken = false;
                prev_filler = false;
            }
            SpanSpec::Utterance {
                text,
                is_filler,
                rate,
                volume,
            } => {
                let xfade_seam = *is_filler || prev_filler;
                let pn = if last_spoken && !xfade_seam {
                    pause_sample_count(job.req.compose_pause_s)
                } else {
                    0
                };
                if pn > 0 {
                    out.push_silence(pn)?;
                }
                let seg = synth_span(job, scratch, text, *rate, *volume, spoken)?;
                if xfade_seam && !out.held.is_empty() {
                    out.crossfade_in(&seg)?;
                } else {
                    out.push(&seg)?;
                }
                spoken += 1;
                last_spoken = true;
                prev_filler = *is_filler;
            }
        }
    }

    if spoken == 0 {
        return Err(EngineError::InvalidArg(
            "Composición: nada que sintetizar".to_string(),
        ));
    }
    let total = out.finish()?;
    if !silent {
        eprintln!(
            "[compuestos {spoken} tramos, {:.2}s]",
            total as f64 / SAMPLE_RATE as f64
        );
    }
    Ok((spoken, total))
}
