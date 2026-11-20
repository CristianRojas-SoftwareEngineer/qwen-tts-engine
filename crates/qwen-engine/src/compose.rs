// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use qwen_core::config::{DEFAULT_SEED, SAMPLE_RATE};
use qwen_core::{MarkupPlanner, SpanSpec};
use qwen_kernels::ComputeRuntime;

use crate::audio::AudioSink;
use crate::error::EngineError;
use crate::model::{Model06B, RequestExecution, SessionScratch, VoiceBindings};
use crate::pipeline::synthesize_to_samples;

pub struct PreparedMarkup {
    pub text: String,
    pub did_para_substitute: bool,
    pub use_compose: bool,
    pub seed: u32,
    pub temperature: f32,
}

/// Aplica sustitución inline de `[laugh]`/`[sigh]`/`[yawn]`/`[wow]`/`[giggle]`/`[scoff]`
/// para `0.6B` y detecta si el marcado restante requiere composición multi-span.
pub fn prepare_markup_06b(
    text: &str,
    no_compose: bool,
    user_seed: Option<u32>,
    user_temp: Option<f32>,
    default_temp: f32,
) -> PreparedMarkup {
    if no_compose {
        return PreparedMarkup {
            text: text.to_string(),
            did_para_substitute: false,
            use_compose: false,
            seed: user_seed.unwrap_or(DEFAULT_SEED),
            temperature: user_temp.unwrap_or(default_temp),
        };
    }

    let sub = MarkupPlanner::para_substitute(text);
    let seed = user_seed.unwrap_or(if sub.did_substitute {
        sub.seed
    } else {
        DEFAULT_SEED
    });
    let temperature = user_temp.unwrap_or(if sub.did_substitute {
        sub.temperature
    } else {
        default_temp
    });
    let use_compose = MarkupPlanner::has_markup(&sub.text);

    PreparedMarkup {
        text: sub.text,
        did_para_substitute: sub.did_substitute,
        use_compose,
        seed,
        temperature,
    }
}

#[allow(clippy::too_many_arguments)]
fn synth_one_span_06b(
    runtime: &ComputeRuntime,
    model: &Model06B,
    voice: &VoiceBindings,
    scratch: &mut SessionScratch,
    exec: &RequestExecution,
    span_text: &str,
    rate_override: f32,
    volume_override: f32,
    apply_base_volume: bool,
    idx: usize,
) -> Result<Vec<f32>, EngineError> {
    let silent = exec.config.silent;
    if !silent {
        eprintln!("Span {idx}: \"{span_text}\"");
    }
    let mut span_cfg = exec.config.clone();
    if rate_override > 0.0 {
        span_cfg.rate = (span_cfg.rate * rate_override).clamp(0.25, 4.0);
    }
    if apply_base_volume {
        if volume_override > 0.0 {
            span_cfg.volume *= volume_override;
        }
    } else {
        // En `render_spans_stream`, el `AudioSink` del llamador aplica `exec.config.volume`.
        span_cfg.volume = if volume_override > 0.0 {
            volume_override
        } else {
            1.0
        };
    }
    // Cada span compuesto ejecuta prefill frío sin streaming hacia MemorySink y luego DSP.
    span_cfg.stream = false;
    let span_exec = RequestExecution {
        config: span_cfg,
        abort: exec.abort.clone(),
    };
    let (samples, _) =
        synthesize_to_samples(runtime, model, voice, scratch, &span_exec, span_text)?;
    if samples.is_empty() {
        return Err(EngineError::Runtime(format!(
            "Compose: synthesis failed for span {idx}"
        )));
    }
    Ok(samples)
}

#[inline]
fn pause_sample_count(seconds: f32) -> usize {
    if seconds.is_finite() && seconds > 0.0 {
        (seconds.min(60.0) * SAMPLE_RATE as f32) as usize
    } else {
        0
    }
}

/// Síntesis multi-span hacia un único buffer `f32` mono a 24 kHz.
/// Aplica pausas entre spans, DSP WSOLA de velocidad/volumen y fundido cruzado de 45 ms en muletillas.
pub fn render_spans_buffer(
    runtime: &ComputeRuntime,
    model: &Model06B,
    voice: &VoiceBindings,
    scratch: &mut SessionScratch,
    exec: &RequestExecution,
    spans: &[SpanSpec],
    default_pause_s: f32,
) -> Result<(Vec<f32>, usize), EngineError> {
    let silent = exec.config.silent;
    let mut out: Vec<f32> = Vec::new();
    let mut spoken = 0usize;
    let mut last_spoken = false;
    let mut prev_filler = false;

    for span in spans {
        if exec.abort.is_cancelled() {
            return Err(EngineError::Aborted);
        }
        match span {
            SpanSpec::Pause { seconds } => {
                let pn = pause_sample_count(*seconds);
                if pn > 0 {
                    out.resize(out.len() + pn, 0.0);
                }
                if !silent {
                    eprintln!("  [pause {:.2}s]", pn as f32 / SAMPLE_RATE as f32);
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
                    pause_sample_count(default_pause_s)
                } else {
                    0
                };
                if pn > 0 {
                    out.resize(out.len() + pn, 0.0);
                }

                let seg = synth_one_span_06b(
                    runtime,
                    model,
                    voice,
                    scratch,
                    exec,
                    text,
                    *rate,
                    *volume,
                    true,
                    spoken,
                )?;

                if xfade_seam && !out.is_empty() && !seg.is_empty() {
                    let mut xf = (0.045f32 * SAMPLE_RATE as f32) as usize;
                    xf = xf.min(out.len()).min(seg.len());
                    let base = out.len() - xf;
                    for k in 0..xf {
                        let a = (xf - k) as f32 / (xf as f32);
                        out[base + k] = out[base + k] * a + seg[k] * (1.0 - a);
                    }
                    out.extend_from_slice(&seg[xf..]);
                } else {
                    out.extend_from_slice(&seg);
                }

                spoken += 1;
                last_spoken = true;
                prev_filler = *is_filler;
            }
        }
    }

    if out.is_empty() {
        return Err(EngineError::InvalidArg(
            "Compose: nothing to synthesize".to_string(),
        ));
    }
    if !silent {
        eprintln!(
            "[composed {spoken} spans, {:.2}s]",
            out.len() as f64 / SAMPLE_RATE as f64
        );
    }
    Ok((out, spoken))
}

/// Síntesis multi-span en streaming entregando las muestras PCM progresivamente a `sink`.
#[allow(clippy::too_many_arguments)]
pub fn render_spans_stream(
    runtime: &ComputeRuntime,
    model: &Model06B,
    voice: &VoiceBindings,
    scratch: &mut SessionScratch,
    exec: &RequestExecution,
    spans: &[SpanSpec],
    default_pause_s: f32,
    sink: &mut (dyn AudioSink + Send),
) -> Result<usize, EngineError> {
    let silent = exec.config.silent;
    let mut spoken = 0usize;
    let mut last_spoken = false;
    let mut prev_filler = false;

    for span in spans {
        if exec.abort.is_cancelled() {
            return Err(EngineError::Aborted);
        }
        match span {
            SpanSpec::Pause { seconds } => {
                let pn = pause_sample_count(*seconds);
                if pn > 0 {
                    let sil = vec![0.0f32; pn];
                    sink.write_samples(&sil)?;
                }
                if !silent {
                    eprintln!("  [pause {:.2}s]", pn as f32 / SAMPLE_RATE as f32);
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
                    pause_sample_count(default_pause_s)
                } else {
                    0
                };
                if pn > 0 {
                    let sil = vec![0.0f32; pn];
                    sink.write_samples(&sil)?;
                }
                let seg = synth_one_span_06b(
                    runtime,
                    model,
                    voice,
                    scratch,
                    exec,
                    text,
                    *rate,
                    *volume,
                    false,
                    spoken,
                )?;
                if !seg.is_empty() {
                    sink.write_samples(&seg)?;
                }
                spoken += 1;
                last_spoken = true;
                prev_filler = *is_filler;
            }
        }
    }

    if spoken == 0 {
        return Err(EngineError::InvalidArg(
            "Compose: nothing to synthesize".to_string(),
        ));
    }
    sink.finish()?;
    Ok(spoken)
}
