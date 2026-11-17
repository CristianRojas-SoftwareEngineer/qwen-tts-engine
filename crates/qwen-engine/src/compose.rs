// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use qwen_core::config::SAMPLE_RATE;
use qwen_core::{MarkupPlanner, SpanSpec, VoiceClass};
use qwen_kernels::ComputeRuntime;

use crate::audio::AudioSink;
use crate::error::EngineError;
use crate::model::{
    Model06B, RequestExecution, SessionScratch, VoiceBindings, VoiceIdentity,
};
use crate::pipeline::synthesize_to_samples;

pub struct PreparedMarkup {
    pub text: String,
    pub did_para_substitute: bool,
    pub use_compose: bool,
    pub seed: u32,
    pub temperature: f32,
}

pub fn voice_class_for(voice: &VoiceBindings) -> VoiceClass {
    match &voice.identity {
        VoiceIdentity::Preset(3065) => VoiceClass::Vivian,
        VoiceIdentity::Preset(_) => VoiceClass::PresetOther,
        VoiceIdentity::XVector(_) | VoiceIdentity::IclGraft { .. } => VoiceClass::Clone,
    }
}

/// Applies inline `[laugh]`/`[sigh]`/`[yawn]`/`[wow]`/`[giggle]`/`[scoff]` substitution
/// (`small_model = true` for 0.6B) and detects whether remaining markup needs multi-span compose.
pub fn prepare_markup_06b(
    text: &str,
    voice: &VoiceBindings,
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
            seed: user_seed.unwrap_or(42),
            temperature: user_temp.unwrap_or(default_temp),
        };
    }

    let vc = voice_class_for(voice);
    let sub = MarkupPlanner::para_substitute(text, vc);
    let seed = user_seed.unwrap_or(if sub.did_substitute { sub.seed } else { 42 });
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

fn synth_one_span_06b(
    runtime: &ComputeRuntime,
    model: &Model06B,
    voice: &VoiceBindings,
    scratch: &mut SessionScratch,
    exec: &RequestExecution,
    span_text: &str,
    mood: Option<&str>,
    rate_override: f32,
    volume_override: f32,
    idx: usize,
) -> Result<Vec<f32>, EngineError> {
    let silent = exec.config.silent;
    if !silent {
        eprintln!(
            "Span {idx}: [{}] \"{span_text}\"",
            mood.unwrap_or("neutral")
        );
    }
    let mut span_cfg = exec.config.clone();
    if rate_override > 0.0 {
        span_cfg.rate = rate_override;
    }
    if volume_override > 0.0 {
        span_cfg.volume = volume_override;
    }
    // Each compose span runs non-streaming cold prefill into MemorySink, then DSP post-process.
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

/// Multi-span synthesis into a single mono 24 kHz `f32` buffer.
/// Applies inter-span pauses, WSOLA rate/volume DSP, and 45 ms filler overlap-add crossfade.
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
                if *seconds > 0.0 {
                    let pn = (*seconds * SAMPLE_RATE as f32) as usize;
                    out.resize(out.len() + pn, 0.0);
                }
                if !silent {
                    eprintln!("  [pause {seconds:.2}s]");
                }
                last_spoken = false;
                prev_filler = false;
            }
            SpanSpec::Utterance {
                text,
                mood,
                is_filler,
                rate,
                volume,
            } => {
                let xfade_seam = *is_filler || prev_filler;
                if last_spoken && default_pause_s > 0.0 && !xfade_seam {
                    let pn = (default_pause_s * SAMPLE_RATE as f32) as usize;
                    out.resize(out.len() + pn, 0.0);
                }

                let seg = synth_one_span_06b(
                    runtime,
                    model,
                    voice,
                    scratch,
                    exec,
                    text,
                    mood.as_deref(),
                    *rate,
                    *volume,
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

/// Streaming multi-span synthesis.
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

    for span in spans {
        if exec.abort.is_cancelled() {
            return Err(EngineError::Aborted);
        }
        match span {
            SpanSpec::Pause { seconds } => {
                if *seconds > 0.0 {
                    let pn = (*seconds * SAMPLE_RATE as f32) as usize;
                    if pn > 0 {
                        let sil = vec![0.0f32; pn];
                        sink.write_samples(&sil)?;
                    }
                }
                if !silent {
                    eprintln!("  [pause {seconds:.2}s]");
                }
                last_spoken = false;
            }
            SpanSpec::Utterance {
                text,
                mood,
                is_filler: _,
                rate,
                volume,
            } => {
                if last_spoken && default_pause_s > 0.0 {
                    let pn = (default_pause_s * SAMPLE_RATE as f32) as usize;
                    if pn > 0 {
                        let sil = vec![0.0f32; pn];
                        sink.write_samples(&sil)?;
                    }
                }
                let seg = synth_one_span_06b(
                    runtime,
                    model,
                    voice,
                    scratch,
                    exec,
                    text,
                    mood.as_deref(),
                    *rate,
                    *volume,
                    spoken,
                )?;
                if !seg.is_empty() {
                    sink.write_samples(&seg)?;
                }
                spoken += 1;
                last_spoken = true;
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
