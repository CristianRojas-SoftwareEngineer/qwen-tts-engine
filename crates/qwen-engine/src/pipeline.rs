// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::Instant;

use qwen_core::config::{
    CODEBOOK_SIZE, CODEC_EOS, CODEC_VOCAB_SIZE, FRAME_RATE, HIDDEN_SIZE,
    NUM_CODEBOOKS, SAMPLE_RATE,
};
use qwen_core::LcgRng;
use qwen_kernels::{ftz_on, ComputeRuntime, KernelDispatch};

use crate::audio::{apply_gain, AudioSink};
use crate::code_predictor::cp_predict;
use crate::error::EngineError;
use crate::model::{
    lookup_codec_embed, AbortHandle, Model06B, SessionScratch, SynthesisTiming, ROPE_MAX_POS,
};
use crate::prompt::{build_prompt, prefill};
use crate::request::SynthesisRequest;
use crate::speech_decoder::speech_decoder_decode_streaming;
use crate::talker::talker_step;
use crate::voice::Voice;

const STREAM_RAMP_START_FRAMES: usize = 4;
const STREAM_RAMP_STEP_FRAMES: usize = 2;
const STREAM_RAMP_HOLD_CHUNKS: usize = 8;

struct DecoderQueueState {
    codes: Vec<i32>,
    write_pos: usize,
    done: bool,
}

struct DecoderQueue {
    state: Mutex<DecoderQueueState>,
    cond: Condvar,
    aborted: AtomicBool,
}

impl DecoderQueue {
    fn new(max_frames: usize) -> Self {
        Self {
            state: Mutex::new(DecoderQueueState {
                codes: Vec::with_capacity(max_frames * NUM_CODEBOOKS),
                write_pos: 0,
                done: false,
            }),
            cond: Condvar::new(),
            aborted: AtomicBool::new(false),
        }
    }

    fn push_frame(&self, codes: &[i32; NUM_CODEBOOKS]) {
        let mut g = self.state.lock().unwrap_or_else(|e| e.into_inner());
        g.codes.extend_from_slice(codes);
        g.write_pos += 1;
        self.cond.notify_one();
    }

    fn finish(&self) {
        let mut g = self.state.lock().unwrap_or_else(|e| e.into_inner());
        g.done = true;
        self.cond.notify_one();
    }
}

/// Calcula los logits del `codec_head` que pueden sobrevivir a `apply_talker_logit_rules`: los
/// códigos de audio (`0..CODEBOOK_SIZE`) y `CODEC_EOS`.
///
/// Las demás filas de `CODEBOOK_SIZE..CODEC_VOCAB_SIZE` (2 MiB de pesos BF16 por frame) no se
/// leen: las reglas las fijan a `-1e30` sin mirar su valor, así que sus posiciones de `logits`
/// quedan sin significado hasta entonces. `CODEC_EOS` se calcula junto a su pareja de fila par para
/// conservar el emparejamiento de filas del kernel BF16 y dar los mismos bits que el cálculo completo.
pub(crate) fn talker_codec_logits(
    runtime: &ComputeRuntime,
    model: &Model06B,
    hidden: &[f32],
    logits: &mut [f32],
) {
    let head = &model.talker.codec_head_bf16;
    let d = runtime.dispatch;
    d.matvec_bf16(
        &runtime.pool,
        &mut logits[..CODEBOOK_SIZE],
        &head[..CODEBOOK_SIZE * HIDDEN_SIZE],
        hidden,
        CODEBOOK_SIZE,
        HIDDEN_SIZE,
    );
    let eos0 = (CODEC_EOS as usize) & !1;
    d.matvec_bf16(
        &runtime.pool,
        &mut logits[eos0..eos0 + 2],
        &head[eos0 * HIDDEN_SIZE..(eos0 + 2) * HIDDEN_SIZE],
        hidden,
        2,
        HIDDEN_SIZE,
    );
}

/// Aplica clamp de logits, supresión de tokens códec especiales y refuerzo progresivo de EOS.
pub(crate) fn apply_talker_logit_rules(
    logits: &mut [f32],
    frame: usize,
    text_content_len: usize,
) {
    for v in logits.iter_mut() {
        *v = v.clamp(-100.0, 100.0);
    }
    for (t, slot) in logits.iter_mut().enumerate().take(CODEC_VOCAB_SIZE).skip(2048) {
        if t as i32 != CODEC_EOS {
            *slot = -1e30;
        }
    }
    if frame < 2 {
        logits[CODEC_EOS as usize] = -1e30;
    }

    let expected_frames = text_content_len * 3;
    let is_short = text_content_len > 0 && text_content_len <= 11;
    let boost_start = if is_short {
        expected_frames * 3 / 2
    } else {
        expected_frames * 2
    };
    if expected_frames > 0 && frame > boost_start {
        let step = if is_short { 1.0 } else { 0.5 };
        let cap = if is_short { 15.0 } else { 10.0 };
        let boost = (step * ((frame - boost_start) as f32)).min(cap);
        logits[CODEC_EOS as usize] += boost;
    }
}

/// Construye el embedding de entrada al siguiente paso del Talker acumulando los 16 codebooks y `tts_pad`.
pub(crate) fn build_step_embed(
    dispatch: &KernelDispatch,
    model: &Model06B,
    code0: i32,
    cp_codes: &[i32],
    out: &mut [f32],
) {
    lookup_codec_embed(dispatch, model, code0, out);
    for (g, &code_g) in cp_codes.iter().enumerate().take(15) {
        if code_g >= 0 && (code_g as usize) < CODEBOOK_SIZE {
            let off = (code_g as usize) * HIDDEN_SIZE;
            let emb = &model.cp.codec_emb_bf16[g][off..off + HIDDEN_SIZE];
            dispatch.bf16_accum_f32(out, emb);
        }
    }
    let tts_pad = model.cached_tts_pad_embed.as_slice();
    for (dst, &pad) in out[..HIDDEN_SIZE].iter_mut().zip(tts_pad[..HIDDEN_SIZE].iter()) {
        *dst += pad;
    }
}

/// Calcula el tamaño objetivo de chunk en frames según la rampa progresiva (la misma con y sin
/// streaming).
#[inline]
pub(crate) fn next_stream_chunk_target(
    first_chunk_emitted: bool,
    ramp_chunks: usize,
    chunk_frames: usize,
) -> (usize, bool) {
    let target = if !first_chunk_emitted && chunk_frames > 2 {
        2
    } else {
        let ramp = STREAM_RAMP_START_FRAMES
            + STREAM_RAMP_STEP_FRAMES * (ramp_chunks / STREAM_RAMP_HOLD_CHUNKS);
        ramp.min(chunk_frames)
    };
    let ramped = first_chunk_emitted || chunk_frames <= 2;
    (target, ramped)
}

/// Datos fijos de una ejecución de texto plano: lo que `generate_to_sink` necesita además del
/// workspace, el texto y el destino.
#[derive(Clone, Copy)]
pub(crate) struct Run<'a> {
    pub runtime: &'a ComputeRuntime,
    pub model: &'a Model06B,
    pub voice: &'a Voice,
    pub req: &'a SynthesisRequest,
    pub abort: &'a AbortHandle,
}

/// Ejecuta la síntesis de un texto plano y entrega progresivamente los chunks de audio PCM a
/// `sink`, con la ganancia ya aplicada.
pub(crate) fn generate_to_sink(
    job: &Run<'_>,
    scratch: &mut SessionScratch,
    text: &str,
    sink: &mut (dyn AudioSink + Send),
) -> Result<SynthesisTiming, EngineError> {
    let Run {
        runtime,
        model,
        voice,
        req,
        abort,
    } = *job;
    let t_start = Instant::now();
    scratch.rng = LcgRng::new(req.sampling.seed);

    let prompt = build_prompt(runtime, model, voice, req.language_id, text, scratch)?;
    let prefill_len = prompt.len;
    let text_content_len = prompt.text_len;
    let silent = req.log.is_silent();
    let debug = req.log.is_debug();
    let sp = &req.sampling;

    if !silent {
        eprintln!(
            "Texto: \"{}\" (prompt: {} posiciones, text_content: {})",
            text, prefill_len, text_content_len
        );
        eprintln!(
            "Prefill: {} posiciones (text_content={})",
            prefill_len, text_content_len
        );
    }

    let nt = runtime.pool.n_threads();
    runtime.blas.set_threads(nt);

    let prefill_ms = prefill(runtime, model, scratch, &prompt, req.log)?;
    let d = runtime.dispatch;

    let mut max_frames = req.max_frames;
    let rope_rem = ROPE_MAX_POS.saturating_sub(prefill_len);
    if max_frames > rope_rem {
        max_frames = rope_rem;
    }

    scratch.codec_codes.clear();
    scratch.codec_codes.reserve(max_frames * NUM_CODEBOOKS);
    scratch.prev_tokens.clear();
    scratch.prev_tokens.reserve(max_frames);

    let chunk_frames = req.chunk_frames.clamp(2, 250);

    let gen_blas = if nt > 1 { nt - 1 } else { 1 };
    runtime.blas.set_threads(gen_blas);

    scratch.sd_stream.reset();
    scratch.sd_stream.cs_tile = true;

    let queue = DecoderQueue::new(max_frames);
    let mut decoder_ms = 0.0f64;
    let mut chunks_decoded = 0usize;
    let mut max_backlog = 0usize;
    let mut ttfa_ms: Option<f64> = None;
    let mut samples_produced = 0usize;
    let mut decoder_err: Option<EngineError> = None;

    let mut talker_ms = 0.0f64;
    let mut cp_ms = 0.0f64;
    let mut drain_ms = 0.0f64;
    let mut was_aborted = false;

    let mut cons_sd_stream = std::mem::take(&mut scratch.sd_stream);
    let mut cons_sd_ws = std::mem::take(&mut scratch.sd_ws);

    std::thread::scope(|s| {
        let dec_handle = s.spawn(|| {
            ftz_on();
            let mut read_pos = 0usize;
            let mut ramp_chunks = 0usize;
            let mut chunk_codes_local = Vec::new();
            let mut first_chunk_emitted = false;

            loop {
                // Mismo calendario de chunks con y sin streaming: las fronteras de
                // chunk fijan el redondeo del decoder, así el PCM es idéntico.
                let (target, ramped) =
                    next_stream_chunk_target(first_chunk_emitted, ramp_chunks, chunk_frames);

                let (avail, is_done) = {
                    let mut g = queue.state.lock().unwrap_or_else(|e| e.into_inner());
                    while (g.write_pos - read_pos) < target && !g.done {
                        g = queue.cond.wait(g).unwrap_or_else(|e| e.into_inner());
                    }
                    let mut avail = g.write_pos - read_pos;
                    if avail > max_backlog {
                        max_backlog = avail;
                    }
                    if avail > target {
                        avail = target;
                    }
                    if avail > 0 {
                        let start = read_pos * NUM_CODEBOOKS;
                        let end = (read_pos + avail) * NUM_CODEBOOKS;
                        chunk_codes_local.clear();
                        chunk_codes_local.extend_from_slice(&g.codes[start..end]);
                    }
                    (avail, g.done)
                };

                if avail == 0 && is_done {
                    break;
                }
                if avail == 0 {
                    continue;
                }
                if queue.aborted.load(Ordering::Acquire) {
                    read_pos += avail;
                    continue;
                }

                chunks_decoded += 1;
                if ramped {
                    ramp_chunks += 1;
                }
                if !first_chunk_emitted {
                    first_chunk_emitted = true;
                }

                let t0 = Instant::now();
                match speech_decoder_decode_streaming(
                    runtime,
                    &model.decoder,
                    &mut cons_sd_stream,
                    &mut cons_sd_ws,
                    &chunk_codes_local,
                    avail,
                ) {
                    Ok(audio) => {
                        let emit = audio.as_slice();
                        if !emit.is_empty() {
                            if ttfa_ms.is_none() {
                                ttfa_ms = Some(t_start.elapsed().as_secs_f64() * 1000.0);
                            }
                            // La ganancia va justo antes del destino (elemento a elemento, mismos
                            // bits que aplicarla dentro de las rampas de borde).
                            let emit = if (req.volume - 1.0).abs() < f32::EPSILON {
                                std::borrow::Cow::Borrowed(emit)
                            } else {
                                let mut g = emit.to_vec();
                                apply_gain(&mut g, req.volume);
                                std::borrow::Cow::Owned(g)
                            };
                            if let Err(e) = sink.write_samples(&emit) {
                                queue.aborted.store(true, Ordering::Release);
                                decoder_err = Some(e);
                            } else {
                                samples_produced += emit.len();
                            }
                        }
                    }
                    Err(e) => {
                        queue.aborted.store(true, Ordering::Release);
                        decoder_err = Some(e);
                    }
                }
                decoder_ms += t0.elapsed().as_secs_f64() * 1000.0;
                read_pos += avail;
            }
        });

        for frame in 0..max_frames {
            if abort.is_cancelled() {
                queue.aborted.store(true, Ordering::Release);
                was_aborted = true;
                break;
            }
            if queue.aborted.load(Ordering::Acquire) {
                break;
            }

            {
                let tb = &mut scratch.talker_bufs;
                talker_codec_logits(runtime, model, &tb.last_hidden, &mut tb.logits);
                apply_talker_logit_rules(&mut tb.logits, frame, text_content_len);
            }

            let code0 = scratch.sampler.sample(
                &mut scratch.talker_bufs.logits,
                sp.temperature,
                sp.top_k as i32,
                sp.top_p,
                sp.rep_penalty,
                &scratch.prev_tokens,
                &mut scratch.rng,
            ) as i32;

            if code0 == CODEC_EOS {
                if !silent {
                    eprintln!("  EOS en el frame {}", frame);
                }
                break;
            }

            scratch.prev_tokens.push(code0);

            let mut codes = [0i32; NUM_CODEBOOKS];
            codes[0] = code0;
            let mut cp_out = [0i32; 15];
            let t_cp = Instant::now();
            // Copia `last_hidden` a un buffer local en pila para evitar aliasing sobre `&mut scratch`
            let mut lh_copy = [0.0f32; HIDDEN_SIZE];
            lh_copy.copy_from_slice(&scratch.talker_bufs.last_hidden);
            cp_predict(
                runtime,
                model,
                scratch,
                &lh_copy,
                code0,
                &mut cp_out,
            );
            cp_ms += t_cp.elapsed().as_secs_f64() * 1000.0;
            codes[1..16].copy_from_slice(&cp_out);

            scratch.codec_codes.extend_from_slice(&codes);
            queue.push_frame(&codes);

            if queue.aborted.load(Ordering::Acquire) {
                break;
            }

            build_step_embed(
                d,
                model,
                code0,
                &codes[1..16],
                &mut scratch.talker_bufs.step_embed,
            );

            let mut step_embed_copy = [0.0f32; HIDDEN_SIZE];
            step_embed_copy.copy_from_slice(&scratch.talker_bufs.step_embed);
            if debug && frame < 2 {
                eprintln!(
                    "  [frame {}] step_embed[:5]=[{:.6},{:.6},{:.6},{:.6},{:.6}]",
                    frame,
                    step_embed_copy[0],
                    step_embed_copy[1],
                    step_embed_copy[2],
                    step_embed_copy[3],
                    step_embed_copy[4]
                );
            }
            let t_tk = Instant::now();
            talker_step(
                runtime,
                model,
                scratch,
                &step_embed_copy,
                &mut lh_copy,
            );
            scratch.talker_bufs.last_hidden.copy_from_slice(&lh_copy);
            if debug && frame < 2 {
                eprintln!(
                    "  [frame {}] POST last_hidden[:5]=[{:.6},{:.6},{:.6},{:.6},{:.6}]",
                    frame,
                    lh_copy[0],
                    lh_copy[1],
                    lh_copy[2],
                    lh_copy[3],
                    lh_copy[4]
                );
            }
            talker_ms += t_tk.elapsed().as_secs_f64() * 1000.0;
        }

        let t_drain = Instant::now();
        queue.finish();
        let _ = dec_handle.join();
        drain_ms = t_drain.elapsed().as_secs_f64() * 1000.0;
    });

    scratch.sd_stream = cons_sd_stream;
    scratch.sd_ws = cons_sd_ws;
    runtime.blas.set_threads(nt);

    if was_aborted {
        return Err(EngineError::Aborted);
    }
    if let Some(e) = decoder_err {
        return Err(e);
    }

    sink.finish()?;

    let frames = scratch.codec_codes.len() / NUM_CODEBOOKS;
    let total_ms = t_start.elapsed().as_secs_f64() * 1000.0;
    if !silent {
        log_synthesis_summary(&SynthesisTiming {
            prefill_ms,
            talker_ms,
            cp_ms,
            decoder_ms,
            drain_ms,
            frames,
            chunks: chunks_decoded,
            max_backlog_frames: max_backlog,
            ttfa_ms,
            total_ms,
            samples_produced,
        }, chunk_frames);
    }

    Ok(SynthesisTiming {
        prefill_ms,
        talker_ms,
        cp_ms,
        decoder_ms,
        drain_ms,
        frames,
        chunks: chunks_decoded,
        max_backlog_frames: max_backlog,
        ttfa_ms,
        total_ms,
        samples_produced,
    })
}

fn log_synthesis_summary(t: &SynthesisTiming, chunk_frames: usize) {
    eprintln!(
        "\n  Generados {} frames ({:.1}s de audio)",
        t.frames,
        (t.frames as f64) / (FRAME_RATE as f64)
    );
    if t.frames > 0 {
        eprintln!(
            "  Paso del Talker: {:.0} ms ({:.1} ms/f), Code Predictor: {:.0} ms ({:.1} ms/f)",
            t.talker_ms,
            t.talker_ms / (t.frames as f64),
            t.cp_ms,
            t.cp_ms / (t.frames as f64)
        );
    }
    eprintln!(
        "  Speech decoder: {:.0} ms en total ({:.0} ms de vaciado tras generar)",
        t.decoder_ms, t.drain_ms
    );
    let audio_dur = (t.samples_produced as f64) / (SAMPLE_RATE as f64);
    let proc_s = t.total_ms / 1000.0;
    if audio_dur > 0.0 {
        eprintln!(
            "Audio: {:.1}s generados en {:.1}s (RTF {:.2})",
            audio_dur,
            proc_s,
            proc_s / audio_dur
        );
    }
    if let Some(ttfa) = t.ttfa_ms {
        eprintln!(
            "  TTFA: {:.0} ms (primer chunk de audio, chunk de {} frames)",
            ttfa, chunk_frames
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Las filas que `talker_codec_logits` no calcula no influyen en los logits tras las reglas,
    /// con cualquier valor previo, en cualquier frame y con y sin refuerzo de EOS.
    #[test]
    fn logit_rules_ignore_rows_skipped_by_codec_head() {
        let eos0 = (CODEC_EOS as usize) & !1;
        for frame in [0usize, 1, 2, 50, 500] {
            for tcl in [0usize, 5, 33] {
                let base: Vec<f32> = (0..CODEC_VOCAB_SIZE).map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.3).collect();
                let mut a = base.clone();
                let mut b = base.clone();
                for (i, v) in b.iter_mut().enumerate().skip(CODEBOOK_SIZE) {
                    if i != eos0 && i != eos0 + 1 {
                        *v = f32::NAN;
                    }
                }
                apply_talker_logit_rules(&mut a, frame, tcl);
                apply_talker_logit_rules(&mut b, frame, tcl);
                for i in 0..CODEC_VOCAB_SIZE {
                    if i != eos0 + 1 {
                        assert_eq!(a[i].to_bits(), b[i].to_bits(), "frame={frame} tcl={tcl} fila={i}");
                    }
                }
                assert_eq!(b[eos0 + 1], -1e30);
            }
        }
    }

    /// `talker_codec_logits` da los mismos bits que el `codec_head` completo en las filas que calcula.
    #[test]
    fn codec_logits_match_full_head_bitwise() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/qwen3-tts-0.6b-base");
        if !dir.exists() {
            eprintln!("Se omite la prueba: falta el modelo en {}", dir.display());
            return;
        }
        let runtime = ComputeRuntime::new(Some(4));
        let model = Model06B::load(&dir, &runtime, true)
            .expect("carga del modelo");
        let hidden: Vec<f32> = (0..HIDDEN_SIZE).map(|i| ((i * 7919 % 211) as f32 - 105.0) * 0.01).collect();
        let mut full = vec![0.0f32; CODEC_VOCAB_SIZE];
        runtime.dispatch.matvec_bf16(
            &runtime.pool,
            &mut full,
            &model.talker.codec_head_bf16,
            &hidden,
            CODEC_VOCAB_SIZE,
            HIDDEN_SIZE,
        );
        let mut part = vec![f32::NAN; CODEC_VOCAB_SIZE];
        talker_codec_logits(&runtime, &model, &hidden, &mut part);
        for i in (0..CODEBOOK_SIZE).chain([CODEC_EOS as usize]) {
            assert_eq!(full[i].to_bits(), part[i].to_bits(), "fila {i}");
        }
    }
}
