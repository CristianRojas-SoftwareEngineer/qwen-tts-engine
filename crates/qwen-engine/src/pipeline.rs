// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::Instant;

use qwen_core::config::{
    CODEBOOK_SIZE, CODEC_BOS, CODEC_EOS, CODEC_NO_THINK, CODEC_PAD, CODEC_THINK, CODEC_THINK_BOS,
    CODEC_THINK_EOS, CODEC_VOCAB_SIZE, FRAME_RATE, HIDDEN_SIZE, HOP_SAMPLES, NUM_CODEBOOKS,
    SAMPLE_RATE, TOKEN_ASSISTANT, TOKEN_IM_START, TOKEN_NEWLINE, TOKEN_TTS_BOS, TOKEN_TTS_EOS,
    TOKEN_TTS_PAD,
};
use qwen_core::LcgRng;
use qwen_kernels::{ftz_on, ComputeRuntime, KernelDispatch};

use crate::audio::{apply_gain, time_stretch, AudioSink, MemorySink};
use crate::code_predictor::cp_predict;
use crate::error::EngineError;
use crate::model::{
    embed_one_text_token_compute, lookup_codec_embed, Model06B, RequestConfig, RequestExecution,
    SessionScratch, SynthesisTiming, VoiceBindings, VoiceIdentity, ROPE_MAX_POS,
};
use crate::speech_decoder::speech_decoder_decode_streaming;
use crate::talker::{talker_prefill, talker_step};

const DEFAULT_CHUNK_FRAMES: usize = 10;
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
/// Las demás filas de `CODEBOOK_SIZE..CODEC_VOCAB_SIZE` (2 MiB de pesos BF16 por fotograma) no se
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

/// Aplica clamp de logits, supresión de tokens códec especiales y boost progresivo de EOS.
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

/// Calcula el tamaño objetivo de chunk en fotogramas según la rampa progresiva (la misma con y sin
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

/// Devuelve el número de muestras PCM iniciales a recortar cuando hay contexto ICL activo.
pub(crate) fn resolve_icl_trim_samples(icl_mode: bool) -> usize {
    if !icl_mode {
        return 0;
    }
    let tf = std::env::var("QWEN_ICL_TRIM_FRAMES")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(2);
    tf.saturating_mul(HOP_SAMPLES)
}

fn embed_text_cached(
    runtime: &ComputeRuntime,
    model: &Model06B,
    scratch: &mut SessionScratch,
    tid: i32,
    out: &mut [f32],
) {
    let (pad_emb, bos_emb, eos_emb) = (
        model.cached_tts_pad_embed.as_slice(),
        model.cached_tts_bos_embed.as_slice(),
        model.cached_tts_eos_embed.as_slice(),
    );

    if tid == TOKEN_TTS_PAD {
        out[..HIDDEN_SIZE].copy_from_slice(pad_emb);
        return;
    }
    if tid == TOKEN_TTS_BOS {
        out[..HIDDEN_SIZE].copy_from_slice(bos_emb);
        return;
    }
    if tid == TOKEN_TTS_EOS {
        out[..HIDDEN_SIZE].copy_from_slice(eos_emb);
        return;
    }

    let tb = &mut scratch.talker_bufs;
    let d = runtime.dispatch;
    let pool = &runtime.pool;
    scratch.emb_lru.get_or_insert(tid, out, |t, dst| {
        embed_one_text_token_compute(
            d,
            pool,
            model,
            t,
            &mut tb.emb_tmp1,
            &mut tb.emb_tmp2,
            dst,
        );
    });
}

pub fn build_prefill_embeds(
    runtime: &ComputeRuntime,
    model: &Model06B,
    voice: &VoiceBindings,
    scratch: &mut SessionScratch,
    req: &RequestConfig,
    text: &str,
) -> Result<(Vec<f32>, usize, usize, bool), EngineError> {
    let text_tokens = model.tokenizer.encode_para(text);
    if text_tokens.is_empty() {
        return Err(EngineError::InvalidArg(
            "Text tokenization produced 0 tokens".to_string(),
        ));
    }

    let (icl_text, ref_codes_slice): (Option<&str>, Option<&[i32]>) =
        if let VoiceIdentity::Icl {
            ref_text,
            ref_codes,
            ..
        } = &voice.identity
        {
            if !ref_text.is_empty() && !ref_codes.is_empty() {
                (Some(ref_text.as_str()), Some(ref_codes.as_slice()))
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

    let ref_text_tokens = icl_text.map(|t| model.tokenizer.encode(t)).unwrap_or_default();
    let mut ref_n_frames = ref_codes_slice
        .map(|c| c.len() / NUM_CODEBOOKS)
        .unwrap_or(0);
    let mut icl_mode = !ref_text_tokens.is_empty() && ref_n_frames > 0;
    if icl_mode {
        let mut cap = voice.icl_frames_cap;
        if let Ok(v) = std::env::var("QWEN_ICL_FRAMES") {
            if let Ok(n) = v.trim().parse::<usize>() {
                cap = n;
            }
        }
        if cap > 0 && cap < ref_n_frames {
            ref_n_frames = cap;
        }
        if ref_n_frames == 0 {
            icl_mode = false;
        }
    }

    let role_prefix = [TOKEN_IM_START, TOKEN_ASSISTANT, TOKEN_NEWLINE];
    let text_content_len = text_tokens.len();

    let mut codec_tokens = Vec::with_capacity(8);
    if let Some(lang_id) = req.language_id {
        codec_tokens.push(CODEC_THINK);
        codec_tokens.push(CODEC_THINK_BOS);
        codec_tokens.push(lang_id);
        codec_tokens.push(CODEC_THINK_EOS);
    } else {
        codec_tokens.push(CODEC_NO_THINK);
        codec_tokens.push(CODEC_THINK_BOS);
        codec_tokens.push(CODEC_THINK_EOS);
    }
    let spk_emb: &[f32] = match &voice.identity {
        VoiceIdentity::XVector(emb) => emb.as_slice(),
        VoiceIdentity::Icl {
            speaker_embedding, ..
        } => speaker_embedding.as_slice(),
    };
    codec_tokens.push(-1);
    codec_tokens.push(CODEC_PAD);
    codec_tokens.push(CODEC_BOS);

    let sec2_len = codec_tokens.len() - 1;
    let (sec3_len, sec4_len) = if icl_mode {
        (ref_text_tokens.len() + text_content_len + 1, ref_n_frames + 1)
    } else {
        (text_content_len + 1, 1)
    };
    let prefill_len = role_prefix.len() + sec2_len + sec3_len + sec4_len;
    if prefill_len > ROPE_MAX_POS {
        return Err(EngineError::InvalidArg(format!(
            "Prompt too long ({prefill_len} tokens > RoPE cache {ROPE_MAX_POS}); shorten the text."
        )));
    }

    let (tts_pad_embed, tts_bos_embed, tts_eos_embed) = (
        model.cached_tts_pad_embed.as_slice(),
        model.cached_tts_bos_embed.as_slice(),
        model.cached_tts_eos_embed.as_slice(),
    );

    let d = runtime.dispatch;
    let mut codec_pad_embed = [0.0f32; HIDDEN_SIZE];
    let mut codec_bos_embed = [0.0f32; HIDDEN_SIZE];
    lookup_codec_embed(d, model, CODEC_PAD, &mut codec_pad_embed);
    lookup_codec_embed(d, model, CODEC_BOS, &mut codec_bos_embed);

    let mut input_embeds = vec![0.0f32; prefill_len * HIDDEN_SIZE];
    let mut tmp_embed = [0.0f32; HIDDEN_SIZE];
    let mut pos = 0usize;

    // Sección 1: prefijo de rol (solo texto)
    for &tid in &role_prefix {
        embed_text_cached(
            runtime,
            model,
            scratch,
            tid,
            &mut input_embeds[pos * HIDDEN_SIZE..(pos + 1) * HIDDEN_SIZE],
        );
        pos += 1;
    }

    // Sección 2: tts_pad*(sec2_len-1) + tts_bos emparejado con codec[0..sec2_len]
    for (i, &ctok) in codec_tokens[..sec2_len].iter().enumerate() {
        let dst = &mut input_embeds[pos * HIDDEN_SIZE..(pos + 1) * HIDDEN_SIZE];
        if i + 1 < sec2_len {
            dst.copy_from_slice(tts_pad_embed);
        } else {
            dst.copy_from_slice(tts_bos_embed);
        }
        if ctok == -1 {
            let emb_norm = spk_emb.iter().map(|&v| v * v).sum::<f32>().sqrt();
            let mut scale = if model.spk_ref_norm > 0.1 && emb_norm > 0.1 {
                model.spk_ref_norm / emb_norm
            } else {
                1.0
            };
            if let Ok(sse) = std::env::var("QWEN_SPK_SCALE") {
                if let Ok(f) = sse.trim().parse::<f32>() {
                    if f.is_finite() {
                        scale *= f;
                    }
                }
            }
            for j in 0..HIDDEN_SIZE {
                dst[j] += spk_emb[j] * scale;
            }
        } else {
            lookup_codec_embed(d, model, ctok, &mut tmp_embed);
            for j in 0..HIDDEN_SIZE {
                dst[j] += tmp_embed[j];
            }
        }
        pos += 1;
    }

    if icl_mode {
        let ref_codes = ref_codes_slice.expect("icl_mode implies ref_codes");
        for i in 0..sec3_len {
            let dst = &mut input_embeds[pos * HIDDEN_SIZE..(pos + 1) * HIDDEN_SIZE];
            if i < ref_text_tokens.len() {
                embed_text_cached(runtime, model, scratch, ref_text_tokens[i], dst);
            } else if i < ref_text_tokens.len() + text_content_len {
                let idx = i - ref_text_tokens.len();
                embed_text_cached(runtime, model, scratch, text_tokens[idx], dst);
            } else {
                dst.copy_from_slice(tts_eos_embed);
            }
            for j in 0..HIDDEN_SIZE {
                dst[j] += codec_pad_embed[j];
            }
            pos += 1;
        }

        for i in 0..sec4_len {
            let dst = &mut input_embeds[pos * HIDDEN_SIZE..(pos + 1) * HIDDEN_SIZE];
            dst.copy_from_slice(tts_pad_embed);
            if i == 0 {
                for j in 0..HIDDEN_SIZE {
                    dst[j] += codec_bos_embed[j];
                }
            } else {
                let frame = i - 1;
                let code0 = ref_codes[frame * NUM_CODEBOOKS];
                lookup_codec_embed(d, model, code0, &mut tmp_embed);
                for j in 0..HIDDEN_SIZE {
                    dst[j] += tmp_embed[j];
                }
                for g in 0..15 {
                    let code_g = ref_codes[frame * NUM_CODEBOOKS + g + 1];
                    if code_g >= 0 && (code_g as usize) < CODEBOOK_SIZE {
                        let off = (code_g as usize) * HIDDEN_SIZE;
                        let emb = &model.cp.codec_emb_bf16[g][off..off + HIDDEN_SIZE];
                        d.bf16_accum_f32(dst, emb);
                    }
                }
            }
            pos += 1;
        }
    } else {
        for &tok in &text_tokens {
            let dst = &mut input_embeds[pos * HIDDEN_SIZE..(pos + 1) * HIDDEN_SIZE];
            embed_text_cached(runtime, model, scratch, tok, dst);
            for (d_val, &cp_val) in dst.iter_mut().zip(codec_pad_embed.iter()) {
                *d_val += cp_val;
            }
            pos += 1;
        }
        let dst = &mut input_embeds[pos * HIDDEN_SIZE..(pos + 1) * HIDDEN_SIZE];
        dst.copy_from_slice(tts_eos_embed);
        for (d_val, &cp_val) in dst.iter_mut().zip(codec_pad_embed.iter()) {
            *d_val += cp_val;
        }
        pos += 1;

        let dst = &mut input_embeds[pos * HIDDEN_SIZE..(pos + 1) * HIDDEN_SIZE];
        dst.copy_from_slice(tts_pad_embed);
        for j in 0..HIDDEN_SIZE {
            dst[j] += codec_bos_embed[j];
        }
        pos += 1;
    }

    debug_assert_eq!(pos, prefill_len);
    Ok((input_embeds, prefill_len, text_content_len, icl_mode))
}

/// Ejecuta la síntesis completa y entrega progresivamente los fragmentos de audio PCM a `sink`.
pub fn synthesize_to_sink(
    runtime: &ComputeRuntime,
    model: &Model06B,
    voice: &VoiceBindings,
    scratch: &mut SessionScratch,
    exec: &RequestExecution,
    text: &str,
    sink: &mut (dyn AudioSink + Send),
) -> Result<SynthesisTiming, EngineError> {
    let req = &exec.config;
    let t_start = Instant::now();
    scratch.rng = LcgRng::new(req.seed);

    let (input_embeds, prefill_len, text_content_len, icl_mode) =
        build_prefill_embeds(runtime, model, voice, scratch, req, text)?;

    if !req.silent {
        eprintln!(
            "Text: \"{}\" (prompt: {} positions, text_content: {})",
            text,
            prefill_len,
            text_content_len
        );
        eprintln!(
            "Prefill: {} positions (text_content={})",
            prefill_len, text_content_len
        );
    }

    let nt = runtime.pool.n_threads();
    runtime.blas.set_threads(nt);

    scratch.talker_kv.len = 0;
    let t_pref = Instant::now();
    talker_prefill(
        runtime,
        model,
        scratch,
        &input_embeds,
        prefill_len,
        req.silent,
        req.debug,
    )?;
    let prefill_ms = t_pref.elapsed().as_secs_f64() * 1000.0;
    if !req.silent {
        eprintln!("  Prefill: {:.0} ms ({} tokens)", prefill_ms, prefill_len);
    }

    let d = runtime.dispatch;
    let eps = model.config.rms_norm_eps;
    {
        let tb = &mut scratch.talker_bufs;
        d.rms_norm(
            &mut tb.last_hidden,
            &tb.dec_x,
            &model.talker.norm,
            1,
            HIDDEN_SIZE,
            eps,
        );
        if req.debug {
            eprint!("[PREFILL] after_norm[:8]:");
            for v in &tb.last_hidden[..8] {
                eprint!(" {:.6}", v);
            }
            eprintln!();
        }
    }

    let mut max_frames = req.max_tokens;
    let rope_rem = ROPE_MAX_POS.saturating_sub(prefill_len);
    if max_frames > rope_rem {
        max_frames = rope_rem;
    }

    scratch.codec_codes.clear();
    scratch.codec_codes.reserve(max_frames * NUM_CODEBOOKS);
    scratch.prev_tokens.clear();
    scratch.prev_tokens.reserve(max_frames);

    let mut dump_codes_writer = match std::env::var("QWEN_DUMP_CODES") {
        Ok(p) if !p.trim().is_empty() => Some(BufWriter::new(File::create(p.trim())?)),
        _ => None,
    };

    let chunk_frames = if req.stream_chunk_frames > 0 {
        req.stream_chunk_frames.clamp(2, 250)
    } else {
        DEFAULT_CHUNK_FRAMES
    };

    let mut trim_head_left = resolve_icl_trim_samples(icl_mode);

    let mut gen_blas = if nt > 1 { nt - 1 } else { 1 };
    if let Ok(v) = std::env::var("QWEN_BLAS_GEN_THREADS") {
        if let Ok(n) = v.trim().parse::<usize>() {
            if n > 0 {
                gen_blas = n;
            }
        }
    }
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
    let mut prod_err: Option<EngineError> = None;
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
                // Mismo calendario de fragmentos con y sin streaming: las fronteras de
                // fragmento fijan el redondeo del decodificador, así el PCM es idéntico.
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
                        let mut emit = audio.as_slice();
                        if trim_head_left > 0 {
                            let cut = trim_head_left.min(emit.len());
                            emit = &emit[cut..];
                            trim_head_left -= cut;
                        }
                        if !emit.is_empty() {
                            if ttfa_ms.is_none() {
                                ttfa_ms = Some(t_start.elapsed().as_secs_f64() * 1000.0);
                            }
                            if let Err(e) = sink.write_samples(emit) {
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
            if exec.abort.is_cancelled() {
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

            let (frame_temp, frame_top_k) = if req.greedy_warmup > 0 && frame < req.greedy_warmup {
                (0.0, 1)
            } else {
                (req.temperature, req.top_k)
            };
            let code0 = scratch.sampler.sample(
                &mut scratch.talker_bufs.logits,
                frame_temp,
                frame_top_k,
                req.top_p,
                req.rep_penalty,
                &scratch.prev_tokens,
                &mut scratch.rng,
            ) as i32;

            if code0 == CODEC_EOS {
                if !req.silent {
                    eprintln!("  EOS at frame {}", frame);
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
                req.cp_roughness,
                &mut cp_out,
            );
            cp_ms += t_cp.elapsed().as_secs_f64() * 1000.0;
            codes[1..16].copy_from_slice(&cp_out);

            if let Some(ref mut w) = dump_codes_writer {
                let _ = write!(w, "{}", codes[0]);
                for &c in &codes[1..] {
                    let _ = write!(w, " {}", c);
                }
                let _ = writeln!(w);
            }

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
            if req.debug && frame < 2 {
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
            if let Err(e) = talker_step(
                runtime,
                model,
                scratch,
                &step_embed_copy,
                &mut lh_copy,
            ) {
                prod_err = Some(e);
                queue.aborted.store(true, Ordering::Release);
                break;
            }
            scratch.talker_bufs.last_hidden.copy_from_slice(&lh_copy);
            if req.debug && frame < 2 {
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

    if let Some(ref mut w) = dump_codes_writer {
        let _ = w.flush();
    }

    if was_aborted {
        return Err(EngineError::Aborted);
    }
    if let Some(e) = prod_err {
        return Err(e);
    }
    if let Some(e) = decoder_err {
        return Err(e);
    }

    sink.finish()?;

    let frames = scratch.codec_codes.len() / NUM_CODEBOOKS;
    let total_ms = t_start.elapsed().as_secs_f64() * 1000.0;
    if !req.silent {
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
        "\n  Generated {} frames ({:.1}s audio)",
        t.frames,
        (t.frames as f64) / (FRAME_RATE as f64)
    );
    if t.frames > 0 {
        eprintln!(
            "  Talker step: {:.0} ms ({:.1} ms/f), Code Predictor: {:.0} ms ({:.1} ms/f)",
            t.talker_ms,
            t.talker_ms / (t.frames as f64),
            t.cp_ms,
            t.cp_ms / (t.frames as f64)
        );
    }
    eprintln!(
        "  Speech decoder: {:.0} ms total ({:.0} ms drain after gen)",
        t.decoder_ms, t.drain_ms
    );
    let audio_dur = (t.samples_produced as f64) / (SAMPLE_RATE as f64);
    let proc_s = t.total_ms / 1000.0;
    if audio_dur > 0.0 {
        eprintln!(
            "Audio: {:.1}s generated in {:.1}s (RTF {:.2})",
            audio_dur,
            proc_s,
            proc_s / audio_dur
        );
    }
    if let Some(ttfa) = t.ttfa_ms {
        eprintln!(
            "  TTFA: {:.0} ms (first audio chunk, {}-frame chunk)",
            ttfa, chunk_frames
        );
    }
}

/// Ejecuta la síntesis completa y devuelve el buffer mono `f32` a 24 kHz junto con las métricas.
pub fn synthesize_to_samples(
    runtime: &ComputeRuntime,
    model: &Model06B,
    voice: &VoiceBindings,
    scratch: &mut SessionScratch,
    exec: &RequestExecution,
    text: &str,
) -> Result<(Vec<f32>, SynthesisTiming), EngineError> {
    let mut sink = MemorySink::new(exec.config.max_tokens.min(1024) * HOP_SAMPLES);
    let mut timing = synthesize_to_sink(runtime, model, voice, scratch, exec, text, &mut sink)?;
    let mut samples = sink.samples;
    apply_gain(&mut samples, exec.config.volume);
    if (exec.config.rate - 1.0).abs() > 1e-4 {
        samples = time_stretch(&samples, exec.config.rate);
    }
    timing.samples_produced = samples.len();
    Ok((samples, timing))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelLoadOptions;
    use std::path::PathBuf;

    /// Las filas que `talker_codec_logits` no calcula no influyen en los logits tras las reglas,
    /// con cualquier valor previo, en cualquier fotograma y con y sin boost de EOS.
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
        let model = Model06B::load(&dir, &runtime, ModelLoadOptions { quant_mixed: true, silent: true, ..Default::default() })
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
