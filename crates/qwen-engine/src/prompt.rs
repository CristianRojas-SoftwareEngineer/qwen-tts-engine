// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Construcción del prompt (embeddings de entrada del Talker) y su *prefill*.

use std::time::Instant;

use qwen_core::config::{
    CODEC_BOS, CODEC_PAD, CODEC_THINK, CODEC_THINK_BOS, CODEC_THINK_EOS, HIDDEN_SIZE,
    TOKEN_ASSISTANT, TOKEN_IM_START, TOKEN_NEWLINE, TOKEN_TTS_BOS, TOKEN_TTS_EOS, TOKEN_TTS_PAD,
};
use qwen_kernels::ComputeRuntime;

use crate::error::EngineError;
use crate::model::{
    embed_one_text_token_compute, lookup_codec_embed, Model06B, SessionScratch, ROPE_MAX_POS,
};
use crate::request::LogLevel;
use crate::talker::talker_prefill;
use crate::voice::Voice;

/// Prompt ya construido: embeddings de entrada del Talker y los datos que el bucle necesita.
pub(crate) struct Prompt {
    /// `[len, HIDDEN_SIZE]` embeddings de entrada.
    pub embeds: Vec<f32>,
    /// Posiciones del prompt.
    pub len: usize,
    /// Tokens del texto a sintetizar.
    pub text_len: usize,
    /// Diferencia entre el embedding de texto de cada paso y `tts_pad`; se suma a la entrada del paso
    /// `i`, que ya lleva `tts_pad`. Sumar la diferencia, y no sustituir `tts_pad`, fija el orden de
    /// redondeo con el que se validó el audio.
    pub trailing_text_delta: Vec<f32>,
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
        embed_one_text_token_compute(d, pool, model, t, &mut tb.emb_tmp1, &mut tb.emb_tmp2, dst);
    });
}

/// Elemento de la sección 2 del prompt: un token de códec o el x-vector del locutor.
enum Sec2Token {
    Codec(i32),
    Speaker,
}

/// Construye el prompt de `text` con la voz `voice` y el idioma `language_id`.
///
/// El texto entra un token por paso, como en el modo por defecto de la implementación oficial: el
/// prompt lleva solo el primer token con `codec_bos`; los siguientes y `tts_eos` se suman a la
/// entrada de los pasos 0, 1, 2…
pub(crate) fn build_prompt(
    runtime: &ComputeRuntime,
    model: &Model06B,
    voice: &Voice,
    language_id: i32,
    text: &str,
    scratch: &mut SessionScratch,
) -> Result<Prompt, EngineError> {
    let text_tokens = model.tokenizer.encode(text);
    if text_tokens.is_empty() {
        return Err(EngineError::InvalidArg(
            "La tokenización del texto produjo 0 tokens".to_string(),
        ));
    }

    let role_prefix = [TOKEN_IM_START, TOKEN_ASSISTANT, TOKEN_NEWLINE];
    let text_content_len = text_tokens.len();

    // Secuencia de códec de la sección 2: prefijo de idioma, locutor y `CODEC_PAD`.
    let codec_prefix = [CODEC_THINK, CODEC_THINK_BOS, language_id, CODEC_THINK_EOS];
    let spk_emb: &[f32] = voice.xvector.as_slice();
    // Posición explícita del locutor, tras el prefijo; después va `CODEC_PAD`.
    let sec2: Vec<Sec2Token> = codec_prefix
        .iter()
        .map(|&t| Sec2Token::Codec(t))
        .chain([Sec2Token::Speaker, Sec2Token::Codec(CODEC_PAD)])
        .collect();
    let sec2_len = sec2.len();
    // Una sola posición de texto en el prompt; el resto entra paso a paso.
    let prefill_len = role_prefix.len() + sec2_len + 1;
    let n_trailing = text_content_len;
    if prefill_len + n_trailing > ROPE_MAX_POS {
        let n = prefill_len + n_trailing;
        return Err(EngineError::InvalidArg(format!(
            "Prompt demasiado largo ({n} posiciones > caché RoPE {ROPE_MAX_POS}); acorta el texto."
        )));
    }

    let (tts_pad_embed, tts_bos_embed, tts_eos_embed) = (
        model.cached_tts_pad_embed.as_slice(),
        model.cached_tts_bos_embed.as_slice(),
        model.cached_tts_eos_embed.as_slice(),
    );

    let d = runtime.dispatch;
    let mut codec_bos_embed = [0.0f32; HIDDEN_SIZE];
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
    for (i, tok) in sec2.iter().enumerate() {
        let dst = &mut input_embeds[pos * HIDDEN_SIZE..(pos + 1) * HIDDEN_SIZE];
        if i + 1 < sec2_len {
            dst.copy_from_slice(tts_pad_embed);
        } else {
            dst.copy_from_slice(tts_bos_embed);
        }
        if let Sec2Token::Codec(ctok) = *tok {
            lookup_codec_embed(d, model, ctok, &mut tmp_embed);
            for j in 0..HIDDEN_SIZE {
                dst[j] += tmp_embed[j];
            }
        } else {
            for j in 0..HIDDEN_SIZE {
                dst[j] += spk_emb[j];
            }
        }
        pos += 1;
    }

    // Sección 3: primer token del texto más `codec_bos`; los demás entran en los pasos del bucle.
    let dst = &mut input_embeds[pos * HIDDEN_SIZE..(pos + 1) * HIDDEN_SIZE];
    embed_text_cached(runtime, model, scratch, text_tokens[0], dst);
    for (d_val, &cb_val) in dst.iter_mut().zip(codec_bos_embed.iter()) {
        *d_val += cb_val;
    }
    pos += 1;

    // Diferencias de los pasos 0, 1, 2…: cada token restante del texto y, al final, `tts_eos`.
    let mut trailing_text_delta = vec![0.0f32; n_trailing * HIDDEN_SIZE];
    for (i, &tok) in text_tokens[1..].iter().enumerate() {
        let delta = &mut trailing_text_delta[i * HIDDEN_SIZE..(i + 1) * HIDDEN_SIZE];
        embed_text_cached(runtime, model, scratch, tok, delta);
        for (d_val, &pad_val) in delta.iter_mut().zip(tts_pad_embed.iter()) {
            *d_val -= pad_val;
        }
    }
    let delta = &mut trailing_text_delta[(n_trailing - 1) * HIDDEN_SIZE..];
    for ((d_val, &eos_val), &pad_val) in delta
        .iter_mut()
        .zip(tts_eos_embed.iter())
        .zip(tts_pad_embed.iter())
    {
        *d_val = eos_val - pad_val;
    }

    debug_assert_eq!(pos, prefill_len);
    Ok(Prompt {
        embeds: input_embeds,
        len: prefill_len,
        text_len: text_content_len,
        trailing_text_delta,
    })
}

/// *Prefill* del Talker sobre `prompt`, con la `rms_norm` final.
///
/// Deja el estado oculto normalizado del último paso en `scratch.talker_bufs.last_hidden` y
/// devuelve la duración del *prefill* en milisegundos.
pub(crate) fn prefill(
    runtime: &ComputeRuntime,
    model: &Model06B,
    scratch: &mut SessionScratch,
    prompt: &Prompt,
    log: LogLevel,
) -> Result<f64, EngineError> {
    scratch.talker_kv.len = 0;
    let t_pref = Instant::now();
    talker_prefill(runtime, model, scratch, &prompt.embeds, prompt.len, log)?;
    let prefill_ms = t_pref.elapsed().as_secs_f64() * 1000.0;
    if !log.is_silent() {
        eprintln!("  Prefill: {:.0} ms ({} tokens)", prefill_ms, prompt.len);
    }

    let eps = model.config.rms_norm_eps;
    let tb = &mut scratch.talker_bufs;
    runtime.dispatch.rms_norm(
        &mut tb.last_hidden,
        &tb.dec_x,
        &model.talker.norm,
        1,
        HIDDEN_SIZE,
        eps,
    );
    if log.is_debug() {
        eprint!("[PREFILL] after_norm[:8]:");
        for v in &tb.last_hidden[..8] {
            eprint!(" {:.6}", v);
        }
        eprintln!();
    }
    Ok(prefill_ms)
}
