// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use qwen_core::config::{
    HEAD_DIM, HIDDEN_SIZE, INTERMEDIATE_SIZE, KV_DIM, NUM_HEADS, NUM_KV_HEADS, NUM_LAYERS, Q_DIM,
};
use qwen_kernels::{ComputeRuntime, Transpose};

use crate::error::EngineError;
use crate::model::{Model06B, SessionScratch, TransformerLayerWeights};

/// Ejecuta el *prefill* completo del Talker (`kv_len = 0`) sobre `input_embeds[seq_len, HIDDEN_SIZE]`.
pub fn talker_prefill(
    runtime: &ComputeRuntime,
    model: &Model06B,
    scratch: &mut SessionScratch,
    input_embeds: &[f32],
    seq_len: usize,
    silent: bool,
    debug: bool,
) -> Result<(), EngineError> {
    let eps = model.config.rms_norm_eps;
    let d = runtime.dispatch;
    let pool = &runtime.pool;
    let blas = &runtime.blas;

    if !silent {
        eprintln!("  Prefill:{} tokens, hidden={}", seq_len, HIDDEN_SIZE);
    }
    if seq_len > model.talker.rope_cos.len() / (HEAD_DIM / 2) {
        return Err(EngineError::Runtime(format!(
            "Prompt too long ({seq_len} tokens > RoPE cache)"
        )));
    }

    scratch.talker_kv.ensure_capacity(seq_len);
    scratch.talker_bufs.ensure_prefill_capacity(seq_len);

    let tb = &mut scratch.talker_bufs;
    let residual = &mut tb.pref_residual[..seq_len * HIDDEN_SIZE];
    residual.copy_from_slice(&input_embeds[..seq_len * HIDDEN_SIZE]);

    let pref_q = &mut tb.pref_q[..seq_len * Q_DIM];
    let pref_k = &mut tb.pref_k[..seq_len * KV_DIM];
    let pref_v = &mut tb.pref_v[..seq_len * KV_DIM];
    let pref_x_norm = &mut tb.pref_x_norm[..seq_len * HIDDEN_SIZE];
    let pref_attn_out = &mut tb.pref_attn_out[..seq_len * Q_DIM];
    let pref_gate = &mut tb.pref_gate[..seq_len * 2 * INTERMEDIATE_SIZE];
    let pref_proj = &mut tb.pref_proj[..seq_len * HIDDEN_SIZE];
    let swiglu_tmp = &mut tb.swiglu_tmp[..INTERMEDIATE_SIZE];

    let wq_f32 = &mut tb.pref_wq_f32[..Q_DIM * HIDDEN_SIZE];
    let wk_f32 = &mut tb.pref_wk_f32[..KV_DIM * HIDDEN_SIZE];
    let wv_f32 = &mut tb.pref_wv_f32[..KV_DIM * HIDDEN_SIZE];
    let wo_f32 = &mut tb.pref_wo_f32[..HIDDEN_SIZE * Q_DIM];
    let gate_up_f32 = &mut tb.pref_gate_up_f32[..2 * INTERMEDIATE_SIZE * HIDDEN_SIZE];
    let down_f32 = &mut tb.pref_down_f32[..HIDDEN_SIZE * INTERMEDIATE_SIZE];

    let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
    let kv_max = scratch.talker_kv.max_pos;

    for (layer_idx, l) in model.talker.layers.iter().enumerate() {
        d.convert_talker_layer_bf16_to_f32(
            pool,
            wq_f32,
            &l.wq_bf16,
            wk_f32,
            &l.wk_bf16,
            wv_f32,
            &l.wv_bf16,
            wo_f32,
            &l.wo_bf16,
            gate_up_f32,
            &l.gate_bf16,
            &l.up_bf16,
            down_f32,
            &l.down_bf16,
            INTERMEDIATE_SIZE,
            HIDDEN_SIZE,
        );

        d.rms_norm(pref_x_norm, residual, &l.input_norm, seq_len, HIDDEN_SIZE, eps);

        blas.sgemm(
            Transpose::NoTrans,
            Transpose::Trans,
            seq_len,
            Q_DIM,
            HIDDEN_SIZE,
            1.0,
            pref_x_norm,
            HIDDEN_SIZE,
            wq_f32,
            HIDDEN_SIZE,
            0.0,
            pref_q,
            Q_DIM,
        );
        blas.sgemm(
            Transpose::NoTrans,
            Transpose::Trans,
            seq_len,
            KV_DIM,
            HIDDEN_SIZE,
            1.0,
            pref_x_norm,
            HIDDEN_SIZE,
            wk_f32,
            HIDDEN_SIZE,
            0.0,
            pref_k,
            KV_DIM,
        );
        blas.sgemm(
            Transpose::NoTrans,
            Transpose::Trans,
            seq_len,
            KV_DIM,
            HIDDEN_SIZE,
            1.0,
            pref_x_norm,
            HIDDEN_SIZE,
            wv_f32,
            HIDDEN_SIZE,
            0.0,
            pref_v,
            KV_DIM,
        );

        d.rms_norm_per_head(pref_q, &l.q_norm, seq_len, NUM_HEADS, HEAD_DIM, eps);
        d.rms_norm_per_head(pref_k, &l.k_norm, seq_len, NUM_KV_HEADS, HEAD_DIM, eps);

        for s in 0..seq_len {
            d.apply_rope_neox_inplace(
                &mut pref_q[s * Q_DIM..(s + 1) * Q_DIM],
                NUM_HEADS,
                HEAD_DIM,
                &model.talker.rope_cos,
                &model.talker.rope_sin,
                s,
            );
            d.apply_rope_neox_inplace(
                &mut pref_k[s * KV_DIM..(s + 1) * KV_DIM],
                NUM_KV_HEADS,
                HEAD_DIM,
                &model.talker.rope_cos,
                &model.talker.rope_sin,
                s,
            );
        }

        let cache_base = layer_idx * kv_max * KV_DIM;
        d.f32_to_bf16_vec(
            &mut scratch.talker_kv.k[cache_base..cache_base + seq_len * KV_DIM],
            pref_k,
        );
        d.f32_to_bf16_vec(
            &mut scratch.talker_kv.v[cache_base..cache_base + seq_len * KV_DIM],
            pref_v,
        );

        d.causal_attention(
            pref_attn_out,
            pref_q,
            pref_k,
            pref_v,
            seq_len,
            seq_len,
            NUM_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            scale,
            0,
        );

        blas.sgemm(
            Transpose::NoTrans,
            Transpose::Trans,
            seq_len,
            HIDDEN_SIZE,
            Q_DIM,
            1.0,
            pref_attn_out,
            Q_DIM,
            wo_f32,
            Q_DIM,
            0.0,
            pref_proj,
            HIDDEN_SIZE,
        );
        d.add_inplace(residual, pref_proj);

        d.rms_norm(
            pref_x_norm,
            residual,
            &l.post_attn_norm,
            seq_len,
            HIDDEN_SIZE,
            eps,
        );

        blas.sgemm(
            Transpose::NoTrans,
            Transpose::Trans,
            seq_len,
            2 * INTERMEDIATE_SIZE,
            HIDDEN_SIZE,
            1.0,
            pref_x_norm,
            HIDDEN_SIZE,
            gate_up_f32,
            HIDDEN_SIZE,
            0.0,
            pref_gate,
            2 * INTERMEDIATE_SIZE,
        );

        for s in 0..seq_len {
            let src_start = s * 2 * INTERMEDIATE_SIZE;
            d.swiglu_inplace(
                &mut pref_gate[src_start..src_start + 2 * INTERMEDIATE_SIZE],
                swiglu_tmp,
                INTERMEDIATE_SIZE,
            );
            let dst_start = s * INTERMEDIATE_SIZE;
            if dst_start != src_start {
                pref_gate.copy_within(src_start..src_start + INTERMEDIATE_SIZE, dst_start);
            }
        }

        blas.sgemm(
            Transpose::NoTrans,
            Transpose::Trans,
            seq_len,
            HIDDEN_SIZE,
            INTERMEDIATE_SIZE,
            1.0,
            &pref_gate[..seq_len * INTERMEDIATE_SIZE],
            INTERMEDIATE_SIZE,
            down_f32,
            INTERMEDIATE_SIZE,
            0.0,
            pref_proj,
            HIDDEN_SIZE,
        );
        d.add_inplace(residual, pref_proj);

        if debug {
            eprintln!(
                "  Layer {}/{} done res[:4]=[{:.4},{:.4},{:.4},{:.4}]",
                layer_idx + 1,
                NUM_LAYERS,
                residual[0],
                residual[1],
                residual[2],
                residual[3]
            );
        }
    }

    scratch.talker_kv.len = seq_len;
    tb.dec_x[..HIDDEN_SIZE]
        .copy_from_slice(&residual[(seq_len - 1) * HIDDEN_SIZE..seq_len * HIDDEN_SIZE]);

    if debug {
        eprint!("[PREFILL] last_hidden[:8]:");
        for v in &tb.dec_x[..8] {
            eprint!(" {:.6}", v);
        }
        eprintln!();
    }

    if !silent {
        eprintln!("  Prefill complete ({seq_len} tokens in KV cache)");
    }
    Ok(())
}

/// Ejecuta un paso autorregresivo del Talker para la posición `scratch.talker_kv.len`.
pub fn talker_step(
    runtime: &ComputeRuntime,
    model: &Model06B,
    scratch: &mut SessionScratch,
    embed: &[f32],
    hidden_out: &mut [f32],
) -> Result<(), EngineError> {
    let pos = scratch.talker_kv.len;
    scratch.talker_kv.ensure_capacity(pos + 1);
    let kv_max = scratch.talker_kv.max_pos;
    let eps = model.config.rms_norm_eps;
    let d = runtime.dispatch;
    let pool = &runtime.pool;
    let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();

    let tb = &mut scratch.talker_bufs;
    tb.dec_x[..HIDDEN_SIZE].copy_from_slice(&embed[..HIDDEN_SIZE]);

    for layer_idx in 0..NUM_LAYERS {
        let l = &model.talker.layers[layer_idx];

        if layer_idx == 0 {
            d.rms_norm(
                &mut tb.dec_x_norm,
                &tb.dec_x,
                &l.input_norm,
                1,
                HIDDEN_SIZE,
                eps,
            );
        }

        l.dispatch_qkv(
            runtime,
            &tb.dec_x_norm,
            &mut tb.dec_q,
            &mut tb.dec_k,
            &mut tb.dec_v,
            HIDDEN_SIZE,
            Q_DIM,
            KV_DIM,
        );

        d.rms_norm_per_head(&mut tb.dec_q, &l.q_norm, 1, NUM_HEADS, HEAD_DIM, eps);
        d.rms_norm_per_head(&mut tb.dec_k, &l.k_norm, 1, NUM_KV_HEADS, HEAD_DIM, eps);

        d.apply_rope_neox_inplace(
            &mut tb.dec_q,
            NUM_HEADS,
            HEAD_DIM,
            &model.talker.rope_cos,
            &model.talker.rope_sin,
            pos,
        );
        d.apply_rope_neox_inplace(
            &mut tb.dec_k,
            NUM_KV_HEADS,
            HEAD_DIM,
            &model.talker.rope_cos,
            &model.talker.rope_sin,
            pos,
        );

        let layer_base = layer_idx * kv_max * KV_DIM;
        let kv_off = layer_base + pos * KV_DIM;
        d.f32_to_bf16_vec(&mut scratch.talker_kv.k[kv_off..kv_off + KV_DIM], &tb.dec_k);
        d.f32_to_bf16_vec(&mut scratch.talker_kv.v[kv_off..kv_off + KV_DIM], &tb.dec_v);

        let layer_k = &scratch.talker_kv.k[layer_base..layer_base + (pos + 1) * KV_DIM];
        let layer_v = &scratch.talker_kv.v[layer_base..layer_base + (pos + 1) * KV_DIM];
        d.causal_attention_bf16kv_heads_parallel(
            pool,
            &mut tb.dec_attn_out,
            &tb.dec_q,
            layer_k,
            layer_v,
            1,
            pos + 1,
            NUM_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            scale,
            pos,
        );

        TransformerLayerWeights::dispatch_proj(
            runtime,
            l.wo_q.as_ref(),
            &l.wo_bf16,
            &tb.dec_attn_out,
            &mut tb.dec_proj_out,
            HIDDEN_SIZE,
            Q_DIM,
        );

        d.rms_norm_residual(
            &mut tb.dec_x_norm,
            &mut tb.dec_x,
            &tb.dec_proj_out,
            &l.post_attn_norm,
            HIDDEN_SIZE,
            eps,
        );

        l.dispatch_gate_up(
            runtime,
            &tb.dec_x_norm,
            &mut tb.dec_gate,
            INTERMEDIATE_SIZE,
            HIDDEN_SIZE,
        )?;
        d.swiglu_inplace(
            &mut tb.dec_gate,
            &mut tb.swiglu_tmp[..INTERMEDIATE_SIZE],
            INTERMEDIATE_SIZE,
        );

        TransformerLayerWeights::dispatch_proj(
            runtime,
            l.down_q.as_ref(),
            &l.down_bf16,
            &tb.dec_gate[..INTERMEDIATE_SIZE],
            &mut tb.dec_proj_out,
            HIDDEN_SIZE,
            INTERMEDIATE_SIZE,
        );

        if layer_idx + 1 < NUM_LAYERS {
            d.rms_norm_residual(
                &mut tb.dec_x_norm,
                &mut tb.dec_x,
                &tb.dec_proj_out,
                &model.talker.layers[layer_idx + 1].input_norm,
                HIDDEN_SIZE,
                eps,
            );
        } else {
            d.add_inplace(&mut tb.dec_x, &tb.dec_proj_out);
        }
    }

    d.rms_norm(
        hidden_out,
        &tb.dec_x,
        &model.talker.norm,
        1,
        HIDDEN_SIZE,
        eps,
    );
    scratch.talker_kv.len = pos + 1;
    Ok(())
}
