// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::sync::OnceLock;
use std::time::Instant;

use qwen_core::config::{
    CODEBOOK_SIZE, CODEC_EOS, CODEC_VOCAB_SIZE, CP_HEAD_DIM, CP_HIDDEN_SIZE,
    CP_INTERMEDIATE_SIZE, CP_KV_DIM, CP_NUM_HEADS, CP_NUM_KV_HEADS, CP_NUM_LAYERS, CP_Q_DIM,
    HEAD_DIM, HIDDEN_SIZE, INTERMEDIATE_SIZE, KV_DIM, NUM_CODEBOOKS, NUM_HEADS, NUM_KV_HEADS,
    NUM_LAYERS, Q_DIM, SAMPLE_RATE,
};
use qwen_core::{LcgRng, Sampler};
use qwen_kernels::{AlignedBuf, ComputeRuntime};

use crate::audio::apply_gain;
use crate::error::EngineError;
use crate::model::{
    lookup_codec_embed, CpLmHead, Model06B, QuantWeight, RequestConfig, SessionScratch,
    VoiceBindings, ROPE_MAX_POS,
};
use crate::pipeline::build_prefill_embeds;
use crate::speech_decoder::{speech_decoder_decode, speech_decoder_decode_streaming};
use crate::talker::talker_step;

pub const MAX_BATCH_SLOTS: usize = 64;
const CP_KV_MAX: usize = 64;

fn env_batch_nomatmul() -> bool {
    static VAL: OnceLock<bool> = OnceLock::new();
    *VAL.get_or_init(|| {
        std::env::var("QWEN_BATCH_NOMATMUL")
            .ok()
            .map(|v| !v.trim().is_empty() && v.trim() != "0")
            .unwrap_or(false)
    })
}

fn env_batch_force_matvec() -> bool {
    std::env::var("QWEN_BATCH_FORCE_MATVEC")
        .ok()
        .map(|v| !v.trim().is_empty() && v.trim() != "0")
        .unwrap_or(false)
}

#[inline]
fn batch_gather(xt: &mut [f32], src: &[f32], b_dim: usize, cols: usize, src_stride: usize) {
    for k in 0..cols {
        let row = &mut xt[k * b_dim..(k + 1) * b_dim];
        for b in 0..b_dim {
            row[b] = src[b * src_stride + k];
        }
    }
}

#[inline]
fn batch_scatter(dst: &mut [f32], yt: &[f32], b_dim: usize, rows: usize) {
    for r in 0..rows {
        let yrow = &yt[r * b_dim..(r + 1) * b_dim];
        for b in 0..b_dim {
            dst[b * rows + r] = yrow[b];
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn batch_proj_q(
    runtime: &ComputeRuntime,
    bw_xt: &mut [f32],
    bw_yt: &mut [f32],
    dst: &mut [f32],
    q_w: Option<&QuantWeight>,
    bf16_w: &[u16],
    src: &[f32],
    rows: usize,
    cols: usize,
    src_stride: usize,
    b_dim: usize,
    force_matvec: bool,
) {
    let d = runtime.dispatch;
    let pool = &runtime.pool;
    if force_matvec || env_batch_nomatmul() {
        for b in 0..b_dim {
            let s = &src[b * src_stride..b * src_stride + cols];
            let out = &mut dst[b * rows..(b + 1) * rows];
            match q_w {
                Some(QuantWeight::Q4_0(w)) => d.matvec_q4_0(pool, out, w, s, rows, cols),
                Some(QuantWeight::Int8 { data, scale }) => {
                    d.matvec_int8(pool, out, data, scale, s, rows, cols);
                }
                None => d.matvec_bf16(pool, out, bf16_w, s, rows, cols),
            }
        }
        return;
    }

    batch_gather(&mut bw_xt[..cols * b_dim], src, b_dim, cols, src_stride);
    let xt = &bw_xt[..cols * b_dim];
    let yt = &mut bw_yt[..rows * b_dim];
    match q_w {
        Some(QuantWeight::Q4_0(w)) => d.matmat_q4_0(pool, yt, w, xt, rows, cols, b_dim),
        Some(QuantWeight::Int8 { data, scale }) => {
            d.matmat_int8(pool, yt, data, scale, xt, rows, cols, b_dim);
        }
        None => d.matmat_bf16(pool, yt, bf16_w, xt, rows, cols, b_dim),
    }
    batch_scatter(dst, yt, b_dim, rows);
}

fn cp_lm_argmax(runtime: &ComputeRuntime, head: &CpLmHead, normed: &[f32]) -> usize {
    let d = runtime.dispatch;
    let pool = &runtime.pool;
    match &head.quant {
        Some(QuantWeight::Q4_0(w)) => {
            d.argmax_matvec_q4_0(pool, normed, w, CP_HIDDEN_SIZE, CODEBOOK_SIZE)
        }
        Some(QuantWeight::Int8 { data, scale }) => {
            d.argmax_matvec_int8(pool, normed, data, scale, CP_HIDDEN_SIZE, CODEBOOK_SIZE)
        }
        None => d.argmax_matvec_bf16(normed, &head.bf16, CP_HIDDEN_SIZE, CODEBOOK_SIZE),
    }
}

pub struct BatchWorkspace {
    pub b_dim: usize,
    pub kv_max: usize,
    pub kv_len: usize,
    pub force_matvec: bool,
    pub x: AlignedBuf<f32>,
    pub x_norm: AlignedBuf<f32>,
    pub q: AlignedBuf<f32>,
    pub k: AlignedBuf<f32>,
    pub v: AlignedBuf<f32>,
    pub attn_out: AlignedBuf<f32>,
    pub proj_out: AlignedBuf<f32>,
    pub gate: AlignedBuf<f32>,
    pub swiglu_tmp: AlignedBuf<f32>,
    pub xt: AlignedBuf<f32>,
    pub yt: AlignedBuf<f32>,
    pub kv_k: AlignedBuf<u16>,
    pub kv_v: AlignedBuf<u16>,
    pub cp_x: AlignedBuf<f32>,
    pub cp_x_norm: AlignedBuf<f32>,
    pub cp_q: AlignedBuf<f32>,
    pub cp_k: AlignedBuf<f32>,
    pub cp_v: AlignedBuf<f32>,
    pub cp_attn: AlignedBuf<f32>,
    pub cp_proj: AlignedBuf<f32>,
    pub cp_gate: AlignedBuf<f32>,
    pub cp_swiglu_tmp: AlignedBuf<f32>,
    pub cp_xt: AlignedBuf<f32>,
    pub cp_yt: AlignedBuf<f32>,
    pub cp_kv_k: AlignedBuf<u16>,
    pub cp_kv_v: AlignedBuf<u16>,
}

impl BatchWorkspace {
    pub fn new(b_dim: usize, kv_max: usize) -> Result<Self, EngineError> {
        if b_dim == 0 || b_dim > MAX_BATCH_SLOTS || kv_max == 0 {
            return Err(EngineError::InvalidArg(format!(
                "Invalid BatchWorkspace dimensions: B={b_dim}, kv_max={kv_max}"
            )));
        }
        let max_rows = (2 * INTERMEDIATE_SIZE).max(Q_DIM).max(HIDDEN_SIZE);
        let max_cols = HIDDEN_SIZE.max(Q_DIM).max(INTERMEDIATE_SIZE);
        let cp_max_rows = (2 * CP_INTERMEDIATE_SIZE)
            .max(CP_Q_DIM)
            .max(CP_HIDDEN_SIZE);
        let cp_max_cols = CP_HIDDEN_SIZE.max(CP_Q_DIM).max(CP_INTERMEDIATE_SIZE);
        let kv_total = b_dim * NUM_LAYERS * kv_max * KV_DIM;
        let cp_kv_total = b_dim * CP_NUM_LAYERS * CP_KV_MAX * CP_KV_DIM;

        Ok(Self {
            b_dim,
            kv_max,
            kv_len: 0,
            force_matvec: env_batch_force_matvec(),
            x: AlignedBuf::new_zeroed(b_dim * HIDDEN_SIZE),
            x_norm: AlignedBuf::new_zeroed(b_dim * HIDDEN_SIZE),
            q: AlignedBuf::new_zeroed(b_dim * Q_DIM),
            k: AlignedBuf::new_zeroed(b_dim * KV_DIM),
            v: AlignedBuf::new_zeroed(b_dim * KV_DIM),
            attn_out: AlignedBuf::new_zeroed(b_dim * Q_DIM),
            proj_out: AlignedBuf::new_zeroed(b_dim * HIDDEN_SIZE),
            gate: AlignedBuf::new_zeroed(b_dim * 2 * INTERMEDIATE_SIZE),
            swiglu_tmp: AlignedBuf::new_zeroed(INTERMEDIATE_SIZE),
            xt: AlignedBuf::new_zeroed(max_cols * b_dim),
            yt: AlignedBuf::new_zeroed(max_rows * b_dim),
            kv_k: AlignedBuf::new_zeroed(kv_total),
            kv_v: AlignedBuf::new_zeroed(kv_total),
            cp_x: AlignedBuf::new_zeroed(b_dim * CP_HIDDEN_SIZE),
            cp_x_norm: AlignedBuf::new_zeroed(b_dim * CP_HIDDEN_SIZE),
            cp_q: AlignedBuf::new_zeroed(b_dim * CP_Q_DIM),
            cp_k: AlignedBuf::new_zeroed(b_dim * CP_KV_DIM),
            cp_v: AlignedBuf::new_zeroed(b_dim * CP_KV_DIM),
            cp_attn: AlignedBuf::new_zeroed(b_dim * CP_Q_DIM),
            cp_proj: AlignedBuf::new_zeroed(b_dim * CP_HIDDEN_SIZE),
            cp_gate: AlignedBuf::new_zeroed(b_dim * 2 * CP_INTERMEDIATE_SIZE),
            cp_swiglu_tmp: AlignedBuf::new_zeroed(CP_INTERMEDIATE_SIZE),
            cp_xt: AlignedBuf::new_zeroed(cp_max_cols * b_dim),
            cp_yt: AlignedBuf::new_zeroed(cp_max_rows * b_dim),
            cp_kv_k: AlignedBuf::new_zeroed(cp_kv_total),
            cp_kv_v: AlignedBuf::new_zeroed(cp_kv_total),
        })
    }

    pub fn load_slot_kv(
        &mut self,
        slot: usize,
        src_k: &[u16],
        src_v: &[u16],
        src_kv_max: usize,
        prompt_len: usize,
    ) {
        for layer in 0..NUM_LAYERS {
            let dst_off = (slot * NUM_LAYERS + layer) * self.kv_max * KV_DIM;
            let src_off = layer * src_kv_max * KV_DIM;
            let len = prompt_len * KV_DIM;
            self.kv_k[dst_off..dst_off + len].copy_from_slice(&src_k[src_off..src_off + len]);
            self.kv_v[dst_off..dst_off + len].copy_from_slice(&src_v[src_off..src_off + len]);
        }
    }
}

fn batch_talker_step_impl(
    runtime: &ComputeRuntime,
    model: &Model06B,
    bw: &mut BatchWorkspace,
    embeds: &[f32],
    pos_arr: Option<&[usize]>,
    active: Option<&[bool]>,
    hidden_out: &mut [f32],
) -> Result<(), EngineError> {
    let b_dim = bw.b_dim;
    let eps = model.config.rms_norm_eps;
    let d = runtime.dispatch;
    let pool = &runtime.pool;
    let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();

    let mut max_pos = 0usize;
    for b in 0..b_dim {
        let p = pos_arr.map(|a| a[b]).unwrap_or(bw.kv_len);
        if p > max_pos {
            max_pos = p;
        }
    }
    if max_pos + 1 > bw.kv_max {
        return Err(EngineError::Runtime(format!(
            "Batch KV overflow: pos {} >= kv_max {}",
            max_pos, bw.kv_max
        )));
    }

    bw.x[..b_dim * HIDDEN_SIZE].copy_from_slice(&embeds[..b_dim * HIDDEN_SIZE]);

    for layer_idx in 0..NUM_LAYERS {
        let l = &model.talker.layers[layer_idx];

        for b in 0..b_dim {
            d.rms_norm(
                &mut bw.x_norm[b * HIDDEN_SIZE..(b + 1) * HIDDEN_SIZE],
                &bw.x[b * HIDDEN_SIZE..(b + 1) * HIDDEN_SIZE],
                &l.input_norm,
                1,
                HIDDEN_SIZE,
                eps,
            );
        }

        batch_proj_q(
            runtime,
            &mut bw.xt,
            &mut bw.yt,
            &mut bw.q,
            l.wq_q.as_ref(),
            &l.wq_bf16,
            &bw.x_norm,
            Q_DIM,
            HIDDEN_SIZE,
            HIDDEN_SIZE,
            b_dim,
            bw.force_matvec,
        );
        batch_proj_q(
            runtime,
            &mut bw.xt,
            &mut bw.yt,
            &mut bw.k,
            l.wk_q.as_ref(),
            &l.wk_bf16,
            &bw.x_norm,
            KV_DIM,
            HIDDEN_SIZE,
            HIDDEN_SIZE,
            b_dim,
            bw.force_matvec,
        );
        batch_proj_q(
            runtime,
            &mut bw.xt,
            &mut bw.yt,
            &mut bw.v,
            l.wv_q.as_ref(),
            &l.wv_bf16,
            &bw.x_norm,
            KV_DIM,
            HIDDEN_SIZE,
            HIDDEN_SIZE,
            b_dim,
            bw.force_matvec,
        );

        for b in 0..b_dim {
            if active.is_some_and(|a| !a[b]) {
                continue;
            }
            let pos = pos_arr.map(|a| a[b]).unwrap_or(bw.kv_len);
            let q_b = &mut bw.q[b * Q_DIM..(b + 1) * Q_DIM];
            let k_b = &mut bw.k[b * KV_DIM..(b + 1) * KV_DIM];
            let v_b = &bw.v[b * KV_DIM..(b + 1) * KV_DIM];
            d.rms_norm_per_head(q_b, &l.q_norm, 1, NUM_HEADS, HEAD_DIM, eps);
            d.rms_norm_per_head(k_b, &l.k_norm, 1, NUM_KV_HEADS, HEAD_DIM, eps);
            d.apply_rope_neox_inplace(
                q_b,
                NUM_HEADS,
                HEAD_DIM,
                &model.talker.rope_cos,
                &model.talker.rope_sin,
                pos,
            );
            d.apply_rope_neox_inplace(
                k_b,
                NUM_KV_HEADS,
                HEAD_DIM,
                &model.talker.rope_cos,
                &model.talker.rope_sin,
                pos,
            );
            let lbase = (b * NUM_LAYERS + layer_idx) * bw.kv_max * KV_DIM;
            let kv_off = lbase + pos * KV_DIM;
            d.f32_to_bf16_vec(&mut bw.kv_k[kv_off..kv_off + KV_DIM], k_b);
            d.f32_to_bf16_vec(&mut bw.kv_v[kv_off..kv_off + KV_DIM], v_b);

            let lk = &bw.kv_k[lbase..lbase + (pos + 1) * KV_DIM];
            let lv = &bw.kv_v[lbase..lbase + (pos + 1) * KV_DIM];
            d.causal_attention_bf16kv_heads_parallel(
                pool,
                &mut bw.attn_out[b * Q_DIM..(b + 1) * Q_DIM],
                q_b,
                lk,
                lv,
                1,
                pos + 1,
                NUM_HEADS,
                NUM_KV_HEADS,
                HEAD_DIM,
                scale,
                pos,
            );
        }

        batch_proj_q(
            runtime,
            &mut bw.xt,
            &mut bw.yt,
            &mut bw.proj_out,
            l.wo_q.as_ref(),
            &l.wo_bf16,
            &bw.attn_out,
            HIDDEN_SIZE,
            Q_DIM,
            Q_DIM,
            b_dim,
            bw.force_matvec,
        );

        for b in 0..b_dim {
            d.rms_norm_residual(
                &mut bw.x_norm[b * HIDDEN_SIZE..(b + 1) * HIDDEN_SIZE],
                &mut bw.x[b * HIDDEN_SIZE..(b + 1) * HIDDEN_SIZE],
                &bw.proj_out[b * HIDDEN_SIZE..(b + 1) * HIDDEN_SIZE],
                &l.post_attn_norm,
                HIDDEN_SIZE,
                eps,
            );
        }

        let gu_bf16 = l.gate_up_fused_bf16.as_deref().unwrap_or(&[]);
        batch_proj_q(
            runtime,
            &mut bw.xt,
            &mut bw.yt,
            &mut bw.gate,
            l.gate_up_q.as_ref(),
            gu_bf16,
            &bw.x_norm,
            2 * INTERMEDIATE_SIZE,
            HIDDEN_SIZE,
            HIDDEN_SIZE,
            b_dim,
            bw.force_matvec,
        );

        for b in 0..b_dim {
            d.swiglu_inplace(
                &mut bw.gate[b * 2 * INTERMEDIATE_SIZE..(b + 1) * 2 * INTERMEDIATE_SIZE],
                &mut bw.swiglu_tmp[..INTERMEDIATE_SIZE],
                INTERMEDIATE_SIZE,
            );
        }

        batch_proj_q(
            runtime,
            &mut bw.xt,
            &mut bw.yt,
            &mut bw.proj_out,
            l.down_q.as_ref(),
            &l.down_bf16,
            &bw.gate,
            HIDDEN_SIZE,
            INTERMEDIATE_SIZE,
            2 * INTERMEDIATE_SIZE,
            b_dim,
            bw.force_matvec,
        );

        for b in 0..b_dim {
            let xb = &mut bw.x[b * HIDDEN_SIZE..(b + 1) * HIDDEN_SIZE];
            let pb = &bw.proj_out[b * HIDDEN_SIZE..(b + 1) * HIDDEN_SIZE];
            for i in 0..HIDDEN_SIZE {
                xb[i] += pb[i];
            }
        }
    }

    for b in 0..b_dim {
        d.rms_norm(
            &mut hidden_out[b * HIDDEN_SIZE..(b + 1) * HIDDEN_SIZE],
            &bw.x[b * HIDDEN_SIZE..(b + 1) * HIDDEN_SIZE],
            &model.talker.norm,
            1,
            HIDDEN_SIZE,
            eps,
        );
    }

    if pos_arr.is_none() {
        bw.kv_len += 1;
    }
    Ok(())
}

pub fn batch_talker_step(
    runtime: &ComputeRuntime,
    model: &Model06B,
    bw: &mut BatchWorkspace,
    embeds: &[f32],
    hidden_out: &mut [f32],
) -> Result<(), EngineError> {
    batch_talker_step_impl(runtime, model, bw, embeds, None, None, hidden_out)
}

pub fn batch_talker_step_ragged(
    runtime: &ComputeRuntime,
    model: &Model06B,
    bw: &mut BatchWorkspace,
    embeds: &[f32],
    pos_arr: &[usize],
    active: &[bool],
    hidden_out: &mut [f32],
) -> Result<(), EngineError> {
    batch_talker_step_impl(
        runtime,
        model,
        bw,
        embeds,
        Some(pos_arr),
        Some(active),
        hidden_out,
    )
}

fn batch_cp_transformer_step(
    runtime: &ComputeRuntime,
    model: &Model06B,
    bw: &mut BatchWorkspace,
    pos: usize,
    active: Option<&[bool]>,
) {
    let b_dim = bw.b_dim;
    let eps = model.config.rms_norm_eps;
    let d = runtime.dispatch;
    let attn_scale = 1.0f32 / (CP_HEAD_DIM as f32).sqrt();

    for layer_idx in 0..CP_NUM_LAYERS {
        let l = &model.cp.layers[layer_idx];

        for b in 0..b_dim {
            d.rms_norm(
                &mut bw.cp_x_norm[b * CP_HIDDEN_SIZE..(b + 1) * CP_HIDDEN_SIZE],
                &bw.cp_x[b * CP_HIDDEN_SIZE..(b + 1) * CP_HIDDEN_SIZE],
                &l.input_norm,
                1,
                CP_HIDDEN_SIZE,
                eps,
            );
        }

        batch_proj_q(
            runtime,
            &mut bw.cp_xt,
            &mut bw.cp_yt,
            &mut bw.cp_q,
            l.wq_q.as_ref(),
            &l.wq_bf16,
            &bw.cp_x_norm,
            CP_Q_DIM,
            CP_HIDDEN_SIZE,
            CP_HIDDEN_SIZE,
            b_dim,
            bw.force_matvec,
        );
        batch_proj_q(
            runtime,
            &mut bw.cp_xt,
            &mut bw.cp_yt,
            &mut bw.cp_k,
            l.wk_q.as_ref(),
            &l.wk_bf16,
            &bw.cp_x_norm,
            CP_KV_DIM,
            CP_HIDDEN_SIZE,
            CP_HIDDEN_SIZE,
            b_dim,
            bw.force_matvec,
        );
        batch_proj_q(
            runtime,
            &mut bw.cp_xt,
            &mut bw.cp_yt,
            &mut bw.cp_v,
            l.wv_q.as_ref(),
            &l.wv_bf16,
            &bw.cp_x_norm,
            CP_KV_DIM,
            CP_HIDDEN_SIZE,
            CP_HIDDEN_SIZE,
            b_dim,
            bw.force_matvec,
        );

        for b in 0..b_dim {
            if active.is_some_and(|a| !a[b]) {
                continue;
            }
            let q_b = &mut bw.cp_q[b * CP_Q_DIM..(b + 1) * CP_Q_DIM];
            let k_b = &mut bw.cp_k[b * CP_KV_DIM..(b + 1) * CP_KV_DIM];
            let v_b = &bw.cp_v[b * CP_KV_DIM..(b + 1) * CP_KV_DIM];
            d.rms_norm_per_head(q_b, &l.q_norm, 1, CP_NUM_HEADS, CP_HEAD_DIM, eps);
            d.rms_norm_per_head(k_b, &l.k_norm, 1, CP_NUM_KV_HEADS, CP_HEAD_DIM, eps);
            d.apply_rope_neox_inplace(
                q_b,
                CP_NUM_HEADS,
                CP_HEAD_DIM,
                &model.cp.rope_cos,
                &model.cp.rope_sin,
                pos,
            );
            d.apply_rope_neox_inplace(
                k_b,
                CP_NUM_KV_HEADS,
                CP_HEAD_DIM,
                &model.cp.rope_cos,
                &model.cp.rope_sin,
                pos,
            );
            let lbase = (b * CP_NUM_LAYERS + layer_idx) * CP_KV_MAX * CP_KV_DIM;
            let kv_off = lbase + pos * CP_KV_DIM;
            d.f32_to_bf16_vec(&mut bw.cp_kv_k[kv_off..kv_off + CP_KV_DIM], k_b);
            d.f32_to_bf16_vec(&mut bw.cp_kv_v[kv_off..kv_off + CP_KV_DIM], v_b);

            let lk = &bw.cp_kv_k[lbase..lbase + (pos + 1) * CP_KV_DIM];
            let lv = &bw.cp_kv_v[lbase..lbase + (pos + 1) * CP_KV_DIM];
            d.causal_attention_bf16kv(
                &mut bw.cp_attn[b * CP_Q_DIM..(b + 1) * CP_Q_DIM],
                q_b,
                lk,
                lv,
                1,
                pos + 1,
                CP_NUM_HEADS,
                CP_NUM_KV_HEADS,
                CP_HEAD_DIM,
                attn_scale,
                pos,
            );
        }

        batch_proj_q(
            runtime,
            &mut bw.cp_xt,
            &mut bw.cp_yt,
            &mut bw.cp_proj,
            l.wo_q.as_ref(),
            &l.wo_bf16,
            &bw.cp_attn,
            CP_HIDDEN_SIZE,
            CP_Q_DIM,
            CP_Q_DIM,
            b_dim,
            bw.force_matvec,
        );

        for b in 0..b_dim {
            d.rms_norm_residual(
                &mut bw.cp_x_norm[b * CP_HIDDEN_SIZE..(b + 1) * CP_HIDDEN_SIZE],
                &mut bw.cp_x[b * CP_HIDDEN_SIZE..(b + 1) * CP_HIDDEN_SIZE],
                &bw.cp_proj[b * CP_HIDDEN_SIZE..(b + 1) * CP_HIDDEN_SIZE],
                &l.post_attn_norm,
                CP_HIDDEN_SIZE,
                eps,
            );
        }

        let gu_bf16 = l.gate_up_fused_bf16.as_deref().unwrap_or(&[]);
        batch_proj_q(
            runtime,
            &mut bw.cp_xt,
            &mut bw.cp_yt,
            &mut bw.cp_gate,
            l.gate_up_q.as_ref(),
            gu_bf16,
            &bw.cp_x_norm,
            2 * CP_INTERMEDIATE_SIZE,
            CP_HIDDEN_SIZE,
            CP_HIDDEN_SIZE,
            b_dim,
            bw.force_matvec,
        );

        for b in 0..b_dim {
            d.swiglu_inplace(
                &mut bw.cp_gate[b * 2 * CP_INTERMEDIATE_SIZE..(b + 1) * 2 * CP_INTERMEDIATE_SIZE],
                &mut bw.cp_swiglu_tmp[..CP_INTERMEDIATE_SIZE],
                CP_INTERMEDIATE_SIZE,
            );
        }

        batch_proj_q(
            runtime,
            &mut bw.cp_xt,
            &mut bw.cp_yt,
            &mut bw.cp_proj,
            l.down_q.as_ref(),
            &l.down_bf16,
            &bw.cp_gate,
            CP_HIDDEN_SIZE,
            CP_INTERMEDIATE_SIZE,
            2 * CP_INTERMEDIATE_SIZE,
            b_dim,
            bw.force_matvec,
        );

        for b in 0..b_dim {
            let xb = &mut bw.cp_x[b * CP_HIDDEN_SIZE..(b + 1) * CP_HIDDEN_SIZE];
            let pb = &bw.cp_proj[b * CP_HIDDEN_SIZE..(b + 1) * CP_HIDDEN_SIZE];
            for i in 0..CP_HIDDEN_SIZE {
                xb[i] += pb[i];
            }
        }
    }
}

pub fn batch_cp_predict(
    runtime: &ComputeRuntime,
    model: &Model06B,
    voice: &VoiceBindings,
    bw: &mut BatchWorkspace,
    talker_hidden: &[f32],
    code0: &[i32],
    out_codes: &mut [i32],
    active: Option<&[bool]>,
) {
    let b_dim = bw.b_dim;
    let eps = model.config.rms_norm_eps;
    let d = runtime.dispatch;
    let mut normed = [0.0f32; CP_HIDDEN_SIZE];

    // Step 0: talker_hidden -> CP (0.6B: identity projection, CP_HIDDEN_SIZE == HIDDEN_SIZE)
    for b in 0..b_dim {
        let dst = &mut bw.cp_x[b * CP_HIDDEN_SIZE..(b + 1) * CP_HIDDEN_SIZE];
        if active.is_some_and(|a| !a[b]) {
            dst.fill(0.0);
        } else {
            dst.copy_from_slice(&talker_hidden[b * HIDDEN_SIZE..(b + 1) * HIDDEN_SIZE]);
        }
    }
    batch_cp_transformer_step(runtime, model, bw, 0, active);

    // Step 1: embed code0 via Talker codec embedding
    for b in 0..b_dim {
        if active.is_some_and(|a| !a[b]) {
            continue;
        }
        let dst = &mut bw.cp_x[b * CP_HIDDEN_SIZE..(b + 1) * CP_HIDDEN_SIZE];
        lookup_codec_embed(d, model, voice, code0[b], dst);
    }
    batch_cp_transformer_step(runtime, model, bw, 1, active);

    // Codebook 0
    for b in 0..b_dim {
        if active.is_some_and(|a| !a[b]) {
            out_codes[b * 15] = 0;
            continue;
        }
        d.rms_norm(
            &mut normed,
            &bw.cp_x[b * CP_HIDDEN_SIZE..(b + 1) * CP_HIDDEN_SIZE],
            &model.cp.norm,
            1,
            CP_HIDDEN_SIZE,
            eps,
        );
        out_codes[b * 15] = cp_lm_argmax(runtime, &model.cp.lm_heads[0], &normed) as i32;
    }

    // Codebooks 1..15
    for g in 1..15 {
        let pos = g + 1;
        for b in 0..b_dim {
            if active.is_some_and(|a| !a[b]) {
                continue;
            }
            let prev = out_codes[b * 15 + (g - 1)];
            let dst = &mut bw.cp_x[b * CP_HIDDEN_SIZE..(b + 1) * CP_HIDDEN_SIZE];
            if prev >= 0 && (prev as usize) < CODEBOOK_SIZE {
                let off = (prev as usize) * CP_HIDDEN_SIZE;
                d.bf16_to_f32_vec(
                    dst,
                    &model.cp.codec_emb_bf16[g - 1][off..off + CP_HIDDEN_SIZE],
                );
            } else {
                dst.fill(0.0);
            }
        }
        batch_cp_transformer_step(runtime, model, bw, pos, active);
        for b in 0..b_dim {
            if active.is_some_and(|a| !a[b]) {
                out_codes[b * 15 + g] = 0;
                continue;
            }
            d.rms_norm(
                &mut normed,
                &bw.cp_x[b * CP_HIDDEN_SIZE..(b + 1) * CP_HIDDEN_SIZE],
                &model.cp.norm,
                1,
                CP_HIDDEN_SIZE,
                eps,
            );
            out_codes[b * 15 + g] = cp_lm_argmax(runtime, &model.cp.lm_heads[g], &normed) as i32;
        }
    }
}

/// Verifies batched Talker step + CP wiring vs single-stream (`--batch-test`).
pub fn batch_self_test(runtime: &ComputeRuntime, model: &Model06B) -> Result<bool, EngineError> {
    let b_dim = std::env::var("QWEN_BATCH_B")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&v| (1..=MAX_BATCH_SLOTS).contains(&v))
        .unwrap_or(8);
    let k_steps = 8usize;
    let kv_max = 64usize;

    let mut bw = BatchWorkspace::new(b_dim, kv_max)?;
    let mut scratch = SessionScratch::new();
    let voice = VoiceBindings::preset(3061);
    let mut seed = 0xC0FFEE42u32;

    let mut embeds_all = vec![0.0f32; k_steps * HIDDEN_SIZE];
    for v in embeds_all.iter_mut() {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *v = ((seed >> 8) as f32 / 16_777_216.0 - 0.5) * 0.2;
    }

    let mut href = vec![0.0f32; k_steps * HIDDEN_SIZE];
    for k in 0..k_steps {
        talker_step(
            runtime,
            model,
            &mut scratch,
            &embeds_all[k * HIDDEN_SIZE..(k + 1) * HIDDEN_SIZE],
            &mut href[k * HIDDEN_SIZE..(k + 1) * HIDDEN_SIZE],
        )?;
    }

    // 1. Force matvec wiring check (bit-exact vs single-stream)
    bw.force_matvec = true;
    bw.kv_len = 0;
    let mut embeds_b = vec![0.0f32; b_dim * HIDDEN_SIZE];
    let mut hbat = vec![0.0f32; b_dim * HIDDEN_SIZE];
    let mut max_wiring_diff = 0.0f32;
    for k in 0..k_steps {
        for b in 0..b_dim {
            embeds_b[b * HIDDEN_SIZE..(b + 1) * HIDDEN_SIZE]
                .copy_from_slice(&embeds_all[k * HIDDEN_SIZE..(k + 1) * HIDDEN_SIZE]);
        }
        batch_talker_step(runtime, model, &mut bw, &embeds_b, &mut hbat)?;
        for b in 0..b_dim {
            for i in 0..HIDDEN_SIZE {
                let d = (hbat[b * HIDDEN_SIZE + i] - href[k * HIDDEN_SIZE + i]).abs();
                if d > max_wiring_diff {
                    max_wiring_diff = d;
                }
            }
        }
    }

    // 2. Matmat path check
    bw.force_matvec = false;
    bw.kv_len = 0;
    let mut max_mm_diff = 0.0f32;
    for k in 0..k_steps {
        for b in 0..b_dim {
            embeds_b[b * HIDDEN_SIZE..(b + 1) * HIDDEN_SIZE]
                .copy_from_slice(&embeds_all[k * HIDDEN_SIZE..(k + 1) * HIDDEN_SIZE]);
        }
        batch_talker_step(runtime, model, &mut bw, &embeds_b, &mut hbat)?;
        for b in 0..b_dim {
            for i in 0..HIDDEN_SIZE {
                let d = (hbat[b * HIDDEN_SIZE + i] - href[k * HIDDEN_SIZE + i]).abs();
                if d > max_mm_diff {
                    max_mm_diff = d;
                }
            }
        }
    }

    // 3. CP batch probe (single-stream reference vs batched wiring + matmat)
    let mut th = vec![0.0f32; b_dim * HIDDEN_SIZE];
    for v in th.iter_mut() {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *v = ((seed >> 8) as f32 / 16_777_216.0 - 0.5) * 1.0;
    }
    let mut c0 = vec![0i32; b_dim];
    for c in c0.iter_mut() {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *c = ((seed >> 16) as i32) % (CODEC_VOCAB_SIZE as i32 - 1);
    }
    let mut cp_ref = vec![0i32; b_dim * 15];
    for b in 0..b_dim {
        crate::code_predictor::cp_predict(
            runtime,
            model,
            &voice,
            &mut scratch,
            &th[b * HIDDEN_SIZE..(b + 1) * HIDDEN_SIZE],
            c0[b],
            0.0,
            (&mut cp_ref[b * 15..(b + 1) * 15]).try_into().unwrap(),
        );
    }
    bw.force_matvec = true;
    let mut cp_mv = vec![0i32; b_dim * 15];
    batch_cp_predict(runtime, model, &voice, &mut bw, &th, &c0, &mut cp_mv, None);
    let diff_mv = cp_mv
        .iter()
        .zip(cp_ref.iter())
        .filter(|(a, b)| a != b)
        .count();

    bw.force_matvec = false;
    let mut cp_mm = vec![0i32; b_dim * 15];
    batch_cp_predict(runtime, model, &voice, &mut bw, &th, &c0, &mut cp_mm, None);
    let diff_mm = cp_mm
        .iter()
        .zip(cp_ref.iter())
        .filter(|(a, b)| a != b)
        .count();

    eprintln!(
        "batch-test (B={b_dim}, K={k_steps}): Talker wiring max_diff={max_wiring_diff:.3e} {}, matmat max_diff={max_mm_diff:.3e} (fp-order)",
        if max_wiring_diff < 1e-5 { "PASS" } else { "FAIL" }
    );
    eprintln!(
        "  CP wiring vs single-stream: {diff_mv}/{} differ {}, matmat: {diff_mm}/{} differ",
        b_dim * 15,
        if diff_mv == 0 { "PASS" } else { "FAIL" },
        b_dim * 15
    );
    let ok = max_wiring_diff < 1e-5 && diff_mv == 0;
    Ok(ok)
}

/// Measures batched-compute throughput vs sequential single-stream (`--batch-bench`).
pub fn batch_bench(runtime: &ComputeRuntime, model: &Model06B) -> Result<(), EngineError> {
    let b_dim = std::env::var("QWEN_BATCH_B")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&v| (1..=MAX_BATCH_SLOTS).contains(&v))
        .unwrap_or(8);
    let k_steps = 32usize;
    let mut bw = BatchWorkspace::new(b_dim, k_steps + 4)?;
    let mut scratch = SessionScratch::new();
    let voice = VoiceBindings::preset(3061);
    let emb_b = vec![0.01f32; b_dim * HIDDEN_SIZE];
    let mut hid = vec![0.0f32; b_dim * HIDDEN_SIZE];
    let c0 = vec![100i32; b_dim];
    let mut codes = vec![0i32; b_dim * 15];

    let t0 = Instant::now();
    for _ in 0..k_steps {
        batch_talker_step(runtime, model, &mut bw, &emb_b, &mut hid)?;
        batch_cp_predict(runtime, model, &voice, &mut bw, &hid, &c0, &mut codes, None);
    }
    let bat_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = Instant::now();
    let mut one_h = [0.0f32; HIDDEN_SIZE];
    let mut cp_one = [0i32; 15];
    for _ in 0..b_dim {
        scratch.talker_kv.len = 0;
        for _ in 0..k_steps {
            talker_step(
                runtime,
                model,
                &mut scratch,
                &emb_b[..HIDDEN_SIZE],
                &mut one_h,
            )?;
            crate::code_predictor::cp_predict(
                runtime,
                model,
                &voice,
                &mut scratch,
                &one_h,
                100,
                0.0,
                &mut cp_one,
            );
        }
    }
    let seq_ms = t1.elapsed().as_secs_f64() * 1000.0;
    let speedup = seq_ms / bat_ms.max(1e-6);
    println!(
        "batch-bench (B={b_dim}, K={k_steps}, threads={}): seq={seq_ms:.1} ms, batched={bat_ms:.1} ms, speedup={speedup:.2}x",
        runtime.pool.n_threads()
    );
    Ok(())
}

// ── Sentence-aware text splitter (--batch) ─────────────────────────────────

fn word_count(s: &str) -> usize {
    s.split_whitespace().count()
}

fn is_abbrev_dot(bytes: &[u8], dot: usize) -> bool {
    let mut s = dot;
    while s > 0 && !matches!(bytes[s - 1], b' ' | b'\n' | b'\t') {
        s -= 1;
    }
    let tl = dot - s;
    if tl == 0 || tl > 4 {
        return false;
    }
    let Ok(tok) = std::str::from_utf8(&bytes[s..dot]) else {
        return false;
    };
    const ABBREVS: &[&str] = &[
        "Sig", "Sigg", "Dott", "Dr", "Prof", "Egr", "On", "Rev", "St", "vs", "pag", "art", "ecc",
        "etc", "es", "cfr",
    ];
    if ABBREVS.iter().any(|a| a.eq_ignore_ascii_case(tok)) {
        return true;
    }
    tl == 1 && bytes[s].is_ascii_uppercase()
}

pub fn split_sentences(text: &str) -> Vec<String> {
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;
    while i < len {
        let ch = bytes[i];
        let mut is_end = matches!(ch, b'.' | b'!' | b'?' | b';' | b'\n');
        if ch == b'.' {
            if i > 0
                && i + 1 < len
                && bytes[i - 1].is_ascii_digit()
                && bytes[i + 1].is_ascii_digit()
            {
                is_end = false;
            }
            if is_abbrev_dot(bytes, i) {
                is_end = false;
            }
        }
        if is_end {
            let mut e = i + 1;
            while e < len && matches!(bytes[e], b'"' | b'\'' | b')' | b']') {
                e += 1;
            }
            let part = text[start..e].trim();
            if !part.is_empty() {
                out.push(part.to_string());
            }
            start = e;
            i = e;
            continue;
        }
        i += 1;
    }
    if start < len {
        let part = text[start..].trim();
        if !part.is_empty() {
            out.push(part.to_string());
        }
    }
    out
}

pub fn split_text_for_batch(text: &str, target_words: usize, min_words: usize) -> Vec<String> {
    let sents = split_sentences(text);
    let mut chunks: Vec<String> = Vec::new();
    let mut acc = String::new();
    let mut acc_words = 0usize;

    for sent in sents {
        let sw = word_count(&sent);
        if !acc.is_empty() {
            acc.push(' ');
        }
        acc.push_str(&sent);
        acc_words += sw;
        if acc_words >= target_words {
            let trimmed = acc.trim();
            if !trimmed.is_empty() {
                chunks.push(trimmed.to_string());
            }
            acc.clear();
            acc_words = 0;
        }
    }
    let tail = acc.trim();
    if !tail.is_empty() {
        if !chunks.is_empty() && acc_words < min_words {
            if let Some(prev) = chunks.last_mut() {
                prev.push(' ');
                prev.push_str(tail);
            }
        } else {
            chunks.push(tail.to_string());
        }
    }
    chunks
}

#[derive(Debug, Clone)]
pub struct BatchReq {
    pub text: String,
    pub speaker_id: i32,
    pub language_id: Option<i32>,
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub rep_penalty: f32,
    pub greedy_warmup: usize,
    pub seed: u32,
    pub want_stream: bool,
    pub chunk_frames: usize,
    pub volume: f32,
}

pub fn generate_batch_multi(
    runtime: &ComputeRuntime,
    model: &Model06B,
    base_voice: &VoiceBindings,
    reqs: &[BatchReq],
    force_matvec: bool,
) -> Result<Vec<Vec<f32>>, EngineError> {
    let nc = reqs.len();
    if nc == 0 {
        return Ok(Vec::new());
    }
    const GMAX: usize = 8;
    let mut out_samples: Vec<Vec<f32>> = vec![Vec::new(); nc];
    let mut scratch = SessionScratch::new();
    let d = runtime.dispatch;
    let eps = model.config.rms_norm_eps;
    let gen_cap = 600usize;

    let nt = runtime.pool.n_threads();
    for g0 in (0..nc).step_by(GMAX) {
        let b_dim = (nc - g0).min(GMAX);
        let mut prompt_lens = vec![0usize; b_dim];
        let mut tcls = vec![0usize; b_dim];
        let mut seed_hidden = vec![0.0f32; b_dim * HIDDEN_SIZE];
        let mut kv_snapshots: Vec<(Vec<u16>, Vec<u16>, usize)> = Vec::with_capacity(b_dim);
        let mut max_pl = 0usize;

        runtime.blas.set_threads(nt);
        for b in 0..b_dim {
            let r = &reqs[g0 + b];
            let preset_override;
            let voice_ref: &VoiceBindings =
                if matches!(base_voice.identity, crate::model::VoiceIdentity::Preset(_)) {
                    preset_override = VoiceBindings {
                        identity: crate::model::VoiceIdentity::Preset(r.speaker_id),
                        ..base_voice.clone()
                    };
                    &preset_override
                } else {
                    base_voice
                };
            let rc = RequestConfig {
                language_id: r.language_id,
                temperature: r.temperature,
                top_k: r.top_k,
                top_p: r.top_p,
                rep_penalty: r.rep_penalty,
                greedy_warmup: r.greedy_warmup,
                seed: r.seed,
                silent: true,
                ..Default::default()
            };
            let (embeds, pl, tcl, _) =
                build_prefill_embeds(runtime, model, voice_ref, &mut scratch, &rc, &r.text)?;
            scratch.talker_kv.len = 0;
            crate::talker::talker_prefill(
                runtime,
                model,
                &mut scratch,
                &embeds,
                pl,
                true,
                false,
            )?;
            d.rms_norm(
                &mut seed_hidden[b * HIDDEN_SIZE..(b + 1) * HIDDEN_SIZE],
                &scratch.talker_bufs.dec_x,
                &model.talker.norm,
                1,
                HIDDEN_SIZE,
                eps,
            );
            prompt_lens[b] = pl;
            tcls[b] = tcl;
            max_pl = max_pl.max(pl);
            kv_snapshots.push((
                scratch.talker_kv.k.as_slice().to_vec(),
                scratch.talker_kv.v.as_slice().to_vec(),
                scratch.talker_kv.max_pos,
            ));
        }

        let kv_max = (max_pl + gen_cap + 4).min(ROPE_MAX_POS);
        let mut bw = BatchWorkspace::new(b_dim, kv_max)?;
        bw.force_matvec = force_matvec || env_batch_force_matvec();
        for (b, (sk, sv, smax)) in kv_snapshots.iter().enumerate() {
            bw.load_slot_kv(b, sk, sv, *smax, prompt_lens[b]);
        }

        let mut pos = prompt_lens;
        let mut active = vec![true; b_dim];
        let mut rngs: Vec<LcgRng> = reqs[g0..g0 + b_dim]
            .iter()
            .map(|r| LcgRng::new(r.seed))
            .collect();
        let mut prev_toks: Vec<Vec<i32>> = vec![Vec::new(); b_dim];
        let mut ch_codes: Vec<Vec<i32>> = vec![Vec::new(); b_dim];
        let mut last_hidden = seed_hidden;
        let mut logits = vec![0.0f32; b_dim * CODEC_VOCAB_SIZE];
        let mut step_embed = vec![0.0f32; b_dim * HIDDEN_SIZE];
        let mut code0 = vec![0i32; b_dim];
        let mut cpcodes = vec![0i32; b_dim * 15];
        let mut sampler = Sampler::new();
        let tts_pad = if let Some(wovr) = &base_voice.wovr {
            wovr.tts_pad_embed.clone()
        } else {
            model.cached_tts_pad_embed.clone()
        };
        let mut n_active = b_dim;

        let gen_blas = if nt > 1 { nt - 1 } else { 1 };
        runtime.blas.set_threads(gen_blas);

        for frame in 0..gen_cap {
            if n_active == 0 {
                break;
            }
            batch_proj_q(
                runtime,
                &mut bw.xt,
                &mut bw.yt,
                &mut logits,
                None,
                &model.talker.codec_head_bf16,
                &last_hidden,
                CODEC_VOCAB_SIZE,
                HIDDEN_SIZE,
                HIDDEN_SIZE,
                b_dim,
                bw.force_matvec,
            );
            for b in 0..b_dim {
                if !active[b] {
                    code0[b] = 0;
                    continue;
                }
                let lg = &mut logits[b * CODEC_VOCAB_SIZE..(b + 1) * CODEC_VOCAB_SIZE];
                for v in lg.iter_mut() {
                    *v = v.clamp(-100.0, 100.0);
                }
                for t in 2048..CODEC_VOCAB_SIZE {
                    if t as i32 != CODEC_EOS {
                        lg[t] = -1e30;
                    }
                }
                if frame < 2 {
                    lg[CODEC_EOS as usize] = -1e30;
                }
                let ef = tcls[b] * 3;
                let bs = ef * 2;
                if ef > 0 && frame > bs {
                    let bo = (0.5f32 * ((frame - bs) as f32)).min(10.0);
                    lg[CODEC_EOS as usize] += bo;
                }
                let r = &reqs[g0 + b];
                let (ft, ftk) = if r.greedy_warmup > 0 && frame < r.greedy_warmup {
                    (0.0, 1)
                } else {
                    (r.temperature, r.top_k)
                };
                let c0 = sampler.sample(
                    lg,
                    ft,
                    ftk,
                    r.top_p,
                    r.rep_penalty,
                    &prev_toks[b],
                    &mut rngs[b],
                ) as i32;
                if c0 == CODEC_EOS || pos[b] >= kv_max - 1 {
                    active[b] = false;
                    n_active -= 1;
                    code0[b] = 0;
                    continue;
                }
                code0[b] = c0;
                prev_toks[b].push(c0);
            }
            if n_active == 0 {
                break;
            }

            batch_cp_predict(
                runtime,
                model,
                base_voice,
                &mut bw,
                &last_hidden,
                &code0,
                &mut cpcodes,
                Some(&active),
            );

            for b in 0..b_dim {
                let se = &mut step_embed[b * HIDDEN_SIZE..(b + 1) * HIDDEN_SIZE];
                if !active[b] {
                    se.fill(0.0);
                    continue;
                }
                ch_codes[b].push(code0[b]);
                ch_codes[b].extend_from_slice(&cpcodes[b * 15..(b + 1) * 15]);
                lookup_codec_embed(d, model, base_voice, code0[b], se);
                for g in 0..15 {
                    let cg = cpcodes[b * 15 + g];
                    if cg >= 0 && (cg as usize) < CODEBOOK_SIZE {
                        let off = (cg as usize) * HIDDEN_SIZE;
                        d.bf16_accum_f32(se, &model.cp.codec_emb_bf16[g][off..off + HIDDEN_SIZE]);
                    }
                }
                for j in 0..HIDDEN_SIZE {
                    se[j] += tts_pad[j];
                }
            }

            let mut next_hidden = vec![0.0f32; b_dim * HIDDEN_SIZE];
            batch_talker_step_ragged(
                runtime,
                model,
                &mut bw,
                &step_embed,
                &pos,
                &active,
                &mut next_hidden,
            )?;
            last_hidden = next_hidden;
            for b in 0..b_dim {
                if active[b] {
                    pos[b] += 1;
                }
            }
        }

        runtime.blas.set_threads(nt);
        for b in 0..b_dim {
            let nf = ch_codes[b].len() / NUM_CODEBOOKS;
            if nf > 0 {
                out_samples[g0 + b] =
                    speech_decoder_decode(runtime, &model.decoder, &ch_codes[b], nf)?;
            }
        }
    }

    Ok(out_samples)
}

pub fn generate_batch_longform(
    runtime: &ComputeRuntime,
    model: &Model06B,
    voice: &VoiceBindings,
    chunks: &[String],
    chunk_pause_s: f32,
    cfg: &RequestConfig,
    speaker_id: i32,
) -> Result<Vec<f32>, EngineError> {
    let reqs: Vec<BatchReq> = chunks
        .iter()
        .map(|text| BatchReq {
            text: text.clone(),
            speaker_id,
            language_id: cfg.language_id,
            temperature: cfg.temperature,
            top_k: cfg.top_k,
            top_p: cfg.top_p,
            rep_penalty: cfg.rep_penalty,
            greedy_warmup: cfg.greedy_warmup,
            seed: cfg.seed,
            want_stream: false,
            chunk_frames: cfg.stream_chunk_frames,
            volume: cfg.volume,
        })
        .collect();
    let bufs = generate_batch_multi(runtime, model, voice, &reqs, false)?;
    let pause_n = if chunk_pause_s > 0.0 {
        (chunk_pause_s * SAMPLE_RATE as f32) as usize
    } else {
        0
    };
    let mut out = Vec::new();
    for (idx, seg) in bufs.into_iter().enumerate() {
        if idx > 0 && pause_n > 0 {
            out.resize(out.len() + pause_n, 0.0);
        }
        out.extend_from_slice(&seg);
    }
    Ok(out)
}

// ── Continuous Batching Driver (vLLM-style, --batch-size N) ──────────────────

pub struct ContinuousSlotJob<T> {
    pub tag: T,
    pub req: BatchReq,
    pub abort: crate::model::AbortHandle,
}

pub trait ContinuousBatchSink<T> {
    fn is_running(&self) -> bool;
    fn next_job(&mut self, block: bool) -> Option<ContinuousSlotJob<T>>;
    /// Return false if client disconnected / stream write failed -> slot aborts.
    fn on_chunk(&mut self, tag: &mut T, samples: &[f32]) -> bool;
    fn on_done(&mut self, tag: T, samples: Option<Vec<f32>>, err: Option<EngineError>);
}

struct ContinuousSlot<T> {
    active: bool,
    tag: Option<T>,
    abort: crate::model::AbortHandle,
    req: Option<BatchReq>,
    pos: usize,
    tcl: usize,
    sframe: usize,
    rng: LcgRng,
    prev_tok: Vec<i32>,
    codes: Vec<i32>,
    sd_stream: crate::model::SdStreamState,
    decoded_frames: usize,
    ramp_chunks: usize,
    first_chunk_emitted: bool,
}

impl<T> ContinuousSlot<T> {
    fn empty() -> Self {
        Self {
            active: false,
            tag: None,
            abort: crate::model::AbortHandle::new(),
            req: None,
            pos: 0,
            tcl: 0,
            sframe: 0,
            rng: LcgRng::new(42),
            prev_tok: Vec::new(),
            codes: Vec::new(),
            sd_stream: crate::model::SdStreamState::new(),
            decoded_frames: 0,
            ramp_chunks: 0,
            first_chunk_emitted: false,
        }
    }
}

pub fn run_continuous_batch<T>(
    runtime: &ComputeRuntime,
    model: &Model06B,
    base_voice: &VoiceBindings,
    max_batch: usize,
    max_tokens: usize,
    sink: &mut dyn ContinuousBatchSink<T>,
) -> Result<(), EngineError> {
    let b_dim = max_batch.clamp(1, MAX_BATCH_SLOTS);
    let gen_cap = max_tokens.clamp(32, 600);
    const MAX_PROMPT: usize = 512;
    let kv_max = (MAX_PROMPT + gen_cap + 4).min(ROPE_MAX_POS);

    let mut bw = BatchWorkspace::new(b_dim, kv_max)?;
    let mut prefill_scratch = SessionScratch::new();
    let mut sd_ws = crate::model::DecoderWorkspace::new();
    let mut slots: Vec<ContinuousSlot<T>> = (0..b_dim).map(|_| ContinuousSlot::empty()).collect();

    let mut last_hidden = vec![0.0f32; b_dim * HIDDEN_SIZE];
    let mut logits = vec![0.0f32; b_dim * CODEC_VOCAB_SIZE];
    let mut step_embed = vec![0.0f32; b_dim * HIDDEN_SIZE];
    let mut code0 = vec![0i32; b_dim];
    let mut cpcodes = vec![0i32; b_dim * 15];
    let mut pos_arr = vec![0usize; b_dim];
    let mut active_arr = vec![false; b_dim];
    let mut sampler = Sampler::new();

    let d = runtime.dispatch;
    let eps = model.config.rms_norm_eps;
    let nt = runtime.pool.n_threads();
    let gen_blas = if nt > 1 { nt - 1 } else { 1 };
    let tts_pad = if let Some(wovr) = &base_voice.wovr {
        wovr.tts_pad_embed.clone()
    } else {
        model.cached_tts_pad_embed.clone()
    };

    let mut n_active = 0usize;

    while sink.is_running() || n_active > 0 {
        // 1. Admit queued jobs into free slots
        for b in 0..b_dim {
            if slots[b].active {
                continue;
            }
            if !sink.is_running() {
                break;
            }
            let block = n_active == 0;
            let Some(job) = sink.next_job(block) else {
                if block {
                    break;
                }
                continue;
            };
            if job.abort.is_cancelled() {
                sink.on_done(job.tag, None, Some(EngineError::Aborted));
                continue;
            }

            runtime.blas.set_threads(nt);
            let preset_override;
            let voice_ref: &VoiceBindings =
                if matches!(base_voice.identity, crate::model::VoiceIdentity::Preset(_)) {
                    preset_override = VoiceBindings {
                        identity: crate::model::VoiceIdentity::Preset(job.req.speaker_id),
                        ..base_voice.clone()
                    };
                    &preset_override
                } else {
                    base_voice
                };
            let rc = RequestConfig {
                language_id: job.req.language_id,
                temperature: job.req.temperature,
                top_k: job.req.top_k,
                top_p: job.req.top_p,
                rep_penalty: job.req.rep_penalty,
                greedy_warmup: job.req.greedy_warmup,
                seed: job.req.seed,
                silent: true,
                ..Default::default()
            };
            let prep =
                build_prefill_embeds(runtime, model, voice_ref, &mut prefill_scratch, &rc, &job.req.text);
            let Ok((embeds, pl, tcl, _)) = prep else {
                sink.on_done(job.tag, None, prep.err());
                continue;
            };
            if pl == 0 || pl > MAX_PROMPT {
                sink.on_done(
                    job.tag,
                    None,
                    Some(EngineError::InvalidArg(format!(
                        "Prompt length {pl} exceeds batch max {MAX_PROMPT}"
                    ))),
                );
                continue;
            }
            prefill_scratch.talker_kv.len = 0;
            if let Err(e) = crate::talker::talker_prefill(
                runtime,
                model,
                &mut prefill_scratch,
                &embeds,
                pl,
                true,
                false,
            ) {
                sink.on_done(job.tag, None, Some(e));
                continue;
            }

            bw.load_slot_kv(
                b,
                prefill_scratch.talker_kv.k.as_slice(),
                prefill_scratch.talker_kv.v.as_slice(),
                prefill_scratch.talker_kv.max_pos,
                pl,
            );
            d.rms_norm(
                &mut last_hidden[b * HIDDEN_SIZE..(b + 1) * HIDDEN_SIZE],
                &prefill_scratch.talker_bufs.dec_x,
                &model.talker.norm,
                1,
                HIDDEN_SIZE,
                eps,
            );

            let slot = &mut slots[b];
            slot.active = true;
            slot.tag = Some(job.tag);
            slot.abort = job.abort;
            slot.pos = pl;
            slot.tcl = tcl;
            slot.sframe = 0;
            slot.rng = LcgRng::new(job.req.seed);
            slot.prev_tok.clear();
            slot.codes.clear();
            slot.sd_stream.reset();
            slot.sd_stream.cs_tile = true;
            slot.decoded_frames = 0;
            slot.ramp_chunks = 0;
            slot.first_chunk_emitted = false;
            slot.req = Some(job.req);
            n_active += 1;
        }

        if n_active == 0 {
            if !sink.is_running() {
                break;
            }
            continue;
        }

        // Check client disconnection per active slot before stepping
        for b in 0..b_dim {
            if slots[b].active && slots[b].abort.is_cancelled() {
                let tag = slots[b].tag.take().unwrap();
                slots[b].active = false;
                slots[b].req = None;
                n_active -= 1;
                sink.on_done(tag, None, Some(EngineError::Aborted));
            }
        }
        if n_active == 0 {
            continue;
        }

        runtime.blas.set_threads(gen_blas);
        batch_proj_q(
            runtime,
            &mut bw.xt,
            &mut bw.yt,
            &mut logits,
            None,
            &model.talker.codec_head_bf16,
            &last_hidden,
            CODEC_VOCAB_SIZE,
            HIDDEN_SIZE,
            HIDDEN_SIZE,
            b_dim,
            bw.force_matvec,
        );

        for b in 0..b_dim {
            if !slots[b].active {
                code0[b] = 0;
                active_arr[b] = false;
                pos_arr[b] = 0;
                continue;
            }
            let lg = &mut logits[b * CODEC_VOCAB_SIZE..(b + 1) * CODEC_VOCAB_SIZE];
            for v in lg.iter_mut() {
                *v = v.clamp(-100.0, 100.0);
            }
            for t in 2048..CODEC_VOCAB_SIZE {
                if t as i32 != CODEC_EOS {
                    lg[t] = -1e30;
                }
            }
            let sf = slots[b].sframe;
            if sf < 2 {
                lg[CODEC_EOS as usize] = -1e30;
            }
            let ef = slots[b].tcl * 3;
            let bs = ef * 2;
            if ef > 0 && sf > bs {
                let bo = (0.5f32 * ((sf - bs) as f32)).min(10.0);
                lg[CODEC_EOS as usize] += bo;
            }
            let (greedy_warmup, r_temp, r_topk, r_topp, r_rep, want_stream, vol) = {
                let r = slots[b].req.as_ref().unwrap();
                (
                    r.greedy_warmup,
                    r.temperature,
                    r.top_k,
                    r.top_p,
                    r.rep_penalty,
                    r.want_stream,
                    r.volume,
                )
            };
            let (ft, ftk) = if greedy_warmup > 0 && sf < greedy_warmup {
                (0.0, 1)
            } else {
                (r_temp, r_topk)
            };
            let slot = &mut slots[b];
            let c0 = sampler.sample(
                lg,
                ft,
                ftk,
                r_topp,
                r_rep,
                &slot.prev_tok,
                &mut slot.rng,
            ) as i32;
            let nf = slot.codes.len() / NUM_CODEBOOKS;
            if c0 == CODEC_EOS || nf >= gen_cap || slot.pos >= kv_max - 1 {
                active_arr[b] = false;
                code0[b] = 0;

                if want_stream {
                    let rem = nf.saturating_sub(slot.decoded_frames);
                    let mut stream_err: Option<EngineError> = None;
                    if rem > 0 {
                        let start = slot.decoded_frames * NUM_CODEBOOKS;
                        let tail_codes = slot.codes[start..].to_vec();
                        match speech_decoder_decode_streaming(
                            runtime,
                            &model.decoder,
                            &mut slot.sd_stream,
                            &mut sd_ws,
                            &tail_codes,
                            rem,
                        ) {
                            Ok(mut aud) => {
                                if (vol - 1.0).abs() > 1e-6 {
                                    apply_gain(&mut aud, vol);
                                }
                                if !aud.is_empty() {
                                    if let Some(tag_ref) = slots[b].tag.as_mut() {
                                        if !sink.on_chunk(tag_ref, &aud) {
                                            stream_err = Some(EngineError::Aborted);
                                        }
                                    }
                                }
                            }
                            Err(e) => stream_err = Some(e),
                        }
                    } else if nf == 0 {
                        stream_err = Some(EngineError::Runtime("no frames generated".to_string()));
                    }
                    let tag = slots[b].tag.take().unwrap();
                    slots[b].active = false;
                    slots[b].req = None;
                    n_active -= 1;
                    sink.on_done(tag, None, stream_err);
                } else {
                    let tag = slots[b].tag.take().unwrap();
                    slots[b].active = false;
                    slots[b].req = None;
                    n_active -= 1;
                    if nf > 0 {
                        runtime.blas.set_threads(nt);
                        match speech_decoder_decode(runtime, &model.decoder, &slots[b].codes, nf) {
                            Ok(mut aud) if !aud.is_empty() => {
                                if (vol - 1.0).abs() > 1e-6 {
                                    apply_gain(&mut aud, vol);
                                }
                                sink.on_done(tag, Some(aud), None);
                            }
                            Ok(_) => sink.on_done(
                                tag,
                                None,
                                Some(EngineError::Runtime("empty decode".to_string())),
                            ),
                            Err(e) => sink.on_done(tag, None, Some(e)),
                        }
                        runtime.blas.set_threads(gen_blas);
                    } else {
                        sink.on_done(
                            tag,
                            None,
                            Some(EngineError::Runtime("no frames generated".to_string())),
                        );
                    }
                }
                continue;
            }
            code0[b] = c0;
            slots[b].prev_tok.push(c0);
            active_arr[b] = true;
            pos_arr[b] = slots[b].pos;
        }
        if n_active == 0 {
            continue;
        }

        batch_cp_predict(
            runtime,
            model,
            base_voice,
            &mut bw,
            &last_hidden,
            &code0,
            &mut cpcodes,
            Some(&active_arr),
        );

        for b in 0..b_dim {
            let se = &mut step_embed[b * HIDDEN_SIZE..(b + 1) * HIDDEN_SIZE];
            if !slots[b].active {
                se.fill(0.0);
                continue;
            }
            slots[b].codes.push(code0[b]);
            slots[b].codes.extend_from_slice(&cpcodes[b * 15..(b + 1) * 15]);
            lookup_codec_embed(d, model, base_voice, code0[b], se);
            for g in 0..15 {
                let cg = cpcodes[b * 15 + g];
                if cg >= 0 && (cg as usize) < CODEBOOK_SIZE {
                    let off = (cg as usize) * HIDDEN_SIZE;
                    d.bf16_accum_f32(se, &model.cp.codec_emb_bf16[g][off..off + HIDDEN_SIZE]);
                }
            }
            for j in 0..HIDDEN_SIZE {
                se[j] += tts_pad[j];
            }

            // Streaming chunk emission following 2, 4, 6, 8, ... ramp up to chunk_frames
            let (want_stream, chunk_frames, vol) = {
                let r = slots[b].req.as_ref().unwrap();
                (r.want_stream, r.chunk_frames.clamp(2, 250), r.volume)
            };
            if want_stream {
                let nf = slots[b].codes.len() / NUM_CODEBOOKS;
                let avail = nf - slots[b].decoded_frames;
                let target = if !slots[b].first_chunk_emitted && chunk_frames > 2 {
                    2
                } else {
                    let ramp = 4 + 2 * (slots[b].ramp_chunks / 8);
                    ramp.min(chunk_frames)
                };
                if avail >= target {
                    let ramped = slots[b].first_chunk_emitted || chunk_frames <= 2;
                    let start = slots[b].decoded_frames * NUM_CODEBOOKS;
                    let end = (slots[b].decoded_frames + target) * NUM_CODEBOOKS;
                    let chunk_codes = slots[b].codes[start..end].to_vec();
                    match speech_decoder_decode_streaming(
                        runtime,
                        &model.decoder,
                        &mut slots[b].sd_stream,
                        &mut sd_ws,
                        &chunk_codes,
                        target,
                    ) {
                        Ok(mut aud) => {
                            if (vol - 1.0).abs() > 1e-6 {
                                apply_gain(&mut aud, vol);
                            }
                            slots[b].decoded_frames += target;
                            if ramped {
                                slots[b].ramp_chunks += 1;
                            }
                            slots[b].first_chunk_emitted = true;
                            if !aud.is_empty() {
                                let ok = slots[b]
                                    .tag
                                    .as_mut()
                                    .map(|tag_ref| sink.on_chunk(tag_ref, &aud))
                                    .unwrap_or(false);
                                if !ok {
                                    let tag = slots[b].tag.take().unwrap();
                                    slots[b].active = false;
                                    slots[b].req = None;
                                    active_arr[b] = false;
                                    n_active -= 1;
                                    sink.on_done(tag, None, Some(EngineError::Aborted));
                                }
                            }
                        }
                        Err(e) => {
                            let tag = slots[b].tag.take().unwrap();
                            slots[b].active = false;
                            slots[b].req = None;
                            active_arr[b] = false;
                            n_active -= 1;
                            sink.on_done(tag, None, Some(e));
                        }
                    }
                }
            }
        }

        if n_active == 0 {
            continue;
        }

        let mut next_hidden = vec![0.0f32; b_dim * HIDDEN_SIZE];
        if let Err(e) = batch_talker_step_ragged(
            runtime,
            model,
            &mut bw,
            &step_embed,
            &pos_arr,
            &active_arr,
            &mut next_hidden,
        ) {
            for b in 0..b_dim {
                if slots[b].active {
                    let tag = slots[b].tag.take().unwrap();
                    slots[b].active = false;
                    slots[b].req = None;
                    sink.on_done(tag, None, Some(EngineError::Runtime(e.to_string())));
                }
            }
            break;
        }
        last_hidden = next_hidden;
        for b in 0..b_dim {
            if slots[b].active {
                slots[b].pos += 1;
                slots[b].sframe += 1;
            }
        }
    }

    for b in 0..b_dim {
        if slots[b].active {
            if let Some(tag) = slots[b].tag.take() {
                sink.on_done(tag, None, Some(EngineError::Aborted));
            }
        }
    }

    runtime.blas.set_threads(nt);
    Ok(())
}
