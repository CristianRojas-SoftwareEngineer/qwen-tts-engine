// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

use crate::quant::{
    bf16_to_f32, f16_to_f32, quantize_act_int8_scalar, Q2_0Block, Q4_0Block, Q2_0_BLOCK_SIZE,
    Q4_0_BLOCK_SIZE,
};

#[inline]
pub fn quantize_act_int8(qx: &mut [i8], x: &[f32]) -> f32 {
    quantize_act_int8_scalar(qx, x)
}

pub fn rms_norm(out: &mut [f32], x: &[f32], weight: &[f32], seq: usize, dim: usize, eps: f32) {
    for s in 0..seq {
        let xs = &x[s * dim..(s + 1) * dim];
        let os = &mut out[s * dim..(s + 1) * dim];
        let mut sum = 0.0f32;
        for &v in xs {
            sum += v * v;
        }
        let inv_rms = 1.0f32 / (sum / dim as f32 + eps).sqrt();
        for i in 0..dim {
            os[i] = xs[i] * inv_rms * weight[i];
        }
    }
}

pub fn rms_norm_residual(
    out: &mut [f32],
    x: &mut [f32],
    residual: &[f32],
    weight: &[f32],
    dim: usize,
    eps: f32,
) {
    let mut sum = 0.0f32;
    for i in 0..dim {
        x[i] += residual[i];
        sum += x[i] * x[i];
    }
    let inv_rms = 1.0f32 / (sum / dim as f32 + eps).sqrt();
    for i in 0..dim {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

pub fn rms_norm_per_head(
    x: &mut [f32],
    weight: &[f32],
    seq: usize,
    n_heads: usize,
    head_dim: usize,
    eps: f32,
) {
    let dim = n_heads * head_dim;
    for s in 0..seq {
        let xs = &mut x[s * dim..(s + 1) * dim];
        for h in 0..n_heads {
            let hs = &mut xs[h * head_dim..(h + 1) * head_dim];
            let mut sum = 0.0f32;
            for &v in hs.iter() {
                sum += v * v;
            }
            let inv_rms = 1.0f32 / (sum / head_dim as f32 + eps).sqrt();
            for i in 0..head_dim {
                hs[i] *= inv_rms * weight[i];
            }
        }
    }
}

pub fn bf16_matvec_slice(y: &mut [f32], x: &[f32], w: &[u16], cols: usize, rows: usize) {
    for o in 0..rows {
        let row = &w[o * cols..(o + 1) * cols];
        let mut sum = 0.0f32;
        for k in 0..cols {
            sum += bf16_to_f32(row[k]) * x[k];
        }
        y[o] = sum;
    }
}

pub fn bf16_matmat_slice(
    y: &mut [f32],
    w: &[u16],
    x: &[f32],
    r0: usize,
    r1: usize,
    cols: usize,
    b_dim: usize,
) {
    let mut acc = [0.0f32; 64];
    for r in r0..r1 {
        let row = &w[r * cols..(r + 1) * cols];
        let yr = &mut y[r * b_dim..(r + 1) * b_dim];
        acc[..b_dim].fill(0.0);
        for k in 0..cols {
            let wv = bf16_to_f32(row[k]);
            let xk = &x[k * b_dim..(k + 1) * b_dim];
            for b in 0..b_dim {
                acc[b] += wv * xk[b];
            }
        }
        yr.copy_from_slice(&acc[..b_dim]);
    }
}

pub fn int8_matvec_slice(
    y: &mut [f32],
    qx: &[i8],
    sx: f32,
    w: &[i8],
    scale: &[f32],
    cols: usize,
    rows: usize,
) {
    for o in 0..rows {
        let row = &w[o * cols..(o + 1) * cols];
        let mut s = 0i32;
        for k in 0..cols {
            s += (row[k] as i32) * (qx[k] as i32);
        }
        y[o] = (s as f32) * scale[o] * sx;
    }
}

pub fn int8_matmat_slice(
    y: &mut [f32],
    w: &[i8],
    scale: &[f32],
    qxt: &[i8],
    sx: &[f32],
    r0: usize,
    r1: usize,
    cols: usize,
    b_dim: usize,
) {
    const MM_TILE: usize = 8;
    let mut yt = [0.0f32; MM_TILE];
    let mut r = r0;
    while r < r1 {
        let n = (r1 - r).min(MM_TILE);
        for b in 0..b_dim {
            int8_matvec_slice(
                &mut yt[..n],
                &qxt[b * cols..(b + 1) * cols],
                sx[b],
                &w[r * cols..(r + n) * cols],
                &scale[r..r + n],
                cols,
                n,
            );
            for i in 0..n {
                y[(r + i) * b_dim + b] = yt[i];
            }
        }
        r += MM_TILE;
    }
}

pub fn q4_0_matvec_slice(
    y: &mut [f32],
    qx: &[i8],
    sx: f32,
    w: &[Q4_0Block],
    cols: usize,
    rows: usize,
) {
    let nb = cols / Q4_0_BLOCK_SIZE;
    for o in 0..rows {
        let row = &w[o * nb..(o + 1) * nb];
        let mut sum = 0.0f32;
        for b in 0..nb {
            let xb = &qx[b * Q4_0_BLOCK_SIZE..(b + 1) * Q4_0_BLOCK_SIZE];
            let qs = &row[b].qs;
            let mut dot = 0i32;
            for i in 0..16 {
                dot += ((qs[i] & 0x0F) as i32 - 8) * (xb[2 * i] as i32);
                dot += ((qs[i] >> 4) as i32 - 8) * (xb[2 * i + 1] as i32);
            }
            sum += f16_to_f32(row[b].scale_f16) * (dot as f32);
        }
        y[o] = sum * sx;
    }
}

pub fn q4_0_matmat_slice(
    y: &mut [f32],
    w: &[Q4_0Block],
    qxt: &[i8],
    sx: &[f32],
    r0: usize,
    r1: usize,
    cols: usize,
    b_dim: usize,
) {
    const MM_TILE: usize = 8;
    let nb = cols / Q4_0_BLOCK_SIZE;
    let mut yt = [0.0f32; MM_TILE];
    let mut r = r0;
    while r < r1 {
        let n = (r1 - r).min(MM_TILE);
        for b in 0..b_dim {
            q4_0_matvec_slice(
                &mut yt[..n],
                &qxt[b * cols..(b + 1) * cols],
                sx[b],
                &w[r * nb..(r + n) * nb],
                cols,
                n,
            );
            for i in 0..n {
                y[(r + i) * b_dim + b] = yt[i];
            }
        }
        r += MM_TILE;
    }
}

pub fn q2_0_matvec(y: &mut [f32], w: &[Q2_0Block], x: &[f32], rows: usize, cols: usize) {
    let bpr = cols / Q2_0_BLOCK_SIZE;
    for o in 0..rows {
        let wr = &w[o * bpr..(o + 1) * bpr];
        let mut sum = 0.0f32;
        for b in 0..bpr {
            let scale = wr[b].scale;
            let qs = &wr[b].qs;
            let xb = &x[b * Q2_0_BLOCK_SIZE..(b + 1) * Q2_0_BLOCK_SIZE];
            for i in 0..Q2_0_BLOCK_SIZE {
                let code = ((qs[i >> 2] >> ((i & 3) * 2)) & 0x3) as f32;
                sum += (code - 1.5f32) * scale * xb[i];
            }
        }
        y[o] = sum;
    }
}

pub fn argmax_matvec_bf16(x: &[f32], w_bf16: &[u16], in_dim: usize, out_dim: usize) -> usize {
    let mut best_idx = 0usize;
    let mut best_val = -1e30f32;
    for o in 0..out_dim {
        let row = &w_bf16[o * in_dim..(o + 1) * in_dim];
        let mut sum = 0.0f32;
        for k in 0..in_dim {
            sum += bf16_to_f32(row[k]) * x[k];
        }
        if sum > best_val {
            best_val = sum;
            best_idx = o;
        }
    }
    best_idx
}

pub fn causal_attention_f32_heads(
    out: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_q: usize,
    seq_k: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    q_offset: usize,
    window: usize,
) {
    let heads_per_kv = n_heads / n_kv_heads;
    let q_hidden = n_heads * head_dim;
    let kv_hidden = n_kv_heads * head_dim;

    for h in 0..n_heads {
        let kv_h = h / heads_per_kv;
        for i in 0..seq_q {
            let q_row = &q[i * q_hidden + h * head_dim..i * q_hidden + (h + 1) * head_dim];
            let o_row = &mut out[i * q_hidden + h * head_dim..i * q_hidden + (h + 1) * head_dim];
            let k_end = (q_offset + i + 1).min(seq_k);
            let k_start = if window > 0 && k_end > window {
                k_end - window
            } else {
                0
            };

            let mut max_score = -1e30f32;
            let mut sum_exp = 0.0f32;
            o_row.fill(0.0);

            for j in k_start..k_end {
                let k_row = &k[j * kv_hidden + kv_h * head_dim..j * kv_hidden + (kv_h + 1) * head_dim];
                let v_row = &v[j * kv_hidden + kv_h * head_dim..j * kv_hidden + (kv_h + 1) * head_dim];
                let mut score = 0.0f32;
                for d in 0..head_dim {
                    score += q_row[d] * k_row[d];
                }
                score *= scale;

                if score > max_score {
                    let correction = (max_score - score).exp();
                    sum_exp = sum_exp * correction + 1.0;
                    for d in 0..head_dim {
                        o_row[d] = o_row[d] * correction + v_row[d];
                    }
                    max_score = score;
                } else {
                    let wt = (score - max_score).exp();
                    sum_exp += wt;
                    for d in 0..head_dim {
                        o_row[d] += v_row[d] * wt;
                    }
                }
            }
            if sum_exp > 0.0 {
                let inv_sum = 1.0f32 / sum_exp;
                for d in 0..head_dim {
                    o_row[d] *= inv_sum;
                }
            }
        }
    }
}

pub fn causal_attention_bf16kv_heads(
    out: &mut [f32],
    q: &[f32],
    k_bf16: &[u16],
    v_bf16: &[u16],
    seq_q: usize,
    seq_k: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    q_offset: usize,
    h_begin: usize,
    h_end: usize,
) {
    let heads_per_kv = n_heads / n_kv_heads;
    let q_hidden = n_heads * head_dim;
    let kv_hidden = n_kv_heads * head_dim;

    for h in h_begin..h_end {
        let kv_h = h / heads_per_kv;
        for i in 0..seq_q {
            let q_row = &q[i * q_hidden + h * head_dim..i * q_hidden + (h + 1) * head_dim];
            let o_row = &mut out[i * q_hidden + h * head_dim..i * q_hidden + (h + 1) * head_dim];
            let k_end = (q_offset + i + 1).min(seq_k);

            let mut max_score = -1e30f32;
            let mut sum_exp = 0.0f32;
            o_row.fill(0.0);

            for j in 0..k_end {
                let k_row = &k_bf16[j * kv_hidden + kv_h * head_dim..j * kv_hidden + (kv_h + 1) * head_dim];
                let v_row = &v_bf16[j * kv_hidden + kv_h * head_dim..j * kv_hidden + (kv_h + 1) * head_dim];
                let mut score = 0.0f32;
                for d in 0..head_dim {
                    score += q_row[d] * bf16_to_f32(k_row[d]);
                }
                score *= scale;

                if score > max_score {
                    let correction = (max_score - score).exp();
                    sum_exp = sum_exp * correction + 1.0;
                    for d in 0..head_dim {
                        o_row[d] = o_row[d] * correction + bf16_to_f32(v_row[d]);
                    }
                    max_score = score;
                } else {
                    let wt = (score - max_score).exp();
                    sum_exp += wt;
                    for d in 0..head_dim {
                        o_row[d] += bf16_to_f32(v_row[d]) * wt;
                    }
                }
            }
            if sum_exp > 0.0 {
                let inv_sum = 1.0f32 / sum_exp;
                for d in 0..head_dim {
                    o_row[d] *= inv_sum;
                }
            }
        }
    }
}

pub fn apply_rope_neox_inplace(
    x: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    cos_cache: &[f32],
    sin_cache: &[f32],
    pos: usize,
) {
    let half = head_dim / 2;
    let cos_ptr = &cos_cache[pos * half..(pos + 1) * half];
    let sin_ptr = &sin_cache[pos * half..(pos + 1) * half];
    for h in 0..n_heads {
        let xh = &mut x[h * head_dim..(h + 1) * head_dim];
        for i in 0..half {
            let x1 = xh[i];
            let x2 = xh[i + half];
            xh[i] = x1 * cos_ptr[i] - x2 * sin_ptr[i];
            xh[i + half] = x2 * cos_ptr[i] + x1 * sin_ptr[i];
        }
    }
}

pub fn bf16_accum_f32(dst: &mut [f32], src_bf16: &[u16]) {
    for (d, &b) in dst.iter_mut().zip(src_bf16.iter()) {
        *d += bf16_to_f32(b);
    }
}

pub fn bf16_to_f32_vec(dst: &mut [f32], src_bf16: &[u16]) {
    for (d, &b) in dst.iter_mut().zip(src_bf16.iter()) {
        *d = bf16_to_f32(b);
    }
}

pub fn f32_to_bf16_vec(dst: &mut [u16], src: &[f32]) {
    let n = dst.len().min(src.len());
    for i in 0..n {
        dst[i] = crate::quant::f32_to_bf16_trunc(src[i]);
    }
}

pub fn f32_matvec_accum(y: &mut [f32], w: &[f32], x: &[f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let row = &w[r * cols..(r + 1) * cols];
        let mut sum = 0.0f32;
        for c in 0..cols {
            sum += row[c] * x[c];
        }
        y[r] += sum;
    }
}

pub fn snake_row(row: &mut [f32], log_alpha: f32, log_beta: f32) {
    let a = log_alpha.exp();
    let inv_b = (-log_beta).exp();
    for v in row.iter_mut() {
        let s = (a * *v).sin();
        *v += inv_b * s * s;
    }
}
