// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use qwen_core::config::{
    CODEBOOK_DIM, CODEBOOK_SIZE, DEC_CODEBOOK_DIM, DEC_HEAD_DIM, DEC_HIDDEN_SIZE,
    DEC_INTERMEDIATE_SIZE, DEC_LATENT_DIM, DEC_NUM_HEADS, DEC_NUM_LAYERS, DEC_RMS_NORM_EPS,
    DEC_SLIDING_WINDOW, DEC_UPSAMPLE_CHANNELS, DEC_UPSAMPLE_RATES, NUM_CODEBOOKS,
};
use qwen_kernels::quant::erff;
use qwen_kernels::{AlignedBuf, ComputeRuntime, Transpose};

use crate::error::EngineError;
use crate::model::{
    DecoderWorkspace, SdConvNeXtWeights, SdStreamState, SpeechDecoderWeights,
};

const SD_TILE_BYTES: usize = 256 * 1024;
/// Primer bloque de upsample que el *streaming* procesa por tiles de `SD_TILE_BYTES`.
pub(crate) const SD_TILE_FROM_BLOCK: usize = 2;

fn conv_add_bias(out: &mut [f32], bias: &[f32], channels: usize, length: usize) {
    for c in 0..channels {
        let bv = bias[c];
        let row = &mut out[c * length..(c + 1) * length];
        for v in row.iter_mut() {
            *v += bv;
        }
    }
}

#[derive(Clone, Copy)]
enum CausalConvWeight<'a> {
    Pointwise(&'a [f32]),
    Reordered(&'a [f32]),
}

#[allow(clippy::too_many_arguments)]
fn causal_conv1d(
    runtime: &ComputeRuntime,
    xp: &mut AlignedBuf<f32>,
    out: &mut [f32],
    input: &[f32],
    weight: CausalConvWeight<'_>,
    bias: &[f32],
    in_ch: usize,
    out_ch: usize,
    length: usize,
    kernel: usize,
    dilation: usize,
) {
    let blas = &runtime.blas;
    match weight {
        CausalConvWeight::Pointwise(w) => {
            debug_assert_eq!(kernel, 1);
            blas.sgemm(
                Transpose::NoTrans,
                Transpose::NoTrans,
                out_ch,
                length,
                in_ch,
                1.0,
                w,
                in_ch,
                input,
                length,
                0.0,
                out,
                length,
            );
            conv_add_bias(out, bias, out_ch, length);
        }
        CausalConvWeight::Reordered(wk) => {
            let pad_left = (kernel - 1) * dilation;
            let ld = pad_left + length;
            xp.resize_zeroed(in_ch * ld);
            let xp_s = xp.as_mut_slice();
            for ic in 0..in_ch {
                let row_off = ic * ld;
                xp_s[row_off..row_off + pad_left].fill(0.0);
                xp_s[row_off + pad_left..row_off + pad_left + length]
                    .copy_from_slice(&input[ic * length..(ic + 1) * length]);
            }
            for k in 0..kernel {
                blas.sgemm(
                    Transpose::NoTrans,
                    Transpose::NoTrans,
                    out_ch,
                    length,
                    in_ch,
                    1.0,
                    &wk[k * out_ch * in_ch..],
                    in_ch,
                    &xp_s[k * dilation..],
                    ld,
                    if k > 0 { 1.0 } else { 0.0 },
                    out,
                    length,
                );
            }
            conv_add_bias(out, bias, out_ch, length);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn causal_conv_transpose1d(
    runtime: &ComputeRuntime,
    ct: &mut AlignedBuf<f32>,
    out: &mut [f32],
    input: &[f32],
    weight: &[f32],
    in_ch: usize,
    out_ch: usize,
    in_len: usize,
    out_len: usize,
    kernel: usize,
    stride: usize,
) {
    out[..out_ch * out_len].fill(0.0);
    let m = out_ch * kernel;
    ct.resize_zeroed(m * in_len);
    let r = ct.as_mut_slice();
    runtime.blas.sgemm(
        Transpose::Trans,
        Transpose::NoTrans,
        m,
        in_len,
        in_ch,
        1.0,
        weight,
        m,
        input,
        in_len,
        0.0,
        r,
        in_len,
    );

    for k in 0..kernel {
        for oc in 0..out_ch {
            let src = &r[(oc * kernel + k) * in_len..(oc * kernel + k + 1) * in_len];
            let dst = &mut out[oc * out_len..(oc + 1) * out_len];
            for (t, &src_val) in src.iter().enumerate().take(in_len) {
                let out_pos = t * stride + k;
                if out_pos < out_len {
                    dst[out_pos] += src_val;
                }
            }
        }
    }
}

/// Cola ConvNeXt: LayerNorm de 2 pasadas -> pw1(+GELU exacto) -> pw2 -> gamma*x + residual.
fn convnext_mlp(
    runtime: &ComputeRuntime,
    pw1: &mut AlignedBuf<f32>,
    cn: &SdConvNeXtWeights,
    signal: &mut [f32],
    residual: &[f32],
    cur_ch: usize,
    cur_len: usize,
) {
    for t in 0..cur_len {
        let mut mean = 0.0f32;
        for c in 0..cur_ch {
            mean += signal[c * cur_len + t];
        }
        mean /= cur_ch as f32;
        let mut var = 0.0f32;
        for c in 0..cur_ch {
            let d = signal[c * cur_len + t] - mean;
            var += d * d;
        }
        let inv_std = 1.0f32 / (var / (cur_ch as f32) + 1e-5f32).sqrt();
        for c in 0..cur_ch {
            let x = (signal[c * cur_len + t] - mean) * inv_std;
            signal[c * cur_len + t] = x * cn.norm_weight[c] + cn.norm_bias[c];
        }
    }

    let pw_dim = 4096usize;
    pw1.resize_zeroed(pw_dim * cur_len);
    let pw1_out = &mut pw1[..pw_dim * cur_len];
    runtime.blas.sgemm(
        Transpose::NoTrans,
        Transpose::NoTrans,
        pw_dim,
        cur_len,
        cur_ch,
        1.0,
        &cn.pwconv1_weight,
        cur_ch,
        signal,
        cur_len,
        0.0,
        pw1_out,
        cur_len,
    );
    conv_add_bias(pw1_out, &cn.pwconv1_bias, pw_dim, cur_len);
    const INV_SQRT2: f32 = std::f32::consts::FRAC_1_SQRT_2;
    for v in pw1_out.iter_mut() {
        *v = 0.5f32 * *v * (1.0f32 + erff(*v * INV_SQRT2));
    }

    runtime.blas.sgemm(
        Transpose::NoTrans,
        Transpose::NoTrans,
        cur_ch,
        cur_len,
        pw_dim,
        1.0,
        &cn.pwconv2_weight,
        pw_dim,
        pw1_out,
        cur_len,
        0.0,
        signal,
        cur_len,
    );
    conv_add_bias(signal, &cn.pwconv2_bias, cur_ch, cur_len);

    for ci in 0..cur_ch {
        let g = cn.gamma[ci];
        let s_row = &mut signal[ci * cur_len..(ci + 1) * cur_len];
        let r_row = &residual[ci * cur_len..(ci + 1) * cur_len];
        for t in 0..cur_len {
            s_row[t] = r_row[t] + s_row[t] * g;
        }
    }
}

#[inline]
fn cur_slot_mut<'a>(
    a: &'a mut AlignedBuf<f32>,
    b: &'a mut AlignedBuf<f32>,
    cur_slot: usize,
) -> &'a mut [f32] {
    if cur_slot == 0 {
        a.as_mut_slice()
    } else {
        b.as_mut_slice()
    }
}

#[inline]
fn buf_pair_mut<'a>(
    a: &'a mut AlignedBuf<f32>,
    b: &'a mut AlignedBuf<f32>,
    cur_slot: usize,
) -> (&'a mut AlignedBuf<f32>, &'a mut AlignedBuf<f32>) {
    if cur_slot == 0 {
        (a, b)
    } else {
        (b, a)
    }
}

fn add_bias_rm(buf: &mut [f32], bias: &[f32], n_frames: usize, dim: usize) {
    for f in 0..n_frames {
        for o in 0..dim {
            buf[f * dim + o] += bias[o];
        }
    }
}

fn add_residual_scaled(
    hidden: &mut [f32],
    proj: &[f32],
    layer_scale: &[f32],
    n_frames: usize,
    dim: usize,
) {
    for s in 0..n_frames {
        let hs = &mut hidden[s * dim..(s + 1) * dim];
        let ps = &proj[s * dim..(s + 1) * dim];
        for o in 0..dim {
            hs[o] += ps[o] * layer_scale[o];
        }
    }
}

fn cs_save_tail(tail: &mut [f32], input: &[f32], in_ch: usize, len: usize, tail_cols: usize) {
    if len >= tail_cols {
        for ic in 0..in_ch {
            tail[ic * tail_cols..(ic + 1) * tail_cols].copy_from_slice(
                &input[ic * len + (len - tail_cols)..ic * len + len],
            );
        }
    } else {
        let keep = tail_cols - len;
        for ic in 0..in_ch {
            let t = &mut tail[ic * tail_cols..(ic + 1) * tail_cols];
            t.copy_within(len..len + keep, 0);
            t[keep..keep + len].copy_from_slice(&input[ic * len..(ic + 1) * len]);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn cs_conv1d(
    runtime: &ComputeRuntime,
    ws_ext: &mut AlignedBuf<f32>,
    ws_xp: &mut AlignedBuf<f32>,
    out: &mut AlignedBuf<f32>,
    input: &[f32],
    in_ch: usize,
    out_ch: usize,
    len: usize,
    kernel: usize,
    dilation: usize,
    wk: &[f32],
    b: &[f32],
    tail: &mut [f32],
    warm: bool,
) {
    let tail_cols = (kernel - 1) * dilation;
    if !warm {
        out.resize_zeroed(out_ch * len);
        causal_conv1d(
            runtime,
            ws_xp,
            &mut out[..out_ch * len],
            input,
            CausalConvWeight::Reordered(wk),
            b,
            in_ch,
            out_ch,
            len,
            kernel,
            dilation,
        );
        cs_save_tail(tail, input, in_ch, len, tail_cols);
        return;
    }

    let ext_len = tail_cols + len;
    ws_ext.resize_zeroed(in_ch * ext_len);
    let ext = ws_ext.as_mut_slice();
    for ic in 0..in_ch {
        ext[ic * ext_len..ic * ext_len + tail_cols]
            .copy_from_slice(&tail[ic * tail_cols..(ic + 1) * tail_cols]);
        ext[ic * ext_len + tail_cols..ic * ext_len + ext_len]
            .copy_from_slice(&input[ic * len..(ic + 1) * len]);
    }
    for ic in 0..in_ch {
        tail[ic * tail_cols..(ic + 1) * tail_cols]
            .copy_from_slice(&ext[ic * ext_len + len..ic * ext_len + len + tail_cols]);
    }

    out.resize_zeroed(out_ch * len);
    let out_s = &mut out[..out_ch * len];
    for k in 0..kernel {
        runtime.blas.sgemm(
            Transpose::NoTrans,
            Transpose::NoTrans,
            out_ch,
            len,
            in_ch,
            1.0,
            &wk[k * out_ch * in_ch..],
            in_ch,
            &ext[k * dilation..],
            ext_len,
            if k > 0 { 1.0 } else { 0.0 },
            out_s,
            len,
        );
    }
    conv_add_bias(out_s, b, out_ch, len);
}

/// Convolución causal k=7 de un tile en *streaming* con una sola `sgemm` (im2col).
///
/// Monta `cols[(k · ch + ic), t] = [tail | input][ic, k · dilation + t]` y multiplica por
/// `wcat` (`[ch][7 · ch]`). Con 96 o 192 canales, `K = 7 · ch` rinde mucho más en OpenBLAS
/// que 7 `sgemm` con `K = ch`. La cola arranca a cero, así que el primer tile no necesita
/// un camino aparte.
#[allow(clippy::too_many_arguments)]
fn cs_conv1d_im2col(
    runtime: &ComputeRuntime,
    ws_cols: &mut AlignedBuf<f32>,
    out: &mut AlignedBuf<f32>,
    input: &[f32],
    ch: usize,
    len: usize,
    dilation: usize,
    wcat: &[f32],
    b: &[f32],
    tail: &mut [f32],
) {
    const KERNEL: usize = 7;
    let tail_cols = (KERNEL - 1) * dilation;
    ws_cols.resize_zeroed(KERNEL * ch * len);
    let cols = &mut ws_cols[..KERNEL * ch * len];
    for k in 0..KERNEL {
        let shift = k * dilation;
        // Columnas que aún caen en la cola: t + shift < tail_cols.
        let from_tail = tail_cols.saturating_sub(shift).min(len);
        for ic in 0..ch {
            let row = &mut cols[(k * ch + ic) * len..(k * ch + ic + 1) * len];
            let tl = &tail[ic * tail_cols..(ic + 1) * tail_cols];
            row[..from_tail].copy_from_slice(&tl[shift..shift + from_tail]);
            // El resto empieza en input[ic, 0]: si quedan columnas, from_tail = tail_cols - shift.
            row[from_tail..].copy_from_slice(&input[ic * len..ic * len + len - from_tail]);
        }
    }
    cs_save_tail(tail, input, ch, len, tail_cols);

    out.resize_zeroed(ch * len);
    let out_s = &mut out[..ch * len];
    runtime.blas.sgemm(
        Transpose::NoTrans,
        Transpose::NoTrans,
        ch,
        len,
        KERNEL * ch,
        1.0,
        wcat,
        KERNEL * ch,
        cols,
        len,
        0.0,
        out_s,
        len,
    );
    conv_add_bias(out_s, b, ch, len);
}

#[allow(clippy::too_many_arguments)]
fn cs_convt(
    runtime: &ComputeRuntime,
    ws_full: &mut AlignedBuf<f32>,
    ws_ct: &mut AlignedBuf<f32>,
    out: &mut AlignedBuf<f32>,
    input: &[f32],
    in_ch: usize,
    out_ch: usize,
    len: usize,
    kernel: usize,
    stride: usize,
    w: &[f32],
    b: &[f32],
    mut carry: Option<&mut [f32]>,
) {
    let cs = kernel - stride;
    let full_len = (len - 1) * stride + kernel;
    let out_len = len * stride;
    ws_full.resize_zeroed(out_ch * full_len);
    let full = &mut ws_full[..out_ch * full_len];
    causal_conv_transpose1d(
        runtime, ws_ct, full, input, w, in_ch, out_ch, len, full_len, kernel, stride,
    );
    out.resize_zeroed(out_ch * out_len);
    let out_s = &mut out[..out_ch * out_len];
    for oc in 0..out_ch {
        let f = &mut full[oc * full_len..(oc + 1) * full_len];
        let o = &mut out_s[oc * out_len..(oc + 1) * out_len];
        if let Some(ref mut cr_all) = carry {
            let cr = &cr_all[oc * cs..(oc + 1) * cs];
            for i in 0..cs {
                f[i] += cr[i];
            }
        }
        let bb = b[oc];
        for t in 0..out_len {
            o[t] = f[t] + bb;
        }
        if let Some(ref mut cr_all) = carry {
            let cr = &mut cr_all[oc * cs..(oc + 1) * cs];
            cr.copy_from_slice(&f[out_len..out_len + cs]);
        }
    }
}

fn cs_dwconv(
    out: &mut AlignedBuf<f32>,
    input: &[f32],
    ch: usize,
    len: usize,
    w: &[f32],
    b: &[f32],
    tail: &mut [f32],
) {
    out.resize_zeroed(ch * len);
    let out_s = &mut out[..ch * len];
    for c in 0..ch {
        let src = &input[c * len..(c + 1) * len];
        let tl = &mut tail[c * 6..(c + 1) * 6];
        let dst = &mut out_s[c * len..(c + 1) * len];
        let bb = b[c];
        let wc = &w[c * 7..(c + 1) * 7];
        for (t, dst_slot) in dst.iter_mut().enumerate().take(len) {
            let mut sum = bb;
            for (k, &wk) in wc.iter().enumerate().take(7) {
                let p = t as isize - 6 + k as isize;
                let v = if p >= 0 {
                    src[p as usize]
                } else {
                    tl[(6 + p) as usize]
                };
                sum += wk * v;
            }
            *dst_slot = sum;
        }
        if len >= 6 {
            tl.copy_from_slice(&src[len - 6..len]);
        } else {
            let mut next_tl = [0.0f32; 6];
            for i in 0..6 {
                next_tl[i] = if i + len < 6 {
                    tl[i + len]
                } else {
                    src[i + len - 6]
                };
            }
            tl.copy_from_slice(&next_tl);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn cs_res_tiled(
    runtime: &ComputeRuntime,
    ws: &mut DecoderWorkspace,
    st: &mut SdStreamState,
    sd: &SpeechDecoderWeights,
    b: usize,
    cur_slot: usize,
    cur_ch: usize,
    cur_len: usize,
) {
    const DILS: [usize; 3] = [1, 3, 9];
    let tile_len = (SD_TILE_BYTES / (4 * cur_ch)).max(16);
    let ub = &sd.upsample_blocks[b];

    let mut t0 = 0usize;
    while t0 < cur_len {
        let n = (cur_len - t0).min(tile_len);
        ws.pw1.resize_zeroed(cur_ch * n);
        ws.ct.resize_zeroed(cur_ch * n);
        {
            let cur = cur_slot_mut(&mut ws.a, &mut ws.b, cur_slot);
            for c in 0..cur_ch {
                ws.pw1[c * n..(c + 1) * n]
                    .copy_from_slice(&cur[c * cur_len + t0..c * cur_len + t0 + n]);
            }
        }

        for (r, res_blk) in ub.res_blocks.iter().enumerate() {
            ws.ct[..cur_ch * n].copy_from_slice(&ws.pw1[..cur_ch * n]);
            runtime.dispatch.snake_activation(
                &runtime.sd_pool,
                &mut ws.pw1[..cur_ch * n],
                cur_ch,
                n,
                &res_blk.snake1_alpha,
                &res_blk.snake1_beta,
            );
            let free_buf = if cur_slot == 0 {
                &mut ws.b
            } else {
                &mut ws.a
            };
            cs_conv1d_im2col(
                runtime,
                &mut ws.ext,
                free_buf,
                &ws.pw1[..cur_ch * n],
                cur_ch,
                n,
                DILS[r],
                &res_blk.conv1_w,
                &res_blk.conv1_bias,
                &mut st.cs_res_tail[b][r],
            );
            runtime.dispatch.snake_activation(
                &runtime.sd_pool,
                &mut free_buf[..cur_ch * n],
                cur_ch,
                n,
                &res_blk.snake2_alpha,
                &res_blk.snake2_beta,
            );
            causal_conv1d(
                runtime,
                &mut ws.xp,
                &mut ws.pw1[..cur_ch * n],
                &free_buf[..cur_ch * n],
                CausalConvWeight::Pointwise(&res_blk.conv2_weight),
                &res_blk.conv2_bias,
                cur_ch,
                cur_ch,
                n,
                1,
                1,
            );
            for i in 0..cur_ch * n {
                ws.pw1[i] += ws.ct[i];
            }
        }

        {
            let cur = cur_slot_mut(&mut ws.a, &mut ws.b, cur_slot);
            for c in 0..cur_ch {
                cur[c * cur_len + t0..c * cur_len + t0 + n]
                    .copy_from_slice(&ws.pw1[c * n..(c + 1) * n]);
            }
        }
        t0 += n;
    }
}

fn conv_decoder_forward_streaming(
    runtime: &ComputeRuntime,
    sd: &SpeechDecoderWeights,
    st: &mut SdStreamState,
    ws: &mut DecoderWorkspace,
    m: usize,
) -> Vec<f32> {
    st.ensure_conv_state();
    let mut cur_ch = DEC_LATENT_DIM;
    let mut cur_len = m;

    for (b, cn) in sd.convnext.iter().enumerate() {
        cs_convt(
            runtime,
            &mut ws.full,
            &mut ws.ct,
            &mut ws.b,
            &ws.a[..cur_ch * cur_len],
            cur_ch,
            cur_ch,
            cur_len,
            2,
            2,
            &cn.conv_weight,
            &cn.conv_bias,
            None,
        );
        cur_len *= 2;
        cs_dwconv(
            &mut ws.a,
            &ws.b[..cur_ch * cur_len],
            cur_ch,
            cur_len,
            &cn.dwconv_weight,
            &cn.dwconv_bias,
            &mut st.cs_cn_dw_tail[b],
        );
        convnext_mlp(
            runtime,
            &mut ws.pw1,
            cn,
            &mut ws.a[..cur_ch * cur_len],
            &ws.b[..cur_ch * cur_len],
            cur_ch,
            cur_len,
        );
    }

    cs_conv1d(
        runtime,
        &mut ws.ext,
        &mut ws.xp,
        &mut ws.b,
        &ws.a[..cur_ch * cur_len],
        cur_ch,
        1536,
        cur_len,
        7,
        1,
        &sd.initial_conv_wk,
        &sd.initial_conv_bias,
        &mut st.cs_init_tail,
        st.cs_warm,
    );
    let mut ci = 1usize;
    cur_ch = 1536;

    const DILS: [usize; 3] = [1, 3, 9];
    for (b, ub) in sd.upsample_blocks.iter().enumerate() {
        let rate = DEC_UPSAMPLE_RATES[b];
        let kernel = rate * 2;
        let out_ch = DEC_UPSAMPLE_CHANNELS[b];

        {
            let cur = cur_slot_mut(&mut ws.a, &mut ws.b, ci);
            runtime.dispatch.snake_activation(
                &runtime.sd_pool,
                &mut cur[..cur_ch * cur_len],
                cur_ch,
                cur_len,
                &ub.snake_alpha,
                &ub.snake_beta,
            );
        }

        {
            let (cur_buf, next_buf) = buf_pair_mut(&mut ws.a, &mut ws.b, ci);
            cs_convt(
                runtime,
                &mut ws.full,
                &mut ws.ct,
                next_buf,
                &cur_buf[..cur_ch * cur_len],
                cur_ch,
                out_ch,
                cur_len,
                kernel,
                rate,
                &ub.conv_weight,
                &ub.conv_bias,
                Some(&mut st.cs_up_carry[b]),
            );
        }
        ci ^= 1;
        cur_ch = out_ch;
        cur_len *= rate;

        if b >= SD_TILE_FROM_BLOCK {
            cs_res_tiled(runtime, ws, st, sd, b, ci, cur_ch, cur_len);
        } else {
            let n_el = cur_ch * cur_len;
            for (r, res_blk) in ub.res_blocks.iter().enumerate() {
                ws.res.resize_zeroed(n_el);
                {
                    let cur = cur_slot_mut(&mut ws.a, &mut ws.b, ci);
                    ws.res[..n_el].copy_from_slice(&cur[..n_el]);
                    runtime.dispatch.snake_activation(
                        &runtime.sd_pool,
                        &mut cur[..n_el],
                        cur_ch,
                        cur_len,
                        &res_blk.snake1_alpha,
                        &res_blk.snake1_beta,
                    );
                }
                let (cur_buf, next_buf) = buf_pair_mut(&mut ws.a, &mut ws.b, ci);
                cs_conv1d(
                    runtime,
                    &mut ws.ext,
                    &mut ws.xp,
                    next_buf,
                    &cur_buf[..n_el],
                    cur_ch,
                    cur_ch,
                    cur_len,
                    7,
                    DILS[r],
                    &res_blk.conv1_w,
                    &res_blk.conv1_bias,
                    &mut st.cs_res_tail[b][r],
                    st.cs_warm,
                );
                runtime.dispatch.snake_activation(
                    &runtime.sd_pool,
                    &mut next_buf[..n_el],
                    cur_ch,
                    cur_len,
                    &res_blk.snake2_alpha,
                    &res_blk.snake2_beta,
                );
                causal_conv1d(
                    runtime,
                    &mut ws.xp,
                    &mut cur_buf[..n_el],
                    &next_buf[..n_el],
                    CausalConvWeight::Pointwise(&res_blk.conv2_weight),
                    &res_blk.conv2_bias,
                    cur_ch,
                    cur_ch,
                    cur_len,
                    1,
                    1,
                );
                for (dst, &res_v) in cur_buf[..n_el].iter_mut().zip(ws.res[..n_el].iter()) {
                    *dst += res_v;
                }
            }
        }
    }

    {
        let cur = cur_slot_mut(&mut ws.a, &mut ws.b, ci);
        runtime.dispatch.snake_activation(
            &runtime.sd_pool,
            &mut cur[..cur_ch * cur_len],
            cur_ch,
            cur_len,
            &sd.final_snake_alpha,
            &sd.final_snake_beta,
        );
    }

    let cur = if ci == 0 {
        &ws.a[..cur_ch * cur_len]
    } else {
        &ws.b[..cur_ch * cur_len]
    };

    let mut audio = vec![sd.final_conv_bias[0]; cur_len];
    let tl = &mut st.cs_final_tail;
    for ic in 0..cur_ch {
        let src = &cur[ic * cur_len..(ic + 1) * cur_len];
        let wc = &sd.final_conv_weight[ic * 7..(ic + 1) * 7];
        let tc = &tl[ic * 6..(ic + 1) * 6];
        for (t, out_s) in audio.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for (k, &wk) in wc.iter().enumerate().take(7) {
                let p = t as isize - 6 + k as isize;
                let v = if p >= 0 {
                    src[p as usize]
                } else {
                    tc[(6 + p) as usize]
                };
                sum += wk * v;
            }
            *out_s += sum;
        }
    }
    for ic in 0..cur_ch {
        let tc = &mut tl[ic * 6..(ic + 1) * 6];
        let src = &cur[ic * cur_len..(ic + 1) * cur_len];
        if cur_len >= 6 {
            tc.copy_from_slice(&src[cur_len - 6..cur_len]);
        } else {
            let mut next_tc = [0.0f32; 6];
            for i in 0..6 {
                next_tc[i] = if i + cur_len < 6 {
                    tc[i + cur_len]
                } else {
                    src[i + cur_len - 6]
                };
            }
            tc.copy_from_slice(&next_tc);
        }
    }
    for s in audio.iter_mut() {
        *s = s.clamp(-1.0, 1.0);
    }

    st.cs_warm = true;
    audio
}

pub fn speech_decoder_decode_streaming(
    runtime: &ComputeRuntime,
    sd: &SpeechDecoderWeights,
    st: &mut SdStreamState,
    ws: &mut DecoderWorkspace,
    new_codes: &[i32],
    new_frames: usize,
) -> Result<Vec<f32>, EngineError> {
    if new_frames == 0 {
        return Ok(Vec::new());
    }
    if new_codes.len() < new_frames * NUM_CODEBOOKS {
        return Err(EngineError::InvalidArg(format!(
            "speech_decoder_decode_streaming: longitud de codes {} < {} ({} frames * {})",
            new_codes.len(),
            new_frames * NUM_CODEBOOKS,
            new_frames,
            NUM_CODEBOOKS
        )));
    }
    let blas = &runtime.blas;
    let d = runtime.dispatch;

    // Paso 1: descuantización VQ por frame
    if st.tmp_vq_out.len() < new_frames * DEC_CODEBOOK_DIM {
        st.tmp_vq_out.resize(new_frames * DEC_CODEBOOK_DIM, 0.0);
    }
    let vq_out = &mut st.tmp_vq_out[..new_frames * DEC_CODEBOOK_DIM];
    vq_out.fill(0.0);
    let mut cb_sum = [0.0f32; CODEBOOK_DIM];
    let rvq_first = sd.rvq_first_output_proj.as_slice();
    let rvq_rest = sd.rvq_rest_output_proj.as_slice();
    for f in 0..new_frames {
        let code0 = new_codes[f * NUM_CODEBOOKS];
        let vq_row = &mut vq_out[f * DEC_CODEBOOK_DIM..(f + 1) * DEC_CODEBOOK_DIM];
        if code0 >= 0 && (code0 as usize) < CODEBOOK_SIZE {
            let emb = &sd.codebooks[0][(code0 as usize) * CODEBOOK_DIM..][..CODEBOOK_DIM];
            d.f32_matvec_accum(vq_row, rvq_first, emb, DEC_CODEBOOK_DIM, CODEBOOK_DIM);
        }
        cb_sum.fill(0.0);
        for k in 1..NUM_CODEBOOKS {
            let code = new_codes[f * NUM_CODEBOOKS + k];
            if code >= 0 && (code as usize) < CODEBOOK_SIZE {
                let emb = &sd.codebooks[k][(code as usize) * CODEBOOK_DIM..][..CODEBOOK_DIM];
                for dd in 0..CODEBOOK_DIM {
                    cb_sum[dd] += emb[dd];
                }
            }
        }
        d.f32_matvec_accum(vq_row, rvq_rest, &cb_sum, DEC_CODEBOOK_DIM, CODEBOOK_DIM);
    }

    // Paso 2: pre-conv con relleno izquierdo de 2 frames del chunk previo
    let pad_frames = if st.vq_pad_valid { 2 } else { 0 };
    let conv_in_len = pad_frames + new_frames;
    if st.tmp_vq_cf.len() < DEC_CODEBOOK_DIM * conv_in_len {
        st.tmp_vq_cf.resize(DEC_CODEBOOK_DIM * conv_in_len, 0.0);
    }
    let vq_cf = &mut st.tmp_vq_cf[..DEC_CODEBOOK_DIM * conv_in_len];
    if st.vq_pad_valid && !st.vq_pad.is_empty() {
        for ch in 0..DEC_CODEBOOK_DIM {
            vq_cf[ch * conv_in_len] = st.vq_pad[ch * 2];
            vq_cf[ch * conv_in_len + 1] = st.vq_pad[ch * 2 + 1];
        }
    }
    for f in 0..new_frames {
        for ch in 0..DEC_CODEBOOK_DIM {
            vq_cf[ch * conv_in_len + pad_frames + f] = vq_out[f * DEC_CODEBOOK_DIM + ch];
        }
    }
    if st.vq_pad.len() < DEC_CODEBOOK_DIM * 2 {
        st.vq_pad.resize(DEC_CODEBOOK_DIM * 2, 0.0);
    }
    let save_start = conv_in_len.saturating_sub(2);
    let save_count = conv_in_len.min(2);
    for ch in 0..DEC_CODEBOOK_DIM {
        for t in 0..save_count {
            st.vq_pad[ch * 2 + (2 - save_count + t)] = vq_cf[ch * conv_in_len + save_start + t];
        }
        if save_count < 2 {
            for t in 0..(2 - save_count) {
                st.vq_pad[ch * 2 + t] = 0.0;
            }
        }
    }
    st.vq_pad_valid = true;

    if st.tmp_pre_conv_out.len() < DEC_LATENT_DIM * conv_in_len {
        st.tmp_pre_conv_out.resize(DEC_LATENT_DIM * conv_in_len, 0.0);
    }
    let pre_conv_out = &mut st.tmp_pre_conv_out[..DEC_LATENT_DIM * conv_in_len];
    causal_conv1d(
        runtime,
        &mut ws.xp,
        pre_conv_out,
        vq_cf,
        CausalConvWeight::Reordered(&sd.pre_conv_wk),
        &sd.pre_conv_bias,
        DEC_CODEBOOK_DIM,
        DEC_LATENT_DIM,
        conv_in_len,
        3,
        1,
    );

    // Paso 3: proyección de entrada sobre los nuevos frames
    if st.tmp_pre_conv_rm.len() < new_frames * DEC_LATENT_DIM {
        st.tmp_pre_conv_rm.resize(new_frames * DEC_LATENT_DIM, 0.0);
    }
    let pre_conv_rm = &mut st.tmp_pre_conv_rm[..new_frames * DEC_LATENT_DIM];
    for f in 0..new_frames {
        for dd in 0..DEC_LATENT_DIM {
            pre_conv_rm[f * DEC_LATENT_DIM + dd] = pre_conv_out[dd * conv_in_len + pad_frames + f];
        }
    }
    if st.tmp_hidden.len() < new_frames * DEC_HIDDEN_SIZE {
        st.tmp_hidden.resize(new_frames * DEC_HIDDEN_SIZE, 0.0);
    }
    let hidden = &mut st.tmp_hidden[..new_frames * DEC_HIDDEN_SIZE];
    blas.sgemm(
        Transpose::NoTrans,
        Transpose::Trans,
        new_frames,
        DEC_HIDDEN_SIZE,
        DEC_LATENT_DIM,
        1.0,
        pre_conv_rm,
        DEC_LATENT_DIM,
        &sd.input_proj_weight,
        DEC_LATENT_DIM,
        0.0,
        hidden,
        DEC_HIDDEN_SIZE,
    );
    add_bias_rm(hidden, &sd.input_proj_bias, new_frames, DEC_HIDDEN_SIZE);

    // Paso 4: pre-transformer con caché KV compactable
    let qkv_dim = DEC_NUM_HEADS * DEC_HEAD_DIM;
    let window = DEC_SLIDING_WINDOW;
    let mut need = (st.kv_len + new_frames) - st.kv_base;
    if need > st.kv_alloc {
        let keep_from = st.kv_len.saturating_sub(window - 1);
        if keep_from > st.kv_base && st.kv_len > st.kv_base {
            let keep = st.kv_len - keep_from;
            let shift = keep_from - st.kv_base;
            for l in 0..DEC_NUM_LAYERS {
                if keep > 0 {
                    st.k_cache[l].copy_within(shift * qkv_dim..(shift + keep) * qkv_dim, 0);
                    st.v_cache[l].copy_within(shift * qkv_dim..(shift + keep) * qkv_dim, 0);
                }
            }
            st.kv_base = keep_from;
            need = (st.kv_len + new_frames) - st.kv_base;
        }
        if need > st.kv_alloc {
            let new_alloc = need + 256;
            for l in 0..DEC_NUM_LAYERS {
                st.k_cache[l].resize(new_alloc * qkv_dim, 0.0);
                st.v_cache[l].resize(new_alloc * qkv_dim, 0.0);
            }
            st.kv_alloc = new_alloc;
        }
    }

    if st.tmp_q.len() < new_frames * qkv_dim {
        st.tmp_q.resize(new_frames * qkv_dim, 0.0);
        st.tmp_new_k.resize(new_frames * qkv_dim, 0.0);
        st.tmp_new_v.resize(new_frames * qkv_dim, 0.0);
        st.tmp_attn_out.resize(new_frames * qkv_dim, 0.0);
    }
    if st.tmp_x_norm.len() < new_frames * DEC_HIDDEN_SIZE {
        st.tmp_x_norm.resize(new_frames * DEC_HIDDEN_SIZE, 0.0);
        st.tmp_ffn_down.resize(new_frames * DEC_HIDDEN_SIZE, 0.0);
    }
    if st.tmp_ffn_gate.len() < new_frames * DEC_INTERMEDIATE_SIZE {
        st.tmp_ffn_gate.resize(new_frames * DEC_INTERMEDIATE_SIZE, 0.0);
        st.tmp_ffn_up.resize(new_frames * DEC_INTERMEDIATE_SIZE, 0.0);
    }
    let q = &mut st.tmp_q[..new_frames * qkv_dim];
    let new_k = &mut st.tmp_new_k[..new_frames * qkv_dim];
    let new_v = &mut st.tmp_new_v[..new_frames * qkv_dim];
    let x_norm = &mut st.tmp_x_norm[..new_frames * DEC_HIDDEN_SIZE];
    let attn_out = &mut st.tmp_attn_out[..new_frames * qkv_dim];
    let ffn_gate = &mut st.tmp_ffn_gate[..new_frames * DEC_INTERMEDIATE_SIZE];
    let ffn_up = &mut st.tmp_ffn_up[..new_frames * DEC_INTERMEDIATE_SIZE];
    let ffn_down_out = &mut st.tmp_ffn_down[..new_frames * DEC_HIDDEN_SIZE];
    let scale = 1.0f32 / (DEC_HEAD_DIM as f32).sqrt();

    for (layer_idx, l) in sd.pre_layers.iter().enumerate() {
        d.rms_norm(
            x_norm,
            hidden,
            &l.attn_norm,
            new_frames,
            DEC_HIDDEN_SIZE,
            DEC_RMS_NORM_EPS,
        );
        blas.sgemm(
            Transpose::NoTrans,
            Transpose::Trans,
            new_frames,
            qkv_dim,
            DEC_HIDDEN_SIZE,
            1.0,
            x_norm,
            DEC_HIDDEN_SIZE,
            &l.attn_q,
            DEC_HIDDEN_SIZE,
            0.0,
            q,
            qkv_dim,
        );
        blas.sgemm(
            Transpose::NoTrans,
            Transpose::Trans,
            new_frames,
            qkv_dim,
            DEC_HIDDEN_SIZE,
            1.0,
            x_norm,
            DEC_HIDDEN_SIZE,
            &l.attn_k,
            DEC_HIDDEN_SIZE,
            0.0,
            new_k,
            qkv_dim,
        );
        blas.sgemm(
            Transpose::NoTrans,
            Transpose::Trans,
            new_frames,
            qkv_dim,
            DEC_HIDDEN_SIZE,
            1.0,
            x_norm,
            DEC_HIDDEN_SIZE,
            &l.attn_v,
            DEC_HIDDEN_SIZE,
            0.0,
            new_v,
            qkv_dim,
        );

        for s in 0..new_frames {
            let abs_pos = st.kv_len + s;
            d.apply_rope_neox_inplace(
                &mut q[s * qkv_dim..(s + 1) * qkv_dim],
                DEC_NUM_HEADS,
                DEC_HEAD_DIM,
                &sd.rope_cos,
                &sd.rope_sin,
                abs_pos,
            );
            d.apply_rope_neox_inplace(
                &mut new_k[s * qkv_dim..(s + 1) * qkv_dim],
                DEC_NUM_HEADS,
                DEC_HEAD_DIM,
                &sd.rope_cos,
                &sd.rope_sin,
                abs_pos,
            );
        }

        let cache_off = (st.kv_len - st.kv_base) * qkv_dim;
        st.k_cache[layer_idx][cache_off..cache_off + new_frames * qkv_dim].copy_from_slice(new_k);
        st.v_cache[layer_idx][cache_off..cache_off + new_frames * qkv_dim].copy_from_slice(new_v);

        // Atención causal con ventana deslizante por el kernel compartido: en
        // coordenadas de la caché compactable, la consulta `sq` está en
        // `kv_len + sq - kv_base` y las claves válidas son `0..kv_len + new_frames - kv_base`.
        // La compactación conserva `window - 1` posiciones, así que la ventana nunca
        // cae por debajo de `kv_base`.
        let seq_k = st.kv_len + new_frames - st.kv_base;
        d.causal_attention_windowed(
            attn_out,
            q,
            &st.k_cache[layer_idx][..seq_k * qkv_dim],
            &st.v_cache[layer_idx][..seq_k * qkv_dim],
            new_frames,
            seq_k,
            DEC_NUM_HEADS,
            DEC_NUM_HEADS,
            DEC_HEAD_DIM,
            scale,
            st.kv_len - st.kv_base,
            window,
        );

        blas.sgemm(
            Transpose::NoTrans,
            Transpose::Trans,
            new_frames,
            DEC_HIDDEN_SIZE,
            qkv_dim,
            1.0,
            attn_out,
            qkv_dim,
            &l.attn_o,
            qkv_dim,
            0.0,
            x_norm,
            DEC_HIDDEN_SIZE,
        );
        add_residual_scaled(
            hidden,
            x_norm,
            &l.attn_layer_scale,
            new_frames,
            DEC_HIDDEN_SIZE,
        );

        d.rms_norm(
            x_norm,
            hidden,
            &l.ffn_norm,
            new_frames,
            DEC_HIDDEN_SIZE,
            DEC_RMS_NORM_EPS,
        );
        blas.sgemm(
            Transpose::NoTrans,
            Transpose::Trans,
            new_frames,
            DEC_INTERMEDIATE_SIZE,
            DEC_HIDDEN_SIZE,
            1.0,
            x_norm,
            DEC_HIDDEN_SIZE,
            &l.ffn_gate,
            DEC_HIDDEN_SIZE,
            0.0,
            ffn_gate,
            DEC_INTERMEDIATE_SIZE,
        );
        blas.sgemm(
            Transpose::NoTrans,
            Transpose::Trans,
            new_frames,
            DEC_INTERMEDIATE_SIZE,
            DEC_HIDDEN_SIZE,
            1.0,
            x_norm,
            DEC_HIDDEN_SIZE,
            &l.ffn_up,
            DEC_HIDDEN_SIZE,
            0.0,
            ffn_up,
            DEC_INTERMEDIATE_SIZE,
        );
        for i in 0..new_frames * DEC_INTERMEDIATE_SIZE {
            ffn_gate[i] = (ffn_gate[i] / (1.0f32 + (-ffn_gate[i]).exp())) * ffn_up[i];
        }
        blas.sgemm(
            Transpose::NoTrans,
            Transpose::Trans,
            new_frames,
            DEC_HIDDEN_SIZE,
            DEC_INTERMEDIATE_SIZE,
            1.0,
            ffn_gate,
            DEC_INTERMEDIATE_SIZE,
            &l.ffn_down,
            DEC_INTERMEDIATE_SIZE,
            0.0,
            ffn_down_out,
            DEC_HIDDEN_SIZE,
        );
        add_residual_scaled(
            hidden,
            ffn_down_out,
            &l.ffn_layer_scale,
            new_frames,
            DEC_HIDDEN_SIZE,
        );
    }
    st.kv_len += new_frames;

    d.rms_norm(
        x_norm,
        hidden,
        &sd.final_norm_weight,
        new_frames,
        DEC_HIDDEN_SIZE,
        DEC_RMS_NORM_EPS,
    );
    hidden.copy_from_slice(x_norm);

    let mut lat_need = (st.latent_frames + new_frames) - st.latent_base;
    if lat_need > st.latent_alloc {
        if st.latent_frames > st.latent_base {
            st.latent_base = st.latent_frames;
            lat_need = new_frames;
        }
        if lat_need > st.latent_alloc {
            let new_alloc = lat_need + 256;
            st.latent_cache.resize(new_alloc * DEC_LATENT_DIM, 0.0);
            st.latent_alloc = new_alloc;
        }
    }
    let lat_off = (st.latent_frames - st.latent_base) * DEC_LATENT_DIM;
    let lat_dst = &mut st.latent_cache[lat_off..lat_off + new_frames * DEC_LATENT_DIM];
    blas.sgemm(
        Transpose::NoTrans,
        Transpose::Trans,
        new_frames,
        DEC_LATENT_DIM,
        DEC_HIDDEN_SIZE,
        1.0,
        hidden,
        DEC_HIDDEN_SIZE,
        &sd.output_proj_weight,
        DEC_HIDDEN_SIZE,
        0.0,
        lat_dst,
        DEC_LATENT_DIM,
    );
    add_bias_rm(lat_dst, &sd.output_proj_bias, new_frames, DEC_LATENT_DIM);
    st.latent_frames += new_frames;

    ws.a.resize_zeroed(DEC_LATENT_DIM * new_frames);
    let lat_new = &st.latent_cache[lat_off..lat_off + new_frames * DEC_LATENT_DIM];
    for f in 0..new_frames {
        for dd in 0..DEC_LATENT_DIM {
            ws.a[dd * new_frames + f] = lat_new[f * DEC_LATENT_DIM + dd];
        }
    }

    let audio = conv_decoder_forward_streaming(runtime, sd, st, ws, new_frames);
    Ok(audio)
}
