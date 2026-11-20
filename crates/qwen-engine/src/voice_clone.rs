// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::f32::consts::PI;
use std::path::Path;

use qwen_core::config::{ENC_DIM, MEL_DIM, SAMPLE_RATE};
use qwen_core::{F32Ref, MmapStore};
use qwen_kernels::{ComputeRuntime, Transpose};

use crate::audio::{read_wav_24khz, trim_trailing_silence};
use crate::error::EngineError;

pub struct SeRes2NetWeights {
    pub tdnn1_conv_w: F32Ref,
    pub tdnn1_conv_b: F32Ref,
    pub res2net_conv_w: [F32Ref; 7],
    pub res2net_conv_b: [F32Ref; 7],
    pub tdnn2_conv_w: F32Ref,
    pub tdnn2_conv_b: F32Ref,
    pub se_conv1_w: F32Ref,
    pub se_conv1_b: F32Ref,
    pub se_conv2_w: F32Ref,
    pub se_conv2_b: F32Ref,
    pub dilation: usize,
}

pub struct SpeakerEncoderWeights {
    pub block0_conv_w: F32Ref,
    pub block0_conv_b: F32Ref,
    pub se_blocks: [SeRes2NetWeights; 3],
    pub mfa_conv_w: F32Ref,
    pub mfa_conv_b: F32Ref,
    pub asp_tdnn_conv_w: F32Ref,
    pub asp_tdnn_conv_b: F32Ref,
    pub asp_conv_w: F32Ref,
    pub asp_conv_b: F32Ref,
    pub fc_w: F32Ref,
    pub fc_b: F32Ref,
}

impl SpeakerEncoderWeights {
    pub fn load(store: &MmapStore, silent: bool) -> Result<Self, EngineError> {
        if !store.contains("speaker_encoder.blocks.0.conv.weight") {
            return Err(EngineError::Model(
                "el checkpoint no tiene speaker encoder (se requiere el modelo Base)".to_string(),
            ));
        }
        let req = |name: &str| -> Result<F32Ref, EngineError> { Ok(store.tensor_f32_ref(name)?) };
        let block0_conv_w = req("speaker_encoder.blocks.0.conv.weight")?;
        let block0_conv_b = req("speaker_encoder.blocks.0.conv.bias")?;

        let dilations = [2usize, 3, 4];
        let mut blocks_vec = Vec::with_capacity(3);
        for (b, &dilation) in dilations.iter().enumerate() {
            let idx = b + 1;
            let mut r2w = Vec::with_capacity(7);
            let mut r2b = Vec::with_capacity(7);
            for r in 0..7 {
                r2w.push(req(&format!(
                    "speaker_encoder.blocks.{idx}.res2net_block.blocks.{r}.conv.weight"
                ))?);
                r2b.push(req(&format!(
                    "speaker_encoder.blocks.{idx}.res2net_block.blocks.{r}.conv.bias"
                ))?);
            }
            let res2net_conv_w: [F32Ref; 7] = r2w.try_into().map_err(|_| {
                EngineError::Model("falló la inicialización de res2net_conv_w".to_string())
            })?;
            let res2net_conv_b: [F32Ref; 7] = r2b.try_into().map_err(|_| {
                EngineError::Model("falló la inicialización de res2net_conv_b".to_string())
            })?;
            blocks_vec.push(SeRes2NetWeights {
                tdnn1_conv_w: req(&format!("speaker_encoder.blocks.{idx}.tdnn1.conv.weight"))?,
                tdnn1_conv_b: req(&format!("speaker_encoder.blocks.{idx}.tdnn1.conv.bias"))?,
                res2net_conv_w,
                res2net_conv_b,
                tdnn2_conv_w: req(&format!("speaker_encoder.blocks.{idx}.tdnn2.conv.weight"))?,
                tdnn2_conv_b: req(&format!("speaker_encoder.blocks.{idx}.tdnn2.conv.bias"))?,
                se_conv1_w: req(&format!(
                    "speaker_encoder.blocks.{idx}.se_block.conv1.weight"
                ))?,
                se_conv1_b: req(&format!("speaker_encoder.blocks.{idx}.se_block.conv1.bias"))?,
                se_conv2_w: req(&format!(
                    "speaker_encoder.blocks.{idx}.se_block.conv2.weight"
                ))?,
                se_conv2_b: req(&format!("speaker_encoder.blocks.{idx}.se_block.conv2.bias"))?,
                dilation,
            });
        }
        let se_blocks: [SeRes2NetWeights; 3] = blocks_vec
            .try_into()
            .map_err(|_| EngineError::Model("falló la inicialización de se_blocks".to_string()))?;

        let mfa_conv_w = req("speaker_encoder.mfa.conv.weight")?;
        let mfa_conv_b = req("speaker_encoder.mfa.conv.bias")?;
        let asp_tdnn_conv_w = req("speaker_encoder.asp.tdnn.conv.weight")?;
        let asp_tdnn_conv_b = req("speaker_encoder.asp.tdnn.conv.bias")?;
        let asp_conv_w = req("speaker_encoder.asp.conv.weight")?;
        let asp_conv_b = req("speaker_encoder.asp.conv.bias")?;
        let fc_w = req("speaker_encoder.fc.weight")?;
        let fc_b = req("speaker_encoder.fc.bias")?;

        if !silent {
            eprintln!(
                "  Speaker encoder (ECAPA-TDNN): cargado (enc_dim={ENC_DIM}, mel_dim={MEL_DIM})"
            );
        }

        Ok(Self {
            block0_conv_w,
            block0_conv_b,
            se_blocks,
            mfa_conv_w,
            mfa_conv_b,
            asp_tdnn_conv_w,
            asp_tdnn_conv_b,
            asp_conv_w,
            asp_conv_b,
            fc_w,
            fc_b,
        })
    }
}

fn fft_radix2(re: &mut [f32], im: &mut [f32]) {
    let n = re.len();
    let log2n = n.trailing_zeros() as usize;
    for i in 0..n {
        let mut j = 0usize;
        for b in 0..log2n {
            if (i & (1 << b)) != 0 {
                j |= 1 << (log2n - 1 - b);
            }
        }
        if j > i {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    for s in 1..=log2n {
        let m = 1usize << s;
        let half = m >> 1;
        let angle = -2.0f32 * PI / (m as f32);
        let wm_re = angle.cos();
        let wm_im = angle.sin();
        for k in (0..n).step_by(m) {
            let mut w_re = 1.0f32;
            let mut w_im = 0.0f32;
            for j in 0..half {
                let idx1 = k + j;
                let idx2 = k + j + half;
                let t_re = w_re * re[idx2] - w_im * im[idx2];
                let t_im = w_re * im[idx2] + w_im * re[idx2];
                re[idx2] = re[idx1] - t_re;
                im[idx2] = im[idx1] - t_im;
                re[idx1] += t_re;
                im[idx1] += t_im;
                let new_w_re = w_re * wm_re - w_im * wm_im;
                w_im = w_re * wm_im + w_im * wm_re;
                w_re = new_w_re;
            }
        }
    }
}

#[inline]
fn hz_to_mel_slaney(hz: f32) -> f32 {
    let f_sp = 200.0f32 / 3.0f32;
    if hz < 1000.0f32 {
        return hz / f_sp;
    }
    let min_log_mel = 1000.0f32 / f_sp;
    let logstep = 6.4f32.ln() / 27.0f32;
    min_log_mel + (hz / 1000.0f32).ln() / logstep
}

#[inline]
fn mel_to_hz_slaney(mel: f32) -> f32 {
    let f_sp = 200.0f32 / 3.0f32;
    let min_log_mel = 1000.0f32 / f_sp;
    let logstep = 6.4f32.ln() / 27.0f32;
    if mel < min_log_mel {
        return mel * f_sp;
    }
    1000.0f32 * ((mel - min_log_mel) * logstep).exp()
}

fn build_mel_filterbank(sr: usize, n_fft: usize, n_mels: usize, fmin: f32, fmax: f32) -> Vec<f32> {
    let n_freqs = n_fft / 2 + 1;
    let mut fb = vec![0.0f32; n_mels * n_freqs];
    let mel_min = hz_to_mel_slaney(fmin);
    let mel_max = hz_to_mel_slaney(fmax);
    let mut mels = Vec::with_capacity(n_mels + 2);
    for i in 0..(n_mels + 2) {
        let m = mel_min + (mel_max - mel_min) * (i as f32) / ((n_mels + 1) as f32);
        mels.push(mel_to_hz_slaney(m));
    }
    let freq_step = (sr as f32) / (n_fft as f32);
    for i in 0..n_mels {
        let f_low = mels[i];
        let f_center = mels[i + 1];
        let f_high = mels[i + 2];
        let enorm = 2.0f32 / (f_high - f_low);
        for j in 0..n_freqs {
            let f = (j as f32) * freq_step;
            let w = if f >= f_low && f <= f_center && f_center > f_low {
                (f - f_low) / (f_center - f_low)
            } else if f > f_center && f <= f_high && f_high > f_center {
                (f_high - f) / (f_high - f_center)
            } else {
                0.0f32
            };
            fb[i * n_freqs + j] = w * enorm;
        }
    }
    fb
}

/// Calcula el espectrograma log-mel (`n_fft=1024`, `hop=256`, `n_mels=128`, `sr=24000`).
pub fn mel_spectrogram(audio: &[f32], sample_rate: usize) -> Result<(Vec<f32>, usize), EngineError> {
    if sample_rate != SAMPLE_RATE {
        return Err(EngineError::InvalidArg(format!(
            "mel_spectrogram requiere audio de {SAMPLE_RATE} Hz (se obtuvo {sample_rate} Hz)"
        )));
    }
    const N_FFT: usize = 1024;
    const N_MELS: usize = 128;
    const HOP_SIZE: usize = 256;
    const WIN_SIZE: usize = 1024;
    const FMIN: f32 = 0.0;
    const FMAX: f32 = 12000.0;

    let n_samples = audio.len();
    let padding = (N_FFT - HOP_SIZE) / 2; // 384
    if n_samples <= padding {
        return Err(EngineError::InvalidArg(
            "Audio demasiado corto para el espectrograma mel".to_string(),
        ));
    }
    let padded_len = n_samples + 2 * padding;
    let mut padded = vec![0.0f32; padded_len];
    for i in 0..padding {
        padded[i] = audio[padding - i];
    }
    padded[padding..padding + n_samples].copy_from_slice(audio);
    for i in 0..padding {
        let src = (n_samples as isize - 2 - i as isize).max(0) as usize;
        padded[padding + n_samples + i] = audio[src];
    }

    if padded_len < N_FFT {
        return Err(EngineError::InvalidArg(
            "Audio demasiado corto para el espectrograma mel".to_string(),
        ));
    }
    let n_frames = (padded_len - N_FFT) / HOP_SIZE + 1;
    let mut window = vec![0.0f32; WIN_SIZE];
    for (i, w) in window.iter_mut().enumerate() {
        *w = 0.5f32 * (1.0f32 - (2.0f32 * PI * (i as f32) / (WIN_SIZE as f32)).cos());
    }

    let mel_fb = build_mel_filterbank(24000, N_FFT, N_MELS, FMIN, FMAX);
    let n_freqs = N_FFT / 2 + 1; // 513
    let mut mel = vec![0.0f32; n_frames * N_MELS];
    let mut fft_re = vec![0.0f32; N_FFT];
    let mut fft_im = vec![0.0f32; N_FFT];
    let mut spec = vec![0.0f32; n_freqs];

    for f in 0..n_frames {
        let start = f * HOP_SIZE;
        for i in 0..N_FFT {
            fft_re[i] = padded[start + i] * window[i];
            fft_im[i] = 0.0f32;
        }
        fft_radix2(&mut fft_re, &mut fft_im);
        for i in 0..n_freqs {
            spec[i] = (fft_re[i] * fft_re[i] + fft_im[i] * fft_im[i] + 1e-9f32).sqrt();
        }
        for m in 0..N_MELS {
            let mut val = 0.0f32;
            let fb_row = &mel_fb[m * n_freqs..(m + 1) * n_freqs];
            for i in 0..n_freqs {
                val += fb_row[i] * spec[i];
            }
            if val < 1e-5f32 {
                val = 1e-5f32;
            }
            mel[f * N_MELS + m] = val.ln();
        }
    }

    Ok((mel, n_frames))
}

#[allow(clippy::too_many_arguments)]
fn conv1d_same_reflect(
    runtime: &ComputeRuntime,
    input: &[f32],
    in_ch: usize,
    in_len: usize,
    weight: &[f32],
    bias: &[f32],
    out_ch: usize,
    kernel: usize,
    dilation: usize,
    output: &mut [f32],
) {
    if kernel == 1 && dilation == 1 {
        runtime.blas.sgemm(
            Transpose::NoTrans,
            Transpose::NoTrans,
            out_ch,
            in_len,
            in_ch,
            1.0,
            weight,
            in_ch,
            input,
            in_len,
            0.0,
            output,
            in_len,
        );
        for oc in 0..out_ch {
            let bv = bias[oc];
            let row = &mut output[oc * in_len..(oc + 1) * in_len];
            for v in row.iter_mut() {
                *v += bv;
            }
        }
        return;
    }

    if in_len == 0 {
        return;
    }
    let effective_k = 1 + (kernel - 1) * dilation;
    let pad_left = (effective_k - 1) / 2;
    let pad_right = effective_k - 1 - pad_left;
    let padded_len = in_len + pad_left + pad_right;
    let mut padded = vec![0.0f32; in_ch * padded_len];

    for c in 0..in_ch {
        let dst = &mut padded[c * padded_len..(c + 1) * padded_len];
        let src = &input[c * in_len..(c + 1) * in_len];
        for (i, d_slot) in dst[..pad_left].iter_mut().enumerate() {
            let idx = (pad_left - i).min(in_len - 1);
            *d_slot = src[idx];
        }
        dst[pad_left..pad_left + in_len].copy_from_slice(src);
        for (i, d_slot) in dst[pad_left + in_len..pad_left + in_len + pad_right]
            .iter_mut()
            .enumerate()
        {
            let idx = (in_len as isize - 2 - i as isize).max(0) as usize;
            *d_slot = src[idx];
        }
    }

    for oc in 0..out_ch {
        let bv = bias[oc];
        let dst = &mut output[oc * in_len..(oc + 1) * in_len];
        for (t, slot) in dst.iter_mut().enumerate() {
            let mut val = bv;
            for ic in 0..in_ch {
                let w = &weight[(oc * in_ch + ic) * kernel..(oc * in_ch + ic + 1) * kernel];
                let p = &padded[ic * padded_len + t..];
                for (k, &wk) in w.iter().enumerate() {
                    val += wk * p[k * dilation];
                }
            }
            *slot = val;
        }
    }
}

#[inline]
fn relu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn tdnn_forward(
    runtime: &ComputeRuntime,
    input: &[f32],
    in_ch: usize,
    in_len: usize,
    weight: &[f32],
    bias: &[f32],
    out_ch: usize,
    kernel: usize,
    dilation: usize,
    output: &mut [f32],
) {
    conv1d_same_reflect(
        runtime, input, in_ch, in_len, weight, bias, out_ch, kernel, dilation, output,
    );
    relu_inplace(output);
}

#[allow(clippy::too_many_arguments)]
fn res2net_forward(
    runtime: &ComputeRuntime,
    input: &[f32],
    channels: usize,
    in_len: usize,
    scale: usize,
    w: &[F32Ref; 7],
    b: &[F32Ref; 7],
    kernel: usize,
    dilation: usize,
    output: &mut [f32],
) {
    let chunk_ch = channels / scale; // 512 / 8 = 64
    let chunk_n = chunk_ch * in_len;
    let mut prev_output = vec![0.0f32; chunk_n];
    let mut block_input = vec![0.0f32; chunk_n];
    let mut block_output = vec![0.0f32; chunk_n];

    for i in 0..scale {
        let chunk = &input[i * chunk_n..(i + 1) * chunk_n];
        if i == 0 {
            output[i * chunk_n..(i + 1) * chunk_n].copy_from_slice(chunk);
            prev_output.copy_from_slice(chunk);
        } else if i == 1 {
            tdnn_forward(
                runtime,
                chunk,
                chunk_ch,
                in_len,
                &w[i - 1],
                &b[i - 1],
                chunk_ch,
                kernel,
                dilation,
                &mut block_output,
            );
            output[i * chunk_n..(i + 1) * chunk_n].copy_from_slice(&block_output);
            prev_output.copy_from_slice(&block_output);
        } else {
            for j in 0..chunk_n {
                block_input[j] = chunk[j] + prev_output[j];
            }
            tdnn_forward(
                runtime,
                &block_input,
                chunk_ch,
                in_len,
                &w[i - 1],
                &b[i - 1],
                chunk_ch,
                kernel,
                dilation,
                &mut block_output,
            );
            output[i * chunk_n..(i + 1) * chunk_n].copy_from_slice(&block_output);
            prev_output.copy_from_slice(&block_output);
        }
    }
}

fn se_res2net_block_forward(
    runtime: &ComputeRuntime,
    input: &[f32],
    channels: usize,
    in_len: usize,
    blk: &SeRes2NetWeights,
    output: &mut [f32],
) {
    let n = channels * in_len;
    let mut h1 = vec![0.0f32; n];
    tdnn_forward(
        runtime,
        input,
        channels,
        in_len,
        &blk.tdnn1_conv_w,
        &blk.tdnn1_conv_b,
        channels,
        1,
        1,
        &mut h1,
    );

    let mut h2 = vec![0.0f32; n];
    res2net_forward(
        runtime,
        &h1,
        channels,
        in_len,
        8,
        &blk.res2net_conv_w,
        &blk.res2net_conv_b,
        3,
        blk.dilation,
        &mut h2,
    );
    drop(h1);

    let mut h3 = vec![0.0f32; n];
    tdnn_forward(
        runtime,
        &h2,
        channels,
        in_len,
        &blk.tdnn2_conv_w,
        &blk.tdnn2_conv_b,
        channels,
        1,
        1,
        &mut h3,
    );
    drop(h2);

    let se_ch = 128usize;
    let mut mean = vec![0.0f32; channels];
    for c in 0..channels {
        let sum: f32 = h3[c * in_len..(c + 1) * in_len].iter().copied().sum();
        mean[c] = sum / (in_len as f32);
    }

    let mut se1 = vec![0.0f32; se_ch];
    for (i, slot) in se1.iter_mut().enumerate() {
        let mut val = blk.se_conv1_b[i];
        let w = &blk.se_conv1_w[i * channels..(i + 1) * channels];
        for c in 0..channels {
            val += w[c] * mean[c];
        }
        *slot = val.max(0.0);
    }

    let mut se2 = vec![0.0f32; channels];
    for (i, slot) in se2.iter_mut().enumerate() {
        let mut val = blk.se_conv2_b[i];
        let w = &blk.se_conv2_w[i * se_ch..(i + 1) * se_ch];
        for c in 0..se_ch {
            val += w[c] * se1[c];
        }
        *slot = 1.0f32 / (1.0f32 + (-val).exp());
    }

    for c in 0..channels {
        let sv = se2[c];
        let out_row = &mut output[c * in_len..(c + 1) * in_len];
        let h3_row = &h3[c * in_len..(c + 1) * in_len];
        let in_row = &input[c * in_len..(c + 1) * in_len];
        for t in 0..in_len {
            out_row[t] = h3_row[t] * sv + in_row[t];
        }
    }
}

pub fn speaker_encoder_forward(
    runtime: &ComputeRuntime,
    enc: &SpeakerEncoderWeights,
    mel: &[f32],
    n_frames: usize,
) -> Vec<f32> {
    let t_len = n_frames;
    let mut x = vec![0.0f32; MEL_DIM * t_len];
    for f in 0..t_len {
        for m in 0..MEL_DIM {
            x[m * t_len + f] = mel[f * MEL_DIM + m];
        }
    }

    let ch0 = 512usize;
    let mut h0 = vec![0.0f32; ch0 * t_len];
    tdnn_forward(
        runtime,
        &x,
        MEL_DIM,
        t_len,
        &enc.block0_conv_w,
        &enc.block0_conv_b,
        ch0,
        5,
        1,
        &mut h0,
    );
    drop(x);

    let mut b0 = vec![0.0f32; ch0 * t_len];
    se_res2net_block_forward(runtime, &h0, ch0, t_len, &enc.se_blocks[0], &mut b0);
    drop(h0);

    let mut b1 = vec![0.0f32; ch0 * t_len];
    se_res2net_block_forward(runtime, &b0, ch0, t_len, &enc.se_blocks[1], &mut b1);

    let mut b2 = vec![0.0f32; ch0 * t_len];
    se_res2net_block_forward(runtime, &b1, ch0, t_len, &enc.se_blocks[2], &mut b2);

    let mfa_ch = 1536usize;
    let mut cat = vec![0.0f32; mfa_ch * t_len];
    cat[0..ch0 * t_len].copy_from_slice(&b0);
    cat[ch0 * t_len..2 * ch0 * t_len].copy_from_slice(&b1);
    cat[2 * ch0 * t_len..3 * ch0 * t_len].copy_from_slice(&b2);
    drop((b0, b1, b2));

    let mut mfa_out = vec![0.0f32; mfa_ch * t_len];
    tdnn_forward(
        runtime,
        &cat,
        mfa_ch,
        t_len,
        &enc.mfa_conv_w,
        &enc.mfa_conv_b,
        mfa_ch,
        1,
        1,
        &mut mfa_out,
    );
    drop(cat);

    let mut asp_mean = vec![0.0f32; mfa_ch];
    let mut asp_std = vec![0.0f32; mfa_ch];
    for c in 0..mfa_ch {
        let row = &mfa_out[c * t_len..(c + 1) * t_len];
        let sum: f32 = row.iter().copied().sum();
        let mean = sum / (t_len as f32);
        asp_mean[c] = mean;
        let sum_sq: f32 = row.iter().map(|&v| (v - mean) * (v - mean)).sum();
        asp_std[c] = (sum_sq / (t_len as f32) + 1e-12f32).sqrt();
    }

    let att_ch = mfa_ch * 3; // 4608
    let mut att_input = vec![0.0f32; att_ch * t_len];
    att_input[..mfa_ch * t_len].copy_from_slice(&mfa_out);
    for c in 0..mfa_ch {
        let m = asp_mean[c];
        let s = asp_std[c];
        att_input[(mfa_ch + c) * t_len..(mfa_ch + c + 1) * t_len].fill(m);
        att_input[(2 * mfa_ch + c) * t_len..(2 * mfa_ch + c + 1) * t_len].fill(s);
    }

    let att_hidden = 128usize;
    let mut att_h = vec![0.0f32; att_hidden * t_len];
    tdnn_forward(
        runtime,
        &att_input,
        att_ch,
        t_len,
        &enc.asp_tdnn_conv_w,
        &enc.asp_tdnn_conv_b,
        att_hidden,
        1,
        1,
        &mut att_h,
    );
    drop(att_input);
    for v in att_h.iter_mut() {
        *v = v.tanh();
    }

    let mut att_w = vec![0.0f32; mfa_ch * t_len];
    conv1d_same_reflect(
        runtime,
        &att_h,
        att_hidden,
        t_len,
        &enc.asp_conv_w,
        &enc.asp_conv_b,
        mfa_ch,
        1,
        1,
        &mut att_w,
    );
    drop(att_h);

    for c in 0..mfa_ch {
        let row = &mut att_w[c * t_len..(c + 1) * t_len];
        let max_val = row.iter().copied().fold(-1e30f32, f32::max);
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - max_val).exp();
            sum += *v;
        }
        if sum > 0.0 && sum.is_finite() {
            let inv = 1.0f32 / sum;
            for v in row.iter_mut() {
                *v *= inv;
            }
        } else if !row.is_empty() {
            row.fill(1.0f32 / (row.len() as f32));
        }
    }

    let mut pooled = vec![0.0f32; 3072];
    for c in 0..mfa_ch {
        let w_row = &att_w[c * t_len..(c + 1) * t_len];
        let m_row = &mfa_out[c * t_len..(c + 1) * t_len];
        let mut mean = 0.0f32;
        for t in 0..t_len {
            mean += w_row[t] * m_row[t];
        }
        let mut var = 0.0f32;
        for t in 0..t_len {
            let d = m_row[t] - mean;
            var += w_row[t] * d * d;
        }
        pooled[c] = mean;
        pooled[mfa_ch + c] = (var + 1e-12f32).sqrt();
    }

    let mut out_emb = vec![0.0f32; ENC_DIM];
    for (i, slot) in out_emb.iter_mut().enumerate() {
        let mut val = enc.fc_b[i];
        let w_row = &enc.fc_w[i * 3072..(i + 1) * 3072];
        for j in 0..3072 {
            val += w_row[j] * pooled[j];
        }
        *slot = val;
    }
    out_emb
}

pub(crate) fn load_and_trim_ref_audio(
    ref_audio_path: &Path,
    max_ref_seconds: f32,
    silent: bool,
) -> Result<Vec<f32>, EngineError> {
    let mut audio = read_wav_24khz(ref_audio_path)?;
    if !silent {
        eprintln!(
            "Audio de referencia: {} ({} muestras, {} Hz, {:.2}s)",
            ref_audio_path.display(),
            audio.len(),
            SAMPLE_RATE,
            audio.len() as f32 / SAMPLE_RATE as f32
        );
    }
    if max_ref_seconds.is_finite() && max_ref_seconds > 0.0 {
        let max_samples = (max_ref_seconds * SAMPLE_RATE as f32) as usize;
        if audio.len() > max_samples {
            if !silent {
                eprintln!(
                    "  Truncado a {:.0}s (eran {:.1}s)",
                    max_ref_seconds,
                    audio.len() as f32 / SAMPLE_RATE as f32
                );
            }
            audio.truncate(max_samples);
        }
    }
    trim_trailing_silence(&mut audio, SAMPLE_RATE, silent);
    Ok(audio)
}
