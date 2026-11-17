use std::f32::consts::PI;
use std::path::Path;

use qwen_core::config::{
    CODEBOOK_SIZE, CODEC_VOCAB_SIZE, CP_HIDDEN_SIZE, CP_INTERMEDIATE_SIZE, CP_KV_DIM,
    CP_Q_DIM, ENC_DIM, HEAD_DIM, HIDDEN_SIZE, INTERMEDIATE_SIZE, KV_DIM, MEL_DIM,
    NUM_CODEBOOKS, Q_DIM, SAMPLE_RATE, TEXT_HIDDEN_SIZE, TEXT_VOCAB_SIZE,
    TOKEN_TTS_BOS, TOKEN_TTS_EOS, TOKEN_TTS_PAD,
};
use qwen_core::{
    encode_wdelta_bf16, lookup_language, read_bin_embedding, write_bin_embedding, F32Ref,
    MmapStore, QVoiceFile, QVoiceMeta, QVoiceTpad, QVoiceWeightTensor, QVoiceWeightsOverride,
    QVoiceWovr,
};
use qwen_kernels::{AlignedBuf, ComputeRuntime, Transpose};

use crate::audio::{read_wav_24khz, trim_trailing_silence};
use crate::error::EngineError;
use crate::model::{Model06B, TextProjOverride, VoiceBindings, VoiceIdentity};
use crate::speech_encoder::speech_encoder_encode;

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
    pub fn load_opt(store: &MmapStore, silent: bool) -> Result<Option<Self>, EngineError> {
        if !store.contains("speaker_encoder.blocks.0.conv.weight") {
            return Ok(None);
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
            let res2net_conv_w: [F32Ref; 7] = r2w
                .try_into()
                .map_err(|_| EngineError::Model("res2net_conv_w init failed".to_string()))?;
            let res2net_conv_b: [F32Ref; 7] = r2b
                .try_into()
                .map_err(|_| EngineError::Model("res2net_conv_b init failed".to_string()))?;
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
            .map_err(|_| EngineError::Model("se_blocks init failed".to_string()))?;

        let mfa_conv_w = req("speaker_encoder.mfa.conv.weight")?;
        let mfa_conv_b = req("speaker_encoder.mfa.conv.bias")?;
        let asp_tdnn_conv_w = req("speaker_encoder.asp.tdnn.conv.weight")?;
        let asp_tdnn_conv_b = req("speaker_encoder.asp.tdnn.conv.bias")?;
        let asp_conv_w = req("speaker_encoder.asp.conv.weight")?;
        let asp_conv_b = req("speaker_encoder.asp.conv.bias")?;
        let fc_w = req("speaker_encoder.fc.weight")?;
        let fc_b = req("speaker_encoder.fc.bias")?;

        if !silent {
            eprintln!("  Speaker encoder (ECAPA-TDNN): loaded (enc_dim={ENC_DIM}, mel_dim={MEL_DIM})");
        }

        Ok(Some(Self {
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
        }))
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

/// Computes log-mel spectrogram (`n_fft=1024`, `hop=256`, `n_mels=128`, `sr=24000`).
pub fn mel_spectrogram(audio: &[f32], _sample_rate: usize) -> Result<(Vec<f32>, usize), EngineError> {
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
            "Audio too short for mel spectrogram".to_string(),
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
            "Audio too short for mel spectrogram".to_string(),
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
    for i in 0..se_ch {
        let mut val = blk.se_conv1_b[i];
        let w = &blk.se_conv1_w[i * channels..(i + 1) * channels];
        for c in 0..channels {
            val += w[c] * mean[c];
        }
        se1[i] = val.max(0.0);
    }

    let mut se2 = vec![0.0f32; channels];
    for i in 0..channels {
        let mut val = blk.se_conv2_b[i];
        let w = &blk.se_conv2_w[i * se_ch..(i + 1) * se_ch];
        for c in 0..se_ch {
            val += w[c] * se1[c];
        }
        se2[i] = 1.0f32 / (1.0f32 + (-val).exp());
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
        let inv = 1.0f32 / sum;
        for v in row.iter_mut() {
            *v *= inv;
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
    for i in 0..ENC_DIM {
        let mut val = enc.fc_b[i];
        let w_row = &enc.fc_w[i * 3072..(i + 1) * 3072];
        for j in 0..3072 {
            val += w_row[j] * pooled[j];
        }
        out_emb[i] = val;
    }
    out_emb
}

pub fn extract_speaker_embedding(
    runtime: &ComputeRuntime,
    model: &Model06B,
    ref_audio_path: &Path,
    max_ref_seconds: f32,
    silent: bool,
) -> Result<Vec<f32>, EngineError> {
    let enc = model.speaker_encoder.as_ref().ok_or_else(|| {
        EngineError::InvalidArg(
            "Error: --ref-audio requires a Base model (not CustomVoice)".to_string(),
        )
    })?;

    let mut audio = read_wav_24khz(ref_audio_path)?;
    if !silent {
        eprintln!(
            "Reference audio: {} ({} samples, {} Hz, {:.2}s)",
            ref_audio_path.display(),
            audio.len(),
            SAMPLE_RATE,
            audio.len() as f32 / SAMPLE_RATE as f32
        );
    }
    if max_ref_seconds > 0.0 {
        let max_samples = (max_ref_seconds * SAMPLE_RATE as f32) as usize;
        if audio.len() > max_samples {
            if !silent {
                eprintln!(
                    "  Truncating to {:.0}s (was {:.1}s)",
                    max_ref_seconds,
                    audio.len() as f32 / SAMPLE_RATE as f32
                );
            }
            audio.truncate(max_samples);
        }
    }
    trim_trailing_silence(&mut audio, SAMPLE_RATE, silent);

    let (mel, n_frames) = mel_spectrogram(&audio, SAMPLE_RATE)?;
    if !silent {
        eprintln!("Mel spectrogram: {n_frames} frames x {MEL_DIM} mels");
    }
    Ok(speaker_encoder_forward(runtime, enc, &mel, n_frames))
}

#[derive(Debug, Clone)]
pub struct VoiceLoadOptions {
    pub icl_only: bool,
    pub xvector_only: bool,
    pub graft_mode: bool,
    pub icl_frames_cap: usize,
    pub voice_strength: f32,
    pub vs_l0: i32,
    pub vs_l1: i32,
    pub silent: bool,
}

impl Default for VoiceLoadOptions {
    fn default() -> Self {
        Self {
            icl_only: false,
            xvector_only: false,
            graft_mode: false,
            icl_frames_cap: 0,
            voice_strength: 1.0,
            vs_l0: 14,
            vs_l1: 27,
            silent: false,
        }
    }
}

pub fn build_wovr_override(
    runtime: &ComputeRuntime,
    model: &Model06B,
    wovr: &QVoiceWovr,
    tpad: Option<&QVoiceTpad>,
) -> TextProjOverride {
    let mut pad = AlignedBuf::new_zeroed(HIDDEN_SIZE);
    let mut bos = AlignedBuf::new_zeroed(HIDDEN_SIZE);
    let mut eos = AlignedBuf::new_zeroed(HIDDEN_SIZE);

    if let Some(tp) = tpad {
        pad[..HIDDEN_SIZE].copy_from_slice(&tp.tts_pad_embed[..HIDDEN_SIZE]);
        bos[..HIDDEN_SIZE].copy_from_slice(&tp.tts_bos_embed[..HIDDEN_SIZE]);
        eos[..HIDDEN_SIZE].copy_from_slice(&tp.tts_eos_embed[..HIDDEN_SIZE]);
    } else {
        let mut tmp1 = AlignedBuf::<f32>::new_zeroed(TEXT_HIDDEN_SIZE);
        let mut tmp2 = AlignedBuf::<f32>::new_zeroed(TEXT_HIDDEN_SIZE);
        let d = runtime.dispatch;
        let pool = &runtime.pool;
        for (tid, out) in [
            (TOKEN_TTS_PAD, &mut pad),
            (TOKEN_TTS_BOS, &mut bos),
            (TOKEN_TTS_EOS, &mut eos),
        ] {
            let idx = tid as usize;
            let emb_row =
                &model.talker.text_embedding_bf16[idx * TEXT_HIDDEN_SIZE..(idx + 1) * TEXT_HIDDEN_SIZE];
            d.bf16_to_f32_vec(&mut tmp1, emb_row);
            d.matvec_bf16(
                pool,
                &mut tmp2,
                &wovr.fc1_bf16,
                &tmp1,
                TEXT_HIDDEN_SIZE,
                TEXT_HIDDEN_SIZE,
            );
            for j in 0..TEXT_HIDDEN_SIZE {
                tmp2[j] += wovr.fc1_bias[j];
            }
            d.silu(&mut tmp2);
            d.matvec_bf16(
                pool,
                out,
                &wovr.fc2_bf16,
                &tmp2,
                HIDDEN_SIZE,
                TEXT_HIDDEN_SIZE,
            );
            for j in 0..HIDDEN_SIZE {
                out[j] += wovr.fc2_bias[j];
            }
        }
    }

    TextProjOverride {
        fc1_bf16: wovr.fc1_bf16.clone(),
        fc1_bias: wovr.fc1_bias.clone(),
        fc2_bf16: wovr.fc2_bf16.clone(),
        fc2_bias: wovr.fc2_bias.clone(),
        codec_emb_0_2048_bf16: wovr.codec_emb_bf16.clone(),
        tts_pad_embed: pad,
        tts_bos_embed: bos,
        tts_eos_embed: eos,
    }
}

pub fn load_voice_from_file(
    runtime: &ComputeRuntime,
    model: &mut Model06B,
    path: &Path,
    ref_text_override: Option<&str>,
    opts: &VoiceLoadOptions,
) -> Result<(VoiceBindings, Option<QVoiceMeta>), EngineError> {
    let is_qvoice = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("qvoice"));

    if !is_qvoice {
        let emb = read_bin_embedding(path)?;
        if !opts.silent {
            let norm = emb.iter().map(|&v| v * v).sum::<f32>().sqrt();
            eprintln!(
                "Voice clone: loaded speaker embedding from {} ({} floats, norm={norm:.4})",
                path.display(),
                emb.len()
            );
        }
        return Ok((
            VoiceBindings {
                identity: VoiceIdentity::XVector(emb),
                wovr: None,
                emo_ref_text: None,
                emo_ref_codes: None,
                icl_frames_cap: opts.icl_frames_cap,
                graft_mode: opts.graft_mode,
            },
            None,
        ));
    }

    let skip_weights = opts.icl_only || opts.graft_mode;
    let qv = QVoiceFile::read_from_file(path, skip_weights)?;
    if let Some(weights) = &qv.weights {
        if !skip_weights {
            if model.config.is_base_model {
                return Err(EngineError::InvalidArg(format!(
                    "ERROR: {} contains weight deltas for CustomVoice, but you're loading it on a Base model.",
                    path.display()
                )));
            }
            model.apply_qvoice_weights_override(
                runtime,
                weights,
                opts.voice_strength,
                opts.vs_l0,
                opts.vs_l1,
                opts.silent,
            )?;
        }
    }

    let skip_wovr = std::env::var_os("QWEN_GRAFT_NO_WOVR").is_some();
    let wovr_override = if !skip_wovr {
        qv.wovr
            .as_ref()
            .map(|w| build_wovr_override(runtime, model, w, qv.tpad.as_ref()))
    } else {
        if !opts.silent {
            eprintln!("  QWEN_GRAFT_NO_WOVR: skipped WOVR override (pure-CV graft)");
        }
        None
    };

    let ref_text = ref_text_override
        .map(str::to_string)
        .unwrap_or_else(|| qv.ref_text.clone());

    let use_icl = !opts.xvector_only && !qv.ref_codes.is_empty() && !ref_text.is_empty();
    let identity = if use_icl {
        VoiceIdentity::IclGraft {
            speaker_embedding: qv.speaker_embedding,
            ref_text,
            ref_codes: qv.ref_codes,
        }
    } else {
        VoiceIdentity::XVector(qv.speaker_embedding)
    };

    if !opts.silent {
        eprintln!(
            "Voice clone: loaded .qvoice v{} from {}{}",
            qv.version,
            path.display(),
            match &identity {
                VoiceIdentity::IclGraft { ref_codes, .. } =>
                    format!(" ({} ICL frames)", ref_codes.len() / NUM_CODEBOOKS),
                _ => String::new(),
            }
        );
    }

    Ok((
        VoiceBindings {
            identity,
            wovr: wovr_override,
            emo_ref_text: None,
            emo_ref_codes: None,
            icl_frames_cap: opts.icl_frames_cap,
            graft_mode: opts.graft_mode,
        },
        qv.meta,
    ))
}

#[derive(Debug, Clone, Default)]
pub struct SaveVoiceOptions<'a> {
    pub ref_text: Option<&'a str>,
    pub xvector_only: bool,
    pub graft_mode: bool,
    pub icl_frames_cap: usize,
    pub max_ref_seconds: f32,
    pub language: Option<&'a str>,
    pub voice_name: Option<&'a str>,
    pub target_cv_dir: Option<&'a Path>,
    pub silent: bool,
    pub debug: bool,
}

pub fn build_voice_from_ref_audio(
    runtime: &ComputeRuntime,
    model: &Model06B,
    ref_audio_path: &Path,
    save_path: Option<&Path>,
    opts: &SaveVoiceOptions<'_>,
) -> Result<VoiceBindings, EngineError> {
    let speaker_embedding = extract_speaker_embedding(
        runtime,
        model,
        ref_audio_path,
        opts.max_ref_seconds,
        opts.silent,
    )?;

    let need_icl = !opts.xvector_only && opts.ref_text.is_some_and(|t| !t.is_empty());
    let mut ref_codes = Vec::new();
    let mut n_ref_frames = 0usize;

    if need_icl {
        if let Some(enc) = model.speech_encoder.as_ref() {
            let mut audio = read_wav_24khz(ref_audio_path)?;
            if opts.max_ref_seconds > 0.0 {
                let max_s = (opts.max_ref_seconds * SAMPLE_RATE as f32) as usize;
                if audio.len() > max_s {
                    audio.truncate(max_s);
                }
            }
            trim_trailing_silence(&mut audio, SAMPLE_RATE, opts.silent);
            let (codes, frames) =
                speech_encoder_encode(runtime, enc, &audio, opts.debug, opts.silent)?;
            ref_codes = codes;
            n_ref_frames = frames;
            if !opts.silent {
                eprintln!("Encoded ref audio: {n_ref_frames} ICL frames");
            }
        } else if !opts.silent {
            eprintln!("Warning: speech encoder unavailable, falling back to x-vector only");
        }
    }

    if let Some(out_path) = save_path {
        let is_qvoice = out_path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("qvoice"));
        if !is_qvoice {
            write_bin_embedding(out_path, &speaker_embedding)?;
            if !opts.silent {
                eprintln!(
                    "Saved speaker embedding to {} ({} floats)",
                    out_path.display(),
                    speaker_embedding.len()
                );
            }
        } else {
            let lang_name = opts.language.unwrap_or("");
            let lang_id = if lang_name.is_empty() {
                0
            } else {
                lookup_language(lang_name).unwrap_or(0) as u32
            };
            let mut flags = 0u32;
            if opts.xvector_only {
                flags |= 1;
            }
            if n_ref_frames > 0 {
                flags |= 2;
            }
            if model.config.is_base_model {
                flags |= 4;
            }

            let meta = QVoiceMeta {
                lang_id,
                lang_name: lang_name.to_string(),
                model_size: HIDDEN_SIZE as u32,
                enc_dim: ENC_DIM as u32,
                ref_dur: opts.max_ref_seconds,
                voice_name: opts.voice_name.unwrap_or("").to_string(),
                flags,
            };
            let tpad = QVoiceTpad {
                hidden: HIDDEN_SIZE as u32,
                tts_pad_embed: model.cached_tts_pad_embed.to_vec(),
                tts_bos_embed: model.cached_tts_bos_embed.to_vec(),
                tts_eos_embed: model.cached_tts_eos_embed.to_vec(),
            };
            let wovr = QVoiceWovr {
                wh: HIDDEN_SIZE as u32,
                wth: TEXT_HIDDEN_SIZE as u32,
                wcv: CODEBOOK_SIZE as u32,
                fc1_bf16: model.talker.text_proj_fc1_bf16.to_vec(),
                fc1_bias: model
                    .talker
                    .text_proj_fc1_bias
                    .as_deref()
                    .map_or_else(|| vec![0.0; TEXT_HIDDEN_SIZE], |b| b.to_vec()),
                fc2_bf16: model.talker.text_proj_fc2_bf16.to_vec(),
                fc2_bias: model
                    .talker
                    .text_proj_fc2_bias
                    .as_deref()
                    .map_or_else(|| vec![0.0; HIDDEN_SIZE], |b| b.to_vec()),
                codec_emb_bf16: model.talker.codec_embedding_bf16[..CODEBOOK_SIZE * HIDDEN_SIZE]
                    .to_vec(),
            };

            let weights = if let Some(cv_dir) = opts.target_cv_dir {
                Some(encode_wdelta_override(model, cv_dir, opts.silent)?)
            } else {
                None
            };

            let qv = QVoiceFile {
                version: 3,
                enc_dim: ENC_DIM as u32,
                speaker_embedding: speaker_embedding.clone(),
                ref_text: opts.ref_text.unwrap_or("").to_string(),
                ref_codes: ref_codes.clone(),
                meta: Some(meta),
                tpad: Some(tpad),
                wovr: Some(wovr),
                weights,
            };
            qv.write_to_file(out_path)?;
            if !opts.silent {
                eprintln!(
                    "Saved .qvoice v3 to {} (embedding + {} ICL frames + metadata + weights)",
                    out_path.display(),
                    n_ref_frames
                );
            }
        }
    }

    let identity = if !ref_codes.is_empty() && !opts.xvector_only {
        VoiceIdentity::IclGraft {
            speaker_embedding,
            ref_text: opts.ref_text.unwrap_or("").to_string(),
            ref_codes,
        }
    } else {
        VoiceIdentity::XVector(speaker_embedding)
    };

    Ok(VoiceBindings {
        identity,
        wovr: None,
        emo_ref_text: None,
        emo_ref_codes: None,
        icl_frames_cap: opts.icl_frames_cap,
        graft_mode: opts.graft_mode,
    })
}

fn encode_wdelta_override(
    model: &Model06B,
    cv_dir: &Path,
    silent: bool,
) -> Result<QVoiceWeightsOverride, EngineError> {
    let cv_store = MmapStore::open_model(cv_dir)?;
    if !silent {
        eprintln!("  Computing deltas vs {} for WDELTA encoding", cv_dir.display());
    }

    let mut tensors = Vec::with_capacity(402);
    let push_bf16 = |tensors: &mut Vec<QVoiceWeightTensor>, name: &str, base: &[u16]| -> Result<(), EngineError> {
        if let Some(cv_bf16) = cv_store.tensor_bf16_opt(name)? {
            tensors.push(encode_wdelta_bf16(name, base, cv_bf16)?);
        } else {
            let mut payload = Vec::with_capacity(base.len() * 2);
            for &u in base {
                payload.extend_from_slice(&u.to_le_bytes());
            }
            tensors.push(QVoiceWeightTensor {
                name: name.to_string(),
                data_bytes: payload.len() as u32,
                dtype_flag: 0,
                payload,
            });
        }
        Ok(())
    };
    let push_f32 = |tensors: &mut Vec<QVoiceWeightTensor>, name: &str, data: &[f32]| {
        let mut payload = Vec::with_capacity(data.len() * 4);
        for &v in data {
            payload.extend_from_slice(&v.to_le_bytes());
        }
        tensors.push(QVoiceWeightTensor {
            name: name.to_string(),
            data_bytes: payload.len() as u32,
            dtype_flag: 0,
            payload,
        });
    };

    // Global 8 tensors
    push_bf16(
        &mut tensors,
        "talker.model.text_embedding.weight",
        &model.talker.text_embedding_bf16[..TEXT_VOCAB_SIZE * TEXT_HIDDEN_SIZE],
    )?;
    push_bf16(
        &mut tensors,
        "talker.text_projection.linear_fc1.weight",
        &model.talker.text_proj_fc1_bf16,
    )?;
    if let Some(b) = model.talker.text_proj_fc1_bias.as_deref() {
        push_f32(&mut tensors, "talker.text_projection.linear_fc1.bias", b);
    }
    push_bf16(
        &mut tensors,
        "talker.text_projection.linear_fc2.weight",
        &model.talker.text_proj_fc2_bf16,
    )?;
    if let Some(b) = model.talker.text_proj_fc2_bias.as_deref() {
        push_f32(&mut tensors, "talker.text_projection.linear_fc2.bias", b);
    }
    push_bf16(
        &mut tensors,
        "talker.model.codec_embedding.weight",
        &model.talker.codec_embedding_bf16[..CODEC_VOCAB_SIZE * HIDDEN_SIZE],
    )?;
    push_bf16(
        &mut tensors,
        "talker.codec_head.weight",
        &model.talker.codec_head_bf16[..CODEC_VOCAB_SIZE * HIDDEN_SIZE],
    )?;
    push_f32(&mut tensors, "talker.model.norm.weight", &model.talker.norm);

    // 28 Talker layers
    for (li, l) in model.talker.layers.iter().enumerate() {
        push_bf16(&mut tensors, &format!("talker.model.layers.{li}.self_attn.q_proj.weight"), &l.wq_bf16[..Q_DIM * HIDDEN_SIZE])?;
        push_bf16(&mut tensors, &format!("talker.model.layers.{li}.self_attn.k_proj.weight"), &l.wk_bf16[..KV_DIM * HIDDEN_SIZE])?;
        push_bf16(&mut tensors, &format!("talker.model.layers.{li}.self_attn.v_proj.weight"), &l.wv_bf16[..KV_DIM * HIDDEN_SIZE])?;
        push_bf16(&mut tensors, &format!("talker.model.layers.{li}.self_attn.o_proj.weight"), &l.wo_bf16[..HIDDEN_SIZE * Q_DIM])?;
        push_f32(&mut tensors, &format!("talker.model.layers.{li}.self_attn.q_norm.weight"), &l.q_norm[..HEAD_DIM]);
        push_f32(&mut tensors, &format!("talker.model.layers.{li}.self_attn.k_norm.weight"), &l.k_norm[..HEAD_DIM]);
        push_f32(&mut tensors, &format!("talker.model.layers.{li}.input_layernorm.weight"), &l.input_norm[..HIDDEN_SIZE]);
        push_f32(&mut tensors, &format!("talker.model.layers.{li}.post_attention_layernorm.weight"), &l.post_attn_norm[..HIDDEN_SIZE]);
        push_bf16(&mut tensors, &format!("talker.model.layers.{li}.mlp.gate_proj.weight"), &l.gate_bf16[..INTERMEDIATE_SIZE * HIDDEN_SIZE])?;
        push_bf16(&mut tensors, &format!("talker.model.layers.{li}.mlp.up_proj.weight"), &l.up_bf16[..INTERMEDIATE_SIZE * HIDDEN_SIZE])?;
        push_bf16(&mut tensors, &format!("talker.model.layers.{li}.mlp.down_proj.weight"), &l.down_bf16[..HIDDEN_SIZE * INTERMEDIATE_SIZE])?;
    }

    // CP tensors
    push_f32(&mut tensors, "talker.code_predictor.model.norm.weight", &model.cp.norm[..CP_HIDDEN_SIZE]);
    for ci in 0..15 {
        push_bf16(&mut tensors, &format!("talker.code_predictor.model.codec_embedding.{ci}.weight"), &model.cp.codec_emb_bf16[ci][..CODEBOOK_SIZE * CP_HIDDEN_SIZE])?;
        push_bf16(&mut tensors, &format!("talker.code_predictor.lm_head.{ci}.weight"), &model.cp.lm_heads[ci].bf16[..CODEBOOK_SIZE * CP_HIDDEN_SIZE])?;
    }
    for (li, cl) in model.cp.layers.iter().enumerate() {
        push_bf16(&mut tensors, &format!("talker.code_predictor.model.layers.{li}.self_attn.q_proj.weight"), &cl.wq_bf16[..CP_Q_DIM * CP_HIDDEN_SIZE])?;
        push_bf16(&mut tensors, &format!("talker.code_predictor.model.layers.{li}.self_attn.k_proj.weight"), &cl.wk_bf16[..CP_KV_DIM * CP_HIDDEN_SIZE])?;
        push_bf16(&mut tensors, &format!("talker.code_predictor.model.layers.{li}.self_attn.v_proj.weight"), &cl.wv_bf16[..CP_KV_DIM * CP_HIDDEN_SIZE])?;
        push_bf16(&mut tensors, &format!("talker.code_predictor.model.layers.{li}.self_attn.o_proj.weight"), &cl.wo_bf16[..CP_HIDDEN_SIZE * CP_Q_DIM])?;
        push_f32(&mut tensors, &format!("talker.code_predictor.model.layers.{li}.self_attn.q_norm.weight"), &cl.q_norm[..HEAD_DIM]);
        push_f32(&mut tensors, &format!("talker.code_predictor.model.layers.{li}.self_attn.k_norm.weight"), &cl.k_norm[..HEAD_DIM]);
        push_f32(&mut tensors, &format!("talker.code_predictor.model.layers.{li}.input_layernorm.weight"), &cl.input_norm[..CP_HIDDEN_SIZE]);
        push_f32(&mut tensors, &format!("talker.code_predictor.model.layers.{li}.post_attention_layernorm.weight"), &cl.post_attn_norm[..CP_HIDDEN_SIZE]);
        push_bf16(&mut tensors, &format!("talker.code_predictor.model.layers.{li}.mlp.gate_proj.weight"), &cl.gate_bf16[..CP_INTERMEDIATE_SIZE * CP_HIDDEN_SIZE])?;
        push_bf16(&mut tensors, &format!("talker.code_predictor.model.layers.{li}.mlp.up_proj.weight"), &cl.up_bf16[..CP_INTERMEDIATE_SIZE * CP_HIDDEN_SIZE])?;
        push_bf16(&mut tensors, &format!("talker.code_predictor.model.layers.{li}.mlp.down_proj.weight"), &cl.down_bf16[..CP_HIDDEN_SIZE * CP_INTERMEDIATE_SIZE])?;
    }

    Ok(QVoiceWeightsOverride {
        target_hidden: HIDDEN_SIZE as u32,
        tensors,
    })
}
