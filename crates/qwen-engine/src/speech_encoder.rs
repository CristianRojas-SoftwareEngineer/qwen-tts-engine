// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::sync::Arc;

use qwen_core::config::{CODEBOOK_DIM, CODEBOOK_SIZE, NUM_CODEBOOKS};
use qwen_core::{F32Ref, MmapStore};
use qwen_kernels::quant::erff;
use qwen_kernels::{AlignedBuf, ComputeRuntime, KernelDispatch, Transpose};

use crate::error::EngineError;

const ENC_HIDDEN: usize = 512;
const ENC_NUM_LAYERS: usize = 8;
const ENC_NUM_HEADS: usize = 8;
const ENC_HEAD_DIM: usize = 64;
const ENC_INTER: usize = 2048;
const ENC_WINDOW: usize = 250;
const ENC_ROPE_MAX_POS: usize = 8000;
const ENC_ROPE_THETA: f32 = 10000.0;

pub struct EncConvWeights {
    pub weight: F32Ref,
    pub bias: Option<F32Ref>,
}

pub struct EncResBlockWeights {
    pub conv1: EncConvWeights,
    pub conv2: EncConvWeights,
}

pub struct EncTransformerLayerWeights {
    pub attn_norm_w: F32Ref,
    pub attn_norm_b: F32Ref,
    pub attn_q: F32Ref,
    pub attn_k: F32Ref,
    pub attn_v: F32Ref,
    pub attn_o: F32Ref,
    pub attn_layer_scale: Option<F32Ref>,
    pub ffn_norm_w: F32Ref,
    pub ffn_norm_b: F32Ref,
    pub ffn_fc1: F32Ref,
    pub ffn_fc2: F32Ref,
    pub ffn_layer_scale: Option<F32Ref>,
}

pub struct SpeechEncoderWeights {
    pub initial_conv: EncConvWeights,
    pub resblocks: [EncResBlockWeights; 4],
    pub stride_convs: [EncConvWeights; 4],
    pub final_conv: EncConvWeights,
    pub transformer: Vec<EncTransformerLayerWeights>,
    pub downsample_weight: F32Ref,
    pub rvq_semantic_input_proj: F32Ref,
    pub rvq_acoustic_input_proj: F32Ref,
    pub rope_cos: AlignedBuf<f32>,
    pub rope_sin: AlignedBuf<f32>,
    pub codebooks: Arc<[AlignedBuf<f32>; NUM_CODEBOOKS]>,
}

impl SpeechEncoderWeights {
    pub fn load_opt(
        store: &MmapStore,
        codebooks: Arc<[AlignedBuf<f32>; NUM_CODEBOOKS]>,
        silent: bool,
    ) -> Result<Option<Self>, EngineError> {
        if !store.contains("encoder.encoder.layers.0.conv.weight") {
            return Ok(None);
        }
        let req = |name: &str| -> Result<F32Ref, EngineError> { Ok(store.tensor_f32_ref(name)?) };
        let opt = |name: &str| -> Result<Option<F32Ref>, EngineError> {
            Ok(store.tensor_f32_ref_opt(name)?)
        };

        let initial_conv = EncConvWeights {
            weight: req("encoder.encoder.layers.0.conv.weight")?,
            bias: opt("encoder.encoder.layers.0.conv.bias")?,
        };

        let res_layers = [1usize, 4, 7, 10];
        let stride_layers = [3usize, 6, 9, 12];
        let mut resblocks_vec = Vec::with_capacity(4);
        let mut stride_vec = Vec::with_capacity(4);
        for i in 0..4 {
            let rl = res_layers[i];
            let sl = stride_layers[i];
            resblocks_vec.push(EncResBlockWeights {
                conv1: EncConvWeights {
                    weight: req(&format!("encoder.encoder.layers.{rl}.block.1.conv.weight"))?,
                    bias: opt(&format!("encoder.encoder.layers.{rl}.block.1.conv.bias"))?,
                },
                conv2: EncConvWeights {
                    weight: req(&format!("encoder.encoder.layers.{rl}.block.3.conv.weight"))?,
                    bias: opt(&format!("encoder.encoder.layers.{rl}.block.3.conv.bias"))?,
                },
            });
            stride_vec.push(EncConvWeights {
                weight: req(&format!("encoder.encoder.layers.{sl}.conv.weight"))?,
                bias: opt(&format!("encoder.encoder.layers.{sl}.conv.bias"))?,
            });
        }
        let resblocks: [EncResBlockWeights; 4] = resblocks_vec
            .try_into()
            .map_err(|_| EngineError::Model("SpeechEncoder resblocks init failed".to_string()))?;
        let stride_convs: [EncConvWeights; 4] = stride_vec
            .try_into()
            .map_err(|_| EngineError::Model("SpeechEncoder stride_convs init failed".to_string()))?;

        let final_conv = EncConvWeights {
            weight: req("encoder.encoder.layers.14.conv.weight")?,
            bias: opt("encoder.encoder.layers.14.conv.bias")?,
        };

        let mut transformer = Vec::with_capacity(ENC_NUM_LAYERS);
        for l in 0..ENC_NUM_LAYERS {
            transformer.push(EncTransformerLayerWeights {
                attn_norm_w: req(&format!(
                    "encoder.encoder_transformer.layers.{l}.input_layernorm.weight"
                ))?,
                attn_norm_b: req(&format!(
                    "encoder.encoder_transformer.layers.{l}.input_layernorm.bias"
                ))?,
                attn_q: req(&format!(
                    "encoder.encoder_transformer.layers.{l}.self_attn.q_proj.weight"
                ))?,
                attn_k: req(&format!(
                    "encoder.encoder_transformer.layers.{l}.self_attn.k_proj.weight"
                ))?,
                attn_v: req(&format!(
                    "encoder.encoder_transformer.layers.{l}.self_attn.v_proj.weight"
                ))?,
                attn_o: req(&format!(
                    "encoder.encoder_transformer.layers.{l}.self_attn.o_proj.weight"
                ))?,
                attn_layer_scale: opt(&format!(
                    "encoder.encoder_transformer.layers.{l}.self_attn_layer_scale.scale"
                ))?,
                ffn_norm_w: req(&format!(
                    "encoder.encoder_transformer.layers.{l}.post_attention_layernorm.weight"
                ))?,
                ffn_norm_b: req(&format!(
                    "encoder.encoder_transformer.layers.{l}.post_attention_layernorm.bias"
                ))?,
                ffn_fc1: req(&format!(
                    "encoder.encoder_transformer.layers.{l}.mlp.fc1.weight"
                ))?,
                ffn_fc2: req(&format!(
                    "encoder.encoder_transformer.layers.{l}.mlp.fc2.weight"
                ))?,
                ffn_layer_scale: opt(&format!(
                    "encoder.encoder_transformer.layers.{l}.mlp_layer_scale.scale"
                ))?,
            });
        }

        let downsample_weight = req("encoder.downsample.conv.weight")?;
        let rvq_semantic_input_proj =
            req("encoder.quantizer.semantic_residual_vector_quantizer.input_proj.weight")?;
        let rvq_acoustic_input_proj =
            req("encoder.quantizer.acoustic_residual_vector_quantizer.input_proj.weight")?;

        let (cos_v, sin_v) = KernelDispatch::compute_rope_neox_table(
            ENC_ROPE_MAX_POS,
            ENC_HEAD_DIM,
            ENC_ROPE_THETA,
        );

        if !silent {
            eprintln!("  Speech encoder: 15/15 components loaded");
        }

        Ok(Some(Self {
            initial_conv,
            resblocks,
            stride_convs,
            final_conv,
            transformer,
            downsample_weight,
            rvq_semantic_input_proj,
            rvq_acoustic_input_proj,
            rope_cos: AlignedBuf::from_slice(&cos_v),
            rope_sin: AlignedBuf::from_slice(&sin_v),
            codebooks,
        }))
    }
}

#[inline]
fn enc_conv1d_out_len(in_len: usize, kernel: usize, stride: usize) -> usize {
    let pad_left = kernel - stride;
    (in_len + pad_left - kernel) / stride + 1
}

fn elu_activation(data: &mut [f32]) {
    for v in data.iter_mut() {
        if *v < 0.0 {
            *v = v.exp() - 1.0;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn enc_causal_conv1d(
    runtime: &ComputeRuntime,
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    in_ch: usize,
    out_ch: usize,
    in_len: usize,
    kernel: usize,
    stride: usize,
) -> Vec<f32> {
    let pad_left = kernel - stride;
    let out_len = (in_len + pad_left - (kernel - 1) - 1) / stride + 1;
    let mut out = vec![0.0f32; out_ch * out_len];

    if kernel == 1 && stride == 1 {
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
            &mut out,
            out_len,
        );
        if let Some(b) = bias {
            for oc in 0..out_ch {
                let bv = b[oc];
                let row = &mut out[oc * out_len..(oc + 1) * out_len];
                for v in row.iter_mut() {
                    *v += bv;
                }
            }
        }
        return out;
    }

    for oc in 0..out_ch {
        let bv = bias.map_or(0.0, |b| b[oc]);
        let dst_row = &mut out[oc * out_len..(oc + 1) * out_len];
        for (t, dst) in dst_row.iter_mut().enumerate() {
            let mut sum = bv;
            let base_in = (t * stride) as isize - (pad_left as isize);
            for ic in 0..in_ch {
                let w_row = &weight[(oc * in_ch + ic) * kernel..(oc * in_ch + ic + 1) * kernel];
                let in_row = &input[ic * in_len..(ic + 1) * in_len];
                for (k, &wk) in w_row.iter().enumerate() {
                    let in_pos = base_in + (k as isize);
                    if in_pos >= 0 && (in_pos as usize) < in_len {
                        sum += wk * in_row[in_pos as usize];
                    }
                }
            }
            *dst = sum;
        }
    }
    out
}

/// Codificador Speech Tokenizer en IEEE-754 estricto: audio mono a 24 kHz -> `[n_frames, 16]` códigos.
pub fn speech_encoder_encode(
    runtime: &ComputeRuntime,
    enc: &SpeechEncoderWeights,
    audio: &[f32],
    debug: bool,
    silent: bool,
) -> Result<(Vec<i32>, usize), EngineError> {
    let n_samples = audio.len();
    if n_samples == 0 {
        return Err(EngineError::InvalidArg(
            "Reference audio is empty".to_string(),
        ));
    }

    if debug {
        eprintln!("[ENC] Input: {n_samples} samples");
    }

    let mut cur_ch = 1usize;
    let mut cur_len = n_samples;
    let mut signal = enc_causal_conv1d(
        runtime,
        audio,
        &enc.initial_conv.weight,
        enc.initial_conv.bias.as_deref(),
        cur_ch,
        64,
        cur_len,
        7,
        1,
    );
    cur_ch = 64;
    cur_len = enc_conv1d_out_len(cur_len, 7, 1);

    if debug {
        eprintln!("[ENC] After initial conv: ch={cur_ch}, len={cur_len}");
    }

    const OUT_CHANNELS: [usize; 4] = [128, 256, 512, 1024];
    const STRIDES: [usize; 4] = [4, 5, 6, 8];
    const KERNELS: [usize; 4] = [8, 10, 12, 16];

    for stage in 0..4 {
        let half_ch = cur_ch / 2;
        let residual = signal.clone();
        elu_activation(&mut signal);
        let mut c1 = enc_causal_conv1d(
            runtime,
            &signal,
            &enc.resblocks[stage].conv1.weight,
            enc.resblocks[stage].conv1.bias.as_deref(),
            cur_ch,
            half_ch,
            cur_len,
            3,
            1,
        );
        elu_activation(&mut c1);
        let c2 = enc_causal_conv1d(
            runtime,
            &c1,
            &enc.resblocks[stage].conv2.weight,
            enc.resblocks[stage].conv2.bias.as_deref(),
            half_ch,
            cur_ch,
            cur_len,
            1,
            1,
        );
        for i in 0..signal.len() {
            signal[i] = residual[i] + c2[i];
        }

        elu_activation(&mut signal);
        let out_ch = OUT_CHANNELS[stage];
        let out_len = enc_conv1d_out_len(cur_len, KERNELS[stage], STRIDES[stage]);
        signal = enc_causal_conv1d(
            runtime,
            &signal,
            &enc.stride_convs[stage].weight,
            enc.stride_convs[stage].bias.as_deref(),
            cur_ch,
            out_ch,
            cur_len,
            KERNELS[stage],
            STRIDES[stage],
        );
        cur_ch = out_ch;
        cur_len = out_len;

        if debug {
            eprintln!("[ENC] After stage {stage}: ch={cur_ch}, len={cur_len}");
        }
    }

    elu_activation(&mut signal);
    signal = enc_causal_conv1d(
        runtime,
        &signal,
        &enc.final_conv.weight,
        enc.final_conv.bias.as_deref(),
        cur_ch,
        ENC_HIDDEN,
        cur_len,
        3,
        1,
    );
    cur_ch = ENC_HIDDEN;
    cur_len = enc_conv1d_out_len(cur_len, 3, 1);

    if debug {
        eprintln!(
            "[ENC] After conv encoder: ch={cur_ch}, len={cur_len} ({:.1} Hz)",
            (cur_len as f32) / ((n_samples as f32) / 24000.0)
        );
    }

    // Etapa 2: Encoder Transformer [n_seq, 512]
    let n_seq = cur_len;
    if n_seq > ENC_ROPE_MAX_POS {
        return Err(EngineError::InvalidArg(format!(
            "Reference audio too long for encoder RoPE cache ({n_seq} > {ENC_ROPE_MAX_POS})"
        )));
    }
    let mut h_buf = vec![0.0f32; n_seq * ENC_HIDDEN];
    for f in 0..n_seq {
        for d in 0..ENC_HIDDEN {
            h_buf[f * ENC_HIDDEN + d] = signal[d * n_seq + f];
        }
    }

    let qkv_dim = ENC_NUM_HEADS * ENC_HEAD_DIM;
    let eps = 1e-5f32;
    let scale = 1.0f32 / (ENC_HEAD_DIM as f32).sqrt();
    let d = runtime.dispatch;
    let blas = &runtime.blas;

    let mut q = vec![0.0f32; n_seq * qkv_dim];
    let mut kk = vec![0.0f32; n_seq * qkv_dim];
    let mut vv = vec![0.0f32; n_seq * qkv_dim];
    let mut x_norm = vec![0.0f32; n_seq * ENC_HIDDEN];
    let mut attn_out = vec![0.0f32; n_seq * qkv_dim];
    let mut fc1_out = vec![0.0f32; n_seq * ENC_INTER];
    let mut fc2_out = vec![0.0f32; n_seq * ENC_HIDDEN];
    let mut scores = Vec::new();

    let layer_norm = |dst: &mut [f32], src: &[f32], w: &[f32], b: &[f32]| {
        for s in 0..n_seq {
            let xs = &src[s * ENC_HIDDEN..(s + 1) * ENC_HIDDEN];
            let xn = &mut dst[s * ENC_HIDDEN..(s + 1) * ENC_HIDDEN];
            let mut mean = 0.0f32;
            for &v in xs {
                mean += v;
            }
            mean /= ENC_HIDDEN as f32;
            let mut var = 0.0f32;
            for &v in xs {
                let diff = v - mean;
                var += diff * diff;
            }
            let inv_std = 1.0f32 / (var / (ENC_HIDDEN as f32) + eps).sqrt();
            for i in 0..ENC_HIDDEN {
                xn[i] = (xs[i] - mean) * inv_std * w[i] + b[i];
            }
        }
    };

    for (layer_idx, l) in enc.transformer.iter().enumerate() {
        layer_norm(&mut x_norm, &h_buf, &l.attn_norm_w, &l.attn_norm_b);

        blas.sgemm(
            Transpose::NoTrans,
            Transpose::Trans,
            n_seq,
            qkv_dim,
            ENC_HIDDEN,
            1.0,
            &x_norm,
            ENC_HIDDEN,
            &l.attn_q,
            ENC_HIDDEN,
            0.0,
            &mut q,
            qkv_dim,
        );
        blas.sgemm(
            Transpose::NoTrans,
            Transpose::Trans,
            n_seq,
            qkv_dim,
            ENC_HIDDEN,
            1.0,
            &x_norm,
            ENC_HIDDEN,
            &l.attn_k,
            ENC_HIDDEN,
            0.0,
            &mut kk,
            qkv_dim,
        );
        blas.sgemm(
            Transpose::NoTrans,
            Transpose::Trans,
            n_seq,
            qkv_dim,
            ENC_HIDDEN,
            1.0,
            &x_norm,
            ENC_HIDDEN,
            &l.attn_v,
            ENC_HIDDEN,
            0.0,
            &mut vv,
            qkv_dim,
        );

        for s in 0..n_seq {
            d.apply_rope_neox_inplace(
                &mut q[s * qkv_dim..(s + 1) * qkv_dim],
                ENC_NUM_HEADS,
                ENC_HEAD_DIM,
                &enc.rope_cos,
                &enc.rope_sin,
                s,
            );
            d.apply_rope_neox_inplace(
                &mut kk[s * qkv_dim..(s + 1) * qkv_dim],
                ENC_NUM_HEADS,
                ENC_HEAD_DIM,
                &enc.rope_cos,
                &enc.rope_sin,
                s,
            );
        }

        for sq in 0..n_seq {
            let out = &mut attn_out[sq * qkv_dim..(sq + 1) * qkv_dim];
            out.fill(0.0);
            let sk_start = (sq + 1).saturating_sub(ENC_WINDOW);
            let n_keys = sq - sk_start + 1;
            if scores.len() < n_keys {
                scores.resize(n_keys, 0.0);
            }

            for h in 0..ENC_NUM_HEADS {
                let qh = &q[sq * qkv_dim + h * ENC_HEAD_DIM..][..ENC_HEAD_DIM];
                let oh = &mut out[h * ENC_HEAD_DIM..(h + 1) * ENC_HEAD_DIM];
                let mut max_score = -1e30f32;
                for (j, sc_slot) in scores[..n_keys].iter_mut().enumerate() {
                    let sk = sk_start + j;
                    let kh = &kk[sk * qkv_dim + h * ENC_HEAD_DIM..][..ENC_HEAD_DIM];
                    let mut dot = 0.0f32;
                    for dd in 0..ENC_HEAD_DIM {
                        dot += qh[dd] * kh[dd];
                    }
                    let sc = dot * scale;
                    *sc_slot = sc;
                    if sc > max_score {
                        max_score = sc;
                    }
                }
                let mut sum_exp = 0.0f32;
                for sc in &mut scores[..n_keys] {
                    let e = (*sc - max_score).exp();
                    *sc = e;
                    sum_exp += e;
                }
                let inv_sum = 1.0f32 / sum_exp;
                for (j, &sc) in scores[..n_keys].iter().enumerate() {
                    let sk = sk_start + j;
                    let vh = &vv[sk * qkv_dim + h * ENC_HEAD_DIM..][..ENC_HEAD_DIM];
                    let w = sc * inv_sum;
                    for dd in 0..ENC_HEAD_DIM {
                        oh[dd] += vh[dd] * w;
                    }
                }
            }
        }

        blas.sgemm(
            Transpose::NoTrans,
            Transpose::Trans,
            n_seq,
            ENC_HIDDEN,
            qkv_dim,
            1.0,
            &attn_out,
            qkv_dim,
            &l.attn_o,
            qkv_dim,
            0.0,
            &mut x_norm,
            ENC_HIDDEN,
        );
        for s in 0..n_seq {
            let xs = &mut h_buf[s * ENC_HIDDEN..(s + 1) * ENC_HIDDEN];
            let ps = &x_norm[s * ENC_HIDDEN..(s + 1) * ENC_HIDDEN];
            if let Some(ls) = l.attn_layer_scale.as_deref() {
                for o in 0..ENC_HIDDEN {
                    xs[o] += ps[o] * ls[o];
                }
            } else {
                for o in 0..ENC_HIDDEN {
                    xs[o] += ps[o];
                }
            }
        }

        layer_norm(&mut x_norm, &h_buf, &l.ffn_norm_w, &l.ffn_norm_b);
        blas.sgemm(
            Transpose::NoTrans,
            Transpose::Trans,
            n_seq,
            ENC_INTER,
            ENC_HIDDEN,
            1.0,
            &x_norm,
            ENC_HIDDEN,
            &l.ffn_fc1,
            ENC_HIDDEN,
            0.0,
            &mut fc1_out,
            ENC_INTER,
        );
        const INV_SQRT2: f32 = std::f32::consts::FRAC_1_SQRT_2;
        for v in fc1_out.iter_mut() {
            *v = 0.5f32 * *v * (1.0f32 + erff(*v * INV_SQRT2));
        }
        blas.sgemm(
            Transpose::NoTrans,
            Transpose::Trans,
            n_seq,
            ENC_HIDDEN,
            ENC_INTER,
            1.0,
            &fc1_out,
            ENC_INTER,
            &l.ffn_fc2,
            ENC_INTER,
            0.0,
            &mut fc2_out,
            ENC_HIDDEN,
        );
        for s in 0..n_seq {
            let xs = &mut h_buf[s * ENC_HIDDEN..(s + 1) * ENC_HIDDEN];
            let fs = &fc2_out[s * ENC_HIDDEN..(s + 1) * ENC_HIDDEN];
            if let Some(ls) = l.ffn_layer_scale.as_deref() {
                for o in 0..ENC_HIDDEN {
                    xs[o] += fs[o] * ls[o];
                }
            } else {
                for o in 0..ENC_HIDDEN {
                    xs[o] += fs[o];
                }
            }
        }

        if debug && layer_idx == 0 {
            eprintln!(
                "[ENC] Transformer L0 out [:5]: [{:.6}, {:.6}, {:.6}, {:.6}, {:.6}]",
                h_buf[0], h_buf[1], h_buf[2], h_buf[3], h_buf[4]
            );
        }
    }

    // Transpone de vuelta a [512, n_seq]
    let mut tr_cf = vec![0.0f32; ENC_HIDDEN * n_seq];
    for f in 0..n_seq {
        for dd in 0..ENC_HIDDEN {
            tr_cf[dd * n_seq + f] = h_buf[f * ENC_HIDDEN + dd];
        }
    }

    // Etapa 3: convolución de submuestreo (stride=2, k=4, sin bias)
    let n_frames = enc_conv1d_out_len(n_seq, 4, 2);
    if n_frames == 0 {
        return Err(EngineError::InvalidArg(
            "Reference audio is too short for speech encoder".to_string(),
        ));
    }
    let ds_out = enc_causal_conv1d(
        runtime,
        &tr_cf,
        &enc.downsample_weight,
        None,
        ENC_HIDDEN,
        ENC_HIDDEN,
        n_seq,
        4,
        2,
    );
    if debug {
        eprintln!(
            "[ENC] After downsample: len={n_frames} ({:.1} Hz)",
            (n_frames as f32) / ((n_samples as f32) / 24000.0)
        );
    }

    // Etapa 4: cuantización RVQ (IEEE-754 estricto)
    let mut enc_hidden = vec![0.0f32; n_frames * ENC_HIDDEN];
    for f in 0..n_frames {
        for dd in 0..ENC_HIDDEN {
            enc_hidden[f * ENC_HIDDEN + dd] = ds_out[dd * n_frames + f];
        }
    }

    let mut codes = vec![0i32; n_frames * NUM_CODEBOOKS];
    let mut cb_norm2 = vec![0.0f32; CODEBOOK_SIZE];

    // Codebook 0 (semántico)
    let mut projected = vec![0.0f32; n_frames * CODEBOOK_DIM];
    blas.sgemm(
        Transpose::NoTrans,
        Transpose::Trans,
        n_frames,
        CODEBOOK_DIM,
        ENC_HIDDEN,
        1.0,
        &enc_hidden,
        ENC_HIDDEN,
        &enc.rvq_semantic_input_proj,
        ENC_HIDDEN,
        0.0,
        &mut projected,
        CODEBOOK_DIM,
    );

    let cb0 = &enc.codebooks[0];
    for e in 0..CODEBOOK_SIZE {
        let row = &cb0[e * CODEBOOK_DIM..(e + 1) * CODEBOOK_DIM];
        let mut s = 0.0f32;
        for &v in row {
            s += v * v;
        }
        cb_norm2[e] = s;
    }
    for f in 0..n_frames {
        let x = &projected[f * CODEBOOK_DIM..(f + 1) * CODEBOOK_DIM];
        let mut x_norm2 = 0.0f32;
        for &v in x {
            x_norm2 += v * v;
        }
        let mut best_idx = 0usize;
        let mut best_dist = 1e30f32;
        for e in 0..CODEBOOK_SIZE {
            let row = &cb0[e * CODEBOOK_DIM..(e + 1) * CODEBOOK_DIM];
            let mut dot = 0.0f32;
            for dd in 0..CODEBOOK_DIM {
                dot += x[dd] * row[dd];
            }
            let dist = x_norm2 + cb_norm2[e] - 2.0f32 * dot;
            if dist < best_dist {
                best_dist = dist;
                best_idx = e;
            }
        }
        codes[f * NUM_CODEBOOKS] = best_idx as i32;
    }

    // Codebooks 1..15 (acústicos)
    let mut residual = vec![0.0f32; n_frames * CODEBOOK_DIM];
    blas.sgemm(
        Transpose::NoTrans,
        Transpose::Trans,
        n_frames,
        CODEBOOK_DIM,
        ENC_HIDDEN,
        1.0,
        &enc_hidden,
        ENC_HIDDEN,
        &enc.rvq_acoustic_input_proj,
        ENC_HIDDEN,
        0.0,
        &mut residual,
        CODEBOOK_DIM,
    );

    for k in 1..NUM_CODEBOOKS {
        let cb = &enc.codebooks[k];
        for e in 0..CODEBOOK_SIZE {
            let row = &cb[e * CODEBOOK_DIM..(e + 1) * CODEBOOK_DIM];
            let mut s = 0.0f32;
            for &v in row {
                s += v * v;
            }
            cb_norm2[e] = s;
        }
        for f in 0..n_frames {
            let x = &mut residual[f * CODEBOOK_DIM..(f + 1) * CODEBOOK_DIM];
            let mut x_n2 = 0.0f32;
            for &v in &*x {
                x_n2 += v * v;
            }
            let mut best = 0usize;
            let mut best_d = 1e30f32;
            for e in 0..CODEBOOK_SIZE {
                let row = &cb[e * CODEBOOK_DIM..(e + 1) * CODEBOOK_DIM];
                let mut dot = 0.0f32;
                for dd in 0..CODEBOOK_DIM {
                    dot += x[dd] * row[dd];
                }
                let dist = x_n2 + cb_norm2[e] - 2.0f32 * dot;
                if dist < best_d {
                    best_d = dist;
                    best = e;
                }
            }
            codes[f * NUM_CODEBOOKS + k] = best as i32;
            let row = &cb[best * CODEBOOK_DIM..(best + 1) * CODEBOOK_DIM];
            for dd in 0..CODEBOOK_DIM {
                x[dd] -= row[dd];
            }
        }
    }

    if !silent {
        eprintln!(
            "  Speech encoder: {n_samples} samples -> {n_frames} frames ({NUM_CODEBOOKS} codebooks)"
        );
    }
    if debug && n_frames > 0 {
        eprintln!("[ENC] ref_codes frame 0: {:?}", &codes[..NUM_CODEBOOKS]);
    }

    Ok((codes, n_frames))
}
