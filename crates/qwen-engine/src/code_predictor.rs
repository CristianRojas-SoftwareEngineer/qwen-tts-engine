use qwen_core::config::{
    CODEBOOK_SIZE, CODEC_VOCAB_SIZE, CP_HEAD_DIM, CP_HIDDEN_SIZE, CP_INTERMEDIATE_SIZE,
    CP_KV_DIM, CP_NUM_HEADS, CP_NUM_KV_HEADS, CP_NUM_LAYERS, CP_Q_DIM, HIDDEN_SIZE,
};
use qwen_kernels::quant::quantize_bf16_to_q2_0;
use qwen_kernels::ComputeRuntime;

use crate::model::{
    lookup_codec_embed, CpLayerWeights, CpLmHead, Model06B, QuantWeight, SessionScratch,
    VoiceBindings,
};

#[inline]
fn cp_ilv2(dst: &mut [f32], a: &[f32], b: &[f32], n: usize) {
    for i in 0..n {
        dst[2 * i] = a[i];
        dst[2 * i + 1] = b[i];
    }
}

#[inline]
fn cp_dcol2(dst: &mut [f32], src: &[f32], n: usize, p: usize) {
    for i in 0..n {
        dst[i] = src[2 * i + p];
    }
}

fn cp_mm2(
    runtime: &ComputeRuntime,
    w: &QuantWeight,
    x2: &[f32],
    y2: &mut [f32],
    rows: usize,
    cols: usize,
) {
    let d = runtime.dispatch;
    let pool = &runtime.pool;
    match w {
        QuantWeight::Q4_0(q4) => d.matmat_q4_0(pool, y2, q4, x2, rows, cols, 2),
        QuantWeight::Int8 { data, scale } => d.matmat_int8(pool, y2, data, scale, x2, rows, cols, 2),
    }
}

fn cp_dispatch_qkv(
    runtime: &ComputeRuntime,
    l: &CpLayerWeights,
    x_norm: &[f32],
    q: &mut [f32],
    k: &mut [f32],
    v: &mut [f32],
) {
    let d = runtime.dispatch;
    let pool = &runtime.pool;
    match (&l.wq_q, &l.wk_q, &l.wv_q) {
        (Some(QuantWeight::Q4_0(wq)), Some(QuantWeight::Q4_0(wk)), Some(QuantWeight::Q4_0(wv))) => {
            d.matvec_q4_0_qkv(
                pool,
                q,
                k,
                v,
                wq,
                wk,
                wv,
                x_norm,
                CP_HIDDEN_SIZE,
                CP_Q_DIM,
                CP_KV_DIM,
            );
        }
        (
            Some(QuantWeight::Int8 { data: wq, scale: sq }),
            Some(QuantWeight::Int8 { data: wk, scale: sk }),
            Some(QuantWeight::Int8 { data: wv, scale: sv }),
        ) => {
            d.matvec_int8_qkv(
                pool,
                q,
                k,
                v,
                wq,
                sq,
                wk,
                sk,
                wv,
                sv,
                x_norm,
                CP_HIDDEN_SIZE,
                CP_Q_DIM,
                CP_KV_DIM,
            );
        }
        _ => {
            d.matvec_bf16_qkv(
                pool,
                q,
                k,
                v,
                &l.wq_bf16,
                &l.wk_bf16,
                &l.wv_bf16,
                x_norm,
                CP_HIDDEN_SIZE,
                CP_Q_DIM,
                CP_KV_DIM,
            );
        }
    }
}

fn cp_dispatch_proj(
    runtime: &ComputeRuntime,
    q_w: Option<&QuantWeight>,
    bf16_w: &[u16],
    x: &[f32],
    y: &mut [f32],
    rows: usize,
    cols: usize,
) {
    let d = runtime.dispatch;
    let pool = &runtime.pool;
    match q_w {
        Some(QuantWeight::Q4_0(w)) => d.matvec_q4_0(pool, y, w, x, rows, cols),
        Some(QuantWeight::Int8 { data, scale }) => {
            d.matvec_int8(pool, y, data, scale, x, rows, cols)
        }
        None => d.matvec_bf16(pool, y, bf16_w, x, rows, cols),
    }
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

fn cp_prefill2(runtime: &ComputeRuntime, model: &Model06B, scratch: &mut SessionScratch) {
    let eps = model.config.rms_norm_eps;
    let attn_scale = 1.0f32 / (CP_HEAD_DIM as f32).sqrt();
    let d = runtime.dispatch;
    let kv_max = scratch.cp_kv.max_pos;
    let cb = &mut scratch.cp_bufs;
    let swiglu_tmp = &mut scratch.talker_bufs.swiglu_tmp[..CP_INTERMEDIATE_SIZE];

    for (layer_idx, l) in model.cp.layers.iter().enumerate() {
        d.rms_norm(&mut cb.p2_xn0, &cb.dec_x, &l.input_norm, 1, CP_HIDDEN_SIZE, eps);
        d.rms_norm(&mut cb.p2_xn1, &cb.x1in, &l.input_norm, 1, CP_HIDDEN_SIZE, eps);
        cp_ilv2(&mut cb.p2_x2, &cb.p2_xn0, &cb.p2_xn1, CP_HIDDEN_SIZE);

        cp_mm2(
            runtime,
            l.wq_q.as_ref().expect("quantized CP wq"),
            &cb.p2_x2,
            &mut cb.p2_q2,
            CP_Q_DIM,
            CP_HIDDEN_SIZE,
        );
        cp_mm2(
            runtime,
            l.wk_q.as_ref().expect("quantized CP wk"),
            &cb.p2_x2,
            &mut cb.p2_k2,
            CP_KV_DIM,
            CP_HIDDEN_SIZE,
        );
        cp_mm2(
            runtime,
            l.wv_q.as_ref().expect("quantized CP wv"),
            &cb.p2_x2,
            &mut cb.p2_v2,
            CP_KV_DIM,
            CP_HIDDEN_SIZE,
        );

        let layer_base = layer_idx * kv_max * CP_KV_DIM;
        for p in 0..2 {
            cp_dcol2(&mut cb.dec_q, &cb.p2_q2, CP_Q_DIM, p);
            cp_dcol2(&mut cb.dec_k, &cb.p2_k2, CP_KV_DIM, p);
            cp_dcol2(&mut cb.dec_v, &cb.p2_v2, CP_KV_DIM, p);

            d.rms_norm_per_head(&mut cb.dec_q, &l.q_norm, 1, CP_NUM_HEADS, CP_HEAD_DIM, eps);
            d.rms_norm_per_head(&mut cb.dec_k, &l.k_norm, 1, CP_NUM_KV_HEADS, CP_HEAD_DIM, eps);

            d.apply_rope_neox_inplace(
                &mut cb.dec_q,
                CP_NUM_HEADS,
                CP_HEAD_DIM,
                &model.cp.rope_cos,
                &model.cp.rope_sin,
                p,
            );
            d.apply_rope_neox_inplace(
                &mut cb.dec_k,
                CP_NUM_KV_HEADS,
                CP_HEAD_DIM,
                &model.cp.rope_cos,
                &model.cp.rope_sin,
                p,
            );

            let kv_off = layer_base + p * CP_KV_DIM;
            d.f32_to_bf16_vec(&mut scratch.cp_kv.k[kv_off..kv_off + CP_KV_DIM], &cb.dec_k);
            d.f32_to_bf16_vec(&mut scratch.cp_kv.v[kv_off..kv_off + CP_KV_DIM], &cb.dec_v);

            let layer_k = &scratch.cp_kv.k[layer_base..layer_base + (p + 1) * CP_KV_DIM];
            let layer_v = &scratch.cp_kv.v[layer_base..layer_base + (p + 1) * CP_KV_DIM];
            let out_attn = if p == 0 {
                &mut cb.p2_attn0
            } else {
                &mut cb.dec_attn_out
            };
            d.causal_attention_bf16kv(
                out_attn,
                &cb.dec_q,
                layer_k,
                layer_v,
                1,
                p + 1,
                CP_NUM_HEADS,
                CP_NUM_KV_HEADS,
                CP_HEAD_DIM,
                attn_scale,
                p,
            );
        }

        cp_ilv2(&mut cb.p2_a2, &cb.p2_attn0, &cb.dec_attn_out, CP_Q_DIM);
        cp_mm2(
            runtime,
            l.wo_q.as_ref().expect("quantized CP wo"),
            &cb.p2_a2,
            &mut cb.p2_p2,
            CP_HIDDEN_SIZE,
            CP_Q_DIM,
        );
        for i in 0..CP_HIDDEN_SIZE {
            cb.dec_x[i] += cb.p2_p2[2 * i];
            cb.x1in[i] += cb.p2_p2[2 * i + 1];
        }

        d.rms_norm(
            &mut cb.p2_xn0,
            &cb.dec_x,
            &l.post_attn_norm,
            1,
            CP_HIDDEN_SIZE,
            eps,
        );
        d.rms_norm(
            &mut cb.p2_xn1,
            &cb.x1in,
            &l.post_attn_norm,
            1,
            CP_HIDDEN_SIZE,
            eps,
        );
        cp_ilv2(&mut cb.p2_x2, &cb.p2_xn0, &cb.p2_xn1, CP_HIDDEN_SIZE);

        cp_mm2(
            runtime,
            l.gate_up_q.as_ref().expect("quantized CP gate_up"),
            &cb.p2_x2,
            &mut cb.p2_g2,
            2 * CP_INTERMEDIATE_SIZE,
            CP_HIDDEN_SIZE,
        );
        cp_dcol2(&mut cb.p2_g0, &cb.p2_g2, 2 * CP_INTERMEDIATE_SIZE, 0);
        cp_dcol2(&mut cb.dec_gate, &cb.p2_g2, 2 * CP_INTERMEDIATE_SIZE, 1);
        d.swiglu_inplace(&mut cb.p2_g0, swiglu_tmp, CP_INTERMEDIATE_SIZE);
        d.swiglu_inplace(&mut cb.dec_gate, swiglu_tmp, CP_INTERMEDIATE_SIZE);
        cp_ilv2(
            &mut cb.p2_gi2,
            &cb.p2_g0[..CP_INTERMEDIATE_SIZE],
            &cb.dec_gate[..CP_INTERMEDIATE_SIZE],
            CP_INTERMEDIATE_SIZE,
        );

        cp_mm2(
            runtime,
            l.down_q.as_ref().expect("quantized CP down"),
            &cb.p2_gi2,
            &mut cb.p2_d2,
            CP_HIDDEN_SIZE,
            CP_INTERMEDIATE_SIZE,
        );
        for i in 0..CP_HIDDEN_SIZE {
            cb.dec_x[i] += cb.p2_d2[2 * i];
            cb.x1in[i] += cb.p2_d2[2 * i + 1];
        }
    }
    scratch.cp_kv.len = 2;
}

fn cp_transformer_step(
    runtime: &ComputeRuntime,
    model: &Model06B,
    scratch: &mut SessionScratch,
    pos: usize,
    cp_roughness: f32,
) {
    let eps = model.config.rms_norm_eps;
    let attn_scale = 1.0f32 / (CP_HEAD_DIM as f32).sqrt();
    let d = runtime.dispatch;
    let pool = &runtime.pool;
    let kv_max = scratch.cp_kv.max_pos;

    let cb = &mut scratch.cp_bufs;
    let swiglu_tmp = &mut scratch.talker_bufs.swiglu_tmp[..CP_INTERMEDIATE_SIZE];
    let x_norm = &mut cb.dec_ffn_out;
    let mut proj = [0.0f32; CP_HIDDEN_SIZE];
    let mut proj_q2 = [0.0f32; CP_HIDDEN_SIZE];

    d.rms_norm(
        x_norm,
        &cb.dec_x,
        &model.cp.layers[0].input_norm,
        1,
        CP_HIDDEN_SIZE,
        eps,
    );

    for layer_idx in 0..CP_NUM_LAYERS {
        let l = &model.cp.layers[layer_idx];

        cp_dispatch_qkv(
            runtime,
            l,
            x_norm,
            &mut cb.dec_q,
            &mut cb.dec_k,
            &mut cb.dec_v,
        );

        d.rms_norm_per_head(&mut cb.dec_q, &l.q_norm, 1, CP_NUM_HEADS, CP_HEAD_DIM, eps);
        d.rms_norm_per_head(&mut cb.dec_k, &l.k_norm, 1, CP_NUM_KV_HEADS, CP_HEAD_DIM, eps);

        d.apply_rope_neox_inplace(
            &mut cb.dec_q,
            CP_NUM_HEADS,
            CP_HEAD_DIM,
            &model.cp.rope_cos,
            &model.cp.rope_sin,
            pos,
        );
        d.apply_rope_neox_inplace(
            &mut cb.dec_k,
            CP_NUM_KV_HEADS,
            CP_HEAD_DIM,
            &model.cp.rope_cos,
            &model.cp.rope_sin,
            pos,
        );

        let layer_base = layer_idx * kv_max * CP_KV_DIM;
        let kv_off = layer_base + pos * CP_KV_DIM;
        d.f32_to_bf16_vec(&mut scratch.cp_kv.k[kv_off..kv_off + CP_KV_DIM], &cb.dec_k);
        d.f32_to_bf16_vec(&mut scratch.cp_kv.v[kv_off..kv_off + CP_KV_DIM], &cb.dec_v);

        let layer_k = &scratch.cp_kv.k[layer_base..layer_base + (pos + 1) * CP_KV_DIM];
        let layer_v = &scratch.cp_kv.v[layer_base..layer_base + (pos + 1) * CP_KV_DIM];
        d.causal_attention_bf16kv(
            &mut cb.dec_attn_out,
            &cb.dec_q,
            layer_k,
            layer_v,
            1,
            pos + 1,
            CP_NUM_HEADS,
            CP_NUM_KV_HEADS,
            CP_HEAD_DIM,
            attn_scale,
            pos,
        );

        cp_dispatch_proj(
            runtime,
            l.wo_q.as_ref(),
            &l.wo_bf16,
            &cb.dec_attn_out,
            &mut proj,
            CP_HIDDEN_SIZE,
            CP_Q_DIM,
        );

        d.rms_norm_residual(
            x_norm,
            &mut cb.dec_x,
            &proj,
            &l.post_attn_norm,
            CP_HIDDEN_SIZE,
            eps,
        );

        match &l.gate_up_q {
            Some(QuantWeight::Q4_0(w)) => d.matvec_q4_0(
                pool,
                &mut cb.dec_gate,
                w,
                x_norm,
                2 * CP_INTERMEDIATE_SIZE,
                CP_HIDDEN_SIZE,
            ),
            Some(QuantWeight::Int8 { data, scale }) => d.matvec_int8(
                pool,
                &mut cb.dec_gate,
                data,
                scale,
                x_norm,
                2 * CP_INTERMEDIATE_SIZE,
                CP_HIDDEN_SIZE,
            ),
            None => {
                let gu = l
                    .gate_up_fused_bf16
                    .as_deref()
                    .expect("CP gate_up_fused_bf16 present in BF16 mode");
                d.matvec_bf16(
                    pool,
                    &mut cb.dec_gate,
                    gu,
                    x_norm,
                    2 * CP_INTERMEDIATE_SIZE,
                    CP_HIDDEN_SIZE,
                );
            }
        }
        d.swiglu_inplace(&mut cb.dec_gate, swiglu_tmp, CP_INTERMEDIATE_SIZE);

        cp_dispatch_proj(
            runtime,
            l.down_q.as_ref(),
            &l.down_bf16,
            &cb.dec_gate[..CP_INTERMEDIATE_SIZE],
            &mut proj,
            CP_HIDDEN_SIZE,
            CP_INTERMEDIATE_SIZE,
        );

        if cp_roughness > 0.0 {
            if let Some(rough_layers) = &cb.down_q2_rough {
                d.matvec_q2_0(
                    &mut proj_q2,
                    &rough_layers[layer_idx],
                    &cb.dec_gate[..CP_INTERMEDIATE_SIZE],
                    CP_HIDDEN_SIZE,
                    CP_INTERMEDIATE_SIZE,
                );
                let r = cp_roughness;
                for i in 0..CP_HIDDEN_SIZE {
                    proj[i] = (1.0 - r) * proj[i] + r * proj_q2[i];
                }
            }
        }

        if layer_idx + 1 < CP_NUM_LAYERS {
            d.rms_norm_residual(
                x_norm,
                &mut cb.dec_x,
                &proj,
                &model.cp.layers[layer_idx + 1].input_norm,
                CP_HIDDEN_SIZE,
                eps,
            );
        } else {
            d.add_inplace(&mut cb.dec_x, &proj);
        }
    }
    scratch.cp_kv.len = pos + 1;
}

pub fn cp_predict(
    runtime: &ComputeRuntime,
    model: &Model06B,
    voice: &VoiceBindings,
    scratch: &mut SessionScratch,
    talker_hidden: &[f32],
    code0: i32,
    cp_roughness: f32,
    out_codes: &mut [i32; 15],
) {
    let eps = model.config.rms_norm_eps;
    let d = runtime.dispatch;

    if cp_roughness > 0.0 && scratch.cp_bufs.down_q2_rough.is_none() {
        let mut q2_layers = Vec::with_capacity(CP_NUM_LAYERS);
        for l in &model.cp.layers {
            q2_layers.push(quantize_bf16_to_q2_0(
                &l.down_bf16,
                CP_HIDDEN_SIZE,
                CP_INTERMEDIATE_SIZE,
            ));
        }
        scratch.cp_bufs.down_q2_rough = Some(q2_layers);
    }

    scratch.cp_kv.len = 0;

    scratch.cp_bufs.dec_x[..CP_HIDDEN_SIZE].copy_from_slice(&talker_hidden[..CP_HIDDEN_SIZE]);
    if code0 >= 0 && (code0 as usize) < CODEC_VOCAB_SIZE {
        lookup_codec_embed(d, model, voice, code0, &mut scratch.cp_bufs.x1in);
    } else {
        scratch.cp_bufs.x1in.fill_zero();
    }

    if cp_roughness <= 0.0 && model.cp.all_quantized {
        cp_prefill2(runtime, model, scratch);
        let cb = &mut scratch.cp_bufs;
        cb.dec_x[..CP_HIDDEN_SIZE].copy_from_slice(&cb.x1in[..CP_HIDDEN_SIZE]);
    } else {
        cp_transformer_step(runtime, model, scratch, 0, cp_roughness);
        let cb = &mut scratch.cp_bufs;
        cb.dec_x[..CP_HIDDEN_SIZE].copy_from_slice(&cb.x1in[..CP_HIDDEN_SIZE]);
        cp_transformer_step(runtime, model, scratch, 1, cp_roughness);
    }

    let mut normed = [0.0f32; CP_HIDDEN_SIZE];
    d.rms_norm(
        &mut normed,
        &scratch.cp_bufs.dec_x,
        &model.cp.norm,
        1,
        CP_HIDDEN_SIZE,
        eps,
    );
    out_codes[0] = cp_lm_argmax(runtime, &model.cp.lm_heads[0], &normed) as i32;

    for g in 1..15 {
        let prev_code = out_codes[g - 1];
        let pos = g + 1;
        if prev_code >= 0 && (prev_code as usize) < CODEBOOK_SIZE {
            let off = (prev_code as usize) * HIDDEN_SIZE;
            let row = &model.cp.codec_emb_bf16[g - 1][off..off + HIDDEN_SIZE];
            d.bf16_to_f32_vec(&mut scratch.cp_bufs.dec_x, row);
        } else {
            scratch.cp_bufs.dec_x.fill_zero();
        }
        cp_transformer_step(runtime, model, scratch, pos, cp_roughness);
        d.rms_norm(
            &mut normed,
            &scratch.cp_bufs.dec_x,
            &model.cp.norm,
            1,
            CP_HIDDEN_SIZE,
            eps,
        );
        out_codes[g] = cp_lm_argmax(runtime, &model.cp.lm_heads[g], &normed) as i32;
    }
}
