// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use qwen_core::config::{
    CODEBOOK_SIZE, CODEC_VOCAB_SIZE, CP_HEAD_DIM, CP_HIDDEN_SIZE, CP_INTERMEDIATE_SIZE,
    CP_KV_DIM, CP_NUM_HEADS, CP_NUM_KV_HEADS, CP_NUM_LAYERS, CP_Q_DIM, HIDDEN_SIZE,
};
use qwen_kernels::quant::quantize_bf16_to_q2_0;
use qwen_kernels::{AlignedBuf, ComputeRuntime, KernelDispatch};

use crate::model::{
    lookup_codec_embed, Model06B, QuantWeight, SessionScratch, TransformerLayerWeights,
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

/// Pasada de 2 posiciones consecutivas del Code Predictor (5 capas) que lee cada matriz de
/// pesos una sola vez para ambas columnas (`matmat` con `B = 2`).
///
/// Entradas: la columna 0 (posición `pos`) en `cb.dec_x` y la columna 1 (posición `pos + 1`)
/// en `cb.x1in`. Salidas: los estados residuales finales (sin la norma final) en `cb.dec_x` y
/// `cb.x1in`, y `cp_kv.len = pos + 2`. Es bit-idéntica a dos `cp_transformer_step`
/// consecutivos en `pos` y `pos + 1`.
fn cp_step2(runtime: &ComputeRuntime, model: &Model06B, scratch: &mut SessionScratch, pos: usize) {
    debug_assert!(pos + 2 <= scratch.cp_kv.max_pos);
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
                pos + p,
            );
            d.apply_rope_neox_inplace(
                &mut cb.dec_k,
                CP_NUM_KV_HEADS,
                CP_HEAD_DIM,
                &model.cp.rope_cos,
                &model.cp.rope_sin,
                pos + p,
            );

            let kv_off = layer_base + (pos + p) * CP_KV_DIM;
            d.f32_to_bf16_vec(&mut scratch.cp_kv.k[kv_off..kv_off + CP_KV_DIM], &cb.dec_k);
            d.f32_to_bf16_vec(&mut scratch.cp_kv.v[kv_off..kv_off + CP_KV_DIM], &cb.dec_v);

            let layer_k = &scratch.cp_kv.k[layer_base..layer_base + (pos + p + 1) * CP_KV_DIM];
            let layer_v = &scratch.cp_kv.v[layer_base..layer_base + (pos + p + 1) * CP_KV_DIM];
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
                pos + p + 1,
                CP_NUM_HEADS,
                CP_NUM_KV_HEADS,
                CP_HEAD_DIM,
                attn_scale,
                pos + p,
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
    scratch.cp_kv.len = pos + 2;
}

/// Carga en `dst` el embedding del código `code` de la tabla `table` del CP (ceros si el código no es válido).
fn cp_code_embed(
    d: &KernelDispatch,
    model: &Model06B,
    table: usize,
    code: i32,
    dst: &mut AlignedBuf<f32>,
) {
    if code >= 0 && (code as usize) < CODEBOOK_SIZE {
        let off = (code as usize) * HIDDEN_SIZE;
        d.bf16_to_f32_vec(dst, &model.cp.codec_emb_bf16[table][off..off + HIDDEN_SIZE]);
    } else {
        dst.fill_zero();
    }
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

        l.dispatch_qkv(
            runtime,
            x_norm,
            &mut cb.dec_q,
            &mut cb.dec_k,
            &mut cb.dec_v,
            CP_HIDDEN_SIZE,
            CP_Q_DIM,
            CP_KV_DIM,
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

        TransformerLayerWeights::dispatch_proj(
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

        if l.dispatch_gate_up(
            runtime,
            x_norm,
            &mut cb.dec_gate,
            CP_INTERMEDIATE_SIZE,
            CP_HIDDEN_SIZE,
        )
        .is_err()
        {
            unreachable!("CP gate_up_fused_bf16 must exist in BF16 mode");
        }
        d.swiglu_inplace(&mut cb.dec_gate, swiglu_tmp, CP_INTERMEDIATE_SIZE);

        TransformerLayerWeights::dispatch_proj(
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

/// Predice los 15 codebooks acústicos (`out_codes[0..15]`) en el Code Predictor de 5 capas
/// a partir de `talker_hidden` y `code0`.
///
/// Con las capas cuantizadas, las posiciones 0 y 1 van juntas en una pasada de 2 posiciones
/// (`cp_step2`). Las siguientes se calculan con verificación especulativa de 1 adivinanza: tras
/// obtener `c[g]` se adivina `c[g + 1]` con la cabeza siguiente sobre el estado actual y una
/// pasada de 2 posiciones calcula el código real y, si coincide con la adivinanza, el siguiente.
/// Cada código sigue siendo el `argmax` de la misma cabeza sobre el mismo estado, así que la
/// salida es bit-idéntica a la ejecución secuencial.
#[allow(clippy::too_many_arguments)]
pub fn cp_predict(
    runtime: &ComputeRuntime,
    model: &Model06B,
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
        lookup_codec_embed(d, model, code0, &mut scratch.cp_bufs.x1in);
    } else {
        scratch.cp_bufs.x1in.fill_zero();
    }

    if cp_roughness <= 0.0 && model.cp.all_quantized {
        cp_step2(runtime, model, scratch, 0);
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
    out_codes[0] = model.cp.lm_heads[0].argmax(runtime, &normed) as i32;

    let fast = cp_roughness <= 0.0 && model.cp.all_quantized;
    // Invariante: out_codes[0..=g] calculados; `normed` = h de la posición g + 1.
    let mut g = 0usize;
    while g < 14 {
        let pos = g + 2; // posición cuya entrada es el embedding de out_codes[g]
        if fast && g + 2 <= 14 {
            let guess = model.cp.lm_heads[g + 1].argmax(runtime, &normed) as i32;
            cp_code_embed(d, model, g, out_codes[g], &mut scratch.cp_bufs.dec_x);
            cp_code_embed(d, model, g + 1, guess, &mut scratch.cp_bufs.x1in);
            cp_step2(runtime, model, scratch, pos);
            d.rms_norm(
                &mut normed,
                &scratch.cp_bufs.dec_x,
                &model.cp.norm,
                1,
                CP_HIDDEN_SIZE,
                eps,
            );
            out_codes[g + 1] = model.cp.lm_heads[g + 1].argmax(runtime, &normed) as i32;
            if out_codes[g + 1] == guess {
                d.rms_norm(
                    &mut normed,
                    &scratch.cp_bufs.x1in,
                    &model.cp.norm,
                    1,
                    CP_HIDDEN_SIZE,
                    eps,
                );
                out_codes[g + 2] = model.cp.lm_heads[g + 2].argmax(runtime, &normed) as i32;
                g += 2;
            } else {
                scratch.cp_kv.len = pos + 1; // se descarta la columna 1
                g += 1;
            }
        } else {
            cp_code_embed(d, model, g, out_codes[g], &mut scratch.cp_bufs.dec_x);
            cp_transformer_step(runtime, model, scratch, pos, cp_roughness);
            d.rms_norm(
                &mut normed,
                &scratch.cp_bufs.dec_x,
                &model.cp.norm,
                1,
                CP_HIDDEN_SIZE,
                eps,
            );
            out_codes[g + 1] = model.cp.lm_heads[g + 1].argmax(runtime, &normed) as i32;
            g += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelLoadOptions;
    use std::path::PathBuf;

    /// Carga `models/qwen3-tts-0.6b-base`; `None` (y la prueba se omite) si el modelo no está.
    fn load_model() -> Option<(ComputeRuntime, Model06B)> {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/qwen3-tts-0.6b-base");
        if !dir.exists() {
            eprintln!("Se omite la prueba: falta el modelo en {}", dir.display());
            return None;
        }
        let runtime = ComputeRuntime::new(Some(4));
        let model = Model06B::load(
            &dir,
            &runtime,
            ModelLoadOptions {
                quant_mixed: true,
                silent: true,
                ..Default::default()
            },
        )
        .expect("carga del modelo");
        assert!(model.cp.all_quantized);
        Some((runtime, model))
    }

    struct Lcg(u32);

    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            self.0 >> 8
        }

        fn code(&mut self) -> i32 {
            (self.next() % CODEBOOK_SIZE as u32) as i32
        }

        fn hidden(&mut self) -> Vec<f32> {
            (0..CP_HIDDEN_SIZE)
                .map(|_| (self.next() as f32 / 16_777_216.0 - 0.5) * 2.0)
                .collect()
        }
    }

    /// Prepara las posiciones `0..pos` de forma determinista (posiciones 0-1 con `cp_step2` y
    /// `2..pos` con `cp_transformer_step`).
    fn build_prefix(
        runtime: &ComputeRuntime,
        model: &Model06B,
        scratch: &mut SessionScratch,
        seed: u32,
        pos: usize,
    ) {
        let d = runtime.dispatch;
        let mut rng = Lcg(seed);
        let hidden = rng.hidden();
        let code0 = rng.code();
        scratch.cp_kv.len = 0;
        scratch.cp_bufs.dec_x[..CP_HIDDEN_SIZE].copy_from_slice(&hidden);
        lookup_codec_embed(d, model, code0, &mut scratch.cp_bufs.x1in);
        cp_step2(runtime, model, scratch, 0);
        let cb = &mut scratch.cp_bufs;
        cb.dec_x[..CP_HIDDEN_SIZE].copy_from_slice(&cb.x1in[..CP_HIDDEN_SIZE]);
        for p in 2..pos {
            let code = rng.code();
            let table = (p - 2) % model.cp.codec_emb_bf16.len();
            cp_code_embed(d, model, table, code, &mut scratch.cp_bufs.dec_x);
            cp_transformer_step(runtime, model, scratch, p, 0.0);
        }
    }

    fn poison_kv_from(scratch: &mut SessionScratch, pos: usize) {
        let max = scratch.cp_kv.max_pos;
        for layer in 0..CP_NUM_LAYERS {
            let base = layer * max * CP_KV_DIM;
            for i in (base + pos * CP_KV_DIM)..(base + max * CP_KV_DIM) {
                scratch.cp_kv.k[i] = 0x7FC0;
                scratch.cp_kv.v[i] = 0x7FC0;
            }
        }
    }

    fn assert_kv_equal(a: &SessionScratch, b: &SessionScratch, upto: usize) {
        let max = a.cp_kv.max_pos;
        for layer in 0..CP_NUM_LAYERS {
            let base = layer * max * CP_KV_DIM;
            let r = base..base + upto * CP_KV_DIM;
            for (name, x, y) in [
                ("K", &a.cp_kv.k[r.clone()], &b.cp_kv.k[r.clone()]),
                ("V", &a.cp_kv.v[r.clone()], &b.cp_kv.v[r.clone()]),
            ] {
                for (i, (u, w)) in x.iter().zip(y.iter()).enumerate() {
                    assert_eq!(
                        u,
                        w,
                        "{name} difiere: capa {layer}, posicion {}, indice {} (bits {u:#06x} vs {w:#06x})",
                        i / CP_KV_DIM,
                        i % CP_KV_DIM
                    );
                }
            }
        }
    }

    fn assert_bits_equal(name: &str, a: &[f32], b: &[f32]) {
        for (i, (u, w)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                u.to_bits(),
                w.to_bits(),
                "{name} difiere en el indice {i} (bits {:#010x} vs {:#010x})",
                u.to_bits(),
                w.to_bits()
            );
        }
    }

    #[test]
    fn step2_equals_two_steps_at_arbitrary_pos() {
        let Some((runtime, model)) = load_model() else { return };
        let d = runtime.dispatch;
        for pos in [2usize, 9] {
            let mut rng = Lcg(0xABCD_0000 + pos as u32);
            let (c0, c1) = (rng.code(), rng.code());

            // Camino A: dos pasos de 1 posicion.
            let mut sa = SessionScratch::new();
            build_prefix(&runtime, &model, &mut sa, 77, pos);
            cp_code_embed(d, &model, 0, c0, &mut sa.cp_bufs.dec_x);
            cp_transformer_step(&runtime, &model, &mut sa, pos, 0.0);
            let xa0: Vec<f32> = sa.cp_bufs.dec_x[..CP_HIDDEN_SIZE].to_vec();
            cp_code_embed(d, &model, 1, c1, &mut sa.cp_bufs.dec_x);
            cp_transformer_step(&runtime, &model, &mut sa, pos + 1, 0.0);
            let xa1: Vec<f32> = sa.cp_bufs.dec_x[..CP_HIDDEN_SIZE].to_vec();

            // Camino B: una pasada de 2 posiciones, con centinela NaN a partir de `pos`.
            let mut sb = SessionScratch::new();
            build_prefix(&runtime, &model, &mut sb, 77, pos);
            poison_kv_from(&mut sb, pos);
            cp_code_embed(d, &model, 0, c0, &mut sb.cp_bufs.dec_x);
            cp_code_embed(d, &model, 1, c1, &mut sb.cp_bufs.x1in);
            cp_step2(&runtime, &model, &mut sb, pos);
            assert_eq!(sb.cp_kv.len, pos + 2);

            assert_bits_equal("dec_x", &sb.cp_bufs.dec_x[..CP_HIDDEN_SIZE], &xa0);
            assert_bits_equal("x1in", &sb.cp_bufs.x1in[..CP_HIDDEN_SIZE], &xa1);
            assert_kv_equal(&sa, &sb, pos + 2);
        }
    }

    #[test]
    fn rejection_rollback_is_exact() {
        let Some((runtime, model)) = load_model() else { return };
        let d = runtime.dispatch;
        let pos = 9usize;
        let mut rng = Lcg(0x5EED_0001);
        let (c0, c1, c1b) = (rng.code(), rng.code(), rng.code());
        assert_ne!(c1, c1b);

        // Camino B: pasada de 2 posiciones, rechazo y paso de 1 posicion con otra entrada.
        let mut sb = SessionScratch::new();
        build_prefix(&runtime, &model, &mut sb, 91, pos);
        cp_code_embed(d, &model, 0, c0, &mut sb.cp_bufs.dec_x);
        cp_code_embed(d, &model, 1, c1, &mut sb.cp_bufs.x1in);
        cp_step2(&runtime, &model, &mut sb, pos);
        sb.cp_kv.len = pos + 1;
        cp_code_embed(d, &model, 1, c1b, &mut sb.cp_bufs.dec_x);
        cp_transformer_step(&runtime, &model, &mut sb, pos + 1, 0.0);

        // Referencia: dos pasos de 1 posicion con la entrada real.
        let mut sr = SessionScratch::new();
        build_prefix(&runtime, &model, &mut sr, 91, pos);
        cp_code_embed(d, &model, 0, c0, &mut sr.cp_bufs.dec_x);
        cp_transformer_step(&runtime, &model, &mut sr, pos, 0.0);
        cp_code_embed(d, &model, 1, c1b, &mut sr.cp_bufs.dec_x);
        cp_transformer_step(&runtime, &model, &mut sr, pos + 1, 0.0);

        assert_bits_equal(
            "dec_x",
            &sb.cp_bufs.dec_x[..CP_HIDDEN_SIZE],
            &sr.cp_bufs.dec_x[..CP_HIDDEN_SIZE],
        );
        assert_kv_equal(&sr, &sb, pos + 2);
    }

    /// Referencia secuencial: devuelve los 15 codigos y cuantas adivinanzas habrian acertado.
    fn sequential_reference(
        runtime: &ComputeRuntime,
        model: &Model06B,
        hidden: &[f32],
        code0: i32,
    ) -> ([i32; 15], usize) {
        let d = runtime.dispatch;
        let eps = model.config.rms_norm_eps;
        let mut scratch = SessionScratch::new();
        scratch.cp_kv.len = 0;
        scratch.cp_bufs.dec_x[..CP_HIDDEN_SIZE].copy_from_slice(&hidden[..CP_HIDDEN_SIZE]);
        lookup_codec_embed(d, model, code0, &mut scratch.cp_bufs.x1in);
        cp_step2(runtime, model, &mut scratch, 0);
        let cb = &mut scratch.cp_bufs;
        cb.dec_x[..CP_HIDDEN_SIZE].copy_from_slice(&cb.x1in[..CP_HIDDEN_SIZE]);

        let mut codes = [0i32; 15];
        let mut guesses = [-1i32; 15];
        let mut normed = [0.0f32; CP_HIDDEN_SIZE];
        d.rms_norm(
            &mut normed,
            &scratch.cp_bufs.dec_x,
            &model.cp.norm,
            1,
            CP_HIDDEN_SIZE,
            eps,
        );
        codes[0] = model.cp.lm_heads[0].argmax(runtime, &normed) as i32;
        guesses[0] = model.cp.lm_heads[1].argmax(runtime, &normed) as i32;
        for g in 1..15 {
            cp_code_embed(d, model, g - 1, codes[g - 1], &mut scratch.cp_bufs.dec_x);
            cp_transformer_step(runtime, model, &mut scratch, g + 1, 0.0);
            d.rms_norm(
                &mut normed,
                &scratch.cp_bufs.dec_x,
                &model.cp.norm,
                1,
                CP_HIDDEN_SIZE,
                eps,
            );
            codes[g] = model.cp.lm_heads[g].argmax(runtime, &normed) as i32;
            if g <= 12 {
                guesses[g] = model.cp.lm_heads[g + 1].argmax(runtime, &normed) as i32;
            }
        }
        let hits = (0..=12).filter(|&g| guesses[g] == codes[g + 1]).count();
        (codes, hits)
    }

    #[test]
    fn cp_predict_matches_sequential_reference() {
        let Some((runtime, model)) = load_model() else { return };
        let mut rng = Lcg(0xC0DE_1234);
        let (mut total_hits, mut total_guesses) = (0usize, 0usize);
        for pair in 0..16 {
            let hidden = rng.hidden();
            let code0 = rng.code();
            let (expected, hits) = sequential_reference(&runtime, &model, &hidden, code0);
            total_hits += hits;
            total_guesses += 13;

            let mut scratch = SessionScratch::new();
            let mut got = [0i32; 15];
            cp_predict(
                &runtime,
                &model,
                &mut scratch,
                &hidden,
                code0,
                0.0,
                &mut got,
            );
            assert_eq!(got, expected, "par {pair}: codigos distintos");
        }
        eprintln!("K1TEST aciertos posibles: {total_hits} de {total_guesses} adivinanzas");
    }
}
