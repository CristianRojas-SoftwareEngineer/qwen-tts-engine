// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use qwen_core::config::{
    CP_HEAD_DIM, CP_HIDDEN_SIZE, CP_INTERMEDIATE_SIZE, CP_KV_DIM, CP_NUM_HEADS, CP_NUM_KV_HEADS,
    CP_NUM_LAYERS, CP_Q_DIM, HIDDEN_SIZE, RMS_NORM_EPS,
};
use qwen_kernels::{AlignedBuf, ComputeRuntime, KernelDispatch};

use crate::model::{int8_qkv, lookup_codec_embed, Model06B, SessionScratch};

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

/// Pasada de 2 posiciones consecutivas del Code Predictor (5 capas) que lee cada matriz de
/// pesos una sola vez para ambas columnas (`matmat` con `B = 2`).
///
/// Entradas: la columna 0 (posición `pos`) en `cb.dec_x` y la columna 1 (posición `pos + 1`)
/// en `cb.x1in`. Salidas: los estados residuales finales (sin la norma final) en `cb.dec_x` y
/// `cb.x1in`, y `cp_kv.len = pos + 2`. Es bit-idéntica a dos `cp_transformer_step`
/// consecutivos en `pos` y `pos + 1`.
fn cp_step2(runtime: &ComputeRuntime, model: &Model06B, scratch: &mut SessionScratch, pos: usize) {
    debug_assert!(pos + 2 <= scratch.cp_kv.max_pos);
    let eps = RMS_NORM_EPS;
    let attn_scale = 1.0f32 / (CP_HEAD_DIM as f32).sqrt();
    let d = runtime.dispatch;
    let pool = &runtime.pool;
    let kv_max = scratch.cp_kv.max_pos;
    let cb = &mut scratch.cp_bufs;
    let swiglu_tmp = &mut scratch.talker_bufs.swiglu_tmp[..CP_INTERMEDIATE_SIZE];

    for (layer_idx, l) in model.cp.layers.iter().enumerate() {
        d.rms_norm(&mut cb.p2_xn0, &cb.dec_x, &l.input_norm, 1, CP_HIDDEN_SIZE, eps);
        d.rms_norm(&mut cb.p2_xn1, &cb.x1in, &l.input_norm, 1, CP_HIDDEN_SIZE, eps);
        cp_ilv2(&mut cb.p2_x2, &cb.p2_xn0, &cb.p2_xn1, CP_HIDDEN_SIZE);

        d.matmat_int8(
            pool,
            &mut cb.p2_q2,
            &l.wq.data,
            &l.wq.scale,
            &cb.p2_x2,
            CP_Q_DIM,
            CP_HIDDEN_SIZE,
            2,
        );
        d.matmat_int8(
            pool,
            &mut cb.p2_k2,
            &l.wk.data,
            &l.wk.scale,
            &cb.p2_x2,
            CP_KV_DIM,
            CP_HIDDEN_SIZE,
            2,
        );
        d.matmat_int8(
            pool,
            &mut cb.p2_v2,
            &l.wv.data,
            &l.wv.scale,
            &cb.p2_x2,
            CP_KV_DIM,
            CP_HIDDEN_SIZE,
            2,
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
        d.matmat_int8(
            pool,
            &mut cb.p2_p2,
            &l.wo.data,
            &l.wo.scale,
            &cb.p2_a2,
            CP_HIDDEN_SIZE,
            CP_Q_DIM,
            2,
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

        d.matmat_q4_0(
            pool,
            &mut cb.p2_g2,
            &l.gate_up,
            &cb.p2_x2,
            2 * CP_INTERMEDIATE_SIZE,
            CP_HIDDEN_SIZE,
            2,
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

        d.matmat_q4_0(
            pool,
            &mut cb.p2_d2,
            &l.down,
            &cb.p2_gi2,
            CP_HIDDEN_SIZE,
            CP_INTERMEDIATE_SIZE,
            2,
        );
        for i in 0..CP_HIDDEN_SIZE {
            cb.dec_x[i] += cb.p2_d2[2 * i];
            cb.x1in[i] += cb.p2_d2[2 * i + 1];
        }
    }
    scratch.cp_kv.len = pos + 2;
}

/// Carga en `dst` el embedding del código `code` de la tabla `table` del CP.
fn cp_code_embed(
    d: &KernelDispatch,
    model: &Model06B,
    table: usize,
    code: i32,
    dst: &mut AlignedBuf<f32>,
) {
    let off = code as usize * HIDDEN_SIZE;
    d.bf16_to_f32_vec(dst, &model.cp.codec_emb_bf16[table][off..off + HIDDEN_SIZE]);
}

fn cp_transformer_step(
    runtime: &ComputeRuntime,
    model: &Model06B,
    scratch: &mut SessionScratch,
    pos: usize,
) {
    let eps = RMS_NORM_EPS;
    let attn_scale = 1.0f32 / (CP_HEAD_DIM as f32).sqrt();
    let d = runtime.dispatch;
    let pool = &runtime.pool;
    let kv_max = scratch.cp_kv.max_pos;

    let cb = &mut scratch.cp_bufs;
    let swiglu_tmp = &mut scratch.talker_bufs.swiglu_tmp[..CP_INTERMEDIATE_SIZE];
    let x_norm = &mut cb.dec_ffn_out;
    let mut proj = [0.0f32; CP_HIDDEN_SIZE];

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

        int8_qkv(
            runtime,
            &l.wq,
            &l.wk,
            &l.wv,
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

        d.matvec_int8(
            pool,
            &mut proj,
            &l.wo.data,
            &l.wo.scale,
            &cb.dec_attn_out,
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

        d.matvec_q4_0(
            pool,
            &mut cb.dec_gate,
            &l.gate_up,
            x_norm,
            2 * CP_INTERMEDIATE_SIZE,
            CP_HIDDEN_SIZE,
        );
        d.swiglu_inplace(&mut cb.dec_gate, swiglu_tmp, CP_INTERMEDIATE_SIZE);

        d.matvec_q4_0(
            pool,
            &mut proj,
            &l.down,
            &cb.dec_gate[..CP_INTERMEDIATE_SIZE],
            CP_HIDDEN_SIZE,
            CP_INTERMEDIATE_SIZE,
        );

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
/// Las posiciones 0 y 1 van juntas en una pasada de 2 posiciones
/// (`cp_step2`). Las siguientes se calculan con verificación especulativa de 1 adivinanza: tras
/// obtener `c[g]` se adivina `c[g + 1]` con la cabeza siguiente sobre el estado actual y una
/// pasada de 2 posiciones calcula el código real y, si coincide con la adivinanza, el siguiente.
/// Cada código sigue siendo el `argmax` de la misma cabeza sobre el mismo estado, así que la
/// salida es bit-idéntica a la ejecución secuencial.
pub fn cp_predict(
    runtime: &ComputeRuntime,
    model: &Model06B,
    scratch: &mut SessionScratch,
    talker_hidden: &[f32],
    code0: i32,
    out_codes: &mut [i32; 15],
) {
    let eps = RMS_NORM_EPS;
    let d = runtime.dispatch;

    scratch.cp_kv.len = 0;

    scratch.cp_bufs.dec_x[..CP_HIDDEN_SIZE].copy_from_slice(&talker_hidden[..CP_HIDDEN_SIZE]);
    lookup_codec_embed(d, model, code0, &mut scratch.cp_bufs.x1in);

    cp_step2(runtime, model, scratch, 0);
    let cb = &mut scratch.cp_bufs;
    cb.dec_x[..CP_HIDDEN_SIZE].copy_from_slice(&cb.x1in[..CP_HIDDEN_SIZE]);

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

    // Invariante: out_codes[0..=g] calculados; `normed` = h de la posición g + 1.
    let mut g = 0usize;
    while g < 14 {
        let pos = g + 2; // posición cuya entrada es el embedding de out_codes[g]
        if g + 2 <= 14 {
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
            cp_transformer_step(runtime, model, scratch, pos);
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
    use std::path::PathBuf;
    use qwen_core::config::CODEBOOK_SIZE;

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
            true,
        )
        .expect("carga del modelo");
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
            cp_transformer_step(runtime, model, scratch, p);
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
                        "{name} difiere: capa {layer}, posición {}, índice {} (bits {u:#06x} vs {w:#06x})",
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
                "{name} difiere en el índice {i} (bits {:#010x} vs {:#010x})",
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

            // Camino A: dos pasos de 1 posición.
            let mut sa = SessionScratch::new();
            build_prefix(&runtime, &model, &mut sa, 77, pos);
            cp_code_embed(d, &model, 0, c0, &mut sa.cp_bufs.dec_x);
            cp_transformer_step(&runtime, &model, &mut sa, pos);
            let xa0: Vec<f32> = sa.cp_bufs.dec_x[..CP_HIDDEN_SIZE].to_vec();
            cp_code_embed(d, &model, 1, c1, &mut sa.cp_bufs.dec_x);
            cp_transformer_step(&runtime, &model, &mut sa, pos + 1);
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

        // Camino B: pasada de 2 posiciones, rechazo y paso de 1 posición con otra entrada.
        let mut sb = SessionScratch::new();
        build_prefix(&runtime, &model, &mut sb, 91, pos);
        cp_code_embed(d, &model, 0, c0, &mut sb.cp_bufs.dec_x);
        cp_code_embed(d, &model, 1, c1, &mut sb.cp_bufs.x1in);
        cp_step2(&runtime, &model, &mut sb, pos);
        sb.cp_kv.len = pos + 1;
        cp_code_embed(d, &model, 1, c1b, &mut sb.cp_bufs.dec_x);
        cp_transformer_step(&runtime, &model, &mut sb, pos + 1);

        // Referencia: dos pasos de 1 posición con la entrada real.
        let mut sr = SessionScratch::new();
        build_prefix(&runtime, &model, &mut sr, 91, pos);
        cp_code_embed(d, &model, 0, c0, &mut sr.cp_bufs.dec_x);
        cp_transformer_step(&runtime, &model, &mut sr, pos);
        cp_code_embed(d, &model, 1, c1b, &mut sr.cp_bufs.dec_x);
        cp_transformer_step(&runtime, &model, &mut sr, pos + 1);

        assert_bits_equal(
            "dec_x",
            &sb.cp_bufs.dec_x[..CP_HIDDEN_SIZE],
            &sr.cp_bufs.dec_x[..CP_HIDDEN_SIZE],
        );
        assert_kv_equal(&sr, &sb, pos + 2);
    }

    /// Referencia secuencial: devuelve los 15 códigos y cuántas adivinanzas habrían acertado.
    fn sequential_reference(
        runtime: &ComputeRuntime,
        model: &Model06B,
        hidden: &[f32],
        code0: i32,
    ) -> ([i32; 15], usize) {
        let d = runtime.dispatch;
        let eps = RMS_NORM_EPS;
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
            cp_transformer_step(runtime, model, &mut scratch, g + 1);
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
                &mut got,
            );
            assert_eq!(got, expected, "par {pair}: códigos distintos");
        }
        eprintln!("K1TEST aciertos posibles: {total_hits} de {total_guesses} adivinanzas");
    }
}
