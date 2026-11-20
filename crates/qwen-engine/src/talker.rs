// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use qwen_core::config::{
    HEAD_DIM, HIDDEN_SIZE, INTERMEDIATE_SIZE, KV_DIM, NUM_HEADS, NUM_KV_HEADS, NUM_LAYERS, Q_DIM,
};
use qwen_kernels::{AlignedBuf, ComputeRuntime, Transpose};

use crate::error::EngineError;
use crate::model::{Model06B, SessionScratch, TransformerLayerWeights};
use crate::request::LogLevel;

/// Elementos F32 del bloque de pesos convertidos del *prefill* (4 MiB).
///
/// Cabe en la L3 (16 MiB en la CPU de referencia) junto a las activaciones, así `sgemm`
/// relee los pesos convertidos desde caché y no desde la RAM. Un bloque del tamaño de la
/// L2 multiplicaría las llamadas a `sgemm` y al pool (~3 400 por *prefill* frente a ~420).
const PREFILL_STAGE_ELEMS: usize = 1 << 20;

/// Pesos BF16 de una proyección del *prefill*, fila a fila (`rows × k`).
#[derive(Clone, Copy)]
enum Bf16Rows<'a> {
    /// Matriz contigua.
    Plain(&'a [u16]),
    /// `gate` y `up` intercaladas por filas (`gate_0`, `up_0`, `gate_1`, …).
    GateUp { gate: &'a [u16], up: &'a [u16] },
}

/// `out[s, n] = Σ_k x[s, k] · W[n, k]` para las `rows` filas de `W` (BF16); `out` es `seq_len × rows`.
///
/// Convierte `W` a F32 por bloques de filas en `stage` y lanza `sgemm` sobre cada bloque:
/// los pesos F32 nunca existen completos en memoria. OpenBLAS (0.3.34) redondea distinto
/// según el ancho del bloque, así que el tamaño de bloque forma parte del contrato numérico.
#[allow(clippy::too_many_arguments)]
fn prefill_matmul_bf16(
    runtime: &ComputeRuntime,
    stage: &mut [f32],
    x: &[f32],
    seq_len: usize,
    k: usize,
    w: Bf16Rows<'_>,
    rows: usize,
    out: &mut [f32],
) {
    // Par, para que cada bloque de `GateUp` lleve parejas completas.
    let block_rows = (stage.len() / k) & !1;
    assert!(
        block_rows > 0,
        "prefill_matmul_bf16: etapa demasiado pequeña para k={k}"
    );
    let mut r0 = 0;
    while r0 < rows {
        let r1 = (r0 + block_rows).min(rows);
        let st = &mut stage[..(r1 - r0) * k];
        match w {
            Bf16Rows::Plain(src) => {
                runtime.dispatch.bf16_to_f32_par(&runtime.pool, st, &src[r0 * k..r1 * k]);
            }
            Bf16Rows::GateUp { gate, up } => {
                let (p0, p1) = (r0 / 2, r1 / 2);
                runtime.dispatch.gate_up_to_f32_par(
                    &runtime.pool,
                    st,
                    &gate[p0 * k..p1 * k],
                    &up[p0 * k..p1 * k],
                    p1 - p0,
                    k,
                );
            }
        }
        runtime.blas.sgemm(
            Transpose::NoTrans,
            Transpose::Trans,
            seq_len,
            r1 - r0,
            k,
            1.0,
            x,
            k,
            st,
            k,
            0.0,
            &mut out[r0..],
            rows,
        );
        r0 = r1;
    }
}

/// Ejecuta el *prefill* completo del Talker (`kv_len = 0`) sobre `input_embeds[seq_len, HIDDEN_SIZE]`.
pub fn talker_prefill(
    runtime: &ComputeRuntime,
    model: &Model06B,
    scratch: &mut SessionScratch,
    input_embeds: &[f32],
    seq_len: usize,
    log: LogLevel,
) -> Result<(), EngineError> {
    let eps = model.config.rms_norm_eps;
    let d = runtime.dispatch;

    if !log.is_silent() {
        eprintln!("  Prefill: {} tokens, hidden={}", seq_len, HIDDEN_SIZE);
    }
    if seq_len > model.talker.rope_cos.len() / (HEAD_DIM / 2) {
        return Err(EngineError::Runtime(format!(
            "Prompt demasiado largo ({seq_len} tokens > caché RoPE)"
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

    // Bloque de pesos convertidos (4 MiB), transitorio: se libera al salir del prefill.
    let mut stage_buf = AlignedBuf::<f32>::new_zeroed(PREFILL_STAGE_ELEMS);
    let stage = &mut stage_buf[..];

    let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
    let kv_max = scratch.talker_kv.max_pos;

    for (layer_idx, l) in model.talker.layers.iter().enumerate() {
        d.rms_norm(pref_x_norm, residual, &l.input_norm, seq_len, HIDDEN_SIZE, eps);

        let x = &*pref_x_norm;
        let h = HIDDEN_SIZE;
        prefill_matmul_bf16(runtime, stage, x, seq_len, h, Bf16Rows::Plain(&l.wq_bf16), Q_DIM, pref_q);
        prefill_matmul_bf16(runtime, stage, x, seq_len, h, Bf16Rows::Plain(&l.wk_bf16), KV_DIM, pref_k);
        prefill_matmul_bf16(runtime, stage, x, seq_len, h, Bf16Rows::Plain(&l.wv_bf16), KV_DIM, pref_v);

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

        prefill_matmul_bf16(
            runtime,
            stage,
            pref_attn_out,
            seq_len,
            Q_DIM,
            Bf16Rows::Plain(&l.wo_bf16),
            HIDDEN_SIZE,
            pref_proj,
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

        prefill_matmul_bf16(
            runtime,
            stage,
            pref_x_norm,
            seq_len,
            HIDDEN_SIZE,
            Bf16Rows::GateUp { gate: &l.gate_bf16, up: &l.up_bf16 },
            2 * INTERMEDIATE_SIZE,
            pref_gate,
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

        prefill_matmul_bf16(
            runtime,
            stage,
            &pref_gate[..seq_len * INTERMEDIATE_SIZE],
            seq_len,
            INTERMEDIATE_SIZE,
            Bf16Rows::Plain(&l.down_bf16),
            HIDDEN_SIZE,
            pref_proj,
        );
        d.add_inplace(residual, pref_proj);

        if log.is_debug() {
            eprintln!(
                "  Capa {}/{} terminada res[:4]=[{:.4},{:.4},{:.4},{:.4}]",
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

    if log.is_debug() {
        eprint!("[PREFILL] last_hidden[:8]:");
        for v in &tb.dec_x[..8] {
            eprint!(" {:.6}", v);
        }
        eprintln!();
    }

    if !log.is_silent() {
        eprintln!("  Prefill completo ({seq_len} tokens en la caché KV)");
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
