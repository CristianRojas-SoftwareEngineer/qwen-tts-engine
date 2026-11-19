// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Kernels NEON de Q4_0 e INT8 para aarch64 (invocados exclusivamente mediante `KernelDispatch`).
//!
//! El mismo código se genera en dos variantes que solo cambian el producto escalar de 4 bytes:
//! - [`sdot`]: la instrucción `SDOT` de la extensión `dotprod` (Apple M1 en adelante, Neoverse N1,
//!   Cortex-A76 en adelante);
//! - [`widen`]: multiplicación ampliada `SMULL` y sumas por pares, para núcleos ARMv8.0 sin `dotprod`.
//!
//! Las dos dan los mismos bits: el producto escalar entero es exacto.
//!
//! **Orden de suma: el de los kernels AVX2** (`simd/avx2.rs`). Cada bloque Q4_0 de 32 pesos da
//! 8 sumas enteras de 4 productos; a cada una se le resta la suma de las 32 activaciones del bloque
//! (los pesos se guardan sin el desplazamiento de 8), se convierte a `f32` y se acumula con FMA en
//! 8 carriles `f32` (dos registros de 4). Al final, la reducción horizontal es la de
//! `hsum256_ps`: `((c0 + c4) + (c2 + c6)) + ((c1 + c5) + (c3 + c7))`. Con el mismo orden, NEON da
//! los mismos bits que AVX2. INT8 acumula en enteros, así que coincide bit a bit con la ruta escalar.
//!
//! El `SDOT` se emite con `asm!` porque el intrínseco `vdotq_s32` no es estable en Rust 1.96
//! (`stdarch_neon_dotprod`).
//!
//! # Safety
//! Las funciones de [`sdot`] exigen que la CPU tenga `dotprod` (`CpuCaps::dotprod`); NEON forma
//! parte de la base de aarch64. Los slices recibidos deben cumplir las dimensiones declaradas: las
//! comprueba `KernelDispatch` antes de llamar.

use core::arch::aarch64::*;

use crate::quant::{f16_to_f32, Q4_0Block, Q4_0_BLOCK_SIZE, QWEN_LONG_ROW_COLS, QWEN_QACT_MAX};

/// Columnas de lote que un microkernel `matmat` acumula a la vez en registros (como en AVX2).
const MM_BATCH_CHUNK: usize = 8;
/// Bloques Q4_0 por fila como máximo.
const MAX_Q4_BLOCKS: usize = QWEN_QACT_MAX / Q4_0_BLOCK_SIZE;

/// `acc[i] += a[4i]·b[4i] + … + a[4i+3]·b[4i+3]` con `SDOT`.
#[inline]
#[target_feature(enable = "dotprod")]
unsafe fn dot4_sdot(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    let mut acc = acc;
    core::arch::asm!(
        "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
        acc = inout(vreg) acc,
        a = in(vreg) a,
        b = in(vreg) b,
        options(pure, nomem, nostack, preserves_flags),
    );
    acc
}

/// `acc[i] += a[4i]·b[4i] + … + a[4i+3]·b[4i+3]` con multiplicación ampliada a 16 bits y sumas por
/// pares a 32 bits (sin desbordamiento posible, también con −128).
#[inline]
unsafe fn dot4_widen(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    let lo = vpaddlq_s16(vmull_s8(vget_low_s8(a), vget_low_s8(b)));
    let hi = vpaddlq_s16(vmull_high_s8(a, b));
    vaddq_s32(acc, vpaddq_s32(lo, hi))
}

/// Reducción horizontal de 8 carriles `f32` en el orden de `hsum256_ps` (AVX2).
#[inline]
unsafe fn hsum_f32x8(lo: float32x4_t, hi: float32x4_t) -> f32 {
    let t = vaddq_f32(lo, hi);
    let u = vadd_f32(vget_low_f32(t), vget_high_f32(t));
    vget_lane_f32::<0>(u) + vget_lane_f32::<1>(u)
}

/// Pesos de un bloque Q4_0 sin desplazar (0..15), en el orden de las activaciones:
/// el elemento `2i` es el nibble bajo de `qs[i]` y el `2i + 1`, el alto.
#[inline]
unsafe fn q4_0_weights(blk: *const Q4_0Block) -> (int8x16_t, int8x16_t) {
    let raw = vld1q_u8((*blk).qs.as_ptr());
    let lo = vandq_u8(raw, vdupq_n_u8(0x0F));
    let hi = vshrq_n_u8::<4>(raw);
    (
        vreinterpretq_s8_u8(vzip1q_u8(lo, hi)),
        vreinterpretq_s8_u8(vzip2q_u8(lo, hi)),
    )
}

/// Suma de las 32 activaciones de cada bloque, que corrige el desplazamiento de 8 de los pesos.
#[inline]
unsafe fn q4_0_sumq(qx: *const i8, nb: usize, sumq: &mut [i32]) {
    for (b, slot) in sumq.iter_mut().enumerate().take(nb) {
        let p = qx.add(b * Q4_0_BLOCK_SIZE);
        *slot = vaddlvq_s8(vld1q_s8(p)) as i32 + vaddlvq_s8(vld1q_s8(p.add(16))) as i32;
    }
}

macro_rules! q4_0_int8_kernels {
    (#[$feat:meta] $dot4:ident) => {
        use super::*;

        /// Acumula un bloque Q4_0 ya decodificado (`w0`, `w1`) contra 32 activaciones.
        #[inline]
        #[$feat]
        unsafe fn q4_0_acc(
            acc: &mut [float32x4_t; 2],
            w0: int8x16_t,
            w1: int8x16_t,
            a: *const i8,
            sumq: i32,
            scale: float32x4_t,
        ) {
            let z = vdupq_n_s32(0);
            let s = vdupq_n_s32(sumq);
            let p0 = vsubq_s32($dot4(z, w0, vld1q_s8(a)), s);
            let p1 = vsubq_s32($dot4(z, w1, vld1q_s8(a.add(16))), s);
            acc[0] = vfmaq_f32(acc[0], vcvtq_f32_s32(p0), scale);
            acc[1] = vfmaq_f32(acc[1], vcvtq_f32_s32(p1), scale);
        }

        #[inline]
        #[$feat]
        unsafe fn q4_0_block(
            acc: &mut [float32x4_t; 2],
            blk: *const Q4_0Block,
            a: *const i8,
            sumq: i32,
        ) {
            let (w0, w1) = q4_0_weights(blk);
            let scale = vdupq_n_f32(f16_to_f32((*blk).scale_f16));
            q4_0_acc(acc, w0, w1, a, sumq, scale);
        }

        #[$feat]
        unsafe fn q4_0_rows1(
            y: *mut f32,
            qx: *const i8,
            sx: f32,
            w: *const Q4_0Block,
            nb: usize,
            sumq: &[i32],
            out_dim: usize,
        ) {
            for o in 0..out_dim {
                let r0 = w.add(o * nb);
                let mut s0 = [vdupq_n_f32(0.0); 2];
                for (b, &sq) in sumq.iter().enumerate().take(nb) {
                    q4_0_block(&mut s0, r0.add(b), qx.add(b * Q4_0_BLOCK_SIZE), sq);
                }
                *y.add(o) = hsum_f32x8(s0[0], s0[1]) * sx;
            }
        }

        #[$feat]
        unsafe fn q4_0_matvec_2row(
            y: *mut f32,
            qx: *const i8,
            sx: f32,
            w: *const Q4_0Block,
            cols: usize,
            out_dim: usize,
        ) {
            let nb = cols / Q4_0_BLOCK_SIZE;
            let mut sumq = [0i32; MAX_Q4_BLOCKS];
            q4_0_sumq(qx, nb, &mut sumq[..nb]);
            let mut o = 0usize;
            while o + 1 < out_dim {
                let r0 = w.add(o * nb);
                let r1 = r0.add(nb);
                let mut s0 = [vdupq_n_f32(0.0); 2];
                let mut s1 = [vdupq_n_f32(0.0); 2];
                for (b, &sq) in sumq.iter().enumerate().take(nb) {
                    let a = qx.add(b * Q4_0_BLOCK_SIZE);
                    q4_0_block(&mut s0, r0.add(b), a, sq);
                    q4_0_block(&mut s1, r1.add(b), a, sq);
                }
                *y.add(o) = hsum_f32x8(s0[0], s0[1]) * sx;
                *y.add(o + 1) = hsum_f32x8(s1[0], s1[1]) * sx;
                o += 2;
            }
            if o < out_dim {
                q4_0_rows1(y.add(o), qx, sx, w.add(o * nb), nb, &sumq[..nb], out_dim - o);
            }
        }

        #[$feat]
        unsafe fn q4_0_matvec_1row(
            y: *mut f32,
            qx: *const i8,
            sx: f32,
            w: *const Q4_0Block,
            cols: usize,
            out_dim: usize,
        ) {
            let nb = cols / Q4_0_BLOCK_SIZE;
            let mut sumq = [0i32; MAX_Q4_BLOCKS];
            q4_0_sumq(qx, nb, &mut sumq[..nb]);
            q4_0_rows1(y, qx, sx, w, nb, &sumq[..nb], out_dim);
        }

        /// Una fila de pesos Q4_0 contra `NB` activaciones: cada bloque se decodifica una sola vez y
        /// cada columna se acumula con la misma secuencia que `q4_0_block`, así que el resultado es
        /// bit a bit el de `NB` llamadas a `matvec`.
        #[inline]
        #[allow(clippy::too_many_arguments)]
        #[$feat]
        unsafe fn q4_0_matmat_row<const NB: usize>(
            yr: *mut f32,
            wr: *const Q4_0Block,
            nb: usize,
            qxt: *const i8,
            sx: *const f32,
            sumq: &[[i32; MAX_Q4_BLOCKS]; MM_BATCH_CHUNK],
            cols: usize,
        ) {
            let mut acc = [[vdupq_n_f32(0.0); 2]; NB];
            for blk in 0..nb {
                let bp = wr.add(blk);
                let (w0, w1) = q4_0_weights(bp);
                let scale = vdupq_n_f32(f16_to_f32((*bp).scale_f16));
                for (j, acc_j) in acc.iter_mut().enumerate() {
                    let a = qxt.add(j * cols + blk * Q4_0_BLOCK_SIZE);
                    q4_0_acc(acc_j, w0, w1, a, *sumq[j].get_unchecked(blk), scale);
                }
            }
            for (j, acc_j) in acc.iter().enumerate() {
                *yr.add(j) = hsum_f32x8(acc_j[0], acc_j[1]) * *sx.add(j);
            }
        }

        #[allow(clippy::too_many_arguments)]
        #[$feat]
        unsafe fn q4_0_matmat(
            y: &mut [f32],
            w: &[Q4_0Block],
            qxt: &[i8],
            sx: &[f32],
            r0: usize,
            r1: usize,
            cols: usize,
            b_dim: usize,
        ) {
            let nb = cols / Q4_0_BLOCK_SIZE;
            let mut sumq = [[0i32; MAX_Q4_BLOCKS]; MM_BATCH_CHUNK];
            let mut b0 = 0usize;
            while b0 < b_dim {
                let nbat = (b_dim - b0).min(MM_BATCH_CHUNK);
                let qx = qxt.as_ptr().add(b0 * cols);
                let sxp = sx.as_ptr().add(b0);
                for (j, sq) in sumq.iter_mut().enumerate().take(nbat) {
                    q4_0_sumq(qx.add(j * cols), nb, &mut sq[..nb]);
                }
                for r in r0..r1 {
                    let yr = y.as_mut_ptr().add(r * b_dim + b0);
                    let wr = w.as_ptr().add(r * nb);
                    match nbat {
                        1 => q4_0_matmat_row::<1>(yr, wr, nb, qx, sxp, &sumq, cols),
                        2 => q4_0_matmat_row::<2>(yr, wr, nb, qx, sxp, &sumq, cols),
                        3 => q4_0_matmat_row::<3>(yr, wr, nb, qx, sxp, &sumq, cols),
                        4 => q4_0_matmat_row::<4>(yr, wr, nb, qx, sxp, &sumq, cols),
                        5 => q4_0_matmat_row::<5>(yr, wr, nb, qx, sxp, &sumq, cols),
                        6 => q4_0_matmat_row::<6>(yr, wr, nb, qx, sxp, &sumq, cols),
                        7 => q4_0_matmat_row::<7>(yr, wr, nb, qx, sxp, &sumq, cols),
                        _ => q4_0_matmat_row::<8>(yr, wr, nb, qx, sxp, &sumq, cols),
                    }
                }
                b0 += nbat;
            }
        }

        /// Producto entero de una fila INT8 por las activaciones, con dos acumuladores independientes.
        #[inline]
        #[$feat]
        unsafe fn int8_dot(w: *const i8, qx: *const i8, cols: usize) -> i32 {
            let mut acc0 = vdupq_n_s32(0);
            let mut acc1 = vdupq_n_s32(0);
            let mut k = 0usize;
            while k + 32 <= cols {
                acc0 = $dot4(acc0, vld1q_s8(w.add(k)), vld1q_s8(qx.add(k)));
                acc1 = $dot4(acc1, vld1q_s8(w.add(k + 16)), vld1q_s8(qx.add(k + 16)));
                k += 32;
            }
            if k + 16 <= cols {
                acc0 = $dot4(acc0, vld1q_s8(w.add(k)), vld1q_s8(qx.add(k)));
                k += 16;
            }
            let mut s = vaddvq_s32(vaddq_s32(acc0, acc1));
            while k < cols {
                s += (*w.add(k) as i32) * (*qx.add(k) as i32);
                k += 1;
            }
            s
        }

        #[$feat]
        unsafe fn int8_matvec(
            y: &mut [f32],
            qx: &[i8],
            sx: f32,
            w: &[i8],
            scale: &[f32],
            cols: usize,
            rows: usize,
        ) {
            let (yp, qp, wp) = (y.as_mut_ptr(), qx.as_ptr(), w.as_ptr());
            for (o, &sc) in scale.iter().enumerate().take(rows) {
                let s = int8_dot(wp.add(o * cols), qp, cols);
                *yp.add(o) = (s as f32) * sc * sx;
            }
        }

        /// Una fila INT8 contra `NB` activaciones: cada vector de pesos se lee una sola vez.
        #[inline]
        #[$feat]
        unsafe fn int8_matmat_row<const NB: usize>(
            yr: *mut f32,
            wr: *const i8,
            scale_r: f32,
            qxt: *const i8,
            sx: *const f32,
            cols: usize,
        ) {
            let mut acc = [vdupq_n_s32(0); NB];
            let mut k = 0usize;
            while k < cols {
                let v = vld1q_s8(wr.add(k));
                for (j, acc_j) in acc.iter_mut().enumerate() {
                    *acc_j = $dot4(*acc_j, v, vld1q_s8(qxt.add(j * cols + k)));
                }
                k += 16;
            }
            for (j, acc_j) in acc.iter().enumerate() {
                *yr.add(j) = (vaddvq_s32(*acc_j) as f32) * scale_r * *sx.add(j);
            }
        }

        #[allow(clippy::too_many_arguments)]
        #[$feat]
        unsafe fn int8_matmat(
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
            debug_assert!(cols.is_multiple_of(32));
            let mut b0 = 0usize;
            while b0 < b_dim {
                let nb = (b_dim - b0).min(MM_BATCH_CHUNK);
                let qx = qxt.as_ptr().add(b0 * cols);
                let sxp = sx.as_ptr().add(b0);
                for (r, &sc) in scale.iter().enumerate().take(r1).skip(r0) {
                    let yr = y.as_mut_ptr().add(r * b_dim + b0);
                    let wr = w.as_ptr().add(r * cols);
                    match nb {
                        1 => int8_matmat_row::<1>(yr, wr, sc, qx, sxp, cols),
                        2 => int8_matmat_row::<2>(yr, wr, sc, qx, sxp, cols),
                        3 => int8_matmat_row::<3>(yr, wr, sc, qx, sxp, cols),
                        4 => int8_matmat_row::<4>(yr, wr, sc, qx, sxp, cols),
                        5 => int8_matmat_row::<5>(yr, wr, sc, qx, sxp, cols),
                        6 => int8_matmat_row::<6>(yr, wr, sc, qx, sxp, cols),
                        7 => int8_matmat_row::<7>(yr, wr, sc, qx, sxp, cols),
                        _ => int8_matmat_row::<8>(yr, wr, sc, qx, sxp, cols),
                    }
                }
                b0 += nb;
            }
        }

        // Entradas de la tabla de despacho, con la firma de la ruta escalar.

        pub(crate) fn int8_matvec_slice(
            y: &mut [f32],
            qx: &[i8],
            sx: f32,
            w: &[i8],
            scale: &[f32],
            cols: usize,
            rows: usize,
        ) {
            unsafe { int8_matvec(y, qx, sx, w, scale, cols, rows) }
        }

        #[allow(clippy::too_many_arguments)]
        pub(crate) fn int8_matmat_slice(
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
            unsafe { int8_matmat(y, w, scale, qxt, sx, r0, r1, cols, b_dim) }
        }

        pub(crate) fn q4_0_matvec_slice(
            y: &mut [f32],
            qx: &[i8],
            sx: f32,
            w: &[Q4_0Block],
            cols: usize,
            rows: usize,
        ) {
            let (yp, qp, wp) = (y.as_mut_ptr(), qx.as_ptr(), w.as_ptr());
            unsafe {
                if cols >= QWEN_LONG_ROW_COLS {
                    q4_0_matvec_1row(yp, qp, sx, wp, cols, rows);
                } else {
                    q4_0_matvec_2row(yp, qp, sx, wp, cols, rows);
                }
            }
        }

        /// QKV fusionado: siempre el kernel de 2 filas, como en AVX2.
        pub(crate) fn q4_0_matvec_qkv_slice(
            y: &mut [f32],
            qx: &[i8],
            sx: f32,
            w: &[Q4_0Block],
            cols: usize,
            rows: usize,
        ) {
            unsafe { q4_0_matvec_2row(y.as_mut_ptr(), qx.as_ptr(), sx, w.as_ptr(), cols, rows) }
        }

        #[allow(clippy::too_many_arguments)]
        pub(crate) fn q4_0_matmat_slice(
            y: &mut [f32],
            w: &[Q4_0Block],
            qxt: &[i8],
            sx: &[f32],
            r0: usize,
            r1: usize,
            cols: usize,
            b_dim: usize,
        ) {
            unsafe { q4_0_matmat(y, w, qxt, sx, r0, r1, cols, b_dim) }
        }
    };
}

/// Variante con `SDOT`. Solo puede usarse si la CPU tiene `dotprod`.
pub(crate) mod sdot {
    q4_0_int8_kernels!(#[target_feature(enable = "dotprod")] dot4_sdot);
}

/// Variante con multiplicación ampliada, para cualquier CPU aarch64.
pub(crate) mod widen {
    q4_0_int8_kernels!(#[target_feature(enable = "neon")] dot4_widen);
}
