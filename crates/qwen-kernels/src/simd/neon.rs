// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Kernels NEON para aarch64 (invocados exclusivamente mediante `KernelDispatch`).
//!
//! Q4_0 e INT8 se generan en dos variantes que solo cambian el producto escalar de 4 bytes:
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
//! El resto de kernels (cuantización de activaciones, normas, BF16, atención, RoPE, conversiones y
//! los del decodificador) usa NEON base y reproduce las operaciones de AVX2 en el mismo orden.
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

// ---------------------------------------------------------------------------------------------
// Kernels de coma flotante y conversión (NEON base, sin `dotprod`).
//
// Reproducen operación por operación los kernels AVX2: un registro AVX2 de 8 `f32` es aquí un
// par de registros NEON (`F8`), con las mismas FMA y la misma reducción final (`F8::hsum`). Donde
// AVX2 tiene un resto escalar, aquí está el mismo resto escalar.
// ---------------------------------------------------------------------------------------------

/// 8 carriles `f32` (los de un `__m256`) como dos registros NEON.
#[derive(Clone, Copy)]
struct F8(float32x4_t, float32x4_t);

impl F8 {
    #[inline]
    unsafe fn zero() -> Self {
        Self(vdupq_n_f32(0.0), vdupq_n_f32(0.0))
    }
    #[inline]
    unsafe fn splat(v: f32) -> Self {
        Self(vdupq_n_f32(v), vdupq_n_f32(v))
    }
    #[inline]
    unsafe fn load(p: *const f32) -> Self {
        Self(vld1q_f32(p), vld1q_f32(p.add(4)))
    }
    /// 8 valores BF16 ampliados a `f32` (desplazamiento de 16 bits, como `loadu_bf16_8`).
    #[inline]
    unsafe fn load_bf16(p: *const u16) -> Self {
        let b = vld1q_u16(p);
        Self(
            vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(b))),
            vreinterpretq_f32_u32(vshll_high_n_u16::<16>(b)),
        )
    }
    #[inline]
    unsafe fn store(self, p: *mut f32) {
        vst1q_f32(p, self.0);
        vst1q_f32(p.add(4), self.1);
    }
    /// `self + a · b` con un solo redondeo (`_mm256_fmadd_ps(a, b, self)`).
    #[inline]
    unsafe fn fma(self, a: Self, b: Self) -> Self {
        Self(vfmaq_f32(self.0, a.0, b.0), vfmaq_f32(self.1, a.1, b.1))
    }
    #[inline]
    unsafe fn add(self, o: Self) -> Self {
        Self(vaddq_f32(self.0, o.0), vaddq_f32(self.1, o.1))
    }
    #[inline]
    unsafe fn mul(self, o: Self) -> Self {
        Self(vmulq_f32(self.0, o.0), vmulq_f32(self.1, o.1))
    }
    /// Reducción en el orden de `hsum256_ps`.
    #[inline]
    unsafe fn hsum(self) -> f32 {
        hsum_f32x8(self.0, self.1)
    }
}

#[inline]
fn bf16_to_f32(b: u16) -> f32 {
    crate::quant::bf16_to_f32(b)
}

/// Mismo cuantizador que `quantize_act_int8_scalar` (y que AVX2): `round_ties_even(x · 127 / amax)`
/// saturado a `[-128, 127]`; da los mismos bits.
pub(crate) fn quantize_act_int8(qx: &mut [i8], x: &[f32]) -> f32 {
    let n = x.len();
    assert!(qx.len() >= n);
    unsafe {
        let xp = x.as_ptr();
        let mut vmax = vdupq_n_f32(0.0);
        let mut i = 0usize;
        while i + 4 <= n {
            vmax = vmaxq_f32(vmax, vabsq_f32(vld1q_f32(xp.add(i))));
            i += 4;
        }
        let mut amax = vmaxvq_f32(vmax);
        while i < n {
            let a = (*xp.add(i)).abs();
            if a > amax {
                amax = a;
            }
            i += 1;
        }
        if amax == 0.0 {
            qx[..n].fill(0);
            return 0.0;
        }
        let inv = 127.0f32 / amax;
        let vinv = vdupq_n_f32(inv);
        let qp = qx.as_mut_ptr();
        let cvt = |p: *const f32| vcvtnq_s32_f32(vmulq_f32(vld1q_f32(p), vinv));
        i = 0;
        while i + 16 <= n {
            let p = xp.add(i);
            let lo = vqmovn_high_s32(vqmovn_s32(cvt(p)), cvt(p.add(4)));
            let hi = vqmovn_high_s32(vqmovn_s32(cvt(p.add(8))), cvt(p.add(12)));
            vst1q_s8(qp.add(i), vqmovn_high_s16(vqmovn_s16(lo), hi));
            i += 16;
        }
        while i < n {
            let v = (*xp.add(i) * inv).round_ties_even() as i32;
            *qp.add(i) = v.clamp(-128, 127) as i8;
            i += 1;
        }
        amax / 127.0f32
    }
}

/// Suma de cuadrados en el orden de AVX2: dos acumuladores de 8 carriles por pasos de 16. Devuelve
/// la suma y dónde empieza el resto escalar.
#[inline]
unsafe fn sum_sq(p: *const f32, n: usize) -> (f32, usize) {
    let (mut s0, mut s1) = (F8::zero(), F8::zero());
    let mut i = 0usize;
    while i + 15 < n {
        let a = F8::load(p.add(i));
        let b = F8::load(p.add(i + 8));
        s0 = s0.fma(a, a);
        s1 = s1.fma(b, b);
        i += 16;
    }
    (s0.add(s1).hsum(), i)
}

/// `out = (x · inv_rms) · w` por pasos de 16 desde 0 (como AVX2); devuelve dónde empieza el resto.
#[inline]
unsafe fn scale_by_weight(out: *mut f32, x: *const f32, w: *const f32, inv_rms: f32, n: usize) -> usize {
    let vinv = F8::splat(inv_rms);
    let mut i = 0usize;
    while i + 15 < n {
        let a = F8::load(x.add(i)).mul(vinv).mul(F8::load(w.add(i)));
        let b = F8::load(x.add(i + 8)).mul(vinv).mul(F8::load(w.add(i + 8)));
        a.store(out.add(i));
        b.store(out.add(i + 8));
        i += 16;
    }
    i
}

pub(crate) fn rms_norm(out: &mut [f32], x: &[f32], weight: &[f32], seq: usize, dim: usize, eps: f32) {
    unsafe {
        let wp = weight.as_ptr();
        for s in 0..seq {
            let xs = x.as_ptr().add(s * dim);
            let os = out.as_mut_ptr().add(s * dim);
            let (mut sum, mut i) = sum_sq(xs, dim);
            while i < dim {
                let v = *xs.add(i);
                sum += v * v;
                i += 1;
            }
            let inv_rms = 1.0f32 / (sum / dim as f32 + eps).sqrt();
            let mut i = scale_by_weight(os, xs, wp, inv_rms, dim);
            while i < dim {
                *os.add(i) = *xs.add(i) * inv_rms * *wp.add(i);
                i += 1;
            }
        }
    }
}

pub(crate) fn rms_norm_residual(
    out: &mut [f32],
    x: &mut [f32],
    residual: &[f32],
    weight: &[f32],
    dim: usize,
    eps: f32,
) {
    unsafe {
        let (xp, rp, wp, op) = (x.as_mut_ptr(), residual.as_ptr(), weight.as_ptr(), out.as_mut_ptr());
        let (mut s0, mut s1) = (F8::zero(), F8::zero());
        let mut i = 0usize;
        while i + 15 < dim {
            let a = F8::load(xp.add(i)).add(F8::load(rp.add(i)));
            let b = F8::load(xp.add(i + 8)).add(F8::load(rp.add(i + 8)));
            a.store(xp.add(i));
            b.store(xp.add(i + 8));
            s0 = s0.fma(a, a);
            s1 = s1.fma(b, b);
            i += 16;
        }
        let mut sum = s0.add(s1).hsum();
        while i < dim {
            let v = *xp.add(i) + *rp.add(i);
            *xp.add(i) = v;
            sum += v * v;
            i += 1;
        }
        let inv_rms = 1.0f32 / (sum / dim as f32 + eps).sqrt();
        let mut i = scale_by_weight(op, xp, wp, inv_rms, dim);
        while i < dim {
            *op.add(i) = *xp.add(i) * inv_rms * *wp.add(i);
            i += 1;
        }
    }
}

pub(crate) fn rms_norm_per_head(
    x: &mut [f32],
    weight: &[f32],
    seq: usize,
    n_heads: usize,
    head_dim: usize,
    eps: f32,
) {
    unsafe {
        let dim = n_heads * head_dim;
        let wp = weight.as_ptr();
        for s in 0..seq {
            for h in 0..n_heads {
                let hs = x.as_mut_ptr().add(s * dim + h * head_dim);
                let (mut sum, mut i) = sum_sq(hs, head_dim);
                while i < head_dim {
                    let v = *hs.add(i);
                    sum += v * v;
                    i += 1;
                }
                let inv_rms = 1.0f32 / (sum / head_dim as f32 + eps).sqrt();
                let mut i = scale_by_weight(hs, hs, wp, inv_rms, head_dim);
                while i < head_dim {
                    *hs.add(i) *= inv_rms * *wp.add(i);
                    i += 1;
                }
            }
        }
    }
}

/// Producto de dos filas BF16 por `x` (cuatro acumuladores de 8 carriles cada una), en el orden de
/// `bf16_matvec_slice` y `argmax_matvec_bf16` de AVX2.
#[inline]
unsafe fn bf16_dot_2rows(w0: *const u16, w1: *const u16, xp: *const f32, in_dim: usize) -> (f32, f32) {
    let (mut a0, mut a1, mut a2, mut a3) = (F8::zero(), F8::zero(), F8::zero(), F8::zero());
    let (mut b0, mut b1, mut b2, mut b3) = (F8::zero(), F8::zero(), F8::zero(), F8::zero());
    let mut k = 0usize;
    while k + 32 <= in_dim {
        let x0 = F8::load(xp.add(k));
        let x1 = F8::load(xp.add(k + 8));
        let x2 = F8::load(xp.add(k + 16));
        let x3 = F8::load(xp.add(k + 24));
        a0 = a0.fma(F8::load_bf16(w0.add(k)), x0);
        a1 = a1.fma(F8::load_bf16(w0.add(k + 8)), x1);
        a2 = a2.fma(F8::load_bf16(w0.add(k + 16)), x2);
        a3 = a3.fma(F8::load_bf16(w0.add(k + 24)), x3);
        b0 = b0.fma(F8::load_bf16(w1.add(k)), x0);
        b1 = b1.fma(F8::load_bf16(w1.add(k + 8)), x1);
        b2 = b2.fma(F8::load_bf16(w1.add(k + 16)), x2);
        b3 = b3.fma(F8::load_bf16(w1.add(k + 24)), x3);
        k += 32;
    }
    while k + 8 <= in_dim {
        let xv = F8::load(xp.add(k));
        a0 = a0.fma(F8::load_bf16(w0.add(k)), xv);
        b0 = b0.fma(F8::load_bf16(w1.add(k)), xv);
        k += 8;
    }
    let mut s0 = a0.add(a2).add(a1.add(a3)).hsum();
    let mut s1 = b0.add(b2).add(b1.add(b3)).hsum();
    while k < in_dim {
        let xv = *xp.add(k);
        s0 += bf16_to_f32(*w0.add(k)) * xv;
        s1 += bf16_to_f32(*w1.add(k)) * xv;
        k += 1;
    }
    (s0, s1)
}

pub(crate) fn bf16_matvec_slice(y: &mut [f32], x: &[f32], w: &[u16], in_dim: usize, out_dim: usize) {
    unsafe {
        let (xp, wp, yp) = (x.as_ptr(), w.as_ptr(), y.as_mut_ptr());
        let mut o = 0usize;
        while o + 1 < out_dim {
            let (s0, s1) = bf16_dot_2rows(wp.add(o * in_dim), wp.add((o + 1) * in_dim), xp, in_dim);
            *yp.add(o) = s0;
            *yp.add(o + 1) = s1;
            o += 2;
        }
        if o < out_dim {
            let w_row = wp.add(o * in_dim);
            let (mut acc0, mut acc1) = (F8::zero(), F8::zero());
            let mut k = 0usize;
            while k + 16 <= in_dim {
                acc0 = acc0.fma(F8::load_bf16(w_row.add(k)), F8::load(xp.add(k)));
                acc1 = acc1.fma(F8::load_bf16(w_row.add(k + 8)), F8::load(xp.add(k + 8)));
                k += 16;
            }
            while k + 8 <= in_dim {
                acc0 = acc0.fma(F8::load_bf16(w_row.add(k)), F8::load(xp.add(k)));
                k += 8;
            }
            let mut sum = acc0.add(acc1).hsum();
            while k < in_dim {
                sum += bf16_to_f32(*w_row.add(k)) * *xp.add(k);
                k += 1;
            }
            *yp.add(o) = sum;
        }
    }
}

/// `y[r, b] = Σ_k W[r, k] · x[k, b]` con una FMA por elemento y por `k` en orden ascendente: la
/// misma secuencia de FMA que las dos variantes de AVX2 (de `B` fijo y genérica), y los mismos bits.
pub(crate) fn bf16_matmat_slice(
    y: &mut [f32],
    w: &[u16],
    x: &[f32],
    r0: usize,
    r1: usize,
    cols: usize,
    b_dim: usize,
) {
    unsafe {
        let mut acc = [0.0f32; 64];
        for r in r0..r1 {
            let wr = w.as_ptr().add(r * cols);
            acc[..b_dim].fill(0.0);
            for k in 0..cols {
                let wv = bf16_to_f32(*wr.add(k));
                let xk = x.as_ptr().add(k * b_dim);
                let vw = vdupq_n_f32(wv);
                let mut b = 0usize;
                while b + 4 <= b_dim {
                    let ap = acc.as_mut_ptr().add(b);
                    vst1q_f32(ap, vfmaq_f32(vld1q_f32(ap), vw, vld1q_f32(xk.add(b))));
                    b += 4;
                }
                while b < b_dim {
                    acc[b] = wv.mul_add(*xk.add(b), acc[b]);
                    b += 1;
                }
            }
            std::ptr::copy_nonoverlapping(acc.as_ptr(), y.as_mut_ptr().add(r * b_dim), b_dim);
        }
    }
}

pub(crate) fn argmax_matvec_bf16(x: &[f32], w_bf16: &[u16], in_dim: usize, out_dim: usize) -> usize {
    unsafe {
        let (xp, wp) = (x.as_ptr(), w_bf16.as_ptr());
        let mut best_idx = 0usize;
        let mut best_val = -1e30f32;
        let mut o = 0usize;
        while o + 1 < out_dim {
            let (s0, s1) = bf16_dot_2rows(wp.add(o * in_dim), wp.add((o + 1) * in_dim), xp, in_dim);
            if s0 > best_val {
                best_val = s0;
                best_idx = o;
            }
            if s1 > best_val {
                best_val = s1;
                best_idx = o + 1;
            }
            o += 2;
        }
        while o < out_dim {
            let row = wp.add(o * in_dim);
            let mut sum = 0.0f32;
            for k in 0..in_dim {
                sum += bf16_to_f32(*row.add(k)) * *xp.add(k);
            }
            if sum > best_val {
                best_val = sum;
                best_idx = o;
            }
            o += 1;
        }
        best_idx
    }
}

/// Una fila de `f32` o de BF16 leída como `f32` (claves y valores de la atención).
trait Row: Copy {
    unsafe fn load8(self, d: usize) -> F8;
    unsafe fn get(self, d: usize) -> f32;
}

impl Row for *const f32 {
    #[inline]
    unsafe fn load8(self, d: usize) -> F8 {
        F8::load(self.add(d))
    }
    #[inline]
    unsafe fn get(self, d: usize) -> f32 {
        *self.add(d)
    }
}

impl Row for *const u16 {
    #[inline]
    unsafe fn load8(self, d: usize) -> F8 {
        F8::load_bf16(self.add(d))
    }
    #[inline]
    unsafe fn get(self, d: usize) -> f32 {
        bf16_to_f32(*self.add(d))
    }
}

/// Producto escalar con cuatro acumuladores de 8 carriles (`dot_f32` y `dot_f32_bf16` de AVX2).
#[inline]
unsafe fn dot<R: Row>(a: *const f32, b: R, n: usize) -> f32 {
    let (mut c0, mut c1, mut c2, mut c3) = (F8::zero(), F8::zero(), F8::zero(), F8::zero());
    let mut d = 0usize;
    while d + 32 <= n {
        c0 = c0.fma(F8::load(a.add(d)), b.load8(d));
        c1 = c1.fma(F8::load(a.add(d + 8)), b.load8(d + 8));
        c2 = c2.fma(F8::load(a.add(d + 16)), b.load8(d + 16));
        c3 = c3.fma(F8::load(a.add(d + 24)), b.load8(d + 24));
        d += 32;
    }
    while d + 8 <= n {
        c0 = c0.fma(F8::load(a.add(d)), b.load8(d));
        d += 8;
    }
    let mut s = c0.add(c2).add(c1.add(c3)).hsum();
    while d < n {
        s += *a.add(d) * b.get(d);
        d += 1;
    }
    s
}

/// `o = o · c + v` (`acc_corr` de AVX2).
#[inline]
unsafe fn acc_corr<R: Row>(o: *mut f32, v: R, c: f32, n: usize) {
    let vc = F8::splat(c);
    let mut d = 0usize;
    while d + 8 <= n {
        v.load8(d).fma(F8::load(o.add(d)), vc).store(o.add(d));
        d += 8;
    }
    while d < n {
        *o.add(d) = *o.add(d) * c + v.get(d);
        d += 1;
    }
}

/// `o = o + v · w` (`acc_wt` de AVX2).
#[inline]
unsafe fn acc_wt<R: Row>(o: *mut f32, v: R, w: f32, n: usize) {
    let vw = F8::splat(w);
    let mut d = 0usize;
    while d + 8 <= n {
        F8::load(o.add(d)).fma(v.load8(d), vw).store(o.add(d));
        d += 8;
    }
    while d < n {
        *o.add(d) += v.get(d) * w;
        d += 1;
    }
}

#[inline]
unsafe fn scale_vec(o: *mut f32, s: f32, n: usize) {
    let vs = F8::splat(s);
    let mut d = 0usize;
    while d + 8 <= n {
        F8::load(o.add(d)).mul(vs).store(o.add(d));
        d += 8;
    }
    while d < n {
        *o.add(d) *= s;
        d += 1;
    }
}

/// Una fila de salida de la atención causal con *online softmax*, como en AVX2: claves y valores
/// de la fila `j` en `k_at(j)` y `v_at(j)`.
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn attn_row<R: Row>(
    o_row: *mut f32,
    q_row: *const f32,
    head_dim: usize,
    scale: f32,
    keys: std::ops::Range<usize>,
    k_at: impl Fn(usize) -> R,
    v_at: impl Fn(usize) -> R,
) {
    let mut max_score = -1e30f32;
    let mut sum_exp = 0.0f32;
    std::ptr::write_bytes(o_row, 0, head_dim);
    for j in keys {
        let score = dot(q_row, k_at(j), head_dim) * scale;
        if score > max_score {
            let correction = (max_score - score).exp();
            sum_exp = sum_exp * correction + 1.0;
            acc_corr(o_row, v_at(j), correction, head_dim);
            max_score = score;
        } else {
            let wt = (score - max_score).exp();
            sum_exp += wt;
            acc_wt(o_row, v_at(j), wt, head_dim);
        }
    }
    if sum_exp > 0.0 {
        scale_vec(o_row, 1.0f32 / sum_exp, head_dim);
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn causal_attention_f32_heads(
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
    let (q_hidden, kv_hidden) = (n_heads * head_dim, n_kv_heads * head_dim);
    unsafe {
        let (qp, kp, vp, op) = (q.as_ptr(), k.as_ptr(), v.as_ptr(), out.as_mut_ptr());
        for h in 0..n_heads {
            let kv_off = (h / heads_per_kv) * head_dim;
            for i in 0..seq_q {
                let k_end = (q_offset + i + 1).min(seq_k);
                let k_start = if window > 0 && k_end > window { k_end - window } else { 0 };
                attn_row(
                    op.add(i * q_hidden + h * head_dim),
                    qp.add(i * q_hidden + h * head_dim),
                    head_dim,
                    scale,
                    k_start..k_end,
                    |j| kp.add(j * kv_hidden + kv_off),
                    |j| vp.add(j * kv_hidden + kv_off),
                );
            }
        }
    }
}

/// # Safety
/// `op` debe apuntar a un búfer de al menos `seq_q * n_heads * head_dim` `f32`; varios hilos solo
/// pueden llamarla a la vez con rangos `[h_begin, h_end)` disjuntos.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn causal_attention_bf16kv_heads_ptr(
    op: *mut f32,
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
    let (q_hidden, kv_hidden) = (n_heads * head_dim, n_kv_heads * head_dim);
    let (qp, kp, vp) = (q.as_ptr(), k_bf16.as_ptr(), v_bf16.as_ptr());
    for h in h_begin..h_end {
        let kv_off = (h / heads_per_kv) * head_dim;
        for i in 0..seq_q {
            let k_end = (q_offset + i + 1).min(seq_k);
            attn_row(
                op.add(i * q_hidden + h * head_dim),
                qp.add(i * q_hidden + h * head_dim),
                head_dim,
                scale,
                0..k_end,
                |j| kp.add(j * kv_hidden + kv_off),
                |j| vp.add(j * kv_hidden + kv_off),
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn causal_attention_bf16kv_heads(
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
    assert!(out.len() >= seq_q * n_heads * head_dim);
    unsafe {
        causal_attention_bf16kv_heads_ptr(
            out.as_mut_ptr(),
            q,
            k_bf16,
            v_bf16,
            seq_q,
            seq_k,
            n_heads,
            n_kv_heads,
            head_dim,
            scale,
            q_offset,
            h_begin,
            h_end,
        )
    }
}

pub(crate) fn apply_rope_neox_inplace(
    x: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    cos_cache: &[f32],
    sin_cache: &[f32],
    pos: usize,
) {
    let half = head_dim / 2;
    unsafe {
        let cp = cos_cache.as_ptr().add(pos * half);
        let sp = sin_cache.as_ptr().add(pos * half);
        for h in 0..n_heads {
            let xh = x.as_mut_ptr().add(h * head_dim);
            let mut i = 0usize;
            while i + 8 <= half {
                let (c, s) = (F8::load(cp.add(i)), F8::load(sp.add(i)));
                let (v1, v2) = (F8::load(xh.add(i)), F8::load(xh.add(i + half)));
                // v1·c − v2·s y v2·c + v1·s, cada una con una FMA (`_mm256_fmsub_ps`, `_mm256_fmadd_ps`).
                let t = v2.mul(s);
                let lo = F8(vfmaq_f32(vnegq_f32(t.0), v1.0, c.0), vfmaq_f32(vnegq_f32(t.1), v1.1, c.1));
                let hi = v1.mul(s).fma(v2, c);
                lo.store(xh.add(i));
                hi.store(xh.add(i + half));
                i += 8;
            }
            while i < half {
                let (x1, x2) = (*xh.add(i), *xh.add(i + half));
                let (ci, si) = (*cp.add(i), *sp.add(i));
                *xh.add(i) = x1 * ci - x2 * si;
                *xh.add(i + half) = x2 * ci + x1 * si;
                i += 1;
            }
        }
    }
}

pub(crate) fn bf16_accum_f32(dst: &mut [f32], src_bf16: &[u16]) {
    let n = dst.len().min(src_bf16.len());
    unsafe {
        let (dp, sp) = (dst.as_mut_ptr(), src_bf16.as_ptr());
        let mut i = 0usize;
        while i + 8 <= n {
            F8::load(dp.add(i)).add(F8::load_bf16(sp.add(i))).store(dp.add(i));
            i += 8;
        }
        while i < n {
            *dp.add(i) += bf16_to_f32(*sp.add(i));
            i += 1;
        }
    }
}

pub(crate) fn bf16_to_f32_vec(dst: &mut [f32], src_bf16: &[u16]) {
    let n = dst.len().min(src_bf16.len());
    unsafe {
        let (dp, sp) = (dst.as_mut_ptr(), src_bf16.as_ptr());
        let mut i = 0usize;
        while i + 8 <= n {
            F8::load_bf16(sp.add(i)).store(dp.add(i));
            i += 8;
        }
        while i < n {
            *dp.add(i) = bf16_to_f32(*sp.add(i));
            i += 1;
        }
    }
}

/// Truncamiento a BF16 (los 16 bits altos), como la ruta escalar y AVX2.
pub(crate) fn f32_to_bf16_vec(dst: &mut [u16], src: &[f32]) {
    let n = dst.len().min(src.len());
    unsafe {
        let (dp, sp) = (dst.as_mut_ptr(), src.as_ptr());
        let mut i = 0usize;
        while i + 8 <= n {
            let lo = vshrn_n_u32::<16>(vreinterpretq_u32_f32(vld1q_f32(sp.add(i))));
            let hi = vshrn_n_u32::<16>(vreinterpretq_u32_f32(vld1q_f32(sp.add(i + 4))));
            vst1q_u16(dp.add(i), vcombine_u16(lo, hi));
            i += 8;
        }
        while i < n {
            *dp.add(i) = crate::quant::f32_to_bf16_trunc(*sp.add(i));
            i += 1;
        }
    }
}

pub(crate) fn f32_matvec_accum(y: &mut [f32], w: &[f32], x: &[f32], rows: usize, cols: usize) {
    unsafe {
        let (wp, xp, yp) = (w.as_ptr(), x.as_ptr(), y.as_mut_ptr());
        for r in 0..rows {
            *yp.add(r) += dot(wp.add(r * cols), xp, cols);
        }
    }
}

/// `sin²(y)`: reducción a `[-π/2, π/2]` y el polinomio de grado 11 de `vsin2_avx2`.
#[inline]
unsafe fn vsin2(y: float32x4_t) -> float32x4_t {
    const SIN_C1: f32 = -1.0 / 6.0;
    const SIN_C2: f32 = 1.0 / 120.0;
    const SIN_C3: f32 = -1.0 / 5040.0;
    const SIN_C4: f32 = 1.0 / 362880.0;
    const SIN_C5: f32 = -1.0 / 39916800.0;
    const PI_HI: f32 = std::f32::consts::PI;
    const PI_LO: f32 = -8.742_278e-8;
    let n = vrndnq_f32(vmulq_f32(y, vdupq_n_f32(std::f32::consts::FRAC_1_PI)));
    let u = vfmsq_f32(y, n, vdupq_n_f32(PI_HI));
    let u = vfmaq_f32(u, n, vdupq_n_f32(PI_LO));
    let u2 = vmulq_f32(u, u);
    let mut p = vdupq_n_f32(SIN_C5);
    for c in [SIN_C4, SIN_C3, SIN_C2, SIN_C1, 1.0] {
        p = vfmaq_f32(vdupq_n_f32(c), p, u2);
    }
    let s = vmulq_f32(u, p);
    vmulq_f32(s, s)
}

/// Snake: `x + exp(-log_beta) · sin²(exp(log_alpha) · x)`. Como en AVX2: por grupos de 8, el
/// polinomio si todos cumplen `|a·x| <= 8192` y el `sin` de la biblioteca si alguno lo supera.
pub(crate) fn snake_row(row: &mut [f32], log_alpha: f32, log_beta: f32) {
    const SIN_POLY_MAX: f32 = 8192.0;
    let a = log_alpha.exp();
    let inv_b = (-log_beta).exp();
    let len = row.len();
    unsafe {
        let rp = row.as_mut_ptr();
        let (va, vinv_b, vmax) = (vdupq_n_f32(a), vdupq_n_f32(inv_b), vdupq_n_f32(SIN_POLY_MAX));
        let mut t = 0usize;
        while t + 8 <= len {
            let x = F8::load(rp.add(t));
            let ax = F8(vmulq_f32(va, x.0), vmulq_f32(va, x.1));
            let over = vorrq_u32(vcagtq_f32(ax.0, vmax), vcagtq_f32(ax.1, vmax));
            let s2 = if vmaxvq_u32(over) == 0 {
                F8(vsin2(ax.0), vsin2(ax.1))
            } else {
                let mut v = [0.0f32; 8];
                ax.store(v.as_mut_ptr());
                for e in v.iter_mut() {
                    let s = e.sin();
                    *e = s * s;
                }
                F8::load(v.as_ptr())
            };
            F8(vfmaq_f32(x.0, vinv_b, s2.0), vfmaq_f32(x.1, vinv_b, s2.1)).store(rp.add(t));
            t += 8;
        }
        while t < len {
            let s = (a * *rp.add(t)).sin();
            *rp.add(t) += inv_b * s * s;
            t += 1;
        }
    }
}
