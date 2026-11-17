// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#![allow(
    clippy::missing_safety_doc,
    clippy::too_many_arguments,
    clippy::needless_range_loop
)]

use core::arch::x86_64::*;

use crate::ftz::ftz_on;
use crate::quant::{
    bf16_to_f32, f16_to_f32, Q4_0Block, Q4_0_BLOCK_SIZE, QWEN_INT8_PF_DIST, QWEN_LONG_ROW_COLS,
    QWEN_QACT_MAX,
};

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn hsum256_ps(v: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps::<1>(v);
    let lo = _mm_add_ps(lo, hi);
    let sh = _mm_movehl_ps(lo, lo);
    let lo = _mm_add_ps(lo, sh);
    let sh = _mm_shuffle_ps::<0x1>(lo, lo);
    let lo = _mm_add_ss(lo, sh);
    _mm_cvtss_f32(lo)
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn loadu_bf16_8(p: *const u16) -> __m256 {
    let b = _mm_loadu_si128(p as *const __m128i);
    _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(b)))
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn hsum_epi32(v: __m256i) -> i32 {
    let s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0x4E>(s));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0xB1>(s));
    _mm_cvtsi128_si32(s)
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn set_m128i(hi: __m128i, lo: __m128i) -> __m256i {
    _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lo), hi)
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn quantize_act_int8(qx: &mut [i8], x: &[f32]) -> f32 {
    let n = x.len();
    let sign = _mm256_set1_ps(-0.0f32);
    let mut vmax = _mm256_setzero_ps();
    let mut i = 0usize;
    let xp = x.as_ptr();
    while i + 8 <= n {
        vmax = _mm256_max_ps(vmax, _mm256_andnot_ps(sign, _mm256_loadu_ps(xp.add(i))));
        i += 8;
    }
    let m4 = _mm_max_ps(_mm256_castps256_ps128(vmax), _mm256_extractf128_ps::<1>(vmax));
    let m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
    let m4 = _mm_max_ss(m4, _mm_shuffle_ps::<1>(m4, m4));
    let mut amax = _mm_cvtss_f32(m4);
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
    let vinv = _mm256_set1_ps(inv);
    let fix = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    let qxp = qx.as_mut_ptr();
    i = 0;
    while i + 32 <= n {
        let q0 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xp.add(i)), vinv));
        let q1 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xp.add(i + 8)), vinv));
        let q2 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xp.add(i + 16)), vinv));
        let q3 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xp.add(i + 24)), vinv));
        let p = _mm256_packs_epi16(_mm256_packs_epi32(q0, q1), _mm256_packs_epi32(q2, q3));
        _mm256_storeu_si256(qxp.add(i) as *mut __m256i, _mm256_permutevar8x32_epi32(p, fix));
        i += 32;
    }
    while i < n {
        let v = (*xp.add(i) * inv).round_ties_even() as i32;
        *qxp.add(i) = v.clamp(-128, 127) as i8;
        i += 1;
    }
    amax / 127.0f32
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn rms_norm(
    out: &mut [f32],
    x: &[f32],
    weight: &[f32],
    seq: usize,
    dim: usize,
    eps: f32,
) {
    let wp = weight.as_ptr();
    for s in 0..seq {
        let xs = x.as_ptr().add(s * dim);
        let os = out.as_mut_ptr().add(s * dim);
        let mut vsum0 = _mm256_setzero_ps();
        let mut vsum1 = _mm256_setzero_ps();
        let mut i = 0usize;
        while i + 15 < dim {
            let v0 = _mm256_loadu_ps(xs.add(i));
            let v1 = _mm256_loadu_ps(xs.add(i + 8));
            vsum0 = _mm256_fmadd_ps(v0, v0, vsum0);
            vsum1 = _mm256_fmadd_ps(v1, v1, vsum1);
            i += 16;
        }
        let mut sum = hsum256_ps(_mm256_add_ps(vsum0, vsum1));
        while i < dim {
            let v = *xs.add(i);
            sum += v * v;
            i += 1;
        }
        let inv_rms = 1.0f32 / (sum / dim as f32 + eps).sqrt();
        let vinv = _mm256_set1_ps(inv_rms);
        i = 0;
        while i + 15 < dim {
            let v0 = _mm256_loadu_ps(xs.add(i));
            let v1 = _mm256_loadu_ps(xs.add(i + 8));
            let w0 = _mm256_loadu_ps(wp.add(i));
            let w1 = _mm256_loadu_ps(wp.add(i + 8));
            _mm256_storeu_ps(os.add(i), _mm256_mul_ps(_mm256_mul_ps(v0, vinv), w0));
            _mm256_storeu_ps(os.add(i + 8), _mm256_mul_ps(_mm256_mul_ps(v1, vinv), w1));
            i += 16;
        }
        while i < dim {
            *os.add(i) = *xs.add(i) * inv_rms * *wp.add(i);
            i += 1;
        }
    }
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn rms_norm_residual(
    out: &mut [f32],
    x: &mut [f32],
    residual: &[f32],
    weight: &[f32],
    dim: usize,
    eps: f32,
) {
    let xp = x.as_mut_ptr();
    let rp = residual.as_ptr();
    let wp = weight.as_ptr();
    let op = out.as_mut_ptr();
    let mut vsum0 = _mm256_setzero_ps();
    let mut vsum1 = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 15 < dim {
        let x0 = _mm256_add_ps(_mm256_loadu_ps(xp.add(i)), _mm256_loadu_ps(rp.add(i)));
        let x1 = _mm256_add_ps(_mm256_loadu_ps(xp.add(i + 8)), _mm256_loadu_ps(rp.add(i + 8)));
        _mm256_storeu_ps(xp.add(i), x0);
        _mm256_storeu_ps(xp.add(i + 8), x1);
        vsum0 = _mm256_fmadd_ps(x0, x0, vsum0);
        vsum1 = _mm256_fmadd_ps(x1, x1, vsum1);
        i += 16;
    }
    let mut sum = hsum256_ps(_mm256_add_ps(vsum0, vsum1));
    while i < dim {
        let v = *xp.add(i) + *rp.add(i);
        *xp.add(i) = v;
        sum += v * v;
        i += 1;
    }
    let inv_rms = 1.0f32 / (sum / dim as f32 + eps).sqrt();
    let vinv = _mm256_set1_ps(inv_rms);
    i = 0;
    while i + 15 < dim {
        let v0 = _mm256_loadu_ps(xp.add(i));
        let v1 = _mm256_loadu_ps(xp.add(i + 8));
        let w0 = _mm256_loadu_ps(wp.add(i));
        let w1 = _mm256_loadu_ps(wp.add(i + 8));
        _mm256_storeu_ps(op.add(i), _mm256_mul_ps(_mm256_mul_ps(v0, vinv), w0));
        _mm256_storeu_ps(op.add(i + 8), _mm256_mul_ps(_mm256_mul_ps(v1, vinv), w1));
        i += 16;
    }
    while i < dim {
        *op.add(i) = *xp.add(i) * inv_rms * *wp.add(i);
        i += 1;
    }
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn rms_norm_per_head(
    x: &mut [f32],
    weight: &[f32],
    seq: usize,
    n_heads: usize,
    head_dim: usize,
    eps: f32,
) {
    let dim = n_heads * head_dim;
    let wp = weight.as_ptr();
    for s in 0..seq {
        let xs = x.as_mut_ptr().add(s * dim);
        for h in 0..n_heads {
            let hs = xs.add(h * head_dim);
            let mut vsum0 = _mm256_setzero_ps();
            let mut vsum1 = _mm256_setzero_ps();
            let mut i = 0usize;
            while i + 15 < head_dim {
                let v0 = _mm256_loadu_ps(hs.add(i));
                let v1 = _mm256_loadu_ps(hs.add(i + 8));
                vsum0 = _mm256_fmadd_ps(v0, v0, vsum0);
                vsum1 = _mm256_fmadd_ps(v1, v1, vsum1);
                i += 16;
            }
            let mut sum = hsum256_ps(_mm256_add_ps(vsum0, vsum1));
            while i < head_dim {
                let v = *hs.add(i);
                sum += v * v;
                i += 1;
            }
            let inv_rms = 1.0f32 / (sum / head_dim as f32 + eps).sqrt();
            let vinv = _mm256_set1_ps(inv_rms);
            i = 0;
            while i + 15 < head_dim {
                let v0 = _mm256_loadu_ps(hs.add(i));
                let v1 = _mm256_loadu_ps(hs.add(i + 8));
                let w0 = _mm256_loadu_ps(wp.add(i));
                let w1 = _mm256_loadu_ps(wp.add(i + 8));
                _mm256_storeu_ps(hs.add(i), _mm256_mul_ps(_mm256_mul_ps(v0, vinv), w0));
                _mm256_storeu_ps(hs.add(i + 8), _mm256_mul_ps(_mm256_mul_ps(v1, vinv), w1));
                i += 16;
            }
            while i < head_dim {
                *hs.add(i) *= inv_rms * *wp.add(i);
                i += 1;
            }
        }
    }
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn bf16_matvec_slice(y: &mut [f32], x: &[f32], w: &[u16], in_dim: usize, out_dim: usize) {
    let xp = x.as_ptr();
    let wp = w.as_ptr();
    let yp = y.as_mut_ptr();
    let mut o = 0usize;
    while o + 1 < out_dim {
        let w0 = wp.add(o * in_dim);
        let w1 = wp.add((o + 1) * in_dim);
        if o + 5 < out_dim {
            let pf0 = wp.add((o + 4) * in_dim);
            let pf1 = wp.add((o + 5) * in_dim);
            _mm_prefetch::<_MM_HINT_NTA>(pf0 as *const i8);
            _mm_prefetch::<_MM_HINT_NTA>(pf0.add(64) as *const i8);
            _mm_prefetch::<_MM_HINT_NTA>(pf1 as *const i8);
            _mm_prefetch::<_MM_HINT_NTA>(pf1.add(64) as *const i8);
        }
        let (mut a0, mut a1, mut a2, mut a3) = (
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
        );
        let (mut b0, mut b1, mut b2, mut b3) = (
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
        );
        let mut k = 0usize;
        while k + 32 <= in_dim {
            let x0 = _mm256_loadu_ps(xp.add(k));
            let x1 = _mm256_loadu_ps(xp.add(k + 8));
            let x2 = _mm256_loadu_ps(xp.add(k + 16));
            let x3 = _mm256_loadu_ps(xp.add(k + 24));
            a0 = _mm256_fmadd_ps(loadu_bf16_8(w0.add(k)), x0, a0);
            a1 = _mm256_fmadd_ps(loadu_bf16_8(w0.add(k + 8)), x1, a1);
            a2 = _mm256_fmadd_ps(loadu_bf16_8(w0.add(k + 16)), x2, a2);
            a3 = _mm256_fmadd_ps(loadu_bf16_8(w0.add(k + 24)), x3, a3);
            b0 = _mm256_fmadd_ps(loadu_bf16_8(w1.add(k)), x0, b0);
            b1 = _mm256_fmadd_ps(loadu_bf16_8(w1.add(k + 8)), x1, b1);
            b2 = _mm256_fmadd_ps(loadu_bf16_8(w1.add(k + 16)), x2, b2);
            b3 = _mm256_fmadd_ps(loadu_bf16_8(w1.add(k + 24)), x3, b3);
            k += 32;
        }
        while k + 8 <= in_dim {
            let xv = _mm256_loadu_ps(xp.add(k));
            a0 = _mm256_fmadd_ps(loadu_bf16_8(w0.add(k)), xv, a0);
            b0 = _mm256_fmadd_ps(loadu_bf16_8(w1.add(k)), xv, b0);
            k += 8;
        }
        let a_sum = _mm256_add_ps(_mm256_add_ps(a0, a2), _mm256_add_ps(a1, a3));
        let b_sum = _mm256_add_ps(_mm256_add_ps(b0, b2), _mm256_add_ps(b1, b3));
        let mut s0 = hsum256_ps(a_sum);
        let mut s1 = hsum256_ps(b_sum);
        while k < in_dim {
            let xv = *xp.add(k);
            s0 += bf16_to_f32(*w0.add(k)) * xv;
            s1 += bf16_to_f32(*w1.add(k)) * xv;
            k += 1;
        }
        *yp.add(o) = s0;
        *yp.add(o + 1) = s1;
        o += 2;
    }
    if o < out_dim {
        let w_row = wp.add(o * in_dim);
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut k = 0usize;
        while k + 16 <= in_dim {
            acc0 = _mm256_fmadd_ps(loadu_bf16_8(w_row.add(k)), _mm256_loadu_ps(xp.add(k)), acc0);
            acc1 = _mm256_fmadd_ps(
                loadu_bf16_8(w_row.add(k + 8)),
                _mm256_loadu_ps(xp.add(k + 8)),
                acc1,
            );
            k += 16;
        }
        while k + 8 <= in_dim {
            acc0 = _mm256_fmadd_ps(loadu_bf16_8(w_row.add(k)), _mm256_loadu_ps(xp.add(k)), acc0);
            k += 8;
        }
        let mut sum = hsum256_ps(_mm256_add_ps(acc0, acc1));
        while k < in_dim {
            sum += bf16_to_f32(*w_row.add(k)) * *xp.add(k);
            k += 1;
        }
        *yp.add(o) = sum;
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn bf16_matmat_fixed_b<const BV: usize>(
    y: &mut [f32],
    w: &[u16],
    x: &[f32],
    r0: usize,
    r1: usize,
    cols: usize,
) {
    let mut r = r0;
    while r + 1 < r1 {
        let w0 = w.as_ptr().add(r * cols);
        let w1 = w.as_ptr().add((r + 1) * cols);
        let y0 = y.as_mut_ptr().add(r * BV);
        let y1 = y.as_mut_ptr().add((r + 1) * BV);
        let mut a = [0.0f32; BV];
        let mut b = [0.0f32; BV];
        for k in 0..cols {
            let w0v = bf16_to_f32(*w0.add(k));
            let w1v = bf16_to_f32(*w1.add(k));
            let xk = x.as_ptr().add(k * BV);
            for j in 0..BV {
                let xv = *xk.add(j);
                a[j] = w0v.mul_add(xv, a[j]);
                b[j] = w1v.mul_add(xv, b[j]);
            }
        }
        std::ptr::copy_nonoverlapping(a.as_ptr(), y0, BV);
        std::ptr::copy_nonoverlapping(b.as_ptr(), y1, BV);
        r += 2;
    }
    while r < r1 {
        let wr = w.as_ptr().add(r * cols);
        let yr = y.as_mut_ptr().add(r * BV);
        let mut acc = [0.0f32; BV];
        for k in 0..cols {
            let wv = bf16_to_f32(*wr.add(k));
            let xk = x.as_ptr().add(k * BV);
            for j in 0..BV {
                acc[j] = wv.mul_add(*xk.add(j), acc[j]);
            }
        }
        std::ptr::copy_nonoverlapping(acc.as_ptr(), yr, BV);
        r += 1;
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn bf16_matmat_generic(
    y: &mut [f32],
    w: &[u16],
    x: &[f32],
    r0: usize,
    r1: usize,
    cols: usize,
    b_dim: usize,
) {
    let mut acc = [0.0f32; 64];
    for r in r0..r1 {
        let wr = w.as_ptr().add(r * cols);
        let yr = y.as_mut_ptr().add(r * b_dim);
        acc[..b_dim].fill(0.0);
        for k in 0..cols {
            let wv = bf16_to_f32(*wr.add(k));
            let xk = x.as_ptr().add(k * b_dim);
            let wq8 = _mm256_set1_ps(wv);
            let mut b = 0usize;
            while b + 8 <= b_dim {
                let a = _mm256_loadu_ps(acc.as_ptr().add(b));
                let xv = _mm256_loadu_ps(xk.add(b));
                _mm256_storeu_ps(acc.as_mut_ptr().add(b), _mm256_fmadd_ps(wq8, xv, a));
                b += 8;
            }
            while b < b_dim {
                acc[b] = wv.mul_add(*xk.add(b), acc[b]);
                b += 1;
            }
        }
        std::ptr::copy_nonoverlapping(acc.as_ptr(), yr, b_dim);
    }
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn bf16_matmat_slice(
    y: &mut [f32],
    w: &[u16],
    x: &[f32],
    r0: usize,
    r1: usize,
    cols: usize,
    b_dim: usize,
) {
    match b_dim {
        1 => bf16_matmat_fixed_b::<1>(y, w, x, r0, r1, cols),
        2 => bf16_matmat_fixed_b::<2>(y, w, x, r0, r1, cols),
        3 => bf16_matmat_fixed_b::<3>(y, w, x, r0, r1, cols),
        4 => bf16_matmat_fixed_b::<4>(y, w, x, r0, r1, cols),
        5 => bf16_matmat_fixed_b::<5>(y, w, x, r0, r1, cols),
        6 => bf16_matmat_fixed_b::<6>(y, w, x, r0, r1, cols),
        7 => bf16_matmat_fixed_b::<7>(y, w, x, r0, r1, cols),
        8 => bf16_matmat_fixed_b::<8>(y, w, x, r0, r1, cols),
        16 => bf16_matmat_fixed_b::<16>(y, w, x, r0, r1, cols),
        _ => bf16_matmat_generic(y, w, x, r0, r1, cols, b_dim),
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn int8_matvec_1row(
    y: *mut f32,
    qx: *const i8,
    sx: f32,
    w: *const i8,
    scale: *const f32,
    in_dim: usize,
    out_dim: usize,
) {
    ftz_on();
    let ones = _mm256_set1_epi16(1);
    for o in 0..out_dim {
        let w0 = w.add(o * in_dim);
        let mut acc0 = _mm256_setzero_si256();
        let mut k = 0usize;
        while k + 32 <= in_dim {
            _mm_prefetch::<_MM_HINT_T0>(w0.add(k + QWEN_INT8_PF_DIST));
            let a = _mm256_loadu_si256(qx.add(k) as *const __m256i);
            let v0 = _mm256_loadu_si256(w0.add(k) as *const __m256i);
            let p0 = _mm256_maddubs_epi16(_mm256_sign_epi8(v0, v0), _mm256_sign_epi8(a, v0));
            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(p0, ones));
            k += 32;
        }
        let mut s0 = hsum_epi32(acc0);
        while k < in_dim {
            s0 += (*w0.add(k) as i32) * (*qx.add(k) as i32);
            k += 1;
        }
        *y.add(o) = (s0 as f32) * *scale.add(o) * sx;
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn int8_matvec_2row(
    y: *mut f32,
    qx: *const i8,
    sx: f32,
    w: *const i8,
    scale: *const f32,
    in_dim: usize,
    out_dim: usize,
) {
    ftz_on();
    let ones = _mm256_set1_epi16(1);
    let mut o = 0usize;
    while o + 1 < out_dim {
        let w0 = w.add(o * in_dim);
        let w1 = w.add((o + 1) * in_dim);
        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();
        let mut k = 0usize;
        while k + 32 <= in_dim {
            _mm_prefetch::<_MM_HINT_T0>(w0.add(k + QWEN_INT8_PF_DIST));
            _mm_prefetch::<_MM_HINT_T0>(w1.add(k + QWEN_INT8_PF_DIST));
            let a = _mm256_loadu_si256(qx.add(k) as *const __m256i);
            let v0 = _mm256_loadu_si256(w0.add(k) as *const __m256i);
            let v1 = _mm256_loadu_si256(w1.add(k) as *const __m256i);
            let p0 = _mm256_maddubs_epi16(_mm256_sign_epi8(v0, v0), _mm256_sign_epi8(a, v0));
            let p1 = _mm256_maddubs_epi16(_mm256_sign_epi8(v1, v1), _mm256_sign_epi8(a, v1));
            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(p0, ones));
            acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(p1, ones));
            k += 32;
        }
        let mut s0 = hsum_epi32(acc0);
        let mut s1 = hsum_epi32(acc1);
        while k < in_dim {
            let qxk = *qx.add(k) as i32;
            s0 += (*w0.add(k) as i32) * qxk;
            s1 += (*w1.add(k) as i32) * qxk;
            k += 1;
        }
        *y.add(o) = (s0 as f32) * *scale.add(o) * sx;
        *y.add(o + 1) = (s1 as f32) * *scale.add(o + 1) * sx;
        o += 2;
    }
    if o < out_dim {
        int8_matvec_1row(
            y.add(o),
            qx,
            sx,
            w.add(o * in_dim),
            scale.add(o),
            in_dim,
            out_dim - o,
        );
    }
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn int8_matvec_slice(
    y: &mut [f32],
    qx: &[i8],
    sx: f32,
    w: &[i8],
    scale: &[f32],
    in_dim: usize,
    out_dim: usize,
) {
    if in_dim >= QWEN_LONG_ROW_COLS {
        int8_matvec_1row(
            y.as_mut_ptr(),
            qx.as_ptr(),
            sx,
            w.as_ptr(),
            scale.as_ptr(),
            in_dim,
            out_dim,
        );
    } else {
        int8_matvec_2row(
            y.as_mut_ptr(),
            qx.as_ptr(),
            sx,
            w.as_ptr(),
            scale.as_ptr(),
            in_dim,
            out_dim,
        );
    }
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn int8_matmat_slice(
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
    const MM_TILE: usize = 8;
    let mut yt = [0.0f32; MM_TILE];
    let mut r = r0;
    while r < r1 {
        let n = (r1 - r).min(MM_TILE);
        for b in 0..b_dim {
            int8_matvec_2row(
                yt.as_mut_ptr(),
                qxt.as_ptr().add(b * cols),
                sx[b],
                w.as_ptr().add(r * cols),
                scale.as_ptr().add(r),
                cols,
                n,
            );
            for i in 0..n {
                y[(r + i) * b_dim + b] = yt[i];
            }
        }
        r += MM_TILE;
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn q4_0_block(acc: __m256, blk: *const Q4_0Block, a: __m256i, sumq: i32) -> __m256 {
    let lomask = _mm_set1_epi8(0x0F);
    let raw = _mm_loadu_si128((*blk).qs.as_ptr() as *const __m128i);
    let lo = _mm_and_si128(raw, lomask);
    let hi = _mm_and_si128(_mm_srli_epi16::<4>(raw), lomask);
    let w = set_m128i(_mm_unpackhi_epi8(lo, hi), _mm_unpacklo_epi8(lo, hi));
    let p = _mm256_madd_epi16(_mm256_maddubs_epi16(w, a), _mm256_set1_epi16(1));
    let p = _mm256_sub_epi32(p, _mm256_set1_epi32(sumq));
    _mm256_fmadd_ps(
        _mm256_cvtepi32_ps(p),
        _mm256_set1_ps(f16_to_f32((*blk).scale_f16)),
        acc,
    )
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn q4_0_sumq(qx: *const i8, nb: usize, sumq: &mut [i32]) {
    let ones8 = _mm256_set1_epi8(1);
    let ones16 = _mm256_set1_epi16(1);
    for b in 0..nb {
        let a = _mm256_loadu_si256(qx.add(b * Q4_0_BLOCK_SIZE) as *const __m256i);
        sumq[b] = hsum_epi32(_mm256_madd_epi16(_mm256_maddubs_epi16(ones8, a), ones16));
    }
}

#[target_feature(enable = "avx2,fma")]
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
        let mut s0 = _mm256_setzero_ps();
        for b in 0..nb {
            let a = _mm256_loadu_si256(qx.add(b * Q4_0_BLOCK_SIZE) as *const __m256i);
            _mm_prefetch::<_MM_HINT_T0>(r0.add(b + 16) as *const i8);
            s0 = q4_0_block(s0, r0.add(b), a, sumq[b]);
        }
        *y.add(o) = hsum256_ps(s0) * sx;
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn q4_0_matvec_2row(
    y: *mut f32,
    qx: *const i8,
    sx: f32,
    w: *const Q4_0Block,
    cols: usize,
    out_dim: usize,
) {
    ftz_on();
    let nb = cols / Q4_0_BLOCK_SIZE;
    let mut sumq = [0i32; QWEN_QACT_MAX / Q4_0_BLOCK_SIZE];
    q4_0_sumq(qx, nb, &mut sumq[..nb]);
    let mut o = 0usize;
    while o + 1 < out_dim {
        let r0 = w.add(o * nb);
        let r1 = r0.add(nb);
        let mut s0 = _mm256_setzero_ps();
        let mut s1 = _mm256_setzero_ps();
        for b in 0..nb {
            let a = _mm256_loadu_si256(qx.add(b * Q4_0_BLOCK_SIZE) as *const __m256i);
            _mm_prefetch::<_MM_HINT_T0>(r0.add(b + 16) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(r1.add(b + 16) as *const i8);
            s0 = q4_0_block(s0, r0.add(b), a, sumq[b]);
            s1 = q4_0_block(s1, r1.add(b), a, sumq[b]);
        }
        *y.add(o) = hsum256_ps(s0) * sx;
        *y.add(o + 1) = hsum256_ps(s1) * sx;
        o += 2;
    }
    if o < out_dim {
        q4_0_rows1(y.add(o), qx, sx, w.add(o * nb), nb, &sumq[..nb], out_dim - o);
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn q4_0_matvec_1row(
    y: *mut f32,
    qx: *const i8,
    sx: f32,
    w: *const Q4_0Block,
    cols: usize,
    out_dim: usize,
) {
    ftz_on();
    let nb = cols / Q4_0_BLOCK_SIZE;
    let mut sumq = [0i32; QWEN_QACT_MAX / Q4_0_BLOCK_SIZE];
    q4_0_sumq(qx, nb, &mut sumq[..nb]);
    q4_0_rows1(y, qx, sx, w, nb, &sumq[..nb], out_dim);
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn q4_0_matvec_slice(
    y: &mut [f32],
    qx: &[i8],
    sx: f32,
    w: &[Q4_0Block],
    cols: usize,
    out_dim: usize,
) {
    if cols >= QWEN_LONG_ROW_COLS {
        q4_0_matvec_1row(y.as_mut_ptr(), qx.as_ptr(), sx, w.as_ptr(), cols, out_dim);
    } else {
        q4_0_matvec_2row(y.as_mut_ptr(), qx.as_ptr(), sx, w.as_ptr(), cols, out_dim);
    }
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn q4_0_matvec_qkv_slice(
    y: &mut [f32],
    qx: &[i8],
    sx: f32,
    w: &[Q4_0Block],
    cols: usize,
    out_dim: usize,
) {
    // QKV fused always uses 2-row kernel, matching C `q4_0_int_rows`.
    q4_0_matvec_2row(y.as_mut_ptr(), qx.as_ptr(), sx, w.as_ptr(), cols, out_dim);
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn q4_0_matmat_slice(
    y: &mut [f32],
    w: &[Q4_0Block],
    qxt: &[i8],
    sx: &[f32],
    r0: usize,
    r1: usize,
    cols: usize,
    b_dim: usize,
) {
    const MM_TILE: usize = 8;
    let nb = cols / Q4_0_BLOCK_SIZE;
    let mut yt = [0.0f32; MM_TILE];
    let mut r = r0;
    while r < r1 {
        let n = (r1 - r).min(MM_TILE);
        for b in 0..b_dim {
            q4_0_matvec_2row(
                yt.as_mut_ptr(),
                qxt.as_ptr().add(b * cols),
                sx[b],
                w.as_ptr().add(r * nb),
                cols,
                n,
            );
            for i in 0..n {
                y[(r + i) * b_dim + b] = yt[i];
            }
        }
        r += MM_TILE;
    }
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn argmax_matvec_bf16(
    x: &[f32],
    w_bf16: &[u16],
    in_dim: usize,
    out_dim: usize,
) -> usize {
    let xp = x.as_ptr();
    let wp = w_bf16.as_ptr();
    let mut best_idx = 0usize;
    let mut best_val = -1e30f32;
    let mut o = 0usize;
    while o + 1 < out_dim {
        let w0 = wp.add(o * in_dim);
        let w1 = wp.add((o + 1) * in_dim);
        if o + 5 < out_dim {
            _mm_prefetch::<_MM_HINT_NTA>(wp.add((o + 4) * in_dim) as *const i8);
            _mm_prefetch::<_MM_HINT_NTA>(wp.add((o + 5) * in_dim) as *const i8);
        }
        let (mut a0, mut a1, mut a2, mut a3) = (
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
        );
        let (mut b0, mut b1, mut b2, mut b3) = (
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
        );
        let mut k = 0usize;
        while k + 32 <= in_dim {
            let x0 = _mm256_loadu_ps(xp.add(k));
            let x1 = _mm256_loadu_ps(xp.add(k + 8));
            let x2 = _mm256_loadu_ps(xp.add(k + 16));
            let x3 = _mm256_loadu_ps(xp.add(k + 24));
            a0 = _mm256_fmadd_ps(loadu_bf16_8(w0.add(k)), x0, a0);
            a1 = _mm256_fmadd_ps(loadu_bf16_8(w0.add(k + 8)), x1, a1);
            a2 = _mm256_fmadd_ps(loadu_bf16_8(w0.add(k + 16)), x2, a2);
            a3 = _mm256_fmadd_ps(loadu_bf16_8(w0.add(k + 24)), x3, a3);
            b0 = _mm256_fmadd_ps(loadu_bf16_8(w1.add(k)), x0, b0);
            b1 = _mm256_fmadd_ps(loadu_bf16_8(w1.add(k + 8)), x1, b1);
            b2 = _mm256_fmadd_ps(loadu_bf16_8(w1.add(k + 16)), x2, b2);
            b3 = _mm256_fmadd_ps(loadu_bf16_8(w1.add(k + 24)), x3, b3);
            k += 32;
        }
        while k + 8 <= in_dim {
            let xv = _mm256_loadu_ps(xp.add(k));
            a0 = _mm256_fmadd_ps(loadu_bf16_8(w0.add(k)), xv, a0);
            b0 = _mm256_fmadd_ps(loadu_bf16_8(w1.add(k)), xv, b0);
            k += 8;
        }
        let mut s0 = hsum256_ps(_mm256_add_ps(_mm256_add_ps(a0, a2), _mm256_add_ps(a1, a3)));
        let mut s1 = hsum256_ps(_mm256_add_ps(_mm256_add_ps(b0, b2), _mm256_add_ps(b1, b3)));
        while k < in_dim {
            s0 += bf16_to_f32(*w0.add(k)) * *xp.add(k);
            s1 += bf16_to_f32(*w1.add(k)) * *xp.add(k);
            k += 1;
        }
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

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32(a: *const f32, b: *const f32, n: usize) -> f32 {
    let (mut c0, mut c1, mut c2, mut c3) = (
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
    );
    let mut d = 0usize;
    while d + 32 <= n {
        c0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(d)), _mm256_loadu_ps(b.add(d)), c0);
        c1 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(d + 8)), _mm256_loadu_ps(b.add(d + 8)), c1);
        c2 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.add(d + 16)),
            _mm256_loadu_ps(b.add(d + 16)),
            c2,
        );
        c3 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.add(d + 24)),
            _mm256_loadu_ps(b.add(d + 24)),
            c3,
        );
        d += 32;
    }
    while d + 8 <= n {
        c0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(d)), _mm256_loadu_ps(b.add(d)), c0);
        d += 8;
    }
    let mut s = hsum256_ps(_mm256_add_ps(_mm256_add_ps(c0, c2), _mm256_add_ps(c1, c3)));
    while d < n {
        s += *a.add(d) * *b.add(d);
        d += 1;
    }
    s
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_bf16(q: *const f32, k: *const u16, n: usize) -> f32 {
    let (mut c0, mut c1, mut c2, mut c3) = (
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
    );
    let mut d = 0usize;
    while d + 32 <= n {
        c0 = _mm256_fmadd_ps(_mm256_loadu_ps(q.add(d)), loadu_bf16_8(k.add(d)), c0);
        c1 = _mm256_fmadd_ps(
            _mm256_loadu_ps(q.add(d + 8)),
            loadu_bf16_8(k.add(d + 8)),
            c1,
        );
        c2 = _mm256_fmadd_ps(
            _mm256_loadu_ps(q.add(d + 16)),
            loadu_bf16_8(k.add(d + 16)),
            c2,
        );
        c3 = _mm256_fmadd_ps(
            _mm256_loadu_ps(q.add(d + 24)),
            loadu_bf16_8(k.add(d + 24)),
            c3,
        );
        d += 32;
    }
    while d + 8 <= n {
        c0 = _mm256_fmadd_ps(_mm256_loadu_ps(q.add(d)), loadu_bf16_8(k.add(d)), c0);
        d += 8;
    }
    let mut s = hsum256_ps(_mm256_add_ps(_mm256_add_ps(c0, c2), _mm256_add_ps(c1, c3)));
    while d < n {
        s += *q.add(d) * bf16_to_f32(*k.add(d));
        d += 1;
    }
    s
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn acc_corr(o: *mut f32, v: *const f32, c: f32, n: usize) {
    let vc = _mm256_set1_ps(c);
    let mut d = 0usize;
    while d + 8 <= n {
        _mm256_storeu_ps(
            o.add(d),
            _mm256_fmadd_ps(_mm256_loadu_ps(o.add(d)), vc, _mm256_loadu_ps(v.add(d))),
        );
        d += 8;
    }
    while d < n {
        *o.add(d) = *o.add(d) * c + *v.add(d);
        d += 1;
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn acc_wt(o: *mut f32, v: *const f32, w: f32, n: usize) {
    let vw = _mm256_set1_ps(w);
    let mut d = 0usize;
    while d + 8 <= n {
        _mm256_storeu_ps(
            o.add(d),
            _mm256_fmadd_ps(_mm256_loadu_ps(v.add(d)), vw, _mm256_loadu_ps(o.add(d))),
        );
        d += 8;
    }
    while d < n {
        *o.add(d) += *v.add(d) * w;
        d += 1;
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn scale_vec(o: *mut f32, s: f32, n: usize) {
    let vs = _mm256_set1_ps(s);
    let mut d = 0usize;
    while d + 8 <= n {
        _mm256_storeu_ps(o.add(d), _mm256_mul_ps(_mm256_loadu_ps(o.add(d)), vs));
        d += 8;
    }
    while d < n {
        *o.add(d) *= s;
        d += 1;
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn acc_corr_bf16(o: *mut f32, v: *const u16, c: f32, n: usize) {
    let vc = _mm256_set1_ps(c);
    let mut d = 0usize;
    while d + 8 <= n {
        _mm256_storeu_ps(
            o.add(d),
            _mm256_fmadd_ps(_mm256_loadu_ps(o.add(d)), vc, loadu_bf16_8(v.add(d))),
        );
        d += 8;
    }
    while d < n {
        *o.add(d) = *o.add(d) * c + bf16_to_f32(*v.add(d));
        d += 1;
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn acc_wt_bf16(o: *mut f32, v: *const u16, w: f32, n: usize) {
    let vw = _mm256_set1_ps(w);
    let mut d = 0usize;
    while d + 8 <= n {
        _mm256_storeu_ps(
            o.add(d),
            _mm256_fmadd_ps(loadu_bf16_8(v.add(d)), vw, _mm256_loadu_ps(o.add(d))),
        );
        d += 8;
    }
    while d < n {
        *o.add(d) += bf16_to_f32(*v.add(d)) * w;
        d += 1;
    }
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn causal_attention_f32_heads(
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
    let q_hidden = n_heads * head_dim;
    let kv_hidden = n_kv_heads * head_dim;
    let qp = q.as_ptr();
    let kp = k.as_ptr();
    let vp = v.as_ptr();
    let op = out.as_mut_ptr();

    for h in 0..n_heads {
        let kv_h = h / heads_per_kv;
        for i in 0..seq_q {
            let q_row = qp.add(i * q_hidden + h * head_dim);
            let o_row = op.add(i * q_hidden + h * head_dim);
            let k_end = (q_offset + i + 1).min(seq_k);
            let k_start = if window > 0 && k_end > window {
                k_end - window
            } else {
                0
            };

            let mut max_score = -1e30f32;
            let mut sum_exp = 0.0f32;
            std::ptr::write_bytes(o_row, 0, head_dim);

            for j in k_start..k_end {
                let k_row = kp.add(j * kv_hidden + kv_h * head_dim);
                let v_row = vp.add(j * kv_hidden + kv_h * head_dim);
                let score = dot_f32(q_row, k_row, head_dim) * scale;
                if score > max_score {
                    let correction = (max_score - score).exp();
                    sum_exp = sum_exp * correction + 1.0;
                    acc_corr(o_row, v_row, correction, head_dim);
                    max_score = score;
                } else {
                    let wt = (score - max_score).exp();
                    sum_exp += wt;
                    acc_wt(o_row, v_row, wt, head_dim);
                }
            }
            if sum_exp > 0.0 {
                scale_vec(o_row, 1.0f32 / sum_exp, head_dim);
            }
        }
    }
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn causal_attention_bf16kv_heads(
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
    let heads_per_kv = n_heads / n_kv_heads;
    let q_hidden = n_heads * head_dim;
    let kv_hidden = n_kv_heads * head_dim;
    let qp = q.as_ptr();
    let kp = k_bf16.as_ptr();
    let vp = v_bf16.as_ptr();
    let op = out.as_mut_ptr();

    for h in h_begin..h_end {
        let kv_h = h / heads_per_kv;
        for i in 0..seq_q {
            let q_row = qp.add(i * q_hidden + h * head_dim);
            let o_row = op.add(i * q_hidden + h * head_dim);
            let k_end = (q_offset + i + 1).min(seq_k);

            let mut max_score = -1e30f32;
            let mut sum_exp = 0.0f32;
            std::ptr::write_bytes(o_row, 0, head_dim);

            for j in 0..k_end {
                let k_row = kp.add(j * kv_hidden + kv_h * head_dim);
                let v_row = vp.add(j * kv_hidden + kv_h * head_dim);
                let score = dot_f32_bf16(q_row, k_row, head_dim) * scale;
                if score > max_score {
                    let correction = (max_score - score).exp();
                    sum_exp = sum_exp * correction + 1.0;
                    acc_corr_bf16(o_row, v_row, correction, head_dim);
                    max_score = score;
                } else {
                    let wt = (score - max_score).exp();
                    sum_exp += wt;
                    acc_wt_bf16(o_row, v_row, wt, head_dim);
                }
            }
            if sum_exp > 0.0 {
                scale_vec(o_row, 1.0f32 / sum_exp, head_dim);
            }
        }
    }
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn apply_rope_neox_inplace(
    x: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    cos_cache: &[f32],
    sin_cache: &[f32],
    pos: usize,
) {
    let half = head_dim / 2;
    let cos_ptr = cos_cache.as_ptr().add(pos * half);
    let sin_ptr = sin_cache.as_ptr().add(pos * half);
    let xp = x.as_mut_ptr();
    for h in 0..n_heads {
        let xh = xp.add(h * head_dim);
        let mut i = 0usize;
        while i + 8 <= half {
            let c = _mm256_loadu_ps(cos_ptr.add(i));
            let s = _mm256_loadu_ps(sin_ptr.add(i));
            let v1 = _mm256_loadu_ps(xh.add(i));
            let v2 = _mm256_loadu_ps(xh.add(i + half));
            _mm256_storeu_ps(xh.add(i), _mm256_fmsub_ps(v1, c, _mm256_mul_ps(v2, s)));
            _mm256_storeu_ps(xh.add(i + half), _mm256_fmadd_ps(v2, c, _mm256_mul_ps(v1, s)));
            i += 8;
        }
        while i < half {
            let x1 = *xh.add(i);
            let x2 = *xh.add(i + half);
            let ci = *cos_ptr.add(i);
            let si = *sin_ptr.add(i);
            *xh.add(i) = x1 * ci - x2 * si;
            *xh.add(i + half) = x2 * ci + x1 * si;
            i += 1;
        }
    }
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn bf16_accum_f32(dst: &mut [f32], src_bf16: &[u16]) {
    let n = dst.len().min(src_bf16.len());
    let dp = dst.as_mut_ptr();
    let sp = src_bf16.as_ptr();
    let mut i = 0usize;
    while i + 7 < n {
        let f = loadu_bf16_8(sp.add(i));
        let d = _mm256_loadu_ps(dp.add(i));
        _mm256_storeu_ps(dp.add(i), _mm256_add_ps(d, f));
        i += 8;
    }
    while i < n {
        *dp.add(i) += bf16_to_f32(*sp.add(i));
        i += 1;
    }
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn bf16_to_f32_vec(dst: &mut [f32], src_bf16: &[u16]) {
    let n = dst.len().min(src_bf16.len());
    let dp = dst.as_mut_ptr();
    let sp = src_bf16.as_ptr();
    let mut i = 0usize;
    while i + 7 < n {
        _mm256_storeu_ps(dp.add(i), loadu_bf16_8(sp.add(i)));
        i += 8;
    }
    while i < n {
        *dp.add(i) = bf16_to_f32(*sp.add(i));
        i += 1;
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn vsin2_avx2(y: __m256) -> __m256 {
    const QWEN_SIN_C1: f32 = -1.0 / 6.0;
    const QWEN_SIN_C2: f32 = 1.0 / 120.0;
    const QWEN_SIN_C3: f32 = -1.0 / 5040.0;
    const QWEN_SIN_C4: f32 = 1.0 / 362880.0;
    const QWEN_SIN_C5: f32 = -1.0 / 39916800.0;
    const QWEN_PI_HI: f32 = std::f32::consts::PI;
    const QWEN_PI_LO: f32 = -8.742_278e-8;
    const QWEN_INV_PI: f32 = std::f32::consts::FRAC_1_PI;

    let inv_pi = _mm256_set1_ps(QWEN_INV_PI);
    let pi_hi = _mm256_set1_ps(QWEN_PI_HI);
    let pi_lo = _mm256_set1_ps(QWEN_PI_LO);
    let n = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
        _mm256_mul_ps(y, inv_pi),
    );
    let u = _mm256_fnmadd_ps(n, pi_hi, y);
    let u = _mm256_fmadd_ps(n, pi_lo, u);
    let u2 = _mm256_mul_ps(u, u);
    let mut p = _mm256_set1_ps(QWEN_SIN_C5);
    p = _mm256_fmadd_ps(p, u2, _mm256_set1_ps(QWEN_SIN_C4));
    p = _mm256_fmadd_ps(p, u2, _mm256_set1_ps(QWEN_SIN_C3));
    p = _mm256_fmadd_ps(p, u2, _mm256_set1_ps(QWEN_SIN_C2));
    p = _mm256_fmadd_ps(p, u2, _mm256_set1_ps(QWEN_SIN_C1));
    p = _mm256_fmadd_ps(p, u2, _mm256_set1_ps(1.0));
    let s = _mm256_mul_ps(u, p);
    _mm256_mul_ps(s, s)
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn f32_to_bf16_vec(dst: &mut [u16], src: &[f32]) {
    let n = dst.len().min(src.len());
    let dp = dst.as_mut_ptr();
    let sp = src.as_ptr();
    let mut i = 0usize;
    while i + 8 <= n {
        let u = _mm256_srli_epi32::<16>(_mm256_castps_si256(_mm256_loadu_ps(sp.add(i))));
        let packed = _mm_packus_epi32(
            _mm256_castsi256_si128(u),
            _mm256_extracti128_si256::<1>(u),
        );
        _mm_storeu_si128(dp.add(i) as *mut __m128i, packed);
        i += 8;
    }
    while i < n {
        *dp.add(i) = (*sp.add(i).cast::<u32>() >> 16) as u16;
        i += 1;
    }
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn f32_matvec_accum(y: &mut [f32], w: &[f32], x: &[f32], rows: usize, cols: usize) {
    let wp = w.as_ptr();
    let xp = x.as_ptr();
    let yp = y.as_mut_ptr();
    for r in 0..rows {
        let row = wp.add(r * cols);
        *yp.add(r) += dot_f32(row, xp, cols);
    }
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn snake_row(row: &mut [f32], log_alpha: f32, log_beta: f32) {
    const QWEN_SIN_POLY_MAX: f32 = 8192.0;
    let a = log_alpha.exp();
    let inv_b = (-log_beta).exp();
    let len = row.len();
    let rp = row.as_mut_ptr();
    let va = _mm256_set1_ps(a);
    let vinv_b = _mm256_set1_ps(inv_b);
    let sign_mask = _mm256_set1_ps(-0.0f32);
    let poly_max = _mm256_set1_ps(QWEN_SIN_POLY_MAX);
    let mut t = 0usize;
    while t + 8 <= len {
        let x = _mm256_loadu_ps(rp.add(t));
        let ax = _mm256_mul_ps(va, x);
        let over = _mm256_cmp_ps::<_CMP_GT_OQ>(_mm256_andnot_ps(sign_mask, ax), poly_max);
        let s2 = if _mm256_movemask_ps(over) == 0 {
            vsin2_avx2(ax)
        } else {
            let mut ax_s = [0.0f32; 8];
            _mm256_storeu_ps(ax_s.as_mut_ptr(), ax);
            let s_arr = [
                ax_s[0].sin(),
                ax_s[1].sin(),
                ax_s[2].sin(),
                ax_s[3].sin(),
                ax_s[4].sin(),
                ax_s[5].sin(),
                ax_s[6].sin(),
                ax_s[7].sin(),
            ];
            let s = _mm256_loadu_ps(s_arr.as_ptr());
            _mm256_mul_ps(s, s)
        };
        _mm256_storeu_ps(rp.add(t), _mm256_fmadd_ps(vinv_b, s2, x));
        t += 8;
    }
    while t < len {
        let s = (a * *rp.add(t)).sin();
        *rp.add(t) += inv_b * s * s;
        t += 1;
    }
}
