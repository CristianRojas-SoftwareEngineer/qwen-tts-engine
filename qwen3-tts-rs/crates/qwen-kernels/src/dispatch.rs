use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::blas::{BlasBackend, FfiBlas};
use crate::caps::CpuCaps;
use crate::ftz::ftz_on;
use crate::pool::{default_num_threads, ComputePool};
use crate::quant::{Q2_0Block, Q4_0Block, Q4_0_BLOCK_SIZE, QWEN_QACT_MAX};
use crate::simd::scalar;

#[cfg(target_arch = "x86_64")]
use crate::simd::avx2;

#[cfg(target_arch = "aarch64")]
use crate::simd::neon;

pub type QuantizeActFn = fn(&mut [i8], &[f32]) -> f32;
pub type RmsNormFn = fn(&mut [f32], &[f32], &[f32], usize, usize, f32);
pub type RmsNormResidualFn = fn(&mut [f32], &mut [f32], &[f32], &[f32], usize, f32);
pub type RmsNormPerHeadFn = fn(&mut [f32], &[f32], usize, usize, usize, f32);
pub type Bf16MatvecSliceFn = fn(&mut [f32], &[f32], &[u16], usize, usize);
pub type Bf16MatmatSliceFn = fn(&mut [f32], &[u16], &[f32], usize, usize, usize, usize);
pub type Int8MatvecSliceFn = fn(&mut [f32], &[i8], f32, &[i8], &[f32], usize, usize);
pub type Int8MatmatSliceFn =
    fn(&mut [f32], &[i8], &[f32], &[i8], &[f32], usize, usize, usize, usize);
pub type Q4MatvecSliceFn = fn(&mut [f32], &[i8], f32, &[Q4_0Block], usize, usize);
pub type Q4MatmatSliceFn = fn(&mut [f32], &[Q4_0Block], &[i8], &[f32], usize, usize, usize, usize);
pub type ArgmaxBf16Fn = fn(&[f32], &[u16], usize, usize) -> usize;
pub type AttnF32Fn =
    fn(&mut [f32], &[f32], &[f32], &[f32], usize, usize, usize, usize, usize, f32, usize, usize);
pub type AttnBf16KvFn = fn(
    &mut [f32],
    &[f32],
    &[u16],
    &[u16],
    usize,
    usize,
    usize,
    usize,
    usize,
    f32,
    usize,
    usize,
    usize,
);
pub type RopeNeoxFn = fn(&mut [f32], usize, usize, &[f32], &[f32], usize);
pub type Bf16VecOpFn = fn(&mut [f32], &[u16]);
pub type F32ToBf16VecFn = fn(&mut [u16], &[f32]);
pub type F32MatvecAccumFn = fn(&mut [f32], &[f32], &[f32], usize, usize);
pub type SnakeRowFn = fn(&mut [f32], f32, f32);

pub struct KernelDispatch {
    pub quantize_act_int8_fn: QuantizeActFn,
    pub rms_norm_fn: RmsNormFn,
    pub rms_norm_residual_fn: RmsNormResidualFn,
    pub rms_norm_per_head_fn: RmsNormPerHeadFn,
    pub bf16_matvec_slice_fn: Bf16MatvecSliceFn,
    pub bf16_matmat_slice_fn: Bf16MatmatSliceFn,
    pub int8_matvec_slice_fn: Int8MatvecSliceFn,
    pub int8_matmat_slice_fn: Int8MatmatSliceFn,
    pub q4_0_matvec_slice_fn: Q4MatvecSliceFn,
    pub q4_0_matvec_qkv_slice_fn: Q4MatvecSliceFn,
    pub q4_0_matmat_slice_fn: Q4MatmatSliceFn,
    pub argmax_bf16_fn: ArgmaxBf16Fn,
    pub attn_f32_fn: AttnF32Fn,
    pub attn_bf16kv_fn: AttnBf16KvFn,
    pub rope_neox_fn: RopeNeoxFn,
    pub bf16_accum_f32_fn: Bf16VecOpFn,
    pub bf16_to_f32_vec_fn: Bf16VecOpFn,
    pub f32_to_bf16_vec_fn: F32ToBf16VecFn,
    pub f32_matvec_accum_fn: F32MatvecAccumFn,
    pub snake_row_fn: SnakeRowFn,
}

static SCALAR_DISPATCH: KernelDispatch = KernelDispatch {
    quantize_act_int8_fn: scalar::quantize_act_int8,
    rms_norm_fn: scalar::rms_norm,
    rms_norm_residual_fn: scalar::rms_norm_residual,
    rms_norm_per_head_fn: scalar::rms_norm_per_head,
    bf16_matvec_slice_fn: scalar::bf16_matvec_slice,
    bf16_matmat_slice_fn: scalar::bf16_matmat_slice,
    int8_matvec_slice_fn: scalar::int8_matvec_slice,
    int8_matmat_slice_fn: scalar::int8_matmat_slice,
    q4_0_matvec_slice_fn: scalar::q4_0_matvec_slice,
    q4_0_matvec_qkv_slice_fn: scalar::q4_0_matvec_slice,
    q4_0_matmat_slice_fn: scalar::q4_0_matmat_slice,
    argmax_bf16_fn: scalar::argmax_matvec_bf16,
    attn_f32_fn: scalar::causal_attention_f32_heads,
    attn_bf16kv_fn: scalar::causal_attention_bf16kv_heads,
    rope_neox_fn: scalar::apply_rope_neox_inplace,
    bf16_accum_f32_fn: scalar::bf16_accum_f32,
    bf16_to_f32_vec_fn: scalar::bf16_to_f32_vec,
    f32_to_bf16_vec_fn: scalar::f32_to_bf16_vec,
    f32_matvec_accum_fn: scalar::f32_matvec_accum,
    snake_row_fn: scalar::snake_row,
};

#[cfg(target_arch = "x86_64")]
mod avx2_wrappers {
    use super::*;
    pub fn quantize_act_int8(qx: &mut [i8], x: &[f32]) -> f32 {
        unsafe { avx2::quantize_act_int8(qx, x) }
    }
    pub fn rms_norm(out: &mut [f32], x: &[f32], weight: &[f32], seq: usize, dim: usize, eps: f32) {
        unsafe { avx2::rms_norm(out, x, weight, seq, dim, eps) }
    }
    pub fn rms_norm_residual(
        out: &mut [f32],
        x: &mut [f32],
        residual: &[f32],
        weight: &[f32],
        dim: usize,
        eps: f32,
    ) {
        unsafe { avx2::rms_norm_residual(out, x, residual, weight, dim, eps) }
    }
    pub fn rms_norm_per_head(
        x: &mut [f32],
        weight: &[f32],
        seq: usize,
        n_heads: usize,
        head_dim: usize,
        eps: f32,
    ) {
        unsafe { avx2::rms_norm_per_head(x, weight, seq, n_heads, head_dim, eps) }
    }
    pub fn bf16_matvec_slice(y: &mut [f32], x: &[f32], w: &[u16], cols: usize, rows: usize) {
        unsafe { avx2::bf16_matvec_slice(y, x, w, cols, rows) }
    }
    pub fn bf16_matmat_slice(
        y: &mut [f32],
        w: &[u16],
        x: &[f32],
        r0: usize,
        r1: usize,
        cols: usize,
        b_dim: usize,
    ) {
        unsafe { avx2::bf16_matmat_slice(y, w, x, r0, r1, cols, b_dim) }
    }
    pub fn int8_matvec_slice(
        y: &mut [f32],
        qx: &[i8],
        sx: f32,
        w: &[i8],
        scale: &[f32],
        cols: usize,
        rows: usize,
    ) {
        unsafe { avx2::int8_matvec_slice(y, qx, sx, w, scale, cols, rows) }
    }
    pub fn int8_matmat_slice(
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
        unsafe { avx2::int8_matmat_slice(y, w, scale, qxt, sx, r0, r1, cols, b_dim) }
    }
    pub fn q4_0_matvec_slice(
        y: &mut [f32],
        qx: &[i8],
        sx: f32,
        w: &[Q4_0Block],
        cols: usize,
        rows: usize,
    ) {
        unsafe { avx2::q4_0_matvec_slice(y, qx, sx, w, cols, rows) }
    }
    pub fn q4_0_matvec_qkv_slice(
        y: &mut [f32],
        qx: &[i8],
        sx: f32,
        w: &[Q4_0Block],
        cols: usize,
        rows: usize,
    ) {
        unsafe { avx2::q4_0_matvec_qkv_slice(y, qx, sx, w, cols, rows) }
    }
    pub fn q4_0_matmat_slice(
        y: &mut [f32],
        w: &[Q4_0Block],
        qxt: &[i8],
        sx: &[f32],
        r0: usize,
        r1: usize,
        cols: usize,
        b_dim: usize,
    ) {
        unsafe { avx2::q4_0_matmat_slice(y, w, qxt, sx, r0, r1, cols, b_dim) }
    }
    pub fn argmax_bf16(x: &[f32], w_bf16: &[u16], in_dim: usize, out_dim: usize) -> usize {
        unsafe { avx2::argmax_matvec_bf16(x, w_bf16, in_dim, out_dim) }
    }
    pub fn attn_f32(
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
        unsafe {
            avx2::causal_attention_f32_heads(
                out, q, k, v, seq_q, seq_k, n_heads, n_kv_heads, head_dim, scale, q_offset, window,
            )
        }
    }
    pub fn attn_bf16kv(
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
        unsafe {
            avx2::causal_attention_bf16kv_heads(
                out, q, k_bf16, v_bf16, seq_q, seq_k, n_heads, n_kv_heads, head_dim, scale,
                q_offset, h_begin, h_end,
            )
        }
    }
    pub fn rope_neox(
        x: &mut [f32],
        n_heads: usize,
        head_dim: usize,
        cos_cache: &[f32],
        sin_cache: &[f32],
        pos: usize,
    ) {
        unsafe { avx2::apply_rope_neox_inplace(x, n_heads, head_dim, cos_cache, sin_cache, pos) }
    }
    pub fn bf16_accum_f32(dst: &mut [f32], src_bf16: &[u16]) {
        unsafe { avx2::bf16_accum_f32(dst, src_bf16) }
    }
    pub fn bf16_to_f32_vec(dst: &mut [f32], src_bf16: &[u16]) {
        unsafe { avx2::bf16_to_f32_vec(dst, src_bf16) }
    }
    pub fn f32_to_bf16_vec(dst: &mut [u16], src: &[f32]) {
        unsafe { avx2::f32_to_bf16_vec(dst, src) }
    }
    pub fn f32_matvec_accum(y: &mut [f32], w: &[f32], x: &[f32], rows: usize, cols: usize) {
        unsafe { avx2::f32_matvec_accum(y, w, x, rows, cols) }
    }
    pub fn snake_row(row: &mut [f32], log_alpha: f32, log_beta: f32) {
        unsafe { avx2::snake_row(row, log_alpha, log_beta) }
    }
}

#[cfg(target_arch = "x86_64")]
static AVX2_DISPATCH: KernelDispatch = KernelDispatch {
    quantize_act_int8_fn: avx2_wrappers::quantize_act_int8,
    rms_norm_fn: avx2_wrappers::rms_norm,
    rms_norm_residual_fn: avx2_wrappers::rms_norm_residual,
    rms_norm_per_head_fn: avx2_wrappers::rms_norm_per_head,
    bf16_matvec_slice_fn: avx2_wrappers::bf16_matvec_slice,
    bf16_matmat_slice_fn: avx2_wrappers::bf16_matmat_slice,
    int8_matvec_slice_fn: avx2_wrappers::int8_matvec_slice,
    int8_matmat_slice_fn: avx2_wrappers::int8_matmat_slice,
    q4_0_matvec_slice_fn: avx2_wrappers::q4_0_matvec_slice,
    q4_0_matvec_qkv_slice_fn: avx2_wrappers::q4_0_matvec_qkv_slice,
    q4_0_matmat_slice_fn: avx2_wrappers::q4_0_matmat_slice,
    argmax_bf16_fn: avx2_wrappers::argmax_bf16,
    attn_f32_fn: avx2_wrappers::attn_f32,
    attn_bf16kv_fn: avx2_wrappers::attn_bf16kv,
    rope_neox_fn: avx2_wrappers::rope_neox,
    bf16_accum_f32_fn: avx2_wrappers::bf16_accum_f32,
    bf16_to_f32_vec_fn: avx2_wrappers::bf16_to_f32_vec,
    f32_to_bf16_vec_fn: avx2_wrappers::f32_to_bf16_vec,
    f32_matvec_accum_fn: avx2_wrappers::f32_matvec_accum,
    snake_row_fn: avx2_wrappers::snake_row,
};

#[cfg(target_arch = "aarch64")]
static NEON_DISPATCH: KernelDispatch = KernelDispatch {
    quantize_act_int8_fn: neon::quantize_act_int8,
    rms_norm_fn: neon::rms_norm,
    rms_norm_residual_fn: neon::rms_norm_residual,
    rms_norm_per_head_fn: neon::rms_norm_per_head,
    bf16_matvec_slice_fn: neon::bf16_matvec_slice,
    bf16_matmat_slice_fn: neon::bf16_matmat_slice,
    int8_matvec_slice_fn: neon::int8_matvec_slice,
    int8_matmat_slice_fn: neon::int8_matmat_slice,
    q4_0_matvec_slice_fn: neon::q4_0_matvec_slice,
    q4_0_matvec_qkv_slice_fn: neon::q4_0_matvec_slice,
    q4_0_matmat_slice_fn: neon::q4_0_matmat_slice,
    argmax_bf16_fn: neon::argmax_matvec_bf16,
    attn_f32_fn: neon::causal_attention_f32_heads,
    attn_bf16kv_fn: neon::causal_attention_bf16kv_heads,
    rope_neox_fn: neon::apply_rope_neox_inplace,
    bf16_accum_f32_fn: neon::bf16_accum_f32,
    bf16_to_f32_vec_fn: neon::bf16_to_f32_vec,
    f32_to_bf16_vec_fn: neon::f32_to_bf16_vec,
    f32_matvec_accum_fn: neon::f32_matvec_accum,
    snake_row_fn: neon::snake_row,
};

impl KernelDispatch {
    pub fn for_caps(caps: &CpuCaps) -> &'static Self {
        #[cfg(target_arch = "x86_64")]
        if caps.has_avx2_fma() {
            return &AVX2_DISPATCH;
        }
        #[cfg(target_arch = "aarch64")]
        if caps.neon {
            return &NEON_DISPATCH;
        }
        let _ = caps;
        &SCALAR_DISPATCH
    }

    pub fn detect() -> &'static Self {
        Self::for_caps(&CpuCaps::detect())
    }

    pub fn scalar() -> &'static Self {
        &SCALAR_DISPATCH
    }

    #[inline]
    pub fn quantize_act_int8(&self, qx: &mut [i8], x: &[f32]) -> f32 {
        (self.quantize_act_int8_fn)(qx, x)
    }

    pub fn quantize_act_int8_cols(
        &self,
        qxt: &mut [i8],
        sx: &mut [f32],
        x: &[f32],
        cols: usize,
        b_dim: usize,
    ) {
        let mut col_uninit = std::mem::MaybeUninit::<[f32; QWEN_QACT_MAX]>::uninit();
        let col = unsafe {
            std::slice::from_raw_parts_mut(col_uninit.as_mut_ptr() as *mut f32, cols)
        };
        for b in 0..b_dim {
            for k in 0..cols {
                col[k] = x[k * b_dim + b];
            }
            sx[b] = self.quantize_act_int8(&mut qxt[b * cols..(b + 1) * cols], col);
        }
    }

    #[inline]
    pub fn rms_norm(
        &self,
        out: &mut [f32],
        x: &[f32],
        weight: &[f32],
        seq: usize,
        dim: usize,
        eps: f32,
    ) {
        (self.rms_norm_fn)(out, x, weight, seq, dim, eps);
    }

    #[inline]
    pub fn rms_norm_residual(
        &self,
        out: &mut [f32],
        x: &mut [f32],
        residual: &[f32],
        weight: &[f32],
        dim: usize,
        eps: f32,
    ) {
        (self.rms_norm_residual_fn)(out, x, residual, weight, dim, eps);
    }

    #[inline]
    pub fn rms_norm_per_head(
        &self,
        x: &mut [f32],
        weight: &[f32],
        seq: usize,
        n_heads: usize,
        head_dim: usize,
        eps: f32,
    ) {
        (self.rms_norm_per_head_fn)(x, weight, seq, n_heads, head_dim, eps);
    }

    pub fn matvec_bf16(
        &self,
        pool: &ComputePool,
        y: &mut [f32],
        w: &[u16],
        x: &[f32],
        rows: usize,
        cols: usize,
    ) {
        let nt = pool.n_threads();
        if nt > 1 && rows >= 256 {
            let f = self.bf16_matvec_slice_fn;
            let y_ptr = y.as_mut_ptr() as usize;
            let x_ref = &x[..cols];
            pool.parallel(nt, |tid, n| {
                let r0 = tid * rows / n;
                let r1 = (tid + 1) * rows / n;
                let y_slice = unsafe {
                    std::slice::from_raw_parts_mut((y_ptr as *mut f32).add(r0), r1 - r0)
                };
                f(y_slice, x_ref, &w[r0 * cols..r1 * cols], cols, r1 - r0);
            });
            return;
        }
        (self.bf16_matvec_slice_fn)(&mut y[..rows], &x[..cols], &w[..rows * cols], cols, rows);
    }

    pub fn matmat_bf16(
        &self,
        pool: &ComputePool,
        y: &mut [f32],
        w: &[u16],
        x: &[f32],
        rows: usize,
        cols: usize,
        b_dim: usize,
    ) {
        if b_dim == 0 {
            return;
        }
        let b_dim = b_dim.min(64);
        let nt = pool.n_threads();
        if nt > 1 && rows >= 256 {
            let f = self.bf16_matmat_slice_fn;
            let y_ptr = y.as_mut_ptr() as usize;
            let y_len = rows * b_dim;
            pool.parallel(nt, |tid, n| {
                let r0 = tid * rows / n;
                let r1 = (tid + 1) * rows / n;
                let y_slice = unsafe { std::slice::from_raw_parts_mut(y_ptr as *mut f32, y_len) };
                f(y_slice, w, x, r0, r1, cols, b_dim);
            });
            return;
        }
        (self.bf16_matmat_slice_fn)(y, w, x, 0, rows, cols, b_dim);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn matvec_bf16_qkv(
        &self,
        pool: &ComputePool,
        q: &mut [f32],
        k: &mut [f32],
        v: &mut [f32],
        wq: &[u16],
        wk: &[u16],
        wv: &[u16],
        x: &[f32],
        in_dim: usize,
        q_dim: usize,
        kv_dim: usize,
    ) {
        let nt = pool.n_threads();
        let total_dim = q_dim + 2 * kv_dim;
        if nt > 1 && total_dim >= 256 {
            let f = self.bf16_matvec_slice_fn;
            let qp = q.as_mut_ptr() as usize;
            let kp = k.as_mut_ptr() as usize;
            let vp = v.as_mut_ptr() as usize;
            let x_ref = &x[..in_dim];
            pool.parallel(nt, |tid, n| {
                let r0 = tid * total_dim / n;
                let r1 = (tid + 1) * total_dim / n;
                let mut r = r0;
                while r < r1 {
                    if r < q_dim {
                        let end = r1.min(q_dim);
                        let len = end - r;
                        let out = unsafe {
                            std::slice::from_raw_parts_mut((qp as *mut f32).add(r), len)
                        };
                        f(out, x_ref, &wq[r * in_dim..end * in_dim], in_dim, len);
                        r = end;
                    } else if r < q_dim + kv_dim {
                        let local = r - q_dim;
                        let end = r1.min(q_dim + kv_dim);
                        let local_end = end - q_dim;
                        let len = local_end - local;
                        let out = unsafe {
                            std::slice::from_raw_parts_mut((kp as *mut f32).add(local), len)
                        };
                        f(
                            out,
                            x_ref,
                            &wk[local * in_dim..local_end * in_dim],
                            in_dim,
                            len,
                        );
                        r = end;
                    } else {
                        let local = r - q_dim - kv_dim;
                        let local_end = r1 - q_dim - kv_dim;
                        let len = local_end - local;
                        let out = unsafe {
                            std::slice::from_raw_parts_mut((vp as *mut f32).add(local), len)
                        };
                        f(
                            out,
                            x_ref,
                            &wv[local * in_dim..local_end * in_dim],
                            in_dim,
                            len,
                        );
                        r = r1;
                    }
                }
            });
            return;
        }
        (self.bf16_matvec_slice_fn)(&mut q[..q_dim], &x[..in_dim], wq, in_dim, q_dim);
        (self.bf16_matvec_slice_fn)(&mut k[..kv_dim], &x[..in_dim], wk, in_dim, kv_dim);
        (self.bf16_matvec_slice_fn)(&mut v[..kv_dim], &x[..in_dim], wv, in_dim, kv_dim);
    }

    pub fn matvec_int8(
        &self,
        pool: &ComputePool,
        y: &mut [f32],
        w: &[i8],
        scale: &[f32],
        x: &[f32],
        rows: usize,
        cols: usize,
    ) {
        debug_assert!(cols > 0 && cols <= QWEN_QACT_MAX && cols % Q4_0_BLOCK_SIZE == 0);
        let mut qx_uninit = std::mem::MaybeUninit::<[i8; QWEN_QACT_MAX]>::uninit();
        let qx_slice = unsafe {
            std::slice::from_raw_parts_mut(qx_uninit.as_mut_ptr() as *mut i8, cols)
        };
        let sx = self.quantize_act_int8(qx_slice, &x[..cols]);
        let nt = pool.n_threads();
        if nt > 1 && rows >= 256 {
            let f = self.int8_matvec_slice_fn;
            let y_ptr = y.as_mut_ptr() as usize;
            let qx_ref: &[i8] = qx_slice;
            pool.parallel(nt, |tid, n| {
                let r0 = tid * rows / n;
                let r1 = (tid + 1) * rows / n;
                let y_slice = unsafe {
                    std::slice::from_raw_parts_mut((y_ptr as *mut f32).add(r0), r1 - r0)
                };
                f(
                    y_slice,
                    qx_ref,
                    sx,
                    &w[r0 * cols..r1 * cols],
                    &scale[r0..r1],
                    cols,
                    r1 - r0,
                );
            });
            return;
        }
        (self.int8_matvec_slice_fn)(&mut y[..rows], qx_slice, sx, w, scale, cols, rows);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn matmat_int8(
        &self,
        pool: &ComputePool,
        y: &mut [f32],
        w: &[i8],
        scale: &[f32],
        x: &[f32],
        rows: usize,
        cols: usize,
        b_dim: usize,
    ) {
        if b_dim == 0 {
            return;
        }
        let b_dim = b_dim.min(64);
        thread_local! {
            static QXT_INT8: RefCell<Vec<i8>> = const { RefCell::new(Vec::new()) };
        }
        QXT_INT8.with(|cell| {
            let mut qxt = cell.borrow_mut();
            let need = b_dim * cols;
            if qxt.len() < need {
                qxt.resize(need, 0);
            }
            let mut sx = [0.0f32; 64];
            self.quantize_act_int8_cols(&mut qxt[..need], &mut sx[..b_dim], x, cols, b_dim);
            let nt = pool.n_threads();
            if nt > 1 && rows >= 256 {
                let f = self.int8_matmat_slice_fn;
                let y_ptr = y.as_mut_ptr() as usize;
                let y_len = rows * b_dim;
                let sx_ref = &sx[..b_dim];
                let qxt_ref = &qxt[..need];
                pool.parallel(nt, |tid, n| {
                    let r0 = tid * rows / n;
                    let r1 = (tid + 1) * rows / n;
                    let y_slice =
                        unsafe { std::slice::from_raw_parts_mut(y_ptr as *mut f32, y_len) };
                    f(y_slice, w, scale, qxt_ref, sx_ref, r0, r1, cols, b_dim);
                });
                return;
            }
            (self.int8_matmat_slice_fn)(
                y,
                w,
                scale,
                &qxt[..need],
                &sx[..b_dim],
                0,
                rows,
                cols,
                b_dim,
            );
        });
    }

    #[allow(clippy::too_many_arguments)]
    pub fn matvec_int8_qkv(
        &self,
        pool: &ComputePool,
        q: &mut [f32],
        k: &mut [f32],
        v: &mut [f32],
        wq: &[i8],
        sq: &[f32],
        wk: &[i8],
        sk: &[f32],
        wv: &[i8],
        sv: &[f32],
        x: &[f32],
        in_dim: usize,
        q_dim: usize,
        kv_dim: usize,
    ) {
        let mut qx_uninit = std::mem::MaybeUninit::<[i8; QWEN_QACT_MAX]>::uninit();
        let qx_slice = unsafe {
            std::slice::from_raw_parts_mut(qx_uninit.as_mut_ptr() as *mut i8, in_dim)
        };
        let sx = self.quantize_act_int8(qx_slice, &x[..in_dim]);
        let nt = pool.n_threads();
        let total = q_dim + 2 * kv_dim;
        if nt > 1 {
            let f = self.int8_matvec_slice_fn;
            let qp = q.as_mut_ptr() as usize;
            let kp = k.as_mut_ptr() as usize;
            let vp = v.as_mut_ptr() as usize;
            let qx_ref: &[i8] = qx_slice;
            pool.parallel(nt, |tid, n| {
                let r0 = tid * total / n;
                let r1 = (tid + 1) * total / n;
                let mut r = r0;
                while r < r1 {
                    if r < q_dim {
                        let end = r1.min(q_dim);
                        let len = end - r;
                        let out = unsafe {
                            std::slice::from_raw_parts_mut((qp as *mut f32).add(r), len)
                        };
                        f(
                            out,
                            qx_ref,
                            sx,
                            &wq[r * in_dim..end * in_dim],
                            &sq[r..end],
                            in_dim,
                            len,
                        );
                        r = end;
                    } else if r < q_dim + kv_dim {
                        let local = r - q_dim;
                        let end = r1.min(q_dim + kv_dim);
                        let local_end = end - q_dim;
                        let len = local_end - local;
                        let out = unsafe {
                            std::slice::from_raw_parts_mut((kp as *mut f32).add(local), len)
                        };
                        f(
                            out,
                            qx_ref,
                            sx,
                            &wk[local * in_dim..local_end * in_dim],
                            &sk[local..local_end],
                            in_dim,
                            len,
                        );
                        r = end;
                    } else {
                        let local = r - q_dim - kv_dim;
                        let local_end = r1 - q_dim - kv_dim;
                        let len = local_end - local;
                        let out = unsafe {
                            std::slice::from_raw_parts_mut((vp as *mut f32).add(local), len)
                        };
                        f(
                            out,
                            qx_ref,
                            sx,
                            &wv[local * in_dim..local_end * in_dim],
                            &sv[local..local_end],
                            in_dim,
                            len,
                        );
                        r = r1;
                    }
                }
            });
            return;
        }
        (self.int8_matvec_slice_fn)(&mut q[..q_dim], qx_slice, sx, wq, sq, in_dim, q_dim);
        (self.int8_matvec_slice_fn)(&mut k[..kv_dim], qx_slice, sx, wk, sk, in_dim, kv_dim);
        (self.int8_matvec_slice_fn)(&mut v[..kv_dim], qx_slice, sx, wv, sv, in_dim, kv_dim);
    }

    pub fn argmax_matvec_int8(
        &self,
        pool: &ComputePool,
        x: &[f32],
        w: &[i8],
        scale: &[f32],
        in_dim: usize,
        out_dim: usize,
    ) -> usize {
        thread_local! {
            static LOGITS: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
        }
        LOGITS.with(|cell| {
            let mut y = cell.borrow_mut();
            if y.len() < out_dim {
                y.resize(out_dim, 0.0);
            }
            self.matvec_int8(pool, &mut y[..out_dim], w, scale, x, out_dim, in_dim);
            let mut best = 0usize;
            let mut best_val = y[0];
            for (o, &val) in y[..out_dim].iter().enumerate().skip(1) {
                if val > best_val {
                    best_val = val;
                    best = o;
                }
            }
            best
        })
    }

    pub fn matvec_q4_0(
        &self,
        pool: &ComputePool,
        y: &mut [f32],
        w: &[Q4_0Block],
        x: &[f32],
        rows: usize,
        cols: usize,
    ) {
        debug_assert!(cols > 0 && cols <= QWEN_QACT_MAX && cols % Q4_0_BLOCK_SIZE == 0);
        let mut qx_uninit = std::mem::MaybeUninit::<[i8; QWEN_QACT_MAX]>::uninit();
        let qx_slice = unsafe {
            std::slice::from_raw_parts_mut(qx_uninit.as_mut_ptr() as *mut i8, cols)
        };
        let sx = self.quantize_act_int8(qx_slice, &x[..cols]);
        let nb = cols / Q4_0_BLOCK_SIZE;
        let nt = pool.n_threads();
        if nt > 1 && rows >= 256 {
            let f = self.q4_0_matvec_slice_fn;
            let y_ptr = y.as_mut_ptr() as usize;
            let qx_ref: &[i8] = qx_slice;
            pool.parallel(nt, |tid, n| {
                let r0 = tid * rows / n;
                let r1 = (tid + 1) * rows / n;
                let y_slice = unsafe {
                    std::slice::from_raw_parts_mut((y_ptr as *mut f32).add(r0), r1 - r0)
                };
                f(y_slice, qx_ref, sx, &w[r0 * nb..r1 * nb], cols, r1 - r0);
            });
            return;
        }
        (self.q4_0_matvec_slice_fn)(&mut y[..rows], qx_slice, sx, w, cols, rows);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn matmat_q4_0(
        &self,
        pool: &ComputePool,
        y: &mut [f32],
        w: &[Q4_0Block],
        x: &[f32],
        rows: usize,
        cols: usize,
        b_dim: usize,
    ) {
        if b_dim == 0 {
            return;
        }
        let b_dim = b_dim.min(64);
        thread_local! {
            static QXT_Q4: RefCell<Vec<i8>> = const { RefCell::new(Vec::new()) };
        }
        QXT_Q4.with(|cell| {
            let mut qxt = cell.borrow_mut();
            let need = b_dim * cols;
            if qxt.len() < need {
                qxt.resize(need, 0);
            }
            let mut sx = [0.0f32; 64];
            self.quantize_act_int8_cols(&mut qxt[..need], &mut sx[..b_dim], x, cols, b_dim);
            let nt = pool.n_threads();
            if nt > 1 && rows >= 256 {
                let f = self.q4_0_matmat_slice_fn;
                let y_ptr = y.as_mut_ptr() as usize;
                let y_len = rows * b_dim;
                let sx_ref = &sx[..b_dim];
                let qxt_ref = &qxt[..need];
                pool.parallel(nt, |tid, n| {
                    let r0 = tid * rows / n;
                    let r1 = (tid + 1) * rows / n;
                    let y_slice =
                        unsafe { std::slice::from_raw_parts_mut(y_ptr as *mut f32, y_len) };
                    f(y_slice, w, qxt_ref, sx_ref, r0, r1, cols, b_dim);
                });
                return;
            }
            (self.q4_0_matmat_slice_fn)(y, w, &qxt[..need], &sx[..b_dim], 0, rows, cols, b_dim);
        });
    }

    #[allow(clippy::too_many_arguments)]
    pub fn matvec_q4_0_qkv(
        &self,
        pool: &ComputePool,
        q: &mut [f32],
        k: &mut [f32],
        v: &mut [f32],
        wq: &[Q4_0Block],
        wk: &[Q4_0Block],
        wv: &[Q4_0Block],
        x: &[f32],
        in_dim: usize,
        q_dim: usize,
        kv_dim: usize,
    ) {
        let mut qx_uninit = std::mem::MaybeUninit::<[i8; QWEN_QACT_MAX]>::uninit();
        let qx_slice = unsafe {
            std::slice::from_raw_parts_mut(qx_uninit.as_mut_ptr() as *mut i8, in_dim)
        };
        let sx = self.quantize_act_int8(qx_slice, &x[..in_dim]);
        let nb = in_dim / Q4_0_BLOCK_SIZE;
        let nt = pool.n_threads();
        let total = q_dim + 2 * kv_dim;
        if nt > 1 {
            let f = self.q4_0_matvec_qkv_slice_fn;
            let qp = q.as_mut_ptr() as usize;
            let kp = k.as_mut_ptr() as usize;
            let vp = v.as_mut_ptr() as usize;
            pool.parallel(nt, |tid, n| {
                let r0 = tid * total / n;
                let r1 = (tid + 1) * total / n;
                let mut r = r0;
                while r < r1 {
                    if r < q_dim {
                        let end = r1.min(q_dim);
                        let len = end - r;
                        let out = unsafe {
                            std::slice::from_raw_parts_mut((qp as *mut f32).add(r), len)
                        };
                        f(out, qx_slice, sx, &wq[r * nb..end * nb], in_dim, len);
                        r = end;
                    } else if r < q_dim + kv_dim {
                        let local = r - q_dim;
                        let end = r1.min(q_dim + kv_dim);
                        let local_end = end - q_dim;
                        let len = local_end - local;
                        let out = unsafe {
                            std::slice::from_raw_parts_mut((kp as *mut f32).add(local), len)
                        };
                        f(out, qx_slice, sx, &wk[local * nb..local_end * nb], in_dim, len);
                        r = end;
                    } else {
                        let local = r - q_dim - kv_dim;
                        let local_end = r1 - q_dim - kv_dim;
                        let len = local_end - local;
                        let out = unsafe {
                            std::slice::from_raw_parts_mut((vp as *mut f32).add(local), len)
                        };
                        f(out, qx_slice, sx, &wv[local * nb..local_end * nb], in_dim, len);
                        r = r1;
                    }
                }
            });
            return;
        }
        (self.q4_0_matvec_qkv_slice_fn)(&mut q[..q_dim], qx_slice, sx, wq, in_dim, q_dim);
        (self.q4_0_matvec_qkv_slice_fn)(&mut k[..kv_dim], qx_slice, sx, wk, in_dim, kv_dim);
        (self.q4_0_matvec_qkv_slice_fn)(&mut v[..kv_dim], qx_slice, sx, wv, in_dim, kv_dim);
    }

    pub fn argmax_matvec_q4_0(
        &self,
        pool: &ComputePool,
        x: &[f32],
        w: &[Q4_0Block],
        in_dim: usize,
        out_dim: usize,
    ) -> usize {
        thread_local! {
            static LOGITS_Q4: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
        }
        LOGITS_Q4.with(|cell| {
            let mut y = cell.borrow_mut();
            if y.len() < out_dim {
                y.resize(out_dim, 0.0);
            }
            self.matvec_q4_0(pool, &mut y[..out_dim], w, x, out_dim, in_dim);
            let mut best = 0usize;
            let mut best_val = y[0];
            for (o, &val) in y[..out_dim].iter().enumerate().skip(1) {
                if val > best_val {
                    best_val = val;
                    best = o;
                }
            }
            best
        })
    }

    #[inline]
    pub fn matvec_q2_0(&self, y: &mut [f32], w: &[Q2_0Block], x: &[f32], rows: usize, cols: usize) {
        scalar::q2_0_matvec(y, w, x, rows, cols);
    }

    #[inline]
    pub fn argmax_matvec_bf16(
        &self,
        x: &[f32],
        w_bf16: &[u16],
        in_dim: usize,
        out_dim: usize,
    ) -> usize {
        (self.argmax_bf16_fn)(x, w_bf16, in_dim, out_dim)
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn causal_attention(
        &self,
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
    ) {
        (self.attn_f32_fn)(
            out, q, k, v, seq_q, seq_k, n_heads, n_kv_heads, head_dim, scale, q_offset, 0,
        );
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn causal_attention_windowed(
        &self,
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
        (self.attn_f32_fn)(
            out, q, k, v, seq_q, seq_k, n_heads, n_kv_heads, head_dim, scale, q_offset, window,
        );
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn causal_attention_bf16kv(
        &self,
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
    ) {
        (self.attn_bf16kv_fn)(
            out, q, k_bf16, v_bf16, seq_q, seq_k, n_heads, n_kv_heads, head_dim, scale, q_offset,
            0, n_heads,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn causal_attention_bf16kv_heads_parallel(
        &self,
        pool: &ComputePool,
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
    ) {
        const ATTN_PARALLEL_MIN_SEQ_K: usize = 48;
        let nt = pool.n_threads();
        if seq_q != 1 || seq_k < ATTN_PARALLEL_MIN_SEQ_K || nt <= 1 {
            self.causal_attention_bf16kv(
                out, q, k_bf16, v_bf16, seq_q, seq_k, n_heads, n_kv_heads, head_dim, scale,
                q_offset,
            );
            return;
        }
        let f = self.attn_bf16kv_fn;
        let out_ptr = out.as_mut_ptr() as usize;
        let out_len = out.len();
        pool.parallel(nt, |tid, n| {
            let h0 = n_heads * tid / n;
            let h1 = n_heads * (tid + 1) / n;
            let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut f32, out_len) };
            f(
                out_slice, q, k_bf16, v_bf16, seq_q, seq_k, n_heads, n_kv_heads, head_dim, scale,
                q_offset, h0, h1,
            );
        });
    }

    #[inline]
    pub fn apply_rope_neox_inplace(
        &self,
        x: &mut [f32],
        n_heads: usize,
        head_dim: usize,
        cos_cache: &[f32],
        sin_cache: &[f32],
        pos: usize,
    ) {
        (self.rope_neox_fn)(x, n_heads, head_dim, cos_cache, sin_cache, pos);
    }

    pub fn compute_rope_neox_table(
        max_pos: usize,
        head_dim: usize,
        theta: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let half = head_dim / 2;
        let mut inv_freq = vec![0.0f32; half];
        for i in 0..half {
            inv_freq[i] = 1.0f32 / theta.powf((2 * i) as f32 / head_dim as f32);
        }
        let mut cos = vec![0.0f32; max_pos * half];
        let mut sin = vec![0.0f32; max_pos * half];
        for pos in 0..max_pos {
            let p = pos as f32;
            for i in 0..half {
                let angle = p * inv_freq[i];
                cos[pos * half + i] = angle.cos();
                sin[pos * half + i] = angle.sin();
            }
        }
        (cos, sin)
    }

    #[inline]
    pub fn silu(&self, x: &mut [f32]) {
        for v in x.iter_mut() {
            *v = *v / (1.0f32 + (-*v).exp());
        }
    }

    #[inline]
    pub fn swiglu_inplace(&self, gate_up: &mut [f32], tmp: &mut [f32], n: usize) {
        for i in 0..n {
            tmp[i] = (-gate_up[2 * i]).exp();
        }
        for i in 0..n {
            let g = gate_up[2 * i];
            let u = gate_up[2 * i + 1];
            gate_up[i] = (g / (1.0f32 + tmp[i])) * u;
        }
    }

    #[inline]
    pub fn add_inplace(&self, y: &mut [f32], x: &[f32]) {
        for (a, &b) in y.iter_mut().zip(x.iter()) {
            *a += b;
        }
    }

    #[inline]
    pub fn mul_inplace(&self, y: &mut [f32], x: &[f32]) {
        for (a, &b) in y.iter_mut().zip(x.iter()) {
            *a *= b;
        }
    }

    #[inline]
    pub fn vec_scale_inplace(&self, y: &mut [f32], s: f32) {
        for a in y.iter_mut() {
            *a *= s;
        }
    }

    #[inline]
    pub fn round_bf16(&self, x: &mut [f32]) {
        for v in x.iter_mut() {
            let bf = (v.to_bits() >> 16) as u16;
            *v = f32::from_bits((bf as u32) << 16);
        }
    }

    #[inline]
    pub fn bf16_accum_f32(&self, dst: &mut [f32], src_bf16: &[u16]) {
        (self.bf16_accum_f32_fn)(dst, src_bf16);
    }

    #[inline]
    pub fn bf16_to_f32_vec(&self, dst: &mut [f32], src_bf16: &[u16]) {
        (self.bf16_to_f32_vec_fn)(dst, src_bf16);
    }

    #[inline]
    pub fn f32_to_bf16_vec(&self, dst: &mut [u16], src: &[f32]) {
        (self.f32_to_bf16_vec_fn)(dst, src);
    }

    #[inline]
    pub fn f32_matvec_accum(
        &self,
        y: &mut [f32],
        w: &[f32],
        x: &[f32],
        rows: usize,
        cols: usize,
    ) {
        (self.f32_matvec_accum_fn)(y, w, x, rows, cols);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn convert_talker_layer_bf16_to_f32(
        &self,
        pool: &ComputePool,
        wq_dst: &mut [f32],
        wq_src: &[u16],
        wk_dst: &mut [f32],
        wk_src: &[u16],
        wv_dst: &mut [f32],
        wv_src: &[u16],
        wo_dst: &mut [f32],
        wo_src: &[u16],
        gate_up_dst: &mut [f32],
        gate_src: &[u16],
        up_src: &[u16],
        down_dst: &mut [f32],
        down_src: &[u16],
        inter: usize,
        hidden: usize,
    ) {
        let nt = pool.n_threads();
        if nt <= 1 {
            self.bf16_to_f32_vec(wq_dst, wq_src);
            self.bf16_to_f32_vec(wk_dst, wk_src);
            self.bf16_to_f32_vec(wv_dst, wv_src);
            self.bf16_to_f32_vec(wo_dst, wo_src);
            self.bf16_to_f32_vec(down_dst, down_src);
            crate::quant::gate_up_to_f32(gate_up_dst, gate_src, up_src, inter, hidden);
            return;
        }
        let nq = wq_dst.len().min(wq_src.len());
        let nk = wk_dst.len().min(wk_src.len());
        let nv = wv_dst.len().min(wv_src.len());
        let no = wo_dst.len().min(wo_src.len());
        let nd = down_dst.len().min(down_src.len());
        let wq_p = wq_dst.as_mut_ptr() as usize;
        let wk_p = wk_dst.as_mut_ptr() as usize;
        let wv_p = wv_dst.as_mut_ptr() as usize;
        let wo_p = wo_dst.as_mut_ptr() as usize;
        let gu_p = gate_up_dst.as_mut_ptr() as usize;
        let dn_p = down_dst.as_mut_ptr() as usize;
        let conv_fn = self.bf16_to_f32_vec_fn;
        pool.parallel(nt, |tid, n| {
            let slice_conv = |dst_ptr: usize, src: &[u16], total: usize| {
                let a = tid * total / n;
                let b = (tid + 1) * total / n;
                if b > a {
                    let dst = unsafe {
                        std::slice::from_raw_parts_mut((dst_ptr as *mut f32).add(a), b - a)
                    };
                    conv_fn(dst, &src[a..b]);
                }
            };
            slice_conv(wq_p, wq_src, nq);
            slice_conv(wk_p, wk_src, nk);
            slice_conv(wv_p, wv_src, nv);
            slice_conv(wo_p, wo_src, no);
            slice_conv(dn_p, down_src, nd);
            let r0 = tid * inter / n;
            let r1 = (tid + 1) * inter / n;
            if r1 > r0 {
                let gu_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        (gu_p as *mut f32).add(2 * r0 * hidden),
                        2 * (r1 - r0) * hidden,
                    )
                };
                crate::quant::gate_up_to_f32(
                    gu_slice,
                    &gate_src[r0 * hidden..r1 * hidden],
                    &up_src[r0 * hidden..r1 * hidden],
                    r1 - r0,
                    hidden,
                );
            }
        });
    }

    pub fn snake_activation(
        &self,
        pool: &ComputePool,
        data: &mut [f32],
        channels: usize,
        length: usize,
        log_alpha: &[f32],
        log_beta: &[f32],
    ) {
        const QWEN_SNAKE_MIN_WORK: usize = 65536;
        let row_fn = self.snake_row_fn;
        if channels * length < QWEN_SNAKE_MIN_WORK || pool.n_threads() <= 1 {
            for c in 0..channels {
                row_fn(
                    &mut data[c * length..(c + 1) * length],
                    log_alpha[c],
                    log_beta[c],
                );
            }
            return;
        }
        let next = AtomicUsize::new(0);
        let data_ptr = data.as_mut_ptr() as usize;
        let nt = pool.n_threads();
        pool.parallel(nt, |_, _| loop {
            let c = next.fetch_add(1, Ordering::Relaxed);
            if c >= channels {
                break;
            }
            let row = unsafe {
                std::slice::from_raw_parts_mut((data_ptr as *mut f32).add(c * length), length)
            };
            row_fn(row, log_alpha[c], log_beta[c]);
        });
    }
}

pub struct ComputeRuntime {
    pub caps: CpuCaps,
    pub dispatch: &'static KernelDispatch,
    pub pool: Arc<ComputePool>,
    pub sd_pool: Arc<ComputePool>,
    pub blas: Arc<dyn BlasBackend>,
}

fn select_blas_backend() -> Arc<dyn BlasBackend> {
    #[cfg(feature = "faer-blas")]
    {
        if std::env::var("QWEN_BLAS")
            .ok()
            .is_some_and(|v| v.trim().eq_ignore_ascii_case("faer"))
        {
            return Arc::new(crate::blas::PureRustBlas::default());
        }
    }
    Arc::new(FfiBlas)
}

impl ComputeRuntime {
    pub fn new(threads: Option<usize>) -> Self {
        ftz_on();
        let caps = CpuCaps::detect();
        let dispatch = KernelDispatch::for_caps(&caps);
        let nt = threads.unwrap_or_else(default_num_threads).max(1);
        let sd_nt = std::env::var("QWEN_SD_THREADS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(nt);
        let pool = Arc::new(ComputePool::new(nt));
        let sd_pool = Arc::new(ComputePool::new(sd_nt));
        let blas = select_blas_backend();
        blas.set_threads(nt);
        Self {
            caps,
            dispatch,
            pool,
            sd_pool,
            blas,
        }
    }

    pub fn with_threads_and_spin(
        threads: usize,
        spin_budget: usize,
        sd_threads: Option<usize>,
    ) -> Self {
        ftz_on();
        let caps = CpuCaps::detect();
        let dispatch = KernelDispatch::for_caps(&caps);
        let nt = threads.max(1);
        let sd_nt = sd_threads.unwrap_or(nt).max(1);
        let pool = Arc::new(ComputePool::with_spin_budget(nt, spin_budget));
        let sd_pool = Arc::new(ComputePool::with_spin_budget(sd_nt, spin_budget));
        let blas = select_blas_backend();
        blas.set_threads(nt);
        Self {
            caps,
            dispatch,
            pool,
            sd_pool,
            blas,
        }
    }

    pub fn caps_report(&self) -> String {
        self.caps
            .format_report(self.pool.n_threads(), self.blas.name())
    }
}
