// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::blas::{BlasBackend, FfiBlas};
use crate::caps::CpuCaps;
use crate::ftz::ftz_on;
use crate::pool::{resolve_num_threads, ComputePool};
use crate::quant::{Q2_0Block, Q4_0Block, Q2_0_BLOCK_SIZE, Q4_0_BLOCK_SIZE, QWEN_QACT_MAX};
use crate::simd::scalar;

#[cfg(target_arch = "x86_64")]
use crate::simd::avx2;
#[cfg(target_arch = "aarch64")]
use crate::simd::neon;

/// Trozos por hilo en `parallel_rows`. Con 1 hilo copiando 256 MiB, 4 trozos bajan el productor
/// (Talker + CP) ~3 ms por fotograma frente a 1 trozo fijo por hilo; con 2 la mejora no fue consistente.
const PARALLEL_ROWS_CHUNKS_PER_THREAD: usize = 4;

type QuantizeActFn = fn(&mut [i8], &[f32]) -> f32;
type RmsNormFn = fn(&mut [f32], &[f32], &[f32], usize, usize, f32);
type RmsNormResidualFn = fn(&mut [f32], &mut [f32], &[f32], &[f32], usize, f32);
type RmsNormPerHeadFn = fn(&mut [f32], &[f32], usize, usize, usize, f32);
type Bf16MatvecSliceFn = fn(&mut [f32], &[f32], &[u16], usize, usize);
type Bf16MatmatSliceFn = fn(&mut [f32], &[u16], &[f32], usize, usize, usize, usize);
type Int8MatvecSliceFn = fn(&mut [f32], &[i8], f32, &[i8], &[f32], usize, usize);
type Int8MatmatSliceFn = fn(&mut [f32], &[i8], &[f32], &[i8], &[f32], usize, usize, usize, usize);
type Q4MatvecSliceFn = fn(&mut [f32], &[i8], f32, &[Q4_0Block], usize, usize);
type Q4MatmatSliceFn = fn(&mut [f32], &[Q4_0Block], &[i8], &[f32], usize, usize, usize, usize);
type ArgmaxBf16Fn = fn(&[f32], &[u16], usize, usize) -> usize;
type AttnF32Fn =
    fn(&mut [f32], &[f32], &[f32], &[f32], usize, usize, usize, usize, usize, f32, usize, usize);
type AttnBf16KvFn = fn(
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
type AttnBf16KvPtrFn = unsafe fn(
    *mut f32,
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
type RopeNeoxFn = fn(&mut [f32], usize, usize, &[f32], &[f32], usize);
type Bf16VecOpFn = fn(&mut [f32], &[u16]);
type F32ToBf16VecFn = fn(&mut [u16], &[f32]);
type F32MatvecAccumFn = fn(&mut [f32], &[f32], &[f32], usize, usize);
type SnakeRowFn = fn(&mut [f32], f32, f32);

/// Puntero mutable compartible entre trabajadores sobre segmentos disjuntos, preservando proveniencia.
#[derive(Copy, Clone)]
struct SendMutPtr<T>(*mut T);

unsafe impl<T> Send for SendMutPtr<T> {}
unsafe impl<T> Sync for SendMutPtr<T> {}

impl<T> SendMutPtr<T> {
    #[inline]
    fn from_slice(s: &mut [T]) -> Self {
        Self(s.as_mut_ptr())
    }

    #[inline]
    unsafe fn slice_mut<'a>(self, offset: usize, len: usize) -> &'a mut [T] {
        std::slice::from_raw_parts_mut(self.0.add(offset), len)
    }

    #[inline]
    fn as_ptr(&self) -> *mut T {
        self.0
    }
}

/// Lee `QWEN_BATCH_B` validando el rango `[1, max_b]` (usando `default_b` si falta o es inválido).
pub fn env_batch_b(default_b: usize, max_b: usize) -> usize {
    std::env::var("QWEN_BATCH_B")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&v| (1..=max_b).contains(&v))
        .unwrap_or(default_b)
}

/// Tabla de punteros a kernels SIMD resuelta en tiempo de ejecución según `CpuCaps`.
pub struct KernelDispatch {
    /// Nombre de la ruta: `scalar`, `avx2`, `neon` o `neon-dotprod`.
    name: &'static str,
    quantize_act_int8_fn: QuantizeActFn,
    rms_norm_fn: RmsNormFn,
    rms_norm_residual_fn: RmsNormResidualFn,
    rms_norm_per_head_fn: RmsNormPerHeadFn,
    bf16_matvec_slice_fn: Bf16MatvecSliceFn,
    bf16_matmat_slice_fn: Bf16MatmatSliceFn,
    int8_matvec_slice_fn: Int8MatvecSliceFn,
    int8_matmat_slice_fn: Int8MatmatSliceFn,
    q4_0_matvec_slice_fn: Q4MatvecSliceFn,
    q4_0_matvec_qkv_slice_fn: Q4MatvecSliceFn,
    q4_0_matmat_slice_fn: Q4MatmatSliceFn,
    argmax_bf16_fn: ArgmaxBf16Fn,
    attn_f32_fn: AttnF32Fn,
    attn_bf16kv_fn: AttnBf16KvFn,
    /// Atención BF16 de las cabezas `[h_begin, h_end)` sobre un puntero de salida compartido: la
    /// usa la versión repartida entre hilos.
    attn_bf16kv_ptr_fn: AttnBf16KvPtrFn,
    rope_neox_fn: RopeNeoxFn,
    bf16_accum_f32_fn: Bf16VecOpFn,
    bf16_to_f32_vec_fn: Bf16VecOpFn,
    f32_to_bf16_vec_fn: F32ToBf16VecFn,
    f32_matvec_accum_fn: F32MatvecAccumFn,
    snake_row_fn: SnakeRowFn,
}

static SCALAR_DISPATCH: KernelDispatch = KernelDispatch {
    name: "scalar",
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
    attn_bf16kv_ptr_fn: scalar::causal_attention_bf16kv_heads_ptr,
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
    #[allow(clippy::too_many_arguments)]
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
    #[allow(clippy::too_many_arguments)]
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
    #[allow(clippy::too_many_arguments)]
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
    #[allow(clippy::too_many_arguments)]
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
    name: "avx2",
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
    attn_bf16kv_ptr_fn: avx2::causal_attention_bf16kv_heads_ptr,
    rope_neox_fn: avx2_wrappers::rope_neox,
    bf16_accum_f32_fn: avx2_wrappers::bf16_accum_f32,
    bf16_to_f32_vec_fn: avx2_wrappers::bf16_to_f32_vec,
    f32_to_bf16_vec_fn: avx2_wrappers::f32_to_bf16_vec,
    f32_matvec_accum_fn: avx2_wrappers::f32_matvec_accum,
    snake_row_fn: avx2_wrappers::snake_row,
};

/// Tabla NEON para aarch64: Q4_0 e INT8 con la variante `$m` (`sdot` o `widen`) y el resto de
/// kernels con NEON base.
#[cfg(target_arch = "aarch64")]
macro_rules! neon_dispatch {
    ($name:literal, $m:ident) => {
        KernelDispatch {
            name: $name,
            quantize_act_int8_fn: neon::quantize_act_int8,
            rms_norm_fn: neon::rms_norm,
            rms_norm_residual_fn: neon::rms_norm_residual,
            rms_norm_per_head_fn: neon::rms_norm_per_head,
            bf16_matvec_slice_fn: neon::bf16_matvec_slice,
            bf16_matmat_slice_fn: neon::bf16_matmat_slice,
            int8_matvec_slice_fn: neon::$m::int8_matvec_slice,
            int8_matmat_slice_fn: neon::$m::int8_matmat_slice,
            q4_0_matvec_slice_fn: neon::$m::q4_0_matvec_slice,
            q4_0_matvec_qkv_slice_fn: neon::$m::q4_0_matvec_qkv_slice,
            q4_0_matmat_slice_fn: neon::$m::q4_0_matmat_slice,
            argmax_bf16_fn: neon::argmax_matvec_bf16,
            attn_f32_fn: neon::causal_attention_f32_heads,
            attn_bf16kv_fn: neon::causal_attention_bf16kv_heads,
            attn_bf16kv_ptr_fn: neon::causal_attention_bf16kv_heads_ptr,
            rope_neox_fn: neon::apply_rope_neox_inplace,
            bf16_accum_f32_fn: neon::bf16_accum_f32,
            bf16_to_f32_vec_fn: neon::bf16_to_f32_vec,
            f32_to_bf16_vec_fn: neon::f32_to_bf16_vec,
            f32_matvec_accum_fn: neon::f32_matvec_accum,
            snake_row_fn: neon::snake_row,
        }
    };
}

/// NEON con `SDOT` (CPU con `dotprod`).
#[cfg(target_arch = "aarch64")]
static NEON_DOTPROD_DISPATCH: KernelDispatch = neon_dispatch!("neon-dotprod", sdot);

/// NEON con multiplicación ampliada (cualquier CPU aarch64).
#[cfg(target_arch = "aarch64")]
static NEON_DISPATCH: KernelDispatch = neon_dispatch!("neon", widen);

/// Ruta de kernels pedida con la variable de entorno `QWEN_KERNELS`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelChoice {
    /// Los kernels más rápidos que soporte la CPU (por defecto).
    Auto,
    /// La ruta escalar aunque la CPU tenga SIMD: referencia para comparar kernels en la misma máquina.
    Scalar,
}

impl KernelChoice {
    /// Interpreta el valor de `QWEN_KERNELS`: `auto` o `scalar`, sin distinguir mayúsculas; ausente o vacío es `auto`.
    pub fn parse(value: Option<&str>) -> Result<Self, String> {
        match value.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
            None | Some("") | Some("auto") => Ok(Self::Auto),
            Some("scalar") => Ok(Self::Scalar),
            Some(other) => Err(format!("QWEN_KERNELS={other} no es válido; use auto o scalar")),
        }
    }

    /// Lee `QWEN_KERNELS` una vez por proceso. Un valor no válido se avisa por stderr y se trata como `auto`.
    pub fn from_env() -> Self {
        static CHOICE: std::sync::OnceLock<KernelChoice> = std::sync::OnceLock::new();
        *CHOICE.get_or_init(|| {
            Self::parse(std::env::var("QWEN_KERNELS").ok().as_deref()).unwrap_or_else(|e| {
                eprintln!("aviso: {e}; se usa auto");
                Self::Auto
            })
        })
    }
}

impl KernelDispatch {
    /// Tabla de kernels para `caps`, salvo que `QWEN_KERNELS=scalar` fuerce la ruta escalar.
    pub fn for_caps(caps: &CpuCaps) -> &'static Self {
        if KernelChoice::from_env() == KernelChoice::Scalar {
            return &SCALAR_DISPATCH;
        }
        Self::available(caps).last().copied().unwrap_or(&SCALAR_DISPATCH)
    }

    /// Tablas que puede ejecutar esta CPU, de la más lenta a la más rápida, sin tener en cuenta
    /// `QWEN_KERNELS`. La primera es siempre la escalar. Sirve para comparar cada ruta con la escalar
    /// en la misma máquina (`--self-test` y pruebas).
    pub fn available(caps: &CpuCaps) -> Vec<&'static Self> {
        let mut tables = vec![&SCALAR_DISPATCH];
        #[cfg(target_arch = "x86_64")]
        if caps.has_avx2_fma() {
            tables.push(&AVX2_DISPATCH);
        }
        #[cfg(target_arch = "aarch64")]
        if caps.neon {
            tables.push(&NEON_DISPATCH);
            if caps.dotprod {
                tables.push(&NEON_DOTPROD_DISPATCH);
            }
        }
        let _ = caps;
        tables
    }

    /// Nombre de la ruta de kernels: `scalar`, `avx2`, `neon` o `neon-dotprod`.
    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn detect() -> &'static Self {
        Self::for_caps(&CpuCaps::detect())
    }

    pub fn scalar() -> &'static Self {
        &SCALAR_DISPATCH
    }

    #[inline]
    pub fn quantize_act_int8(&self, qx: &mut [i8], x: &[f32]) -> f32 {
        assert!(qx.len() >= x.len(), "quantize_act_int8 qx buffer too small");
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
        assert!(cols <= QWEN_QACT_MAX, "cols={cols} > QWEN_QACT_MAX={QWEN_QACT_MAX}");
        assert!(qxt.len() >= b_dim * cols, "quantize_act_int8_cols qxt overflow");
        assert!(sx.len() >= b_dim, "quantize_act_int8_cols sx overflow");
        assert!(x.len() >= cols * b_dim, "quantize_act_int8_cols x overflow");
        let mut col_uninit = [std::mem::MaybeUninit::<f32>::uninit(); QWEN_QACT_MAX];
        for b in 0..b_dim {
            for (k, slot) in col_uninit[..cols].iter_mut().enumerate() {
                slot.write(x[k * b_dim + b]);
            }
            let col = unsafe {
                std::slice::from_raw_parts(col_uninit.as_ptr() as *const f32, cols)
            };
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
        if seq == 0 || dim == 0 {
            return;
        }
        assert!(
            out.len() >= seq * dim && x.len() >= seq * dim && weight.len() >= dim,
            "rms_norm slice bounds mismatch"
        );
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
        if dim == 0 {
            return;
        }
        assert!(
            out.len() >= dim && x.len() >= dim && residual.len() >= dim && weight.len() >= dim,
            "rms_norm_residual slice bounds mismatch"
        );
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
        if seq == 0 || n_heads == 0 || head_dim == 0 {
            return;
        }
        assert!(
            x.len() >= seq * n_heads * head_dim && weight.len() >= head_dim,
            "rms_norm_per_head slice bounds mismatch"
        );
        (self.rms_norm_per_head_fn)(x, weight, seq, n_heads, head_dim, eps);
    }

    #[allow(clippy::too_many_arguments)]
    fn parallel_qkv_split<F>(
        pool: &ComputePool,
        q: &mut [f32],
        k: &mut [f32],
        v: &mut [f32],
        q_dim: usize,
        kv_dim: usize,
        run_segment: F,
    ) where
        F: Fn(usize, &mut [f32], usize, usize) + Sync,
    {
        let nt = pool.n_threads();
        let total = q_dim + 2 * kv_dim;
        if nt > 1 && total > 0 {
            let qp = SendMutPtr::from_slice(q);
            let kp = SendMutPtr::from_slice(k);
            let vp = SendMutPtr::from_slice(v);
            pool.parallel(nt, |tid, n| {
                let r0 = tid * total / n;
                let r1 = (tid + 1) * total / n;
                let mut r = r0;
                while r < r1 {
                    if r < q_dim {
                        let end = r1.min(q_dim);
                        let len = end - r;
                        let out = unsafe { qp.slice_mut(r, len) };
                        run_segment(0, out, r, len);
                        r = end;
                    } else if r < q_dim + kv_dim {
                        let local = r - q_dim;
                        let end = r1.min(q_dim + kv_dim);
                        let len = end - r;
                        let out = unsafe { kp.slice_mut(local, len) };
                        run_segment(1, out, local, len);
                        r = end;
                    } else {
                        let local = r - q_dim - kv_dim;
                        let len = r1 - r;
                        let out = unsafe { vp.slice_mut(local, len) };
                        run_segment(2, out, local, len);
                        r = r1;
                    }
                }
            });
            return;
        }
        run_segment(0, &mut q[..q_dim], 0, q_dim);
        run_segment(1, &mut k[..kv_dim], 0, kv_dim);
        run_segment(2, &mut v[..kv_dim], 0, kv_dim);
    }

    /// Reparte las `rows` filas de `y` entre los hilos del pool en `PARALLEL_ROWS_CHUNKS_PER_THREAD`
    /// trozos por hilo que cada hilo toma de un contador compartido según queda libre.
    ///
    /// Con la CPU ocupada, un hilo frenado por la carga ya no retrasa la barrera: los demás se
    /// quedan con sus trozos pendientes. Los bordes de trozo son múltiplos de 64 filas y cada fila
    /// se calcula con el mismo kernel y el mismo orden de suma, así que el resultado no depende del
    /// reparto.
    fn parallel_rows<F>(pool: &ComputePool, y: &mut [f32], rows: usize, row_stride: usize, f: F)
    where
        F: Fn(&mut [f32], usize, usize) + Sync,
    {
        let nt = pool.n_threads();
        if nt > 1 && rows >= 256 {
            let chunks = nt * PARALLEL_ROWS_CHUNKS_PER_THREAD;
            let bound = |c: usize| if c >= chunks { rows } else { (c * rows / chunks) & !63 };
            let next = AtomicUsize::new(0);
            let y_ptr = SendMutPtr::from_slice(y);
            pool.parallel(nt, |_, _| loop {
                let c = next.fetch_add(1, Ordering::Relaxed);
                if c >= chunks {
                    break;
                }
                let (r0, r1) = (bound(c), bound(c + 1));
                if r1 > r0 {
                    let y_slice = unsafe { y_ptr.slice_mut(r0 * row_stride, (r1 - r0) * row_stride) };
                    f(y_slice, r0, r1);
                }
            });
            return;
        }
        f(&mut y[..rows * row_stride], 0, rows);
    }

    fn quant_matmat_with_tls<F>(&self, x: &[f32], cols: usize, b_dim: usize, f: F)
    where
        F: FnOnce(&[i8], &[f32]),
    {
        thread_local! {
            #[allow(clippy::missing_const_for_thread_local)]
            static QXT_TLS: RefCell<Vec<i8>> = const { RefCell::new(Vec::new()) };
        }
        QXT_TLS.with(|cell| {
            let mut qxt = cell.borrow_mut();
            let need = b_dim * cols;
            if qxt.len() < need {
                qxt.resize(need, 0);
            }
            let mut sx = [0.0f32; 64];
            self.quantize_act_int8_cols(&mut qxt[..need], &mut sx[..b_dim], x, cols, b_dim);
            f(&qxt[..need], &sx[..b_dim]);
        });
    }

    fn argmax_with_tls<F>(out_dim: usize, fill: F) -> usize
    where
        F: FnOnce(&mut [f32]),
    {
        if out_dim == 0 {
            return 0;
        }
        thread_local! {
            #[allow(clippy::missing_const_for_thread_local)]
            static LOGITS_TLS: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
        }
        LOGITS_TLS.with(|cell| {
            let mut y = cell.borrow_mut();
            if y.len() < out_dim {
                y.resize(out_dim, 0.0);
            }
            fill(&mut y[..out_dim]);
            Self::argmax_slice(&y[..out_dim])
        })
    }

    #[inline]
    fn argmax_slice(vals: &[f32]) -> usize {
        let mut best = 0usize;
        let mut best_val = vals[0];
        for (o, &val) in vals.iter().enumerate().skip(1) {
            if val > best_val {
                best_val = val;
                best = o;
            }
        }
        best
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
        if rows == 0 {
            return;
        }
        assert!(y.len() >= rows && x.len() >= cols && w.len() >= rows * cols);
        let f = self.bf16_matvec_slice_fn;
        let x_ref = &x[..cols];
        Self::parallel_rows(pool, y, rows, 1, |y_slice, r0, r1| {
            f(y_slice, x_ref, &w[r0 * cols..r1 * cols], cols, r1 - r0);
        });
    }

    #[allow(clippy::too_many_arguments)]
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
        if b_dim == 0 || rows == 0 {
            return;
        }
        assert!(b_dim <= 64, "matmat_bf16 b_dim={b_dim} > 64");
        assert!(y.len() >= rows * b_dim && x.len() >= cols * b_dim && w.len() >= rows * cols);
        let f = self.bf16_matmat_slice_fn;
        Self::parallel_rows(pool, y, rows, b_dim, |y_slice, r0, r1| {
            f(y_slice, &w[r0 * cols..r1 * cols], x, 0, r1 - r0, cols, b_dim);
        });
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
        assert!(
            q.len() >= q_dim
                && k.len() >= kv_dim
                && v.len() >= kv_dim
                && x.len() >= in_dim
                && wq.len() >= q_dim * in_dim
                && wk.len() >= kv_dim * in_dim
                && wv.len() >= kv_dim * in_dim
        );
        let f = self.bf16_matvec_slice_fn;
        let x_ref = &x[..in_dim];
        let total_dim = q_dim + 2 * kv_dim;
        if pool.n_threads() > 1 && total_dim >= 256 {
            Self::parallel_qkv_split(pool, q, k, v, q_dim, kv_dim, |seg, out, local, len| {
                let w = match seg {
                    0 => wq,
                    1 => wk,
                    _ => wv,
                };
                f(out, x_ref, &w[local * in_dim..(local + len) * in_dim], in_dim, len);
            });
            return;
        }
        f(&mut q[..q_dim], x_ref, wq, in_dim, q_dim);
        f(&mut k[..kv_dim], x_ref, wk, in_dim, kv_dim);
        f(&mut v[..kv_dim], x_ref, wv, in_dim, kv_dim);
    }

    #[allow(clippy::too_many_arguments)]
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
        if rows == 0 {
            return;
        }
        ftz_on();
        assert!(
            cols > 0 && cols <= QWEN_QACT_MAX && cols.is_multiple_of(Q4_0_BLOCK_SIZE),
            "matvec_int8 invalid cols={cols}"
        );
        assert!(
            y.len() >= rows && x.len() >= cols && w.len() >= rows * cols && scale.len() >= rows
        );
        let mut qx_buf = [0i8; QWEN_QACT_MAX];
        let qx_slice = &mut qx_buf[..cols];
        let sx = self.quantize_act_int8(qx_slice, &x[..cols]);
        let f = self.int8_matvec_slice_fn;
        let qx_ref: &[i8] = qx_slice;
        Self::parallel_rows(pool, y, rows, 1, |y_slice, r0, r1| {
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
        if b_dim == 0 || rows == 0 {
            return;
        }
        ftz_on();
        assert!(
            cols > 0 && cols <= QWEN_QACT_MAX && cols.is_multiple_of(Q4_0_BLOCK_SIZE),
            "matmat_int8 invalid cols={cols}"
        );
        assert!(b_dim <= 64, "matmat_int8 b_dim={b_dim} > 64");
        assert!(
            y.len() >= rows * b_dim
                && x.len() >= cols * b_dim
                && w.len() >= rows * cols
                && scale.len() >= rows
        );
        let f = self.int8_matmat_slice_fn;
        self.quant_matmat_with_tls(x, cols, b_dim, |qxt_ref, sx_ref| {
            Self::parallel_rows(pool, y, rows, b_dim, |y_slice, r0, r1| {
                f(
                    y_slice,
                    &w[r0 * cols..r1 * cols],
                    &scale[r0..r1],
                    qxt_ref,
                    sx_ref,
                    0,
                    r1 - r0,
                    cols,
                    b_dim,
                );
            });
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
        ftz_on();
        assert!(
            in_dim > 0 && in_dim <= QWEN_QACT_MAX && in_dim.is_multiple_of(Q4_0_BLOCK_SIZE),
            "matvec_int8_qkv invalid in_dim={in_dim}"
        );
        assert!(
            q.len() >= q_dim
                && k.len() >= kv_dim
                && v.len() >= kv_dim
                && x.len() >= in_dim
                && wq.len() >= q_dim * in_dim
                && sq.len() >= q_dim
                && wk.len() >= kv_dim * in_dim
                && sk.len() >= kv_dim
                && wv.len() >= kv_dim * in_dim
                && sv.len() >= kv_dim
        );
        let mut qx_buf = [0i8; QWEN_QACT_MAX];
        let qx_slice = &mut qx_buf[..in_dim];
        let sx = self.quantize_act_int8(qx_slice, &x[..in_dim]);
        let f = self.int8_matvec_slice_fn;
        let qx_ref: &[i8] = qx_slice;
        Self::parallel_qkv_split(pool, q, k, v, q_dim, kv_dim, |seg, out, local, len| {
            let (w, s) = match seg {
                0 => (wq, sq),
                1 => (wk, sk),
                _ => (wv, sv),
            };
            f(
                out,
                qx_ref,
                sx,
                &w[local * in_dim..(local + len) * in_dim],
                &s[local..local + len],
                in_dim,
                len,
            );
        });
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
        Self::argmax_with_tls(out_dim, |y| {
            self.matvec_int8(pool, y, w, scale, x, out_dim, in_dim);
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
        if rows == 0 {
            return;
        }
        ftz_on();
        assert!(
            cols > 0 && cols <= QWEN_QACT_MAX && cols.is_multiple_of(Q4_0_BLOCK_SIZE),
            "matvec_q4_0 invalid cols={cols}"
        );
        let nb = cols / Q4_0_BLOCK_SIZE;
        assert!(y.len() >= rows && x.len() >= cols && w.len() >= rows * nb);
        let mut qx_buf = [0i8; QWEN_QACT_MAX];
        let qx_slice = &mut qx_buf[..cols];
        let sx = self.quantize_act_int8(qx_slice, &x[..cols]);
        let f = self.q4_0_matvec_slice_fn;
        let qx_ref: &[i8] = qx_slice;
        Self::parallel_rows(pool, y, rows, 1, |y_slice, r0, r1| {
            f(y_slice, qx_ref, sx, &w[r0 * nb..r1 * nb], cols, r1 - r0);
        });
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
        if b_dim == 0 || rows == 0 {
            return;
        }
        ftz_on();
        assert!(
            cols > 0 && cols <= QWEN_QACT_MAX && cols.is_multiple_of(Q4_0_BLOCK_SIZE),
            "matmat_q4_0 invalid cols={cols}"
        );
        assert!(b_dim <= 64, "matmat_q4_0 b_dim={b_dim} > 64");
        let nb = cols / Q4_0_BLOCK_SIZE;
        assert!(y.len() >= rows * b_dim && x.len() >= cols * b_dim && w.len() >= rows * nb);
        let f = self.q4_0_matmat_slice_fn;
        self.quant_matmat_with_tls(x, cols, b_dim, |qxt_ref, sx_ref| {
            Self::parallel_rows(pool, y, rows, b_dim, |y_slice, r0, r1| {
                f(
                    y_slice,
                    &w[r0 * nb..r1 * nb],
                    qxt_ref,
                    sx_ref,
                    0,
                    r1 - r0,
                    cols,
                    b_dim,
                );
            });
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
        ftz_on();
        assert!(
            in_dim > 0 && in_dim <= QWEN_QACT_MAX && in_dim.is_multiple_of(Q4_0_BLOCK_SIZE),
            "matvec_q4_0_qkv invalid in_dim={in_dim}"
        );
        let nb = in_dim / Q4_0_BLOCK_SIZE;
        assert!(
            q.len() >= q_dim
                && k.len() >= kv_dim
                && v.len() >= kv_dim
                && x.len() >= in_dim
                && wq.len() >= q_dim * nb
                && wk.len() >= kv_dim * nb
                && wv.len() >= kv_dim * nb
        );
        let mut qx_buf = [0i8; QWEN_QACT_MAX];
        let qx_slice = &mut qx_buf[..in_dim];
        let sx = self.quantize_act_int8(qx_slice, &x[..in_dim]);
        let f = self.q4_0_matvec_qkv_slice_fn;
        let qx_ref: &[i8] = qx_slice;
        Self::parallel_qkv_split(pool, q, k, v, q_dim, kv_dim, |seg, out, local, len| {
            let w = match seg {
                0 => wq,
                1 => wk,
                _ => wv,
            };
            f(out, qx_ref, sx, &w[local * nb..(local + len) * nb], in_dim, len);
        });
    }

    pub fn argmax_matvec_q4_0(
        &self,
        pool: &ComputePool,
        x: &[f32],
        w: &[Q4_0Block],
        in_dim: usize,
        out_dim: usize,
    ) -> usize {
        Self::argmax_with_tls(out_dim, |y| {
            self.matvec_q4_0(pool, y, w, x, out_dim, in_dim);
        })
    }

    #[inline]
    pub fn matvec_q2_0(&self, y: &mut [f32], w: &[Q2_0Block], x: &[f32], rows: usize, cols: usize) {
        if rows == 0 {
            return;
        }
        assert!(
            cols > 0 && cols.is_multiple_of(Q2_0_BLOCK_SIZE),
            "matvec_q2_0 invalid cols={cols}"
        );
        let bpr = cols / Q2_0_BLOCK_SIZE;
        assert!(
            y.len() >= rows && w.len() >= rows * bpr && x.len() >= cols,
            "matvec_q2_0 slice bounds mismatch"
        );
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
        if out_dim == 0 {
            return 0;
        }
        assert!(
            x.len() >= in_dim && w_bf16.len() >= out_dim * in_dim,
            "argmax_matvec_bf16 slice bounds mismatch"
        );
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
        self.causal_attention_windowed(
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
        if seq_q == 0 || n_heads == 0 || head_dim == 0 {
            return;
        }
        assert!(
            n_kv_heads > 0 && n_heads.is_multiple_of(n_kv_heads),
            "causal_attention invalid head counts {n_heads}/{n_kv_heads}"
        );
        assert!(
            out.len() >= seq_q * n_heads * head_dim
                && q.len() >= seq_q * n_heads * head_dim
                && k.len() >= seq_k * n_kv_heads * head_dim
                && v.len() >= seq_k * n_kv_heads * head_dim,
            "causal_attention slice bounds mismatch"
        );
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
        if seq_q == 0 || n_heads == 0 || head_dim == 0 {
            return;
        }
        assert!(
            n_kv_heads > 0 && n_heads.is_multiple_of(n_kv_heads),
            "causal_attention_bf16kv invalid head counts {n_heads}/{n_kv_heads}"
        );
        assert!(
            out.len() >= seq_q * n_heads * head_dim
                && q.len() >= seq_q * n_heads * head_dim
                && k_bf16.len() >= seq_k * n_kv_heads * head_dim
                && v_bf16.len() >= seq_k * n_kv_heads * head_dim,
            "causal_attention_bf16kv slice bounds mismatch"
        );
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
        if seq_q == 0 || n_heads == 0 || head_dim == 0 {
            return;
        }
        assert!(
            n_kv_heads > 0 && n_heads.is_multiple_of(n_kv_heads),
            "causal_attention_bf16kv_heads_parallel invalid head counts {n_heads}/{n_kv_heads}"
        );
        assert!(
            out.len() >= seq_q * n_heads * head_dim
                && q.len() >= seq_q * n_heads * head_dim
                && k_bf16.len() >= seq_k * n_kv_heads * head_dim
                && v_bf16.len() >= seq_k * n_kv_heads * head_dim,
            "causal_attention_bf16kv_heads_parallel slice bounds mismatch"
        );
        let nt = pool.n_threads();
        if seq_q != 1 || seq_k < ATTN_PARALLEL_MIN_SEQ_K || nt <= 1 {
            self.causal_attention_bf16kv(
                out, q, k_bf16, v_bf16, seq_q, seq_k, n_heads, n_kv_heads, head_dim, scale,
                q_offset,
            );
            return;
        }
        let out_ptr = SendMutPtr::from_slice(out);
        let heads_fn = self.attn_bf16kv_ptr_fn;
        pool.parallel(nt, |tid, n| {
            let h0 = n_heads * tid / n;
            let h1 = n_heads * (tid + 1) / n;
            if h0 >= h1 {
                return;
            }
            // SAFETY: cada hilo escribe solo las cabezas `[h0, h1)` de `out`, disjuntas entre hilos.
            unsafe {
                heads_fn(
                    out_ptr.as_ptr(),
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
                    h0,
                    h1,
                );
            }
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
        if n_heads == 0 || head_dim == 0 {
            return;
        }
        assert!(head_dim.is_multiple_of(2), "RoPE head_dim={head_dim} must be even");
        let half = head_dim / 2;
        assert!(
            x.len() >= n_heads * head_dim
                && cos_cache.len() >= (pos + 1) * half
                && sin_cache.len() >= (pos + 1) * half,
            "apply_rope_neox_inplace slice bounds mismatch"
        );
        (self.rope_neox_fn)(x, n_heads, head_dim, cos_cache, sin_cache, pos);
    }

    pub fn compute_rope_neox_table(
        max_pos: usize,
        head_dim: usize,
        theta: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let half = head_dim / 2;
        let mut inv_freq = vec![0.0f32; half];
        for (i, slot) in inv_freq.iter_mut().enumerate() {
            *slot = 1.0f32 / theta.powf((2 * i) as f32 / head_dim as f32);
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
        assert!(
            gate_up.len() >= 2 * n && tmp.len() >= n,
            "swiglu_inplace slice bounds mismatch"
        );
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
        if rows == 0 {
            return;
        }
        assert!(
            y.len() >= rows && w.len() >= rows * cols && x.len() >= cols,
            "f32_matvec_accum slice bounds mismatch"
        );
        (self.f32_matvec_accum_fn)(y, w, x, rows, cols);
    }

    #[allow(clippy::too_many_arguments)]
    /// Convierte `src` (BF16) a `dst` (F32) repartiendo el trabajo entre los hilos de `pool`.
    pub fn bf16_to_f32_par(&self, pool: &ComputePool, dst: &mut [f32], src: &[u16]) {
        let total = dst.len();
        assert!(src.len() >= total, "bf16_to_f32_par slice bounds mismatch");
        let nt = pool.n_threads();
        if nt <= 1 {
            self.bf16_to_f32_vec(dst, &src[..total]);
            return;
        }
        let dst_p = SendMutPtr::from_slice(dst);
        let conv_fn = self.bf16_to_f32_vec_fn;
        pool.parallel(nt, |tid, n| {
            let a = tid * total / n;
            let b = (tid + 1) * total / n;
            if b > a {
                let d = unsafe { dst_p.slice_mut(a, b - a) };
                conv_fn(d, &src[a..b]);
            }
        });
    }

    /// Convierte las filas `gate` y `up` (BF16, `rows × cols`) a F32 intercaladas
    /// (`gate_0`, `up_0`, `gate_1`, …) repartiendo las filas entre los hilos de `pool`.
    pub fn gate_up_to_f32_par(
        &self,
        pool: &ComputePool,
        dst: &mut [f32],
        gate: &[u16],
        up: &[u16],
        rows: usize,
        cols: usize,
    ) {
        assert!(
            dst.len() >= 2 * rows * cols && gate.len() >= rows * cols && up.len() >= rows * cols,
            "gate_up_to_f32_par slice bounds mismatch"
        );
        let nt = pool.n_threads();
        if nt <= 1 {
            crate::quant::gate_up_to_f32(dst, gate, up, rows, cols);
            return;
        }
        let dst_p = SendMutPtr::from_slice(dst);
        pool.parallel(nt, |tid, n| {
            let r0 = tid * rows / n;
            let r1 = (tid + 1) * rows / n;
            if r1 > r0 {
                let d = unsafe { dst_p.slice_mut(2 * r0 * cols, 2 * (r1 - r0) * cols) };
                crate::quant::gate_up_to_f32(
                    d,
                    &gate[r0 * cols..r1 * cols],
                    &up[r0 * cols..r1 * cols],
                    r1 - r0,
                    cols,
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
        assert!(
            data.len() >= channels * length
                && log_alpha.len() >= channels
                && log_beta.len() >= channels
        );
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
        let data_ptr = SendMutPtr::from_slice(data);
        let nt = pool.n_threads();
        pool.parallel(nt, |_, _| loop {
            let c = next.fetch_add(1, Ordering::Relaxed);
            if c >= channels {
                break;
            }
            let row = unsafe { data_ptr.slice_mut(c * length, length) };
            row_fn(row, log_alpha[c], log_beta[c]);
        });
    }
}

/// Contexto de cómputo inyectable con despacho SIMD, pools de hilos y backend BLAS.
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
        let nt = resolve_num_threads(threads);
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
            .format_report(
            self.pool.n_threads(),
            self.blas.name(),
            KernelChoice::from_env(),
            self.dispatch.name(),
        )
    }
}
