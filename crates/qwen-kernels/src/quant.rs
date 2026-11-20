// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use crate::aligned::AlignedBuf;

pub const QWEN_QACT_MAX: usize = 8192;
pub const Q4_0_BLOCK_SIZE: usize = 32;
pub const Q2_0_BLOCK_SIZE: usize = 32;
pub const QWEN_INT8_PF_DIST: usize = 5120;
pub const QWEN_LONG_ROW_COLS: usize = 2048;

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct Q4_0Block {
    pub scale_f16: u16,
    pub qs: [u8; 16],
}

const _: () = assert!(std::mem::size_of::<Q4_0Block>() == 18);

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct Q2_0Block {
    pub scale: f32,
    pub qs: [u8; 8],
}

const _: () = assert!(std::mem::size_of::<Q2_0Block>() == 12);

#[inline]
pub fn bf16_to_f32(bf: u16) -> f32 {
    f32::from_bits((bf as u32) << 16)
}

#[inline]
pub fn f32_to_bf16_trunc(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

#[inline]
pub fn f32_to_bf16_round(v: f32) -> u16 {
    let bits = v.to_bits();
    ((bits.wrapping_add(0x8000u32)) >> 16) as u16
}

#[inline(always)]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let em = (h & 0x7FFF) as u32;
    let bits = if em >= 0x7C00 {
        sign | 0x7F80_0000u32 | ((em & 0x03FF) << 13)
    } else if em >= 0x0400 {
        sign | ((em + ((127u32 - 15u32) << 10)) << 13)
    } else if em == 0 {
        sign
    } else {
        let mut shift = 0u32;
        let mut m = em;
        while (m & 0x0400) == 0 {
            m <<= 1;
            shift += 1;
        }
        sign | ((127u32 - 15u32 - shift) << 23) | ((m & 0x03FF) << 13)
    };
    f32::from_bits(bits)
}

#[inline]
pub fn f32_to_f16(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let e = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let m = bits & 0x007F_FFFF;
    if e >= 0x1F {
        return (sign | 0x7C00) as u16;
    }
    if e <= 0 {
        if e < -10 {
            return sign as u16;
        }
        let m2 = m | 0x0080_0000;
        let shift = (14 - e) as u32;
        let mut sub = (m2 >> shift) as u16;
        if ((m2 >> (shift - 1)) & 1) != 0 {
            sub = sub.wrapping_add(1);
        }
        return (sign as u16) | sub;
    }
    let mut out = (sign | ((e as u32) << 10) | (m >> 13)) as u16;
    if (m & 0x1000) != 0 {
        out = out.wrapping_add(1);
    }
    out
}

pub fn quant_shape_check(name: &str, cols: usize) -> Result<(), String> {
    if cols == 0 || cols > QWEN_QACT_MAX || !cols.is_multiple_of(Q4_0_BLOCK_SIZE) {
        return Err(format!(
            "error: la matriz cuantizada {name} tiene {cols} columnas; el motor exige un múltiplo de 32 y como máximo {QWEN_QACT_MAX}"
        ));
    }
    Ok(())
}

/// Cuantizador escalar de activaciones (referencia y backend escalar).
/// Usa redondeo IEEE round-half-even (`round_ties_even`), equivalente a `cvtps_epi32` en AVX2.
#[inline]
pub fn quantize_act_int8_scalar(qx: &mut [i8], x: &[f32]) -> f32 {
    let n = x.len();
    debug_assert!(qx.len() >= n);
    let mut amax = 0.0f32;
    for &xi in x {
        let a = xi.abs();
        if a > amax {
            amax = a;
        }
    }
    if amax == 0.0 {
        qx[..n].fill(0);
        return 0.0;
    }
    let inv = 127.0f32 / amax;
    for i in 0..n {
        let v = (x[i] * inv).round_ties_even() as i32;
        qx[i] = v.clamp(-128, 127) as i8;
    }
    amax / 127.0f32
}

/// Cuantiza `src_bf16[rows, cols]` a INT8 simétrico por fila (absmax).
pub fn quantize_bf16_to_int8(
    src_bf16: &[u16],
    rows: usize,
    cols: usize,
) -> (AlignedBuf<i8>, AlignedBuf<f32>) {
    let mut dst_int8 = AlignedBuf::<i8>::new_zeroed(rows * cols);
    let mut dst_scale = AlignedBuf::<f32>::new_zeroed(rows);
    quantize_bf16_to_int8_into(src_bf16, rows, cols, &mut dst_int8, &mut dst_scale);
    (dst_int8, dst_scale)
}

pub fn quantize_bf16_to_int8_into(
    src_bf16: &[u16],
    rows: usize,
    cols: usize,
    dst_int8: &mut [i8],
    dst_scale: &mut [f32],
) {
    assert!(
        src_bf16.len() >= rows * cols && dst_int8.len() >= rows * cols && dst_scale.len() >= rows,
        "quantize_bf16_to_int8_into: los límites de los slices no coinciden"
    );
    for r in 0..rows {
        let row = &src_bf16[r * cols..(r + 1) * cols];
        let mut amax = 0.0f32;
        for &b in row {
            let a = bf16_to_f32(b).abs();
            if a > amax {
                amax = a;
            }
        }
        let s = amax / 127.0f32;
        dst_scale[r] = s;
        let inv_s = if s > 0.0 { 127.0f32 / amax } else { 0.0 };
        let dst_row = &mut dst_int8[r * cols..(r + 1) * cols];
        for (k, &b) in row.iter().enumerate() {
            let val = bf16_to_f32(b);
            let v = (val * inv_s).round_ties_even() as i32;
            dst_row[k] = v.clamp(-128, 127) as i8;
        }
    }
}

/// Cuantización Q4_0 por mínimos cuadrados ponderados (`q4_0s1`) con nibbles intercalados (`2*i` bajo, `2*i+1` alto).
pub fn quantize_bf16_to_q4_0(src_bf16: &[u16], rows: usize, cols: usize) -> AlignedBuf<Q4_0Block> {
    assert_eq!(cols % Q4_0_BLOCK_SIZE, 0);
    let bpr = cols / Q4_0_BLOCK_SIZE;
    let mut dst = AlignedBuf::<Q4_0Block>::new_zeroed(rows * bpr);
    quantize_bf16_to_q4_0_into(src_bf16, rows, cols, &mut dst);
    dst
}

pub fn quantize_bf16_to_q4_0_into(
    src_bf16: &[u16],
    rows: usize,
    cols: usize,
    dst: &mut [Q4_0Block],
) {
    assert!(
        cols.is_multiple_of(Q4_0_BLOCK_SIZE),
        "quantize_bf16_to_q4_0_into: cols={cols} no es múltiplo de {Q4_0_BLOCK_SIZE}"
    );
    let bpr = cols / Q4_0_BLOCK_SIZE;
    assert!(
        src_bf16.len() >= rows * cols && dst.len() >= rows * bpr,
        "quantize_bf16_to_q4_0_into: los límites de los slices no coinciden"
    );
    for r in 0..rows {
        let row = &src_bf16[r * cols..(r + 1) * cols];
        let dst_row = &mut dst[r * bpr..(r + 1) * bpr];
        for (b, block_out) in dst_row.iter_mut().enumerate() {
            let blk = &row[b * Q4_0_BLOCK_SIZE..(b + 1) * Q4_0_BLOCK_SIZE];
            let mut vals = [0.0f32; Q4_0_BLOCK_SIZE];
            let mut amax = 0.0f32;
            let mut vmax = 0.0f32;
            for i in 0..Q4_0_BLOCK_SIZE {
                let v = bf16_to_f32(blk[i]);
                vals[i] = v;
                let a = v.abs();
                if a > amax {
                    amax = a;
                    vmax = v;
                }
            }
            let mut q = [0i32; Q4_0_BLOCK_SIZE];
            let isc = if vmax != 0.0 { -8.0f32 / vmax } else { 0.0 };
            let mut num = 0.0f64;
            let mut den = 0.0f64;
            for i in 0..Q4_0_BLOCK_SIZE {
                let v = (vals[i] * isc).round() as i32;
                let vc = v.clamp(-8, 7);
                q[i] = vc;
                let w = (vals[i] as f64) * (vals[i] as f64);
                num += w * (vals[i] as f64) * (vc as f64);
                den += w * (vc as f64) * (vc as f64);
            }
            let s = if den > 0.0 { (num / den) as f32 } else { 0.0 };
            block_out.scale_f16 = f32_to_f16(s);
            for i in 0..16 {
                block_out.qs[i] = ((q[2 * i] + 8) | ((q[2 * i + 1] + 8) << 4)) as u8;
            }
        }
    }
}

/// Cuantización Q2_0 (12 bytes / 32 pesos) usada con `--roughness > 0` en `down_proj` del Code Predictor.
pub fn quantize_bf16_to_q2_0(src_bf16: &[u16], rows: usize, cols: usize) -> AlignedBuf<Q2_0Block> {
    assert_eq!(cols % Q2_0_BLOCK_SIZE, 0);
    let bpr = cols / Q2_0_BLOCK_SIZE;
    let mut dst = AlignedBuf::<Q2_0Block>::new_zeroed(rows * bpr);
    for r in 0..rows {
        let row = &src_bf16[r * cols..(r + 1) * cols];
        let drow = &mut dst[r * bpr..(r + 1) * bpr];
        for (b, block_out) in drow.iter_mut().enumerate() {
            let blk = &row[b * Q2_0_BLOCK_SIZE..(b + 1) * Q2_0_BLOCK_SIZE];
            let mut vals = [0.0f32; Q2_0_BLOCK_SIZE];
            let mut amax = 0.0f32;
            for i in 0..Q2_0_BLOCK_SIZE {
                vals[i] = bf16_to_f32(blk[i]);
                let a = vals[i].abs();
                if a > amax {
                    amax = a;
                }
            }
            let scale = amax / 1.5f32;
            block_out.scale = scale;
            let inv = if scale > 0.0 { 1.0f32 / scale } else { 0.0 };
            block_out.qs = [0u8; 8];
            for (i, &val) in vals.iter().enumerate().take(Q2_0_BLOCK_SIZE) {
                let code = ((val * inv + 1.5f32).round_ties_even() as i32).clamp(0, 3) as u8;
                block_out.qs[i >> 2] |= code << ((i & 3) * 2);
            }
        }
    }
    dst
}

pub fn interleave_gate_up_bf16(
    dst: &mut [u16],
    gate: &[u16],
    up: &[u16],
    rows: usize,
    cols: usize,
) {
    assert!(
        dst.len() >= 2 * rows * cols && gate.len() >= rows * cols && up.len() >= rows * cols,
        "interleave_gate_up_bf16: los límites de los slices no coinciden"
    );
    for r in 0..rows {
        dst[2 * r * cols..(2 * r + 1) * cols].copy_from_slice(&gate[r * cols..(r + 1) * cols]);
        dst[(2 * r + 1) * cols..(2 * r + 2) * cols].copy_from_slice(&up[r * cols..(r + 1) * cols]);
    }
}

pub fn quantize_gate_up_int8(
    gate: &[u16],
    up: &[u16],
    rows: usize,
    cols: usize,
) -> (AlignedBuf<i8>, AlignedBuf<f32>) {
    assert!(
        gate.len() >= rows * cols && up.len() >= rows * cols,
        "quantize_gate_up_int8: los límites de los slices no coinciden"
    );
    let mut dst_int8 = AlignedBuf::<i8>::new_zeroed(2 * rows * cols);
    let mut dst_scale = AlignedBuf::<f32>::new_zeroed(2 * rows);
    for r in 0..rows {
        quantize_bf16_to_int8_into(
            &gate[r * cols..(r + 1) * cols],
            1,
            cols,
            &mut dst_int8[2 * r * cols..(2 * r + 1) * cols],
            &mut dst_scale[2 * r..2 * r + 1],
        );
        quantize_bf16_to_int8_into(
            &up[r * cols..(r + 1) * cols],
            1,
            cols,
            &mut dst_int8[(2 * r + 1) * cols..(2 * r + 2) * cols],
            &mut dst_scale[2 * r + 1..2 * r + 2],
        );
    }
    (dst_int8, dst_scale)
}

pub fn quantize_gate_up_q4_0(
    gate: &[u16],
    up: &[u16],
    rows: usize,
    cols: usize,
) -> AlignedBuf<Q4_0Block> {
    assert!(
        cols.is_multiple_of(Q4_0_BLOCK_SIZE)
            && gate.len() >= rows * cols
            && up.len() >= rows * cols,
        "quantize_gate_up_q4_0: los límites de los slices no coinciden"
    );
    let bpr = cols / Q4_0_BLOCK_SIZE;
    let mut dst = AlignedBuf::<Q4_0Block>::new_zeroed(2 * rows * bpr);
    for r in 0..rows {
        quantize_bf16_to_q4_0_into(
            &gate[r * cols..(r + 1) * cols],
            1,
            cols,
            &mut dst[2 * r * bpr..(2 * r + 1) * bpr],
        );
        quantize_bf16_to_q4_0_into(
            &up[r * cols..(r + 1) * cols],
            1,
            cols,
            &mut dst[(2 * r + 1) * bpr..(2 * r + 2) * bpr],
        );
    }
    dst
}

#[inline]
pub fn erff(x: f32) -> f32 {
    extern "C" {
        fn erff(x: f32) -> f32;
    }
    unsafe { erff(x) }
}

pub fn gate_up_to_f32(dst: &mut [f32], gate: &[u16], up: &[u16], rows: usize, cols: usize) {
    assert!(
        dst.len() >= 2 * rows * cols && gate.len() >= rows * cols && up.len() >= rows * cols,
        "gate_up_to_f32: los límites de los slices no coinciden"
    );
    for r in 0..rows {
        let g = &mut dst[2 * r * cols..(2 * r + 1) * cols];
        let sg = &gate[r * cols..(r + 1) * cols];
        for k in 0..cols {
            g[k] = bf16_to_f32(sg[k]);
        }
        let u = &mut dst[(2 * r + 1) * cols..(2 * r + 2) * cols];
        let su = &up[r * cols..(r + 1) * cols];
        for k in 0..cols {
            u[k] = bf16_to_f32(su[k]);
        }
    }
}
