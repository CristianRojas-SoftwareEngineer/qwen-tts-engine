// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::fmt::Write;

use crate::aligned::AlignedBuf;
use crate::dispatch::ComputeRuntime;
use crate::quant::{
    bf16_to_f32, f16_to_f32, f32_to_bf16_round, quantize_act_int8_scalar, quantize_bf16_to_int8,
    quantize_bf16_to_q4_0, Q4_0_BLOCK_SIZE,
};

struct Lcg(u64);

impl Lcg {
    #[inline]
    fn next_f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005u64)
            .wrapping_add(1442695040888963407u64);
        ((self.0 >> 40) as f32) / ((1u32 << 24) as f32) * 2.0f32 - 1.0f32
    }
}

pub fn run_kernel_selftest(rt: &ComputeRuntime) -> (usize, String) {
    let mut out = String::new();
    let mut rng = Lcg(0x9E3779B97F4A7C15u64);
    let cases = [
        (3072usize, 1024usize),
        (2048, 1024),
        (257, 320),
        (1024, 2048),
        (255, 3072),
    ];
    let mut failures = 0usize;

    let _ = writeln!(out, "qwen-tts-engine kernel self-test (matvec correctness vs f32 reference)");
    out.push_str(&rt.caps_report());
    let _ = writeln!(out);

    let d = rt.dispatch;
    let pool = &rt.pool;

    for &(rows, cols) in &cases {
        let mut x = AlignedBuf::<f32>::new_zeroed(cols);
        for v in x.iter_mut() {
            *v = rng.next_f();
        }
        let mut wb = AlignedBuf::<u16>::new_zeroed(rows * cols);
        for v in wb.iter_mut() {
            *v = f32_to_bf16_round(rng.next_f());
        }
        let mut ref_y = vec![0.0f64; rows];
        let mut y = AlignedBuf::<f32>::new_zeroed(rows);

        // 1. matvec bf16
        for r in 0..rows {
            let row = &wb[r * cols..(r + 1) * cols];
            let mut s = 0.0f32;
            for k in 0..cols {
                s += bf16_to_f32(row[k]) * x[k];
            }
            ref_y[r] = s as f64;
        }
        d.matvec_bf16(pool, &mut y, &wb, &x, rows, cols);
        let mut max_rel_bf16 = 0.0f64;
        for r in 0..rows {
            let denom = ref_y[r].abs() + 1e-3;
            let rel = ((y[r] as f64) - ref_y[r]).abs() / denom;
            if rel > max_rel_bf16 {
                max_rel_bf16 = rel;
            }
        }

        // 2. matmat bf16 por lotes
        {
            const B: usize = 8;
            let mut xb_mat = AlignedBuf::<f32>::new_zeroed(cols * B);
            let mut yb_mat = AlignedBuf::<f32>::new_zeroed(rows * B);
            let mut xb_col = AlignedBuf::<f32>::new_zeroed(cols);
            let mut yc_col = AlignedBuf::<f32>::new_zeroed(rows);
            for k in 0..cols {
                for b in 0..B {
                    xb_mat[k * B + b] = x[k] * (1.0f32 + 0.05f32 * b as f32);
                }
            }
            d.matmat_bf16(pool, &mut yb_mat, &wb, &xb_mat, rows, cols, B);
            let mut l2n = 0.0f64;
            let mut l2d = 0.0f64;
            for b in 0..B {
                for k in 0..cols {
                    xb_col[k] = x[k] * (1.0f32 + 0.05f32 * b as f32);
                }
                d.matvec_bf16(pool, &mut yc_col, &wb, &xb_col, rows, cols);
                for r in 0..rows {
                    let diff = (yb_mat[r * B + b] as f64) - (yc_col[r] as f64);
                    l2n += diff * diff;
                    l2d += (yc_col[r] as f64) * (yc_col[r] as f64);
                }
            }
            let l2rel = if l2d > 0.0 { (l2n / l2d).sqrt() } else { 0.0 };
            let ok = l2rel < 1e-4;
            let _ = writeln!(
                out,
                "  [{:4}x{:4}] matmat(B={}) vs B*matvec: L2_rel={:.2e}  {}",
                rows,
                cols,
                B,
                l2rel,
                if ok { "PASS" } else { "FAIL" }
            );
            if !ok {
                failures += 1;
            }
        }

        // 3. matvec int8 + argmax
        let (wi, sc) = quantize_bf16_to_int8(&wb, rows, cols);
        for r in 0..rows {
            let row = &wi[r * cols..(r + 1) * cols];
            let mut s = 0.0f32;
            for k in 0..cols {
                s += (row[k] as f32) * x[k];
            }
            ref_y[r] = (sc[r] * s) as f64;
        }
        d.matvec_int8(pool, &mut y, &wi, &sc, &x, rows, cols);
        let mut l2_num = 0.0f64;
        let mut l2_den = 0.0f64;
        for r in 0..rows {
            let diff = (y[r] as f64) - ref_y[r];
            l2_num += diff * diff;
            l2_den += ref_y[r] * ref_y[r];
        }
        let rel_l2_i8 = (l2_num / (l2_den + 1e-12)).sqrt();

        let mut amax_ref = 0usize;
        let mut amax_val = ref_y[0];
        for (r, &v) in ref_y.iter().enumerate().take(rows).skip(1) {
            if v > amax_val {
                amax_val = v;
                amax_ref = r;
            }
        }
        let amax_got = d.argmax_matvec_int8(pool, &x, &wi, &sc, cols, rows);
        let argmax_ok = (amax_got == amax_ref)
            || (amax_got < rows && (amax_val - ref_y[amax_got]) < 0.02 * (amax_val.abs() + 1e-3));

        let bf16_ok = max_rel_bf16 < 1e-2;
        let i8_ok = rel_l2_i8 < 3e-2;
        if !bf16_ok || !i8_ok || !argmax_ok {
            failures += 1;
        }
        let _ = writeln!(
            out,
            "  [{:4}x{:<4}] bf16 max_rel={:.2e} {} | int8 rel_L2={:.2e} {} | argmax {} (ref={} got={})",
            rows,
            cols,
            max_rel_bf16,
            if bf16_ok { "OK" } else { "FAIL" },
            rel_l2_i8,
            if i8_ok { "OK" } else { "FAIL" },
            if argmax_ok { "OK" } else { "FAIL" },
            amax_ref,
            amax_got
        );

        // 4. matmat int8 por lotes == B*matvec (0.0 exacto)
        {
            const B: usize = 8;
            let mut xb_mat = AlignedBuf::<f32>::new_zeroed(cols * B);
            let mut yb_mat = AlignedBuf::<f32>::new_zeroed(rows * B);
            let mut xb_col = AlignedBuf::<f32>::new_zeroed(cols);
            let mut yc_col = AlignedBuf::<f32>::new_zeroed(rows);
            for k in 0..cols {
                for b in 0..B {
                    xb_mat[k * B + b] = x[k] * (1.0f32 + 0.05f32 * b as f32);
                }
            }
            d.matmat_int8(pool, &mut yb_mat, &wi, &sc, &xb_mat, rows, cols, B);
            let mut l2n = 0.0f64;
            let mut l2d = 0.0f64;
            for b in 0..B {
                for k in 0..cols {
                    xb_col[k] = x[k] * (1.0f32 + 0.05f32 * b as f32);
                }
                d.matvec_int8(pool, &mut yc_col, &wi, &sc, &xb_col, rows, cols);
                for r in 0..rows {
                    let diff = (yb_mat[r * B + b] as f64) - (yc_col[r] as f64);
                    l2n += diff * diff;
                    l2d += (yc_col[r] as f64) * (yc_col[r] as f64);
                }
            }
            let l2rel = if l2d > 0.0 { (l2n / l2d).sqrt() } else { 0.0 };
            let ok = l2rel == 0.0;
            let _ = writeln!(
                out,
                "  [{:4}x{:4}] matmat_int8(B={}) vs B*matvec_int8: L2_rel={:.2e}  {}",
                rows,
                cols,
                B,
                l2rel,
                if ok { "PASS" } else { "FAIL" }
            );
            if !ok {
                failures += 1;
            }
        }

        // 5. q4_0 matvec, matmat, qkv y 1 fila vs bloque
        if cols % Q4_0_BLOCK_SIZE == 0 {
            const B: usize = 8;
            let nb = cols / Q4_0_BLOCK_SIZE;
            let wq = quantize_bf16_to_q4_0(&wb, rows, cols);

            for r in 0..rows {
                let row = &wq[r * nb..(r + 1) * nb];
                let mut s = 0.0f32;
                for b in 0..nb {
                    let bs = f16_to_f32(row[b].scale_f16);
                    let xk = &x[b * Q4_0_BLOCK_SIZE..(b + 1) * Q4_0_BLOCK_SIZE];
                    for i in 0..16 {
                        s += bs * ((row[b].qs[i] & 0x0F) as i32 - 8) as f32 * xk[2 * i];
                        s += bs * ((row[b].qs[i] >> 4) as i32 - 8) as f32 * xk[2 * i + 1];
                    }
                }
                ref_y[r] = s as f64;
            }
            d.matvec_q4_0(pool, &mut y, &wq, &x, rows, cols);
            let mut l2n = 0.0f64;
            let mut l2d = 0.0f64;
            for r in 0..rows {
                let diff = (y[r] as f64) - ref_y[r];
                l2n += diff * diff;
                l2d += ref_y[r] * ref_y[r];
            }
            let l2rel = (l2n / (l2d + 1e-12)).sqrt();
            let ok = l2rel < 3e-2;
            let _ = writeln!(
                out,
                "  [{:4}x{:4}] matvec_q4_0 vs dequant ref: rel_L2={:.2e}  {}",
                rows,
                cols,
                l2rel,
                if ok { "PASS" } else { "FAIL" }
            );
            if !ok {
                failures += 1;
            }

            let mut xb_mat = AlignedBuf::<f32>::new_zeroed(cols * B);
            let mut yb_mat = AlignedBuf::<f32>::new_zeroed(rows * B);
            let mut xb_col = AlignedBuf::<f32>::new_zeroed(cols);
            let mut yc_col = AlignedBuf::<f32>::new_zeroed(rows);
            for k in 0..cols {
                for b in 0..B {
                    xb_mat[k * B + b] = x[k] * (1.0f32 + 0.05f32 * b as f32);
                }
            }
            d.matmat_q4_0(pool, &mut yb_mat, &wq, &xb_mat, rows, cols, B);
            let mut l2n = 0.0f64;
            let mut l2d = 0.0f64;
            for b in 0..B {
                for k in 0..cols {
                    xb_col[k] = x[k] * (1.0f32 + 0.05f32 * b as f32);
                }
                d.matvec_q4_0(pool, &mut yc_col, &wq, &xb_col, rows, cols);
                for r in 0..rows {
                    let diff = (yb_mat[r * B + b] as f64) - (yc_col[r] as f64);
                    l2n += diff * diff;
                    l2d += (yc_col[r] as f64) * (yc_col[r] as f64);
                }
            }
            let l2rel = if l2d > 0.0 { (l2n / l2d).sqrt() } else { 0.0 };
            let ok = l2rel <= 1e-6;
            let _ = writeln!(
                out,
                "  [{:4}x{:4}] matmat_q4_0(B={}) vs B*matvec_q4_0: L2_rel={:.2e}  {}",
                rows,
                cols,
                B,
                l2rel,
                if ok { "PASS" } else { "FAIL" }
            );
            if !ok {
                failures += 1;
            }

            // qkv_q4_0 vs 3*matvec_q4_0
            let qd = rows / 2;
            let kvd = rows / 4;
            let wq_slice = &wq[..qd * nb];
            let wk_slice = &wq[qd * nb..(qd + kvd) * nb];
            let wv_slice = &wq[(qd + kvd) * nb..(qd + 2 * kvd) * nb];
            let (q_out, rest) = y.split_at_mut(qd);
            let (k_out, v_rest) = rest.split_at_mut(kvd);
            let v_out = &mut v_rest[..kvd];
            d.matvec_q4_0_qkv(
                pool, q_out, k_out, v_out, wq_slice, wk_slice, wv_slice, &x, cols, qd, kvd,
            );
            d.matvec_q4_0(pool, &mut yc_col[..qd], wq_slice, &x, qd, cols);
            d.matvec_q4_0(pool, &mut yc_col[qd..qd + kvd], wk_slice, &x, kvd, cols);
            d.matvec_q4_0(
                pool,
                &mut yc_col[qd + kvd..qd + 2 * kvd],
                wv_slice,
                &x,
                kvd,
                cols,
            );
            let ok_qkv = (0..qd + 2 * kvd).all(|i| y[i].to_bits() == yc_col[i].to_bits());
            let _ = writeln!(
                out,
                "  [{:4}x{:4}] qkv_q4_0 vs 3*matvec_q4_0: {}",
                rows,
                cols,
                if ok_qkv { "PASS (exacto)" } else { "FAIL" }
            );
            if !ok_qkv {
                failures += 1;
            }

            // fila sola vs en bloque
            let mut y0 = [0.0f32; 1];
            d.matvec_q4_0(pool, &mut y0, &wq[..nb], &x, 1, cols);
            d.matvec_q4_0(pool, &mut yc_col, &wq, &x, rows, cols);
            let ok_row0 = y0[0].to_bits() == yc_col[0].to_bits();
            let _ = writeln!(
                out,
                "  [{:4}x{:4}] matvec_q4_0 fila sola vs en bloque: {}",
                rows,
                cols,
                if ok_row0 { "PASS (exacto)" } else { "FAIL" }
            );
            if !ok_row0 {
                failures += 1;
            }
        }
    }

    // 6. quantize_act_int8 vector + ceros + columnas vs referencia escalar (bit a bit)
    for &n in &[1024usize, 3072, 320] {
        const QB: usize = 5;
        let mut xv = vec![0.0f32; n];
        for v in xv.iter_mut() {
            *v = rng.next_f() * 100.0f32;
        }
        xv[0] = 127.0;
        xv[1] = 2.5;
        xv[2] = 3.5;
        xv[3] = -2.5;
        xv[4] = 0.5;

        let mut qa = vec![0i8; n];
        let mut qr = vec![0i8; n];
        let sa = d.quantize_act_int8(&mut qa, &xv);
        let sr = quantize_act_int8_scalar(&mut qr, &xv);
        let bad = qa != qr || sa.to_bits() != sr.to_bits();
        let _ = writeln!(
            out,
            "  [quantize n={:4}] vector vs referencia escalar (bit a bit): {}",
            n,
            if bad { "FAIL" } else { "PASS" }
        );
        if bad {
            failures += 1;
        }

        let xc = vec![0.0f32; n];
        let sa = d.quantize_act_int8(&mut qa, &xc);
        let sr = quantize_act_int8_scalar(&mut qr, &xc);
        let bad = qa != qr || sa.to_bits() != sr.to_bits();
        let _ = writeln!(
            out,
            "  [quantize n={:4}] ceros vs referencia escalar (bit a bit): {}",
            n,
            if bad { "FAIL" } else { "PASS" }
        );
        if bad {
            failures += 1;
        }

        let mut x_mat = vec![0.0f32; n * QB];
        for k in 0..n {
            for b in 0..QB {
                x_mat[k * QB + b] = match b {
                    0 => xv[k],
                    1 => xv[k] * 0.37,
                    4 => 0.0,
                    _ => rng.next_f() * (1.0 + b as f32),
                };
            }
        }
        let mut qc = vec![0i8; n * QB];
        let mut sc5 = [0.0f32; QB];
        d.quantize_act_int8_cols(&mut qc, &mut sc5, &x_mat, n, QB);
        let mut bad_cols = false;
        let mut col_tmp = vec![0.0f32; n];
        for b in 0..QB {
            for k in 0..n {
                col_tmp[k] = x_mat[k * QB + b];
            }
            let sr = quantize_act_int8_scalar(&mut qr, &col_tmp);
            if qc[b * n..(b + 1) * n] != qr[..] || sc5[b].to_bits() != sr.to_bits() {
                bad_cols = true;
                break;
            }
        }
        let _ = writeln!(
            out,
            "  [quantize n={:4}] columnas (B={}) vs referencia escalar (bit a bit): {}",
            n,
            QB,
            if bad_cols { "FAIL" } else { "PASS" }
        );
        if bad_cols {
            failures += 1;
        }
    }

    let _ = writeln!(
        out,
        "\n{} ({} case{} failed)",
        if failures > 0 {
            "SELF-TEST FAILED"
        } else {
            "SELF-TEST PASSED"
        },
        failures,
        if failures == 1 { "" } else { "s" }
    );
    (failures, out)
}
