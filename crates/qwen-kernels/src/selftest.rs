// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::fmt::Write;

use crate::aligned::AlignedBuf;
use crate::dispatch::{ComputeRuntime, KernelDispatch};
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

fn check_matmat_vs_matvec<FMat, FVec>(
    x: &[f32],
    rows: usize,
    cols: usize,
    b_dim: usize,
    run_matmat: FMat,
    mut run_matvec: FVec,
) -> f64
where
    FMat: FnOnce(&mut [f32], &[f32]),
    FVec: FnMut(&mut [f32], &[f32]),
{
    let mut xb_mat = AlignedBuf::<f32>::new_zeroed(cols * b_dim);
    let mut yb_mat = AlignedBuf::<f32>::new_zeroed(rows * b_dim);
    let mut xb_col = AlignedBuf::<f32>::new_zeroed(cols);
    let mut yc_col = AlignedBuf::<f32>::new_zeroed(rows);
    for k in 0..cols {
        for b in 0..b_dim {
            xb_mat[k * b_dim + b] = x[k] * (1.0f32 + 0.05f32 * b as f32);
        }
    }
    run_matmat(&mut yb_mat, &xb_mat);
    let mut l2n = 0.0f64;
    let mut l2d = 0.0f64;
    for b in 0..b_dim {
        for k in 0..cols {
            xb_col[k] = x[k] * (1.0f32 + 0.05f32 * b as f32);
        }
        run_matvec(&mut yc_col, &xb_col);
        for r in 0..rows {
            let diff = (yb_mat[r * b_dim + b] as f64) - (yc_col[r] as f64);
            l2n += diff * diff;
            l2d += (yc_col[r] as f64) * (yc_col[r] as f64);
        }
    }
    if l2d > 0.0 {
        (l2n / l2d).sqrt()
    } else {
        0.0
    }
}

fn rel_l2_against_ref(got: &[f32], ref_y: &[f64]) -> f64 {
    let mut l2n = 0.0f64;
    let mut l2d = 0.0f64;
    for (&g, &r) in got.iter().zip(ref_y.iter()) {
        let diff = (g as f64) - r;
        l2n += diff * diff;
        l2d += r * r;
    }
    (l2n / (l2d + 1e-12)).sqrt()
}

fn selftest_shape_case(
    rt: &ComputeRuntime,
    rng: &mut Lcg,
    rows: usize,
    cols: usize,
    out: &mut String,
) -> usize {
    const B: usize = 8;
    let d = rt.dispatch;
    let pool = &rt.pool;
    let mut failures = 0usize;

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
    let l2rel_bf16 = check_matmat_vs_matvec(
        &x,
        rows,
        cols,
        B,
        |ym, xm| d.matmat_bf16(pool, ym, &wb, xm, rows, cols, B),
        |yc, xc| d.matvec_bf16(pool, yc, &wb, xc, rows, cols),
    );
    let ok = l2rel_bf16 < 1e-4;
    let _ = writeln!(
        out,
        "  [{:4}x{:4}] matmat(B={}) frente a B*matvec: L2_rel={:.2e}  {}",
        rows,
        cols,
        B,
        l2rel_bf16,
        if ok { "OK" } else { "FALLO" }
    );
    if !ok {
        failures += 1;
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
    let rel_l2_i8 = rel_l2_against_ref(&y, &ref_y);

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
        "  [{:4}x{:<4}] bf16 max_rel={:.2e} {} | int8 rel_L2={:.2e} {} | argmax {} (ref={} obtenido={})",
        rows,
        cols,
        max_rel_bf16,
        if bf16_ok { "OK" } else { "FALLO" },
        rel_l2_i8,
        if i8_ok { "OK" } else { "FALLO" },
        if argmax_ok { "OK" } else { "FALLO" },
        amax_ref,
        amax_got
    );

    // 4. matmat int8 por lotes == B*matvec (0.0 exacto)
    let l2rel_i8 = check_matmat_vs_matvec(
        &x,
        rows,
        cols,
        B,
        |ym, xm| d.matmat_int8(pool, ym, &wi, &sc, xm, rows, cols, B),
        |yc, xc| d.matvec_int8(pool, yc, &wi, &sc, xc, rows, cols),
    );
    let ok = l2rel_i8 == 0.0;
    let _ = writeln!(
        out,
        "  [{:4}x{:4}] matmat_int8(B={}) frente a B*matvec_int8: L2_rel={:.2e}  {}",
        rows,
        cols,
        B,
        l2rel_i8,
        if ok { "OK" } else { "FALLO" }
    );
    if !ok {
        failures += 1;
    }

    // 5. q4_0 matvec, matmat, qkv y 1 fila vs bloque
    if cols.is_multiple_of(Q4_0_BLOCK_SIZE) {
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
        let l2rel = rel_l2_against_ref(&y, &ref_y);
        let ok = l2rel < 3e-2;
        let _ = writeln!(
            out,
            "  [{:4}x{:4}] matvec_q4_0 frente a la referencia descuantizada: rel_L2={:.2e}  {}",
            rows,
            cols,
            l2rel,
            if ok { "OK" } else { "FALLO" }
        );
        if !ok {
            failures += 1;
        }

        let l2rel_q4 = check_matmat_vs_matvec(
            &x,
            rows,
            cols,
            B,
            |ym, xm| d.matmat_q4_0(pool, ym, &wq, xm, rows, cols, B),
            |yc, xc| d.matvec_q4_0(pool, yc, &wq, xc, rows, cols),
        );
        let ok = l2rel_q4 <= 1e-6;
        let _ = writeln!(
            out,
            "  [{:4}x{:4}] matmat_q4_0(B={}) frente a B*matvec_q4_0: L2_rel={:.2e}  {}",
            rows,
            cols,
            B,
            l2rel_q4,
            if ok { "OK" } else { "FALLO" }
        );
        if !ok {
            failures += 1;
        }

        // qkv_q4_0 vs 3*matvec_q4_0
        let mut yc_col = AlignedBuf::<f32>::new_zeroed(rows);
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
            "  [{:4}x{:4}] qkv_q4_0 frente a 3*matvec_q4_0: {}",
            rows,
            cols,
            if ok_qkv { "OK (exacto)" } else { "FALLO" }
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
            "  [{:4}x{:4}] matvec_q4_0 fila sola frente a en bloque: {}",
            rows,
            cols,
            if ok_row0 { "OK (exacto)" } else { "FALLO" }
        );
        if !ok_row0 {
            failures += 1;
        }
    }

    failures
}

fn selftest_quantize_case(rt: &ComputeRuntime, rng: &mut Lcg, n: usize, out: &mut String) -> usize {
    const QB: usize = 5;
    let d = rt.dispatch;
    let mut failures = 0usize;

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
        "  [quantize n={:4}] vector frente a referencia escalar (bit a bit): {}",
        n,
        if bad { "FALLO" } else { "OK" }
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
        "  [quantize n={:4}] ceros frente a referencia escalar (bit a bit): {}",
        n,
        if bad { "FALLO" } else { "OK" }
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
        "  [quantize n={:4}] columnas (B={}) frente a referencia escalar (bit a bit): {}",
        n,
        QB,
        if bad_cols { "FALLO" } else { "OK" }
    );
    if bad_cols {
        failures += 1;
    }
    failures
}

/// `||a - b|| / ||b||` en `f64`.
fn rel_l2_f32(a: &[f32], b: &[f32]) -> f64 {
    let mut l2n = 0.0f64;
    let mut l2d = 0.0f64;
    for (&x, &r) in a.iter().zip(b.iter()) {
        let diff = (x as f64) - (r as f64);
        l2n += diff * diff;
        l2d += (r as f64) * (r as f64);
    }
    (l2n / (l2d + 1e-30)).sqrt()
}

/// Compara una ruta SIMD que puede ejecutar esta CPU con la ruta escalar en las mismas entradas.
/// INT8 debe dar los mismos bits (la suma entera es exacta). Q4_0 suma en otro orden (8 carriles
/// con FMA frente a una suma por bloque), así que se exige `L2_rel <= 1e-6`.
fn selftest_simd_vs_scalar_case(
    rt: &ComputeRuntime,
    simd: &KernelDispatch,
    rng: &mut Lcg,
    rows: usize,
    cols: usize,
    out: &mut String,
) -> usize {
    const BATCHES: [usize; 2] = [3, 8];
    const Q4_TOL: f64 = 1e-6;
    let sc = KernelDispatch::scalar();
    let pool = &rt.pool;
    let name = simd.name();

    let x: Vec<f32> = (0..cols * 8).map(|_| rng.next_f()).collect();
    let wb: Vec<u16> = (0..rows * cols).map(|_| f32_to_bf16_round(rng.next_f())).collect();
    let (wi, si) = quantize_bf16_to_int8(&wb, rows, cols);
    let wq = quantize_bf16_to_q4_0(&wb, rows, cols);
    let nb = cols / Q4_0_BLOCK_SIZE;
    let same_bits = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(p, q)| p.to_bits() == q.to_bits());

    let mut results: Vec<(String, bool, String)> = Vec::new();
    let (mut ya, mut yb) = (vec![0.0f32; rows * 8], vec![0.0f32; rows * 8]);

    simd.matvec_int8(pool, &mut ya[..rows], &wi, &si, &x, rows, cols);
    sc.matvec_int8(pool, &mut yb[..rows], &wi, &si, &x, rows, cols);
    let ok = same_bits(&ya[..rows], &yb[..rows]);
    results.push(("matvec_int8".into(), ok, "bit a bit".into()));
    for b in BATCHES {
        simd.matmat_int8(pool, &mut ya[..rows * b], &wi, &si, &x, rows, cols, b);
        sc.matmat_int8(pool, &mut yb[..rows * b], &wi, &si, &x, rows, cols, b);
        let ok = same_bits(&ya[..rows * b], &yb[..rows * b]);
        results.push((format!("matmat_int8(B={b})"), ok, "bit a bit".into()));
    }

    simd.matvec_q4_0(pool, &mut ya[..rows], &wq, &x, rows, cols);
    sc.matvec_q4_0(pool, &mut yb[..rows], &wq, &x, rows, cols);
    let l2 = rel_l2_f32(&ya[..rows], &yb[..rows]);
    results.push(("matvec_q4_0".into(), l2 <= Q4_TOL, format!("L2_rel={l2:.2e}")));

    let (qd, kvd) = (rows / 2, rows / 4);
    let (wqq, wqk, wqv) = (
        &wq[..qd * nb],
        &wq[qd * nb..(qd + kvd) * nb],
        &wq[(qd + kvd) * nb..(qd + 2 * kvd) * nb],
    );
    for (d, y) in [(simd, &mut ya), (sc, &mut yb)] {
        let (q, rest) = y.split_at_mut(qd);
        let (k, rest) = rest.split_at_mut(kvd);
        d.matvec_q4_0_qkv(pool, q, k, &mut rest[..kvd], wqq, wqk, wqv, &x, cols, qd, kvd);
    }
    let n = qd + 2 * kvd;
    let l2 = rel_l2_f32(&ya[..n], &yb[..n]);
    results.push(("qkv_q4_0".into(), l2 <= Q4_TOL, format!("L2_rel={l2:.2e}")));

    for b in BATCHES {
        simd.matmat_q4_0(pool, &mut ya[..rows * b], &wq, &x, rows, cols, b);
        sc.matmat_q4_0(pool, &mut yb[..rows * b], &wq, &x, rows, cols, b);
        let l2 = rel_l2_f32(&ya[..rows * b], &yb[..rows * b]);
        results.push((format!("matmat_q4_0(B={b})"), l2 <= Q4_TOL, format!("L2_rel={l2:.2e}")));
    }

    let mut failures = 0usize;
    for (what, ok, detail) in results {
        let _ = writeln!(
            out,
            "  [{name} {rows:4}x{cols:4}] {what} frente a escalar: {detail}  {}",
            if ok { "OK" } else { "FALLO" }
        );
        if !ok {
            failures += 1;
        }
    }
    failures
}

/// Compara con la ruta escalar los kernels de coma flotante y de conversión de una ruta SIMD:
/// cuantización de activaciones y conversiones BF16, bit a bit; el resto (otro orden de suma o, en
/// el *snake*, un polinomio en vez de `sin`), `L2_rel <= 1e-6`. Las longitudes no son múltiplos
/// del ancho del vector, para cubrir los restos escalares.
fn selftest_simd_float_vs_scalar(
    rt: &ComputeRuntime,
    simd: &KernelDispatch,
    rng: &mut Lcg,
    out: &mut String,
) -> usize {
    const TOL: f64 = 1e-6;
    let sc = KernelDispatch::scalar();
    let pool = &rt.pool;
    let name = simd.name();
    let mut vec_f = |n: usize, scale: f32| (0..n).map(|_| rng.next_f() * scale).collect::<Vec<f32>>();
    let same_bits = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(p, q)| p.to_bits() == q.to_bits());
    let l2 = |what: &str, a: &[f32], b: &[f32]| {
        let e = rel_l2_f32(a, b);
        (what.to_string(), e <= TOL, format!("L2_rel={e:.2e}"))
    };
    let mut results: Vec<(String, bool, String)> = Vec::new();

    // Cuantización de activaciones (n con resto).
    let x = vec_f(1029, 3.0);
    let (mut qa, mut qb) = (vec![0i8; 1029], vec![0i8; 1029]);
    let (sa, sb) = (simd.quantize_act_int8(&mut qa, &x), sc.quantize_act_int8(&mut qb, &x));
    let ok = qa == qb && sa.to_bits() == sb.to_bits();
    results.push(("quantize_act_int8".into(), ok, "bit a bit".into()));

    // Normas RMS.
    let (seq, dim) = (3usize, 1029usize);
    let x = vec_f(seq * dim, 2.0);
    let w = vec_f(dim, 1.5);
    let (mut ya, mut yb) = (vec![0.0f32; seq * dim], vec![0.0f32; seq * dim]);
    simd.rms_norm(&mut ya, &x, &w, seq, dim, 1e-6);
    sc.rms_norm(&mut yb, &x, &w, seq, dim, 1e-6);
    results.push(l2("rms_norm", &ya, &yb));
    let res = vec_f(dim, 1.0);
    let (mut xa, mut xb) = (x[..dim].to_vec(), x[..dim].to_vec());
    simd.rms_norm_residual(&mut ya[..dim], &mut xa, &res, &w, dim, 1e-6);
    sc.rms_norm_residual(&mut yb[..dim], &mut xb, &res, &w, dim, 1e-6);
    let e = rel_l2_f32(&ya[..dim], &yb[..dim]);
    let ok = same_bits(&xa, &xb) && e <= TOL;
    results.push(("rms_norm_residual".into(), ok, format!("L2_rel={e:.2e}, suma residual bit a bit")));
    let (heads, hd) = (16usize, 128usize);
    let x = vec_f(2 * heads * hd, 2.0);
    let (mut xa, mut xb) = (x.clone(), x);
    simd.rms_norm_per_head(&mut xa, &w[..hd], 2, heads, hd, 1e-6);
    sc.rms_norm_per_head(&mut xb, &w[..hd], 2, heads, hd, 1e-6);
    results.push(l2("rms_norm_per_head", &xa, &xb));

    // BF16: matvec (filas impares, columnas con resto), matmat y argmax.
    let (rows, cols) = (257usize, 1027usize);
    let wb: Vec<u16> = vec_f(rows * cols, 1.0).iter().map(|&v| f32_to_bf16_round(v)).collect();
    let x = vec_f(cols * 8, 1.0);
    let (mut ya, mut yb) = (vec![0.0f32; rows * 8], vec![0.0f32; rows * 8]);
    simd.matvec_bf16(pool, &mut ya[..rows], &wb, &x, rows, cols);
    sc.matvec_bf16(pool, &mut yb[..rows], &wb, &x, rows, cols);
    results.push(l2("matvec_bf16", &ya[..rows], &yb[..rows]));
    for b in [3usize, 8] {
        simd.matmat_bf16(pool, &mut ya[..rows * b], &wb, &x, rows, cols, b);
        sc.matmat_bf16(pool, &mut yb[..rows * b], &wb, &x, rows, cols, b);
        results.push(l2(&format!("matmat_bf16(B={b})"), &ya[..rows * b], &yb[..rows * b]));
    }
    sc.matvec_bf16(pool, &mut yb[..rows], &wb, &x, rows, cols);
    let ia = simd.argmax_matvec_bf16(&x, &wb, cols, rows);
    let ib = sc.argmax_matvec_bf16(&x, &wb, cols, rows);
    let tie = (yb[ia] - yb[ib]).abs() <= 1e-6 * yb[ib].abs();
    results.push(("argmax_bf16".into(), ia == ib || tie, format!("simd={ia} escalar={ib}")));

    // Atención: f32 con y sin ventana, y caché BF16 en serie y repartida entre hilos.
    let (n_heads, n_kv, seq_q, seq_k) = (16usize, 8usize, 5usize, 64usize);
    let (qh, kvh) = (n_heads * hd, n_kv * hd);
    let q = vec_f(seq_q * qh, 1.0);
    let k = vec_f(seq_k * kvh, 1.0);
    let v = vec_f(seq_k * kvh, 1.0);
    let scale = 1.0 / (hd as f32).sqrt();
    let q_off = seq_k - seq_q;
    let (mut oa, mut ob) = (vec![0.0f32; seq_q * qh], vec![0.0f32; seq_q * qh]);
    for window in [0usize, 16] {
        simd.causal_attention_windowed(
            &mut oa, &q, &k, &v, seq_q, seq_k, n_heads, n_kv, hd, scale, q_off, window,
        );
        sc.causal_attention_windowed(
            &mut ob, &q, &k, &v, seq_q, seq_k, n_heads, n_kv, hd, scale, q_off, window,
        );
        results.push(l2(&format!("attn_f32(window={window})"), &oa, &ob));
    }
    let kb: Vec<u16> = k.iter().map(|&f| f32_to_bf16_round(f)).collect();
    let vb: Vec<u16> = v.iter().map(|&f| f32_to_bf16_round(f)).collect();
    simd.causal_attention_bf16kv(&mut oa, &q, &kb, &vb, seq_q, seq_k, n_heads, n_kv, hd, scale, q_off);
    sc.causal_attention_bf16kv(&mut ob, &q, &kb, &vb, seq_q, seq_k, n_heads, n_kv, hd, scale, q_off);
    results.push(l2("attn_bf16kv", &oa, &ob));
    simd.causal_attention_bf16kv_heads_parallel(
        pool, &mut oa[..qh], &q[..qh], &kb, &vb, 1, seq_k, n_heads, n_kv, hd, scale, seq_k - 1,
    );
    sc.causal_attention_bf16kv_heads_parallel(
        pool, &mut ob[..qh], &q[..qh], &kb, &vb, 1, seq_k, n_heads, n_kv, hd, scale, seq_k - 1,
    );
    results.push(l2("attn_bf16kv repartida entre hilos", &oa[..qh], &ob[..qh]));

    // RoPE.
    let half = hd / 2;
    let cos = vec_f(4 * half, 1.0);
    let sin = vec_f(4 * half, 1.0);
    let x = vec_f(n_heads * hd, 2.0);
    let (mut xa, mut xb) = (x.clone(), x);
    simd.apply_rope_neox_inplace(&mut xa, n_heads, hd, &cos, &sin, 3);
    sc.apply_rope_neox_inplace(&mut xb, n_heads, hd, &cos, &sin, 3);
    results.push(l2("rope_neox", &xa, &xb));

    // Conversiones BF16 (n con resto).
    let n = 1029usize;
    let src = vec_f(n, 4.0);
    let sb16: Vec<u16> = src.iter().map(|&f| f32_to_bf16_round(f)).collect();
    let (mut da, mut db) = (src.clone(), src.clone());
    simd.bf16_accum_f32(&mut da, &sb16);
    sc.bf16_accum_f32(&mut db, &sb16);
    let mut ok = same_bits(&da, &db);
    simd.bf16_to_f32_vec(&mut da, &sb16);
    sc.bf16_to_f32_vec(&mut db, &sb16);
    ok &= same_bits(&da, &db);
    let (mut ua, mut ub) = (vec![0u16; n], vec![0u16; n]);
    simd.f32_to_bf16_vec(&mut ua, &src);
    sc.f32_to_bf16_vec(&mut ub, &src);
    ok &= ua == ub;
    results.push(("bf16_accum, bf16_to_f32, f32_to_bf16".into(), ok, "bit a bit".into()));

    // Decoder: matvec f32 acumulado y snake.
    let (rows, cols) = (97usize, 1027usize);
    let wf = vec_f(rows * cols, 1.0);
    let x = vec_f(cols, 1.0);
    let y0 = vec_f(rows, 1.0);
    let (mut ya, mut yb) = (y0.clone(), y0);
    simd.f32_matvec_accum(&mut ya, &wf, &x, rows, cols);
    sc.f32_matvec_accum(&mut yb, &wf, &x, rows, cols);
    results.push(l2("f32_matvec_accum", &ya, &yb));
    let (channels, length) = (64usize, 1029usize);
    let data = vec_f(channels * length, 4.0);
    let la = vec_f(channels, 1.0);
    let lb = vec_f(channels, 1.0);
    let (mut da, mut db) = (data.clone(), data);
    simd.snake_activation(pool, &mut da, channels, length, &la, &lb);
    sc.snake_activation(pool, &mut db, channels, length, &la, &lb);
    results.push(l2("snake", &da, &db));

    let mut failures = 0usize;
    for (what, ok, detail) in results {
        let _ = writeln!(
            out,
            "  [{name}] {what} frente a escalar: {detail}  {}",
            if ok { "OK" } else { "FALLO" }
        );
        if !ok {
            failures += 1;
        }
    }
    failures
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

    let _ = writeln!(
        out,
        "Autoprueba de kernels de qwen-tts-engine (corrección de matvec frente a la referencia f32)"
    );
    out.push_str(&rt.caps_report());
    let _ = writeln!(out);

    for &(rows, cols) in &cases {
        failures += selftest_shape_case(rt, &mut rng, rows, cols, &mut out);
    }
    for &n in &[1024usize, 3072, 320] {
        failures += selftest_quantize_case(rt, &mut rng, n, &mut out);
    }
    // Cada ruta SIMD que puede ejecutar esta CPU frente a la escalar (filas cortas y largas,
    // número de filas impar).
    for &simd in KernelDispatch::available(&rt.caps).iter().skip(1) {
        for &(rows, cols) in &[(3072usize, 1024usize), (257, 320), (1024, 2048), (255, 3072)] {
            failures += selftest_simd_vs_scalar_case(rt, simd, &mut rng, rows, cols, &mut out);
        }
        failures += selftest_simd_float_vs_scalar(rt, simd, &mut rng, &mut out);
    }

    let _ = writeln!(
        out,
        "\n{} ({} {} con fallo)",
        if failures > 0 {
            "AUTOPRUEBA FALLIDA"
        } else {
            "AUTOPRUEBA SUPERADA"
        },
        failures,
        if failures == 1 { "caso" } else { "casos" }
    );
    (failures, out)
}
