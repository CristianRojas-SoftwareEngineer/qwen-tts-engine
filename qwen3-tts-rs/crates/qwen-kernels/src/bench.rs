use std::fmt::Write;
use std::time::Instant;

use crate::aligned::AlignedBuf;
use crate::dispatch::ComputeRuntime;
use crate::quant::{f32_to_bf16_round, quantize_bf16_to_int8, quantize_bf16_to_q4_0};

pub fn run_matmat_bench(rt: &ComputeRuntime) -> String {
    let mut out = String::new();
    let b_dim = std::env::var("QWEN_BATCH_B")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&v| (1..=64).contains(&v))
        .unwrap_or(8);

    let shapes = [(3072usize, 1024usize), (1024, 3072), (2048, 1024)];
    let mut rng: u64 = 0x1234567u64;
    let mut rf = || -> f32 {
        rng = rng
            .wrapping_mul(6364136223846793005u64)
            .wrapping_add(1442695040888963407u64);
        ((rng >> 40) as f32) / ((1u32 << 24) as f32) * 2.0f32 - 1.0f32
    };

    let d = rt.dispatch;
    let pool = &rt.pool;

    let _ = writeln!(
        out,
        "matmat-bench: B={}, threads={}  (B*matvec [seq] vs matmat [batched])",
        b_dim,
        pool.n_threads()
    );
    let _ = writeln!(
        out,
        "  speedup>1 => batching (weight read+unpack once) beats re-reading per stream\n"
    );

    for &(rows, cols) in &shapes {
        let mut wb = AlignedBuf::<u16>::new_zeroed(rows * cols);
        for v in wb.iter_mut() {
            *v = f32_to_bf16_round(rf());
        }
        let (wi, sc) = quantize_bf16_to_int8(&wb, rows, cols);
        let wq = quantize_bf16_to_q4_0(&wb, rows, cols);
        let mut x_mat = AlignedBuf::<f32>::new_zeroed(cols * b_dim);
        for v in x_mat.iter_mut() {
            *v = rf();
        }
        let mut xb = AlignedBuf::<f32>::new_zeroed(cols);
        for k in 0..cols {
            xb[k] = x_mat[k * b_dim];
        }
        let mut y_mat = AlignedBuf::<f32>::new_zeroed(rows * b_dim);
        let mut yc = AlignedBuf::<f32>::new_zeroed(rows);

        let mb = (rows * cols * 2) as f64 / (1024.0 * 1024.0);
        let reps = if mb > 8.0 { 8 } else { 24 };
        let _ = writeln!(out, "  [{:4}x{:4}]  ({:.1} MB bf16)", rows, cols, mb);

        for (p, pn) in [(0, "bf16"), (1, "int8"), (2, "int4")] {
            match p {
                0 => {
                    d.matvec_bf16(pool, &mut yc, &wb, &xb, rows, cols);
                    d.matmat_bf16(pool, &mut y_mat, &wb, &x_mat, rows, cols, b_dim);
                }
                1 => {
                    d.matvec_int8(pool, &mut yc, &wi, &sc, &xb, rows, cols);
                    d.matmat_int8(pool, &mut y_mat, &wi, &sc, &x_mat, rows, cols, b_dim);
                }
                _ => {
                    d.matvec_q4_0(pool, &mut yc, &wq, &xb, rows, cols);
                    d.matmat_q4_0(pool, &mut y_mat, &wq, &x_mat, rows, cols, b_dim);
                }
            }

            let t0 = Instant::now();
            for _ in 0..reps {
                for b in 0..b_dim {
                    for k in 0..cols {
                        xb[k] = x_mat[k * b_dim + b];
                    }
                    match p {
                        0 => d.matvec_bf16(pool, &mut yc, &wb, &xb, rows, cols),
                        1 => d.matvec_int8(pool, &mut yc, &wi, &sc, &xb, rows, cols),
                        _ => d.matvec_q4_0(pool, &mut yc, &wq, &xb, rows, cols),
                    }
                }
            }
            let t_seq = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;

            let t0 = Instant::now();
            for _ in 0..reps {
                match p {
                    0 => d.matmat_bf16(pool, &mut y_mat, &wb, &x_mat, rows, cols, b_dim),
                    1 => d.matmat_int8(pool, &mut y_mat, &wi, &sc, &x_mat, rows, cols, b_dim),
                    _ => d.matmat_q4_0(pool, &mut y_mat, &wq, &x_mat, rows, cols, b_dim),
                }
            }
            let t_batch = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;

            let _ = writeln!(
                out,
                "     {:<5}  seq {:7.2} ms   batch {:7.2} ms   SPEEDUP {:.2}x",
                pn,
                t_seq,
                t_batch,
                t_seq / t_batch
            );
        }
    }
    out
}
