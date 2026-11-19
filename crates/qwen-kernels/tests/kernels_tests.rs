// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use qwen_kernels::aligned::AlignedBuf;
use qwen_kernels::blas::Transpose;
use qwen_kernels::quant::{
    f16_to_f32, f32_to_bf16_round, f32_to_f16, quant_shape_check, quantize_bf16_to_q2_0,
};
use qwen_kernels::selftest::run_kernel_selftest;
use qwen_kernels::{ComputePool, ComputeRuntime};

#[test]
fn test_aligned_buf_64() {
    let mut buf = AlignedBuf::<f32, 64>::new_zeroed(1025);
    assert_eq!((buf.as_ptr() as usize) % 64, 0);
    assert_eq!(buf.len(), 1025);
    buf.resize_zeroed(4096);
    assert_eq!((buf.as_ptr() as usize) % 64, 0);
    assert_eq!(buf.len(), 4096);
}

#[test]
fn test_selftest_zero_failures() {
    let rt = ComputeRuntime::new(Some(4));
    let (failures, report) = run_kernel_selftest(&rt);
    assert_eq!(failures, 0, "kernel self-test failed:\n{report}");
}

#[test]
fn test_compute_pool_stress() {
    let pool = Arc::new(ComputePool::new(4));
    for round in 0..200 {
        let sum = AtomicUsize::new(0);
        let nt = (round % 8) + 1;
        pool.parallel(nt, |tid, n| {
            assert_eq!(n, nt);
            sum.fetch_add(tid + 1, Ordering::Relaxed);
        });
        let expected = nt * (nt + 1) / 2;
        assert_eq!(sum.load(Ordering::SeqCst), expected);
    }

    // Múltiples pools independientes simultáneos (verifica ausencia de singleton global)
    let p1 = Arc::new(ComputePool::new(2));
    let p2 = Arc::new(ComputePool::new(2));
    std::thread::scope(|s| {
        s.spawn(|| {
            for _ in 0..100 {
                let c = AtomicUsize::new(0);
                p1.parallel(8, |_, _| {
                    c.fetch_add(1, Ordering::Relaxed);
                });
                assert_eq!(c.load(Ordering::SeqCst), 8);
            }
        });
        s.spawn(|| {
            for _ in 0..100 {
                let c = AtomicUsize::new(0);
                p2.parallel(8, |_, _| {
                    c.fetch_add(1, Ordering::Relaxed);
                });
                assert_eq!(c.load(Ordering::SeqCst), 8);
            }
        });
    });
}

#[test]
fn test_blas_sgemm_and_q2_0() {
    let rt = ComputeRuntime::new(Some(2));
    // C = A[2x3] * B[3x2]
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
    let mut c = [0.0f32; 4];
    rt.blas.sgemm(
        Transpose::NoTrans,
        Transpose::NoTrans,
        2,
        2,
        3,
        1.0,
        &a,
        3,
        &b,
        2,
        0.0,
        &mut c,
        2,
    );
    assert!((c[0] - 58.0).abs() < 1e-4);
    assert!((c[1] - 64.0).abs() < 1e-4);
    assert!((c[2] - 139.0).abs() < 1e-4);
    assert!((c[3] - 154.0).abs() < 1e-4);

    // Prueba básica de Q2_0
    let wb: Vec<u16> = (0..64).map(|i| f32_to_bf16_round((i as f32 - 32.0) * 0.1)).collect();
    let wq2 = quantize_bf16_to_q2_0(&wb, 2, 32);
    let x = vec![1.0f32; 32];
    let mut y = [0.0f32; 2];
    rt.dispatch.matvec_q2_0(&mut y, &wq2, &x, 2, 32);
    assert!(y[0] < 0.0 && y[1] > 0.0);

    // Verificación de dimensiones y roundtrip f16
    assert!(quant_shape_check("ok", 1024).is_ok());
    assert!(quant_shape_check("bad_mod", 33).is_err());
    assert!(quant_shape_check("bad_max", 8224).is_err());
    assert!((f16_to_f32(f32_to_f16(0.125)) - 0.125).abs() < 1e-6);
}

#[test]
fn test_compute_pool_drop_joins_cleanly() {
    for _ in 0..8 {
        let pool = ComputePool::new(4);
        let sum = AtomicUsize::new(0);
        pool.parallel(4, |tid, _| {
            sum.fetch_add(tid + 1, Ordering::Relaxed);
        });
        assert_eq!(sum.load(Ordering::SeqCst), 10);
        drop(pool);
    }
}

#[test]
fn test_compute_pool_panic_safety() {
    let pool = ComputePool::new(4);
    let completed = AtomicUsize::new(0);
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pool.parallel(4, |tid, _| {
            if tid == 0 {
                panic!("intentional test panic");
            }
            completed.fetch_add(1, Ordering::SeqCst);
        });
    }));
    assert!(res.is_err());
    assert_eq!(completed.load(Ordering::SeqCst), 3);

    // El pool sigue operativo tras el pánico en una tarea previa
    let next = AtomicUsize::new(0);
    pool.parallel(4, |_, _| {
        next.fetch_add(1, Ordering::SeqCst);
    });
    assert_eq!(next.load(Ordering::SeqCst), 4);
}

#[test]
fn test_compute_pool_worker_panic_propagates() {
    // Todas las tareas entran en pánico, así que al menos una cae en un hilo trabajador: el pánico
    // llega a quien llama y los trabajadores siguen vivos (la llamada siguiente no se cuelga).
    let pool = ComputePool::new(4);
    for _ in 0..20 {
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.parallel(4, |_, _| panic!("intentional test panic"));
        }));
        assert!(res.is_err());
    }
    let next = AtomicUsize::new(0);
    pool.parallel(4, |_, _| {
        next.fetch_add(1, Ordering::SeqCst);
    });
    assert_eq!(next.load(Ordering::SeqCst), 4);
}

#[test]
fn test_scalar_dispatch_parity_and_bounds() {
    use qwen_kernels::KernelDispatch;

    let sc = KernelDispatch::scalar();
    let det = KernelDispatch::detect();
    let pool = ComputePool::new(2);

    // Paridad quantize_act_int8 escalar vs dispatch detectado
    let x: Vec<f32> = (0..64).map(|i| (i as f32 - 31.5) * 0.25).collect();
    let mut q_sc = vec![0i8; 64];
    let mut q_det = vec![0i8; 64];
    let s_sc = sc.quantize_act_int8(&mut q_sc, &x);
    let s_det = det.quantize_act_int8(&mut q_det, &x);
    assert_eq!(q_sc, q_det);
    assert_eq!(s_sc.to_bits(), s_det.to_bits());

    // Paridad rms_norm + snake_activation + causal_attention
    let w = vec![1.25f32; 64];
    let mut o_sc = vec![0.0f32; 64];
    let mut o_det = vec![0.0f32; 64];
    sc.rms_norm(&mut o_sc, &x, &w, 1, 64, 1e-6);
    det.rms_norm(&mut o_det, &x, &w, 1, 64, 1e-6);
    for (a, b) in o_sc.iter().zip(o_det.iter()) {
        assert!((a - b).abs() < 1e-5);
    }

    // Paridad matvec_int8 / matvec_q4_0 escalar vs detectado
    let wb: Vec<u16> = (0..32 * 64)
        .map(|i| f32_to_bf16_round(((i % 19) as f32 - 9.0) * 0.05))
        .collect();
    let (wi, si) = qwen_kernels::quant::quantize_bf16_to_int8(&wb, 32, 64);
    let mut y_sc = vec![0.0f32; 32];
    let mut y_det = vec![0.0f32; 32];
    sc.matvec_int8(&pool, &mut y_sc, &wi, &si, &x, 32, 64);
    det.matvec_int8(&pool, &mut y_det, &wi, &si, &x, 32, 64);
    for (a, b) in y_sc.iter().zip(y_det.iter()) {
        assert!((a - b).abs() < 1e-5);
    }

    // Validación de límites seguros en KernelDispatch
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut tiny = [0.0f32; 4];
        det.rms_norm(&mut tiny, &x, &w, 1, 64, 1e-6);
    }));
    assert!(res.is_err());

    let res_b = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut y = vec![0.0f32; 65];
        det.matmat_bf16(&pool, &mut y, &wb, &x, 1, 64, 65);
    }));
    assert!(res_b.is_err());

    // Paridad matmat_int8 y matmat_q4_0 escalar vs dispatch detectado
    let wq = qwen_kernels::quant::quantize_bf16_to_q4_0(&wb, 32, 64);
    let mut x_mat = vec![0.0f32; 64 * 4];
    for (i, v) in x_mat.iter_mut().enumerate() {
        *v = ((i % 23) as f32 - 11.0) * 0.1;
    }
    let mut ym_sc = vec![0.0f32; 32 * 4];
    let mut ym_det = vec![0.0f32; 32 * 4];
    sc.matmat_int8(&pool, &mut ym_sc, &wi, &si, &x_mat, 32, 64, 4);
    det.matmat_int8(&pool, &mut ym_det, &wi, &si, &x_mat, 32, 64, 4);
    assert_eq!(ym_sc, ym_det);

    sc.matmat_q4_0(&pool, &mut ym_sc, &wq, &x_mat, 32, 64, 4);
    det.matmat_q4_0(&pool, &mut ym_det, &wq, &x_mat, 32, 64, 4);
    for (a, b) in ym_sc.iter().zip(ym_det.iter()) {
        assert!((a - b).abs() < 1e-5);
    }
}

#[test]
fn test_parallel_rows_partition_bit_exact() {
    use qwen_kernels::quant::{quantize_bf16_to_int8, quantize_bf16_to_q4_0};
    use qwen_kernels::KernelDispatch;

    // El reparto dinámico por trozos (4 hilos) da los mismos bits que una sola pasada (1 hilo),
    // también con un número de filas que no es múltiplo de 64.
    let d = KernelDispatch::detect();
    let serial = ComputePool::new(1);
    let par = ComputePool::new(4);
    let cols = 1024;
    let x: Vec<f32> = (0..2 * cols).map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.01).collect();
    for rows in [1000usize, 3072] {
        let wb: Vec<u16> = (0..rows * cols)
            .map(|i| f32_to_bf16_round(((i * 7919 % 211) as f32 - 105.0) * 0.002))
            .collect();
        let (wi, si) = quantize_bf16_to_int8(&wb, rows, cols);
        let wq = quantize_bf16_to_q4_0(&wb, rows, cols);
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<u32>>();
        let run = |pool: &ComputePool| {
            let mut a = vec![0.0f32; rows];
            let mut b = vec![0.0f32; rows];
            let mut c = vec![0.0f32; rows];
            let mut e = vec![0.0f32; 2 * rows];
            let mut g = vec![0.0f32; 2 * rows];
            d.matvec_bf16(pool, &mut a, &wb, &x, rows, cols);
            d.matvec_int8(pool, &mut b, &wi, &si, &x, rows, cols);
            d.matvec_q4_0(pool, &mut c, &wq, &x, rows, cols);
            d.matmat_int8(pool, &mut e, &wi, &si, &x, rows, cols, 2);
            d.matmat_q4_0(pool, &mut g, &wq, &x, rows, cols, 2);
            [bits(&a), bits(&b), bits(&c), bits(&e), bits(&g)]
        };
        assert_eq!(run(&serial), run(&par), "rows={rows}");
    }
}

#[test]
fn test_kernel_choice_parse() {
    use qwen_kernels::KernelChoice;

    for v in [None, Some(""), Some("auto"), Some(" AUTO ")] {
        assert_eq!(KernelChoice::parse(v), Ok(KernelChoice::Auto), "{v:?}");
    }
    for v in [Some("scalar"), Some("Scalar"), Some(" scalar\n")] {
        assert_eq!(KernelChoice::parse(v), Ok(KernelChoice::Scalar), "{v:?}");
    }
    let err = KernelChoice::parse(Some("neon")).unwrap_err();
    assert!(err.contains("QWEN_KERNELS=neon"), "{err}");
}

#[test]
fn test_available_dispatch_tables() {
    use qwen_kernels::{CpuCaps, KernelChoice, KernelDispatch};

    let caps = CpuCaps::detect();
    let tables = KernelDispatch::available(&caps);
    let names: Vec<&str> = tables.iter().map(|t| t.name()).collect();
    assert_eq!(names[0], "scalar", "{names:?}");
    #[cfg(target_arch = "aarch64")]
    {
        assert!(names.contains(&"neon"), "{names:?}");
        assert_eq!(names.contains(&"neon-dotprod"), caps.dotprod, "{names:?}");
    }
    #[cfg(target_arch = "x86_64")]
    assert_eq!(names.contains(&"avx2"), caps.has_avx2_fma(), "{names:?}");
    if KernelChoice::from_env() == KernelChoice::Auto {
        assert_eq!(KernelDispatch::for_caps(&caps).name(), *names.last().unwrap());
    }
}

/// Las rutas SIMD suman Q4_0 en el mismo orden (el de AVX2), así que dan los mismos bits en x86 y
/// en ARM. La huella de referencia sale de la ruta AVX2; si cambia el orden de suma de cualquier
/// ruta SIMD, esta prueba falla.
#[test]
fn test_simd_q4_0_matches_avx2_summation_order() {
    use qwen_kernels::quant::quantize_bf16_to_q4_0;
    use qwen_kernels::{CpuCaps, KernelDispatch};

    const AVX2_FINGERPRINT: u64 = 0x5df1d827ee451ac1;
    let pool = ComputePool::new(1);
    let fnv = |h: u64, v: &[f32]| {
        v.iter().fold(h, |h, f| (h ^ f.to_bits() as u64).wrapping_mul(0x100000001b3))
    };
    for &simd in KernelDispatch::available(&CpuCaps::detect()).iter().skip(1) {
        let mut h = 0xcbf29ce484222325u64;
        // 1024 columnas usa el kernel de 2 filas y 2048, el de 1 fila; 257 filas, la fila suelta.
        for (rows, cols) in [(257usize, 1024usize), (64, 2048)] {
            let wb: Vec<u16> = (0..rows * cols)
                .map(|i| f32_to_bf16_round(((i * 7919 % 211) as f32 - 105.0) * 0.002))
                .collect();
            let x: Vec<f32> = (0..3 * cols).map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.01).collect();
            let wq = quantize_bf16_to_q4_0(&wb, rows, cols);
            let mut y = vec![0.0f32; 3 * rows];
            simd.matvec_q4_0(&pool, &mut y[..rows], &wq, &x, rows, cols);
            h = fnv(h, &y[..rows]);
            simd.matmat_q4_0(&pool, &mut y, &wq, &x, rows, cols, 3);
            h = fnv(h, &y);
        }
        println!("{}: huella Q4_0 = {h:#018x}", simd.name());
        assert_eq!(h, AVX2_FINGERPRINT, "{}", simd.name());
    }
}

/// Como `test_simd_q4_0_matches_avx2_summation_order`, para los kernels de coma flotante que no
/// usan `libm` (la atención usa `exp` y el *snake*, `exp` y `sin`, que cambian entre plataformas):
/// normas RMS, BF16 (`matvec`, `matmat`, `argmax`), RoPE, matvec `f32` acumulado, cuantización de
/// activaciones y conversiones. Longitudes con resto para cubrir las colas escalares.
#[test]
fn test_simd_float_kernels_match_avx2_summation_order() {
    use qwen_kernels::{CpuCaps, KernelDispatch};

    const AVX2_FINGERPRINT: u64 = 0x841cba85449e95a2;
    let pool = ComputePool::new(1);
    let fnv = |h: u64, v: &[f32]| {
        v.iter().fold(h, |h, f| (h ^ f.to_bits() as u64).wrapping_mul(0x100000001b3))
    };
    let val = |i: usize, m: usize| ((i * 7919 % m) as f32 - (m / 2) as f32) * 0.003;
    for &simd in KernelDispatch::available(&CpuCaps::detect()).iter().skip(1) {
        let mut h = 0xcbf29ce484222325u64;
        let (rows, cols) = (37usize, 1027usize);
        let x: Vec<f32> = (0..3 * cols).map(|i| val(i, 211)).collect();
        let w: Vec<f32> = (0..cols).map(|i| 1.0 + val(i, 97)).collect();
        let wb: Vec<u16> = (0..rows * cols).map(|i| f32_to_bf16_round(val(i, 389))).collect();
        let mut y = vec![0.0f32; 3 * cols];

        simd.rms_norm(&mut y, &x, &w, 3, cols, 1e-6);
        h = fnv(h, &y);
        let mut xh = x[..2 * 16 * 64].to_vec();
        simd.rms_norm_per_head(&mut xh, &w[..64], 2, 16, 64, 1e-6);
        h = fnv(h, &xh);
        simd.matvec_bf16(&pool, &mut y[..rows], &wb, &x, rows, cols);
        h = fnv(h, &y[..rows]);
        simd.matmat_bf16(&pool, &mut y[..3 * rows], &wb, &x, rows, cols, 3);
        h = fnv(h, &y[..3 * rows]);
        h = fnv(h, &[simd.argmax_matvec_bf16(&x, &wb, cols, rows) as f32]);
        let cs: Vec<f32> = (0..4 * 64).map(|i| val(i, 53)).collect();
        let mut xr = x[..16 * 128].to_vec();
        simd.apply_rope_neox_inplace(&mut xr, 16, 128, &cs, &cs[64..], 2);
        h = fnv(h, &xr);
        let wf: Vec<f32> = (0..rows * cols).map(|i| val(i, 173)).collect();
        let mut ya = x[..rows].to_vec();
        simd.f32_matvec_accum(&mut ya, &wf, &x, rows, cols);
        h = fnv(h, &ya);
        let mut q = vec![0i8; cols];
        let s = simd.quantize_act_int8(&mut q, &x[..cols]);
        h = fnv(h, &[s]);
        h = fnv(h, &q.iter().map(|&b| b as f32).collect::<Vec<f32>>());
        let mut u = vec![0u16; cols];
        simd.f32_to_bf16_vec(&mut u, &x[..cols]);
        let mut d = x[..cols].to_vec();
        simd.bf16_accum_f32(&mut d, &u);
        h = fnv(h, &d);

        println!("{}: huella de los kernels de coma flotante = {h:#018x}", simd.name());
        assert_eq!(h, AVX2_FINGERPRINT, "{}", simd.name());
    }
}

#[test]
fn test_spin_budget_for_cores() {
    use qwen_kernels::pool::QWEN_POOL_SPIN_DEFAULT;
    use qwen_kernels::spin_budget_for;

    // Solo sin QWEN_POOL_SPIN: la variable tiene prioridad sobre la regla.
    if std::env::var_os("QWEN_POOL_SPIN").is_some() {
        return;
    }
    assert_eq!(spin_budget_for(4, 4), 0, "los hilos llenan los núcleos");
    assert_eq!(spin_budget_for(4, 2), 0);
    assert_eq!(spin_budget_for(4, 6), QWEN_POOL_SPIN_DEFAULT);
    assert_eq!(spin_budget_for(4, 8), QWEN_POOL_SPIN_DEFAULT);
    assert_eq!(spin_budget_for(1, 0), 0, "cero núcleos cuenta como uno");
}
