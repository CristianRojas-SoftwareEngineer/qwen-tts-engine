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
