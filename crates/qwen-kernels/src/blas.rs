// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::os::raw::c_int;
#[cfg(not(target_os = "macos"))]
use std::{
    cell::Cell,
    sync::{
        atomic::{AtomicUsize, Ordering},
        OnceLock,
    },
};

/// Modo de transposición de matrices para `BlasBackend::sgemm`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Transpose {
    NoTrans,
    Trans,
}

/// Abstracción del backend BLAS denso en `f32` (`FfiBlas` por defecto o `PureRustBlas`).
pub trait BlasBackend: Send + Sync {
    fn name(&self) -> &'static str;
    fn set_threads(&self, n: usize);
    #[allow(clippy::too_many_arguments)]
    fn sgemm(
        &self,
        transa: Transpose,
        transb: Transpose,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &[f32],
        lda: usize,
        b: &[f32],
        ldb: usize,
        beta: f32,
        c: &mut [f32],
        ldc: usize,
    );
}

#[repr(C)]
#[allow(non_camel_case_types)]
enum CblasOrder {
    RowMajor = 101,
}

#[repr(C)]
#[allow(non_camel_case_types)]
enum CblasTranspose {
    NoTrans = 111,
    Trans = 112,
}

extern "C" {
    fn cblas_sgemm(
        order: c_int,
        transa: c_int,
        transb: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: f32,
        a: *const f32,
        lda: c_int,
        b: *const f32,
        ldb: c_int,
        beta: f32,
        c: *mut f32,
        ldc: c_int,
    );

    #[cfg(not(target_os = "macos"))]
    fn openblas_set_num_threads(num_threads: c_int);

    #[cfg(not(target_os = "macos"))]
    fn openblas_get_num_threads() -> c_int;

    #[cfg(all(windows, target_env = "gnu"))]
    fn _putenv(envstring: *const std::ffi::c_char) -> c_int;
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn validate_sgemm_bounds(
    transa: Transpose,
    transb: Transpose,
    m: usize,
    n: usize,
    k: usize,
    a_len: usize,
    lda: usize,
    b_len: usize,
    ldb: usize,
    c_len: usize,
    ldc: usize,
    check_c_int: bool,
) {
    if check_c_int {
        assert!(
            m <= c_int::MAX as usize
                && n <= c_int::MAX as usize
                && k <= c_int::MAX as usize
                && lda <= c_int::MAX as usize
                && ldb <= c_int::MAX as usize
                && ldc <= c_int::MAX as usize,
            "sgemm dimensions exceed c_int::MAX"
        );
    } else {
        assert!(
            lda <= isize::MAX as usize && ldb <= isize::MAX as usize && ldc <= isize::MAX as usize,
            "sgemm leading dimensions exceed isize::MAX"
        );
    }

    let (a_rows, a_cols) = match transa {
        Transpose::NoTrans => (m, k),
        Transpose::Trans => (k, m),
    };
    assert!(lda >= a_cols.max(1), "sgemm invalid lda={lda} < {a_cols}");
    let a_req = (a_rows - 1).checked_mul(lda).and_then(|v| v.checked_add(a_cols)).expect("sgemm a size overflow");
    assert!(a_len >= a_req, "sgemm buffer a too small: {} < {}", a_len, a_req);

    let (b_rows, b_cols) = match transb {
        Transpose::NoTrans => (k, n),
        Transpose::Trans => (n, k),
    };
    assert!(ldb >= b_cols.max(1), "sgemm invalid ldb={ldb} < {b_cols}");
    let b_req = (b_rows - 1).checked_mul(ldb).and_then(|v| v.checked_add(b_cols)).expect("sgemm b size overflow");
    assert!(b_len >= b_req, "sgemm buffer b too small: {} < {}", b_len, b_req);

    assert!(ldc >= n.max(1), "sgemm invalid ldc={ldc} < {n}");
    let c_req = (m - 1).checked_mul(ldc).and_then(|v| v.checked_add(n)).expect("sgemm c size overflow");
    assert!(c_len >= c_req, "sgemm buffer c too small: {} < {}", c_len, c_req);
}

#[inline]
fn handle_zero_k_sgemm(m: usize, n: usize, beta: f32, c: &mut [f32], ldc: usize) {
    if m > 0 && n > 0 && beta != 1.0 {
        let req = (m - 1)
            .checked_mul(ldc)
            .and_then(|v| v.checked_add(n))
            .expect("sgemm zero-k c overflow");
        assert!(ldc >= n && c.len() >= req, "sgemm zero-k c bounds");
        for row in 0..m {
            let slice = &mut c[row * ldc..row * ldc + n];
            if beta == 0.0 {
                slice.fill(0.0);
            } else {
                for v in slice.iter_mut() {
                    *v *= beta;
                }
            }
        }
    }
}

/// Hilos de OpenBLAS pedidos con `set_threads` (0 = sin fijar: manda el entorno).
///
/// El estado de OpenBLAS es global al proceso, así que el valor también lo es.
#[cfg(not(target_os = "macos"))]
static BLAS_THREADS: AtomicUsize = AtomicUsize::new(0);

// Valor de `BLAS_THREADS` ya aplicado en cada hilo.
#[cfg(not(target_os = "macos"))]
thread_local! {
    #[allow(clippy::missing_const_for_thread_local)]
    static APPLIED_THREADS: Cell<usize> = const { Cell::new(0) };
}

/// Hilos con que se inicializó OpenBLAS: tope de `set_threads` (pedir más cuelga `sgemm`).
#[cfg(not(target_os = "macos"))]
fn initial_threads() -> usize {
    static INITIAL: OnceLock<usize> = OnceLock::new();
    *INITIAL.get_or_init(|| unsafe { openblas_get_num_threads() }.max(1) as usize)
}

/// Aplica en el hilo actual los hilos pedidos con `set_threads`, sin pasar de los iniciales.
///
/// Con OpenMP, cada `sgemm` de OpenBLAS se reajusta a `omp_get_max_threads()` del hilo
/// que la llama, y ese valor es por hilo. Sin esto, un hilo que no llamó a `set_threads`
/// (p. ej. el del decoder) usaría el valor global de `OMP_NUM_THREADS`.
#[cfg(not(target_os = "macos"))]
fn apply_threads_on_this_thread() {
    let n = BLAS_THREADS.load(Ordering::Relaxed).min(initial_threads());
    if n != 0 && APPLIED_THREADS.with(Cell::get) != n {
        unsafe {
            openblas_set_num_threads(n as c_int);
        }
        APPLIED_THREADS.with(|a| a.set(n));
    }
}

/// Fija los hilos con que se inicializa OpenBLAS, si el usuario no los fijó por entorno.
///
/// En `x86_64-pc-windows-gnu` OpenBLAS se enlaza estático con OpenMP. Su constructor corre
/// antes de `main` y reserva un búfer de 128 MiB (memoria comprometida) por cada hilo que
/// indique `OMP_NUM_THREADS`; sin la variable, uno por CPU lógica. `openblas_set_num_threads`
/// posterior no devuelve esa memoria, y pedir más hilos que los iniciales cuelga la primera
/// `sgemm` (OpenBLAS 0.3.34). Por eso el binario llama a esta función antes que los
/// constructores de OpenBLAS y libgomp, con el número de hilos que va a usar.
///
/// `_putenv` actualiza la tabla de entorno de la CRT, que es la que lee `getenv`. En el resto
/// de plataformas no hace nada: Linux no cobra el commit por adelantado y macOS usa Accelerate.
pub fn set_initial_openblas_threads(n: usize) {
    #[cfg(all(windows, target_env = "gnu"))]
    {
        let user_set = ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS"]
            .iter()
            .any(|v| std::env::var_os(v).is_some());
        if user_set {
            return;
        }
        let Ok(var) = std::ffi::CString::new(format!("OMP_NUM_THREADS={}", n.max(1))) else {
            return;
        };
        // Si falla, OpenBLAS arranca con su valor por defecto: más memoria, mismo resultado.
        unsafe {
            _putenv(var.as_ptr());
        }
    }
    #[cfg(not(all(windows, target_env = "gnu")))]
    {
        let _ = n;
    }
}

/// Backend BLAS por defecto enlazado vía FFI a OpenBLAS (Linux/Windows) o Accelerate (macOS).
#[derive(Debug, Default, Clone, Copy)]
pub struct FfiBlas;

impl BlasBackend for FfiBlas {
    fn name(&self) -> &'static str {
        if cfg!(target_os = "macos") {
            "Accelerate"
        } else {
            "OpenBLAS"
        }
    }

    fn set_threads(&self, n: usize) {
        #[cfg(not(target_os = "macos"))]
        {
            if std::env::var_os("OPENBLAS_NUM_THREADS").is_some() {
                return;
            }
            BLAS_THREADS.store(n.max(1), Ordering::Relaxed);
            apply_threads_on_this_thread();
        }
        #[cfg(target_os = "macos")]
        {
            let _ = n;
        }
    }

    fn sgemm(
        &self,
        transa: Transpose,
        transb: Transpose,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &[f32],
        lda: usize,
        b: &[f32],
        ldb: usize,
        beta: f32,
        c: &mut [f32],
        ldc: usize,
    ) {
        if m == 0 || n == 0 || k == 0 {
            handle_zero_k_sgemm(m, n, beta, c, ldc);
            return;
        }
        validate_sgemm_bounds(
            transa,
            transb,
            m,
            n,
            k,
            a.len(),
            lda,
            b.len(),
            ldb,
            c.len(),
            ldc,
            true,
        );
        let ta = match transa {
            Transpose::NoTrans => CblasTranspose::NoTrans as c_int,
            Transpose::Trans => CblasTranspose::Trans as c_int,
        };
        let tb = match transb {
            Transpose::NoTrans => CblasTranspose::NoTrans as c_int,
            Transpose::Trans => CblasTranspose::Trans as c_int,
        };
        #[cfg(not(target_os = "macos"))]
        apply_threads_on_this_thread();
        unsafe {
            cblas_sgemm(
                CblasOrder::RowMajor as c_int,
                ta,
                tb,
                m as c_int,
                n as c_int,
                k as c_int,
                alpha,
                a.as_ptr(),
                lda as c_int,
                b.as_ptr(),
                ldb as c_int,
                beta,
                c.as_mut_ptr(),
                ldc as c_int,
            );
        }
    }
}

#[cfg(feature = "faer-blas")]
#[derive(Debug)]
pub struct PureRustBlas {
    threads: std::sync::atomic::AtomicUsize,
}

#[cfg(feature = "faer-blas")]
impl Default for PureRustBlas {
    fn default() -> Self {
        Self {
            threads: std::sync::atomic::AtomicUsize::new(1),
        }
    }
}

#[cfg(feature = "faer-blas")]
impl BlasBackend for PureRustBlas {
    fn name(&self) -> &'static str {
        "faer (pure Rust)"
    }

    fn set_threads(&self, n: usize) {
        self.threads
            .store(n.max(1), std::sync::atomic::Ordering::Relaxed);
    }

    fn sgemm(
        &self,
        transa: Transpose,
        transb: Transpose,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &[f32],
        lda: usize,
        b: &[f32],
        ldb: usize,
        beta: f32,
        c: &mut [f32],
        ldc: usize,
    ) {
        use faer::linalg::matmul::matmul;
        use faer::{Accum, MatMut, MatRef, Par};

        if m == 0 || n == 0 || k == 0 {
            handle_zero_k_sgemm(m, n, beta, c, ldc);
            return;
        }
        validate_sgemm_bounds(
            transa,
            transb,
            m,
            n,
            k,
            a.len(),
            lda,
            b.len(),
            ldb,
            c.len(),
            ldc,
            false,
        );

        let (a_rs, a_cs) = match transa {
            Transpose::NoTrans => (lda as isize, 1isize),
            Transpose::Trans => (1isize, lda as isize),
        };
        let (b_rs, b_cs) = match transb {
            Transpose::NoTrans => (ldb as isize, 1isize),
            Transpose::Trans => (1isize, ldb as isize),
        };

        let a_mat = unsafe { MatRef::<'_, f32>::from_raw_parts(a.as_ptr(), m, k, a_rs, a_cs) };
        let b_mat = unsafe { MatRef::<'_, f32>::from_raw_parts(b.as_ptr(), k, n, b_rs, b_cs) };
        let mut c_mat =
            unsafe { MatMut::<'_, f32>::from_raw_parts_mut(c.as_mut_ptr(), m, n, ldc as isize, 1) };

        let accum = if beta == 0.0 {
            Accum::Replace
        } else {
            if beta != 1.0 {
                for r in 0..m {
                    for col in 0..n {
                        c[r * ldc + col] *= beta;
                    }
                }
            }
            Accum::Add
        };

        let nt = self.threads.load(std::sync::atomic::Ordering::Relaxed);
        let par = if nt <= 1 { Par::Seq } else { Par::rayon(nt) };
        matmul(c_mat.as_mut(), accum, a_mat, b_mat, alpha, par);
    }
}
