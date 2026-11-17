use std::os::raw::c_int;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Transpose {
    NoTrans,
    Trans,
}

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
}

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
            unsafe {
                openblas_set_num_threads(n.max(1) as c_int);
            }
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
            if beta == 0.0 && m > 0 && n > 0 {
                for row in 0..m {
                    c[row * ldc..row * ldc + n].fill(0.0);
                }
            }
            return;
        }
        let ta = match transa {
            Transpose::NoTrans => CblasTranspose::NoTrans as c_int,
            Transpose::Trans => CblasTranspose::Trans as c_int,
        };
        let tb = match transb {
            Transpose::NoTrans => CblasTranspose::NoTrans as c_int,
            Transpose::Trans => CblasTranspose::Trans as c_int,
        };
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
            if beta == 0.0 && m > 0 && n > 0 {
                for row in 0..m {
                    c[row * ldc..row * ldc + n].fill(0.0);
                }
            }
            return;
        }

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
