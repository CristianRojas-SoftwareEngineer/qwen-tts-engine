// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::os::raw::c_int;
#[cfg(not(target_os = "macos"))]
use std::sync::OnceLock;

/// Modo de transposición de matrices para `FfiBlas::sgemm`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Transpose {
    NoTrans,
    Trans,
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

    #[cfg(not(target_os = "macos"))]
    fn openblas_get_corename() -> *const std::ffi::c_char;

    #[cfg(windows)]
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
) {
    assert!(
        m <= c_int::MAX as usize
            && n <= c_int::MAX as usize
            && k <= c_int::MAX as usize
            && lda <= c_int::MAX as usize
            && ldb <= c_int::MAX as usize
            && ldc <= c_int::MAX as usize,
        "sgemm: las dimensiones superan c_int::MAX"
    );

    let (a_rows, a_cols) = match transa {
        Transpose::NoTrans => (m, k),
        Transpose::Trans => (k, m),
    };
    assert!(lda >= a_cols.max(1), "sgemm: lda={lda} no válido, menor que {a_cols}");
    let a_req = (a_rows - 1).checked_mul(lda).and_then(|v| v.checked_add(a_cols)).expect("sgemm: desbordamiento del tamaño de a");
    assert!(a_len >= a_req, "sgemm: el buffer a es demasiado pequeño: {} < {}", a_len, a_req);

    let (b_rows, b_cols) = match transb {
        Transpose::NoTrans => (k, n),
        Transpose::Trans => (n, k),
    };
    assert!(ldb >= b_cols.max(1), "sgemm: ldb={ldb} no válido, menor que {b_cols}");
    let b_req = (b_rows - 1).checked_mul(ldb).and_then(|v| v.checked_add(b_cols)).expect("sgemm: desbordamiento del tamaño de b");
    assert!(b_len >= b_req, "sgemm: el buffer b es demasiado pequeño: {} < {}", b_len, b_req);

    assert!(ldc >= n.max(1), "sgemm: ldc={ldc} no válido, menor que {n}");
    let c_req = (m - 1).checked_mul(ldc).and_then(|v| v.checked_add(n)).expect("sgemm: desbordamiento del tamaño de c");
    assert!(c_len >= c_req, "sgemm: el buffer c es demasiado pequeño: {} < {}", c_len, c_req);
}

#[inline]
fn handle_zero_k_sgemm(m: usize, n: usize, beta: f32, c: &mut [f32], ldc: usize) {
    if m > 0 && n > 0 && beta != 1.0 {
        let req = (m - 1)
            .checked_mul(ldc)
            .and_then(|v| v.checked_add(n))
            .expect("sgemm: desbordamiento del tamaño de c con k = 0");
        assert!(
            ldc >= n && c.len() >= req,
            "sgemm: límites de c no válidos con k = 0"
        );
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

/// Hilos con que se inicializó OpenBLAS: tope de `set_threads` (pedir más cuelga `sgemm`).
#[cfg(not(target_os = "macos"))]
fn initial_threads() -> usize {
    static INITIAL: OnceLock<usize> = OnceLock::new();
    *INITIAL.get_or_init(|| unsafe { openblas_get_num_threads() }.max(1) as usize)
}

/// Indica si el usuario fijó los hilos de OpenBLAS por entorno, con `OPENBLAS_NUM_THREADS`,
/// `GOTO_NUM_THREADS` u `OMP_NUM_THREADS`, las tres que OpenBLAS lee al arrancar.
///
/// Se evalúa la primera vez que se llama y se conserva. Debe llamarse antes de que el propio
/// binario toque el entorno (`_putenv`): después vería la variable que fija el binario y se
/// confundiría con una del usuario.
#[cfg(not(target_os = "macos"))]
fn user_fixed_threads() -> bool {
    static FIXED: OnceLock<bool> = OnceLock::new();
    *FIXED.get_or_init(|| {
        ["OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS", "OMP_NUM_THREADS"]
            .iter()
            .any(|v| std::env::var_os(v).is_some())
    })
}

/// Fija los hilos con que se inicializa OpenBLAS, si el usuario no los fijó por entorno.
///
/// En `x86_64-pc-windows-gnu` se enlaza el OpenBLAS propio, estático y sin OpenMP. Su
/// inicialización corre antes de `main` y reserva 128 MiB de memoria comprometida por cada
/// hilo. Por eso el binario fija `OPENBLAS_NUM_THREADS` antes, salvo que el usuario haya fijado
/// alguna de las tres variables de hilos (`OPENBLAS_NUM_THREADS`, `GOTO_NUM_THREADS` u
/// `OMP_NUM_THREADS`), que se respetan; sin ninguna, OpenBLAS usa un hilo por CPU lógica.
/// `openblas_set_num_threads` posterior no devuelve esa memoria, y pedir más hilos que los
/// iniciales cuelga la primera `sgemm`. Por eso el binario llama a esta función antes que los
/// constructores de OpenBLAS, con el número de hilos que va a usar.
///
/// `_putenv` actualiza la tabla de entorno de la CRT, que es la que lee `getenv`. En el resto
/// de plataformas no hace nada: Linux no cobra la memoria comprometida (*commit*) por adelantado y macOS usa Accelerate.
pub fn set_initial_openblas_threads(n: usize) {
    #[cfg(windows)]
    {
        // Invariante: `user_fixed_threads` se evalúa siempre antes de cualquier `_putenv`, que
        // actualiza también el entorno del proceso; evaluada después, vería la variable que fija
        // el propio binario.
        if user_fixed_threads() {
            return;
        }
        let Ok(var) = std::ffi::CString::new(format!("OPENBLAS_NUM_THREADS={}", n.max(1))) else {
            return;
        };
        // Si falla, OpenBLAS arranca con su valor por defecto: más memoria, mismo resultado.
        unsafe {
            _putenv(var.as_ptr());
        }
    }
    #[cfg(not(windows))]
    {
        let _ = n;
    }
}

/// Backend BLAS por defecto enlazado vía FFI a OpenBLAS (Linux/Windows) o Accelerate (macOS).
#[derive(Debug, Default, Clone, Copy)]
pub struct FfiBlas;

impl FfiBlas {
    pub fn name(&self) -> &'static str {
        if cfg!(target_os = "macos") {
            "Accelerate"
        } else {
            "OpenBLAS"
        }
    }

    /// Detalle de la implementación para `--caps`: en OpenBLAS, el kernel para esta CPU
    /// (`openblas_get_corename`), que con `DYNAMIC_ARCH` se elige al arrancar.
    pub fn detail(&self) -> Option<String> {
        #[cfg(not(target_os = "macos"))]
        {
            let p = unsafe { openblas_get_corename() };
            if p.is_null() {
                return None;
            }
            let core = unsafe { std::ffi::CStr::from_ptr(p) };
            Some(format!("núcleo {}", core.to_string_lossy()))
        }
        #[cfg(target_os = "macos")]
        {
            None
        }
    }

    /// Fija los hilos de OpenBLAS, sin pasar de los iniciales, salvo que el usuario los haya
    /// fijado por entorno. El número de hilos de OpenBLAS es global al proceso.
    pub fn set_threads(&self, n: usize) {
        #[cfg(not(target_os = "macos"))]
        {
            if user_fixed_threads() {
                return;
            }
            unsafe {
                openblas_set_num_threads(n.max(1).min(initial_threads()) as c_int);
            }
        }
        #[cfg(target_os = "macos")]
        {
            let _ = n;
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sgemm(
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
        );
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

#[cfg(all(test, not(target_os = "macos")))]
mod tests {
    use super::*;

    #[test]
    fn set_threads_acts_on_openblas() {
        if user_fixed_threads() {
            eprintln!("se omite: el usuario fijó los hilos de OpenBLAS por entorno");
            return;
        }
        // En un equipo de un solo hilo, la comprobación con n = 1 pasa de forma trivial.
        FfiBlas.set_threads(1);
        assert_eq!(unsafe { openblas_get_num_threads() }, 1, "set_threads(1) no fijó un hilo en OpenBLAS");
        // Restaura el estado global del proceso.
        FfiBlas.set_threads(initial_threads());
    }
}
