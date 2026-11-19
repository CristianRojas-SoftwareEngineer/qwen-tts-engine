// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::env;
use std::path::PathBuf;

fn main() {
    let target = env::var("TARGET").unwrap_or_default();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    println!("cargo:rerun-if-env-changed=OPENBLAS_LIB_DIR");

    if target_os == "macos" {
        println!("cargo:rustc-link-lib=framework=Accelerate");
        return;
    }

    if target_os == "windows" {
        let lib_dir = env::var("OPENBLAS_LIB_DIR")
            .unwrap_or_else(|_| "C:/msys64/ucrt64/lib".to_string());
        println!("cargo:rustc-link-search=native={lib_dir}");

        if target.contains("gnu") {
            // OpenBLAS estático + OpenMP + runtime de gfortran (MSYS2 UCRT64)
            println!("cargo:rustc-link-lib=static=openblas");
            println!("cargo:rustc-link-lib=static=gomp");
            println!("cargo:rustc-link-lib=static=gfortran");
            println!("cargo:rustc-link-lib=static=quadmath");
            println!("cargo:rustc-link-lib=static=winpthread");
            println!("cargo:rustc-link-lib=ws2_32");
        } else {
            println!("cargo:rustc-link-lib=openblas");
        }
        return;
    }

    // Linux / Unix: con OPENBLAS_LIB_DIR y libopenblas.a, enlace estático; si no, la libopenblas
    // dinámica del sistema. Las bibliotecas de Fortran solo hacen falta si ese OpenBLAS se compiló
    // con LAPACK en Fortran: se enlazan si están en el mismo directorio.
    if let Ok(lib_dir) = env::var("OPENBLAS_LIB_DIR") {
        println!("cargo:rustc-link-search=native={lib_dir}");
        let dir = PathBuf::from(&lib_dir);
        if dir.join("libopenblas.a").exists() {
            println!("cargo:rustc-link-lib=static=openblas");
            for lib in ["gfortran", "quadmath"] {
                if dir.join(format!("lib{lib}.a")).exists() {
                    println!("cargo:rustc-link-lib=static={lib}");
                }
            }
            return;
        }
    }
    println!("cargo:rustc-link-lib=openblas");
}
