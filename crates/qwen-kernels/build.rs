// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::env;
use std::path::Path;

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    println!("cargo:rerun-if-env-changed=OPENBLAS_LIB_DIR");

    if target_os == "macos" {
        println!("cargo:rustc-link-lib=framework=Accelerate");
        return;
    }

    // Linux y Windows enlazan siempre el OpenBLAS propio estático de `scripts/build-openblas.sh`
    // (sin OpenMP ni Fortran), en el directorio que indica `OPENBLAS_LIB_DIR`. Esa es la única
    // ruta de búsqueda: así ningún otro `libopenblas` instalado en el equipo entra en el enlace.
    let windows = target_os == "windows";
    let build_hint = if windows {
        "compílalo con `scripts/build-openblas.sh /c/openblas` en el bash de MSYS2 UCRT64 y fija OPENBLAS_LIB_DIR=C:/openblas/lib"
    } else {
        "compílalo con `scripts/build-openblas.sh \"$HOME/openblas\"` y fija OPENBLAS_LIB_DIR=$HOME/openblas/lib"
    };
    let Ok(lib_dir) = env::var("OPENBLAS_LIB_DIR") else {
        panic!("el motor enlaza un OpenBLAS propio y falta OPENBLAS_LIB_DIR: {build_hint}");
    };
    if !Path::new(&lib_dir).join("libopenblas.a").exists() {
        panic!(
            "el motor enlaza un OpenBLAS propio y falta libopenblas.a en `{lib_dir}`: {build_hint}"
        );
    }
    println!("cargo:rustc-link-search=native={lib_dir}");
    println!("cargo:rustc-link-lib=static=openblas");
    if windows {
        // OpenBLAS usa símbolos `pthread_*` en MinGW. `libwinpthread.a` es del toolchain de MSYS2
        // UCRT64 y la resuelve gcc al enlazar, por eso va con `-bundle` y sin ruta de búsqueda
        // propia. Debe ir después de `openblas`, que usa sus símbolos.
        println!("cargo:rustc-link-lib=static:-bundle=winpthread");
        println!("cargo:rustc-link-lib=ws2_32");
    }
}
