// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=qwen_app.exe.manifest");
    println!("cargo:rerun-if-changed=qwen_app_manifest.rc");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "windows" {
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let rc_path = manifest_dir.join("qwen_app_manifest.rc");
    let res_obj = PathBuf::from(env::var("OUT_DIR").unwrap()).join("qwen_app_manifest.o");

    // El manifiesto fija UTF-8 como página de códigos del proceso: sin él, los argumentos y
    // las rutas con tildes llegan mal. `windres` viene con binutils de MSYS2 UCRT64, en la
    // carpeta de su gcc, y preprocesa el `.rc` con `gcc` buscado en el PATH. Si Cargo tiene un
    // enlazador configurado (`RUSTC_LINKER`), se usa el `windres.exe` de su carpeta y esa
    // carpeta va primera en el PATH de `windres`; si no, ambos salen del PATH.
    let mut windres = Command::new("windres");
    if let Ok(linker) = env::var("RUSTC_LINKER") {
        let linker_dir = PathBuf::from(linker).parent().unwrap().to_path_buf();
        let path = env::var_os("PATH").unwrap_or_default();
        let dirs = std::iter::once(linker_dir.clone()).chain(env::split_paths(&path));
        let path = env::join_paths(dirs).unwrap();
        windres = Command::new(linker_dir.join("windres.exe"));
        windres.env("PATH", path);
    }
    let ok = windres
        .arg("-i")
        .arg(&rc_path)
        .arg("-o")
        .arg(&res_obj)
        .arg("-O")
        .arg("coff")
        .current_dir(&manifest_dir)
        .status()
        .is_ok_and(|st| st.success());
    if !ok {
        panic!(
            "no se pudo compilar el manifiesto de Windows con `{}`: instala \
             mingw-w64-ucrt-x86_64-binutils y pon C:\\msys64\\ucrt64\\bin en el PATH",
            windres.get_program().to_string_lossy()
        );
    }
    println!("cargo:rustc-link-arg={}", res_obj.display());
}
