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

    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let rc_path = manifest_dir.join("qwen_app_manifest.rc");
    let manifest_path = manifest_dir.join("qwen_app.exe.manifest");

    if target_env == "msvc" {
        println!("cargo:rustc-link-arg=/MANIFEST:EMBED");
        println!(
            "cargo:rustc-link-arg=/MANIFESTINPUT:{}",
            manifest_path.display()
        );
        return;
    }

    // GNU / MinGW: compila .rc a objeto COFF .o usando windres
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let res_obj = out_dir.join("qwen_app_manifest.o");

    let candidates = [
        env::var("WINDRES").ok(),
        Some("windres".to_string()),
        Some("x86_64-w64-mingw32-windres".to_string()),
        Some("llvm-windres".to_string()),
        Some(r"C:\msys64\ucrt64\bin\windres.exe".to_string()),
    ];

    for cand in candidates.into_iter().flatten() {
        let status = Command::new(&cand)
            .arg("-i")
            .arg(&rc_path)
            .arg("-o")
            .arg(&res_obj)
            .arg("-O")
            .arg("coff")
            .current_dir(&manifest_dir)
            .status();
        if let Ok(st) = status {
            if st.success() {
                println!("cargo:rustc-link-arg={}", res_obj.display());
                return;
            }
        }
    }

    println!(
        "cargo:warning=windres no encontrado; no se incrustó el manifiesto UTF-8 de Windows"
    );
}
