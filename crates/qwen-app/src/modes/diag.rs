// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Modos de diagnóstico: `--caps`, `--self-test`, `--matmat-bench` y `decode-dump`.

use std::fs;
use std::path::Path;

use qwen_core::config::SAMPLE_RATE;
use qwen_engine::{write_wav, Model06B, ModelLoadOptions};
use qwen_kernels::bench::run_matmat_bench;
use qwen_kernels::selftest::run_kernel_selftest;
use qwen_kernels::ComputeRuntime;

use crate::cli::CliArgs;

/// `--caps`: informe de capacidades del procesador.
pub fn caps(cfg: &CliArgs) -> i32 {
    let rt = ComputeRuntime::new(cfg.threads);
    print!("{}", rt.caps_report());
    0
}

/// `--self-test`: autoprueba de los kernels; `1` si algún caso falla.
pub fn self_test(cfg: &CliArgs) -> i32 {
    let rt = ComputeRuntime::new(cfg.threads);
    let (failures, report) = run_kernel_selftest(&rt);
    print!("{report}");
    if failures == 0 {
        0
    } else {
        1
    }
}

/// `--matmat-bench`: microbanco de los kernels `matmat`.
pub fn matmat_bench(cfg: &CliArgs) -> i32 {
    let rt = ComputeRuntime::new(cfg.threads);
    print!("{}", run_matmat_bench(&rt));
    0
}

/// `decode-dump <codes.txt> <model_dir> <out.wav>`: decodifica códigos a WAV.
pub fn decode_dump(codes_path: &str, model_dir: &str, out_wav: &str) -> Result<i32, String> {
    let raw = fs::read_to_string(codes_path)
        .map_err(|e| format!("No se pudo leer el archivo de códigos {codes_path}: {e}"))?;
    let mut codes = Vec::new();
    for tok in raw.split_whitespace() {
        let v = tok
            .parse::<i32>()
            .map_err(|_| format!("Token entero inválido '{tok}' en {codes_path}"))?;
        codes.push(v);
    }
    if codes.is_empty() || codes.len() % 16 != 0 {
        return Err(format!(
            "Cantidad de códigos inválida {} en {codes_path} (se esperaba un múltiplo de 16)",
            codes.len()
        ));
    }
    let n_frames = codes.len() / 16;
    let rt = ComputeRuntime::new(None);
    let opts = ModelLoadOptions {
        silent: true,
        ..Default::default()
    };
    let model = Model06B::load(Path::new(model_dir), &rt, opts).map_err(|e| e.to_string())?;
    let audio = qwen_engine::speech_decoder_decode(&rt, &model.decoder, &codes, n_frames)
        .map_err(|e| e.to_string())?;
    write_wav(Path::new(out_wav), &audio, SAMPLE_RATE as u32).map_err(|e| e.to_string())?;
    eprintln!(
        "decode-dump: {n_frames} frames -> {} muestras -> {out_wav}",
        audio.len()
    );
    Ok(0)
}
