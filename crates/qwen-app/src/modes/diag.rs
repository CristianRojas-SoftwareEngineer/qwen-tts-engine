// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Modos de diagnóstico: `--caps` y `--self-test`.

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
