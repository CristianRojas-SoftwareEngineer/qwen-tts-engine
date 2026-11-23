// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Modo `--serve`: servidor HTTP de un solo trabajador.

use qwen_engine::{Model06B, Voice};
use qwen_kernels::ComputeRuntime;

use crate::cli::CliArgs;
use crate::server::{serve, ServerContext};

pub fn run(
    cfg: &CliArgs,
    runtime: ComputeRuntime,
    model: Model06B,
    voice: Voice,
    port: u16,
) -> Result<i32, String> {
    serve(
        ServerContext {
            runtime,
            model,
            voice,
        },
        cfg.serve_host,
        port,
    )?;
    Ok(0)
}
