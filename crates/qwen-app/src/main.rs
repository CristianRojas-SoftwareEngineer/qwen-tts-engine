// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

mod affinity;
mod cli;
mod modes;
mod request;
mod server;

use std::process;

use cli::{parse_cli, print_usage, CliArgs, CliMode, ParseOutcome};
use modes::{diag, run_with_model, voice_cmd};

/// Despacha el modo ya resuelto por el parser.
fn run_app(mut cfg: CliArgs) -> Result<i32, String> {
    // Un entorno no válido es un error en cualquier modo.
    qwen_kernels::KernelChoice::from_env().map_err(|e| format!("Error: {e}"))?;

    if cfg.stdout_pcm {
        cfg.silent = true;
    }
    let physical_cores = affinity::pin_one_thread_per_core(cfg.silent);

    match &cfg.mode {
        CliMode::Caps => Ok(diag::caps(&cfg)),
        CliMode::SelfTest => Ok(diag::self_test(&cfg)),
        CliMode::ListVoices(dir) => Ok(voice_cmd::list_voices(dir)),
        CliMode::DeleteVoice(path) => Ok(voice_cmd::delete_voice(path)),
        CliMode::CreateVoice | CliMode::Serve(_) | CliMode::Synthesize | CliMode::NoJob => {
            run_with_model(cfg, physical_cores)
        }
    }
}

/// Fija los hilos iniciales de OpenBLAS con el `-j` efectivo antes de que se inicialice.
///
/// Corre en `.CRT$XCT`: dentro de `_initterm`, con la CRT ya iniciada y antes de los
/// constructores (`.ctors`) de OpenBLAS. Ver [`qwen_kernels::set_initial_openblas_threads`].
/// Si los argumentos no se pueden leer o no son válidos, no hace nada: `main` informa del error.
#[cfg(windows)]
mod openblas_preinit {
    extern "C" fn set_initial_threads() {
        let Ok(args) = std::env::args_os()
            .map(std::ffi::OsString::into_string)
            .collect::<Result<Vec<String>, _>>()
        else {
            return;
        };
        if let Ok(crate::cli::ParseOutcome::Run(cfg)) = crate::cli::parse_cli(&args) {
            qwen_kernels::set_initial_openblas_threads(qwen_kernels::resolve_num_threads(
                cfg.threads,
            ));
        }
    }

    #[used]
    #[link_section = ".CRT$XCT"]
    static SET_INITIAL_THREADS: extern "C" fn() = set_initial_threads;
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let prog = args.first().map(String::as_str).unwrap_or("qwen_tts");
    let cfg = match parse_cli(&args) {
        Ok(ParseOutcome::Help) => {
            print_usage(prog);
            process::exit(0);
        }
        Ok(ParseOutcome::Run(c)) => *c,
        Err(msg) => {
            eprintln!("{msg}");
            process::exit(1);
        }
    };

    match run_app(cfg) {
        Ok(code) => process::exit(code),
        Err(msg) => {
            eprintln!("{msg}");
            process::exit(1);
        }
    }
}
