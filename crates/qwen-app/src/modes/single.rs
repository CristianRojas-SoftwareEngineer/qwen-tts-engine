// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Modo de una síntesis desde la línea de órdenes: WAV completo o PCM por --stdout.

use std::path::Path;

use qwen_core::config::SAMPLE_RATE;
use qwen_engine::{
    write_wav, AbortHandle, Delivery, EdgeRampSink, SessionScratch, StdoutWriter, Synthesizer,
};

use crate::cli::CliArgs;
use crate::request;

fn write_output_wav(cfg: &CliArgs, audio: &[f32]) -> Result<(), String> {
    write_wav(Path::new(&cfg.output), audio, SAMPLE_RATE as u32).map_err(|e| e.to_string())?;
    if !cfg.silent {
        eprintln!(
            "Escrito {} ({} muestras, {:.2}s)",
            cfg.output,
            audio.len(),
            audio.len() as f64 / SAMPLE_RATE as f64
        );
    }
    Ok(())
}

/// Una síntesis desde la CLI con la voz del contexto `synth`.
pub fn run(cfg: &CliArgs, synth: &Synthesizer<'_>) -> Result<i32, String> {
    let (req, script) = request::from_cli(cfg, synth.voice);
    let abort = AbortHandle::new();
    let mut scratch = SessionScratch::new();

    // PCM crudo por stdout (--stdout)
    if cfg.stdout_pcm {
        let mut sink = EdgeRampSink::new(StdoutWriter, SAMPLE_RATE as u32);
        synth
            .synthesize(
                &mut scratch,
                &req,
                &abort,
                &script,
                Delivery::Stream(&mut sink),
            )
            .map_err(|e| e.to_string())?;
        return Ok(0);
    }

    // Entrega completa
    let audio = synth
        .synthesize(&mut scratch, &req, &abort, &script, Delivery::Buffer)
        .map_err(|e| e.to_string())?
        .samples
        .unwrap_or_default();

    write_output_wav(cfg, &audio)?;
    Ok(0)
}
