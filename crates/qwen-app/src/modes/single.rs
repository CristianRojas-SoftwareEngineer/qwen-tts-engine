// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Modo de una síntesis desde la línea de órdenes: WAV completo, `--stream`, `--stdout` y
//! `--seed-audition`.

use std::path::Path;

use qwen_core::config::{FRAME_RATE, SAMPLE_RATE};
use qwen_core::Script;
use qwen_engine::{
    finish_full_delivery, tail_glitch_score, write_wav, AbortHandle, Delivery, EdgeRampSink,
    SessionScratch, StdoutWriter, SynthesisRequest, Synthesizer, WavFileWriter,
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

fn run_seed_audition(
    cfg: &CliArgs,
    synth: &Synthesizer<'_>,
    scratch: &mut SessionScratch,
    req: &SynthesisRequest,
    script: &Script,
) -> Result<Vec<f32>, String> {
    let n_takes = cfg.seed_audition.min(64);
    let mut candidates: Vec<(Vec<f32>, u32, f32)> = Vec::with_capacity(n_takes);
    let abort = AbortHandle::new();
    for i in 0..n_takes {
        // Las tomas se sintetizan sin `tail_trim` ni `onset_fade`: el acabado va solo a la elegida.
        let mut ccfg = req.clone();
        ccfg.sampling.seed = req.sampling.seed.wrapping_add(i as u32);
        ccfg.tail_trim = false;
        ccfg.onset_fade_ms = 0.0;
        if let Ok(out) = synth.synthesize(scratch, &ccfg, &abort, script, Delivery::Buffer) {
            let a = out.samples.unwrap_or_default();
            if !a.is_empty() {
                let (g, _) = tail_glitch_score(&a, SAMPLE_RATE);
                if !cfg.silent {
                    eprintln!(
                        "  audición semilla {}: {:.2}s glitch={g:.2}",
                        ccfg.sampling.seed,
                        a.len() as f32 / SAMPLE_RATE as f32
                    );
                }
                if cfg.audition_keep {
                    let out_p = Path::new(&cfg.output);
                    let stem = out_p
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("output");
                    let keep_name = match out_p.parent() {
                        Some(parent) if !parent.as_os_str().is_empty() => {
                            parent.join(format!("{stem}.seed{}.wav", ccfg.sampling.seed))
                        }
                        _ => Path::new(&format!("{stem}.seed{}.wav", ccfg.sampling.seed))
                            .to_path_buf(),
                    };
                    if write_wav(&keep_name, &a, SAMPLE_RATE as u32).is_ok() && !cfg.silent {
                        eprintln!("    conservada -> {}", keep_name.display());
                    }
                }
                candidates.push((a, ccfg.sampling.seed, g));
            }
        }
    }
    if candidates.is_empty() {
        return Err("La generación falló".to_string());
    }
    let mut lengths: Vec<usize> = candidates.iter().map(|(a, _, _)| a.len()).collect();
    lengths.sort_unstable();
    let med_n = lengths[lengths.len() / 2];
    let mut best_idx = 0usize;
    let mut best_cost = f32::MAX;
    let mut n_rej = 0usize;
    for (idx, (a, _, g)) in candidates.iter().enumerate() {
        let r = if med_n > 0 {
            a.len() as f32 / med_n as f32
        } else {
            1.0
        };
        let dd = (r - 1.0).abs();
        let bad = !(0.55..=2.50).contains(&r) || *g >= 0.5;
        if bad {
            n_rej += 1;
        }
        let cost = g * 10.0 + dd + if bad { 100.0 } else { 0.0 };
        if cost < best_cost {
            best_cost = cost;
            best_idx = idx;
        }
    }
    let best_seed = candidates[best_idx].1;
    let best_g = candidates[best_idx].2;
    let best_len = candidates[best_idx].0.len();
    if !cfg.silent {
        eprintln!(
            "  audición -> elegida semilla {best_seed} (glitch={best_g:.2}, {:.2}s de {} tomas; {n_rej} rechazadas por defectuosas)",
            best_len as f32 / SAMPLE_RATE as f32,
            candidates.len()
        );
    }
    Ok(candidates.swap_remove(best_idx).0)
}

/// Una síntesis desde la CLI con la voz del contexto `synth`.
pub fn run(cfg: &CliArgs, synth: &Synthesizer<'_>) -> Result<i32, String> {
    let (req, planned) = request::from_cli(cfg, synth.voice);
    let abort = AbortHandle::new();
    let mut scratch = SessionScratch::new();
    let script = planned.script;

    // Entrega en streaming (--stream / --stdout)
    if cfg.stream {
        if !cfg.silent {
            eprintln!(
                "Streaming: chunk={} frames ({:.1}s), {}",
                req.chunk_frames,
                req.chunk_frames as f32 / FRAME_RATE,
                if cfg.stdout_pcm {
                    "PCM crudo a stdout"
                } else {
                    &cfg.output
                }
            );
        }
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
        } else {
            let writer = WavFileWriter::create(Path::new(&cfg.output), SAMPLE_RATE as u32)
                .map_err(|e| e.to_string())?;
            let mut sink = EdgeRampSink::new(writer, SAMPLE_RATE as u32);
            let out = synth
                .synthesize(
                    &mut scratch,
                    &req,
                    &abort,
                    &script,
                    Delivery::Stream(&mut sink),
                )
                .map_err(|e| e.to_string())?;
            if !cfg.silent {
                eprintln!(
                    "Escrito {} ({} muestras, {:.2}s) [por streaming]",
                    cfg.output,
                    out.timing.samples_produced,
                    out.timing.samples_produced as f64 / SAMPLE_RATE as f64
                );
            }
        }
        return Ok(0);
    }

    // Entrega completa (toma única o --seed-audition N)
    if cfg.audition_keep && cfg.seed_audition <= 1 && !cfg.silent {
        eprintln!(
            "Aviso: --audition-keep no tiene efecto sin --seed-audition N (N>1, sin streaming)"
        );
    }

    let audio = if cfg.seed_audition > 1 {
        let mut chosen = run_seed_audition(cfg, synth, &mut scratch, &req, &script)?;
        // Las tomas no llevan acabado: `tail_trim` y `onset_fade` van solo a la elegida.
        finish_full_delivery(&mut chosen, &req);
        chosen
    } else {
        // `synthesize` ya aplica el acabado tras `time_stretch`.
        synth
            .synthesize(&mut scratch, &req, &abort, &script, Delivery::Buffer)
            .map_err(|e| e.to_string())?
            .samples
            .unwrap_or_default()
    };

    write_output_wav(cfg, &audio)?;
    Ok(0)
}
