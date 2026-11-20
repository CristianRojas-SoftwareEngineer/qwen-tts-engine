// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::net::{Ipv4Addr, SocketAddr, TcpListener};
use std::sync::Arc;

use qwen_engine::{Model06B, SessionScratch, Voice};
use qwen_kernels::ComputeRuntime;

use super::http::{handle_connection, warm_up_session};

pub struct ServeConfig {
    pub runtime: Arc<ComputeRuntime>,
    pub model: Arc<Model06B>,
    pub voice: Arc<Voice>,
    pub host: Ipv4Addr,
    pub port: u16,
}

pub fn serve(cfg: ServeConfig) -> Result<(), String> {
    let ServeConfig {
        runtime,
        model,
        voice,
        host,
        port,
    } = cfg;
    let addr = SocketAddr::from((host, port));
    let listener = TcpListener::bind(addr)
        .map_err(|e| format!("Failed to bind http://{host}:{port}: {e}"))?;

    eprint!("Server listening on http://{host}:{port}");
    eprintln!("
Endpoints:");
    eprintln!("  POST /v1/tts          — generate speech (returns WAV)");
    eprintln!("  POST /v1/tts/stream   — generate speech (chunked PCM stream)");
    eprintln!("  POST /v1/audio/speech — OpenAI-compatible TTS");
    eprintln!("  GET  /v1/voices       — list the loaded voice");
    eprintln!("  GET  /v1/health       — health check
");
    eprintln!("Press Ctrl+C to stop.
");

    let mut scratch = SessionScratch::new();
    warm_up_session(&runtime, &model, &voice, &mut scratch);
    for incoming in listener.incoming() {
        match incoming {
            Ok(stream) => {
                let peer_ip = stream
                    .peer_addr()
                    .map(|a| a.ip().to_string())
                    .unwrap_or_else(|_| "0.0.0.0".to_string());
                handle_connection(
                    &runtime,
                    &model,
                    &voice,
                    &mut scratch,
                    stream,
                    &peer_ip,
                );
            }
            Err(e) => {
                if e.kind() == std::io::ErrorKind::Interrupted {
                    continue;
                }
                eprintln!("accept error: {e}");
            }
        }
    }
    Ok(())
}
