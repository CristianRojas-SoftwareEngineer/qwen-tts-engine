// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Servidor HTTP (`--serve`): un único trabajador que atiende una conexión a la vez.

pub mod http;
pub mod validation;

use std::net::{Ipv4Addr, SocketAddr, TcpListener};
use std::sync::Arc;

use qwen_engine::{Model06B, SessionScratch, Synthesizer, Voice};
use qwen_kernels::ComputeRuntime;

/// Contexto fijo del servidor: runtime, modelo y la voz cargada.
pub struct ServerContext {
    pub runtime: Arc<ComputeRuntime>,
    pub model: Arc<Model06B>,
    pub voice: Arc<Voice>,
}

impl ServerContext {
    /// El contexto fijo de síntesis de este servidor.
    pub fn synthesizer(&self) -> Synthesizer<'_> {
        Synthesizer {
            runtime: &self.runtime,
            model: &self.model,
            voice: &self.voice,
        }
    }
}

/// Escucha en `host:port`, calienta una vez y atiende las conexiones en línea (sin hilos ni cola).
pub fn serve(ctx: ServerContext, host: Ipv4Addr, port: u16) -> Result<(), String> {
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
    http::warm_up(&ctx.synthesizer(), &mut scratch);
    for incoming in listener.incoming() {
        match incoming {
            Ok(stream) => {
                let peer_ip = stream
                    .peer_addr()
                    .map(|a| a.ip().to_string())
                    .unwrap_or_else(|_| "0.0.0.0".to_string());
                http::handle_connection(&ctx, &mut scratch, stream, &peer_ip);
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
