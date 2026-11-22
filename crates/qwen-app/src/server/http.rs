// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::io::{ErrorKind, Read, Write};
use std::net::TcpStream;
use std::thread;
use std::time::{Duration, Instant};

use qwen_core::config::{effective_language, language_name, SAMPLE_RATE};
use qwen_core::{PlannedScript, SamplingParams, Script};
use qwen_engine::{
    encode_wav_bytes, AbortHandle, Delivery, EdgeRampSink, EngineError, LogLevel, MemorySink,
    PcmWrite, SessionScratch, SynthesisOutput, SynthesisRequest, SynthesisTiming, Synthesizer,
    Voice,
};

use super::validation::ValidatedTtsRequest;
use super::ServerContext;
use crate::request::from_http;

pub const MAX_HTTP_BODY: usize = 1024 * 1024;

pub struct AbortGuard {
    pub abort: AbortHandle,
}

impl AbortGuard {
    pub fn for_socket(stream: &TcpStream) -> Result<Self, std::io::Error> {
        let cloned = stream.try_clone()?;
        let abort = AbortHandle::with_poll(move || socket_disconnected(&cloned));
        Ok(Self { abort })
    }
}

impl Drop for AbortGuard {
    fn drop(&mut self) {
        self.abort.cancel();
    }
}

pub fn socket_disconnected(stream: &TcpStream) -> bool {
    let mut buf = [0u8; 1];
    match stream.peek(&mut buf) {
        Err(e) if e.kind() == ErrorKind::WouldBlock || e.kind() == ErrorKind::Interrupted => false,
        Ok(0) => true,
        Ok(_) => false,
        Err(_) => true,
    }
}

pub fn send_all(stream: &mut TcpStream, mut data: &[u8]) -> std::io::Result<()> {
    let start = Instant::now();
    while !data.is_empty() {
        match stream.write(data) {
            Ok(0) => return Err(std::io::Error::from(ErrorKind::WriteZero)),
            Ok(n) => data = &data[n..],
            Err(e) if e.kind() == ErrorKind::WouldBlock || e.kind() == ErrorKind::Interrupted => {
                if start.elapsed() > Duration::from_secs(30) {
                    return Err(std::io::Error::from(ErrorKind::TimedOut));
                }
                thread::sleep(Duration::from_millis(1));
            }
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

fn find_header_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n").map(|p| p + 4)
}

fn parse_content_length(headers: &str) -> Result<usize, ()> {
    for line in headers.split("\r\n") {
        if let Some((k, v)) = line.split_once(':') {
            if k.trim().eq_ignore_ascii_case("content-length") {
                return v.trim().parse::<usize>().map_err(|_| ());
            }
        }
    }
    Ok(0)
}

pub struct ParsedHttpRequest {
    pub method: String,
    pub path: String,
    pub body: String,
}

pub enum ReadHttpResult {
    Ok(ParsedHttpRequest),
    TooLarge,
    InvalidContentLength,
    Empty,
}

pub fn read_http_request(stream: &mut TcpStream) -> ReadHttpResult {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(30)));
    let mut buf: Vec<u8> = Vec::with_capacity(4096);
    let mut tmp = [0u8; 8192];
    let mut header_end: Option<usize> = None;
    let mut content_length: usize = 0;

    loop {
        if buf.len() >= MAX_HTTP_BODY {
            return ReadHttpResult::TooLarge;
        }
        let max_read = (MAX_HTTP_BODY - buf.len()).min(tmp.len());
        match stream.read(&mut tmp[..max_read]) {
            Ok(0) => break,
            Ok(n) => {
                buf.extend_from_slice(&tmp[..n]);
                if header_end.is_none() {
                    if let Some(hend) = find_header_end(&buf) {
                        header_end = Some(hend);
                        let hdr_str = String::from_utf8_lossy(&buf[..hend]);
                        match parse_content_length(&hdr_str) {
                            Ok(len) => {
                                if len > MAX_HTTP_BODY - hend {
                                    return ReadHttpResult::TooLarge;
                                }
                                content_length = len;
                            }
                            Err(()) => return ReadHttpResult::InvalidContentLength,
                        }
                    }
                }
                if let Some(hend) = header_end {
                    if buf.len() - hend >= content_length {
                        break;
                    }
                }
            }
            Err(e) if e.kind() == ErrorKind::Interrupted => continue,
            Err(_) => break,
        }
    }

    if buf.is_empty() {
        return ReadHttpResult::Empty;
    }

    let hend = header_end.unwrap_or(buf.len());
    let head = String::from_utf8_lossy(&buf[..hend]);
    let first_line = head.lines().next().unwrap_or("");
    let mut parts = first_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let raw_path = parts.next().unwrap_or("");
    let path = raw_path.split('?').next().unwrap_or(raw_path).to_string();

    let body_end = if header_end.is_some() {
        (hend + content_length).min(buf.len())
    } else {
        buf.len()
    };
    let body = String::from_utf8_lossy(&buf[hend..body_end]).to_string();

    // Cambiar a modo no bloqueante para que AbortGuard sondee sin bloquear; send_all gestiona WouldBlock.
    let _ = stream.set_nonblocking(true);

    ReadHttpResult::Ok(ParsedHttpRequest { method, path, body })
}

pub fn log_http_request(req: &ParsedHttpRequest, peer_ip: &str) {
    eprintln!(
        "[HTTP] {} {} {} desde {}",
        req.method,
        req.path,
        if req.method == "POST" && !req.body.is_empty() {
            "(con cuerpo)"
        } else {
            ""
        },
        peer_ip
    );
}

pub fn send_response(
    stream: &mut TcpStream,
    status: u16,
    content_type: &str,
    body: &[u8],
) -> std::io::Result<()> {
    let status_text = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        503 => "Service Unavailable",
        _ => "Internal Server Error",
    };
    let header = format!(
        "HTTP/1.1 {status} {status_text}\r\n\
         Content-Type: {content_type}\r\n\
         Content-Length: {}\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Connection: close\r\n\r\n",
        body.len()
    );
    send_all(stream, header.as_bytes())?;
    if !body.is_empty() {
        send_all(stream, body)?;
    }
    Ok(())
}

pub fn send_json(stream: &mut TcpStream, status: u16, json: &str) {
    let _ = send_response(stream, status, "application/json", json.as_bytes());
}

pub fn send_error(stream: &mut TcpStream, status: u16, msg: &str) {
    let escaped = serde_json::to_string(msg).unwrap_or_else(|_| "\"error\"".to_string());
    let json = format!("{{\"error\":{escaped}}}");
    send_json(stream, status, &json);
}

pub fn send_cors_options(stream: &mut TcpStream) {
    let cors = b"HTTP/1.1 204 No Content\r\n\
        Access-Control-Allow-Origin: *\r\n\
        Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n\
        Access-Control-Allow-Headers: Content-Type\r\n\
        Connection: close\r\n\r\n";
    let _ = send_all(stream, cors);
}

pub fn send_chunked_header(stream: &mut TcpStream) -> std::io::Result<()> {
    let hdr = b"HTTP/1.1 200 OK\r\n\
        Content-Type: audio/pcm\r\n\
        X-Sample-Rate: 24000\r\n\
        X-Sample-Format: s16le\r\n\
        X-Channels: 1\r\n\
        Transfer-Encoding: chunked\r\n\
        Access-Control-Allow-Origin: *\r\n\
        Connection: close\r\n\r\n";
    send_all(stream, hdr)
}

/// Envía un chunk de la respuesta *chunked* con PCM s16le ya procesado (ganancia y rampas de borde aplicadas
/// por `EdgeRampSink`). Un chunk vacío no se envía: cerraría la respuesta.
pub fn send_pcm_chunk(stream: &mut TcpStream, pcm: &[u8]) -> std::io::Result<()> {
    if pcm.is_empty() {
        return Ok(());
    }
    let ch_hdr = format!("{:x}\r\n", pcm.len());
    send_all(stream, ch_hdr.as_bytes())?;
    send_all(stream, pcm)?;
    send_all(stream, b"\r\n")?;
    Ok(())
}

pub fn send_chunked_end(stream: &mut TcpStream) -> std::io::Result<()> {
    send_all(stream, b"0\r\n\r\n")
}

pub fn handle_health(stream: &mut TcpStream) {
    send_json(stream, 200, "{\"status\":\"ok\"}");
}

/// `GET /v1/voices`: voz cargada con su idioma.
pub fn handle_voices(stream: &mut TcpStream, voice: &Voice) {
    let language = voice
        .language_id
        .and_then(language_name)
        .map(serde_json::Value::from)
        .unwrap_or(serde_json::Value::Null);
    let body = serde_json::json!({
        "voices": [{ "name": voice.name, "language": language }]
    });
    send_json(stream, 200, &body.to_string());
}

pub fn log_wav_sent(bytes: usize, samples_len: usize, elapsed_s: f64) {
    let audio_secs = samples_len as f64 / SAMPLE_RATE as f64;
    eprintln!(
        "[HTTP] Enviados {} bytes de WAV ({:.2}s de audio) en {:.1}s (RTF {:.2})",
        bytes,
        audio_secs,
        elapsed_s,
        elapsed_s / audio_secs.max(1e-6)
    );
}

pub fn log_stream_sent(samples_len: usize, elapsed_s: f64) {
    let audio_secs = samples_len as f64 / SAMPLE_RATE as f64;
    eprintln!(
        "[HTTP] Transmitidas {} muestras ({:.2}s de audio) en {:.1}s (RTF {:.2})",
        samples_len,
        audio_secs,
        elapsed_s,
        elapsed_s / audio_secs.max(1e-6)
    );
}

pub fn log_timing_breakdown(t: &SynthesisTiming) {
    if t.frames == 0 {
        return;
    }
    let f = t.frames as f64;
    eprintln!(
        "[HTTP] Desglose: prefill {:.0} ms, Talker {:.0} ms ({:.1} ms/f), CP {:.0} ms ({:.1} ms/f), decoder {:.0} ms, drenaje {:.0} ms, {} frames en {} chunks, cola máxima {} frames",
        t.prefill_ms,
        t.talker_ms,
        t.talker_ms / f,
        t.cp_ms,
        t.cp_ms / f,
        t.decoder_ms,
        t.drain_ms,
        t.frames,
        t.chunks,
        t.max_backlog_frames
    );
}

/// Escritor HTTP *chunked* de PCM s16le para `EdgeRampSink`. Un fallo de envío (cliente
/// desconectado) se señala como `ConnectionAborted`.
pub struct HttpChunkWriter<'a> {
    stream: &'a mut TcpStream,
}

impl<'a> HttpChunkWriter<'a> {
    pub fn new(stream: &'a mut TcpStream) -> Self {
        Self { stream }
    }
}

impl PcmWrite for HttpChunkWriter<'_> {
    fn write_pcm(&mut self, pcm: &[u8]) -> std::io::Result<()> {
        send_pcm_chunk(self.stream, pcm)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::ConnectionAborted, e.to_string()))
    }

    fn finish(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Frames de la síntesis de calentamiento: el primer chunk (2) y uno completo (10).
const WARM_UP_FRAMES: usize = 12;

/// Síntesis corta y descartada al arrancar el servidor (único calentamiento).
///
/// Deja `scratch` (Talker, Code Predictor y decoder) dimensionado y en caliente, para que la
/// primera petición no pague el arranque en frío (~60–80 ms de TTFA). No cambia el audio de
/// las peticiones: cada una reinicia su estado y su semilla.
pub fn warm_up(synth: &Synthesizer<'_>, scratch: &mut SessionScratch) {
    let req = SynthesisRequest {
        language_id: effective_language(None, synth.voice.language_id),
        sampling: SamplingParams {
            seed: 1,
            ..SamplingParams::default()
        },
        max_frames: WARM_UP_FRAMES,
        log: LogLevel::Silent,
        ..SynthesisRequest::default()
    };
    let abort = AbortHandle::new();
    let mut sink = MemorySink::new(0);
    let t0 = Instant::now();
    match synth.synthesize(
        scratch,
        &req,
        &abort,
        &Script::Plain("Hello.".to_string()),
        Delivery::Stream(&mut sink),
    ) {
        Ok(_) => eprintln!(
            "[HTTP] Calentamiento: {:.0} ms",
            t0.elapsed().as_secs_f64() * 1000.0
        ),
        Err(e) => eprintln!("[HTTP] Calentamiento fallido (sin efecto en las peticiones): {e}"),
    }
}

/// Prepara la petición y registra la línea `[HTTP] TTS…` (`label` = `TTS` o `TTS stream`).
fn prepare(
    ctx: &ServerContext,
    vreq: &ValidatedTtsRequest,
    label: &str,
) -> (SynthesisRequest, PlannedScript) {
    let (req, planned) = from_http(vreq, &ctx.voice);
    eprintln!(
        "[HTTP] {label}: \"{}\" (voz={}, idioma={}, semilla={})",
        planned.text, ctx.voice.name, req.language_id, req.sampling.seed
    );
    (req, planned)
}

/// Guarda de aborto ligada al socket; responde 500 si no se puede crear.
fn abort_guard(stream: &mut TcpStream) -> Option<AbortGuard> {
    match AbortGuard::for_socket(stream) {
        Ok(g) => Some(g),
        Err(_) => {
            send_error(stream, 500, "error interno del socket");
            None
        }
    }
}

/// Única llamada a `synthesize` del servidor. `markup_label` es el texto del log de marcado en
/// línea (`compose` o `compose stream`).
fn run_synthesis(
    ctx: &ServerContext,
    scratch: &mut SessionScratch,
    req: &SynthesisRequest,
    script: &Script,
    abort: &AbortHandle,
    delivery: Delivery<'_>,
    markup_label: &str,
) -> Result<SynthesisOutput, EngineError> {
    if let Script::Spans(spans) = script {
        if spans.is_empty() {
            return Err(EngineError::InvalidArg(
                "falló el análisis del marcado".to_string(),
            ));
        }
        eprintln!(
            "[HTTP] marcado en línea -> {markup_label} por frase ({} tramos)",
            spans.len()
        );
    }
    ctx.synthesizer()
        .synthesize(scratch, req, abort, script, delivery)
}

/// Desglose de tiempos: solo para texto plano (la composición no lo produce).
fn plain_timing(script: &Script, out: &SynthesisOutput) -> Option<SynthesisTiming> {
    matches!(script, Script::Plain(_)).then(|| out.timing.clone())
}

fn handle_tts_full(
    ctx: &ServerContext,
    scratch: &mut SessionScratch,
    stream: &mut TcpStream,
    vreq: &ValidatedTtsRequest,
) {
    let (req, planned) = prepare(ctx, vreq, "TTS");
    let Some(guard) = abort_guard(stream) else {
        return;
    };

    let t0 = Instant::now();
    let res = run_synthesis(
        ctx,
        scratch,
        &req,
        &planned.script,
        &guard.abort,
        Delivery::Buffer,
        "compose",
    );
    let timing_opt = res
        .as_ref()
        .ok()
        .and_then(|o| plain_timing(&planned.script, o));

    let samples = match res.map(|o| o.samples.unwrap_or_default()) {
        Ok(s) if !s.is_empty() => s,
        Err(EngineError::Aborted) => {
            eprintln!(
                "[HTTP] TTS cancelado: el cliente se desconectó tras {:.1} s",
                t0.elapsed().as_secs_f64()
            );
            return;
        }
        Err(_) | Ok(_) => {
            if socket_disconnected(stream) {
                eprintln!(
                    "[HTTP] TTS cancelado: el cliente se desconectó tras {:.1} s",
                    t0.elapsed().as_secs_f64()
                );
            } else {
                send_error(stream, 500, "la generación falló");
            }
            return;
        }
    };

    let wav = encode_wav_bytes(&samples, SAMPLE_RATE as u32);
    if send_response(stream, 200, "audio/wav", &wav).is_ok() {
        log_wav_sent(wav.len(), samples.len(), t0.elapsed().as_secs_f64());
        if let Some(ref t) = timing_opt {
            log_timing_breakdown(t);
        }
    }
}

fn handle_tts_stream(
    ctx: &ServerContext,
    scratch: &mut SessionScratch,
    stream: &mut TcpStream,
    vreq: &ValidatedTtsRequest,
) {
    let (req, planned) = prepare(ctx, vreq, "TTS stream");
    let Some(guard) = abort_guard(stream) else {
        return;
    };

    if send_chunked_header(stream).is_err() {
        return;
    }

    let t0 = Instant::now();
    let mut chunk_sink = EdgeRampSink::new(HttpChunkWriter::new(stream), SAMPLE_RATE as u32);
    let res = run_synthesis(
        ctx,
        scratch,
        &req,
        &planned.script,
        &guard.abort,
        Delivery::Stream(&mut chunk_sink),
        "compose stream",
    );
    let timing_opt = res
        .as_ref()
        .ok()
        .and_then(|o| plain_timing(&planned.script, o));
    let gen_res = res.map(|_| ());

    let total_samples = chunk_sink.total_samples();
    if gen_res.is_err() || total_samples == 0 {
        if socket_disconnected(stream)
            || matches!(gen_res, Err(EngineError::Aborted))
            || matches!(&gen_res, Err(EngineError::Io(e)) if e.kind() == std::io::ErrorKind::ConnectionAborted)
        {
            eprintln!(
                "[HTTP] TTS stream cancelado: el cliente se desconectó tras {:.1} s",
                t0.elapsed().as_secs_f64()
            );
        } else {
            eprintln!("[HTTP] TTS stream: la generación falló; la respuesta queda sin cierre");
        }
        // Omitir `0\r\n\r\n` intencionadamente para que el cliente detecte transferencia truncada.
        return;
    }

    if send_chunked_end(stream).is_ok() {
        log_stream_sent(total_samples, t0.elapsed().as_secs_f64());
        if let Some(ref t) = timing_opt {
            log_timing_breakdown(t);
        }
    }
}

/// Único manejo de rutas: `/v1/health`, `/v1/voices`, `/v1/tts` y `/v1/tts/stream`.
pub fn handle_connection(
    ctx: &ServerContext,
    scratch: &mut SessionScratch,
    mut stream: TcpStream,
    peer_ip: &str,
) {
    let req = match read_http_request(&mut stream) {
        ReadHttpResult::Ok(req) => req,
        ReadHttpResult::TooLarge => {
            send_error(
                &mut stream,
                400,
                "el cuerpo de la petición excede el límite de 1 MiB",
            );
            return;
        }
        ReadHttpResult::InvalidContentLength => {
            send_error(&mut stream, 400, "cabecera Content-Length inválida");
            return;
        }
        ReadHttpResult::Empty => return,
    };

    log_http_request(&req, peer_ip);

    if req.method == "OPTIONS" {
        send_cors_options(&mut stream);
        return;
    }

    let is_stream = match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/v1/health") => return handle_health(&mut stream),
        ("GET", "/v1/voices") => return handle_voices(&mut stream, &ctx.voice),
        ("POST", "/v1/tts") => false,
        ("POST", "/v1/tts/stream") => true,
        _ => return send_error(&mut stream, 404, "no encontrado"),
    };
    match ValidatedTtsRequest::parse(&req.body, is_stream, &ctx.voice.name) {
        Ok(vreq) if is_stream => handle_tts_stream(ctx, scratch, &mut stream, &vreq),
        Ok(vreq) => handle_tts_full(ctx, scratch, &mut stream, &vreq),
        Err(msg) => send_error(&mut stream, 400, &msg),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    #[test]
    fn test_http_content_length_bounds_and_overflow() {
        assert_eq!(parse_content_length("Content-Length: 42\r\n"), Ok(42));
        assert_eq!(parse_content_length("Host: localhost\r\n"), Ok(0));
        assert!(parse_content_length("Content-Length: -5\r\n").is_err());
        assert!(parse_content_length("Content-Length: 99999999999999999999999\r\n").is_err());

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let client_t = std::thread::spawn(move || {
            let mut c = TcpStream::connect(addr).unwrap();
            let req =
                b"POST /v1/tts HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2000000\r\n\r\n{}";
            c.write_all(req).unwrap();
            let mut resp = String::new();
            let _ = c.read_to_string(&mut resp);
            resp
        });
        let (mut srv, _) = listener.accept().unwrap();
        let res = read_http_request(&mut srv);
        assert!(matches!(res, ReadHttpResult::TooLarge));
        send_error(
            &mut srv,
            400,
            "el cuerpo de la petición excede el límite de 1 MiB",
        );
        drop(srv);
        let resp = client_t.join().unwrap();
        assert!(resp.contains("400 Bad Request"));
    }
}
