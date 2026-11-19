// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::io::{ErrorKind, Read, Write};
use std::net::TcpStream;
use std::thread;
use std::time::{Duration, Instant};

use qwen_core::config::{PRESET_SPEAKERS, SAMPLE_RATE};
use qwen_core::MarkupPlanner;
use qwen_engine::{
    encode_wav_bytes, prepare_markup_06b, render_spans_buffer,
    render_spans_stream, synthesize_to_samples, synthesize_to_sink, AbortHandle, AudioSink,
    EngineError, MemorySink, Model06B, RequestConfig, RequestExecution, SessionScratch,
    StreamEdgeRamp, SynthesisTiming, VoiceBindings, VoiceIdentity,
};
use qwen_kernels::ComputeRuntime;

use super::validation::ValidatedTtsRequest;

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
        Err(e) if e.kind() == ErrorKind::WouldBlock || e.kind() == ErrorKind::Interrupted => {
            false
        }
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
            Err(e)
                if e.kind() == ErrorKind::WouldBlock || e.kind() == ErrorKind::Interrupted =>
            {
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
    buf.windows(4)
        .position(|w| w == b"\r\n\r\n")
        .map(|p| p + 4)
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
        "[HTTP] {} {} {} from {}",
        req.method,
        req.path,
        if req.method == "POST" && !req.body.is_empty() {
            "(has body)"
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

/// Envía un fragmento *chunked* con PCM s16le ya procesado (ganancia y rampas de borde aplicadas
/// por `StreamEdgeRamp`). Un fragmento vacío no se envía: cerraría la respuesta.
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

pub fn handle_speakers(stream: &mut TcpStream) {
    let mut out = String::from("{\"speakers\":[");
    for (idx, spk) in PRESET_SPEAKERS.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        out.push_str(&format!(
            "{{\"name\":\"{}\",\"language\":\"{}\",\"gender\":\"{}\"}}",
            spk.name, spk.language, spk.gender
        ));
    }
    out.push_str("]}");
    send_json(stream, 200, &out);
}

pub fn default_voice_context(base_voice: &VoiceBindings) -> (bool, i32, Option<i32>) {
    let has_clone = !matches!(base_voice.identity, VoiceIdentity::Preset(_));
    let (def_spk, def_lang) = match &base_voice.identity {
        VoiceIdentity::Preset(id) => (*id, Some(2050)),
        _ => (3061, None),
    };
    (has_clone, def_spk, def_lang)
}

pub fn log_wav_sent(bytes: usize, samples_len: usize, elapsed_s: f64) {
    let audio_secs = samples_len as f64 / SAMPLE_RATE as f64;
    eprintln!(
        "[HTTP] Sent {} bytes WAV ({:.2}s audio) in {:.1}s (RTF {:.2})",
        bytes,
        audio_secs,
        elapsed_s,
        elapsed_s / audio_secs.max(1e-6)
    );
}

pub fn log_stream_sent(samples_len: usize, elapsed_s: f64) {
    let audio_secs = samples_len as f64 / SAMPLE_RATE as f64;
    eprintln!(
        "[HTTP] Streamed {} samples ({:.2}s audio) in {:.1}s (RTF {:.2})",
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
        "[HTTP] Desglose: prefill {:.0} ms, Talker {:.0} ms ({:.1} ms/f), CP {:.0} ms ({:.1} ms/f), decodificador {:.0} ms, drenaje {:.0} ms, {} fotogramas en {} fragmentos, cola máxima {} fotogramas",
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

pub struct HttpAudioChunkSink<'a> {
    stream: &'a mut TcpStream,
    ramp: StreamEdgeRamp,
    total_samples: usize,
}

impl<'a> HttpAudioChunkSink<'a> {
    pub fn new(stream: &'a mut TcpStream, volume: f32) -> Self {
        Self {
            stream,
            ramp: StreamEdgeRamp::new(SAMPLE_RATE as u32, volume),
            total_samples: 0,
        }
    }

    fn send(&mut self, pcm: &[u8]) -> Result<(), EngineError> {
        send_pcm_chunk(self.stream, pcm).map_err(|_| EngineError::Aborted)
    }
}

impl AudioSink for HttpAudioChunkSink<'_> {
    fn write_samples(&mut self, samples: &[f32]) -> Result<(), EngineError> {
        if samples.is_empty() {
            return Ok(());
        }
        let mut pcm = Vec::new();
        self.ramp.push(samples, &mut pcm);
        self.send(&pcm)?;
        self.total_samples += samples.len();
        Ok(())
    }

    fn finish(&mut self) -> Result<(), EngineError> {
        let mut pcm = Vec::new();
        self.ramp.finish(&mut pcm);
        self.send(&pcm)
    }
}

struct PreparedHttpTts<'a> {
    voice_ref: &'a VoiceBindings,
    prep: qwen_engine::PreparedMarkup,
    effective_seed: u32,
}

fn prepare_http_tts<'a>(
    base_voice: &'a VoiceBindings,
    vreq: &ValidatedTtsRequest,
    owned_voice: &'a mut Option<VoiceBindings>,
) -> PreparedHttpTts<'a> {
    let seed = vreq.resolve_seed_or_time();
    if matches!(base_voice.identity, VoiceIdentity::Preset(_)) {
        // Las voces preset no tienen pesos WOVR; solo cambia la identidad del preset.
        *owned_voice = Some(VoiceBindings {
            identity: VoiceIdentity::Preset(vreq.speaker_id),
            ..base_voice.clone()
        });
    }
    let voice_ref = owned_voice.as_ref().unwrap_or(base_voice);
    let prep = prepare_markup_06b(
        &vreq.text,
        voice_ref,
        false,
        if vreq.seed_set { Some(seed) } else { None },
        if vreq.temp_set {
            Some(vreq.temperature)
        } else {
            None
        },
        vreq.temperature,
    );
    let effective_seed = if !vreq.seed_set && prep.did_para_substitute {
        prep.seed
    } else {
        seed
    };
    PreparedHttpTts {
        voice_ref,
        prep,
        effective_seed,
    }
}

/// Fotogramas de la síntesis de calentamiento: el primer fragmento (2) y uno completo (10).
const WARM_UP_FRAMES: usize = 12;

/// Síntesis corta y descartada al arrancar un worker del servidor.
///
/// Deja `scratch` (Talker, Code Predictor y decoder) dimensionado y en caliente, para que la
/// primera petición no pague el arranque en frío (~60–80 ms de TTFA). No cambia el audio de
/// las peticiones: cada una reinicia su estado y su semilla.
pub fn warm_up_session(
    runtime: &ComputeRuntime,
    model: &Model06B,
    base_voice: &VoiceBindings,
    scratch: &mut SessionScratch,
) {
    let (has_clone, def_spk, def_lang) = default_voice_context(base_voice);
    let vreq = match ValidatedTtsRequest::parse(
        r#"{"text":"Hello.","seed":1}"#,
        true,
        has_clone,
        def_spk,
        def_lang,
    ) {
        Ok(v) => v,
        Err(msg) => {
            eprintln!("[HTTP] Calentamiento omitido: {msg}");
            return;
        }
    };
    let mut owned_voice = None;
    let ctx = prepare_http_tts(base_voice, &vreq, &mut owned_voice);
    let mut exec = build_http_exec(&vreq, &ctx.prep, ctx.effective_seed, true, AbortHandle::new());
    exec.config.max_tokens = WARM_UP_FRAMES;
    let mut sink = MemorySink::new(0);
    let t0 = Instant::now();
    match synthesize_to_sink(
        runtime,
        model,
        ctx.voice_ref,
        scratch,
        &exec,
        &ctx.prep.text,
        &mut sink,
    ) {
        Ok(_) => eprintln!("[HTTP] Calentamiento: {:.0} ms", t0.elapsed().as_secs_f64() * 1000.0),
        Err(e) => eprintln!("[HTTP] Calentamiento fallido (sin efecto en las peticiones): {e}"),
    }
}

fn build_http_exec(
    vreq: &ValidatedTtsRequest,
    prep: &qwen_engine::PreparedMarkup,
    effective_seed: u32,
    is_stream: bool,
    abort: AbortHandle,
) -> RequestExecution {
    RequestExecution {
        config: RequestConfig {
            speaker_id: vreq.speaker_id,
            language_id: vreq.language_id,
            temperature: prep.temperature,
            top_k: vreq.top_k,
            top_p: vreq.top_p,
            rep_penalty: vreq.rep_penalty,
            max_tokens: 8192,
            greedy_warmup: 0,
            seed: effective_seed,
            stream: is_stream,
            stream_chunk_frames: if is_stream { vreq.chunk_frames } else { 10 },
            volume: vreq.volume,
            rate: if is_stream { 1.0 } else { vreq.rate },
            silent: true,
            debug: false,
            cp_roughness: 0.0,
        },
        abort,
    }
}

pub fn handle_tts_full(
    runtime: &ComputeRuntime,
    model: &Model06B,
    base_voice: &VoiceBindings,
    scratch: &mut SessionScratch,
    stream: &mut TcpStream,
    vreq: &ValidatedTtsRequest,
) {
    let mut owned_voice = None;
    let ctx = prepare_http_tts(base_voice, vreq, &mut owned_voice);
    let voice_ref = ctx.voice_ref;
    let prep = ctx.prep;
    let effective_seed = ctx.effective_seed;

    eprintln!(
        "[HTTP] TTS: \"{}\" (speaker={}, lang={}, seed={})",
        prep.text,
        vreq.speaker_id,
        vreq.language_id.unwrap_or(0),
        effective_seed
    );

    let guard = match AbortGuard::for_socket(stream) {
        Ok(g) => g,
        Err(_) => {
            send_error(stream, 500, "internal socket error");
            return;
        }
    };
    let exec = build_http_exec(vreq, &prep, effective_seed, false, guard.abort.clone());

    let t0 = Instant::now();
    let mut timing_opt: Option<SynthesisTiming> = None;
    let res = if prep.use_compose {
        let spans = MarkupPlanner::parse_spans(&prep.text);
        if !spans.is_empty() {
            eprintln!(
                "[HTTP] inline markup -> per-sentence compose ({} spans)",
                spans.len()
            );
            render_spans_buffer(runtime, model, voice_ref, scratch, &exec, &spans, 0.12)
                .map(|(buf, _)| buf)
        } else {
            Err(EngineError::InvalidArg("markup parse failed".to_string()))
        }
    } else {
        synthesize_to_samples(runtime, model, voice_ref, scratch, &exec, &prep.text).map(
            |(buf, timing)| {
                timing_opt = Some(timing);
                buf
            },
        )
    };

    let samples = match res {
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
                send_error(stream, 500, "generation failed");
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

pub fn handle_tts_stream(
    runtime: &ComputeRuntime,
    model: &Model06B,
    base_voice: &VoiceBindings,
    scratch: &mut SessionScratch,
    stream: &mut TcpStream,
    vreq: &ValidatedTtsRequest,
) {
    let mut owned_voice = None;
    let ctx = prepare_http_tts(base_voice, vreq, &mut owned_voice);
    let voice_ref = ctx.voice_ref;
    let prep = ctx.prep;
    let effective_seed = ctx.effective_seed;

    eprintln!(
        "[HTTP] TTS stream: \"{}\" (speaker={}, lang={}, seed={})",
        prep.text,
        vreq.speaker_id,
        vreq.language_id.unwrap_or(0),
        effective_seed
    );

    let guard = match AbortGuard::for_socket(stream) {
        Ok(g) => g,
        Err(_) => {
            send_error(stream, 500, "internal socket error");
            return;
        }
    };

    if send_chunked_header(stream).is_err() {
        return;
    }

    let t0 = Instant::now();
    let exec = build_http_exec(vreq, &prep, effective_seed, true, guard.abort.clone());

    let mut chunk_sink = HttpAudioChunkSink::new(stream, vreq.volume);

    let mut timing_opt: Option<SynthesisTiming> = None;
    let gen_res: Result<(), EngineError> = if prep.use_compose {
        let spans = MarkupPlanner::parse_spans(&prep.text);
        if !spans.is_empty() {
            eprintln!(
                "[HTTP] inline markup -> per-sentence compose stream ({} spans)",
                spans.len()
            );
            render_spans_stream(
                runtime,
                model,
                voice_ref,
                scratch,
                &exec,
                &spans,
                0.12,
                &mut chunk_sink,
            )
            .map(|_| ())
        } else {
            Err(EngineError::InvalidArg("markup parse failed".to_string()))
        }
    } else {
        synthesize_to_sink(
            runtime,
            model,
            voice_ref,
            scratch,
            &exec,
            &prep.text,
            &mut chunk_sink,
        )
        .map(|t| {
            timing_opt = Some(t);
        })
    };

    let total_samples = chunk_sink.total_samples;
    if gen_res.is_err() || total_samples == 0 {
        if socket_disconnected(stream) || matches!(gen_res, Err(EngineError::Aborted)) {
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

pub fn handle_connection(
    runtime: &ComputeRuntime,
    model: &Model06B,
    base_voice: &VoiceBindings,
    scratch: &mut SessionScratch,
    mut stream: TcpStream,
    peer_ip: &str,
) {
    let req = match read_http_request(&mut stream) {
        ReadHttpResult::Ok(req) => req,
        ReadHttpResult::TooLarge => {
            send_error(&mut stream, 400, "request body exceeds 1 MiB limit");
            return;
        }
        ReadHttpResult::InvalidContentLength => {
            send_error(&mut stream, 400, "invalid Content-Length header");
            return;
        }
        ReadHttpResult::Empty => return,
    };

    log_http_request(&req, peer_ip);

    if req.method == "OPTIONS" {
        send_cors_options(&mut stream);
        return;
    }

    let (has_clone, def_spk, def_lang) = default_voice_context(base_voice);

    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/v1/health") => handle_health(&mut stream),
        ("GET", "/v1/speakers") => handle_speakers(&mut stream),
        ("POST", "/v1/tts") | ("POST", "/v1/audio/speech") => {
            match ValidatedTtsRequest::parse(&req.body, false, has_clone, def_spk, def_lang) {
                Ok(vreq) => handle_tts_full(runtime, model, base_voice, scratch, &mut stream, &vreq),
                Err(msg) => send_error(&mut stream, 400, &msg),
            }
        }
        ("POST", "/v1/tts/stream") => {
            match ValidatedTtsRequest::parse(&req.body, true, has_clone, def_spk, def_lang) {
                Ok(vreq) => {
                    handle_tts_stream(runtime, model, base_voice, scratch, &mut stream, &vreq)
                }
                Err(msg) => send_error(&mut stream, 400, &msg),
            }
        }
        _ => send_error(&mut stream, 404, "not found"),
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
            let req = b"POST /v1/tts HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2000000\r\n\r\n{}";
            c.write_all(req).unwrap();
            let mut resp = String::new();
            let _ = c.read_to_string(&mut resp);
            resp
        });
        let (mut srv, _) = listener.accept().unwrap();
        let res = read_http_request(&mut srv);
        assert!(matches!(res, ReadHttpResult::TooLarge));
        send_error(&mut srv, 400, "request body exceeds 1 MiB limit");
        drop(srv);
        let resp = client_t.join().unwrap();
        assert!(resp.contains("400 Bad Request"));
    }
}
