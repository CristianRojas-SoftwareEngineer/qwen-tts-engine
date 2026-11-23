// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Pruebas de integración del contrato para clientes de proceso y HTTP: lanzan el binario real
//! `qwen_tts` y comprueban solo lo que un cliente puede asumir (códigos de salida, estados,
//! claves JSON, cabeceras, formato de audio y comportamiento). Ninguna aserción compara el
//! texto de un mensaje.
//!
//! Las pruebas sin modelo se ejecutan siempre. Las que lo necesitan viven en una sola prueba
//! secuencial (`contract_with_model`), para que nunca haya dos procesos con el modelo cargado a
//! la vez. Si falta el modelo, esa prueba se omite con un aviso, salvo que
//! `QWEN_TTS_REQUIRE_MODEL` valga `1`: entonces falla.

use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// Utilidades de proceso
// ---------------------------------------------------------------------------

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

fn model_dir() -> PathBuf {
    repo_root().join("models").join("qwen3-tts-0.6b-base")
}

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("qwen_app_contract_tests").join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// Comando base: sin las variables de hilos de OpenBLAS (el contrato pide al cliente no
/// definirlas) y sin salida capturada (no se compara texto).
fn command(args: &[&str]) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_qwen_tts"));
    cmd.args(args)
        .env_remove("OPENBLAS_NUM_THREADS")
        .env_remove("GOTO_NUM_THREADS")
        .env_remove("OMP_NUM_THREADS")
        .env_remove("QWEN_KERNELS")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    cmd
}

/// Espera a que el proceso termine y devuelve su código de salida y lo que tardó. Si pasa de
/// `limit`, lo mata (es un proceso lanzado por la prueba) y falla.
fn wait_exit(child: &mut Child, limit: Duration, started: Instant) -> (Option<i32>, Duration) {
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            return (status.code(), started.elapsed());
        }
        if started.elapsed() > limit {
            let _ = child.kill();
            let _ = child.wait();
            panic!("el proceso no terminó en {limit:?}");
        }
        thread::sleep(Duration::from_millis(20));
    }
}

/// Lanza el binario con la entrada cerrada y espera su salida.
fn run(args: &[&str], envs: &[(&str, &str)], limit: Duration) -> (Option<i32>, Duration) {
    let mut cmd = command(args);
    for (k, v) in envs {
        cmd.env(k, v);
    }
    let started = Instant::now();
    let mut child = cmd.spawn().expect("no se pudo lanzar qwen_tts");
    wait_exit(&mut child, limit, started)
}

fn model_arg() -> String {
    model_dir().to_string_lossy().into_owned()
}

// ---------------------------------------------------------------------------
// Pruebas sin modelo
// ---------------------------------------------------------------------------

#[test]
fn unknown_flag_exits_nonzero() {
    let (code, _) = run(&["--opcion-inexistente"], &[], Duration::from_secs(60));
    assert_ne!(code, Some(0));
}

#[test]
fn self_test_exits_zero() {
    let (code, _) = run(&["--self-test"], &[], Duration::from_secs(300));
    assert_eq!(code, Some(0));
}

#[test]
fn caps_exits_zero() {
    let (code, _) = run(&["--caps"], &[], Duration::from_secs(60));
    assert_eq!(code, Some(0));
}

#[test]
fn threads_flag_is_accepted() {
    let (code, _) = run(&["--caps", "-j", "2"], &[], Duration::from_secs(60));
    assert_eq!(code, Some(0));
}

#[test]
fn invalid_host_exits_1_quickly() {
    let model = model_arg();
    let (code, elapsed) = run(
        &["-d", &model, "--serve", "8080", "--host", "no-es-una-ip"],
        &[],
        Duration::from_secs(30),
    );
    assert_eq!(code, Some(1));
    assert!(elapsed < Duration::from_secs(2), "tardó {elapsed:?}");
}

#[test]
fn serve_port_zero_exits_1() {
    let model = model_arg();
    let (code, _) = run(&["-d", &model, "--serve", "0"], &[], Duration::from_secs(30));
    assert_eq!(code, Some(1));
}

#[test]
fn invalid_kernels_value_exits_1() {
    let (code, _) = run(&["--caps"], &[("QWEN_KERNELS", "otro")], Duration::from_secs(60));
    assert_eq!(code, Some(1));
}

#[test]
fn valid_kernels_values_exit_0() {
    for value in ["auto", "scalar"] {
        let (code, _) = run(&["--caps"], &[("QWEN_KERNELS", value)], Duration::from_secs(60));
        assert_eq!(code, Some(0), "QWEN_KERNELS={value}");
    }
}

#[test]
fn ref_audio_with_load_voice_exits_1() {
    let model = model_arg();
    let (code, _) = run(
        &[
            "-d", &model, "--text", "x", "--ref-audio", "a.wav", "--load-voice", "b.qvoice",
        ],
        &[],
        Duration::from_secs(30),
    );
    assert_eq!(code, Some(1));
}

#[test]
fn save_voice_without_ref_audio_exits_1() {
    let model = model_arg();
    let (code, _) = run(
        &["-d", &model, "--save-voice", "v.qvoice"],
        &[],
        Duration::from_secs(30),
    );
    assert_eq!(code, Some(1));
}

#[test]
fn missing_model_dir_exits_1() {
    let dir = temp_dir("missing_model");
    let missing = dir.join("no-existe");
    let (code, _) = run(
        &["-d", &missing.to_string_lossy(), "--text", "hola"],
        &[],
        Duration::from_secs(60),
    );
    assert_eq!(code, Some(1));
}

// ---------------------------------------------------------------------------
// WAV mínimo, escrito y leído por la prueba (sin ffmpeg)
// ---------------------------------------------------------------------------

struct Wav {
    channels: u16,
    rate: u32,
    bits: u16,
    pcm: Vec<u8>,
}

fn parse_wav(bytes: &[u8]) -> Wav {
    assert_eq!(&bytes[0..4], b"RIFF");
    assert_eq!(&bytes[8..12], b"WAVE");
    let mut pos = 12;
    let (mut channels, mut rate, mut bits) = (0, 0, 0);
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let body = pos + 8;
        if id == b"fmt " {
            assert_eq!(u16::from_le_bytes([bytes[body], bytes[body + 1]]), 1, "no es PCM");
            channels = u16::from_le_bytes([bytes[body + 2], bytes[body + 3]]);
            rate = u32::from_le_bytes(bytes[body + 4..body + 8].try_into().unwrap());
            bits = u16::from_le_bytes([bytes[body + 14], bytes[body + 15]]);
        } else if id == b"data" {
            let end = (body + size).min(bytes.len());
            return Wav {
                channels,
                rate,
                bits,
                pcm: bytes[body..end].to_vec(),
            };
        }
        pos = body + size + (size & 1);
    }
    panic!("el WAV no tiene fragmento data");
}

fn write_wav(path: &Path, channels: u16, rate: u32, samples: &[i16]) {
    let data_len = (samples.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + samples.len() * 2);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&(rate * u32::from(channels) * 2).to_le_bytes());
    out.extend_from_slice(&(channels * 2).to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    fs::write(path, out).unwrap();
}

// ---------------------------------------------------------------------------
// Cliente HTTP mínimo
// ---------------------------------------------------------------------------

struct HttpResponse {
    status: u16,
    headers: Vec<(String, String)>,
    /// Cuerpo con `Content-Length` (vacío si la respuesta es *chunked*).
    body: Vec<u8>,
    /// Fragmentos de una respuesta *chunked* (sin contar el terminador).
    chunks: Vec<Vec<u8>>,
    /// La respuesta *chunked* terminó con el fragmento final de tamaño cero.
    terminated: bool,
}

impl HttpResponse {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }

    /// PCM de una respuesta *chunked*: todos los fragmentos seguidos.
    fn chunked_pcm(&self) -> Vec<u8> {
        self.chunks.concat()
    }
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn parse_response(raw: &[u8]) -> HttpResponse {
    let head_end = find_subslice(raw, b"\r\n\r\n").expect("la respuesta no tiene cabeceras") + 4;
    let head = String::from_utf8_lossy(&raw[..head_end]).into_owned();
    let mut lines = head.split("\r\n");
    let status: u16 = lines
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse().ok())
        .expect("línea de estado ilegible");
    let headers: Vec<(String, String)> = lines
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_string()))
        .collect();
    let chunked = headers
        .iter()
        .any(|(k, v)| k == "transfer-encoding" && v.eq_ignore_ascii_case("chunked"));

    let rest = &raw[head_end..];
    let mut resp = HttpResponse {
        status,
        headers,
        body: Vec::new(),
        chunks: Vec::new(),
        terminated: false,
    };
    if !chunked {
        resp.body = rest.to_vec();
        return resp;
    }
    let mut pos = 0;
    while let Some(line_len) = find_subslice(&rest[pos..], b"\r\n") {
        let size_text = String::from_utf8_lossy(&rest[pos..pos + line_len]).into_owned();
        let Ok(size) = usize::from_str_radix(size_text.trim(), 16) else {
            break;
        };
        pos += line_len + 2;
        if size == 0 {
            resp.terminated = true;
            break;
        }
        if rest.len() < pos + size + 2 {
            break;
        }
        resp.chunks.push(rest[pos..pos + size].to_vec());
        pos += size + 2;
    }
    resp
}

/// Envía una petición cruda, lee hasta EOF y la interpreta. No cierra la mitad de escritura.
fn exchange(port: u16, request: &[u8]) -> HttpResponse {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("no se pudo conectar");
    stream.set_read_timeout(Some(Duration::from_secs(180))).unwrap();
    let _ = stream.write_all(request);
    let mut raw = Vec::new();
    let _ = stream.read_to_end(&mut raw);
    parse_response(&raw)
}

fn get(port: u16, path: &str) -> HttpResponse {
    exchange(
        port,
        format!("GET {path} HTTP/1.1\r\nHost: localhost\r\n\r\n").as_bytes(),
    )
}

fn post_raw(port: u16, path: &str, body: &str) -> HttpResponse {
    exchange(
        port,
        format!(
            "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\n\r\n{body}",
            body.len()
        )
        .as_bytes(),
    )
}

fn post(port: u16, path: &str, body: &Value) -> HttpResponse {
    post_raw(port, path, &body.to_string())
}

fn assert_json_error(resp: &HttpResponse, status: u16, caso: &str) {
    assert_eq!(resp.status, status, "{caso}: estado");
    assert!(
        resp.header("content-type")
            .is_some_and(|v| v.starts_with("application/json")),
        "{caso}: Content-Type no es JSON"
    );
    let value: Value =
        serde_json::from_slice(&resp.body).unwrap_or_else(|_| panic!("{caso}: cuerpo no es JSON"));
    assert!(value["error"].is_string(), "{caso}: falta `error` de tipo cadena");
}

// ---------------------------------------------------------------------------
// Servidor lanzado por la prueba
// ---------------------------------------------------------------------------

struct Server {
    child: Child,
    stdin: Option<ChildStdin>,
    port: u16,
}

impl Server {
    fn start(model: &str) -> Server {
        let port = {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            listener.local_addr().unwrap().port()
        };
        let mut cmd = command(&["-d", model, "--serve", &port.to_string(), "--watch-stdin"]);
        cmd.stdin(Stdio::piped());
        let mut child = cmd.spawn().expect("no se pudo lanzar el servidor");
        let stdin = child.stdin.take();
        Server {
            child,
            stdin,
            port,
        }
    }

    /// Sondea `/v1/health` cada 250 ms, con un tope de 120 s.
    fn wait_ready(&mut self) {
        let started = Instant::now();
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                panic!("el servidor terminó antes de estar listo ({status:?})");
            }
            if started.elapsed() > Duration::from_secs(120) {
                panic!("el servidor no respondió a /v1/health en 120 s");
            }
            if let Ok(mut s) =
                TcpStream::connect_timeout(&([127, 0, 0, 1], self.port).into(), Duration::from_secs(1))
            {
                s.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
                let _ = s.write_all(b"GET /v1/health HTTP/1.1\r\nHost: localhost\r\n\r\n");
                let mut raw = Vec::new();
                let _ = s.read_to_end(&mut raw);
                if find_subslice(&raw, b"\r\n\r\n").is_some() && parse_response(&raw).status == 200 {
                    return;
                }
            }
            thread::sleep(Duration::from_millis(250));
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        // Red de seguridad: si la prueba falla a medias, no deja el servidor vivo.
        if matches!(self.child.try_wait(), Ok(None)) {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

// ---------------------------------------------------------------------------
// Prueba con modelo
// ---------------------------------------------------------------------------

/// Texto largo: la síntesis dura varios segundos, suficiente para cancelarla o para que otra
/// conexión espere.
fn long_text() -> String {
    "Esta es una frase bastante larga para que la síntesis tarde más de un segundo en \
     completarse. "
        .repeat(8)
}

fn medium_text() -> String {
    "Esta es una frase de prueba. Y esta es otra frase, un poco más larga que la anterior. \
     Y esta es la tercera."
        .to_string()
}

#[test]
fn contract_with_model() {
    if !model_dir().join("config.json").exists() {
        if std::env::var("QWEN_TTS_REQUIRE_MODEL").as_deref() == Ok("1") {
            panic!("falta el modelo en {} y QWEN_TTS_REQUIRE_MODEL=1", model_dir().display());
        }
        eprintln!("Se omite contract_with_model (falta el modelo)");
        return;
    }
    let model = model_arg();
    let tmp = temp_dir("with_model");

    // 1. Referencia de clonado: la voz `default` dice una frase; sale en WAV PCM mono de
    // 16 bits a 24 kHz. De ahí se derivan una versión estéreo y otra a 48 kHz.
    let ref24 = tmp.join("ref24.wav");
    let (code, _) = run(
        &[
            "-d",
            &model,
            "--text",
            "Hola, esta es una frase de referencia para probar el clonado de voz.",
            "-o",
            &ref24.to_string_lossy(),
            "-S",
        ],
        &[],
        Duration::from_secs(300),
    );
    assert_eq!(code, Some(0), "síntesis de la referencia");
    let wav = parse_wav(&fs::read(&ref24).unwrap());
    assert_eq!((wav.channels, wav.rate, wav.bits), (1, 24000, 16));
    let samples: Vec<i16> = wav
        .pcm
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect();
    assert!(samples.len() > 24000, "la referencia es demasiado corta");
    let stereo: Vec<i16> = samples.iter().flat_map(|&s| [s, s]).collect();
    let rate48: Vec<i16> = samples.iter().flat_map(|&s| [s, s]).collect();
    let ref_stereo = tmp.join("ref_stereo.wav");
    let ref_48k = tmp.join("ref_48k.wav");
    write_wav(&ref_stereo, 2, 24000, &stereo);
    write_wav(&ref_48k, 1, 48000, &rate48);

    // 2. Estéreo y 48 kHz como referencia salen con 1 antes de cargar el modelo.
    for (nombre, wav_path) in [("estéreo", &ref_stereo), ("48 kHz", &ref_48k)] {
        let out = tmp.join("rechazada.qvoice");
        let (code, elapsed) = run(
            &[
                "-d",
                &model,
                "--ref-audio",
                &wav_path.to_string_lossy(),
                "--save-voice",
                &out.to_string_lossy(),
            ],
            &[],
            Duration::from_secs(30),
        );
        assert_eq!(code, Some(1), "referencia {nombre}");
        assert!(elapsed < Duration::from_secs(2), "referencia {nombre}: tardó {elapsed:?}");
        assert!(!out.exists(), "referencia {nombre}: no debe dejar archivo");
    }

    // 3. Clonado con --watch-stdin y la entrada abierta hasta que el proceso termina.
    let voice_ok = tmp.join("clon.qvoice");
    let mut cmd = command(&[
        "-d",
        &model,
        "--ref-audio",
        &ref24.to_string_lossy(),
        "--save-voice",
        &voice_ok.to_string_lossy(),
        "--watch-stdin",
        "-S",
    ]);
    cmd.stdin(Stdio::piped());
    let started = Instant::now();
    let mut child = cmd.spawn().expect("no se pudo lanzar el clonado");
    let stdin_abierta = child.stdin.take();
    let (code, _) = wait_exit(&mut child, Duration::from_secs(300), started);
    drop(stdin_abierta);
    assert_eq!(code, Some(0), "clonado con la entrada abierta");
    let qvoice = fs::read(&voice_ok).expect("el clonado no dejó el .qvoice");
    assert_eq!(qvoice.len(), 4112);
    assert_eq!(&qvoice[0..4], b"QVCE");
    assert_eq!(u32::from_le_bytes(qvoice[4..8].try_into().unwrap()), 6, "version");
    assert_eq!(
        u32::from_le_bytes(qvoice[4108..4112].try_into().unwrap()),
        0,
        "lang_id (sin -l)"
    );

    // 4. Clonado con --watch-stdin y la entrada cerrada al instante: abandona con 0 y no deja
    // un .qvoice válido.
    let voice_abandoned = tmp.join("abandonado.qvoice");
    let mut cmd = command(&[
        "-d",
        &model,
        "--ref-audio",
        &ref24.to_string_lossy(),
        "--save-voice",
        &voice_abandoned.to_string_lossy(),
        "--watch-stdin",
        "-S",
    ]);
    cmd.stdin(Stdio::piped());
    let started = Instant::now();
    let mut child = cmd.spawn().expect("no se pudo lanzar el clonado");
    drop(child.stdin.take());
    let (code, elapsed) = wait_exit(&mut child, Duration::from_secs(60), started);
    assert_eq!(code, Some(0), "clonado abandonado");
    assert!(elapsed < Duration::from_secs(5), "tardó {elapsed:?}");
    let valido = fs::metadata(&voice_abandoned).is_ok_and(|m| m.len() == 4112);
    assert!(!valido, "no debe quedar un .qvoice válido");

    // 5. Servidor.
    let mut server = Server::start(&model);
    server.wait_ready();
    let port = server.port;

    // 6. Salud, voces y 404.
    let health = get(port, "/v1/health");
    assert_eq!(health.status, 200);
    let health_json: Value = serde_json::from_slice(&health.body).unwrap();
    assert_eq!(health_json, json!({"status": "ok"}));

    let voices = get(port, "/v1/voices");
    assert_eq!(voices.status, 200);
    let voices_json: Value = serde_json::from_slice(&voices.body).unwrap();
    let list = voices_json["voices"].as_array().expect("`voices` no es una lista");
    assert_eq!(list.len(), 1);
    assert_eq!(list[0]["name"], "default");
    assert!(list[0].get("language").is_some_and(|l| l.is_string() || l.is_null()));
    assert_eq!(list[0]["language"], "Spanish");

    assert_json_error(&get(port, "/v1/no-existe"), 404, "ruta desconocida");
    assert_json_error(&get(port, "/v1/tts"), 404, "método desconocido en /v1/tts");
    assert_json_error(
        &exchange(
            port,
            b"OPTIONS /v1/tts HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\n\r\n",
        ),
        404,
        "OPTIONS /v1/tts",
    );
    assert!(
        server.child.try_wait().unwrap().is_none(),
        "--serve no debe terminar por sí solo"
    );

    // 7. Peticiones inválidas: 400 con `error` de tipo cadena y JSON.
    let casos: Vec<(&str, &str, Value)> = vec![
        ("campo desconocido", "/v1/tts", json!({"text": "Hola.", "speaker": "x"})),
        ("idioma desconocido", "/v1/tts", json!({"text": "Hola.", "language": "Klingon"})),
        ("texto vacío", "/v1/tts", json!({"text": ""})),
        ("texto de 8193 bytes", "/v1/tts", json!({"text": "a".repeat(8193)})),
        ("voz distinta de la de arranque", "/v1/tts", json!({"text": "Hola.", "voice": "otra"})),
        ("temperature fuera de rango", "/v1/tts", json!({"text": "Hola.", "temperature": 5.0})),
        ("top_k fuera de rango", "/v1/tts", json!({"text": "Hola.", "top_k": 99999})),
        ("seed mayor que u32::MAX", "/v1/tts", json!({"text": "Hola.", "seed": 4294967296u64})),
        ("rate no admitido en el stream", "/v1/tts/stream", json!({"text": "Hola.", "rate": 1.0})),
        ("chunk_frames no admitido en /v1/tts", "/v1/tts", json!({"text": "Hola.", "chunk_frames": 10})),
        ("temperature fuera de rango en el stream", "/v1/tts/stream", json!({"text": "Hola.", "temperature": -1.0})),
    ];
    for (caso, path, body) in &casos {
        assert_json_error(&post(port, path, body), 400, caso);
    }
    assert_json_error(&post_raw(port, "/v1/tts", "{"), 400, "JSON inválido");
    assert_json_error(&post_raw(port, "/v1/tts", "[1,2]"), 400, "JSON que no es un objeto");
    assert_json_error(
        &exchange(
            port,
            b"POST /v1/tts HTTP/1.1\r\nHost: localhost\r\nContent-Length: abc\r\n\r\n",
        ),
        400,
        "Content-Length: abc",
    );
    // Un cuerpo de 4 MiB enviado entero: el 400 llega aunque el cliente siga enviando.
    let mut grande =
        b"POST /v1/tts HTTP/1.1\r\nHost: localhost\r\nContent-Length: 4194304\r\n\r\n".to_vec();
    grande.extend(std::iter::repeat_n(b'a', 4 * 1024 * 1024));
    for intento in 0..3 {
        assert_json_error(&exchange(port, &grande), 400, &format!("cuerpo de 4 MiB, intento {intento}"));
    }

    // 8. `/v1/tts/stream` completo.
    let stream = post(port, "/v1/tts/stream", &json!({"text": "Hola, mundo.", "seed": 4}));
    assert_eq!(stream.status, 200);
    assert_eq!(stream.header("transfer-encoding"), Some("chunked"));
    assert_eq!(stream.header("content-type"), Some("audio/pcm"));
    assert_eq!(stream.header("x-sample-rate"), Some("24000"));
    assert_eq!(stream.header("x-sample-format"), Some("s16le"));
    assert_eq!(stream.header("x-channels"), Some("1"));
    assert!(!stream.chunks.is_empty(), "el stream no trae fragmentos");
    assert!(
        stream.chunks.iter().all(|c| c.len() % 2 == 0),
        "hay un fragmento de tamaño impar"
    );
    assert!(stream.terminated, "falta el fragmento final de tamaño cero");

    // 9. Texto solo de pausas: el stream termina sin terminador y /v1/tts responde 500.
    let pausas_stream = post(port, "/v1/tts/stream", &json!({"text": "[pause:1]"}));
    assert_eq!(pausas_stream.status, 200);
    assert!(!pausas_stream.terminated, "el stream fallido no debe llevar terminador");
    assert_json_error(
        &post(port, "/v1/tts", &json!({"text": "[pause:1]"})),
        500,
        "texto solo de pausas",
    );

    // 10. Marcado: el PCM del stream es el del WAV sin sus 44 bytes de cabecera.
    let marcado = json!({"text": "Uno. | [pause:0.3] Dos.", "seed": 4});
    let stream_marcado = post(port, "/v1/tts/stream", &marcado);
    assert!(stream_marcado.terminated);
    let wav_marcado = post(port, "/v1/tts", &marcado);
    assert_eq!(wav_marcado.status, 200);
    assert_eq!(wav_marcado.header("content-type"), Some("audio/wav"));
    let wav_parsed = parse_wav(&wav_marcado.body);
    assert_eq!((wav_parsed.channels, wav_parsed.rate, wav_parsed.bits), (1, 24000, 16));
    assert_eq!(wav_marcado.body.len(), 44 + wav_parsed.pcm.len(), "cabecera WAV de 44 bytes");
    assert_eq!(
        stream_marcado.chunked_pcm(),
        wav_marcado.body[44..],
        "el PCM del stream y el del WAV difieren"
    );

    // Voz e idioma sin distinguir mayúsculas.
    let mayusculas = post(
        port,
        "/v1/tts",
        &json!({"text": "Ok.", "voice": "DEFAULT", "language": "ENGLISH"}),
    );
    assert_eq!(mayusculas.status, 200);

    // 11. Cancelación: el cliente cierra el socket a mitad del stream; el servidor sigue.
    {
        let body = json!({"text": long_text(), "seed": 4}).to_string();
        let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(60))).unwrap();
        s.write_all(
            format!(
                "POST /v1/tts/stream HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            )
            .as_bytes(),
        )
        .unwrap();
        let mut raw = Vec::new();
        let mut buf = [0u8; 4096];
        while find_subslice(&raw, b"\r\n\r\n").is_none() {
            let n = s.read(&mut buf).expect("no llegó la cabecera del stream");
            assert!(n > 0, "el servidor cerró antes de la cabecera");
            raw.extend_from_slice(&buf[..n]);
        }
        thread::sleep(Duration::from_millis(500));
        drop(s);
        let t = Instant::now();
        let health = get(port, "/v1/health");
        assert_eq!(health.status, 200);
        assert!(
            t.elapsed() < Duration::from_secs(2),
            "/v1/health tardó {:?} tras la desconexión",
            t.elapsed()
        );
        let despues = post(port, "/v1/tts/stream", &json!({"text": "Hola.", "seed": 4}));
        assert!(despues.terminated, "la síntesis posterior no terminó con terminador");
    }

    // 12. Determinismo: la misma semilla da los mismos bytes.
    let pedido = json!({"text": "Hola mundo.", "seed": 7});
    let a = post(port, "/v1/tts", &pedido);
    let b = post(port, "/v1/tts", &pedido);
    assert_eq!(a.status, 200);
    assert_eq!(a.body, b.body, "misma semilla, audio distinto");

    // 13. Una conexión a la vez: con una síntesis en curso, otra conexión espera.
    {
        let terminada = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&terminada);
        let pedido = json!({"text": medium_text(), "seed": 4});
        let primera = thread::spawn(move || {
            let r = post(port, "/v1/tts", &pedido);
            flag.store(true, Ordering::SeqCst);
            r.status
        });
        thread::sleep(Duration::from_millis(700));
        assert!(
            !terminada.load(Ordering::SeqCst),
            "la primera síntesis terminó antes de probar la espera"
        );
        let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
        s.write_all(b"GET /v1/health HTTP/1.1\r\nHost: localhost\r\n\r\n").unwrap();
        s.set_read_timeout(Some(Duration::from_millis(300))).unwrap();
        let mut buf = [0u8; 1024];
        let respuesta_inmediata = matches!(s.read(&mut buf), Ok(n) if n > 0);
        assert!(
            !respuesta_inmediata,
            "la segunda conexión recibió respuesta mientras la primera seguía en curso"
        );
        s.set_read_timeout(Some(Duration::from_secs(180))).unwrap();
        let mut raw = Vec::new();
        let _ = s.read_to_end(&mut raw);
        assert_eq!(parse_response(&raw).status, 200);
        assert_eq!(primera.join().unwrap(), 200);
    }

    // 14. Cerrar la entrada estándar del servidor: sale con 0 en menos de 20 s.
    drop(server.stdin.take());
    let started = Instant::now();
    let (code, elapsed) = wait_exit(&mut server.child, Duration::from_secs(60), started);
    assert_eq!(code, Some(0), "salida del servidor");
    assert!(elapsed < Duration::from_secs(20), "tardó {elapsed:?}");
}
