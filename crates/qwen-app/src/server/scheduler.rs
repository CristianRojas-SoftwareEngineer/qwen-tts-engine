// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::collections::VecDeque;
use std::net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::thread;
use std::time::Instant;

use qwen_core::config::SAMPLE_RATE;
use qwen_engine::batch::MAX_BATCH_SLOTS;
use qwen_engine::{
    encode_wav_bytes, prepare_markup_06b, run_continuous_batch, AbortHandle, BatchReq,
    ContinuousBatchSink, ContinuousSlotJob, EngineError, Model06B, SessionScratch, VoiceBindings,
};
use qwen_kernels::ComputeRuntime;

use super::http::{
    default_voice_context, handle_connection, handle_health, handle_speakers, handle_tts_full,
    handle_tts_stream, log_http_request, log_stream_sent, log_wav_sent, read_http_request,
    send_chunked_end, send_chunked_header, send_cors_options, send_error, send_pcm_chunk,
    send_response, socket_disconnected, ReadHttpResult,
};
use super::validation::ValidatedTtsRequest;

const CONN_QUEUE_CAP: usize = 256;

pub trait SynthesisScheduler {
    fn run(self: Box<Self>, listener: TcpListener) -> Result<(), String>;
}

pub struct WorkerPoolScheduler {
    pub runtime: Arc<ComputeRuntime>,
    pub model: Arc<Model06B>,
    pub voice: Arc<VoiceBindings>,
    pub n_workers: usize,
    pub physical_cores: usize,
}

impl SynthesisScheduler for WorkerPoolScheduler {
    fn run(self: Box<Self>, listener: TcpListener) -> Result<(), String> {
        if self.n_workers <= 1 {
            let mut scratch = SessionScratch::new();
            for incoming in listener.incoming() {
                match incoming {
                    Ok(stream) => {
                        let peer_ip = stream
                            .peer_addr()
                            .map(|a| a.ip().to_string())
                            .unwrap_or_else(|_| "0.0.0.0".to_string());
                        handle_connection(
                            &self.runtime,
                            &self.model,
                            &self.voice,
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
            return Ok(());
        }

        let total_threads = self.runtime.pool.n_threads();
        let worker_threads = (total_threads / self.n_workers).max(1);
        let spin_budget = if self.n_workers * worker_threads >= self.physical_cores.max(1) {
            0
        } else {
            4096
        };

        let (tx, rx) = mpsc::sync_channel::<(TcpStream, String)>(CONN_QUEUE_CAP);
        let rx = Arc::new(Mutex::new(rx));

        let mut handles = Vec::with_capacity(self.n_workers);
        for _ in 0..self.n_workers {
            let rx_c = Arc::clone(&rx);
            let model_c = Arc::clone(&self.model);
            let voice_c = Arc::clone(&self.voice);
            let h = thread::Builder::new()
                .name("qwen-http-worker".to_string())
                .spawn(move || {
                    let rt = ComputeRuntime::with_threads_and_spin(
                        worker_threads,
                        spin_budget,
                        Some(worker_threads),
                    );
                    let mut scratch = SessionScratch::new();
                    loop {
                        let item = {
                            let guard = rx_c.lock().unwrap_or_else(|e| e.into_inner());
                            guard.recv()
                        };
                        let Ok((stream, peer_ip)) = item else {
                            break;
                        };
                        handle_connection(&rt, &model_c, &voice_c, &mut scratch, stream, &peer_ip);
                    }
                })
                .map_err(|e| format!("failed to spawn HTTP worker: {e}"))?;
            handles.push(h);
        }

        for incoming in listener.incoming() {
            match incoming {
                Ok(stream) => {
                    let peer_ip = stream
                        .peer_addr()
                        .map(|a| a.ip().to_string())
                        .unwrap_or_else(|_| "0.0.0.0".to_string());
                    if tx.send((stream, peer_ip)).is_err() {
                        break;
                    }
                }
                Err(e) => {
                    if e.kind() == std::io::ErrorKind::Interrupted {
                        continue;
                    }
                    eprintln!("accept error: {e}");
                }
            }
        }
        drop(tx);
        for h in handles {
            let _ = h.join();
        }
        Ok(())
    }
}

// ── Planificador Continuous Batch (--batch-size N) ──────────────────────────

struct JobQueue<T> {
    items: Mutex<VecDeque<T>>,
    cv: Condvar,
    shutdown: AtomicBool,
}

impl<T> JobQueue<T> {
    fn new() -> Self {
        Self {
            items: Mutex::new(VecDeque::new()),
            cv: Condvar::new(),
            shutdown: AtomicBool::new(false),
        }
    }

    fn push(&self, item: T) -> Result<(), T> {
        let mut q = self.items.lock().unwrap_or_else(|e| e.into_inner());
        if self.shutdown.load(Ordering::Acquire) {
            return Err(item);
        }
        q.push_back(item);
        self.cv.notify_one();
        Ok(())
    }

    fn pop(&self, block: bool) -> Option<T> {
        let mut q = self.items.lock().unwrap_or_else(|e| e.into_inner());
        if block {
            while q.is_empty() && !self.shutdown.load(Ordering::Acquire) {
                q = self.cv.wait(q).unwrap_or_else(|e| e.into_inner());
            }
        }
        q.pop_front()
    }

    fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Release);
        self.cv.notify_all();
    }
}

struct BatchHttpTag {
    stream: TcpStream,
    is_stream: bool,
    header_sent: bool,
    volume: f32,
    samples_sent: usize,
    t0: Instant,
}

struct SingleHttpJob {
    stream: TcpStream,
    is_stream: bool,
    vreq: ValidatedTtsRequest,
}

struct BatchSinkAdapter {
    jq: Arc<JobQueue<ContinuousSlotJob<BatchHttpTag>>>,
    running: Arc<AtomicBool>,
    admitted: usize,
    done: usize,
}

impl ContinuousBatchSink<BatchHttpTag> for BatchSinkAdapter {
    fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }

    fn next_job(&mut self, block: bool) -> Option<ContinuousSlotJob<BatchHttpTag>> {
        let j = self.jq.pop(block);
        if j.is_some() {
            self.admitted += 1;
        }
        j
    }

    fn on_chunk(&mut self, tag: &mut BatchHttpTag, samples: &[f32]) -> bool {
        if samples.is_empty() {
            return true;
        }
        if !tag.header_sent {
            if send_chunked_header(&mut tag.stream).is_err() {
                return false;
            }
            tag.header_sent = true;
        }
        if send_pcm_chunk(&mut tag.stream, samples, tag.volume).is_err() {
            return false;
        }
        tag.samples_sent += samples.len();
        true
    }

    fn on_done(
        &mut self,
        mut tag: BatchHttpTag,
        samples: Option<Vec<f32>>,
        err: Option<EngineError>,
    ) {
        let streamed = tag.is_stream;
        if streamed {
            if err.is_none() && tag.samples_sent > 0 {
                if !tag.header_sent && send_chunked_header(&mut tag.stream).is_ok() {
                    tag.header_sent = true;
                }
                let _ = send_chunked_end(&mut tag.stream);
                log_stream_sent(tag.samples_sent, tag.t0.elapsed().as_secs_f64());
            } else if matches!(err, Some(EngineError::Aborted))
                || socket_disconnected(&tag.stream)
            {
                eprintln!(
                    "[HTTP] TTS stream cancelado: el cliente se desconectó tras {:.1} s",
                    tag.t0.elapsed().as_secs_f64()
                );
            } else {
                eprintln!("[HTTP] TTS stream: la generación falló; la respuesta queda sin cierre");
            }
        } else if let Some(aud) = samples.filter(|a| !a.is_empty()) {
            let wav = encode_wav_bytes(&aud, SAMPLE_RATE as u32);
            if send_response(&mut tag.stream, 200, "audio/wav", &wav).is_ok() {
                log_wav_sent(wav.len(), aud.len(), tag.t0.elapsed().as_secs_f64());
            }
        } else if !matches!(err, Some(EngineError::Aborted)) && !socket_disconnected(&tag.stream) {
            send_error(&mut tag.stream, 500, "generation failed");
        }

        self.done += 1;
        eprintln!(
            "[BATCH] done #{} ({}, in-flight admitted={})",
            self.done,
            if streamed { "streamed" } else { "wav" },
            self.admitted
        );
    }
}

pub struct ContinuousBatchScheduler {
    pub runtime: Arc<ComputeRuntime>,
    pub model: Arc<Model06B>,
    pub voice: Arc<VoiceBindings>,
    pub max_batch: usize,
}

impl SynthesisScheduler for ContinuousBatchScheduler {
    fn run(self: Box<Self>, listener: TcpListener) -> Result<(), String> {
        let max_batch = self.max_batch.max(2);
        let n_readers = max_batch.clamp(2, 16);

        let (conn_tx, conn_rx) = mpsc::sync_channel::<(TcpStream, String)>(CONN_QUEUE_CAP);
        let conn_rx = Arc::new(Mutex::new(conn_rx));

        let jq = Arc::new(JobQueue::<ContinuousSlotJob<BatchHttpTag>>::new());
        let jq_single = Arc::new(JobQueue::<SingleHttpJob>::new());
        let running = Arc::new(AtomicBool::new(true));

        let (has_clone, def_spk, def_lang) = default_voice_context(&self.voice);

        let mut reader_handles = Vec::with_capacity(n_readers);
        for _ in 0..n_readers {
            let rx_c = Arc::clone(&conn_rx);
            let jq_c = Arc::clone(&jq);
            let jqs_c = Arc::clone(&jq_single);
            let voice_c = Arc::clone(&self.voice);
            let running_c = Arc::clone(&running);
            let h = thread::Builder::new()
                .name("qwen-http-reader".to_string())
                .spawn(move || loop {
                    let item = {
                        let g = rx_c.lock().unwrap_or_else(|e| e.into_inner());
                        g.recv()
                    };
                    let Ok((mut stream, peer_ip)) = item else {
                        break;
                    };
                    let req = match read_http_request(&mut stream) {
                        ReadHttpResult::Ok(req) => req,
                        ReadHttpResult::TooLarge => {
                            send_error(&mut stream, 400, "request body exceeds 1 MiB limit");
                            continue;
                        }
                        ReadHttpResult::InvalidContentLength => {
                            send_error(&mut stream, 400, "invalid Content-Length header");
                            continue;
                        }
                        ReadHttpResult::Empty => continue,
                    };
                    log_http_request(&req, &peer_ip);

                    if req.method == "OPTIONS" {
                        send_cors_options(&mut stream);
                        continue;
                    }
                    if req.method == "GET" && req.path == "/v1/health" {
                        handle_health(&mut stream);
                        continue;
                    }
                    if req.method == "GET" && req.path == "/v1/speakers" {
                        handle_speakers(&mut stream);
                        continue;
                    }
                    if req.method == "POST"
                        && matches!(
                            req.path.as_str(),
                            "/v1/tts" | "/v1/audio/speech" | "/v1/tts/stream"
                        )
                    {
                        let is_stream = req.path == "/v1/tts/stream";
                        let vreq = match ValidatedTtsRequest::parse(
                            &req.body,
                            is_stream,
                            has_clone,
                            def_spk,
                            def_lang,
                        ) {
                            Ok(v) => v,
                            Err(msg) => {
                                send_error(&mut stream, 400, &msg);
                                continue;
                            }
                        };

                        let seed = vreq.resolve_seed_or_time();
                        let prep = prepare_markup_06b(
                            &vreq.text,
                            &voice_c,
                            false,
                            if vreq.seed_set { Some(seed) } else { None },
                            if vreq.temp_set {
                                Some(vreq.temperature)
                            } else {
                                None
                            },
                            vreq.temperature,
                        );
                        let needs_single =
                            prep.use_compose || (vreq.rate - 1.0).abs() > 1e-6;
                        if needs_single {
                            if let Err(mut job) = jqs_c.push(SingleHttpJob {
                                stream,
                                is_stream,
                                vreq,
                            }) {
                                send_error(&mut job.stream, 503, "server shutting down");
                            }
                        } else {
                            if !running_c.load(Ordering::Acquire) {
                                send_error(
                                    &mut stream,
                                    503,
                                    "batch scheduler unavailable (startup failure)",
                                );
                                continue;
                            }
                            let effective_seed =
                                if !vreq.seed_set && prep.did_para_substitute {
                                    prep.seed
                                } else {
                                    seed
                                };
                            let cloned_sock = match stream.try_clone() {
                                Ok(s) => s,
                                Err(_) => continue,
                            };
                            let abort = AbortHandle::with_poll(move || {
                                socket_disconnected(&cloned_sock)
                            });
                            let breq = BatchReq {
                                text: prep.text,
                                speaker_id: vreq.speaker_id,
                                language_id: vreq.language_id,
                                temperature: prep.temperature,
                                top_k: vreq.top_k,
                                top_p: vreq.top_p,
                                rep_penalty: vreq.rep_penalty,
                                greedy_warmup: 0,
                                seed: effective_seed,
                                want_stream: is_stream,
                                chunk_frames: vreq.chunk_frames,
                                volume: vreq.volume,
                            };
                            let tag = BatchHttpTag {
                                stream,
                                is_stream,
                                header_sent: false,
                                volume: vreq.volume,
                                samples_sent: 0,
                                t0: Instant::now(),
                            };
                            if let Err(mut job) = jq_c.push(ContinuousSlotJob {
                                tag,
                                req: breq,
                                abort,
                            }) {
                                send_error(
                                    &mut job.tag.stream,
                                    503,
                                    "batch scheduler unavailable (startup failure)",
                                );
                            }
                        }
                    } else {
                        send_error(&mut stream, 404, "not found");
                    }
                })
                .map_err(|e| format!("failed to spawn reader: {e}"))?;
            reader_handles.push(h);
        }

        // Hilo fallback para trabajos individuales (compose inline / rate != 1.0)
        let single_rt = Arc::clone(&self.runtime);
        let single_model = Arc::clone(&self.model);
        let single_voice = Arc::clone(&self.voice);
        let jqs_worker = Arc::clone(&jq_single);
        let single_handle = thread::Builder::new()
            .name("qwen-http-single".to_string())
            .spawn(move || {
                let mut scratch = SessionScratch::new();
                while let Some(mut job) = jqs_worker.pop(true) {
                    if job.is_stream {
                        handle_tts_stream(
                            &single_rt,
                            &single_model,
                            &single_voice,
                            &mut scratch,
                            &mut job.stream,
                            &job.vreq,
                        );
                    } else {
                        handle_tts_full(
                            &single_rt,
                            &single_model,
                            &single_voice,
                            &mut scratch,
                            &mut job.stream,
                            &job.vreq,
                        );
                    }
                }
            })
            .map_err(|e| format!("failed to spawn single-job worker: {e}"))?;

        // Hilo planificador de continuous batching
        let sched_rt = Arc::clone(&self.runtime);
        let sched_model = Arc::clone(&self.model);
        let sched_voice = Arc::clone(&self.voice);
        let jq_sched = Arc::clone(&jq);
        let running_sched = Arc::clone(&running);
        let sched_handle = thread::Builder::new()
            .name("qwen-batch-sched".to_string())
            .spawn(move || {
                let mut adapter = BatchSinkAdapter {
                    jq: jq_sched.clone(),
                    running: running_sched.clone(),
                    admitted: 0,
                    done: 0,
                };
                if let Err(e) = run_continuous_batch(
                    &sched_rt,
                    &sched_model,
                    &sched_voice,
                    max_batch,
                    600,
                    &mut adapter,
                ) {
                    eprintln!("[BATCH] FATAL: continuous scheduler failed ({e}) — draining with 503");
                    running_sched.store(false, Ordering::Release);
                    jq_sched.shutdown();
                    while let Some(mut j) = jq_sched.pop(false) {
                        send_error(
                            &mut j.tag.stream,
                            503,
                            "batch scheduler unavailable (startup failure)",
                        );
                    }
                }
            })
            .map_err(|e| format!("failed to spawn batch scheduler: {e}"))?;

        for incoming in listener.incoming() {
            match incoming {
                Ok(stream) => {
                    let peer_ip = stream
                        .peer_addr()
                        .map(|a| a.ip().to_string())
                        .unwrap_or_else(|_| "0.0.0.0".to_string());
                    if conn_tx.send((stream, peer_ip)).is_err() {
                        break;
                    }
                }
                Err(e) => {
                    if e.kind() == std::io::ErrorKind::Interrupted {
                        continue;
                    }
                    eprintln!("accept error: {e}");
                }
            }
        }

        running.store(false, Ordering::Release);
        drop(conn_tx);
        for h in reader_handles {
            let _ = h.join();
        }
        jq.shutdown();
        jq_single.shutdown();
        let _ = sched_handle.join();
        let _ = single_handle.join();
        Ok(())
    }
}

pub struct ServeConfig {
    pub runtime: Arc<ComputeRuntime>,
    pub model: Arc<Model06B>,
    pub voice: Arc<VoiceBindings>,
    pub host: Ipv4Addr,
    pub port: u16,
    pub workers: usize,
    pub batch_size: usize,
    pub physical_cores: usize,
}

pub fn serve(cfg: ServeConfig) -> Result<(), String> {
    let ServeConfig {
        runtime,
        model,
        voice,
        host,
        port,
        workers,
        batch_size,
        physical_cores,
    } = cfg;
    let addr = SocketAddr::from((host, port));
    let listener = TcpListener::bind(addr)
        .map_err(|e| format!("Failed to bind http://{host}:{port}: {e}"))?;

    if batch_size >= 2 {
        let max_batch = batch_size.min(MAX_BATCH_SLOTS);
        let n_readers = max_batch.clamp(2, 16);
        eprintln!(
            "Server listening on http://{host}:{port} (continuous request-batching: max_batch={max_batch}, {n_readers} readers, +1 single-job fallback)"
        );
        eprintln!("Endpoints:");
        eprintln!("  POST /v1/tts          — generate speech (returns WAV, BATCHED)");
        eprintln!("  POST /v1/tts/stream   — generate speech (chunked PCM, BATCHED)");
        eprintln!("  POST /v1/audio/speech — OpenAI-compatible TTS (BATCHED)");
        eprintln!("  GET  /v1/speakers     — list speakers");
        eprintln!("  GET  /v1/health       — health check\n");
        eprintln!("Press Ctrl+C to stop.\n");

        let sched: Box<dyn SynthesisScheduler> = Box::new(ContinuousBatchScheduler {
            runtime,
            model,
            voice,
            max_batch,
        });
        sched.run(listener)
    } else {
        eprint!("Server listening on http://{host}:{port}");
        if workers > 1 {
            eprint!(" ({workers} workers)");
        }
        eprintln!("\nEndpoints:");
        eprintln!("  POST /v1/tts          — generate speech (returns WAV)");
        eprintln!("  POST /v1/tts/stream   — generate speech (chunked PCM stream)");
        eprintln!("  POST /v1/audio/speech — OpenAI-compatible TTS");
        eprintln!("  GET  /v1/speakers     — list speakers");
        eprintln!("  GET  /v1/health       — health check\n");
        eprintln!("Press Ctrl+C to stop.\n");

        let sched: Box<dyn SynthesisScheduler> = Box::new(WorkerPoolScheduler {
            runtime,
            model,
            voice,
            n_workers: workers,
            physical_cores,
        });
        sched.run(listener)
    }
}
