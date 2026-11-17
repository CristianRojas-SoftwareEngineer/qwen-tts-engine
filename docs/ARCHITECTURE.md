# Arquitectura del Motor en Rust (`qwen-tts-rs`, actualmente `0.6B`)

Define la estructura del workspace Cargo, la separación de responsabilidades y el modelo de concurrencia del nuevo motor en Rust, diseñado para cumplir [INTERFACE_CONTRACT.md](INTERFACE_CONTRACT.md), [SPEC_06B.md](SPEC_06B.md) y [NUMERIC_CONTRACT.md](NUMERIC_CONTRACT.md).

---

## 1. Grafo de Crates (DAG estricto) y Política `unsafe`

```text
   qwen-app          (Binario CLI + Servidor HTTP + Schedulers; puntual Win32 affinity)
      │
      ▼
  qwen-engine        (#![forbid(unsafe_code)] Modelo 0.6B, Talker, CP, Decoder, Encoder, Audio)
      │
      ▼
   qwen-core         (Constantes 0.6B, MmapStore, BPE, Sampler, .qvoice/.bin, MarkupPlanner)
      │
      ▼
  qwen-kernels       (AlignedBuf<64>, CpuCaps, ComputePool, FTZ, SIMD, Quant, FFI BLAS)
```

| Crate | Módulos principales | Política `unsafe` |
|---|---|---|
| **`qwen-kernels`** | `aligned` (`AlignedBuf<64>`), `caps`, `pool` (`ComputePool`), `ftz`, `blas` (`FfiBlas` default + `PureRustBlas` opcional), `quant` (`int8`/`q4_0`/`q2_0`), `simd/{scalar,avx2}`, `dispatch`, `selftest`, `bench`. | **Permitido y aislado** (intrínsecos SIMD, `std::alloc`, `MXCSR`/`FPCR`, FFI BLAS). |
| **`qwen-core`** | `config`, `mmap_store` (`memmap2`), `tokenizer` (`BpeTokenizer`), `sampling` (`Sampler`/`LcgRng`), `qvoice` (`QVCE`/`WOVR`/`WDELTA`), `markup` (`MarkupPlanner`). | **Solo `Mmap::map` y cast de slices alineadas** en `mmap_store.rs`; resto 100 % seguro. |
| **`qwen-engine`** | `model` (`Model06B`, `VoiceBindings`, `SessionScratch`), `talker`, `code_predictor`, `speech_decoder`, `speech_encoder`, `voice_clone`, `emotion`, `compose`, `batch`, `audio`, `pipeline`. | **`#![forbid(unsafe_code)]`** |
| **`qwen-app`** | `main`, `cli`, `affinity` (SMT Win32), `build.rs` + `qwen_app.exe.manifest`, `server/{mod,http,validation,scheduler}`. | **Solo llamada puntual** `SetProcessAffinityMask` / `GetLogicalProcessorInformationEx` en Windows. |

---

## 2. Separación del Estado en 4 Tiempos de Vida

Separa el estado de inferencia en 4 tiempos de vida para compartir `Model06B` entre workers sin copias superficiales ni fugas de estado entre peticiones:

```text
Arc<Model06B>  [Inmutable, Send + Sync — compartido por todo el proceso]
 ├── _mmaps: Vec<Mmap>                    // mantiene vivos safetensors BF16/F32
 ├── tokenizer: BpeTokenizer
 ├── talker: TalkerWeights                // 28 capas (BF16 mmap o INT8/Q4_0) + embeddings
 ├── cp: CodePredictorWeights             // 5 capas + 15 lm_head + 15 codec_emb
 ├── decoder: SpeechDecoderWeights        // F32 + Arc<Codebooks> + W_k pre-reordenado
 └── encoder: Option<SpeechEncoderWeights>// ECAPA-TDNN + RVQ encoder (comparte Arc<Codebooks>)

Arc<VoiceBindings>  [Inmutable por voz — 0 B en preset, ~25 MB con WOVR]
 ├── identity: VoiceIdentity              // Preset(SpeakerId) | XVector([f32; 1024]) | IclGraft
 └── bindings: TextAndCodecView           // Borrowed(&Model06B) | Overridden(text_proj, codec_emb[0..2048], special_embeds)
     (Si se carga WDELTA pesado, se construye un Model06B derivado al arranque sin mutar el base.)

SessionScratch  [Mutable, exclusivo por worker o sesión — cero locks internos]
 ├── talker_kv / cp_kv                    // BF16 KV caches (Talker inicia en 1024 pos)
 ├── talker_bufs / cp_bufs                // AlignedBuf<64> + down_q2_rough opcional
 ├── sd_stream + sd_ws                    // colas/acarreos streaming + 8 ranuras DecoderWorkspace
 └── emb_lru + sampler + rng: LcgRng      // caché texto local + muestreador + RNG propio

RequestExecution  [Efímero por petición — se destruye al finalizar]
 ├── config: RequestConfig                // speaker_id, temperature, top_k, top_p, rep_penalty, chunk_frames, rate, volume
 └── abort: AbortHandle                   // Arc<AtomicBool> + poll de socket opcional acoplado por RAII (Drop) al socket HTTP
```

---

## 3. Abstracciones de Cómputo (`qwen-kernels`)

1. **`ComputeRuntime`**: inyectable (no singleton global), compuesto por `pool: Arc<ComputePool>` y `blas: Arc<dyn BlasBackend>`.
2. **`ComputePool`** *(ver ADR-003)*: pool persistente instanciable con contador atómico de chunks (`AtomicUsize`), presupuesto de *spin-wait* (`QWEN_POOL_SPIN=4096`) con `std::hint::spin_loop()`, `Condvar` y `ftz_on()` al iniciar cada trabajador.
3. **`KernelDispatch`**: tabla resuelta una vez en tiempo de ejecución según `CpuCaps` *(ver ADR-004)* para `matvec_{bf16,int8,q4_0,q2_0}`, `matmat_*`, `argmax_matvec_*`, `rms_norm*`, `causal_attention*`, `rope_neox_inplace`, `swiglu_inplace` y `snake_row`. `qwen-engine` nunca contiene `#[cfg(target_arch)]`.
4. **`BlasBackend`** *(ver ADR-002)*: trait que abstrae `sgemm` y `set_threads(n)`. Por defecto usa `FfiBlas` (`OpenBLAS` / `Accelerate`), con `PureRustBlas` (`faer`) disponible bajo la feature opcional `faer-blas` (`QWEN_BLAS=faer`). Respeta `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS`.

---

## 4. Pipeline Productor-Consumidor y Streaming (`qwen-engine`)

1. **Planificación pura (`MarkupPlanner`)**: transforma el texto y parámetros de paralingüística `0.6B` en `Vec<SpanSpec>` (`Utterance` o `Pause`) sin mutar ningún contexto global.
2. **Ejecución por `SpanSpec`**:
   - **Prefill**: siempre completo (`kv_len = 0`) vía `BlasBackend` usando `threads = -j`.
   - **Paso de generación**: fija `blas.set_threads(gen_blas)` antes de iniciar el hilo decodificador y lo mantiene constante hasta drenar.
   - **Paralelismo de pipeline (`std::thread::scope`)**:
     - Hilo productor (`Talker + CodePredictor`) genera `[i32; 16]` por fotograma, comprueba `abort.is_cancelled()` antes de cada paso y envía al canal acotado.
     - Hilo consumidor (`SpeechDecoder`) agrupa tramos según el calendario creciente (`2, 4, 6, 8, 10` hasta `chunk_frames`), ejecuta `conv_decoder_forward` con `DecoderWorkspace` + tiling `up2`/`up3` (256 KB) y entrega PCM a `AudioSink` (`WavFileSink`, `StdoutRawSink`, `HttpChunkSink`, `MemorySink`).
     - Un error de escritura en `AudioSink` activa `abort` inmediatamente y `thread::scope` garantiza *join* limpio sin hilos huérfanos.

---

## 5. Servidor HTTP y Schedulers (`qwen-app`)

1. **Frontend único**: valida todas las peticiones en `ValidatedTtsRequest::parse(body)` (HTTP `400` ante cualquier error, ver [INTERFACE_CONTRACT.md](INTERFACE_CONTRACT.md)) y asocia un `AbortGuard` (`Drop` → `abort.cancel()`) a la conexión abierta.
2. **Trait `SynthesisScheduler`** con dos estrategias tras el mismo contrato:
   - **`ContinuousBatchScheduler` (`--batch-size N`, recomendado)**: un único hilo planificador con `1` `ComputePool` compartido y `BatchWorkspace` ragged; lee los pesos una vez para hasta `N` peticiones simultáneas (WAV y streaming), derivando peticiones con markup multi-span o `rate != 1.0` a un worker dedicado.
   - **`WorkerPoolScheduler` (`--workers N`)**: `N` workers con `SessionScratch` propios; calcula el presupuesto de hilos por worker y ajusta el *spin-wait* para no exceder los núcleos físicos del host.
