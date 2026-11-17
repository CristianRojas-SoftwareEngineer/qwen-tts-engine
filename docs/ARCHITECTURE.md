# Arquitectura de Qwen TTS Engine (`ARCHITECTURE.md`)

> Describe la topología del workspace Cargo, la política `unsafe`, la separación de estado en cuatro tiempos de vida y el modelo de concurrencia en CPU de **Qwen TTS Engine** (`qwen-tts-engine`), diseñado para cumplir [`INTERFACE_CONTRACT.md`](INTERFACE_CONTRACT.md), [`SPEC_06B.md`](SPEC_06B.md) y [`NUMERIC_CONTRACT.md`](NUMERIC_CONTRACT.md).

---

## 1. Grafo de crates (DAG estricto) y política `unsafe`

```text
   qwen-app          (Binario CLI + Servidor HTTP + Schedulers; afinidad SMT Win32)
      │
      ▼
  qwen-engine        (#![forbid(unsafe_code)] Modelo 0.6B, Talker, CP, Decoder, Encoder, Audio)
      │
      ▼
   qwen-core         (Constantes 0.6B, MmapStore, BPE, Sampler, .qvoice/.bin, MarkupPlanner)
      │
      ▼
  qwen-kernels       (AlignedBuf<64>, CpuCaps, ComputePool, FTZ, SIMD, Quant, FFI/Rust BLAS)
```

| Crate | Módulos principales | Política `unsafe` |
|---|---|---|
| **`qwen-kernels`** | `aligned`, `caps`, `pool`, `ftz`, `blas`, `quant`, `simd/{scalar,avx2}`, `dispatch`, `selftest`, `bench`. | **Permitido y aislado** (intrínsecos `AVX2+FMA`, `std::alloc`, `MXCSR`/`FPCR`, FFI BLAS / `faer`). |
| **`qwen-core`** | `config`, `error`, `mmap_store`, `tokenizer`, `sampling`, `qvoice`, `markup`. | **Solo `Mmap::map` y cast de slices alineadas** en `mmap_store.rs`; resto 100 % seguro. |
| **`qwen-engine`** | `model`, `talker`, `code_predictor`, `speech_decoder`, `speech_encoder`, `voice_clone`, `emotion`, `compose`, `batch`, `audio`, `pipeline`, `error`. | **`#![forbid(unsafe_code)]`** |
| **`qwen-app`** | `main`, `cli`, `affinity`, `build.rs` + manifiesto UTF-8, `server/{mod,http,validation,scheduler}`. | **Solo llamada puntual** a Win32 (`SetProcessAffinityMask` / `GetLogicalProcessorInformationEx`). |

---

## 2. Separación del estado en cuatro tiempos de vida

El diseño separa los datos en cuatro niveles de vida para compartir pesos inmutables entre hilos sin copias superficiales ni fugas de estado entre peticiones:

```text
Arc<Model06B>  [Inmutable, Send + Sync — compartido por todo el proceso]
 ├── _mmaps (en Bf16Weight/F32Ref)        // mantiene vivos los safetensors BF16/F32
 ├── tokenizer: BpeTokenizer
 ├── talker: TalkerWeights                // 28 capas (BF16 mmap o INT8/Q4_0) + embeddings
 ├── cp: CodePredictorWeights             // 5 capas + 15 lm_head + 15 codec_emb
 ├── decoder: SpeechDecoderWeights        // F32 + Arc<Codebooks> + W_k pre-reordenado
 └── encoder: Option<SpeechEncoderWeights>// ECAPA-TDNN + RVQ encoder (comparte Arc<Codebooks>)

Arc<VoiceBindings>  [Inmutable por voz — 0 B en preset, ~25 MB con WOVR]
 ├── identity: VoiceIdentity              // Preset(i32) | XVector(Vec<f32>) | IclGraft
 └── wovr / emo_ref_*                     // TextProjOverride y ancla prosódica opcionales
     (Al cargar WDLT en QVoice, load_voice_from_file muta Model06B antes de envolver en Arc.)

SessionScratch  [Mutable, exclusivo por worker o sesión — cero locks internos]
 ├── talker_kv / cp_kv                    // KV caches BF16 (Talker inicia en 1024 posiciones)
 ├── talker_bufs / cp_bufs                // AlignedBuf<64> + down_q2_rough opcional
 ├── sd_stream + sd_ws                    // colas/acarreos streaming + 8 ranuras DecoderWorkspace
 └── emb_lru + sampler + rng: LcgRng      // caché texto local + muestreador + RNG propio

RequestExecution  [Efímero por petición — se destruye al finalizar]
 ├── config: RequestConfig                // speaker_id, temperature, top_k, top_p, rep_penalty, ...
 └── abort: AbortHandle                   // Arc<AtomicBool> + poll opcional acoplado por RAII (Drop)
```

---

## 3. Runtime de cómputo (`qwen-kernels`)

1. **`ComputeRuntime`**: instancia inyectable (sin estado global) con cinco componentes:
   - `caps: CpuCaps` — capacidades SIMD detectadas en runtime.
   - `dispatch: &'static KernelDispatch` — tabla de punteros resuelta según `CpuCaps` *(ver [ADR-004](adr/004-simd-baseline.md))*. `qwen-engine` nunca contiene `#[cfg(target_arch)]`.
   - `pool: Arc<ComputePool>` — pool persistente para `Talker` y `CodePredictor` *(ver [ADR-003](adr/003-concurrency-pool.md))*.
   - `sd_pool: Arc<ComputePool>` — pool dedicado para `SpeechDecoder` (configurable con `QWEN_SD_THREADS`).
   - `blas: Arc<dyn BlasBackend>` — abstracción de `sgemm` y `set_threads` *(ver [ADR-002](adr/002-blas-strategy.md))*.
2. **`ComputePool`**: sincronización con contador atómico de chunks (`AtomicUsize`), presupuesto de *spin-wait* (`QWEN_POOL_SPIN=4096`) usando `std::hint::spin_loop()`, `Condvar` de respaldo y llamada única a `ftz_on()` al iniciar cada hilo trabajador.
3. **`BlasBackend`**: `FfiBlas` (`OpenBLAS` en Linux/Windows, `Accelerate` en macOS) por defecto; `PureRustBlas` (`faer`) disponible con `--features faer-blas` y seleccionable con `QWEN_BLAS=faer`.

---

## 4. Pipeline productor-consumidor y *streaming* (`qwen-engine`)

1. **Planificación pura (`MarkupPlanner`)**: `prepare_markup_06b` sustituye `[tags]` paralingüísticos y `MarkupPlanner::parse_spans` divide el texto en `Vec<SpanSpec>` (`Utterance` o `Pause`) sin estado global.
2. **Ejecución por `SpanSpec`**:
   - **Prefill**: siempre completo (`kv_len = 0`) vía `BlasBackend` usando `nt` hilos.
   - **Paso de generación**: fija `blas.set_threads(gen_blas)` antes de iniciar el hilo decodificador y lo mantiene constante hasta terminar el drenaje final.
   - **Paralelismo productor-consumidor (`std::thread::scope`)**:
     - **Hilo productor (`Talker + CodePredictor`)**: genera `[i32; 16]` por fotograma, comprueba `exec.abort.is_cancelled()` antes de cada paso y empuja los códigos a `DecoderQueue`.
     - **Hilo consumidor (`SpeechDecoder`)**: agrupa tramos según el calendario creciente (`2` en el primer fragmento y `min(chunk_frames, 4 + 2 * (k / 8))` para `k = 0, 1, 2, ...` en los siguientes), ejecuta `speech_decoder_decode_streaming` con `DecoderWorkspace` + tiling en `up2`/`up3` (`256 KB`) y entrega PCM a `AudioSink`.
     - Cualquier error de escritura en `AudioSink` activa `queue.aborted` de inmediato y `thread::scope` garantiza *join* limpio sin hilos huérfanos.

---

## 5. Servidor HTTP y *schedulers* (`qwen-app`)

1. **Validación única**: `ValidatedTtsRequest::parse(body, ...)` valida todas las peticiones antes de encolar (HTTP `400` ante cualquier parámetro fuera de contrato) y `AbortGuard` acopla el ciclo de vida del socket a `AbortHandle::cancel()` vía `Drop`.
2. **Trait `SynthesisScheduler`** con dos estrategias tras el mismo contrato:
   - **`ContinuousBatchScheduler` (`--batch-size N`, recomendado en servidor)**: un hilo planificador (`run_continuous_batch`) comparte un único `ComputePool` y `BatchWorkspace` *ragged* para procesar hasta `N` peticiones simultáneas (WAV y streaming), derivando peticiones con markup multi-span o `rate != 1.0` a un hilo *fallback* dedicado.
   - **`WorkerPoolScheduler` (`--workers N`)**: `N` hilos *worker* con `SessionScratch` y `ComputeRuntime` propios; divide el presupuesto de hilos por *worker* y desactiva el *spin-wait* (`spin_budget = 0`) cuando hay sobresuscripción frente a núcleos físicos.
