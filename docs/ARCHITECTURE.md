# Arquitectura de Qwen TTS Engine (`ARCHITECTURE.md`)

> Describe la topología del workspace Cargo, la política `unsafe`, la separación de estado en tiempos de vida, el camino único de síntesis y el modelo de concurrencia en CPU de **Qwen TTS Engine** (`qwen-tts-engine`), diseñado para cumplir [`INTERFACE_CONTRACT.md`](INTERFACE_CONTRACT.md), [`SPEC_06B.md`](SPEC_06B.md) y [`NUMERIC_CONTRACT.md`](NUMERIC_CONTRACT.md).

---

## 1. Grafo de crates (DAG estricto) y política `unsafe`

```text
   qwen-app          (Binario CLI + Servidor HTTP; afinidad SMT Win32)
      │
      ▼
  qwen-engine        (#![forbid(unsafe_code)] Modelo 0.6B, Talker, CP, Decoder, Encoder, Audio)
      │
      ▼
   qwen-core         (Constantes 0.6B, MmapStore, BPE, Sampler, .qvoice v4, planificación de marcado)
      │
      ▼
  qwen-kernels       (AlignedBuf<64>, CpuCaps, ComputePool, FTZ, SIMD, Quant, FFI/Rust BLAS)
```

| Crate | Módulos principales | Política `unsafe` |
|---|---|---|
| **`qwen-kernels`** | `aligned`, `caps`, `pool`, `ftz`, `blas`, `quant`, `simd/{scalar,avx2,neon}` *(internos del crate)*, `dispatch`, `selftest`, `bench`. | **Permitido y aislado** (intrínsecos `AVX2+FMA` y NEON, `asm!` para `SDOT`, `std::alloc`, `MXCSR`/`FPCR`, FFI BLAS / `faer`); la API segura `KernelDispatch` valida límites antes de despachar. |
| **`qwen-core`** | `config`, `error`, `mmap_store`, `tokenizer`, `sampling`, `qvoice`, `markup`. | **Solo `Mmap::map` y cast de slices alineadas** en `mmap_store.rs`; resto 100 % seguro. |
| **`qwen-engine`** | `model`, `talker`, `code_predictor`, `speech_decoder`, `speech_encoder`, `voice`, `voice_clone`, `request`, `prompt`, `synth`, `compose`, `audio`, `pipeline`, `error`. | **`#![forbid(unsafe_code)]`** |
| **`qwen-app`** | `main`, `cli`, `request`, `affinity`, `build.rs` + manifiesto UTF-8, `modes/{mod,single,serve,voice_cmd,diag}`, `server/{mod,http,validation}`. | **Solo llamada puntual** a Win32 (`SetProcessAffinityMask` / `GetLogicalProcessorInformationEx`). |

---

## 2. Separación del estado en tiempos de vida

El diseño separa los datos en niveles de vida para compartir pesos inmutables entre hilos sin copias superficiales ni fugas de estado entre peticiones:

```text
Arc<Model06B>  [Inmutable, Send + Sync — compartido por todo el proceso]
 ├── Bf16Weight::Mmap / F32Ref::Mmap      // Arc<Mmap> manteniendo vivos los safetensors
 ├── tokenizer: BpeTokenizer
 ├── talker: TalkerWeights                // 28 capas (BF16 mmap o INT8/Q4_0) + embeddings
 ├── cp: CodePredictorWeights             // 5 capas + 15 lm_head + 15 codec_emb
 ├── decoder: SpeechDecoderWeights        // F32 + Arc<Codebooks> + W_k pre-reordenado
 ├── speaker_encoder: Arc<SpeakerEncoderWeights>         // ECAPA-TDNN (0.6B-Base)
 └── speech_encoder: Arc<SpeechEncoderWeights>           // RVQ encoder (comparte Arc<Codebooks>)

Voice  [Inmutable por voz — unos pocos KB; la voz `default` va embebida en el binario]
 ├── name: String                         // `default`, el nombre guardado en el `.qvoice` o el del archivo
 ├── language_id: Option<i32>             // idioma declarado (`default` declara español)
 ├── xvector: Vec<f32>                    // x-vector de 1024 valores
 └── icl: Option<IclPrompt>               // { ref_text, ref_codes }, ya recortado al tope de frames
     (`Voice::load` solo lee el `.qvoice`; no modifica `Model06B`.)

Synthesizer<'a>  [Efímero, `Copy`: referencias a los tres niveles anteriores]
 └── { runtime: &ComputeRuntime, model: &Model06B, voice: &Voice }   // el contexto fijo de una sesión

SessionScratch  [Mutable, exclusivo por sesión — cero locks internos]
 ├── talker_kv / cp_kv                    // cachés KV BF16 (Talker inicia en KV_INITIAL_POSITIONS = 256 y duplica bajo demanda)
 ├── talker_bufs / cp_bufs                // AlignedBuf<64> + down_q2_rough opcional (el staging F32 de pesos del prefill es local y transitorio de talker_prefill)
 ├── sd_stream + sd_ws                    // colas/acarreos streaming + 8 slots DecoderWorkspace
 └── emb_lru + sampler + rng: LcgRng      // caché texto local + muestreador + RNG propio

SynthesisRequest  [Efímero por síntesis — plano, `Clone`]
 ├── language_id: i32                     // ya resuelto: pedido, voz o español
 ├── sampling: SamplingParams             // temperature (0.325), top_k, top_p, rep_penalty, greedy_warmup, seed (4)
 └── max_frames, chunk_frames, cp_roughness, volume, rate, compose_pause_s, tail_trim, onset_fade_ms, log

AbortHandle  [Efímero por síntesis]      // Arc<AtomicBool> + poll opcional acoplado por RAII (Drop); es un parámetro, no parte de la petición
```

---

## 3. Runtime de cómputo (`qwen-kernels`)

1. **`ComputeRuntime`**: instancia inyectable (sin estado global) con cinco componentes:
   - `caps: CpuCaps` — capacidades SIMD detectadas en runtime.
   - `dispatch: &'static KernelDispatch` — tabla de punteros resuelta según `CpuCaps` *(ver [ADR-004](adr/004-simd-baseline.md))*. `qwen-engine` nunca contiene `#[cfg(target_arch)]`.
   - `pool: Arc<ComputePool>` — pool persistente para `Talker` y `CodePredictor` *(ver [ADR-003](adr/003-concurrency-pool.md))*.
   - `sd_pool: Arc<ComputePool>` — pool dedicado para `SpeechDecoder` (configurable con `QWEN_SD_THREADS`).
   - `blas: Arc<dyn BlasBackend>` — abstracción de `sgemm` y `set_threads` *(ver [ADR-002](adr/002-blas-strategy.md))*.
2. **`ComputePool`**: sincronización con contador atómico de chunks (`AtomicUsize`), presupuesto de *spin-wait* (`QWEN_POOL_SPIN=4096`, o `0` si los hilos del productor ocupan todos los núcleos físicos: `spin_budget_for`) usando `std::hint::spin_loop()`, `Condvar` de respaldo y llamada única a `ftz_on()` al iniciar cada hilo trabajador.
   Si una tarea entra en pánico en un hilo trabajador, `parallel` lo relanza en quien llama y el trabajador sigue vivo.
3. **`BlasBackend`**: `FfiBlas` (`OpenBLAS` en Linux/Windows, `Accelerate` en macOS; en Linux, estático y con `DYNAMIC_ARCH` si se compila con `scripts/build-openblas.sh`) por defecto; `PureRustBlas` (`faer`) disponible con `--features faer-blas` y seleccionable con `QWEN_BLAS=faer`.

---

## 4. Pipeline productor-consumidor y *streaming* (`qwen-engine`)

1. **Un solo camino de síntesis** (`Synthesizer::synthesize`): dos ejes independientes, el guion y la entrega.
   - **`Script`** (`qwen-core`): `Plain(String)` o `Spans(Vec<SpanSpec>)`. `plan_script(texto, MarkupMode)` es el único sitio que sustituye los `[tags]` paralingüísticos y divide el texto en tramos (`Utterance` o `Pause`), sin estado global.
   - **`Delivery`**: `Buffer` (todo el audio en memoria, con `time_stretch`, `tail_trim` y `onset_fade`) o `Stream(&mut dyn AudioSink)` (chunks a un destino, sin acabado).
   - Las cuatro combinaciones (`Plain`/`Spans` por `Buffer`/`Stream`) comparten el mismo generador, y `Buffer` y `Stream` dan el mismo PCM (salvo el acabado de la entrega completa). `Plain` es un bucle productor-consumidor; `Spans` sintetiza cada tramo como `Plain` con ganancia `x·(v·o)` y los une con pausas y fundido cruzado de 45 ms (`compose.rs`).
   - **`AudioSink`** es el único destino de audio; `EdgeRampSink<W: PcmWrite>` hace solo rampas de borde y retención, y los escritores (`WavFileWriter`, `StdoutWriter`, `Vec<u8>` y el HTTP *chunked* de `qwen-app`) solo escriben bytes. La ganancia la aplica `synthesize` justo antes del destino.
   - **Una regla de idioma** (`effective_language`): pedido, voz o español. La preparación de la petición está en `qwen-app/src/request.rs` (`from_cli` y `from_http`), el único sitio que aplica `plan_script`, `resolve_sampling` y `effective_language`.
2. **Ejecución por tramo**:
   - **Prefill**: siempre completo (`kv_len = 0`) vía `BlasBackend` usando `nt` hilos.
   - **Paso de generación**: fija `blas.set_threads(gen_blas)` antes de iniciar el hilo decoder y lo mantiene constante hasta terminar el drenaje final.
   - **Paralelismo productor-consumidor (`std::thread::scope`)**:
     - **Hilo productor (`Talker + CodePredictor`)**: genera `[i32; 16]` por frame, comprueba `abort.is_cancelled()` antes de cada paso y empuja los códigos a `DecoderQueue`.
     - **Hilo consumidor (`SpeechDecoder`)**: agrupa los frames en chunks según el calendario creciente (`2` en el primer chunk y `min(chunk_frames, 4 + 2 * (k / 8))` para `k = 0, 1, 2, ...` en los siguientes), ejecuta `speech_decoder_decode_streaming` con `DecoderWorkspace` + tiling en `up2`/`up3` (`256 KB`) y la misma atención con ventana deslizante que la decodificación completa (`causal_attention_windowed` sobre la caché KV compactable), y entrega PCM a `AudioSink`.
     - Cualquier error de escritura en `AudioSink` activa `queue.aborted` de inmediato y `thread::scope` garantiza *join* limpio sin hilos huérfanos.

---

## 5. Aplicación (`qwen-app`)

1. **Modos**: `cli.rs` resuelve una sola vez el `CliMode` (`Caps`, `SelfTest`, `MatmatBench`, `ListVoices`, `DeleteVoice`, `CreateVoice`, `Serve`, `Synthesize` o `NoJob`) según la precedencia de los flags, y `main.rs` solo despacha. `modes/diag.rs` y `modes/voice_cmd.rs` no cargan el modelo; `modes/mod.rs` prepara runtime, modelo y voz para `modes/single.rs` (una síntesis, `--seed-audition` incluido) y `modes/serve.rs`.
2. **Servidor simple**: `ServerContext { runtime, model, voice }` y `serve(ctx, host, port)`. Usa el `ComputeRuntime` global, calienta una sesión (`warm_up`, con un `SynthesisRequest` directo, sin JSON) y atiende cada conexión en línea (`accept` + `handle_connection`), con un único trabajador, sin hilos ni cola. El presupuesto de *spin-wait* lo fija `ComputeRuntime::for_cores` según los núcleos físicos.
3. **Un solo enrutado**: `handle_connection` atiende `/v1/health`, `/v1/voices`, `/v1/tts`, `/v1/tts/stream` y `/v1/audio/speech`. `ValidatedTtsRequest::parse` valida todas las peticiones (HTTP `400` ante cualquier parámetro fuera de contrato), `from_http` las convierte en `SynthesisRequest` y `Script`, y `AbortGuard` acopla el ciclo de vida del socket a `AbortHandle::cancel()` vía `Drop`.

---

## 6. Rendimiento

El tiempo de una síntesis es `prefill + max(productor, consumidor) + drenaje final`:

- **Productor** (`Talker + CodePredictor`, en `pool`): debe generar cada frame en menos de 80 ms para que el *streaming* no se corte. Lo limita el ancho de banda de memoria: cada frame lee ~1 GiB de pesos cuantizados.
- **Consumidor** (`SpeechDecoder`, en `sd_pool` y BLAS): casi todo su tiempo está en las convoluciones de los bloques de *upsample* (`sgemm`).
- **Primer audio**: lo gobierna el prefill, y después el primer chunk del decoder (2 frames).

Con 4 núcleos físicos, productor y consumidor se reparten las mismas CPU y ninguna plataforma llega a tiempo real. Las optimizaciones adoptadas y descartadas, con su efecto medido, y el rendimiento por plataforma están en [ADR-005](adr/005-cpu-inference-optimizations.md).
