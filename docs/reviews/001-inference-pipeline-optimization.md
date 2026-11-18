# Revisión de Arquitectura y Plan de Optimización del Pipeline de Inferencia (`0.6B`)

> **Propósito**: Documento técnico autocontenido sobre la arquitectura de inferencia CPU de `qwen-tts-engine` (especializado en **Qwen3-TTS 0.6B**), el análisis de sus cuellos de botella en **CPU**, **memoria RAM** y **ancho de banda de memoria**, el impacto real esperado sobre el **RTF** (*Real-Time Factor*) y una **planificación progresiva de implementación organizada en Fases y Pasos** verificables.
>
> **Referencias normativas**: [`ARCHITECTURE.md`](../ARCHITECTURE.md), [`SPEC_06B.md`](../SPEC_06B.md), [`NUMERIC_CONTRACT.md`](../NUMERIC_CONTRACT.md), [`VERIFICATION_PLAN.md`](../VERIFICATION_PLAN.md) y ADRs `001`–`004`.
>
> **Fecha**: 2026-11-17

> **Evidencia de medición (local, no versionada)**: los valores citados proceden de `.work/plans/optimizations/phase-0/estado.md` (mediciones y Anexo A) y de `.work/plans/optimizations/00-estado.md` (tablero acumulado). `.work/` está ignorado por Git: las rutas se indican como texto para no dejar enlaces rotos en el repositorio publicado.

---

## 1. Resumen ejecutivo y contexto

**Qwen TTS Engine** (`qwen-tts-engine`) ejecuta síntesis de texto a voz a `24 kHz` en CPU sin aceleración GPU ([`SPEC_06B.md`](../SPEC_06B.md)). En su configuración de producción recomendada (`--quant-mixed`, que combina `Q4_0` en el Talker y FFN del Code Predictor con `INT8` en atención y cabezas del Code Predictor), la línea base medida en el hardware objetivo con `4` hilos y `OpenBLAS` es `RTF 0.82` en frase corta y `0.72` en frase larga (`--seed 4 --quant-mixed -j 4 --stream`), registrada en `.work/plans/optimizations/phase-0/estado.md`. La estrategia BLAS se documenta en [ADR-002](../adr/002-blas-strategy.md).

> **Definiciones operativas**:
> - **Fotograma (*frame*)**: unidad de `16` códigos acústicos (`code0` del Talker + `codes[1..15]` del Code Predictor) equivalente a `80 ms` de audio (`FRAME_RATE = 12.5 Hz`, `HOP_SAMPLES = 1920` muestras a `24 kHz`).
> - **RTF (*Real-Time Factor*)**: `tiempo_total_proceso / duración_audio_generado`. Valores `< 1.0` indican síntesis más rápida que el tiempo real.
> - **TTFA (*Time To First Audio*)**: latencia desde el inicio de la petición hasta la emisión del primer fragmento PCM en modo streaming.

El diseño actual ya incorpora optimizaciones clave para CPU: cuantización `Q4_0`/`INT8` con intrínsecos `AVX2+FMA`, pesos `safetensors` inmutables sobre `mmap` compartidos vía `Arc<Model06B>`, activación `FTZ/DAZ` (`MXCSR`/`FPCR`), un pool de hilos propio (`ComputePool`) y *tiling* L2 de `256 KB` en el vocoder. Sin embargo, el análisis estático del código fuente y del grafo de llamadas identifica **12 oportunidades concretas** agrupadas en tres frentes:

1. **Ancho de banda de memoria (cuello de botella principal)**:
   - `int8_matmat_slice` y `q4_0_matmat_slice` (`crates/qwen-kernels/src/simd/avx2.rs`) iteran `for b in 0..b_dim` llamando a `matvec`, releyendo los mismos pesos `b_dim` veces (impacta a `cp_prefill2` con `B=2` en cada fotograma y a todo `ContinuousBatchScheduler`).
   - `batch_proj_q` (`crates/qwen-engine/src/batch.rs`) realiza una doble transposición (`batch_gather` y `quantize_act_int8_cols`) más `batch_scatter` (solo en el camino batch: fuera de alcance, ver [`ROADMAP.md`](../../ROADMAP.md)).
   - `talker_prefill` (`crates/qwen-engine/src/talker.rs`) expande las 28 capas `BF16` a `F32` (`60.0 MiB`/capa = `3.36 GiB` de escritura+relectura extra en todo el prefill) y provoca el *page-in* de `840 MiB` `BF16` desde disco aunque el modelo corra cuantizado.
   - `talker.codec_head_bf16` (`3072 × 1024`, `6.0 MiB`) permanece en `BF16` sin cuantizar.

2. **Consumo de CPU (rutas escalares en *hot paths* y exceso de barreras)**:
   - La atención causal con ventana deslizante en `speech_decoder_decode_streaming` (`crates/qwen-engine/src/speech_decoder.rs:1679-1723`) corre en bucles escalares puros en lugar de reutilizar `KernelDispatch::causal_attention_windowed` (AVX2+FMA).
   - `swiglu_inplace`, `silu` (`crates/qwen-kernels/src/dispatch.rs:1204-1224`), `convnext_mlp` (`erff` escalar) y varias convoluciones del vocoder (`cs_dwconv`, `final_conv`) carecen de vectorización SIMD, acumulando **~331,776 llamadas escalares a `expf()` por fotograma** (~4.15M/s a RTF 1.0) solo en Talker + Code Predictor.
   - Se despachan hasta **466 barreras `ComputePool::parallel` por fotograma**, se invoca `ftz_on()` en cada `matvec_*`, `sd_pool` hace *spin-wait* concurrente con el productor y `batch_talker_step_impl` lanza `b_dim` barreras de atención por capa sin compactar slots inactivos.

3. **Consumo de memoria RAM**:
   - Cada `SessionScratch` preasigna **~181.6 MiB** al crearse (`112.0 MiB` de `TalkerKvCache` para `1024` posiciones + `60.0 MiB` de `pref_*_f32` + `8.0 MiB` de `EmbLruCache`), lo que suma **~726 MiB** con `--workers 4` antes de procesar la primera petición.
   - `BatchWorkspace` reserva **`122.08 MiB`/slot** contiguos (`~976.6 MiB` para `B=8`) y `generate_batch_multi` clonaba `TalkerKvCache` completos en un snapshot antes de copiar solo el prompt (**`112 MiB`/petición** pre-KV inicial de 256 posiciones; **`28,0 MiB`/petición** post-KV inicial de 256 posiciones con `KV_INITIAL_POSITIONS=256`; el snapshot completo se eliminó en snapshot compacto de KV, Fase 1 Paso 1.2).

---

## 2. Arquitectura e interacciones del pipeline de inferencia

### 2.1 Grafo de crates y aislamiento de responsabilidades

El workspace sigue un DAG estricto en cuatro capas:

| Crate | Responsabilidad en inferencia | Símbolos principales |
|---|---|---|
| `qwen-app` | CLI, afinidad SMT Win32 (`affinity.rs`), servidor HTTP, *schedulers* (`WorkerPoolScheduler`, `ContinuousBatchScheduler`) | `pin_one_thread_per_core`, `serve` |
| `qwen-engine` | Motor (`#![forbid(unsafe_code)]`) y separación de estado en 4 tiempos de vida (`Model06B`, `VoiceBindings`, `SessionScratch`, `RequestExecution`) | `synthesize_to_sink`, `talker_prefill`, `talker_step`, `cp_predict`, `speech_decoder_decode_streaming`, `run_continuous_batch` |
| `qwen-core` | Constantes `0.6B` (`config.rs`), `MmapStore` sin copias, `BpeTokenizer`, `Sampler` (top-k/top-p/rep-penalty + LCG), `.qvoice`, `MarkupPlanner` | `MmapStore`, `Sampler::sample`, `BpeTokenizer::encode_para` |
| `qwen-kernels` | Despacho SIMD en runtime (`CpuCaps`), kernels `AVX2+FMA` y escalar, cuantización, `ComputePool`, `BlasBackend`, `AlignedBuf<T, 64>` | `ComputeRuntime`, `KernelDispatch`, `ComputePool::parallel`, `FfiBlas`, `PureRustBlas` |

### 2.2 Flujo de ejecución y sincronización entre componentes

```text
Texto ──► MarkupPlanner (spans) ──► BpeTokenizer ──► build_prefill_embeds (EmbLruCache + text_proj BF16)
                                                               │
                                                               ▼
                                          talker_prefill (28 capas, BF16->F32 + BlasBackend::sgemm)
                                                               │ escribe TalkerKvCache (BF16)
                                                               ▼
     ┌───────────────────────── std::thread::scope (Productor / Consumidor) ─────────────────────────┐
     │                                                                                               │
     │  HILO PRODUCTOR (12.5 Hz = 1 fotograma / 80 ms)                                               │
     │   1. codec_head_bf16 (3072x1024) ──► Sampler ──► code0                                        │
     │   2. cp_predict (5 capas): cp_prefill2 (B=2) + 14 pasos autorregresivos ──► codes[1..15]      │
     │   3. DecoderQueue::push_frame([i32; 16])  ──(Mutex + Condvar)──►                              │
     │   4. build_step_embed (suma 16 codebooks + tts_pad)                                           │
     │   5. talker_step (28 capas Q4_0/INT8/BF16 + GQA atención BF16-KV)                              │
     │                                                                                               │
     │  HILO CONSUMIDOR (rampa 2, 4, 6... hasta chunk_frames=10)                                     │
     │   speech_decoder_decode_streaming (F32):                                                      │
     │     RVQ codebooks ──► pre_conv ──► Pre-Transformer (8 capas, ventana 72) ──►                  │
     │     ConvNeXt (2x2) ──► initial_conv ──► 4x Upsample (8x5x4x3) + ResBlocks (tiling 256 KB)     │
     │     ──► final_conv ──► PCM f32 24 kHz ──► AudioSink                                           │
     └───────────────────────────────────────────────────────────────────────────────────────────────┘
```

1. **Preparación (`build_prefill_embeds`, `pipeline.rs:213-454`)**: tokeniza texto (`encode_para`) y referencia ICL opcional. Resuelve embeddings de texto mediante `EmbLruCache` (`2048 × 1024` floats) llamando a `compute_text_proj_token` (`text_embedding_bf16[151936, 2048]` → `fc1` → `silu` → `fc2`).
2. **Prefill (`talker_prefill`, `talker.rs:15-279`)**: fija `blas.set_threads(nt)` y recorre `28` capas expandiendo pesos `BF16` a `F32`, llamando `6` veces a `sgemm` por capa, aplicando `rms_norm_per_head` + RoPE NeoX y escribiendo `TalkerKvCache` (`BF16`).
3. **Generación paralela (`synthesize_to_sink`, `pipeline.rs:556-825`)**: fija `blas.set_threads(max(1, nt - 1))` y ejecuta productor (`Talker + CodePredictor` sobre `runtime.pool`) en paralelo con el consumidor (`SpeechDecoder` sobre `runtime.blas` + `runtime.sd_pool`).
4. **Modo servidor (`qwen-app/src/server/scheduler.rs`)**:
   - `ContinuousBatchScheduler` (`--batch-size N`): comparte un único `ComputePool` y `BatchWorkspace` *ragged* sobre `run_continuous_batch`, derivando markup multi-span o `rate != 1.0` a un hilo *fallback* dedicado.
   - `WorkerPoolScheduler` (`--workers N`): `N` hilos worker independientes con su propio `ComputeRuntime` y `SessionScratch`.

---

## 3. Línea base cuantitativa e impacto esperado en el RTF

Dimensiones verificadas en `crates/qwen-core/src/config.rs` y `crates/qwen-engine/src/model.rs`:

### 3.1 Pesos y huella en memoria

| Componente | Parámetros | Tamaño BF16 / F32 | Tamaño bajo `--quant-mixed` | Notas |
|---|---:|---:|---:|---|
| **Talker (28 capas)** | `440.40 M` (`15.73 M/capa`) | `840.25 MiB` (`BF16` + norms `F32`) | **`236.25 MiB`** (`Q4_0`) | Mantiene además `BF16` en `MmapStore` (`talker_prefill` los lee) |
| **Talker embeddings + proj + head** | `323.75 M` (`text_emb` = `311.16 M`) | `617.51 MiB` (`BF16` + biases `F32`) | `617.51 MiB` (`BF16`) | `text_embedding` (`593.50 MiB`) es disperso; `text_proj` = `12.00 MiB`; `codec_emb` = `6.00 MiB`; `codec_head` = `6.00 MiB` |
| **Code Predictor (5 capas)** | `78.64 M` | `150.04 MiB` (`BF16` + norms `F32`) | **`55.31 MiB`** (`30.00 MiB` `INT8` attn + `25.31 MiB` `Q4_0` FFN) | Más `15 × codec_emb` (`60.00 MiB` `BF16`) + `15 × lm_head` (`30.00 MiB` `INT8`) |
| **Speech Decoder** | `114.32 M` (`436.11 MiB` en disco) | `436.11 MiB` (`F32` `mmap`) + `~32+ MiB` heap | `436.11 MiB` (`F32`) + heap | Incluye `16` `codebooks` dequantizados (`32.00 MiB`) y pesos reordenados (`pre_conv_wk`, `initial_conv_wk`, `conv1_wk`) |

> **Nota sobre codificadores (`Base`)**: `speech_tokenizer/model.safetensors` contiene también `encoder.*` (`214.51 MiB` `F32`, `56.23 M` parámetros), y `0.6B-Base` incluye `speaker_encoder.*` (`F32`). Al estar respaldados por `MmapStore`, no generan presión de *working set* salvo al invocar clonación de voz (`--ref-audio` / `--emo-ref`).

### 3.2 Memoria mutable por sesión (`SessionScratch::new()`) y lote (`BatchWorkspace::new()`)

| Estructura | Cálculo exacto | RAM reservada al crear |
|---|---|---|
| `TalkerKvCache::new()` | `28 × 256 × 1024 × 2 (K,V) × 2 B` (`KV_INITIAL_POSITIONS = 256` desde KV inicial de 256 posiciones; antes 1024 = `112.0 MiB`) | **`28.0 MiB`** |
| `TalkerBuffers::new()` (dec + `pref_*` de activación diferidos) | sin staging de pesos desde staging F32 del prefill transitorio (antes `(2048+1024+1024+2048+6144+3072)×1024 × 4 B` = `60.0 MiB` residentes) | **`~0.1 MiB`** *(staging F32 de `60.0 MiB` solo durante el prefill)* |
| `EmbLruCache::new()` | `2048 × 1024 × 4 B` | **`8.0 MiB`** |
| `CpKvCache` + `CpBuffers` + otros | `5 × 64 × 1024 × 2 × 2 B` + bufs | **`~1.5 MiB`** |
| **Total base por `SessionScratch`** | Antes de `ensure_prefill_capacity` / `ensure_conv_state` | **`~37.6 MiB`** tras KV inicial de 256 posiciones+staging F32 del prefill transitorio *(antes `~181.6 MiB`, ~726 MiB con 4 workers)* |
| `BatchWorkspace` KV (`kv_cap` inicial 256, crece hasta `kv_max = 1116`) | `B × 28 × 256 × 1024 × 2 × 2 B` (desde KV de `BatchWorkspace` bajo demanda; antes preasignado a `kv_max`: `122.08 MiB/slot`) | **`28.0 MiB / slot`** *(`224 MiB` para B=8)* |

### 3.3 Tráfico y sincronización por fotograma generado (`80 ms` de audio)

- **Barreras `ComputePool::parallel`**: `1` (`codec_head`) + `325` (`cp_predict`: `30` en `cp_prefill2` + `280` en 14 pasos + `15` `argmax`) + hasta `140` (`talker_step`: `112` proyecciones + hasta `28` en atención paralela para `seq_k >= 48`) = **hasta 466 barreras/fotograma** (**~5,825 barreras/s** a RTF 1.0).
- **Llamadas escalares a `f32::exp()` en `swiglu_inplace`**: `28 × 3072` (Talker) + `16 × 5 × 3072` (Code Predictor) = **331,776 `expf()` escalares/fotograma** (~4.15M/s a RTF 1.0), más `new_frames × 8 × 1024` en el Pre-Transformer del `SpeechDecoder`.
- **Lectura mínima teórica de pesos por fotograma (`--quant-mixed`)**: `236.25 MiB` (Talker `Q4_0`) + `6.0 MiB` (`codec_head` BF16) + `2 × 55.31 MiB` (`cp_prefill2` con lectura doble actual) + `14 × 55.31 MiB` (14 pasos CP) + `30.0 MiB` (`15 × lm_head` INT8) ≈ **`1,157 MiB` (~1.13 GiB) leídos por fotograma** (~14.1 GiB/s sostenidos para RTF 1.0).

### 3.4 Modelo de rendimiento e impacto esperado en el RTF

El tiempo total de síntesis sigue la relación:

$$\text{Tiempo total} \approx \text{prefill} + \max(\text{Productor}, \text{Consumidor}) + \text{drain final}$$

De este modelo se desprenden tres conclusiones clave sobre el **RTF**:

1. **En petición individual (`B=1`, CLI o HTTP simple)**:
   - La reducción de RTF es **moderada en frases largas** y **más pronunciada en frases cortas y TTFA**.
   - **Techo de implementación, no de volumen de bytes**: cada fotograma lee `~1.04 GiB` de pesos cuantizados (`236.25 MiB` del Talker `Q4_0` + `774.34 MiB` de los 14 pasos del CP + `30.0 MiB` de `lm_head`). Con esos mismos bytes, el gemelo C completa el Code Predictor en `33.0–37.0 ms/f` y este motor en `38.7 ms/f`: existe un **5–17 % de margen sin cambiar bitwidth**. **Fundamento**: `Anexo A de phase-0/estado.md`.
   - **Palancas con efecto en `B=1`**: `codec_head` en INT8 (`-3 MiB/frame`), vocoder AVX2 (libera el `drain` cuando mejora el productor; atención del decoder por kernel compartido resultó neutro), prefill por bloques BF16→F32 (`prefill_ms`, crítico en textos breves) y microkernels `matmat` INT8/Q4_0 solo en `cp_prefill2` (2 posiciones/fotograma; el matmat entero es exacto pero no más rápido: **1.26×** E2E y **1.2–1.7×** en microbanco). **Fundamento**: `Anexo A de phase-0/estado.md`.
2. **En servidor concurrente (`--batch-size N`, `--workers N`)**: **fuera de alcance**. El objetivo del plan es el uso local, con una síntesis a la vez en un equipo modesto (laptop, solo CPU, Windows/Linux/macOS). Las optimizaciones que solo rinden con varios clientes concurrentes (escalado batch de microkernels `matmat` INT8/Q4_0, doble transposición en `batch_proj_q`, barreras del batch, parte multi-worker de hilos de OpenBLAS) están en [`ROADMAP.md`](../../ROADMAP.md).
3. **Optimizaciones orientadas a RAM y arranque (huella base de `SessionScratch`, snapshot compacto de KV, cuantización paralela en la carga)**:
   - Reducen cientos de MiB de huella residente (evitando *swapping* en equipos de poca memoria) y aceleran el arranque, sin alterar directamente la velocidad por fotograma cuando hay RAM libre.

---

## 4. Hallazgos detallados y oportunidades de optimización

### 4.1 Ancho de banda de memoria (RAM ↔ Cachés ↔ CPU)

#### Microkernels `matmat` reales para `INT8` y `Q4_0`
- **Ubicación**: `crates/qwen-kernels/src/simd/avx2.rs` (`int8_matmat_slice` L571–603, `q4_0_matmat_slice` L735–766) y `crates/qwen-kernels/src/simd/scalar.rs` (L134–166, L195–226).
- **Problema**: Para cada tile de `MM_TILE = 8` filas, el kernel ejecuta `for b in 0..b_dim` llamando a `int8_matvec_2row` / `q4_0_matvec_2row`. Esto relee las mismas filas de peso `w` desde caché/DRAM `b_dim` veces y en `Q4_0` recalcula `q4_0_sumq` en cada iteración, limitando los resultados de `--matmat-bench` (`crates/qwen-kernels/src/bench.rs`).
- **Impacto**: Ahorra **`~55.3 MiB/frame`** en `cp_prefill2` (`B=2`) y reduce el tráfico de `B` relecturas a **1** (techo teórico `B×`). Medido: **1.26×** E2E en `--batch-bench` y **1.2–1.7×** en microbanco `INT8`/`INT4`. `matmat_bf16` queda fuera de alcance: **0.76–0.97×** en este hardware y descartado en el gemelo C. **Fundamento**: `Anexo A de phase-0/estado.md`.
- **Propuesta**: Escribir microkernels `AVX2+FMA` especializados para `B=2, 4, 8`: cargar 32 bytes de peso (`INT8` o un `Q4_0Block`), decodificar nibbles una sola vez y acumular `B` vectores `a_b` en registros YMM independientes, precalculando `sumq[0..b_dim]` una vez por bloque. Mantener equivalente estructurado en `simd/scalar.rs` para ARM64 y *fallback* x86.
- **Contrato numérico**: **Compatible 100 %** (`L2_rel == 0.0` en `INT8`, `L2_rel <= 1e-6` en `Q4_0` en `--self-test`).
- **Estado**: `vigente` con **alcance local**: `cp_prefill2` del Code Predictor (B=2 por frame, en toda síntesis). El escalado batch (`B = 4, 8`) queda **fuera de alcance** → [`ROADMAP.md`](../../ROADMAP.md), «Microkernels `matmat` que lean los pesos una sola vez por lote». **Fundamento**: `.work/plans/optimizations/phase-0/estado.md`. Plan: **Fase 4, Paso 4.4**; techo según el perfil del 2026-11-18: ~6 % del CP (`cp_prefill2` es 1 de cada 16 pasadas de pesos por frame). **Si se pierde**: `cp_prefill2` lee cada matriz dos veces por frame.

#### Eliminar la doble transposición en `batch_proj_q` + `quantize_act_int8_cols`
- **Ubicación**: `crates/qwen-engine/src/batch.rs:53-115` y `crates/qwen-kernels/src/dispatch.rs:357-379`.
- **Problema**: `BatchWorkspace` guarda activaciones en `[B, cols]`. `batch_proj_q` las transpone a `[cols, B]` (`batch_gather`), `quantize_act_int8_cols` las retranspone a `[B, cols]` en `QXT_TLS` elemento a elemento, y `batch_scatter` transpone `[rows, B] → [B, rows]`.
- **Propuesta**: Cuantizar directamente cada fila contigua `src[b * cols..(b+1) * cols]` con `quantize_act_int8` AVX2 sin `batch_gather` y alinear la salida con el microkernel de microkernels `matmat` INT8/Q4_0.
- **Contrato numérico**: **Bit-idéntico**.
- **Estado**: **`fuera de alcance`** (solo existe en el camino batch) → [`ROADMAP.md`](../../ROADMAP.md), «Eliminar la doble transposición de activaciones en `batch_proj_q`».

#### Cuantización de `talker.codec_head_bf16` bajo `--quant-mixed` / `--int8`
- **Ubicación**: `crates/qwen-engine/src/model.rs:1729` y `crates/qwen-engine/src/pipeline.rs:697-704`.
- **Problema**: En cada fotograma lee `6.0 MiB` (`3072 × 1024 × 2 B`) en `BF16` con FMA de coma flotante.
- **Propuesta**: Soportar `QuantWeight::Int8` para `codec_head` (baja a **`3.0 MiB/frame`** con aritmética entera `maddubs`), manteniendo el criterio conservador de [`NUMERIC_CONTRACT.md`](../NUMERIC_CONTRACT.md) §1.3 de evitar `Q4_0` en cabezas de salida salvo validación explícita.
- **Contrato numérico**: **Modifica [`NUMERIC_CONTRACT.md`](../NUMERIC_CONTRACT.md) §1.3** (cambia la precisión de logits `code0`); requiere evaluación A/B de calidad.
- **Estado**: `descartado` (reestructuración del 2026-11-18). **Fundamento**: `codec_head` lee `6.0 MiB/frame` en `BF16`; `INT8` los reduce a `3.0 MiB/frame`, el 0,6 % del tráfico por frame, y obliga a cambiar `NUMERIC_CONTRACT.md`: el techo no compensa el riesgo.

#### Prefill cuantizado o conversión `BF16→F32` por bloques L2 en `talker_prefill`
- **Ubicación**: `crates/qwen-engine/src/talker.rs:64-82` y `crates/qwen-kernels/src/dispatch.rs:1268-1341`.
- **Problema**: `convert_talker_layer_bf16_to_f32` expande `60.0 MiB/capa` a `F32` (`1,680 MiB` escritos y `1,680 MiB` re-leídos en 28 capas) y fuerza el *page-in* de `840 MiB` `BF16` del `mmap` aunque el Talker corra en `Q4_0`.
- **Propuesta**:
  - *Opción A (bit-exacta en F32)*: convertir `BF16→F32` en bloques pequeños de L2 dentro de `sgemm` sin materializar la capa completa (conserva `F32` exacto y libera `60 MiB` de `TalkerBuffers`).
  - *Opción B (cuantizada)*: `descartado`. No habilitar `matmat_q4_0` en el prefill bajo `--quant-mixed`.
- **Contrato numérico**: Opción A es **compatible**; Opción B altera la precisión del prefill y requiere validación en `NUMERIC_CONTRACT.md`.
- **Estado**: Opción A `vigente` (**Fase 3, Paso 3.1**), Opción B `descartado`. **Perfil del 2026-11-18**: la conversión `BF16→F32` cuesta 127 / 133 ms y `sgemm` 199 / 248 ms (corta / larga) de un prefill de 346 / 417 ms, que es el 57 % del TTFA; techo de la Opción A: −100 a −200 ms. **Efecto medido**: la Opción B produce `+4.7 st` en `ryan`, WER `+1.8/+2.7 pp` y prefill largo `+41 %`. **Fundamento**: `Anexo A §A.3 de phase-0/estado.md`. **Si se pierde**: regenerar audio con el tono alterado.

---

### 4.2 Consumo de CPU (vectorización SIMD y sincronización)

#### Vectorizar la atención con ventana deslizante en `speech_decoder_decode_streaming`
- **Ubicación**: `crates/qwen-engine/src/speech_decoder.rs:1679-1723` vs `733-746`.
- **Problema**: `speech_decoder_decode` (no streaming) usa `d.causal_attention_windowed` (`avx2.rs:1020-1078`), mientras `speech_decoder_decode_streaming` usa bucles escalares `for dd in 0..DEC_HEAD_DIM` y `exp()` escalar sobre hasta `72` claves × `16` cabezas × `8` capas.
- **Propuesta**: Reutilizar `d.causal_attention_windowed` (adaptando el offset de base del KV cache compactable `st.kv_base`).
- **Contrato numérico**: **Compatible (`SNR >= 100 dB`)**.
- **Estado**: `implementado` (Paso 1.5). **Efecto medido**: consolidación del camino canónico, con rendimiento **neutro**. A/B intercalado (3 pares × corta/larga): `decoder_ms` (tiempo de cómputo puro del decoder) −13 ms en la corta y +163 ms en la larga (pares +771 / +163 / −54; el control del par 1 fue anómalo), `drain` −5 / −5 ms, RTF −0,01 / +0,02. La atención (16 cabezas × ≤72 claves × 64 × 8 capas ≈ 1,2 MFLOP/frame) no es el punto caliente del decoder: su coste está en las convoluciones del vocoder. Paridad: códigos idénticos (mismo número de frames, decoder aguas abajo y sin realimentación); PCM con SNR 101,5 / 101,1 dB frente a `baseline_*.wav` (solo ±1 LSB), bit-idéntico entre corridas, también con compactación (`kv_base > 0`, texto de 417 frames: 99,7 dB frente al binario viejo, 379 muestras de 800 640 con ±1 LSB, en el suelo de cuantización PCM16); `test_decoder_streaming_equivalence` (SNR ≥ 100 dB en f32) en verde. **Consecuencia**: el KPI de `decoder_ms` −10 % de la Fase 1 no lo puede aportar atención del decoder por kernel compartido; se traslada al vocoder vectorizado (Fase 4). **Fundamento**: `.work/plans/optimizations/phase-1/estado.md` (Paso 1.5), `Anexo A §A.4 de phase-0/estado.md`.

#### Vectorizar `swiglu_inplace`, `silu` y `convnext_mlp` GELU con `AVX2+FMA`
- **Ubicación**: `crates/qwen-kernels/src/dispatch.rs:1204-1224`, `crates/qwen-engine/src/speech_decoder.rs:219-221, 1786-1788`.
- **Problema**: `swiglu_inplace` ejecuta ~`331,776` `expf()` escalares/frame en Talker + CP; `speech_decoder_decode_streaming` ejecuta `expf()` y `erff()` escalares en FFN y `convnext_mlp`.
- **Propuesta**: Despachar `swiglu_inplace` y `silu` por `KernelDispatch` con aproximación polinomial `AVX2+FMA` de `expf` (8 lanes, similar a `vsin2_avx2` en `avx2.rs:1245`), y vectorizar GELU + LayerNorm en `convnext_mlp`.
- **Contrato numérico**: Diferencias de 1 ULP en `expf()` pueden variar desempates en muestreo respecto a `libm`; requiere medir ULP/SNR y validar que no degrada pruebas de regresión.
- **Estado**: `descartado` para SwiGLU/SiLU y para la fusión `swiglu + quantize_act_int8` (reestructuración del 2026-11-18); GELU/LayerNorm de `convnext_mlp` entra en el vocoder vectorizado (Fase 4, Paso 4.2) solo si su perfil supera el 5 % del decoder (ConvNeXt completo: 5–7 %). **Efecto medido**: SwiGLU es el **1.7 %** del CP en el gemelo C; repartirlo entre hilos ahorró **0.68 ms/f**. Ahorro esperado ≤1 ms/f: `gate_up` (37 %) y QKV (27 %) concentran el coste del CP. **Fundamento**: `Anexo A §A.1 de phase-0/estado.md`.

#### Reducir barreras de `ComputePool`, eliminar `ftz_on()` en `matvec_*` y paralelizar batch unificado
- **Ubicaciones**:
  1. `crates/qwen-kernels/src/dispatch.rs:668, 709, 757, 823, 854, 894` (`ftz_on()` repetido ~466 veces/frame).
  2. ~~`crates/qwen-engine/src/batch.rs`: barreras por slot y slots inactivos en `batch_talker_step_impl`~~ (**fuera de alcance**, camino batch → [`ROADMAP.md`](../../ROADMAP.md), «Compactar los slots activos y usar una sola barrera de atención por capa»).
  3. `crates/qwen-engine/src/code_predictor.rs:206-360`: ~325 barreras por fotograma sobre matrices de `1024`–`3072` filas.
- **Propuestas**:
  - ~~Dejar `ftz_on()` solo al inicializar hilos/runtimes~~ (**descartado por evidencia**, Fase 1 Paso 1.6): `ftz_on()` es `stmxcsr` + comparación (el `ldmxcsr` solo se ejecuta si FTZ/DAZ está apagado): ~466 llamadas/frame ≈ 5 k ciclos ≈ 2 µs/frame frente a ~45 ms/frame (≈ 0,005 %, por debajo de cualquier resolución de A/B). La auditoría de hilos muestra que `qwen-batch-sched` y `qwen-http-single` (`scheduler.rs`) usan el `ComputeRuntime` creado en `main` y no activan FTZ al nacer: para ellos, la llamada en la entrada del kernel es la única garantía. Retirarla obligaría a añadir `ftz_on()` en cada `spawn` (más superficie de error, con riesgo de denormales) a cambio de nada medible.
  - Ajustar el umbral de paralelización en `CodePredictor` para reducir salidas del pool (Fase 4, Paso 4.3).
- **Contrato numérico**: **Bit-idéntico**.
- **Estado**: parte `ftz_on()` `descartado` (Paso 1.6); parte batch **fuera de alcance** ([`ROADMAP.md`](../../ROADMAP.md), «Compactar los slots activos y usar una sola barrera de atención por capa»); parte Code Predictor `vigente` (**Fase 4, Paso 4.3**). **Perfil del 2026-11-18**: 325 despachos al pool por frame; la espera en la barrera es el 6,5 % / 4,2 % del CP sin carga (1,4–2,2 ms/frame); falta medirla con carga, primera tarea del paso. **Fundamento**: el gemelo C registra ~495 despachos al pool por fotograma. **Si se pierde**: barreras redundantes en el CP.

#### Presupuesto de espera en `sd_pool` y vectorizar convoluciones del `SpeechDecoder`
- **Ubicación**: `crates/qwen-kernels/src/dispatch.rs:1411-1417`, `crates/qwen-engine/src/speech_decoder.rs:32-43, 505-525, 1013-1057, 1366-1383`.
- **Problema**: `sd_pool` hereda un presupuesto de espera activa; `cs_dwconv`, `final_conv`, `conv_add_bias` y `add_residual_scaled` son escalares.
- **Propuesta**: mantener la espera activa acotada en `sd_pool` (medido `0` frente a `4 096` en la Fase 1: **descartado**, ver Estado); vectorizar en AVX2 `cs_dwconv`, `final_conv` y `add_residual_scaled` sobre el eje temporal `t`.
- **Contrato numérico**: **Compatible (`SNR >= 100 dB`)**.
- **Estado**: presupuesto de espera `descartado` (Fase 1 Paso 1.7: A/B intercalado (3 pares × corta/larga, variante con `sd_pool = ComputePool::with_spin_budget(sd_nt, 0)`): `decoder_ms` **+44 / +110 ms** (pares cortos +51 / +44 / +2), Talker +0,2 / +0,1 ms/f, CP +0,7 / +0,1 ms/f, RTF +0,01 / 0,00; `drain` −5 / −8 ms. Paridad 12/12 (bit-idéntico). Confirma el prior del motor C: en Windows, dormir el pool cuesta más que la espera activa acotada (4 096).); vocoder vectorizado `vigente` (**Fase 4, Pasos 4.1–4.2**). **Perfil del 2026-11-18**: los 4 bloques de upsample son el 78 % / 83 % del decoder; ConvNeXt 7 % / 5 %; VQ + transformer 11 % / 8 %. El decoder y el productor van empatados (RTF ~0,56 cada uno en la larga): el RTF solo baja si bajan los dos. **Efecto medido (gemelo C)**: dormir el pool cuesta **3–4 %** de RTF (0.76 → 0.79 sin carga; 0.87 → 0.91 con 4 núcleos); `4 096` rinde más que `0` y que `20 000`. **Fundamento**: `Anexo A §A.3 de phase-0/estado.md`. **Si se pierde**: regresión silenciosa en Windows.

#### Contención global de `openblas_set_num_threads` en `WorkerPoolScheduler`
- **Ubicación**: `crates/qwen-kernels/src/blas.rs:161-175` y `crates/qwen-app/src/server/scheduler.rs:92-109`.
- **Problema**: `FfiBlas::set_threads` modifica el estado global de proceso de OpenBLAS. Con `--workers N > 1`, múltiples workers se pisan entre prefill y generación. Un `OMP_NUM_THREADS` ausente en el lanzador compromete los hilos de OpenMP antes de `main`.
- **Propuesta**: fijar `OMP_NUM_THREADS = -j` en el lanzador y `openblas_set_num_threads` una sola vez al iniciar cuando `n_workers > 1` (sin reconfigurar por petición). `OPENBLAS_NUM_THREADS` permanece sin definir. `--batch-size N` (o `faer-blas`) es la opción preferente en concurrencia.
- **Contrato numérico**: **Bit-idéntico** por petición con el mismo número de hilos efectivo.
- **Estado**: parte lanzador `implementada` (**Fase 2**, Paso 2.2: el binario fija `OMP_NUM_THREADS = -j` antes de que se inicialice OpenBLAS; sin la variable, el servidor comprometía **+1 539 MiB** y el RTF empeoraba un 12 %); parte multi-worker (`openblas_set_num_threads` con `n_workers > 1`) **fuera de alcance** → [`ROADMAP.md`](../../ROADMAP.md), «Configurar los hilos de OpenBLAS una sola vez con varios workers». **Efecto medido (gemelo C)**: en Windows `OMP_NUM_THREADS = -j` reduce **1 410 MiB** comprometidos y el tiempo un **6 %**, sin cambiar el audio. **Fundamento**: `Anexo A §A.6 de phase-0/estado.md`.

---

### 4.3 Consumo de memoria RAM

#### Reducir la huella base de `SessionScratch` de `~181.6 MiB` a `~37.6 MiB`
- **Ubicación**: `crates/qwen-engine/src/model.rs:600-609, 707-756, 861-937`.
- **Problema**: `TalkerKvCache::new()` asigna `1024` posiciones (`112.0 MiB`); `TalkerBuffers::new()` asigna `pref_*_f32` (`60.0 MiB`) de inmediato; `SdStreamState` retiene 14 `Vec<f32>` temporales (`tmp_*`) por cada `ContinuousSlot` aunque decodifiquen secuencialmente con un único `sd_ws`.
- **Propuesta**:
  - Cambiar `KV_INITIAL_POSITIONS` a `256` (`28.0 MiB`, ahorro de **`84.0 MiB`/sesión**, creciendo dinámicamente vía `ensure_capacity`).
  - Sacar el staging `pref_*_f32` de `TalkerBuffers`: transitorio y local de `talker_prefill` (se asigna al entrar y se libera al salir). Asignarlo «bajo demanda y retenerlo» no ahorraría nada en el pico, porque toda síntesis prefila (ahorro de **`60.0 MiB`/sesión** fuera del prefill).
  - ~~Mover los 14 `tmp_*` de `SdStreamState` a `DecoderWorkspace`~~ (**descartado por evidencia**: nacen vacíos y con `stream_chunk_frames = 10` suman ≈ 444 KiB/slot, < 2 MiB en batch4). En su lugar, KV de `BatchWorkspace` bajo demanda ataca el driver real de la RAM por slot: el KV de `BatchWorkspace` arranca en `KV_INITIAL_POSITIONS` y crece bajo demanda hasta `kv_max`, con un bloque `[NUM_LAYERS, kv_cap, KV_DIM]` por slot (el crecimiento reasigna slot a slot). Con KV inicial de 256 posiciones+staging F32 del prefill transitorio, la huella base de `SessionScratch::new()` baja de `~181.6 MiB` a **`~37.6 MiB`**.
- **Contrato numérico**: **Bit-idéntico**.
- **Estado**: KV inicial de 256 posiciones `implementado` (Paso 1.1); staging F32 del prefill transitorio `implementado` (Paso 1.3); KV de `BatchWorkspace` bajo demanda `implementado` (Paso 1.4, KV de `BatchWorkspace` bajo demanda; la parte `tmp_*` se descarta por evidencia). **Efecto medido KV de `BatchWorkspace` bajo demanda** (A/B ABBA, hashes `bm_0..7` idénticos): servidor `--batch-size 4` **−377 MiB** de pico de commit (2 883→2 506) y de reposo, WS sin cambio, RTF −2,2 %; con texto de crecimiento (>256 posiciones) −265 MiB y RTF −3,3 %; CLI `--batch-multi-test 8` largo −343,6 MiB, y con crecimiento hasta `kv_max` ±0 (pico transitorio acotado a un slot: WS +15 MiB). Una primera versión con bloques por (slot, capa) de 512 KiB subía el WS en reposo +112 MiB: el heap sirve bloques de ese tamaño desde segmentos y los pone a cero tocándolos; los bloques por slot (≥ 14 MiB) usan páginas cero diferidas. **Efecto medido staging F32 del prefill transitorio** (A/B intercalado, pico exacto de commit `PeakPagefileUsage`, paridad 12/12 con la referencia): CLI **−60,3 / −60,1 MiB** (corta / larga; WS pico −60 MiB, porque el pico ocurre en el decoder, cuando el staging ya se liberó); servidor `--workers 4` **−241 MiB** de pico de commit (1 878→1 637) y de reposo; `--batch-size 4` **−120 MiB** (3 003→2 883). `prefill_ms` sin cambio. RTF por petición en workers4 **+1,9 %** de mediana pareada (ABBA, 4 repeticiones, rango +0,4…+3,5 %, sin calma plena), dentro de la compuerta ±3 %; CLI y batch4 neutros. **Efecto medido KV inicial de 256 posiciones**: ahorra `84 MiB`/sesión solo si no hay crecimiento. `256` cubre 211 frames de audio + prompt; un texto de 450 caracteres usa ~319. El gemelo C probó `512`, creció a `1024` en textos largos y lo descartó; su valor validado es `1024`, que este motor ya tiene. **Prueba obligatoria**: medir el pico *durante* `ensure_capacity`. **Fundamento**: `Anexo A §A.3 de phase-0/estado.md`. **Si se pierde**: reasignación del KV en frases largas.

#### Snapshot compacto de KV en `generate_batch_multi`
- **Ubicación**: `crates/qwen-engine/src/batch.rs:163, 200-219, 1144-1181` (y `run_continuous_batch` en `1485`, ruta directa sin snapshot).
- **Problema**: `generate_batch_multi` materializaba el KV completo del scratch por slot en `kv_snapshots` antes de copiar solo `prompt_lens[b]` posiciones a `BatchWorkspace` (**`112 MiB`/petición** pre-KV inicial de 256 posiciones, **`896 MiB`** para 8; post-KV inicial de 256 posiciones `28,0 MiB`/slot, **`224 MiB`** por grupo de 8). `run_continuous_batch`/`BatchWorkspace` preasignan `122.08 MiB`/slot contiguos (`kv_max = 1116`), huella independiente del snapshot.
- **Implementación (Fase 1, Paso 1.2)**: se elimina el snapshot completo y se conserva un staging compacto con layout `(layer × pl) × KV_DIM` (`NUM_LAYERS * pl * KV_DIM` elementos `u16` por buffer), cargado al slot por la **única ruta** existente `load_slot_kv(b, sk, sv, pl, pl)` (`src_kv_max = prompt_len`, stride de fuente igual al de destino lógico del prompt). Se mantiene el staging porque `bw.kv_max` depende del `max_pl` de todo el grupo y se conoce solo tras prefilar todos los slots; la copia directa scratch→`bw` exigiría `BatchWorkspace` ragged (stride por slot) o pre-dimensionarlo a 1116 (peor para RAM) — opción futura junto a bloques KV paginados si se escala a `B >= 16`.
- **Contrato numérico**: **Bit-idéntico**.
- **Estado**: `implementado` (Fase 1, Paso 1.2). **Alcance**: modo batch del CLI (`generate_batch_multi` / `--batch-multi-test` / `--batch`), **no** el pico del servidor `--batch-size` (que usa `run_continuous_batch` sin snapshot; su pico lo gobierna `BatchWorkspace.kv_k/kv_v` ≈ 488 MiB committed con `kv_max=1116`, a tratar en KV de `BatchWorkspace` bajo demanda). **Efecto medido** (A/B intercalado, pico de privados pareado, 3 pares × 2 textos, hashes 48/48 por texto): **corto (pl=21) −206,0 MiB**, **largo (pl=44) −169,5 MiB** de mediana (Δ en la fase de snapshots 185,9 MiB, acorde a 917 504 B × (256 − pl)); wall ±3%; servidor workers4/batch4 sin cambio (commit pico exacto −2,8 / 0,0 MiB). **Compuerta**: hashes bit-idénticos y no-regresión de rutas no tocadas (no «WS ≤ 2 250 MiB»). **Fundamento**: `.work/plans/optimizations/phase-0/estado.md`, `.work/plans/optimizations/phase-1/estado.md`. **Si se pierde**: volver al snapshot completo y al pico de privados medido en el control.

#### Carga paralela de cuantización al iniciar `Model06B::load`
- **Ubicación**: `crates/qwen-engine/src/model.rs:1487-1489` y `crates/qwen-kernels/src/quant.rs:144-236`.
- **Problema**: `quantize_bf16_to_q4_0_into` y `quantize_bf16_to_int8_into` se ejecutan secuencialmente en un hilo durante `Model06B::load`.
- **Propuesta**: Paralelizar por capas/filas sobre `runtime.pool` durante la carga (acelera ~3–4× el startup en `--quant-mixed`).
- **Contrato numérico**: **Bit-idéntico**.
- **Estado**: `descartado` (reestructuración del 2026-11-18): el uso real es un servidor residente que carga el modelo una sola vez. **Efecto medido**: sin medir en Fase 0; la carga del modelo osciló entre 4 400 y 5 397 ms. **Fundamento**: `.work/plans/optimizations/phase-0/estado.md`.

---

## 5. Matriz de priorización y dependencias

Prioridades para el uso real (servidor residente, una petición cada vez por `/v1/tts/stream`, laptop solo CPU, multiplataforma), ordenadas por impacto medido en el perfil fino del 2026-11-18 (§6, «Hallazgos del perfil fino»). El servidor con varios clientes y el batch están fuera de alcance: [`ROADMAP.md`](../../ROADMAP.md).

| Prioridad | Optimización | Objetivo al que sirve | Techo | Riesgo numérico | Esfuerzo | Fase |
|---|---|---|---|---|---|---|
| **1** | Hilos de OpenMP fijados por el binario (`OMP_NUM_THREADS = -j`) | Menos RAM; menos hilos compitiendo con carga | medido: −1 539 MiB de commit en el servidor arrancado sin la variable; RTF 0,74–0,77 → 0,66–0,69 | Ninguno | Bajo | 2 |
| **2** | Prefill por bloques `BF16→F32` (Opción A) | Oír la voz antes | −100 a −200 ms de TTFA (conversión 127–133 ms más la relectura) | Ninguno esperado (bit-idéntico) | Medio | 3.1 |
| **3** | Primer fragmento del decoder en caliente | Oír la voz antes | ~90 ms de TTFA, si el arranque en frío persiste en el servidor | Ninguno | Bajo | 3.2 |
| **4** | Vocoder vectorizado (bloques de upsample) | Audio sin cortes con carga | Bloques de upsample = 78–83 % del decoder, que limita el RTF junto al productor | `SNR >= 100 dB` | Medio | 4.1–4.2 |
| **5** | Barreras del Code Predictor | Audio sin cortes con carga | 4–7 % del CP sin carga; mayor con carga (por medir) | Ninguno (bit-idéntico) | Bajo–Medio | 4.3 |
| **6** | Microkernel `matmat` INT8/Q4_0 en `cp_prefill2` | Audio sin cortes con carga | ~6 % del CP; parche exacto listo | Ninguno (bit-idéntico) | Bajo | 4.4 |
| **7** | Kernels NEON (`matvec` INT8/Q4_0, conversión `BF16→F32`, vocoder) | Rendimiento en macOS/ARM64 | Toda la ruta ARM64 es hoy escalar | Contrato frente a la ruta escalar | Alto | 5 |

**Descartados** (evidencia en §4): SwiGLU/SiLU vectorizados y su fusión con la cuantización de `down_proj` (1,7 % del CP, ≤ 0,68 ms/f); `codec_head` en INT8 (0,6 % del tráfico y cambia el contrato); cuantización paralela en la carga (el servidor carga una vez); prefill cuantizado (Opción B: +4,7 st de tono); espera de `sd_pool` a 0 (Paso 1.7) y retirada de `ftz_on()` (Paso 1.6).

**Historial (implementado en la Fase 1)**: KV inicial de 256 posiciones, snapshot compacto de KV, staging F32 del prefill transitorio, KV de `BatchWorkspace` bajo demanda y atención del decoder por kernel compartido.

**Fundamento de los estados**: `Anexo A §A.3 de phase-0/estado.md` y el perfil fino citado arriba.

---

## 6. Planificación de implementación progresiva (Fases y Pasos)

Las mejoras **no deben aplicarse en un único cambio monolítico**: mezclar ajustes de memoria, microkernels SIMD y cambios de precisión numérica dificultaría aislar regresiones de determinismo o concurrencia. El plan se estructura en dos niveles coherentes:

- **Nivel Mapa (Fases 0–5)**: agrupan los cambios por el objetivo de uso que persiguen (RAM, primer audio, RTF con carga, ARM64), con un criterio de cierre medible por fase.
- **Nivel Ejecución (Pasos `X.Y`)**: dentro de cada Fase, cada mejora o sub-mejora se ejecuta como **1 Paso aislado** siguiendo siempre el mismo bucle de 5 reglas:
  1. **Chequeo previo**: confirmar archivos exactos, invariantes de [`NUMERIC_CONTRACT.md`](../NUMERIC_CONTRACT.md) y riesgos de borde antes de editar.
  2. **Implementar solo ese Paso**: sin mezclar cambios de otros pasos.
  3. **Compuerta de exactitud**: ejecutar `clippy`, `cargo test --workspace --release`, `--self-test` y comparar paridad frente al paso anterior o línea base.
  4. **Compuerta de rendimiento**: verificación mínima suficiente del efecto del Paso (microbanco, corrida de commit o 3 corridas sueltas si el efecto supera con holgura el ruido); el A/B E2E intercalado se hace una sola vez, al cierre de la fase (§7).
  5. **Decisión binaria**: `commit` si aporta mejora comprobada sin regresiones; revertir de inmediato si no aporta ganancia o degrada otra métrica.

---

### Fase 0 — Congelación de línea base y perfilado real (sin cambios de código)
- **Propósito**: fijar una referencia reproducible en el hardware objetivo antes de alterar código, evitando optimizar a ciegas.
- **Pasos**:
  - **Paso 0.1 (Capturar línea base)**: ejecutar `--seed 4 --quant-mixed -j 4` en frase corta, frase larga e ICL (si aplica), además de `--matmat-bench` y `--batch-bench`; registrar `prefill_ms`, `talker_ms`, `cp_ms`, `decoder_ms`, `drain_ms`, `TTFA`, `RTF` y pico de RSS (CLI y servidor `--workers 4` / `--batch-size 4`), conservando códigos códec y WAV de referencia.
  - **Paso 0.2 (Perfilado 80/20)**: contrastar qué bloque domina el tiempo real de pared para validar el orden de los pasos siguientes y detectar mejoras cuyo peso sea marginal (`< 2–3 %`).
- **Criterio de cierre**: tabla de métricas base documentada y archivos de referencia guardados.

**Techo del consumidor**: con `prefill 475`, productor `9 666`, `SpeechDecoder 10 185` y `drain 388` ms, un CP −20 % sin tocar el decoder deja el wall gobernado por el decoder (RTF ≈ **0.68–0.70**, −3 %); el mismo CP −20 % con decoder −20 % da RTF ≈ **0.59–0.60** (−18 %). Para que el ahorro del productor no se traslade al `drain`, hace falta acelerar el decoder; la Fase 1 midió que su coste está en el vocoder (Fase 4, Pasos 4.1–4.2) y no en la atención (atención del decoder por kernel compartido, implementado y neutro). **Fundamento**: `Anexo A §A.4 de phase-0/estado.md`.

---

### Fase 1 — *Quick wins* de bajo riesgo en CPU y RAM
- **Alcance**: huella base de `SessionScratch` (KV inicial, staging del prefill, KV de lote), snapshot compacto de KV, atención del decoder por kernel compartido, retirada de `ftz_on()` redundante y presupuesto de espera de `sd_pool` (en último lugar y condicionado).
- **Prerequisitos**: Fase 0 completada.
- **Pasos (ordenados de menor a mayor riesgo de borde)**:
  - **Paso 1.1 (KV inicial de 256 posiciones)**: **Qué cambia**: `KV_INITIAL_POSITIONS` de `1024` a `256` (`crates/qwen-engine/src/model.rs:33`). **Por qué**: −84.0 MiB/sesión. **Riesgo a vigilar**: `256` cubre 211 frames de audio + prompt; un texto de 450 caracteres usa ~319 → prueba de crecimiento obligatoria y medición del pico durante `ensure_capacity`. **Compuerta**: RSS sin subir durante el crecimiento. **Decisión**: conservar solo si no aparece ese pico.
  - **Paso 1.2 (snapshot compacto de KV)**: **Qué cambia**: sustituir el snapshot completo de `generate_batch_multi` por un snapshot compacto `(layer × pl) × KV_DIM` cargado por la ruta única `load_slot_kv(b, sk, sv, pl, pl)` (`crates/qwen-engine/src/batch.rs`); `run_continuous_batch` (servidor `--batch-size`) no usa snapshots y no se modifica. **Por qué**: reducir el pico de privados del modo batch del CLI (el pico del servidor lo gobierna `BatchWorkspace`, ~488 MiB committed con `kv_max=1116`). **Riesgo a vigilar**: offsets de capa/stride (`layer * src_stride * KV_DIM` en origen). **Compuerta**: hashes bit-idénticos y Δ de pico privado pareado en banda (medido: −206,0 MiB corto pl=21, −169,5 MiB largo pl=44; servidor ± commit exacto). **Decisión**: conservado (implementado).
  - **Paso 1.3 (staging F32 del prefill transitorio)**: **Qué cambia**: el staging F32 de pesos del prefill (`60.0 MiB`) sale de `TalkerBuffers` y pasa a ser local y transitorio de `talker_prefill` (`crates/qwen-engine/src/talker.rs`). **Por qué**: el pico de commit ocurre en el decoder, después del prefill; liberar el staging lo baja (CLI −60 MiB, workers4 −241 MiB, batch4 −120 MiB). **Compuerta**: pico de commit CLI y servidor; `prefill_ms` y RTF ±3 %. **Decisión**: `implementado`.
  - **Paso 1.4 (KV de `BatchWorkspace` bajo demanda)**: **Qué cambia**: el KV de `BatchWorkspace` (`crates/qwen-engine/src/batch.rs`) separa `kv_max` (límite) de `kv_cap` (posiciones asignadas, inicial `KV_INITIAL_POSITIONS`) y crece bajo demanda con `ensure_kv_capacity`, en bloques por slot. Los `tmp_*` de `SdStreamState` se descartan (≈ 444 KiB/slot). **Por qué**: el KV preasignado a 1 116 posiciones (122 MiB/slot) era el driver del pico batch. **Compuerta**: pico de commit batch4 y BatchMulti; WS y RTF sin regresión; hashes idénticos. **Decisión**: `implementado` (batch4 −377 MiB).
  - **Paso 1.5 (atención del decoder por kernel compartido)**: **Qué cambia**: reemplazar la atención escalar de `speech_decoder_decode_streaming` (`crates/qwen-engine/src/speech_decoder.rs:1679-1723`) por `d.causal_attention_windowed` con `q_offset = st.kv_len - st.kv_base` y `seq_k = st.kv_len + new_frames - st.kv_base`. **Por qué**: el decoder gobierna el wall cuando mejora el productor. **Riesgo a vigilar**: compactación con `st.kv_base > 0`. **Compuerta**: paridad (`SNR >= 100 dB`, bit-identidad entre corridas) y `decoder_ms`/`drain` sin regresión. **Decisión**: `implementado`; rendimiento neutro: la atención no es el punto caliente del decoder, cuyo coste está en el vocoder.
  - **Paso 1.6 (`ftz_on()` redundante)**: **Decisión**: `descartado` sin cambio de código. `ftz_on()` es `stmxcsr` + comparación (el `ldmxcsr` solo se ejecuta si FTZ/DAZ está apagado): ~466 llamadas/frame ≈ 5 k ciclos ≈ 2 µs/frame frente a ~45 ms/frame (≈ 0,005 %, por debajo de cualquier resolución de A/B). La auditoría de hilos muestra que `qwen-batch-sched` y `qwen-http-single` (`scheduler.rs`) usan el `ComputeRuntime` creado en `main` y no activan FTZ al nacer: para ellos, la llamada en la entrada del kernel es la única garantía. Retirarla obligaría a añadir `ftz_on()` en cada `spawn` (más superficie de error, con riesgo de denormales) a cambio de nada medible.
  - **Paso 1.7 (espera de `sd_pool`)**: **Qué cambia**: medir `spin_budget = 0` frente a `4 096` en `sd_pool` (`crates/qwen-kernels/src/dispatch.rs:1417, 1440`). **Por qué**: el valor por defecto rinde más en Windows. **Riesgo a vigilar**: `decoder_ms` y `drain` saturan. **Compuerta**: sin regresión en ambas condiciones. **Decisión**: `descartado` (revertido). A/B intercalado (3 pares × corta/larga, variante con `sd_pool = ComputePool::with_spin_budget(sd_nt, 0)`): `decoder_ms` **+44 / +110 ms** (pares cortos +51 / +44 / +2), Talker +0,2 / +0,1 ms/f, CP +0,7 / +0,1 ms/f, RTF +0,01 / 0,00; `drain` −5 / −8 ms. Paridad 12/12 (bit-idéntico). Confirma el prior del motor C: en Windows, dormir el pool cuesta más que la espera activa acotada (4 096).
- **Criterio de cierre de Fase 1**:
  1. `cargo test --workspace --release` y `qwen_tts --self-test` verdes.
  2. **Códigos códec** (`Talker + CodePredictor`) **bit-idénticos** frente a la Fase 0 con `--seed 4`; **audio PCM** del `SpeechDecoder` con **`SNR >= 100 dB`** (Paso 1.5) y **bit-idéntico entre corridas** del nuevo binario.
  3. Pico `--batch-size 4` ≤**2 250 MiB** y `--workers 4` ≤**2 100 MiB**; RSS CLI ≤**2 070 MiB**; `decoder_ms` largo ≤**9 150 ms**, `drain` ≤**350 ms**; RTF corta ≤**0.78** sin regresión en larga.
  4. Compuerta con 1 hilo copiando 256 MB: el RTF no empeora respecto a la Fase 0 en la misma condición.
- **Resultado del cierre (2026-11-18)**: KV inicial de 256 posiciones, snapshot compacto de KV, staging F32 del prefill transitorio, KV de `BatchWorkspace` bajo demanda y atención del decoder por kernel compartido implementados; `ftz_on()` redundante (`ftz_on`) descartado y espera de `sd_pool` (espera de `sd_pool`) revertido. Frente al binario de Fase 0 en A/B intercalado: pico de commit CLI −144 MiB; `--workers 4` 2 213 → 1 638 MiB; `--batch-size 4` 3 172 → 2 506 MiB; WS sin regresión; RTF neutro (0,66 / 0,64; −1,5 % / +1,6 %; L1 sin regresión); `drain` 297 / 326 ms. Los umbrales de RSS de este criterio se fijaron en working set; el ahorro es commit diferido (métrica primaria adoptada en la Fase 1). El objetivo de `decoder_ms` no se cumple (9 151 ms): su coste está en el vocoder (Fase 4, Pasos 4.1–4.2), no en la atención (atención del decoder por kernel compartido, neutro). **El cierre de la Fase 1 es la línea base única de la Fase 2 en adelante** (tiempos, RAM y audio de referencia). **Fundamento**: `.work/plans/optimizations/phase-1/estado.md`.

---

### Reestructuración de las fases pendientes (2026-11-18)
Las fases 2–4 originales priorizaban el servidor multi-cliente y el batch, estaban ordenadas por técnica y partían de dos supuestos que el perfil refutó: el coste del decoder no está en su atención, sino en los bloques de upsample del vocoder, y el primer audio lo gobierna el **prefill**, no el Code Predictor. Se sustituyen por fases ordenadas por el efecto que nota el usuario. **Línea base única**: el cierre de la Fase 1 (`.work/plans/optimizations/00-estado.md`, fila «Cierre Fase 1»; audio `phase-1/baseline_fase1_{short,long}.wav`); cada fase se compara contra el cierre de la anterior.

**Hallazgos del perfil fino** (corta / larga). Una corrida por texto de referencia con HEAD `78d4bc9` más instrumentación temporal de tiempos por hilo, sin carga; el audio salió con los hashes de la línea base, así que la instrumentación no altera el cálculo. Con una sola corrida (±5 % de ruido) sirve para repartir el tiempo, no para comparar binarios. Evidencia e instrumentación reutilizable: `.work/plans/optimizations/phase-4/instrumentacion/`.
- TTFA 602 / 663 ms = prefill 346 / 417 ms + 2 frames de Talker y CP ~72 ms + primer fragmento del decoder 190 / 179 ms (~90 ms de arranque en frío sobre el régimen).
- Prefill: conversión `BF16→F32` 37 % / 32 %, `sgemm` 57 % / 59 %.
- Code Predictor: 325 despachos al pool por frame; espera en barreras 6,5 % / 4,2 % del CP.
- Decoder: bloques de upsample 78 % / 83 %; casi no usa su pool (su cálculo va por `sgemm` y bucles en un solo hilo); productor y decoder empatados (RTF ~0,56 cada uno).

---

### Fase 2 — Hilos de OpenMP fijados por el binario
- **Objetivo**: con el servidor arrancado sin variables de entorno, el pico de commit queda dentro de ±20 MiB del medido con `OMP_NUM_THREADS=4`.
- **Por qué primero**: es el cambio más barato con el mayor techo conocido y fija las condiciones reales de medición del resto del plan.
- **Pasos**:
  - **Paso 2.1 (medir el coste de no fijar `OMP_NUM_THREADS`)**: servidor con la variable sin definir y con `=4`, una petición por `/v1/tts/stream`: pico y reposo de commit, número de hilos y RTF. Si la diferencia es < 50 MiB y el RTF no cambia más de un 3 %, la fase se cierra sin cambio de código.
  - **Paso 2.2 (fijar los hilos desde el binario)**: hilos de OpenMP = `-j` antes de la primera región paralela de OpenBLAS (`crates/qwen-kernels/src/blas.rs`); `OPENBLAS_NUM_THREADS` del usuario mantiene la prioridad; macOS (Accelerate) no cambia. Verificación: repetir la corrida del Paso 2.1.
- **Resultado de los Pasos 2.1 y 2.2 (2026-11-18)**: OpenBLAS 0.3.34 va enlazado estático con OpenMP y su constructor, antes de `main`, reserva 128 MiB de commit por hilo según `OMP_NUM_THREADS` (sin ella, 16). Servidor sin la variable: reposo 2 604 → 1 065 MiB, pico 2 740 → 1 328 MiB (igual que con `=4`), 25 → 13 hilos, RTF 0,74–0,77 → 0,66–0,69, PCM idéntico. Implementación: un inicializador en `.CRT$XCT` (`qwen-app`) parsea los argumentos con `parse_cli` y llama a `set_initial_openblas_threads(-j)` antes de los constructores de OpenBLAS y libgomp. Hallazgo: pedir a OpenBLAS más hilos que los iniciales colgaba la primera `sgemm` (ya pasaba con `OMP_NUM_THREADS=4` y `-j 6`); `set_threads` ahora se limita a los hilos iniciales y los aplica en cada hilo que llama a `sgemm` (con OpenMP el valor es por hilo). **Fundamento**: `.work/plans/optimizations/phase-2/estado.md`.
- **Criterio de cierre**: objetivo cumplido; audio bit-idéntico; A/B E2E de la fase sin regresión de RTF, sin carga y con carga.
- **Cierre (2026-11-18, HEAD `f7647f8`)**: **cerrada**. Criterios cumplidos: diferencia de pico 0,4 MiB, audio bit-idéntico y A/B intercalado neutro (con carga, −3,6 % / 0 % de RTF; sin carga, empate por pares con el equipo muy ruidoso). **El cierre de la Fase 2 es la línea base de la Fase 3.**

---

### Fase 3 — Primer audio antes
- **Objetivo**: TTFA ≤ 420 ms (corta) y ≤ 500 ms (larga), hoy 570 / 678 ms, sin carga y con carga.
- **Pasos**:
  - **Paso 3.1 (prefill por bloques `BF16→F32`)**: en `talker_prefill` (`crates/qwen-engine/src/talker.rs`), convertir por bloques de filas del tamaño de L2 y lanzar `sgemm` por bloque, sin materializar la capa en F32. Verificación: hashes de audio y `prefill_ms` de 3 corridas sueltas.
  - **Paso 3.2 (primer fragmento del decoder en caliente)**: medir el primer fragmento en la segunda petición del servidor; si el arranque en frío persiste (> 30 ms sobre el régimen), dejar el decoder dimensionado y calentado al arrancar el servidor. Verificación: log de dos peticiones consecutivas y audio bit-idéntico.
- **Resultado del Paso 3.1 (2026-11-18)**: partir `N` en bloques de 4 MiB bajaba el prefill 100–140 ms pero **cambiaba el audio**: OpenBLAS 0.3.34 redondea distinto al partir `N`, y el muestreo autorregresivo amplifica la diferencia hasta cambiar los códigos (SNR −2 dB). En el Talker no hay término medio: o bit-idéntico o audio distinto. Variante conservada: convertir cada proyección justo antes de su `sgemm` (staging 60 → 24 MiB): prefill −56 / −58 ms y TTFA −27 / −51 ms, bit-idéntico. **Fundamento**: `.work/plans/optimizations/phase-3/estado.md`.
- **Paso 3.3 (prefill por bloques con audio nuevo, decisión del usuario)**: se aplican los bloques de 4 MiB y se fija una línea base de audio nueva (`phase-3/baseline_fase3_*`; la de la Fase 1 queda como histórico; contrato en `NUMERIC_CONTRACT.md`). Servidor por `/v1/tts/stream`: TTFA ~549 → ~486 ms (corta) y ~613 → ~557 ms (larga), pico de commit −17 MiB. Frente al Paso 3.1 solo, el CLI gana −63 / −13 ms de prefill.
- **Criterio de cierre**: metas de TTFA en el A/B E2E y en `/v1/tts/stream`; audio bit-idéntico (o `SNR >= 100 dB` documentado en `NUMERIC_CONTRACT.md`); RTF y commit sin regresión.
- **Cierre (2026-11-18, HEAD `66d2174`)**: **cerrada con la meta ajustada por el usuario** a TTFA ≤ 500 / 600 ms sin carga (la original, 420 / 500, no se alcanzó). A/B intercalado frente al cierre de la Fase 2: TTFA 618 → 493 / 702 → 590 ms sin carga y 909 → 710 / 994 → 811 ms con carga; prefill −124 / −118 ms; RTF −1 a −5 %; commit sin cambio. Audio de referencia nuevo desde el Paso 3.3. **El cierre de la Fase 3 es la línea base de la Fase 4.**

---

### Fase 4 — Síntesis estable con la laptop ocupada
- **Objetivo**: RTF ≤ 0,85 con 1 hilo copiando 256 MiB (hoy 1,06 / 0,98) y ≤ 0,55 sin carga (hoy 0,66 / 0,64). Como productor y decoder van empatados, la fase baja los dos.
- **Pasos**:
  - **Paso 4.1 (perfil interno de los bloques de upsample)**: reparto entre *snake*, convolución transpuesta y convoluciones dilatadas, y uso de hilos; instrumentación temporal, una corrida por texto.
  - **Paso 4.2 (vocoder vectorizado)**: vectorizar en AVX2, con equivalente escalar, la parte caliente que señale el Paso 4.1 (`speech_decoder.rs`, `simd/avx2.rs`). Verificación: `test_decoder_streaming_equivalence`, `SNR >= 100 dB` y `decoder_ms` de 3 corridas.
  - **Paso 4.3 (barreras del Code Predictor)**: medir la espera con carga; si supera el 8 % del CP, fusionar despachos del pool sin dependencia de datos dentro de cada capa (`crates/qwen-engine/src/code_predictor.rs`). Bit-idéntico.
  - **Paso 4.4 (microkernel `matmat` INT8/Q4_0 en `cp_prefill2`)**: aplicar el parche exacto ya verificado (`--self-test` `L2_rel = 0`). Verificación: `--matmat-bench` (B=2) y audio bit-idéntico.
- **Resultado del Paso 4.1 (2026-11-18)**: las convoluciones dilatadas k=7 de los bloques residuales son el 58 % del decoder, la transpuesta el 18 %, la `pointwise` el 9 % y *snake* solo el 3,6 %. Todo va por `sgemm`. En los bloques 2–3 (192 y 96 canales) la k=7 rinde 1,7× peor por FLOP que en los bloques 0–1 (K de `sgemm` = 96/192). El Paso 4.2 pasa de «vectorizar el vocoder» a **im2col en esas convoluciones** (una `sgemm` con K = 7·canales). **Fundamento**: `.work/plans/optimizations/phase-4/estado.md`.
- **Resultado del Paso 4.2 (2026-11-18)**: im2col en la `conv1` k=7 de los bloques 2–3 (una `sgemm` con `K = 7 · canales`): decoder −3 % / −7 %, drenaje −38 % / −10 %, SNR 105,8 / 104,7 dB. El RTF no cambia: el productor (Talker + CP) va por delante del decoder.
- **Paso 4.3 (barreras del CP): descartado (2026-11-18)**. La espera con carga es el 11,7 % del CP (6,9 % sin carga), pero los 4 despachos por capa (QKV fusionado, `wo`, `gate_up`, `down`) dependen cada uno del resultado completo del anterior: no hay despachos que fusionar. Repartir en más trozos ya estaba descartado (Anexo A de la Fase 0).
- **Resultado del Paso 4.4 (2026-11-18)**: microkernel `matmat` INT8/Q4_0 en `cp_prefill2` (lee cada fila de pesos una vez para las 2 posiciones): `matmat` Q4_0 B=2 −40 a −50 %; RTF corta 0,72 → 0,69, larga 0,67 → 0,66; audio bit-idéntico.
- **Criterio de cierre**: metas de RTF en el A/B E2E y en `/v1/tts/stream`; códigos códec bit-idénticos y PCM bit-idéntico o `SNR >= 100 dB` documentado; TTFA y commit sin regresión.

---

### Fase 5 — Kernels NEON para macOS/ARM64
- **Objetivo**: RTF < 1 sin carga en ARM64 (hoy todo pasa por `simd/scalar.rs`).
- **Prerrequisito**: una máquina ARM64 con el modelo; compilar en cruzado no verifica ni exactitud ni velocidad.
- **Pasos**:
  - **Paso 5.1 (línea base en ARM64)**: tiempos por etapa, TTFA, RTF y hashes con la ruta escalar.
  - **Paso 5.2 (`matvec` INT8 y Q4_0 en NEON)**: kernels del Talker y del Code Predictor. Verificación: `--self-test` en ARM64.
  - **Paso 5.3 (conversión `BF16→F32` y vocoder en NEON)**: incluye los kernels nuevos de las Fases 3–4. Verificación: `SNR >= 100 dB` frente a la ruta escalar.
- **Criterio de cierre**: `--self-test` y `cargo test` verdes en ARM64; A/B E2E escalar frente a NEON con RTF < 1.

---

## 7. Protocolo de verificación y perfilado entre Fases y Pasos

Cada Paso (en su alcance aplicable) y el cierre de cada Fase deben validarse con los comandos estándar del proyecto:

```bash
# Linux / macOS
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --release
./target/release/qwen_tts --self-test
./target/release/qwen_tts --matmat-bench
./target/release/qwen_tts --batch-test -d models/qwen3-tts-0.6b --quant-mixed
./target/release/qwen_tts --batch-bench -d models/qwen3-tts-0.6b --quant-mixed
```

En **Windows (`x86_64-pc-windows-gnu`, MSYS2 UCRT64)** manteniendo `OPENBLAS_NUM_THREADS` sin definir:

```powershell
$env:PATH = "C:\msys64\ucrt64\bin;$env:PATH"
Remove-Item Env:OPENBLAS_NUM_THREADS -ErrorAction SilentlyContinue

cargo clippy --workspace --all-targets --target x86_64-pc-windows-gnu -- -D warnings
cargo test --workspace --release --target x86_64-pc-windows-gnu
.\target\x86_64-pc-windows-gnu\release\qwen_tts.exe --self-test
.\target\x86_64-pc-windows-gnu\release\qwen_tts.exe --matmat-bench
.\target\x86_64-pc-windows-gnu\release\qwen_tts.exe --batch-test -d models\qwen3-tts-0.6b --quant-mixed
.\target\x86_64-pc-windows-gnu\release\qwen_tts.exe --batch-bench -d models\qwen3-tts-0.6b --quant-mixed
```

**Método** (detalle en `.work/plans/optimizations/00-plan.md`, §3):
- **Predecir antes de medir** y perfilar antes de optimizar algo cuyo coste no se conoce (instrumentación temporal que no se commitea).
- **Verificación por Paso con la herramienta mínima suficiente**: los comandos de arriba, hashes de audio, una corrida de commit para la RAM (determinista) o 3 corridas sueltas cuando el efecto esperado supera con holgura el ruido (±5 %). Prohibida la verificación por fuerza bruta.
- **Un único A/B E2E por fase**, al cierre, contra el cierre de la fase anterior: CLI `--stream`, `--seed 4 --quant-mixed -j 4`, textos de referencia corto y largo, warmups descartados, mediana de 3 pares en orden ABBA (cancela la deriva térmica), más una corrida del servidor por `POST /v1/tts/stream`.
- **Condición con carga**: el A/B se repite con 1 hilo copiando 256 MiB en bucle; ahí se verifica el requisito de RTF < 1.
- **Métricas**: `Prefill (ms)`, `Talker step (ms/f)`, `Code Predictor (ms/f)`, `Speech decoder (ms)`, `drain (ms)`, `TTFA (ms)` y `RTF`; pico de commit privado (`PeakPagefileUsage`, métrica primaria de RAM) y pico de *working set* (no regresión), en CLI y en el servidor en reposo y en pico.
- **Pico durante reasignación**: en Pasos que cambien capacidades iniciales, registrar el pico *durante* `ensure_capacity`.
- Nunca compilar ni lanzar tests en paralelo con una medición; los procesos del motor se lanzan con la salida redirigida a fichero.

