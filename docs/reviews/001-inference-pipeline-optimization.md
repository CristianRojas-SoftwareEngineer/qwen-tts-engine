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

El diseño actual ya incorpora optimizaciones clave para CPU: cuantización `Q4_0`/`INT8` con intrínsecos `AVX2+FMA`, pesos `safetensors` inmutables sobre `mmap` compartidos vía `Arc<Model06B>`, activación `FTZ/DAZ` (`MXCSR`/`FPCR`), un pool de hilos propio (`ComputePool`) y *tiling* L2 de `256 KB` en el vocoder. Sin embargo, el análisis estático del código fuente y del grafo de llamadas identifica **12 oportunidades concretas (`O1`–`O12`)** agrupadas en tres frentes:

1. **Ancho de banda de memoria (cuello de botella principal)**:
   - `int8_matmat_slice` y `q4_0_matmat_slice` (`crates/qwen-kernels/src/simd/avx2.rs`) iteran `for b in 0..b_dim` llamando a `matvec`, releyendo los mismos pesos `b_dim` veces (impacta a `cp_prefill2` con `B=2` en cada fotograma y a todo `ContinuousBatchScheduler`).
   - `batch_proj_q` (`crates/qwen-engine/src/batch.rs`) realiza una doble transposición (`batch_gather` y `quantize_act_int8_cols`) más `batch_scatter`.
   - `talker_prefill` (`crates/qwen-engine/src/talker.rs`) expande las 28 capas `BF16` a `F32` (`60.0 MiB`/capa = `3.36 GiB` de escritura+relectura extra en todo el prefill) y provoca el *page-in* de `840 MiB` `BF16` desde disco aunque el modelo corra cuantizado.
   - `talker.codec_head_bf16` (`3072 × 1024`, `6.0 MiB`) permanece en `BF16` sin cuantizar.

2. **Consumo de CPU (rutas escalares en *hot paths* y exceso de barreras)**:
   - La atención causal con ventana deslizante en `speech_decoder_decode_streaming` (`crates/qwen-engine/src/speech_decoder.rs:1679-1723`) corre en bucles escalares puros en lugar de reutilizar `KernelDispatch::causal_attention_windowed` (AVX2+FMA).
   - `swiglu_inplace`, `silu` (`crates/qwen-kernels/src/dispatch.rs:1204-1224`), `convnext_mlp` (`erff` escalar) y varias convoluciones del vocoder (`cs_dwconv`, `final_conv`) carecen de vectorización SIMD, acumulando **~331,776 llamadas escalares a `expf()` por fotograma** (~4.15M/s a RTF 1.0) solo en Talker + Code Predictor.
   - Se despachan hasta **466 barreras `ComputePool::parallel` por fotograma**, se invoca `ftz_on()` en cada `matvec_*`, `sd_pool` hace *spin-wait* concurrente con el productor y `batch_talker_step_impl` lanza `b_dim` barreras de atención por capa sin compactar slots inactivos.

3. **Consumo de memoria RAM**:
   - Cada `SessionScratch` preasigna **~181.6 MiB** al crearse (`112.0 MiB` de `TalkerKvCache` para `1024` posiciones + `60.0 MiB` de `pref_*_f32` + `8.0 MiB` de `EmbLruCache`), lo que suma **~726 MiB** con `--workers 4` antes de procesar la primera petición.
   - `BatchWorkspace` reserva **`122.08 MiB`/slot** contiguos (`~976.6 MiB` para `B=8`) y `generate_batch_multi` clonaba `TalkerKvCache` completos en un snapshot antes de copiar solo el prompt (**`112 MiB`/petición** pre-`O10a`; **`28,0 MiB`/petición** post-`O10a` con `KV_INITIAL_POSITIONS=256`; el snapshot completo se eliminó en `O11`, Fase 1 Paso 1.2).

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
| `TalkerKvCache::new()` | `28 × 256 × 1024 × 2 (K,V) × 2 B` (`KV_INITIAL_POSITIONS = 256` desde `O10a`; antes 1024 = `112.0 MiB`) | **`28.0 MiB`** |
| `TalkerBuffers::new()` (dec + `pref_*` de activación diferidos) | sin staging de pesos desde `O10b` (antes `(2048+1024+1024+2048+6144+3072)×1024 × 4 B` = `60.0 MiB` residentes) | **`~0.1 MiB`** *(staging F32 de `60.0 MiB` solo durante el prefill)* |
| `EmbLruCache::new()` | `2048 × 1024 × 4 B` | **`8.0 MiB`** |
| `CpKvCache` + `CpBuffers` + otros | `5 × 64 × 1024 × 2 × 2 B` + bufs | **`~1.5 MiB`** |
| **Total base por `SessionScratch`** | Antes de `ensure_prefill_capacity` / `ensure_conv_state` | **`~37.6 MiB`** tras `O10a`+`O10b` *(antes `~181.6 MiB`, ~726 MiB con 4 workers)* |
| `BatchWorkspace` KV (`kv_cap` inicial 256, crece hasta `kv_max = 1116`) | `B × 28 × 256 × 1024 × 2 × 2 B` (desde `O10c`; antes preasignado a `kv_max`: `122.08 MiB/slot`) | **`28.0 MiB / slot`** *(`224 MiB` para B=8)* |

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
   - **Palancas con efecto en `B=1`**: `O3` (`-3 MiB/frame`), `O5` (libera el `drain` cuando mejora el productor), `O4` (`prefill_ms`, crítico en textos breves) y `O1` solo en `cp_prefill2` (2 posiciones/fotograma; el matmat entero es exacto pero no más rápido: **1.26×** E2E y **1.2–1.7×** en microbanco). **Fundamento**: `Anexo A de phase-0/estado.md`.
2. **En servidor concurrente (`--batch-size N`, `ContinuousBatchScheduler`)**:
   - Aquí se produce la **mayor reducción de RTF agregado** (tiempo total / audio total generado). Actualmente, `int8_matmat_slice` y `q4_0_matmat_slice` re-leen los pesos `B` veces, multiplicando el tráfico de DRAM casi por `B`. Con `O1 + O2 + O7`, el costo de leer `~1.04 GiB/frame` se amortiza entre las `B` peticiones activas simultáneamente.
3. **Optimizaciones orientadas a RAM y arranque (`O10`, `O11`, `O12`)**:
   - Reducen cientos de MiB de huella residente (evitando *swapping* en equipos de poca memoria) y aceleran el arranque, sin alterar directamente la velocidad por fotograma cuando hay RAM libre.

---

## 4. Hallazgos detallados y oportunidades de optimización (`O1`–`O12`)

### 4.1 Ancho de banda de memoria (RAM ↔ Cachés ↔ CPU)

#### O1. Microkernels `matmat` reales para `INT8` y `Q4_0`
- **Ubicación**: `crates/qwen-kernels/src/simd/avx2.rs` (`int8_matmat_slice` L571–603, `q4_0_matmat_slice` L735–766) y `crates/qwen-kernels/src/simd/scalar.rs` (L134–166, L195–226).
- **Problema**: Para cada tile de `MM_TILE = 8` filas, el kernel ejecuta `for b in 0..b_dim` llamando a `int8_matvec_2row` / `q4_0_matvec_2row`. Esto relee las mismas filas de peso `w` desde caché/DRAM `b_dim` veces y en `Q4_0` recalcula `q4_0_sumq` en cada iteración, limitando los resultados de `--matmat-bench` (`crates/qwen-kernels/src/bench.rs`).
- **Impacto**: Ahorra **`~55.3 MiB/frame`** en `cp_prefill2` (`B=2`) y reduce el tráfico de `B` relecturas a **1** (techo teórico `B×`). Medido: **1.26×** E2E en `--batch-bench` y **1.2–1.7×** en microbanco `INT8`/`INT4`. `matmat_bf16` queda fuera de alcance: **0.76–0.97×** en este hardware y descartado en el gemelo C. **Fundamento**: `Anexo A de phase-0/estado.md`.
- **Propuesta**: Escribir microkernels `AVX2+FMA` especializados para `B=2, 4, 8`: cargar 32 bytes de peso (`INT8` o un `Q4_0Block`), decodificar nibbles una sola vez y acumular `B` vectores `a_b` en registros YMM independientes, precalculando `sumq[0..b_dim]` una vez por bloque. Mantener equivalente estructurado en `simd/scalar.rs` para ARM64 y *fallback* x86.
- **Contrato numérico**: **Compatible 100 %** (`L2_rel == 0.0` en `INT8`, `L2_rel <= 1e-6` en `Q4_0` en `--self-test`).
- **Estado**: `vigente` (solo `INT8`/`Q4_0`). **Efecto medido**: 1.26× E2E. **Fundamento**: `.work/plans/optimizations/phase-0/estado.md`. **Si se pierde**: volver a leer los pesos `B` veces por lote.

#### O2. Eliminar la doble transposición en `batch_proj_q` + `quantize_act_int8_cols`
- **Ubicación**: `crates/qwen-engine/src/batch.rs:53-115` y `crates/qwen-kernels/src/dispatch.rs:357-379`.
- **Problema**: `BatchWorkspace` guarda activaciones en `[B, cols]`. `batch_proj_q` las transpone a `[cols, B]` (`batch_gather`), `quantize_act_int8_cols` las retranspone a `[B, cols]` en `QXT_TLS` elemento a elemento, y `batch_scatter` transpone `[rows, B] → [B, rows]`.
- **Propuesta**: Cuantizar directamente cada fila contigua `src[b * cols..(b+1) * cols]` con `quantize_act_int8` AVX2 sin `batch_gather` y alinear la salida con el microkernel de `O1`.
- **Contrato numérico**: **Bit-idéntico**.
- **Estado**: `pendiente`. **Efecto medido**: sin medir en Fase 0. **Fundamento**: depende de `O1`.

#### O3. Cuantización de `talker.codec_head_bf16` bajo `--quant-mixed` / `--int8`
- **Ubicación**: `crates/qwen-engine/src/model.rs:1729` y `crates/qwen-engine/src/pipeline.rs:697-704`.
- **Problema**: En cada fotograma lee `6.0 MiB` (`3072 × 1024 × 2 B`) en `BF16` con FMA de coma flotante.
- **Propuesta**: Soportar `QuantWeight::Int8` para `codec_head` (baja a **`3.0 MiB/frame`** con aritmética entera `maddubs`), manteniendo el criterio conservador de [`NUMERIC_CONTRACT.md`](../NUMERIC_CONTRACT.md) §1.3 de evitar `Q4_0` en cabezas de salida salvo validación explícita.
- **Contrato numérico**: **Modifica [`NUMERIC_CONTRACT.md`](../NUMERIC_CONTRACT.md) §1.3** (cambia la precisión de logits `code0`); requiere evaluación A/B de calidad.
- **Estado**: `pendiente`. **Efecto medido**: sin medir en Fase 0. **Fundamento**: `codec_head` lee `6.0 MiB/frame` en `BF16`; `INT8` los reduce a `3.0 MiB/frame`.

#### O4. Prefill cuantizado o conversión `BF16→F32` por bloques L2 en `talker_prefill`
- **Ubicación**: `crates/qwen-engine/src/talker.rs:64-82` y `crates/qwen-kernels/src/dispatch.rs:1268-1341`.
- **Problema**: `convert_talker_layer_bf16_to_f32` expande `60.0 MiB/capa` a `F32` (`1,680 MiB` escritos y `1,680 MiB` re-leídos en 28 capas) y fuerza el *page-in* de `840 MiB` `BF16` del `mmap` aunque el Talker corra en `Q4_0`.
- **Propuesta**:
  - *Opción A (bit-exacta en F32)*: convertir `BF16→F32` en bloques pequeños de L2 dentro de `sgemm` sin materializar la capa completa (conserva `F32` exacto y libera `60 MiB` de `TalkerBuffers`).
  - *Opción B (cuantizada)*: `descartado`. No habilitar `matmat_q4_0` en el prefill bajo `--quant-mixed`.
- **Contrato numérico**: Opción A es **compatible**; Opción B altera la precisión del prefill y requiere validación en `NUMERIC_CONTRACT.md`.
- **Estado**: Opción A `vigente`, Opción B `descartado`. **Efecto medido**: la Opción B produce `+4.7 st` en `ryan`, WER `+1.8/+2.7 pp` y prefill largo `+41 %`. **Fundamento**: `Anexo A §A.3 de phase-0/estado.md`. **Si se pierde**: regenerar audio con el tono alterado.

---

### 4.2 Consumo de CPU (vectorización SIMD y sincronización)

#### O5. Vectorizar la atención con ventana deslizante en `speech_decoder_decode_streaming`
- **Ubicación**: `crates/qwen-engine/src/speech_decoder.rs:1679-1723` vs `733-746`.
- **Problema**: `speech_decoder_decode` (no streaming) usa `d.causal_attention_windowed` (`avx2.rs:1020-1078`), mientras `speech_decoder_decode_streaming` usa bucles escalares `for dd in 0..DEC_HEAD_DIM` y `exp()` escalar sobre hasta `72` claves × `16` cabezas × `8` capas.
- **Propuesta**: Reutilizar `d.causal_attention_windowed` (adaptando el offset de base del KV cache compactable `st.kv_base`).
- **Contrato numérico**: **Compatible (`SNR >= 100 dB`)**.
- **Estado**: `implementado` (Paso 1.5). **Efecto medido (Fase 1, Paso 1.5)**: `implementado` como consolidación del camino canónico, con rendimiento **neutro**. A/B intercalado (3 pares × corta/larga): `decoder_ms` (tiempo de cómputo puro del decoder) −13 ms en la corta y +163 ms en la larga (pares +771 / +163 / −54; el control del par 1 fue anómalo), `drain` −5 / −5 ms, RTF −0,01 / +0,02. La atención (16 cabezas × ≤72 claves × 64 × 8 capas ≈ 1,2 MFLOP/frame) no es el punto caliente del decoder: su coste está en las convoluciones del vocoder (`O8`). Paridad: códigos idénticos (mismo número de frames, decoder aguas abajo y sin realimentación); PCM con SNR 101,5 / 101,1 dB frente a `baseline_*.wav` (solo ±1 LSB), bit-idéntico entre corridas, también con compactación (`kv_base > 0`, texto de 417 frames: 99,7 dB frente al binario viejo, 379 muestras de 800 640 con ±1 LSB, en el suelo de cuantización PCM16); `test_decoder_streaming_equivalence` (SNR ≥ 100 dB en f32) en verde. **Consecuencia**: el KPI de `decoder_ms` −10 % de la Fase 1 no lo puede aportar `O5`; se traslada a `O8` (vocoder AVX2, Fase 3). **Fundamento**: `.work/plans/optimizations/phase-1/estado.md` (Paso 1.5), `Anexo A §A.4 de phase-0/estado.md`.

#### O6. Vectorizar `swiglu_inplace`, `silu` y `convnext_mlp` GELU con `AVX2+FMA`
- **Ubicación**: `crates/qwen-kernels/src/dispatch.rs:1204-1224`, `crates/qwen-engine/src/speech_decoder.rs:219-221, 1786-1788`.
- **Problema**: `swiglu_inplace` ejecuta ~`331,776` `expf()` escalares/frame en Talker + CP; `speech_decoder_decode_streaming` ejecuta `expf()` y `erff()` escalares en FFN y `convnext_mlp`.
- **Propuesta**: Despachar `swiglu_inplace` y `silu` por `KernelDispatch` con aproximación polinomial `AVX2+FMA` de `expf` (8 lanes, similar a `vsin2_avx2` en `avx2.rs:1245`), y vectorizar GELU + LayerNorm en `convnext_mlp`.
- **Contrato numérico**: Diferencias de 1 ULP en `expf()` pueden variar desempates en muestreo respecto a `libm`; requiere medir ULP/SNR y validar que no degrada pruebas de regresión.
- **Estado**: `condicionado`. **Efecto medido**: SwiGLU es el **1.7 %** del CP en el gemelo C; repartirlo entre hilos ahorró **0.68 ms/f**. Ahorro esperado ≤1 ms/f: `gate_up` (37 %) y QKV (27 %) concentran el coste del CP. **Fundamento**: `Anexo A §A.1 de phase-0/estado.md`.

#### O7. Reducir barreras de `ComputePool`, eliminar `ftz_on()` en `matvec_*` y paralelizar batch unificado
- **Ubicaciones**:
  1. `crates/qwen-kernels/src/dispatch.rs:668, 709, 757, 823, 854, 894` (`ftz_on()` repetido ~466 veces/frame).
  2. `crates/qwen-engine/src/batch.rs:252-426`: `causal_attention_bf16kv_heads_parallel` se invoca dentro de `for b in 0..b_dim` (`b_dim` barreras por capa) y procesa proyecciones para slots con `active[b] == false`.
  3. `crates/qwen-engine/src/code_predictor.rs:206-360`: ~325 barreras por fotograma sobre matrices de `1024`–`3072` filas.
- **Propuestas**:
  - ~~Dejar `ftz_on()` solo al inicializar hilos/runtimes~~ (**descartado por evidencia**, Fase 1 Paso 1.6): `ftz_on()` es `stmxcsr` + comparación (el `ldmxcsr` solo se ejecuta si FTZ/DAZ está apagado): ~466 llamadas/frame ≈ 5 k ciclos ≈ 2 µs/frame frente a ~45 ms/frame (≈ 0,005 %, por debajo de cualquier resolución de A/B). La auditoría de hilos muestra que `qwen-batch-sched` y `qwen-http-single` (`scheduler.rs`) usan el `ComputeRuntime` creado en `main` y no activan FTZ al nacer: para ellos, la llamada en la entrada del kernel es la única garantía. Retirarla obligaría a añadir `ftz_on()` en cada `spawn` (más superficie de error, con riesgo de denormales) a cambio de nada medible.
  - En `batch_talker_step_impl`, compactar slots activos (`n_act`) antes de `batch_proj_q` y paralelizar `(slot activo, cabeza)` en una sola barrera por capa.
  - Ajustar el umbral de paralelización en `CodePredictor` para reducir salidas del pool (y combinar la fusión `swiglu + quantize_act_int8` en `down_proj` junto con `O6` en la **Fase 3**).
- **Contrato numérico**: **Bit-idéntico**.
- **Estado**: `vigente`. **Efecto medido**: sin medir en Fase 0. **Fundamento**: el gemelo C registra ~495 despachos al pool por fotograma. **Si se pierde**: barreras redundantes por capa y por slot inactivo.

#### O8. Presupuesto de espera en `sd_pool` y vectorizar convoluciones del `SpeechDecoder`
- **Ubicación**: `crates/qwen-kernels/src/dispatch.rs:1411-1417`, `crates/qwen-engine/src/speech_decoder.rs:32-43, 505-525, 1013-1057, 1366-1383`.
- **Problema**: `sd_pool` hereda un presupuesto de espera activa; `cs_dwconv`, `final_conv`, `conv_add_bias` y `add_residual_scaled` son escalares.
- **Propuesta**: mantener la espera activa acotada en `sd_pool` (medido `0` frente a `4 096` en la Fase 1: **descartado**, ver Estado); vectorizar en AVX2 `cs_dwconv`, `final_conv` y `add_residual_scaled` sobre el eje temporal `t`.
- **Contrato numérico**: **Compatible (`SNR >= 100 dB`)**.
- **Estado**: presupuesto de espera `descartado` (Fase 1 Paso 1.7: A/B intercalado (3 pares × corta/larga, variante con `sd_pool = ComputePool::with_spin_budget(sd_nt, 0)`): `decoder_ms` **+44 / +110 ms** (pares cortos +51 / +44 / +2), Talker +0,2 / +0,1 ms/f, CP +0,7 / +0,1 ms/f, RTF +0,01 / 0,00; `drain` −5 / −8 ms. Paridad 12/12 (bit-idéntico). Confirma el prior del motor C: en Windows, dormir el pool cuesta más que la espera activa acotada (4 096).); vocoder AVX2 `vigente` (Paso 3.3). **Efecto medido (gemelo C)**: dormir el pool cuesta **3–4 %** de RTF (0.76 → 0.79 sin carga; 0.87 → 0.91 con 4 núcleos); `4 096` rinde más que `0` y que `20 000`. **Fundamento**: `Anexo A §A.3 de phase-0/estado.md`. **Si se pierde**: regresión silenciosa en Windows.

#### O9. Contención global de `openblas_set_num_threads` en `WorkerPoolScheduler`
- **Ubicación**: `crates/qwen-kernels/src/blas.rs:161-175` y `crates/qwen-app/src/server/scheduler.rs:92-109`.
- **Problema**: `FfiBlas::set_threads` modifica el estado global de proceso de OpenBLAS. Con `--workers N > 1`, múltiples workers se pisan entre prefill y generación. Un `OMP_NUM_THREADS` ausente en el lanzador compromete los hilos de OpenMP antes de `main`.
- **Propuesta**: fijar `OMP_NUM_THREADS = -j` en el lanzador y `openblas_set_num_threads` una sola vez al iniciar cuando `n_workers > 1` (sin reconfigurar por petición). `OPENBLAS_NUM_THREADS` permanece sin definir. `--batch-size N` (o `faer-blas`) es la opción preferente en concurrencia.
- **Contrato numérico**: **Bit-idéntico** por petición con el mismo número de hilos efectivo.
- **Estado**: `vigente`. **Efecto medido**: en Windows `OMP_NUM_THREADS = -j` reduce **1 410 MiB** comprometidos y el tiempo un **6 %**, sin cambiar el audio. **Fundamento**: `Anexo A §A.6 de phase-0/estado.md`.

---

### 4.3 Consumo de memoria RAM

#### O10. Reducir la huella base de `SessionScratch` de `~181.6 MiB` a `~37.6 MiB`
- **Ubicación**: `crates/qwen-engine/src/model.rs:600-609, 707-756, 861-937`.
- **Problema**: `TalkerKvCache::new()` asigna `1024` posiciones (`112.0 MiB`); `TalkerBuffers::new()` asigna `pref_*_f32` (`60.0 MiB`) de inmediato; `SdStreamState` retiene 14 `Vec<f32>` temporales (`tmp_*`) por cada `ContinuousSlot` aunque decodifiquen secuencialmente con un único `sd_ws`.
- **Propuesta**:
  - Cambiar `KV_INITIAL_POSITIONS` a `256` (`28.0 MiB`, ahorro de **`84.0 MiB`/sesión**, creciendo dinámicamente vía `ensure_capacity`).
  - Sacar el staging `pref_*_f32` de `TalkerBuffers`: transitorio y local de `talker_prefill` (se asigna al entrar y se libera al salir). Asignarlo «bajo demanda y retenerlo» no ahorraría nada en el pico, porque toda síntesis prefila (ahorro de **`60.0 MiB`/sesión** fuera del prefill).
  - ~~Mover los 14 `tmp_*` de `SdStreamState` a `DecoderWorkspace`~~ (**descartado por evidencia**: nacen vacíos y con `stream_chunk_frames = 10` suman ≈ 444 KiB/slot, < 2 MiB en batch4). En su lugar, `O10c` ataca el driver real de la RAM por slot: el KV de `BatchWorkspace` arranca en `KV_INITIAL_POSITIONS` y crece bajo demanda hasta `kv_max`, con un bloque `[NUM_LAYERS, kv_cap, KV_DIM]` por slot (el crecimiento reasigna slot a slot). Con `O10a`+`O10b`, la huella base de `SessionScratch::new()` baja de `~181.6 MiB` a **`~37.6 MiB`**.
- **Contrato numérico**: **Bit-idéntico**.
- **Estado**: `O10a` `implementado` (Paso 1.1); `O10b` `implementado` (Paso 1.3); `O10c` `implementado` (Paso 1.4, KV de `BatchWorkspace` bajo demanda; la parte `tmp_*` se descarta por evidencia). **Efecto medido `O10c`** (A/B ABBA, hashes `bm_0..7` idénticos): servidor `--batch-size 4` **−377 MiB** de pico de commit (2 883→2 506) y de reposo, WS sin cambio, RTF −2,2 %; con texto de crecimiento (>256 posiciones) −265 MiB y RTF −3,3 %; CLI `--batch-multi-test 8` largo −343,6 MiB, y con crecimiento hasta `kv_max` ±0 (pico transitorio acotado a un slot: WS +15 MiB). Una primera versión con bloques por (slot, capa) de 512 KiB subía el WS en reposo +112 MiB: el heap sirve bloques de ese tamaño desde segmentos y los pone a cero tocándolos; los bloques por slot (≥ 14 MiB) usan páginas cero diferidas. **Efecto medido `O10b`** (A/B intercalado, pico exacto de commit `PeakPagefileUsage`, paridad 12/12 con la referencia): CLI **−60,3 / −60,1 MiB** (corta / larga; WS pico −60 MiB, porque el pico ocurre en el decoder, cuando el staging ya se liberó); servidor `--workers 4` **−241 MiB** de pico de commit (1 878→1 637) y de reposo; `--batch-size 4` **−120 MiB** (3 003→2 883). `prefill_ms` sin cambio. RTF por petición en workers4 **+1,9 %** de mediana pareada (ABBA, 4 repeticiones, rango +0,4…+3,5 %, sin calma plena), dentro de la compuerta ±3 %; CLI y batch4 neutros. **Efecto medido `O10a`**: ahorra `84 MiB`/sesión solo si no hay crecimiento. `256` cubre 211 frames de audio + prompt; un texto de 450 caracteres usa ~319. El gemelo C probó `512`, creció a `1024` en textos largos y lo descartó; su valor validado es `1024`, que este motor ya tiene. **Prueba obligatoria**: medir el pico *durante* `ensure_capacity`. **Fundamento**: `Anexo A §A.3 de phase-0/estado.md`. **Si se pierde**: reasignación del KV en frases largas.

#### O11. Snapshot compacto de KV en `generate_batch_multi`
- **Ubicación**: `crates/qwen-engine/src/batch.rs:163, 200-219, 1144-1181` (y `run_continuous_batch` en `1485`, ruta directa sin snapshot).
- **Problema**: `generate_batch_multi` materializaba el KV completo del scratch por slot en `kv_snapshots` antes de copiar solo `prompt_lens[b]` posiciones a `BatchWorkspace` (**`112 MiB`/petición** pre-`O10a`, **`896 MiB`** para 8; post-`O10a` `28,0 MiB`/slot, **`224 MiB`** por grupo de 8). `run_continuous_batch`/`BatchWorkspace` preasignan `122.08 MiB`/slot contiguos (`kv_max = 1116`), huella independiente del snapshot.
- **Implementación (Fase 1, Paso 1.2)**: se elimina el snapshot completo y se conserva un staging compacto con layout `(layer × pl) × KV_DIM` (`NUM_LAYERS * pl * KV_DIM` elementos `u16` por buffer), cargado al slot por la **única ruta** existente `load_slot_kv(b, sk, sv, pl, pl)` (`src_kv_max = prompt_len`, stride de fuente igual al de destino lógico del prompt). Se mantiene el staging porque `bw.kv_max` depende del `max_pl` de todo el grupo y se conoce solo tras prefilar todos los slots; la copia directa scratch→`bw` exigiría `BatchWorkspace` ragged (stride por slot) o pre-dimensionarlo a 1116 (peor para RAM) — opción futura junto a bloques KV paginados si se escala a `B >= 16`.
- **Contrato numérico**: **Bit-idéntico**.
- **Estado**: `implementado` (Fase 1, Paso 1.2). **Alcance**: modo batch del CLI (`generate_batch_multi` / `--batch-multi-test` / `--batch`), **no** el pico del servidor `--batch-size` (que usa `run_continuous_batch` sin snapshot; su pico lo gobierna `BatchWorkspace.kv_k/kv_v` ≈ 488 MiB committed con `kv_max=1116`, a tratar en `O10c`). **Efecto medido** (A/B intercalado, pico de privados pareado, 3 pares × 2 textos, hashes 48/48 por texto): **corto (pl=21) −206,0 MiB**, **largo (pl=44) −169,5 MiB** de mediana (Δ en la fase de snapshots 185,9 MiB, acorde a 917 504 B × (256 − pl)); wall ±3%; servidor workers4/batch4 sin cambio (commit pico exacto −2,8 / 0,0 MiB). **Compuerta**: hashes bit-idénticos y no-regresión de rutas no tocadas (no «WS ≤ 2 250 MiB»). **Fundamento**: `.work/plans/optimizations/phase-0/estado.md`, `.work/plans/optimizations/phase-1/estado.md`. **Si se pierde**: volver al snapshot completo y al pico de privados medido en el control.

#### O12. Carga paralela de cuantización al iniciar `Model06B::load`
- **Ubicación**: `crates/qwen-engine/src/model.rs:1487-1489` y `crates/qwen-kernels/src/quant.rs:144-236`.
- **Problema**: `quantize_bf16_to_q4_0_into` y `quantize_bf16_to_int8_into` se ejecutan secuencialmente en un hilo durante `Model06B::load`.
- **Propuesta**: Paralelizar por capas/filas sobre `runtime.pool` durante la carga (acelera ~3–4× el startup en `--quant-mixed`).
- **Contrato numérico**: **Bit-idéntico**.
- **Estado**: `pendiente`. **Efecto medido**: sin medir en Fase 0; la carga del modelo osciló entre 4 400 y 5 397 ms. **Fundamento**: `.work/plans/optimizations/phase-0/estado.md`.

---

## 5. Matriz de priorización y dependencias

| Prioridad | ID | Estado | Optimización | Recurso | Dependencias | Riesgo numérico | Esfuerzo |
|---|---|---|---|---|---|---|---|
| **1** | **O5** | `implementado` (neutro en RTF; camino único) | `causal_attention_windowed` en `SpeechDecoder` streaming | **CPU** | Ninguna | Ninguno (`SNR >= 100 dB`) | Bajo |
| **2** | **O1** | `vigente` | Microkernels `matmat_int8` / `matmat_q4_0` reales (sin `bf16`) | **Ancho de banda + CPU** | Ninguna | Ninguno (bit-idéntico) | Medio |
| **3** | **O2** | `pendiente` | Eliminar doble transposición en `batch_proj_q` | **Ancho de banda + CPU** | Depende de **O1** | Ninguno (bit-idéntico) | Bajo–Medio |
| **4** | **O10** | `implementado` (`O10a`/`O10b`/`O10c`) | `KV_INITIAL_POSITIONS`, staging `pref_*_f32` transitorio, KV de `BatchWorkspace` bajo demanda | **RAM** | Ninguna | Ninguno (bit-idéntico) | Bajo |
| **5** | **O11** | `implementado` | Snapshot compacto de KV en `generate_batch_multi` (CLI batch) | **RAM** | Ninguna | Ninguno (bit-idéntico) | Bajo |
| **6** | **O7** | `vigente` | Quitar `ftz_on()` hot path, compactar batch y barrera única atención | **CPU + Ancho de banda** | Ninguna (mejor tras **O1**) | Ninguno (bit-idéntico) | Bajo–Medio |
| **7** | **O9** | `vigente` | Fijar `OMP_NUM_THREADS = -j` en el lanzador y `openblas_set_num_threads` una sola vez | **CPU (servidor)** | Ninguna | Ninguno | Bajo |
| **8** | **O12** | `pendiente` | Cuantización paralela en `Model06B::load` | **Tiempo de carga** | Ninguna | Ninguno (bit-idéntico) | Bajo |
| **9** | **O6** | `condicionado` | Vectorizar `swiglu_inplace`, `silu`, GELU en `AVX2+FMA` | **CPU** | Ninguna | Revisar ULP vs `libm` | Medio |
| **10** | **O3** | `pendiente` | Cuantizar `talker.codec_head` a `INT8` | **Ancho de banda** | Ninguna | Requiere `NUMERIC_CONTRACT.md` | Bajo |
| **11** | **O4** | `vigente` / `descartado` | Opción A: tiling L2 en `F32`; Opción B: `matmat_q4_0` en prefill | **Ancho de banda + RAM RSS** | Opción B depende de **O1** | Opción A: ninguno; Opción B: requiere contrato | Medio–Alto |
| **12** | **O8** | espera `descartado`; vocoder `vigente` | Presupuesto de espera en `sd_pool` + convoluciones vocoder AVX2 | **CPU** | Ninguna | Ninguno (`SNR >= 100 dB`) | Bajo–Medio |

**Fundamento de los estados**: `Anexo A §A.3 de phase-0/estado.md`. `O8` se ejecuta en último lugar y `O4` Opción B no se habilita.

---

## 6. Planificación de implementación progresiva (Fases y Pasos)

Las 12 mejoras **no deben aplicarse en un único cambio monolítico**: mezclar ajustes de memoria, microkernels SIMD y cambios de precisión numérica dificultaría aislar regresiones de determinismo o concurrencia. El plan se estructura en dos niveles coherentes:

- **Nivel Mapa (Fases 0–4)**: agrupan los cambios por tipo de riesgo y dependencias técnicas, con un criterio de cierre claro por etapa.
- **Nivel Ejecución (Pasos `X.Y`)**: dentro de cada Fase, cada mejora o sub-mejora se ejecuta como **1 Paso aislado** siguiendo siempre el mismo bucle de 5 reglas:
  1. **Chequeo previo**: confirmar archivos exactos, invariantes de [`NUMERIC_CONTRACT.md`](../NUMERIC_CONTRACT.md) y riesgos de borde antes de editar.
  2. **Implementar solo ese Paso**: sin mezclar cambios de otros pasos.
  3. **Compuerta de exactitud**: ejecutar `clippy`, `cargo test --workspace --release`, `--self-test` y comparar paridad frente al paso anterior o línea base.
  4. **Compuerta de rendimiento (A/B)**: medir mediana de 3 corridas con parámetros fijos frente al estado inmediatamente anterior.
  5. **Decisión binaria**: `commit` si aporta mejora comprobada sin regresiones; revertir de inmediato si no aporta ganancia o degrada otra métrica.

---

### Fase 0 — Congelación de línea base y perfilado real (sin cambios de código)
- **Propósito**: fijar una referencia reproducible en el hardware objetivo antes de alterar código, evitando optimizar a ciegas.
- **Pasos**:
  - **Paso 0.1 (Capturar línea base)**: ejecutar `--seed 4 --quant-mixed -j 4` en frase corta, frase larga e ICL (si aplica), además de `--matmat-bench` y `--batch-bench`; registrar `prefill_ms`, `talker_ms`, `cp_ms`, `decoder_ms`, `drain_ms`, `TTFA`, `RTF` y pico de RSS (CLI y servidor `--workers 4` / `--batch-size 4`), conservando códigos códec y WAV de referencia.
  - **Paso 0.2 (Perfilado 80/20)**: contrastar qué bloque domina el tiempo real de pared para validar el orden de los pasos siguientes y detectar mejoras cuyo peso sea marginal (`< 2–3 %`).
- **Criterio de cierre**: tabla de métricas base documentada y archivos de referencia guardados.

**Techo del consumidor**: con `prefill 475`, productor `9 666`, `SpeechDecoder 10 185` y `drain 388` ms, un CP −20 % sin tocar el decoder deja el wall gobernado por el decoder (RTF ≈ **0.68–0.70**, −3 %); el mismo CP −20 % con decoder −20 % da RTF ≈ **0.59–0.60** (−18 %). `O5` es requisito de las Fases 2–3: sin él el ahorro del productor se traslada al `drain`. **Fundamento**: `Anexo A §A.4 de phase-0/estado.md`.

---

### Fase 1 — *Quick wins* de bajo riesgo en CPU y RAM
- **Alcance**: `O10`, `O11`, `O5`, `O7 (retirar ftz_on() redundante)` y `O8 (presupuesto de espera en sd_pool, en último lugar y condicionado)`.
- **Prerequisitos**: Fase 0 completada.
- **Pasos (ordenados de menor a mayor riesgo de borde)**:
  - **Paso 1.1 (`O10a`)**: **Qué cambia**: `KV_INITIAL_POSITIONS` de `1024` a `256` (`crates/qwen-engine/src/model.rs:33`). **Por qué**: −84.0 MiB/sesión. **Riesgo a vigilar**: `256` cubre 211 frames de audio + prompt; un texto de 450 caracteres usa ~319 → prueba de crecimiento obligatoria y medición del pico durante `ensure_capacity`. **Compuerta**: RSS sin subir durante el crecimiento. **Decisión**: conservar solo si no aparece ese pico.
  - **Paso 1.2 (`O11`)**: **Qué cambia**: sustituir el snapshot completo de `generate_batch_multi` por un snapshot compacto `(layer × pl) × KV_DIM` cargado por la ruta única `load_slot_kv(b, sk, sv, pl, pl)` (`crates/qwen-engine/src/batch.rs`); `run_continuous_batch` (servidor `--batch-size`) no usa snapshots y no se modifica. **Por qué**: reducir el pico de privados del modo batch del CLI (el pico del servidor lo gobierna `BatchWorkspace`, ~488 MiB committed con `kv_max=1116`). **Riesgo a vigilar**: offsets de capa/stride (`layer * src_stride * KV_DIM` en origen). **Compuerta**: hashes bit-idénticos y Δ de pico privado pareado en banda (medido: −206,0 MiB corto pl=21, −169,5 MiB largo pl=44; servidor ± commit exacto). **Decisión**: conservado (implementado).
  - **Paso 1.3 (`O10b`)**: **Qué cambia**: el staging F32 de pesos del prefill (`60.0 MiB`) sale de `TalkerBuffers` y pasa a ser local y transitorio de `talker_prefill` (`crates/qwen-engine/src/talker.rs`). **Por qué**: el pico de commit ocurre en el decoder, después del prefill; liberar el staging lo baja (CLI −60 MiB, workers4 −241 MiB, batch4 −120 MiB). **Compuerta**: pico de commit CLI y servidor; `prefill_ms` y RTF ±3 %. **Decisión**: `implementado`.
  - **Paso 1.4 (`O10c`)**: **Qué cambia**: el KV de `BatchWorkspace` (`crates/qwen-engine/src/batch.rs`) separa `kv_max` (límite) de `kv_cap` (posiciones asignadas, inicial `KV_INITIAL_POSITIONS`) y crece bajo demanda con `ensure_kv_capacity`, en bloques por slot. Los `tmp_*` de `SdStreamState` se descartan (≈ 444 KiB/slot). **Por qué**: el KV preasignado a 1 116 posiciones (122 MiB/slot) era el driver del pico batch. **Compuerta**: pico de commit batch4 y BatchMulti; WS y RTF sin regresión; hashes idénticos. **Decisión**: `implementado` (batch4 −377 MiB).
  - **Paso 1.5 (`O5`)**: **Qué cambia**: reemplazar la atención escalar de `speech_decoder_decode_streaming` (`crates/qwen-engine/src/speech_decoder.rs:1679-1723`) por `d.causal_attention_windowed` con `q_offset = st.kv_len - st.kv_base` y `seq_k = st.kv_len + new_frames - st.kv_base`. **Por qué**: el decoder gobierna el wall cuando mejora el productor. **Riesgo a vigilar**: compactación con `st.kv_base > 0`. **Compuerta**: paridad (`SNR >= 100 dB`, bit-identidad entre corridas) y `decoder_ms`/`drain` sin regresión. **Decisión**: `implementado`; rendimiento neutro: la atención no es el punto caliente del decoder, cuyo coste está en `O8`.
  - **Paso 1.6 (`O7` parcial)**: **Decisión**: `descartado` sin cambio de código. `ftz_on()` es `stmxcsr` + comparación (el `ldmxcsr` solo se ejecuta si FTZ/DAZ está apagado): ~466 llamadas/frame ≈ 5 k ciclos ≈ 2 µs/frame frente a ~45 ms/frame (≈ 0,005 %, por debajo de cualquier resolución de A/B). La auditoría de hilos muestra que `qwen-batch-sched` y `qwen-http-single` (`scheduler.rs`) usan el `ComputeRuntime` creado en `main` y no activan FTZ al nacer: para ellos, la llamada en la entrada del kernel es la única garantía. Retirarla obligaría a añadir `ftz_on()` en cada `spawn` (más superficie de error, con riesgo de denormales) a cambio de nada medible.
  - **Paso 1.7 (`O8` parcial)**: **Qué cambia**: medir `spin_budget = 0` frente a `4 096` en `sd_pool` (`crates/qwen-kernels/src/dispatch.rs:1417, 1440`). **Por qué**: el valor por defecto rinde más en Windows. **Riesgo a vigilar**: `decoder_ms` y `drain` saturan. **Compuerta**: sin regresión en ambas condiciones. **Decisión**: `descartado` (revertido). A/B intercalado (3 pares × corta/larga, variante con `sd_pool = ComputePool::with_spin_budget(sd_nt, 0)`): `decoder_ms` **+44 / +110 ms** (pares cortos +51 / +44 / +2), Talker +0,2 / +0,1 ms/f, CP +0,7 / +0,1 ms/f, RTF +0,01 / 0,00; `drain` −5 / −8 ms. Paridad 12/12 (bit-idéntico). Confirma el prior del motor C: en Windows, dormir el pool cuesta más que la espera activa acotada (4 096).
- **Criterio de cierre de Fase 1**:
  1. `cargo test --workspace --release` y `qwen_tts --self-test` verdes.
  2. **Códigos códec** (`Talker + CodePredictor`) **bit-idénticos** frente a la Fase 0 con `--seed 4`; **audio PCM** del `SpeechDecoder` con **`SNR >= 100 dB`** (Paso 1.5) y **bit-idéntico entre corridas** del nuevo binario.
  3. Pico `--batch-size 4` ≤**2 250 MiB** y `--workers 4` ≤**2 100 MiB**; RSS CLI ≤**2 070 MiB**; `decoder_ms` largo ≤**9 150 ms**, `drain` ≤**350 ms**; RTF corta ≤**0.78** sin regresión en larga.
  4. Compuerta con 1 hilo copiando 256 MB: el RTF no empeora respecto a la Fase 0 en la misma condición.
- **Resultado del cierre (2026-11-18)**: `O10a`, `O11`, `O10b`, `O10c` y `O5` implementados; `O7` parcial (`ftz_on`) descartado y `O8` parcial (espera de `sd_pool`) revertido. Frente al binario de Fase 0 en A/B intercalado: pico de commit CLI −144 MiB; `--workers 4` 2 213 → 1 638 MiB; `--batch-size 4` 3 172 → 2 506 MiB; WS sin regresión; RTF −1,5 % (L1 sin regresión); `drain` 307 / 313 ms. Los umbrales de RSS de este criterio se fijaron en working set; el ahorro es commit diferido (métrica primaria adoptada en la Fase 1). El objetivo de `decoder_ms` no se cumple (9 266 ms): su coste está en el vocoder (`O8`, Paso 3.3), no en la atención (`O5`, neutro). **Fundamento**: `.work/plans/optimizations/phase-1/estado.md`.

---

### Fase 2 — Microkernels `matmat` reales y escalado del servidor
- **Alcance**: `O1` (solo `INT8`/`Q4_0`; `matmat_bf16` queda fuera de alcance), `O2`, `O7 (compactación de slots activos + barrera única de atención batch + umbral en CP)` y `O9`.
- **Prerequisitos**: Fase 1 completada.
- **Pasos**:
  - **Paso 2.1 (`O1a`)**: implementar microkernel real `int8_matmat_slice` para `B=2,4,8` en `crates/qwen-kernels/src/simd/avx2.rs` (y paridad en `simd/scalar.rs`).
  - **Paso 2.2 (`O1b`)**: implementar microkernel real `q4_0_matmat_slice` para `B=2,4,8` con `sumq[0..b_dim]` precalculado por bloque (`avx2.rs` y `scalar.rs`).
  - **Paso 2.3 (`O2`)**: eliminar la doble transposición en `batch_proj_q` cuantizando directamente filas contiguas `[B, cols]` (`crates/qwen-engine/src/batch.rs` y `dispatch.rs`).
  - **Paso 2.4 (`O7` batch)**: compactar slots activos (`n_act`) antes de `batch_proj_q` y unificar `causal_attention_bf16kv_heads_parallel` en una sola barrera por capa en `batch_talker_step_impl` (`crates/qwen-engine/src/batch.rs`).
  - **Paso 2.5 (`O7` CP)**: ajustar umbral de paralelización en `CodePredictor` (`crates/qwen-engine/src/code_predictor.rs`) y verificar con `--batch-test`.
  - **Paso 2.6 (`O9`)**: fijar `OMP_NUM_THREADS = -j` en el lanzador y `openblas_set_num_threads` una sola vez al inicializar cuando `n_workers > 1` (`crates/qwen-app/src/server/scheduler.rs` y `crates/qwen-kernels/src/blas.rs`). `OPENBLAS_NUM_THREADS` permanece sin definir.
- **Criterio de cierre de Fase 2**:
  1. `qwen_tts --self-test` con `L2_rel == 0.0` (`INT8`) y `L2_rel <= 1e-6` (`Q4_0`) entre `matmat(B=8)` y `8 × matvec`.
  2. `qwen_tts --batch-test` aprobado y salida E2E **bit-idéntica** respecto al cierre de la Fase 1.
  3. `qwen_tts --matmat-bench` (`QWEN_BATCH_B=2,4,8`) y `--batch-bench` con techo realista **1.26×** E2E y **1.2–1.7×** en microbanco `INT8`/`INT4` frente a la Fase 0, y caída medible de `cp_ms` en `cp_prefill2`. `matmat_bf16` no se evalúa: mide 0.76–0.97×. **Fundamento**: `Anexo A §A.3 de phase-0/estado.md`.

---

### Fase 3 — Vectorización SIMD de activaciones, convoluciones y carga paralela
- **Alcance**: `O6`, `O7 (fusión swiglu + quantize_act_int8 en down_proj)`, `O8 (cs_dwconv, final_conv, add_residual_scaled en AVX2)` y `O12`. El ahorro esperado de `O6` es ≤1 ms/f: el coste del CP se concentra en `gate_up` (37 %) y QKV (27 %), no en SwiGLU (1.7 %). **Fundamento**: `Anexo A §A.1 de phase-0/estado.md`.
- **Prerequisitos**: Fases 1–2 estabilizadas (para aislar cualquier variación ULP introducida por `O6` sin tocar `swiglu` en fases previas).
- **Pasos**:
  - **Paso 3.1 (`O6a`)**: despachar `swiglu_inplace` y `silu` por `KernelDispatch` con `expf` polinomial `AVX2+FMA` (`avx2.rs` y `dispatch.rs`).
  - **Paso 3.2 (`O7` fusión)**: fusionar `swiglu + quantize_act_int8` para `down_proj` reutilizando la implementación de `Paso 3.1`.
  - **Paso 3.3 (`O6b` + `O8` vocoder)**: vectorizar `convnext_mlp` (GELU/LayerNorm), `cs_dwconv`, `final_conv` y `add_residual_scaled` (`crates/qwen-engine/src/speech_decoder.rs` y `avx2.rs`).
  - **Paso 3.4 (`O12`)**: paralelizar la cuantización durante `Model06B::load` sobre `runtime.pool` (`crates/qwen-engine/src/model.rs` y `crates/qwen-kernels/src/quant.rs`).
- **Criterio de cierre de Fase 3**:
  1. Error máximo relativo de `swiglu_inplace` / `silu` / GELU AVX2 frente a `libm` `< 1e-6`.
  2. `SNR >= 100 dB` en `SpeechDecoder` streaming vs full, determinismo E2E **bit-idéntico** entre corridas del mismo binario, y validación auditiva/métrica en frases cortas, medias y largas frente a la Fase 2.
  3. Reducción medible de `talker_ms`, `cp_ms`, `decoder_ms` y tiempo de carga de modelo.

---

### Fase 4 — Cambios con impacto en `NUMERIC_CONTRACT.md` (Ancho de banda Talker / Prefill)
- **Alcance**: `O4` Opción A (*tiling* L2 `BF16→F32` sin cambio numérico) y `O3` (`codec_head` en `INT8`). `O4` Opción B (prefill con `matmat_q4_0`) está `descartado` y no se habilita. La atención del Code Predictor permanece en `INT8`: su cuantización a `Q4_0` produce `+2–7 st` de tono. **Fundamento**: `Anexo A §A.3 de phase-0/estado.md`.
- **Prerequisitos**: Fase 2 completada.
- **Pasos**:
  - **Paso 4.1 (`O4` Opción A)**: convertir `BF16→F32` por bloques L2 dentro de `talker_prefill` (`crates/qwen-engine/src/talker.rs`) sin materializar la capa completa.
  - **Paso 4.2 (`O3`)**: añadir soporte `QuantWeight::Int8` en `talker.codec_head` (`model.rs`, `pipeline.rs`) tras evaluación A/B.
  - **Paso 4.3 (`O4` Opción B)**: `descartado`. No habilitar prefill cuantizado bajo `--quant-mixed`: `+4.7 st` en `ryan`, WER `+1.8/+2.7 pp` y prefill largo `+41 %`.
- **Criterio de cierre de Fase 4**:
  1. Evaluación A/B con pesos reales de `0.6B` (`CustomVoice` y `Base` con clonación): comparación de códigos generados, SNR/calidad perceptual, ausencia de artefactos al inicio/fin y medición de `prefill_ms`, RSS y RTF final.
  2. Actualización documentada de [`NUMERIC_CONTRACT.md`](../NUMERIC_CONTRACT.md) antes de habilitar cualquier cambio de precisión por defecto.

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
$env:OMP_NUM_THREADS = "4"
Remove-Item Env:OPENBLAS_NUM_THREADS -ErrorAction SilentlyContinue

cargo clippy --workspace --all-targets --target x86_64-pc-windows-gnu -- -D warnings
cargo test --workspace --release --target x86_64-pc-windows-gnu
.\target\x86_64-pc-windows-gnu\release\qwen_tts.exe --self-test
.\target\x86_64-pc-windows-gnu\release\qwen_tts.exe --matmat-bench
.\target\x86_64-pc-windows-gnu\release\qwen_tts.exe --batch-test -d models\qwen3-tts-0.6b --quant-mixed
.\target\x86_64-pc-windows-gnu\release\qwen_tts.exe --batch-bench -d models\qwen3-tts-0.6b --quant-mixed
```

Para registrar el progreso real de cada Fase y Paso, medir antes y después con el mismo texto, semilla (`--seed 4`) e hilos (`-j 4 --quant-mixed`), en **CLI y servidor**:
- **Latencia y RTF**: `Prefill (ms)`, `Talker step (ms/f)`, `Code Predictor (ms/f)`, `Speech decoder (ms)`, `drain (ms)`, `TTFA (ms)` y `RTF`.
- **Condición con carga**: además del equipo libre, 1 hilo copiando 256 MB en bucle durante toda la serie; su mediana se registra por separado. Las ganancias de ancho de banda aparecen en esta condición y aquí se verifica el requisito de RTF < 1.
- **Memoria**: pico de `WorkingSet64` del proceso en CLI (frase corta, frase larga, ICL) y en servidor bajo `--serve --batch-size 4` y `--workers 4` con 4 peticiones concurrentes del mismo texto. La memoria privada es otra magnitud: no se compara con el *working set* y se instrumenta por separado cuando se necesite.
- **Pico durante reasignación**: en Pasos que modifiquen capacidades iniciales, registrar el pico *durante* `ensure_capacity`, no solo el de régimen.
