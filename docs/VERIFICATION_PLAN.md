# Plan de Verificación (`VERIFICATION_PLAN.md`)

> Define las suites automatizadas, comprobaciones numéricas y matriz de CI que validan **Qwen TTS Engine** (`qwen-tts-engine`) en Rust estable. Pensado para QA, colaboradores y mantenimiento de CI.

---

## 1. Suites automatizadas (`cargo test`)

Ejecuta todas las pruebas unitarias y de integración del workspace con el perfil `release`:

```bash
# Linux / macOS
cargo test --workspace --release
cargo clippy --workspace --all-targets -- -D warnings
```

```powershell
# Windows (x86_64-pc-windows-gnu)
$env:PATH = "C:\msys64\ucrt64\bin;$env:PATH"
$env:OPENBLAS_LIB_DIR = "C:/openblas/lib"

cargo test --workspace --release --target x86_64-pc-windows-gnu
cargo clippy --workspace --all-targets --target x86_64-pc-windows-gnu -- -D warnings
```

| Suite | Qué valida |
|---|---|
| `qwen-kernels/tests/kernels_tests.rs` | `AlignedBuf<64>`, batería `--self-test` (`0` fallos en `int8`/`q4_0`/`bf16`), selección de ruta con `QWEN_KERNELS` (`auto`/`scalar`), tablas disponibles por CPU (`KernelDispatch::available`), hashes de las rutas SIMD iguales a los de AVX2 (Q4_0 y kernels de coma flotante sin `libm`: mismo orden de suma en NEON), regla de spin-wait por núcleos (`spin_budget_for`), paridad escalar vs dispatch detectado (`quantize_act_int8`, `rms_norm`, `matvec_int8`, `matmat_int8`, `matmat_q4_0`) y validación de límites seguros en `KernelDispatch`, reparto de filas entre hilos bit a bit (1 frente a 4 hilos, `matvec`/`matmat` INT8 y Q4_0 y `matvec` BF16), estrés concurrente y *seguridad ante pánicos* de `ComputePool`, `FfiBlas::sgemm`, y conversión `f16`. |
| `qwen-core/tests/core_tests.rs` | Precedencia de idioma (`effective_language`: pedido, voz, español), solo el modelo Base carga (config sintética y checkpoint real) y todo otro checkpoint se rechaza, rechazo del mensaje genérico `Este motor solo carga Qwen3-TTS 0.6B Base: <detalle>` (geometría distinta, tipo que no es Base y falta del codificador de locutor), idiomas sin dialectos (`language_name`), `MmapStore`, tokenización BPE, `.qvoice` v6 (ida y vuelta con y sin idioma, tamaño exacto de 4112 bytes, rechazo de otras versiones, magic incorrecto, `enc_dim` y `lang_id` inválidos, truncado, bytes sobrantes), `MarkupPlanner` (pausas, separador `\|`; un tag desconocido como `[xyz]` queda como texto), saneamiento de pausas negativas, manejo seguro de `NaN` y determinismo `Sampler`/`LcgRng`. |
| `qwen-engine/tests/engine_tests.rs` | Rechazo de códigos truncados, determinismo E2E bit-idéntico y cancelación cooperativa vía `AbortHandle`. |
| `qwen-core` (unitarios en `markup.rs`) | `plan_script` en sus dos modos (`Auto` y `Literal`) y `resolve_sampling` (primero lo explícito del usuario, después el valor por defecto). |
| `qwen-engine` (unitarios en `model.rs`) | Crecimiento bajo demanda de la caché KV con preservación de contenido: `TalkerKvCache::ensure_capacity` (desde `KV_INITIAL_POSITIONS`). |
| `qwen-engine` (unitarios en `audio.rs`) | `EdgeRampSink` da el mismo PCM bit a bit que `apply_gain` + `encode_wav_bytes` sobre el audio completo, con cualquier troceo, longitudes en el límite de las rampas y volumen con saturación; retiene solo los 40 ms de la rampa de salida. |
| `qwen-engine` (unitarios en `pipeline.rs`; el segundo usa `models/qwen3-tts-0.6b-base`) | Las reglas de logits del Talker descartan todas las filas que `talker_codec_logits` no calcula, con cualquier valor previo y en cualquier frame; `talker_codec_logits` da los mismos bits que el `codec_head` completo en las filas que calcula. |
| `qwen-engine` (unitarios en `code_predictor.rs`, usan `models/qwen3-tts-0.6b-base`) | Verificación especulativa del Code Predictor bit a bit: `cp_step2` en una posición arbitraria (2 y 9) frente a dos `cp_transformer_step` (estados y caché K/V, con centinela NaN), retroceso exacto tras un rechazo y `cp_predict` frente a la referencia secuencial (16 entradas). |
| `qwen-engine/tests/voice_clone_tests.rs` | Lector WAV de la clonación: solo PCM mono de 16 bits a 24 kHz (rechazo de otros formatos, de otra frecuencia, de estéreo y de `sample_rate = 0`) con la guía `ffmpeg`, `trim_trailing_silence`, `mel_spectrogram`, voz `default` incorporada (1024 valores finitos, norma L2 y bits exactos), errores de extensión y de versión al cargar una voz, marcado, y el flujo de extremo a extremo: la voz `default` dice un texto, ese audio se clona con el x-vector, tamaño exacto del `.qvoice` (4112 bytes), carga y síntesis con esa voz, y composición multitramo con `Synthesizer::synthesize`. |
| `qwen-engine/tests/synthesis_paths_tests.rs` | El camino único de síntesis: `Delivery::Buffer` frente a `Delivery::Stream(EdgeRampSink<Vec<u8>>)` dan los mismos bytes PCM en texto plano, con voz clonada y con marcado (el buffer se compara con el WAV sin cabecera, porque el encoder WAV aplica las rampas de borde) y `timing.samples_produced` igual en las dos entregas; `Voice::load`; y `clone_voice`, que devuelve solo el `QVoiceFile` (bytes exactos del archivo completo) y la voz se construye con `Voice::from_qvoice`. Las pruebas con síntesis se omiten si falta el modelo. |
| `qwen-app` (`cli.rs`, `request.rs`, `server/validation.rs`, `server/http.rs`) | Resolución única del modo (`test_cli_mode_precedence`), preparación de la petición (`from_cli` y `from_http`: idioma efectivo, semilla, temperatura y marcado), `--host` inválido, valores por defecto `DEFAULT_TEMPERATURE = 0.325` y `DEFAULT_SEED = 4`, rangos CLI (`--temperature`, `--top-k`, `--top-p`, `--rep-penalty`, `--volume`, `--rate`, `--seed`), implicación `--stdout`, validación HTTP `400` en `ValidatedTtsRequest` (incluido el campo `voice`, la lista estricta de campos por ruta y `rate`/`chunk_frames` solo en su ruta) y rechazo de `Content-Length` inválido o `> 1 MiB` en `http.rs`. |

> Las pruebas con pesos reales (los de integración `engine_tests.rs` y `voice_clone_tests.rs`, y los unitarios de `code_predictor.rs` y `pipeline.rs`) se ejecutan cuando el modelo Base está presente en `models/qwen3-tts-0.6b-base`. Si faltan, se omiten con un aviso.

---

## 2. Verificación binaria y numérica

Tras compilar `qwen_tts`, verifica el runtime según [`INTERFACE_CONTRACT.md`](INTERFACE_CONTRACT.md) y [`NUMERIC_CONTRACT.md`](NUMERIC_CONTRACT.md):

- **Capacidades y kernels**: `qwen_tts --caps`, `qwen_tts --self-test` (`0` fallos).
- **Carga sintética para medir RTF**: [`scripts/SyntheticLoad.ps1`](../scripts/SyntheticLoad.ps1) crea 0–2 procesos `pwsh` que copian 256 MB en bucle (`-Action Start -Threads 1 -ChunkMB 256 -Tag rtf`), los inspecciona con `Status` y los detiene con `Stop -Handle $load` en un `finally`. Es una herramienta de medición, no parte del binario.

---

## 3. Integración continua en CircleCI (`.circleci/config.yml`)

La CI del repositorio es CircleCI, con tres workflows sobre las mismas máquinas:

| Sistema | Máquina | Target | BLAS |
|---|---|---|---|
| Linux ARM64 | Docker `cimg/base`, `arm.large` (Neoverse N1, 4 vCPU) | `aarch64-unknown-linux-gnu` | OpenBLAS 0.3.34 propio (`scripts/build-openblas.sh`, en caché) |
| macOS ARM64 | macOS Xcode 26.4, `m4pro.medium` (Apple M4 Pro) | `aarch64-apple-darwin` | `Accelerate` |
| Windows x86_64 | Windows Server 2022 (orb `circleci/windows`), `large` (4 núcleos físicos) | `x86_64-pc-windows-gnu` | OpenBLAS 0.3.34 propio (`scripts/build-openblas.sh`, en caché) |

**Workflows**:
- **`test`**: en cada push.
  - `clippy --workspace --all-targets -- -D warnings`;
  - `cargo test --workspace --release`, con el modelo Base restaurado de la caché, así que las pruebas que lo necesitan se ejecutan;
  - `--caps` y `--self-test`.
- **`measure`**: a mano, con el parámetro de pipeline `measure = true`, desde la web de CircleCI («Trigger Pipeline») o desde la API v2. Ejecuta `scripts/ci-measure-rtf.sh`:
  - `--caps` y `--self-test`;
  - 3 síntesis de la frase corta y la larga (voz `default`, inglés, `-j 4 --seed 4`), con RTF, tiempos por etapa, TTFA y hash del PCM.
- **`kernels`**: a mano, con `kernels = true`, en los tres sistemas. Compara `QWEN_KERNELS=auto` con `QWEN_KERNELS=scalar`, intercalados por corrida (`scripts/ci-compare-kernels.sh`): tiempos, RTF y hash del PCM de cada ruta, y el número de rutas SIMD que pasan el `--self-test`.

Las pruebas que necesitan pesos reales se omiten (con un aviso en la salida) cuando el modelo no está en `models/`.

---

## 4. Medición de rendimiento

Para decidir si un cambio de rendimiento se integra ([ADR-005](adr/005-cpu-inference-optimizations.md)):

- **Paridad primero**: `--self-test`, `cargo test` y el hash del PCM frente a la tabla de [`NUMERIC_CONTRACT.md`](NUMERIC_CONTRACT.md) («Audio de referencia»). Si el audio cambia, el cambio necesita una escucha y una tabla nueva.
- **A/B intercalado**: binario de control y binario nuevo alternados por corrida (o en orden ABBA), con las corridas de calentamiento descartadas, para que la deriva térmica no favorezca a ninguno. `-j 4 --seed 4`, frase corta y larga de referencia, `--stdout`, y una corrida del servidor por `POST /v1/tts/stream`. Mediana de 3 pares como mínimo.
- **Con carga**: el A/B se repite con 1 hilo copiando 256 MiB en bucle (`scripts/SyntheticLoad.ps1 -Action Start -Threads 1 -ChunkMB 256`). Ahí se verifica el tiempo real.
- **Métricas** (las imprime el motor): `Prefill (ms)`, `Paso del Talker (ms/f)`, `Code Predictor (ms/f)`, `Speech decoder (ms)`, drenaje, TTFA y RTF. En RAM, el pico de memoria comprometida (*commit*) privada (`PeakPagefileUsage` en Windows) es la métrica principal y el pico de *working set*, de no regresión. En cambios de capacidad inicial, el pico *durante* el crecimiento (`ensure_capacity`).
- **Herramienta mínima**: una corrida para la RAM (es determinista) o 3 corridas sueltas bastan para descartar; el A/B completo es para la decisión.
- **Sin interferencias**: nunca compilar ni lanzar pruebas en paralelo con una medición; los procesos del motor escriben su salida en un archivo.
- **ARM64 y otras máquinas**: workflows `measure` y `kernels` de la CI (§3). El audio de ARM se compara con la ruta escalar de la misma máquina.
