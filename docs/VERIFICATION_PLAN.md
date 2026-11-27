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

La línea del `PATH` sobra si el `gcc` de UCRT64 está fijado como enlazador en la configuración de Cargo de usuario (README, «Windows (MSYS2 UCRT64)»).

Los temporales de las pruebas caen en `target/` (lo fija `.cargo/config.toml`) y cada prueba borra los suyos al terminar, también si falla.

| Suite | Qué valida |
|---|---|
| `qwen-kernels/tests/kernels_tests.rs` | `AlignedBuf<64>`, batería `--self-test` (`0` fallos en `int8`/`q4_0`/`bf16`), selección de ruta con `QWEN_KERNELS` (`auto`/`scalar`), tablas disponibles por CPU (`KernelDispatch::available`), hashes de las rutas SIMD iguales a los de AVX2 (Q4_0 y kernels de coma flotante sin `libm`: mismo orden de suma en NEON), regla de spin-wait por núcleos (`spin_budget_for`), paridad escalar vs dispatch detectado (`quantize_act_int8`, `rms_norm`, `matvec_int8`, `matmat_int8`, `matmat_q4_0`) y validación de límites seguros en `KernelDispatch`, reparto de filas entre hilos bit a bit (1 frente a 4 hilos, `matvec`/`matmat` INT8 y Q4_0 y `matvec` BF16), estrés concurrente y *seguridad ante pánicos* de `ComputePool`, `FfiBlas::sgemm`, y conversión `f16`, `quantize_gate_up_int8` igual bit a bit a cuantizar la matriz entrelazada. |
| `qwen-core/tests/core_tests.rs` | Precedencia de idioma (`effective_language`: pedido, voz, español), solo el modelo Base carga (config sintética y checkpoint real) y todo otro checkpoint se rechaza, rechazo del mensaje genérico `Este motor solo carga Qwen3-TTS 0.6B Base: <detalle>` (geometría distinta, tipo que no es Base y falta del codificador de locutor), idiomas sin dialectos (`language_name`), `MmapStore`, tokenización BPE, `.qvoice` v6 (ida y vuelta con y sin idioma, tamaño exacto de 4112 bytes, rechazo de otras versiones, magic incorrecto, `enc_dim` y `lang_id` inválidos, truncado, bytes sobrantes), `MarkupPlanner` (pausas, separador `\|`; un tag desconocido como `[xyz]` queda como texto), saneamiento de pausas negativas, manejo seguro de `NaN` y determinismo `Sampler`/`LcgRng`. |
| `qwen-engine/tests/engine_tests.rs` | Rechazo de códigos truncados, determinismo E2E bit-idéntico y cancelación cooperativa vía `AbortHandle`; retención tras el primer EOS (`test_eos_hold_frames`: frames = frame del primer EOS + 2, 1 920 muestras por cada frame entregado, es decir `(frames - leading_silence_frames) * 1920`, y la retención sin pasar de `max_frames`); descarte de la cadena de silencio inicial (`test_leading_silence_discard`: con la voz `default` en español, `leading_silence_frames` es mayor que 0 e igual a los frames iniciales seguidos de la cadena, y con `max_frames = 3`, los tres frames son de la cadena, no se descarta ninguno y se entregan 3 × 1 920 muestras); audio de referencia (`test_reference_audio_hashes`: las cuatro frases de la tabla «Audio de referencia (x86-64)» de `NUMERIC_CONTRACT.md` dan sus hashes; solo corre en el equipo de referencia, con `QWEN_TTS_CHECK_REFERENCE_AUDIO=1` y la ruta `avx2`, porque otra CPU da otros hashes). Las pruebas con el modelo se ejecutan una a una. |
| `qwen-core` (unitarios en `markup.rs`) | `plan_script` en sus dos modos (`Auto` y `Literal`), el punto final (`close_final_sentence_cases` y `plan_script_closes_final_sentence`: texto plano, último tramo hablado y casos sin cambio) y `resolve_sampling` (primero lo explícito del usuario, después el valor por defecto). |
| `qwen-kernels` (unitarios en `quant.rs`) | El redondeo propio de los cuantizadores (`round_ties_even_small`, `round_half_away_small`) da lo mismo que `f32::round_ties_even` y `f32::round` alrededor de cada entero y semientero de ±130 y en un barrido del rango. |
| `qwen-engine` (unitarios en `model.rs`) | Crecimiento bajo demanda de la caché KV con preservación de contenido: `TalkerKvCache::ensure_capacity` (desde `KV_INITIAL_POSITIONS`). |
| `qwen-engine` (unitarios en `audio.rs`) | `EdgeRampSink` da el mismo PCM bit a bit que `apply_gain` + `encode_wav_bytes` sobre el audio completo, con cualquier troceo, longitudes en el límite de las rampas y volumen con saturación; retiene solo los 40 ms de la rampa de salida. `time_stretch` (WSOLA de `--rate`): sin factor válido (1, 0, negativo, `NaN`, infinito) o con menos de 16 muestras devuelve la entrada; el factor se acota a `[0.25, 4.0]`; con una senoide, la duración es `(n - 1024) / rate + 1024` con el error de un salto (512 muestras), la frecuencia no cambia (cruces por cero) y el pico no supera el de la entrada. |
| `qwen-engine` (unitarios en `pipeline.rs`; el segundo usa `models/qwen3-tts-0.6b-base`) | Las reglas de logits del Talker descartan todas las filas que `talker_codec_logits` no calcula, con cualquier valor previo y en cualquier frame; `talker_codec_logits` da los mismos bits que el `codec_head` completo en las filas que calcula. |
| `qwen-engine` (unitarios en `code_predictor.rs`, usan `models/qwen3-tts-0.6b-base`) | Verificación especulativa del Code Predictor bit a bit: `cp_step2` en una posición arbitraria (2 y 9) frente a dos `cp_transformer_step` (estados y caché K/V, con centinela NaN), retroceso exacto tras un rechazo y `cp_predict` frente a la referencia secuencial (16 entradas). |
| `qwen-engine/tests/voice_clone_tests.rs` | Lector WAV de la clonación: solo PCM mono de 16 bits a 24 kHz (rechazo de otros formatos, de otra frecuencia, de estéreo y de `sample_rate = 0`) con la guía `ffmpeg`, `preprocess_ref_audio` (pasa-altos 70 Hz, limpieza/compactación de silencios y normalización RMS), `mel_spectrogram`, voz `default` incorporada (1024 valores finitos, norma L2 y bits exactos), errores de extensión y de versión al cargar una voz, marcado, y el flujo de extremo a extremo: la voz `default` dice un texto, ese audio se clona con `clone_voice` y `clone_voice_from_samples` (equivalentes), tamaño exacto del `.qvoice` (4112 bytes), carga y síntesis con esa voz, y composición multitramo con `Synthesizer::synthesize`. |
| `qwen-engine/tests/synthesis_paths_tests.rs` | El camino único de síntesis: `Delivery::Buffer` frente a `Delivery::Stream(EdgeRampSink<Vec<u8>>)` dan los mismos bytes PCM en texto plano, con voz clonada y con marcado (el buffer se compara con el WAV sin cabecera, porque el encoder WAV aplica las rampas de borde) y `timing.samples_produced` igual en las dos entregas; `Voice::load`; y `clone_voice`, que devuelve solo el `QVoiceFile` (bytes exactos del archivo completo) y la voz se construye con `Voice::from_qvoice`. Las pruebas con síntesis se omiten si falta el modelo. |
| `qwen-app` (`cli.rs`, `request.rs`, `server/validation.rs`, `server/http.rs`, `modes/voice_cmd.rs`) | Resolución única del modo (`test_cli_mode_precedence`), preparación de la petición HTTP (`from_http`: idioma efectivo, reglas por ruta, semilla y temperatura), `--host` inválido, valores por defecto `DEFAULT_TEMPERATURE = 0.45` y `DEFAULT_SEED = 4`, rangos CLI (`--temperature`, `--top-k`, `--top-p`, `--rep-penalty`, `--volume`, `--rate`, `--seed`, `--max-duration`), flags desconocidos, reglas de las flags de voz, flag `--stdout` y semilla de reloj (`test_stdout_and_clock_seed`), nombre de la voz clonada (`cloned_voice_name`), validación HTTP `400` en `ValidatedTtsRequest` (incluido el campo `voice`, la lista estricta de campos por ruta y `rate`/`chunk_frames` solo en su ruta) y rechazo de `Content-Length` inválido o `> 1 MiB` en `http.rs`. |
| `qwen-app/tests/contract_tests.rs` | Contrato para clientes de proceso y HTTP (`INTERFACE_CONTRACT.md` §5), lanzando el binario `qwen_tts` real y sin comparar texto de mensajes. Sin modelo, una prueba por caso: flag desconocido, `--self-test` y `--caps` (`0`), `-j`, `--host` inválido (`1` en menos de 2 s), `--serve 0`, `QWEN_KERNELS` (`auto` y `scalar` válidos, otro valor `1`), opciones de voz incoherentes y modelo inexistente (`1`). Con modelo, una sola prueba secuencial (`contract_with_model`, nunca dos procesos con el modelo cargado): referencia de clonado estéreo y a 48 kHz rechazadas (`1` en menos de 2 s), clonado con `--watch-stdin` (`.qvoice` v6 de 4112 bytes sin idioma; con la entrada cerrada al instante, salida `0` sin `.qvoice` válido) y un servidor real: salud, voces, `404`, los `400` con `error` JSON (incluido un cuerpo de 4 MiB enviado entero), stream completo (cabeceras, fragmentos pares, terminador), texto solo de pausas (stream sin terminador y `500`), mismo PCM por stream y por WAV, cancelación por desconexión, determinismo, una conexión a la vez y salida `0` al cerrar la entrada. |

> Las pruebas con pesos reales (los de integración `engine_tests.rs`, `voice_clone_tests.rs` y `contract_tests.rs`, y los unitarios de `code_predictor.rs` y `pipeline.rs`) se ejecutan cuando el modelo Base está presente en `models/qwen3-tts-0.6b-base`. Si faltan, se omiten con un aviso. La excepción es `contract_with_model`: con la variable de entorno `QWEN_TTS_REQUIRE_MODEL=1` (que fija la CI en los jobs del workflow `test`) falla en lugar de omitirse, para que la CI no pueda pasar sin ejecutarla.

---

## 2. Verificación binaria y numérica

Tras compilar `qwen_tts`, verifica el runtime según [`INTERFACE_CONTRACT.md`](INTERFACE_CONTRACT.md) y [`NUMERIC_CONTRACT.md`](NUMERIC_CONTRACT.md):

- **Capacidades y kernels**: `qwen_tts --caps`, `qwen_tts --self-test` (`0` fallos).
- **Carga sintética para medir RTF**: [`scripts/SyntheticLoad.ps1`](../scripts/SyntheticLoad.ps1) crea 0–8 procesos `pwsh` que copian 256 MB en bucle (`-Action Start -Threads 1 -ChunkMB 256 -Tag rtf`), los inspecciona con `Status` y los detiene con `Stop -Handle $load` en un `finally`. Es una herramienta de medición, no parte del binario.

---

## 3. Integración continua en CircleCI (`.circleci/config.yml`)

La CI del repositorio es CircleCI, con cinco workflows sobre las mismas máquinas:

| Sistema | Máquina | Target | BLAS |
|---|---|---|---|
| Linux x86_64 | Docker `cimg/base`, `large` (x86_64, solo en el workflow `test`) | `x86_64-unknown-linux-gnu` | OpenBLAS 0.3.34 propio (`scripts/build-openblas.sh`, en caché) |
| Linux ARM64 | Docker `cimg/base`, `arm.large` (Neoverse N1, 4 vCPU) | `aarch64-unknown-linux-gnu` | OpenBLAS 0.3.34 propio (`scripts/build-openblas.sh`, en caché) |
| macOS ARM64 | macOS Xcode 26.4, `m4pro.medium` (Apple M4 Pro) | `aarch64-apple-darwin` | `Accelerate` |
| Windows x86_64 | Windows Server 2022 (orb `circleci/windows`), `large` (4 núcleos físicos) | `x86_64-pc-windows-gnu` | OpenBLAS 0.3.34 propio (`scripts/build-openblas.sh`, en caché) |

**Workflows**:
- **`test`**: en cada push, en los cuatro sistemas (Linux x86_64, Linux ARM64, macOS ARM64 y Windows x86_64).
  - solo en Linux x86_64, antes de las pruebas, `scripts/ci-check-licenses.sh`: compara los crates y versiones que entran en el binario de los cuatro targets (`cargo metadata` recorrido desde `qwen-app`, sin dependencias de compilación ni macros procedurales) con la tabla «Crates que entran en el binario» de `THIRD-PARTY-LICENSES.md`, y falla si falta o sobra alguno;
  - `clippy --workspace --all-targets -- -D warnings`;
  - `cargo test --workspace --release`, con el modelo Base restaurado de la caché, así que las pruebas que lo necesitan se ejecutan;
  - `--caps` y `--self-test`.
- **`measure`**: a mano, con el parámetro de pipeline `measure = true`, desde la web de CircleCI («Trigger Pipeline») o desde la API v2. Ejecuta `scripts/ci-measure-rtf.sh`:
  - `--caps` y `--self-test`;
  - 3 síntesis de la frase corta y la larga (voz `default`, inglés, `-j 4 --seed 4`), con RTF, tiempos por etapa, TTFA y hash del PCM.
- **`kernels`**: a mano, con `kernels = true`, en Linux ARM64, macOS ARM64 y Windows x86_64. Compara `QWEN_KERNELS=auto` con `QWEN_KERNELS=scalar`, intercalados por corrida (`scripts/ci-compare-kernels.sh`): tiempos, RTF y hash del PCM de cada ruta, y el número de rutas SIMD que pasan el `--self-test`.

- **`release-dry-run`**: a mano, con `release_dry_run = true` (la API v2 con `{"branch": "main", "parameters": {"release_dry_run": true}}`, o «Trigger Pipeline»); con ese parámetro el workflow `test` no corre. Es la release completa salvo la publicación:
  - `release-linux-x64` y `release-linux-arm64`: compilan en la imagen `manylinux_2_28` (glibc 2.28) con el OpenBLAS propio, reenlazan `qwen-app` con `-Wl,-Map` para obtener el mapa del enlace y auditan el binario con `scripts/ci-audit-binary.sh` (solo bibliotecas del sistema, símbolos `GLIBC_*` ≤ 2.28 y sin `GLIBCXX_*`); después ejecutan `--caps` y `--self-test` y generan el `tar.gz`;
  - `release-macos-arm64`: compila con `MACOSX_DEPLOYMENT_TARGET=26.0`, audita que `minos` sea ≤ 26.0 y que solo enlace bibliotecas de `/usr/lib/` y `/System/Library/`, ejecuta `--self-test` y genera el `tar.gz`;
  - `release-windows-x64`: compila como el workflow `test`, genera el mapa del enlace, audita las DLL importadas (solo `KERNEL32`, `ntdll`, `WS2_32`, `bcryptprimitives` y `api-ms-win-*`), ejecuta `--self-test` y genera el `zip`;
  - `cleanroom-almalinux8-*` y `cleanroom-debian-*` (`almalinux:8` y `debian:stable-slim`, x86_64 y ARM64): ejecutan `--caps` y `--self-test` con el binario Linux ya compilado, sin compilar nada;
  - `release-assemble`: genera las fuentes (`git archive`) y `SHA256SUMS` de los cuatro paquetes y las fuentes, y los guarda como artefactos junto con los mapas del enlace (`link-map-*.txt`).
  - Cada paquete contiene `qwen_tts[.exe]`, `LICENSE`, `NOTICE` y `THIRD-PARTY-LICENSES.md` bajo la carpeta `qwen-tts-engine-<versión>-<triple>/`. Si el pipeline viene de un tag, cada job falla si el tag no coincide con la versión de `Cargo.toml`.
- **`release`**: solo con un tag `v*` (lo crea el usuario; ningún otro workflow corre en tags). Ejecuta los mismos jobs que `release-dry-run`, todos con el filtro de tags, y añade `release-publish` (contexto `github-release`, variable `GH_TOKEN`), que con `scripts/ci-publish-release.sh` falla si ya existe una release con ese tag (publicada o en borrador), la crea como borrador, sube los seis archivos (cuatro paquetes, las fuentes y `SHA256SUMS`), comprueba que los lista y solo entonces la publica. Una release publicada no cambia: una corrección es otra versión.

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
