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
Remove-Item Env:OPENBLAS_NUM_THREADS -ErrorAction SilentlyContinue

cargo test --workspace --release --target x86_64-pc-windows-gnu
cargo clippy --workspace --all-targets --target x86_64-pc-windows-gnu -- -D warnings
```

| Suite | Qué valida |
|---|---|
| `qwen-kernels/tests/kernels_tests.rs` | `AlignedBuf<64>`, batería `--self-test` (`0` fallos en `int8`/`q4_0`/`bf16`), selección de ruta con `QWEN_KERNELS` (`auto`/`scalar`), paridad escalar vs dispatch detectado (`quantize_act_int8`, `rms_norm`, `matvec_int8`, `matmat_int8`, `matmat_q4_0`) y validación de límites seguros en `KernelDispatch`, reparto de filas entre hilos bit a bit (1 frente a 4 hilos, `matvec`/`matmat` BF16, INT8 y Q4_0), estrés concurrente y *panic safety* de `ComputePool`, `BlasBackend::sgemm`, `quantize_bf16_to_q2_0` y conversión `f16`. |
| `qwen-core/tests/core_tests.rs` | Carga `0.6B` (`CustomVoice`/`Base`), rechazo de `1.7B` y `VoiceDesign`, `MmapStore`, tokenización BPE, roundtrip `.bin` y `.qvoice` v3 (`META`/`TPAD`/`WOVR` + `WDLT` LZ4), validación de payload raw `dtype_flag == 0`, rechazo de `.qvoice` v2 y archivos truncados, `MarkupPlanner` (incluido separador `\|`), saneamiento de `voice_strength=NaN`/pausas negativas, manejo seguro de `NaN` y determinismo `Sampler`/`LcgRng`. |
| `qwen-engine/tests/engine_tests.rs` | Equivalencia de `SpeechDecoder` en *streaming* (`2`, luego `min(10, 4 + 2 * (k / 8))` con tiling de 256 KB) frente a decodificación completa (`SNR >= 100 dB`), rechazo de códigos truncados, determinismo E2E bit-idéntico y cancelación cooperativa vía `AbortHandle`. |
| `qwen-engine` (unitarios en `model.rs`, `batch.rs`) | Crecimiento bajo demanda de las cachés KV con preservación de contenido: `TalkerKvCache::ensure_capacity` (desde `KV_INITIAL_POSITIONS`) y `BatchWorkspace::ensure_kv_capacity` (por slot, tope `kv_max`). |
| `qwen-engine` (unitarios en `audio.rs`) | `StreamEdgeRamp` da el mismo PCM bit a bit que `apply_gain` + `encode_wav_bytes` sobre el audio completo, con cualquier troceo, longitudes en el límite de las rampas y volumen con saturación; retiene solo los 40 ms de la rampa de salida. |
| `qwen-engine` (unitarios en `pipeline.rs`; el segundo usa `models/qwen3-tts-0.6b`) | Las reglas de logits del Talker descartan todas las filas que `talker_codec_logits` no calcula, con cualquier valor previo y en cualquier fotograma; `talker_codec_logits` da los mismos bits que el `codec_head` completo en las filas que calcula. |
| `qwen-engine` (unitarios en `code_predictor.rs`, usan `models/qwen3-tts-0.6b`) | Verificación especulativa del Code Predictor bit a bit: `cp_step2` en una posición arbitraria (2 y 9) frente a dos `cp_transformer_step` (estados y caché K/V, con centinela NaN), retroceso exacto tras un rechazo y `cp_predict` frente a la referencia secuencial (16 entradas). |
| `qwen-engine/tests/voice_clone_emotion_tests.rs` | Validación temprana de WAV 24 kHz (incluido rechazo de `sample_rate = 0`) y guías `ffmpeg`, `trim_trailing_silence`, `mel_spectrogram`, resolución `emovoice` + norma L2 `dir_*.bin`, flujo `0.6B-Base` → `.qvoice` (`WDLT`), rechazo de `WDLT` sobre `Q4_0` sin `--icl-only`, `--emo-ref`, `--roughness` (`Q2_0`) y composición multi-span (`render_spans_buffer` y `render_spans_stream`). |
| `qwen-app` (`cli.rs`, `server/validation.rs`, `server/http.rs`) | Rechazo de flags no soportadas (`--voice-design`, `--instruct`), `--host` inválido, valores por defecto `DEFAULT_TEMPERATURE = 0.325` y `DEFAULT_SEED = 4`, rangos CLI (`--temperature`, `--top-k`, `--top-p`, `--rep-penalty`, `--max-tokens`, `--vs-layers`, `--volume`, `--rate`, `--roughness`, `--seed`), implicaciones `--graft`/`--stdout`, mínimo de `--batch-words`, validación HTTP `400` en `ValidatedTtsRequest` y rechazo de `Content-Length` inválido o `> 1 MiB` en `http.rs`. |

> Los tests con pesos reales (los de integración `engine_tests.rs` y `voice_clone_emotion_tests.rs`, y los unitarios de `code_predictor.rs` y `pipeline.rs`) se ejecutan cuando los modelos `0.6B` están presentes en `models/`. Si faltan, se omiten con un aviso.

---

## 2. Verificación binaria y numérica

Tras compilar `qwen_tts`, verifica el runtime y las rutas *batched* según [`INTERFACE_CONTRACT.md`](INTERFACE_CONTRACT.md) y [`NUMERIC_CONTRACT.md`](NUMERIC_CONTRACT.md):

- **Capacidades y kernels**: `qwen_tts --caps`, `qwen_tts --self-test` (`0` fallos) y `qwen_tts --matmat-bench`.
- **Equivalencia *batching***: `qwen_tts -d models/qwen3-tts-0.6b --quant-mixed --batch-test` y `--batch-bench`.
- **Decodificación aislada**: `qwen_tts decode-dump <codes.txt> <model_dir> <out.wav>`.
- **Carga sintética para medir RTF**: [`scripts/SyntheticLoad.ps1`](../scripts/SyntheticLoad.ps1) crea 0–2 procesos `pwsh` que copian 256 MB en bucle (`-Action Start -Threads 1 -ChunkMB 256 -Tag rtf`), los inspecciona con `Status` y los detiene con `Stop -Handle $load` en un `finally`. Es una herramienta de medición, no parte del binario.

---

## 3. Integración continua en CircleCI (`.circleci/config.yml`)

La CI del repositorio es CircleCI, con tres workflows sobre las mismas máquinas:

| Sistema | Máquina | Target | BLAS |
|---|---|---|---|
| Linux ARM64 | Docker `cimg/base`, `arm.large` (Neoverse N1, 4 vCPU) | `aarch64-unknown-linux-gnu` | `libopenblas-dev` |
| macOS ARM64 | macOS Xcode 26.4, `m4pro.medium` (Apple M4 Pro) | `aarch64-apple-darwin` | `Accelerate` |
| Windows x86_64 | Windows Server 2022 (orb `circleci/windows`), `large` (4 núcleos físicos) | `x86_64-pc-windows-gnu` | MSYS2 UCRT64 (`libopenblas.a` estático) |

**Workflows**:
- **`pruebas`**: en cada push.
  - `clippy --workspace --all-targets -- -D warnings`;
  - `cargo test --workspace --release`, con el modelo CustomVoice restaurado de la caché, así que las pruebas que lo necesitan se ejecutan. Las que requieren también el modelo Base se omiten;
  - `--caps` y `--self-test`.
- **`medicion`**: a mano, con el parámetro de pipeline `medir = true`, desde la web de CircleCI («Trigger Pipeline») o desde la API v2. Ejecuta `scripts/ci-medir-rtf.sh`:
  - `--caps` y `--self-test`;
  - 3 síntesis de la frase corta y la larga (`ryan`, inglés, `--quant-mixed -j 4 --seed 4`), con RTF, tiempos por etapa, TTFA y hash del PCM.
- **`hilos-windows`**: a mano, con `hilos_windows = true`. Compara `-j 4`, `-j 2` y `-j 4` con `QWEN_SD_THREADS=2` en Windows (`scripts/ci-comparar-hilos.sh`).

Las pruebas que necesitan pesos reales se omiten (con un aviso en la salida) cuando el modelo no está en `models/`.
