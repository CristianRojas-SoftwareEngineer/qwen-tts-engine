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
$env:OMP_NUM_THREADS = "4"
Remove-Item Env:OPENBLAS_NUM_THREADS -ErrorAction SilentlyContinue

cargo test --workspace --release --target x86_64-pc-windows-gnu
cargo clippy --workspace --all-targets --target x86_64-pc-windows-gnu -- -D warnings
```

| Suite | Qué valida |
|---|---|
| `qwen-kernels/tests/kernels_tests.rs` | `AlignedBuf<64>`, batería `--self-test` (`0` fallos en `int8`/`q4_0`/`bf16`), paridad escalar vs dispatch detectado (`quantize_act_int8`, `rms_norm`, `matvec_int8`, `matmat_int8`, `matmat_q4_0`) y validación de límites seguros en `KernelDispatch`, estrés concurrente y *panic safety* de `ComputePool`, `BlasBackend::sgemm`, `quantize_bf16_to_q2_0` y conversión `f16`. |
| `qwen-core/tests/core_tests.rs` | Carga `0.6B` (`CustomVoice`/`Base`), rechazo de `1.7B` y `VoiceDesign`, `MmapStore`, tokenización BPE, roundtrip `.bin` y `.qvoice` v3 (`META`/`TPAD`/`WOVR` + `WDLT` LZ4), validación de payload raw `dtype_flag == 0`, rechazo de `.qvoice` v2 y archivos truncados, `MarkupPlanner` (incluido separador `\|`), saneamiento de `voice_strength=NaN`/pausas negativas, manejo seguro de `NaN` y determinismo `Sampler`/`LcgRng`. |
| `qwen-engine/tests/engine_tests.rs` | Equivalencia de `SpeechDecoder` en *streaming* (`2`, luego `min(10, 4 + 2 * (k / 8))` con tiling de 256 KB) frente a decodificación completa (`SNR >= 100 dB`), rechazo de códigos truncados, determinismo E2E bit-idéntico y cancelación cooperativa vía `AbortHandle`. |
| `qwen-engine/tests/voice_clone_emotion_tests.rs` | Validación temprana de WAV 24 kHz (incluido rechazo de `sample_rate = 0`) y guías `ffmpeg`, `trim_trailing_silence`, `mel_spectrogram`, resolución `emovoice` + norma L2 `dir_*.bin`, flujo `0.6B-Base` → `.qvoice` (`WDLT`), rechazo de `WDLT` sobre `Q4_0` sin `--icl-only`, `--emo-ref`, `--roughness` (`Q2_0`) y composición multi-span (`render_spans_buffer` y `render_spans_stream`). |
| `qwen-app` (`cli.rs`, `server/validation.rs`, `server/http.rs`) | Rechazo de flags no soportadas (`--voice-design`, `--instruct`), `--host` inválido, valores por defecto `DEFAULT_TEMPERATURE = 0.325` y `DEFAULT_SEED = 4`, rangos CLI (`--temperature`, `--top-k`, `--top-p`, `--rep-penalty`, `--max-tokens`, `--vs-layers`, `--volume`, `--rate`, `--roughness`, `--seed`), implicaciones `--graft`/`--stdout`, mínimo de `--batch-words`, validación HTTP `400` en `ValidatedTtsRequest` y rechazo de `Content-Length` inválido o `> 1 MiB` en `http.rs`. |

> Los tests de integración con pesos reales (`engine_tests.rs`, `voice_clone_emotion_tests.rs`) se ejecutan cuando los modelos `0.6B` están presentes en `models/`.

---

## 2. Verificación binaria y numérica

Tras compilar `qwen_tts`, verifica el runtime y las rutas *batched* según [`INTERFACE_CONTRACT.md`](INTERFACE_CONTRACT.md) y [`NUMERIC_CONTRACT.md`](NUMERIC_CONTRACT.md):

- **Capacidades y kernels**: `qwen_tts --caps`, `qwen_tts --self-test` (`0` fallos) y `qwen_tts --matmat-bench`.
- **Equivalencia *batching***: `qwen_tts -d models/qwen3-tts-0.6b --quant-mixed --batch-test` y `--batch-bench`.
- **Decodificación aislada**: `qwen_tts decode-dump <codes.txt> <model_dir> <out.wav>`.

---

## 3. Matriz de integración continua (`.github/workflows/ci.yml`)

En cada `push` y `pull_request` sobre `main` (y tags `v*`), CI ejecuta pruebas del workspace, compilación `release`, `--caps` y `--self-test` en:

| Runner / Target | Toolchain BLAS | Verificaciones extra |
|---|---|---|
| `linux-x86_64` | `libopenblas-dev` | `cargo check -p qwen-app --features faer-blas` |
| `linux-aarch64` | `libopenblas-dev` | Verificación en arquitectura ARM64 (*fallback* escalar) |
| `macos-arm64` | `Accelerate` | Verificación en Apple Silicon |
| `windows-x86_64-gnu` | MSYS2 UCRT64 (`libopenblas.a` estático) | Comprobación de manifiesto UTF-8 (`activeCodePage`) embebido en `qwen_tts.exe` |
