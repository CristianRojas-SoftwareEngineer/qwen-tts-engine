# Qwen3-TTS (`0.6B`) en Rust (`qwen3-tts-rs`)

Motor de inferencia TTS en CPU para **Qwen3-TTS 0.6B** (`CustomVoice` y `Base`) escrito en Rust estable, con soporte de CLI, streaming progresivo, clonación de voz (`.qvoice`/`.bin`, `ECAPA-TDNN`, `SpeechEncoder` RVQ), expresividad `0.6B` (`emovoice`, `--emo-ref`, `--roughness`, marcado inline), y servidor HTTP compatible con OpenAI (`/v1/tts`, `/v1/tts/stream`, `/v1/audio/speech`, `/v1/health`, `/v1/speakers`) con `WorkerPoolScheduler` y `ContinuousBatchScheduler`.

## Estructura del repositorio

- `qwen3-tts-rs/` — Workspace Cargo:
  - `crates/qwen-kernels` — `AlignedBuf<64>`, `CpuCaps`, `ComputePool`, `FfiBlas` (`OpenBLAS`/`Accelerate`) + `PureRustBlas` opcional (`--features faer-blas`), cuantización `int8`/`q4_0`/`q2_0`, kernels SIMD (`AVX2+FMA` en x86-64, fallback entero escalar portable), `--self-test` y `--matmat-bench`.
  - `crates/qwen-core` — Constantes `0.6B`, `MmapStore`, `BpeTokenizer`, `Sampler`/`LcgRng`, `.qvoice`/`.bin`, `MarkupPlanner`.
  - `crates/qwen-engine` (`#![forbid(unsafe_code)]`) — `Model06B`, `VoiceBindings`, `SessionScratch`, `Talker`, `CodePredictor`, `SpeechDecoder`, `SpeechEncoder`, `ECAPA-TDNN`, `emotion`, `compose`, `batch`, `audio`, `pipeline`.
  - `crates/qwen-app` — Binario `qwen_tts`: CLI completa, afinidad SMT Win32, manifiesto UTF-8, `--watch-stdin`, servidor HTTP y schedulers (`--workers`, `--batch-size`).
- `docs/` — Contratos y arquitectura (`INTERFACE_CONTRACT.md`, `SPEC_06B.md`, `NUMERIC_CONTRACT.md`, `ARCHITECTURE.md`, `VERIFICATION_PLAN.md`, `MIGRATION_PLAN.md`) y `docs/adr/`.
- `.github/workflows/ci.yml` — CI multiplataforma (Windows `x86_64-pc-windows-gnu`, Linux `x86_64`/`aarch64`, macOS `arm64`).
- `models/` — *(Opcional, ignorado en Git)* Pesos `0.6B` (`qwen3-tts-0.6b` y `qwen3-tts-0.6b-base`).

## Compilación y tests (Windows UCRT64 / Linux / macOS)

En Windows (`x86_64-pc-windows-gnu` con OpenBLAS estática de MSYS2 UCRT64):

```powershell
$env:PATH = "C:\msys64\ucrt64\bin;$env:PATH"
$env:OMP_NUM_THREADS = "4"
Remove-Item Env:OPENBLAS_NUM_THREADS -ErrorAction SilentlyContinue

cargo test --manifest-path qwen3-tts-rs/Cargo.toml --workspace --release --target x86_64-pc-windows-gnu
cargo build --manifest-path qwen3-tts-rs/Cargo.toml -p qwen-app --release --target x86_64-pc-windows-gnu
```

Ejecución rápida:

```powershell
.\qwen3-tts-rs\target\x86_64-pc-windows-gnu\release\qwen_tts.exe --caps
.\qwen3-tts-rs\target\x86_64-pc-windows-gnu\release\qwen_tts.exe --self-test
.\qwen3-tts-rs\target\x86_64-pc-windows-gnu\release\qwen_tts.exe -d models\qwen3-tts-0.6b -j 4 --seed 42 -s ryan -l English --quant-mixed --text "Hello, world!" -o output.wav
```
