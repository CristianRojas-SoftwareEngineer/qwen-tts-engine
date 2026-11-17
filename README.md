# Qwen TTS Engine (`qwen-tts-engine`)

Motor de inferencia TTS en CPU escrito en Rust estable (actualmente con soporte para **Qwen3-TTS 0.6B** `CustomVoice` y `Base`, sin dependencias de Python, LibTorch ni Candle), con soporte de CLI, streaming progresivo, clonación de voz (`.qvoice`/`.bin`, `ECAPA-TDNN`, `SpeechEncoder` RVQ), expresividad `0.6B` (`emovoice`, `--emo-ref`, `--roughness`, marcado inline), y servidor HTTP compatible con OpenAI (`/v1/tts`, `/v1/tts/stream`, `/v1/audio/speech`, `/v1/health`, `/v1/speakers`) con `WorkerPoolScheduler` y `ContinuousBatchScheduler`.

## Estructura del repositorio

- `Cargo.toml` / `Cargo.lock` — Workspace Cargo en la raíz:
  - `crates/qwen-kernels` — `AlignedBuf<64>`, `CpuCaps`, `ComputePool`, `FfiBlas` (`OpenBLAS`/`Accelerate`) + `PureRustBlas` opcional (`--features faer-blas`), cuantización `int8`/`q4_0`/`q2_0`, kernels SIMD (`AVX2+FMA` en x86-64, fallback entero escalar portable), `--self-test` y `--matmat-bench`.
  - `crates/qwen-core` — Constantes `0.6B`, `MmapStore`, `BpeTokenizer`, `Sampler`/`LcgRng`, `.qvoice`/`.bin`, `MarkupPlanner`.
  - `crates/qwen-engine` (`#![forbid(unsafe_code)]`) — `Model06B`, `VoiceBindings`, `SessionScratch`, `Talker`, `CodePredictor`, `SpeechDecoder`, `SpeechEncoder`, `ECAPA-TDNN`, `emotion`, `compose`, `batch`, `audio`, `pipeline`.
  - `crates/qwen-app` — Binario `qwen_tts`: CLI completa, afinidad SMT Win32, manifiesto UTF-8, `--watch-stdin`, servidor HTTP y schedulers (`--workers`, `--batch-size`).
- `docs/` — Contratos y arquitectura ([INTERFACE_CONTRACT.md](docs/INTERFACE_CONTRACT.md), [SPEC_06B.md](docs/SPEC_06B.md), [NUMERIC_CONTRACT.md](docs/NUMERIC_CONTRACT.md), [ARCHITECTURE.md](docs/ARCHITECTURE.md), [VERIFICATION_PLAN.md](docs/VERIFICATION_PLAN.md)) y [docs/adr/](docs/adr/001-only-06b.md).
- `.github/workflows/ci.yml` — CI multiplataforma (Windows `x86_64-pc-windows-gnu`, Linux `x86_64`/`aarch64`, macOS `arm64`).
- `models/` — *(Opcional, ignorado en Git)* Pesos locales `0.6B` (`models/qwen3-tts-0.6b` y `models/qwen3-tts-0.6b-base`).
- `LICENSE` — Licencia Mozilla Public License 2.0 (`MPL-2.0`; ver `NOTICE` para atribución upstream).

## Requisitos de entorno

- **Windows (`x86_64-pc-windows-gnu`)**: toolchain GNU de MSYS2 UCRT64 (`mingw-w64-ucrt-x86_64-gcc`, `mingw-w64-ucrt-x86_64-openblas`, `mingw-w64-ucrt-x86_64-binutils`) en `PATH` (`C:\msys64\ucrt64\bin`). Si OpenBLAS está instalado en otra ruta, define `OPENBLAS_LIB_DIR` (por defecto `C:/msys64/ucrt64/lib`).
- **Linux**: `libopenblas-dev` (o `OPENBLAS_LIB_DIR` apuntando a `libopenblas.a`).
- **macOS**: framework del sistema `Accelerate` (incluido con Xcode Command Line Tools).
- **Backend BLAS puro en Rust (opcional)**: añade `--features faer-blas` (sin dependencia externa de OpenBLAS en tiempo de enlace si se configura para ello, o seleccionable en ejecución con `QWEN_BLAS=faer`).

## Modelos (`0.6B`)

Descarga los pesos desde Hugging Face y colócalos bajo `models/`:

- `models/qwen3-tts-0.6b` ← `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`
- `models/qwen3-tts-0.6b-base` ← `Qwen/Qwen3-TTS-12Hz-0.6B-Base` *(necesario para clonación de voz)*

```bash
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --local-dir models/qwen3-tts-0.6b
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base --local-dir models/qwen3-tts-0.6b-base
```

## Compilación y tests

### Windows (MSYS2 UCRT64)

```powershell
$env:PATH = "C:\msys64\ucrt64\bin;$env:PATH"
$env:OMP_NUM_THREADS = "4"
Remove-Item Env:OPENBLAS_NUM_THREADS -ErrorAction SilentlyContinue

cargo test --workspace --release --target x86_64-pc-windows-gnu
cargo build -p qwen-app --release --target x86_64-pc-windows-gnu
```

Ejecución rápida en Windows:

```powershell
.\target\x86_64-pc-windows-gnu\release\qwen_tts.exe --caps
.\target\x86_64-pc-windows-gnu\release\qwen_tts.exe --self-test
.\target\x86_64-pc-windows-gnu\release\qwen_tts.exe -d models\qwen3-tts-0.6b -j 4 --seed 42 -s ryan -l English --quant-mixed --text "Hello, world!" -o output.wav
```

### Linux / macOS

```bash
cargo test --workspace --release
cargo build -p qwen-app --release

./target/release/qwen_tts --caps
./target/release/qwen_tts --self-test
./target/release/qwen_tts -d models/qwen3-tts-0.6b -j 4 --seed 42 -s ryan -l English --quant-mixed --text "Hello, world!" -o output.wav
```

## Licencia

Este proyecto está licenciado bajo **Mozilla Public License Version 2.0 (`MPL-2.0`)**. Consulta [`LICENSE`](LICENSE) para el texto completo de la licencia y [`NOTICE`](NOTICE) para la atribución upstream.
