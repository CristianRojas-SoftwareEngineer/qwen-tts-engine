# Qwen TTS Engine (`qwen-tts-engine`)

Motor de inferencia TTS autónomo en CPU escrito en Rust estable para Qwen3-TTS 0.6B.

[![CI](https://github.com/CristianRojas-SoftwareEngineer/qwen-tts-engine/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)
[![License: MPL-2.0](https://img.shields.io/badge/License-MPL--2.0-blue.svg)](LICENSE)
![Rust: Stable](https://img.shields.io/badge/Rust-stable-orange.svg)
![Model: Qwen3--TTS 0.6B](https://img.shields.io/badge/Model-Qwen3--TTS%200.6B-informational.svg)

**Qwen TTS Engine** convierte texto en audio PCM de 24 kHz directamente en CPU mediante un binario CLI (`qwen_tts`), un servidor HTTP compatible con OpenAI (`--serve`) y una librería Rust embebida (`qwen-engine`).

---

## Tabla de contenidos

- [Alcance y capacidades](#alcance-y-capacidades)
- [Requisitos de entorno](#requisitos-de-entorno)
- [Descarga de modelos](#descarga-de-modelos)
- [Uso rápido](#uso-rápido)
- [Verificación rápida](#verificación-rápida)
- [Documentación y arquitectura](#documentación-y-arquitectura)
- [Contribución y atribuciones](#contribución-y-atribuciones)
- [Licencia](#licencia)

---

## Alcance y capacidades

- **Síntesis local en CPU**: ejecuta **Qwen3-TTS `0.6B`** con cuantización recomendada `--quant-mixed` (`Q4_0` + `INT8`), despacho en runtime `AVX2 + FMA` en x86-64 (y *fallback* escalar en ARM64) y *streaming* progresivo.
- **Clonación de voz y expresividad**: soporta perfiles `.qvoice` v3 / `.bin`, extracción `ECAPA-TDNN` + RVQ desde audio de referencia, activos `emovoice`, `--emo-ref`, `--roughness` (`Q2_0`) y marcado inline (`[laugh]`, `[sigh]`, pausas).
- **Tres interfaces de integración**:
  - **CLI (`qwen_tts`)**: síntesis directa a WAV, *streaming* a WAV o `stdout` PCM crudo, lotes `--batch` y creación de voces.
  - **Servidor HTTP (`--serve`)**: rutas `/v1/audio/speech` (compatible con OpenAI), `/v1/tts` y `/v1/tts/stream`, con *continuous batching* (`--batch-size N`).
  - **Librería Rust (`qwen-engine`)**: API segura (`#![forbid(unsafe_code)]`) para incrustar el motor compartiendo `Arc<Model06B>` entre hilos.

> **Modelo soportado (`Qwen3-TTS 0.6B`)**: este motor ejecuta exclusivamente checkpoints **Qwen3-TTS `0.6B`** (`CustomVoice` para voces preset y `Base` para clonación de voz). Al cargar un modelo, `config.json` valida automáticamente esta geometría (`HIDDEN_SIZE=1024`, `28` capas Talker, `5` capas Code Predictor).

---

## Requisitos de entorno

### Dependencias por plataforma (perfil BLAS nativo por defecto)

| Plataforma | Toolchain y librerías | Notas |
|---|---|---|
| **Windows (`x86_64-pc-windows-gnu`)** | Rust estable (`x86_64-pc-windows-gnu`) + MSYS2 UCRT64 (`mingw-w64-ucrt-x86_64-gcc`, `mingw-w64-ucrt-x86_64-openblas`, `mingw-w64-ucrt-x86_64-binutils`) en `PATH` (`C:\msys64\ucrt64\bin`) | Si OpenBLAS está en otra ruta, define `OPENBLAS_LIB_DIR` (default `C:/msys64/ucrt64/lib`). |
| **Linux** | Rust estable + `libopenblas-dev` | Opcionalmente define `OPENBLAS_LIB_DIR` apuntando a `libopenblas.a` para enlace estático. |
| **macOS** | Rust estable + Xcode Command Line Tools | Enlaza con el framework del sistema `Accelerate`. |

### Variante opcional 100 % Rust (`--features faer-blas`)

Para compilar una variante en Rust puro mediante el backend `PureRustBlas` (basado en `faer`), habilita `--features faer-blas` y selecciónalo en ejecución con `QWEN_BLAS=faer`. Consulta la comparativa técnica en [`docs/adr/002-blas-strategy.md`](docs/adr/002-blas-strategy.md).

---

## Descarga de modelos

Descarga los pesos oficiales `0.6B` desde Hugging Face dentro de `models/`:

- **`models/qwen3-tts-0.6b`** ← `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` *(9 voces preset integradas)*
- **`models/qwen3-tts-0.6b-base`** ← `Qwen/Qwen3-TTS-12Hz-0.6B-Base` *(requerido para `--ref-audio` / clonación de voz)*

```bash
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --local-dir models/qwen3-tts-0.6b
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base --local-dir models/qwen3-tts-0.6b-base
```

---

## Uso rápido

### 1. CLI (`qwen_tts`)

#### Windows (MSYS2 UCRT64)

```powershell
$env:PATH = "C:\msys64\ucrt64\bin;$env:PATH"
$env:OMP_NUM_THREADS = "4"
Remove-Item Env:OPENBLAS_NUM_THREADS -ErrorAction SilentlyContinue

cargo build -p qwen-app --release --target x86_64-pc-windows-gnu
.\target\x86_64-pc-windows-gnu\release\qwen_tts.exe --caps
.\target\x86_64-pc-windows-gnu\release\qwen_tts.exe -d models\qwen3-tts-0.6b -j 4 --seed 42 -s ryan -l English --quant-mixed --text "Hello, world!" -o output.wav
```

#### Linux / macOS

```bash
cargo build -p qwen-app --release
./target/release/qwen_tts --caps
./target/release/qwen_tts -d models/qwen3-tts-0.6b -j 4 --seed 42 -s ryan -l English --quant-mixed --text "Hello, world!" -o output.wav
```

> **Nota**: `--quant-mixed` es el perfil recomendado para producción en `0.6B`; añade `--stream` si necesitas escribir audio progresivamente conforme se decodifica. En Windows GNU, mantén `OPENBLAS_NUM_THREADS` sin definir para que el motor ajuste los hilos entre *prefill* y generación.

### 2. Clonación de voz (`0.6B-Base`)

El audio de referencia debe ser **WAV PCM de 24 kHz mono** (si parte de MP3/MP4/OGG/FLAC, conviértelo antes con `ffmpeg -i entrada.mp3 -ar 24000 -ac 1 ref_24k.wav`).

```bash
# 1. Extraer perfil .qvoice reutilizable y sintetizar el primer audio
./target/release/qwen_tts -d models/qwen3-tts-0.6b-base -j 4 --quant-mixed \
  --ref-audio ref_24k.wav --ref-text "Texto exacto del audio de referencia." \
  --save-voice mi_voz.qvoice -l Spanish --text "Hola, esta es mi voz clonada." -o clon.wav

# 2. Reutilizar el perfil guardado (sin volver a codificar la referencia)
./target/release/qwen_tts -d models/qwen3-tts-0.6b-base -j 4 --quant-mixed \
  --load-voice mi_voz.qvoice -l Spanish --text "Segunda frase con la misma voz." -o clon2.wav
```

### 3. Servidor HTTP compatible con OpenAI (`--serve`)

Inicia el servidor en `127.0.0.1:8080` con *continuous batching* (`--batch-size 4`) y 4 hilos:

```bash
./target/release/qwen_tts -d models/qwen3-tts-0.6b -j 4 --quant-mixed --serve 8080 --batch-size 4
```

Envía peticiones síncronas (WAV completo) o en *streaming* (PCM `s16le` 24 kHz mono):

```bash
# WAV completo (OpenAI-compatible)
curl -X POST http://127.0.0.1:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello from Qwen TTS Engine!","voice":"ryan","language":"English"}' \
  --output speech.wav

# Streaming PCM s16le 24 kHz mono (Transfer-Encoding: chunked)
curl -X POST http://127.0.0.1:8080/v1/tts/stream \
  -H "Content-Type: application/json" \
  -d '{"text":"Streaming audio frame by frame.","speaker":"ryan","language":"English"}' \
  --output stream.pcm
```

### 4. Librería embebida en Rust (`qwen-engine`)

Agrega las dependencias del workspace en tu `Cargo.toml`:

```toml
[dependencies]
qwen-engine = { path = "crates/qwen-engine" }
qwen-core = { path = "crates/qwen-core" }
qwen-kernels = { path = "crates/qwen-kernels" }
```

Ejemplo mínimo de síntesis directa en memoria:

```rust
use std::path::Path;
use std::sync::Arc;
use qwen_engine::{
    synthesize_to_samples, write_wav, Model06B, ModelLoadOptions, RequestConfig,
    RequestExecution, SessionScratch, VoiceBindings,
};
use qwen_kernels::ComputeRuntime;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = Arc::new(ComputeRuntime::new(Some(4)));
    let opts = ModelLoadOptions { quant_mixed: true, ..Default::default() };
    let model = Model06B::load(Path::new("models/qwen3-tts-0.6b"), &runtime, opts)?;

    let voice = VoiceBindings::preset(3061); // ryan
    let mut scratch = SessionScratch::new();
    let exec = RequestExecution::new(RequestConfig::default());

    let (samples, _timing) = synthesize_to_samples(
        &runtime, &model, &voice, &mut scratch, &exec, "Hello from Rust!",
    )?;
    write_wav(Path::new("out.wav"), &samples, 24_000)?;
    Ok(())
}
```

Para *streaming* sobre `AudioSink` (`synthesize_to_sink`), cancelación cooperativa (`AbortHandle`) y carga de voces clonadas, consulta [`docs/INTERFACE_CONTRACT.md`](docs/INTERFACE_CONTRACT.md).

---

## Verificación rápida

Comprueba los kernels numéricos del binario compilado y la batería de pruebas del workspace:

```bash
# Linux / macOS
./target/release/qwen_tts --self-test
cargo test --workspace --release
```

En Windows (`x86_64-pc-windows-gnu`):

```powershell
$env:PATH = "C:\msys64\ucrt64\bin;$env:PATH"
$env:OMP_NUM_THREADS = "4"
Remove-Item Env:OPENBLAS_NUM_THREADS -ErrorAction SilentlyContinue

.\target\x86_64-pc-windows-gnu\release\qwen_tts.exe --self-test
cargo test --workspace --release --target x86_64-pc-windows-gnu
```

Consulta el detalle de suites, pruebas de integración con pesos reales y matriz de CI en [`docs/VERIFICATION_PLAN.md`](docs/VERIFICATION_PLAN.md).

---

## Documentación y arquitectura

La documentación sigue el modelo **Diátaxis** para separar guías prácticas, contratos de referencia, arquitectura y decisiones de diseño:

| Documento | Rol Diátaxis | Audiencia principal | Qué contiene |
|---|---|---|---|
| [`docs/INTERFACE_CONTRACT.md`](docs/INTERFACE_CONTRACT.md) | Referencia | Usuarios avanzados, integradores HTTP y Rust | Flags CLI (`qwen_tts`), endpoints HTTP (`--serve`), API Rust embebida, variables de entorno y logs contractuales. |
| [`docs/SPEC_06B.md`](docs/SPEC_06B.md) | Referencia | Ingenieros del motor y mantenedores | Geometría fija `0.6B`, mapa de tensores `safetensors`, formatos `.qvoice` v3 / `.bin`, vocabularios, *prefill* y marcado inline. |
| [`docs/NUMERIC_CONTRACT.md`](docs/NUMERIC_CONTRACT.md) | Referencia | Ingenieros de kernels y optimización | Cuantización `INT8`/`Q4_0`/`Q2_0`, `FTZ`/`DAZ`, reglas IEEE-754/`FMA`, determinismo de *streaming* y tolerancias. |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Explicación | Desarrolladores y contribuidores | Grafo de *crates* (DAG), política `unsafe`, cuatro tiempos de vida del estado, `ComputeRuntime`, pipeline y *schedulers*. |
| [`docs/VERIFICATION_PLAN.md`](docs/VERIFICATION_PLAN.md) | Guía operativa | QA, contribuidores y mantenimiento de CI | Suites `cargo test`, verificación numérica de kernels y matriz de `.github/workflows/ci.yml`. |
| [`docs/adr/001-only-06b.md`](docs/adr/001-only-06b.md) | ADR | Mantenedores | Especialización del motor para Qwen3-TTS `0.6B`. |
| [`docs/adr/002-blas-strategy.md`](docs/adr/002-blas-strategy.md) | ADR | Mantenedores | Estrategia BLAS (`FfiBlas` por defecto y `faer-blas` opcional). |
| [`docs/adr/003-concurrency-pool.md`](docs/adr/003-concurrency-pool.md) | ADR | Mantenedores | Pool de hilos `ComputePool` y control de sobresuscripción. |
| [`docs/adr/004-simd-baseline.md`](docs/adr/004-simd-baseline.md) | ADR | Mantenedores | Despacho SIMD en runtime sobre Rust estable. |

---

## Contribución y atribuciones

- **Issues y Pull Requests**: reporta errores o propón cambios mediante GitHub Issues y Pull Requests verificando previamente `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace --release` y la alineación con los contratos de `docs/`.
- **Atribución upstream**: implementación independiente en Rust inspirada y portada a partir del motor en C `gabriele-mastrapasqua/qwen3-tts` (licencia MIT). Consulta [`NOTICE`](NOTICE) para el detalle completo.

---

## Licencia

Este proyecto está licenciado bajo **Mozilla Public License Version 2.0 (`MPL-2.0`)**. Consulta [`LICENSE`](LICENSE) para el texto completo de la licencia y [`NOTICE`](NOTICE) para los avisos de atribución.
