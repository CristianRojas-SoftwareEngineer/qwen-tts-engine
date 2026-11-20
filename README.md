# Qwen TTS Engine (`qwen-tts-engine`)

Motor de inferencia TTS autónomo en CPU escrito en Rust estable para Qwen3-TTS 0.6B.

[![CircleCI](https://dl.circleci.com/status-badge/img/gh/CristianRojas-SoftwareEngineer/qwen-tts-engine/tree/main.svg?style=shield)](https://dl.circleci.com/status-badge/redirect/gh/CristianRojas-SoftwareEngineer/qwen-tts-engine/tree/main)
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

- **Síntesis local en CPU**: ejecuta **Qwen3-TTS `0.6B`** con cuantización recomendada `--quant-mixed` (`Q4_0` + `INT8`), despacho en runtime de kernels `AVX2 + FMA` en x86-64 y NEON en ARM64 (con `SDOT` si la CPU tiene `dotprod`; ver [requisitos mínimos](#requisitos-mínimos-recomendados-síntesis-en-tiempo-real)) y *streaming* progresivo.
- **Clonación de voz y expresividad**: soporta perfiles `.qvoice` v3 / `.bin`, extracción `ECAPA-TDNN` + RVQ desde audio de referencia, activos `emovoice`, `--emo-ref`, `--roughness` (`Q2_0`) y marcado inline (`[laugh]`, `[sigh]`, pausas).
- **Tres interfaces de integración**:
  - **CLI (`qwen_tts`)**: síntesis directa a WAV, *streaming* a WAV o `stdout` PCM crudo, lotes `--batch` y creación de voces.
  - **Servidor HTTP (`--serve`)**: rutas `/v1/audio/speech` (compatible con OpenAI), `/v1/tts` y `/v1/tts/stream`, con *continuous batching* (`--batch-size N`).
  - **Librería Rust (`qwen-engine`)**: API segura (`#![forbid(unsafe_code)]`) para incrustar el motor compartiendo `Arc<Model06B>` entre hilos.

> **Modelo soportado (`Qwen3-TTS 0.6B`)**: este motor ejecuta exclusivamente checkpoints **Qwen3-TTS `0.6B`** (`CustomVoice` para voces preset y `Base` para clonación de voz). Al cargar un modelo, `config.json` valida automáticamente esta geometría (`HIDDEN_SIZE=1024`, `28` capas Talker, `5` capas Code Predictor).

---

## Requisitos de entorno

### Requisitos mínimos recomendados (síntesis en tiempo real)

«Tiempo real» significa RTF < 1, es decir, generar el audio más rápido de lo que dura, para que el *streaming* no se corte. Se cumple tanto con el equipo libre como con una carga de fondo moderada (1 hilo copiando 256 MiB en bucle).

| Plataforma | CPU | Núcleos físicos | RAM | Disco | Estado |
|---|---|---|---|---|---|
| **Windows x86_64** (`x86_64-pc-windows-gnu`) | x86_64 con **AVX2 y FMA** (Intel Core de 4.ª generación / Haswell o posterior; AMD Ryzen / Zen o posterior) | **8** | 8 GB, doble canal | ~5 GB (los dos modelos) | ✅ Verificado en el equipo de referencia (AMD Ryzen 7 5825U, 8 núcleos), frase larga: RTF 0,58–0,64 sin carga y 0,89–0,92 con carga |
| **Linux x86_64** (`x86_64-unknown-linux-gnu`) | Igual que Windows | **8** | 8 GB, doble canal | ~5 GB | ⚠️ Misma ruta de kernels AVX2 que Windows; requisitos esperados iguales, sin medición propia en Linux |
| **macOS ARM64** (`aarch64-apple-darwin`) | Apple Silicon (M1 o posterior) | — | 8 GB | ~5 GB | ✅ Sin carga: RTF 0,25–0,26 en un Apple M4 Pro virtual de 6 núcleos (CircleCI `m4pro.medium`), con todos los kernels en NEON; la ruta escalar daba 1,36–1,56 en la misma máquina. Sin medir con carga ni en otros Mac, así que no hay mínimo de núcleos verificado. |
| **Linux ARM64** (`aarch64-unknown-linux-gnu`) | ARMv8.2 o posterior con NEON y `dotprod` (Neoverse N1/V1, Cortex-A76 o posterior); también corre en ARMv8.0, más lento | más de 4 (sin verificar) | 8 GB | ~5 GB | ❌ Sin tiempo real con 4 vCPU, aunque cerca: RTF 1,12–1,21 en un Arm Neoverse V1 de 4 vCPU con el OpenBLAS propio (1,20–1,33 con el de Ubuntu 22.04) y 1,54–1,62 en un Neoverse N1 de 4 vCPU; la ruta escalar daba ~5,4. Con 4 núcleos, el productor (85–118 ms por fotograma de 80) y el decoder se reparten las mismas CPU, igual que en x86 con 4 núcleos. No se ha podido medir con 8 vCPU: CircleCI no ofrece a este proyecto máquinas ARM más grandes. |

Notas sobre estos requisitos:
- **De dónde salen las cifras.** Windows x86_64 se midió en el equipo de referencia, con y sin carga. macOS y Linux ARM64 se midieron en máquinas virtuales de CircleCI, solo sin carga, con la mayor máquina disponible (6 y 4 núcleos), así que no son comparables entre sí ni con Windows. Detalle de cada medición en [ADR-005](docs/adr/005-cpu-inference-optimizations.md).
- **Núcleos físicos, no hilos.** El motor se fija a un hilo lógico por núcleo físico (Windows). Con menos de 8 núcleos, el decodificador y OpenBLAS compiten con el productor por las mismas CPU. Mediciones en el mismo equipo, limitado por afinidad:
  - **6 núcleos:** RTF 0,89 sin carga, pero 1,13 con carga.
  - **4 núcleos:** 0,97 sin carga y 1,45 con carga.
- **Sin AVX2**, el motor usa la ruta escalar y va unas 5–6 veces más lento (RTF ~3,8 en el mismo equipo).
- **Memoria.** El proceso usa ~1,3 GB en el pico. El motor está limitado por el ancho de banda de memoria, por eso se recomienda doble canal.
- **Ajuste de hilos.** `-j` fija los hilos del productor (4 por defecto). Menos hilos no compensa la falta de núcleos: con 4 núcleos, `-j 2` tampoco bajó de RTF 1.

### Dependencias por plataforma (perfil BLAS nativo por defecto)

| Plataforma | Toolchain y librerías | Notas |
|---|---|---|
| **Windows (`x86_64-pc-windows-gnu`)** | Rust estable (`x86_64-pc-windows-gnu`) + MSYS2 UCRT64 (`mingw-w64-ucrt-x86_64-gcc`, `mingw-w64-ucrt-x86_64-openblas`, `mingw-w64-ucrt-x86_64-binutils`) en `PATH` (`C:\msys64\ucrt64\bin`) | Si OpenBLAS está en otra ruta, define `OPENBLAS_LIB_DIR` (default `C:/msys64/ucrt64/lib`). |
| **Linux** | Rust estable + `libopenblas-dev`, o un OpenBLAS propio (abajo) | **Para distribuir el binario, y sobre todo en ARM64, enlaza un OpenBLAS propio**: `scripts/build-openblas.sh "$HOME/openblas"` (OpenBLAS 0.3.30 estático, solo BLAS, `DYNAMIC_ARCH`: elige el kernel de cada CPU al arrancar; ~4–6 min en 4 vCPU ARM) y `OPENBLAS_LIB_DIR="$HOME/openblas/lib" cargo build --release -p qwen-app`. El de la distribución puede traer solo el kernel genérico: Ubuntu 22.04 en ARM64 usa `armv8` y el decoder va un 30 % más lento. `qwen_tts --caps` muestra el kernel en la línea `BLAS (prefill)`. |
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
Remove-Item Env:OPENBLAS_NUM_THREADS -ErrorAction SilentlyContinue

cargo build -p qwen-app --release --target x86_64-pc-windows-gnu
.\target\x86_64-pc-windows-gnu\release\qwen_tts.exe --caps
.\target\x86_64-pc-windows-gnu\release\qwen_tts.exe -d models\qwen3-tts-0.6b -j 4 --seed 4 -s ryan -l English --quant-mixed --text "Hello, world!" -o output.wav
```

#### Linux / macOS

```bash
cargo build -p qwen-app --release
./target/release/qwen_tts --caps
./target/release/qwen_tts -d models/qwen3-tts-0.6b -j 4 --seed 4 -s ryan -l English --quant-mixed --text "Hello, world!" -o output.wav
```

> **Nota**: `--quant-mixed` es el perfil recomendado para producción en `0.6B` (los valores por defecto de muestreo son `temperature = 0.325` y `seed = 4`); añade `--stream` si necesitas escribir audio progresivamente conforme se decodifica. En Windows GNU, mantén `OPENBLAS_NUM_THREADS` sin definir para que el motor ajuste los hilos entre *prefill* y generación.

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
| [`docs/VERIFICATION_PLAN.md`](docs/VERIFICATION_PLAN.md) | Guía operativa | QA, contribuidores y mantenimiento de CI | Suites `cargo test`, verificación numérica de kernels, CI en CircleCI (`.circleci/config.yml`) y protocolo de medición de rendimiento. |
| [`docs/adr/001-only-06b.md`](docs/adr/001-only-06b.md) | ADR | Mantenedores | Especialización del motor para Qwen3-TTS `0.6B`. |
| [`docs/adr/002-blas-strategy.md`](docs/adr/002-blas-strategy.md) | ADR | Mantenedores | Estrategia BLAS (`FfiBlas` por defecto y `faer-blas` opcional). |
| [`docs/adr/003-concurrency-pool.md`](docs/adr/003-concurrency-pool.md) | ADR | Mantenedores | Pool de hilos `ComputePool` y control de sobresuscripción. |
| [`docs/adr/004-simd-baseline.md`](docs/adr/004-simd-baseline.md) | ADR | Mantenedores | Despacho SIMD en runtime sobre Rust estable (AVX2, NEON y escalar). |
| [`docs/adr/005-cpu-inference-optimizations.md`](docs/adr/005-cpu-inference-optimizations.md) | ADR | Mantenedores | Optimizaciones de la inferencia en CPU adoptadas y descartadas, con su efecto medido, y rendimiento por plataforma. |

---

## Contribución y atribuciones

- **Issues y Pull Requests**: reporta errores o propón cambios mediante GitHub Issues y Pull Requests verificando previamente `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace --release` y la alineación con los contratos de `docs/`.
- **Atribución upstream**: implementación independiente en Rust inspirada y portada a partir del motor en C `gabriele-mastrapasqua/qwen3-tts` (licencia MIT). Consulta [`NOTICE`](NOTICE) para el detalle completo.

---

## Licencia

Este proyecto está licenciado bajo **Mozilla Public License Version 2.0 (`MPL-2.0`)**. Consulta [`LICENSE`](LICENSE) para el texto completo de la licencia y [`NOTICE`](NOTICE) para los avisos de atribución.
