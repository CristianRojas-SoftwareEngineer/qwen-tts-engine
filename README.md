# Qwen TTS Engine (`qwen-tts-engine`)

Motor de inferencia TTS autónomo en CPU escrito en Rust estable para Qwen3-TTS 0.6B.

[![CircleCI](https://dl.circleci.com/status-badge/img/gh/CristianRojas-SoftwareEngineer/qwen-tts-engine/tree/main.svg?style=shield)](https://dl.circleci.com/status-badge/redirect/gh/CristianRojas-SoftwareEngineer/qwen-tts-engine/tree/main)
[![License: MPL-2.0](https://img.shields.io/badge/License-MPL--2.0-blue.svg)](LICENSE)
![Rust: Stable](https://img.shields.io/badge/Rust-stable-orange.svg)
![Model: Qwen3--TTS 0.6B](https://img.shields.io/badge/Model-Qwen3--TTS%200.6B-informational.svg)

**Qwen TTS Engine** convierte texto en audio PCM de 24 kHz directamente en CPU mediante un binario CLI (`qwen_tts`), un servidor HTTP (`--serve`) y una biblioteca Rust embebida (`qwen-engine`).

---

## Tabla de contenidos

- [Alcance y capacidades](#alcance-y-capacidades)
- [Descargas](#descargas)
- [Requisitos de entorno](#requisitos-de-entorno)
- [Descarga de modelos](#descarga-de-modelos)
- [Uso rápido](#uso-rápido)
- [Verificación rápida](#verificación-rápida)
- [Documentación y arquitectura](#documentación-y-arquitectura)
- [Contribución y atribuciones](#contribución-y-atribuciones)
- [Licencia](#licencia)

---

## Alcance y capacidades

- **Síntesis local en CPU**: ejecuta **Qwen3-TTS `0.6B`** con cuantización fija (`Q4_0` + `INT8`), despacho en runtime de kernels `AVX2 + FMA` en x86-64 y NEON en ARM64 (con `SDOT` si la CPU tiene `dotprod`; ver [requisitos mínimos](#requisitos-mínimos-recomendados-síntesis-en-tiempo-real)) y *streaming* progresivo.
- **Una sola forma de voz: la clonada**: el motor trae incorporada la voz `default` y cualquier otra la creas clonándola desde un audio de referencia (x-vector `ECAPA-TDNN`, que fija el timbre). Las voces se guardan como perfiles `.qvoice` v6 de 4 KB. Además: marcado en línea (pausas y muletillas como `[hmm]`).
- **Tres interfaces de integración**:
  - **CLI (`qwen_tts`)**: síntesis directa a WAV, PCM crudo por `stdout` y creación de voces.
  - **Servidor HTTP (`--serve`)**: rutas `/v1/tts` y `/v1/tts/stream`, una síntesis a la vez.
  - **Biblioteca Rust (`qwen-engine`)**: API segura (`#![forbid(unsafe_code)]`) para incrustar el motor compartiendo `&Model06B` (`Send + Sync`) entre hilos.

> **Modelo soportado (`Qwen3-TTS 0.6B Base`)**: este motor ejecuta exclusivamente el checkpoint **Qwen3-TTS `0.6B` Base**; cualquier otro se rechaza al cargar. Al cargar un modelo, `config.json` valida automáticamente esta geometría (`HIDDEN_SIZE=1024`, `28` capas Talker, `5` capas Code Predictor).

---

## Descargas

Las releases están en [GitHub Releases](https://github.com/CristianRojas-SoftwareEngineer/qwen-tts-engine/releases). Cada una trae un paquete por sistema, las fuentes y `SHA256SUMS`:

| Sistema | Paquete |
|---|---|
| Windows x86_64 | `qwen-tts-engine-<versión>-x86_64-pc-windows-gnu.zip` |
| Linux x86_64 | `qwen-tts-engine-<versión>-x86_64-unknown-linux-gnu.tar.gz` |
| Linux ARM64 | `qwen-tts-engine-<versión>-aarch64-unknown-linux-gnu.tar.gz` |
| macOS ARM64 | `qwen-tts-engine-<versión>-aarch64-apple-darwin.tar.gz` |

Cada paquete contiene el binario `qwen_tts` (`qwen_tts.exe` en Windows), `LICENSE`, `NOTICE` y `THIRD-PARTY-LICENSES.md`.

- **Requisitos de los binarios**: Linux con glibc 2.28 o posterior, macOS 26.0 o posterior y Windows x86_64. En x86_64 la ruta rápida necesita AVX2 y FMA; sin ellos el motor usa la ruta escalar, mucho más lenta.
- **Verificación**: descarga todos los archivos en una misma carpeta y ejecuta `sha256sum -c SHA256SUMS`.
- **Modelo**: los pesos no van en el paquete; se descargan aparte como indica [Descarga de modelos](#descarga-de-modelos).
- **Versionado**: las versiones siguen el [versionado semántico](https://semver.org/lang/es/) sobre el contrato de uso; ver la sección «Versionado» de [`docs/INTERFACE_CONTRACT.md`](docs/INTERFACE_CONTRACT.md). Lo que cambia en cada versión está en [`CHANGELOG.md`](CHANGELOG.md).

---

## Requisitos de entorno

### Requisitos mínimos recomendados (síntesis en tiempo real)

«Tiempo real» significa RTF < 1, es decir, generar el audio más rápido de lo que dura, para que el *streaming* no se corte. En el equipo de referencia se cumple con margen con el equipo libre; con una carga de fondo moderada (1 hilo copiando 256 MiB en bucle), la frase larga queda al límite y la corta pasa de 1.

| Plataforma | CPU | Núcleos físicos | RAM | Disco | Estado |
|---|---|---|---|---|---|
| **Windows x86_64** (`x86_64-pc-windows-gnu`) | x86_64 con **AVX2 y FMA** (Intel Core de 4.ª generación / Haswell o posterior; AMD Ryzen / Zen o posterior) | **8** | 8 GB, doble canal | ~2,5 GB (el modelo) | ✅ Verificado en el equipo de referencia (AMD Ryzen 7 5825U, 8 núcleos), v1.2.1, frase larga en español con la voz `default`, CLI y servidor: RTF 0,67 sin carga y 0,99–1,02 con carga (repeticiones de 0,93 a 1,06), así que con carga no hay margen. En CircleCI, con 4 núcleos (`large`): 1,16–1,28 sin carga, sin tiempo real (la misma clase de máquina daba 0,89–0,98 cuando se documentó y hoy la v1.0.0 da 1,47–1,63 en ella, así que cambió la máquina, no el motor) |
| **Linux x86_64** (`x86_64-unknown-linux-gnu`) | Igual que Windows | **8** | 8 GB, doble canal | ~2,5 GB | ⚠️ Misma ruta de kernels AVX2 que Windows; requisitos esperados iguales. La CI lo compila y lo prueba, pero no mide su rendimiento. Sin afinidad de hilos: el motor cuenta los hilos lógicos; con SMT y `-j` entre los núcleos físicos y los hilos lógicos, los hilos esperan girando |
| **macOS ARM64** (`aarch64-apple-darwin`) | Apple Silicon (M1 o posterior) | — | 8 GB | ~2,5 GB | ✅ Sin carga: RTF 0,23–0,25 en un Apple M4 Pro virtual de 6 núcleos (CircleCI `m4pro.medium`), con todos los kernels en NEON. Sin medir con carga ni en otros Mac, así que no hay mínimo de núcleos verificado. |
| **Linux ARM64** (`aarch64-unknown-linux-gnu`) | ARMv8.2 o posterior con NEON y `dotprod` (Neoverse N1/V1, Cortex-A76 o posterior); también corre en ARMv8.0, más lento | más de 4 (sin verificar) | 8 GB | ~2,5 GB | ❌ Sin tiempo real con 4 vCPU: RTF 1,43–1,48 en un Arm Neoverse N1 de 4 vCPU (CircleCI Docker `arm.large`). Con 4 núcleos, el productor (Talker y Code Predictor, 100–107 ms por frame de 80 ms) y el decoder se reparten las mismas CPU. No se ha podido medir con 8 vCPU: CircleCI no ofrece a este proyecto máquinas ARM más grandes. |

Notas sobre estos requisitos:
- **De dónde salen las cifras.** Windows x86_64 se midió en el equipo de referencia, con y sin carga (1 hilo copiando 256 MiB en bucle); macOS, Linux ARM64 y el Windows de 4 núcleos, en máquinas virtuales de CircleCI, solo sin carga, con la voz `default` en inglés (frase corta y larga), así que no son comparables entre sí ni con el equipo de referencia. Detalle de cada medición en [ADR-005](docs/adr/005-cpu-inference-optimizations.md).
- **Núcleos físicos, no hilos.** El motor se fija a un hilo lógico por núcleo físico (Windows). Con menos de 8 núcleos, el decoder y OpenBLAS compiten con el productor por las mismas CPU. Mediciones en el mismo equipo, limitado por afinidad:
  - **6 núcleos:** RTF 0,93 sin carga y 1,30 con carga.
  - **4 núcleos:** 1,01 sin carga y 1,49 con carga.
- **Sin AVX2**, el motor usa la ruta escalar y va unas 5 veces más lento (RTF ~3,6 en el mismo equipo).
- **Memoria.** El proceso llega a ~1 353 MiB de memoria comprometida privada en el pico (CLI, frase larga); el *working set* llega a ~2 362 MiB porque incluye los pesos del modelo mapeados desde el disco. El motor está limitado por el ancho de banda de memoria, por eso se recomienda doble canal.
- **Ajuste de hilos.** `-j` fija los hilos de cómputo; por defecto, los hilos lógicos de la CPU hasta un máximo de 4. Con 4 núcleos, `-j 2` rinde algo mejor que `-j 4` (RTF 0,92 sin carga y 1,31 con carga, una corrida), pero con carga tampoco llega a tiempo real.

### Dependencias por plataforma (BLAS nativo)

| Plataforma | Toolchain y bibliotecas | Notas |
|---|---|---|
| **Windows (`x86_64-pc-windows-gnu`)** | Rust con host `x86_64-pc-windows-gnu` solo para este repositorio: en su carpeta, `rustup toolchain install 1.96.0-x86_64-pc-windows-gnu --component clippy --target x86_64-pc-windows-gnu` y `rustup override set 1.96.0-x86_64-pc-windows-gnu`, con la versión de `channel` en `rust-toolchain.toml` (al cambiarla, repite los dos comandos con la nueva). El override no cambia el host por defecto de la máquina, así que no afecta a otros proyectos Rust que usen MSVC, y no hace falta Visual Studio. Además: MSYS2 UCRT64 (`mingw-w64-ucrt-x86_64-gcc`, `make`, `mingw-w64-ucrt-x86_64-binutils`) en `PATH` (`C:\msys64\ucrt64\bin`) y el OpenBLAS propio (abajo) | El motor enlaza un OpenBLAS 0.3.34 propio, estático y sin OpenMP ni Fortran: `OPENBLAS_LIB_DIR` es obligatoria (`C:/openblas/lib`). |
| **Linux** | Rust estable + OpenBLAS propio | El motor enlaza siempre un OpenBLAS propio: `scripts/build-openblas.sh "$HOME/openblas"` (OpenBLAS 0.3.34 estático, solo BLAS, `DYNAMIC_ARCH`: elige el kernel de cada CPU al arrancar; ~4–6 min en 4 vCPU ARM) y `OPENBLAS_LIB_DIR="$HOME/openblas/lib" cargo build --release -p qwen-app`; `OPENBLAS_LIB_DIR` es obligatoria. El de la distribución puede traer solo el kernel genérico: Ubuntu 22.04 en ARM64 usa `armv8`. `qwen_tts --caps` muestra el kernel en la línea `BLAS (prefill)`. |
| **macOS** | Rust estable + Xcode Command Line Tools | Enlaza con el framework del sistema `Accelerate`. |

---

## Descarga de modelos

Descarga el único modelo que usa el motor, **Qwen3-TTS `0.6B` Base**, desde Hugging Face dentro de `models/`:

- **`models/qwen3-tts-0.6b-base`** ← `Qwen/Qwen3-TTS-12Hz-0.6B-Base`

```bash
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base --local-dir models/qwen3-tts-0.6b-base
```

---

## Uso rápido

### 1. CLI (`qwen_tts`)

#### Windows (MSYS2 UCRT64)

Antes de la primera compilación, compila una vez el OpenBLAS propio en `C:/openblas` (unos 30 minutos con 8 núcleos):

```powershell
$env:MSYSTEM = "UCRT64"; $env:CHERE_INVOKING = "1"
C:\msys64\usr\bin\bash.exe -lc 'pacman -S --needed --noconfirm make && bash scripts/build-openblas.sh /c/openblas 0.3.34'
```

```powershell
$env:PATH = "C:\msys64\ucrt64\bin;$env:PATH"
$env:OPENBLAS_LIB_DIR = "C:/openblas/lib"

cargo build -p qwen-app --release --target x86_64-pc-windows-gnu
.\target\x86_64-pc-windows-gnu\release\qwen_tts.exe --caps
.\target\x86_64-pc-windows-gnu\release\qwen_tts.exe -d models\qwen3-tts-0.6b-base -j 4 --seed 4 -l English --text "Hello, world!" -o output.wav
```

#### Linux / macOS

```bash
cargo build -p qwen-app --release
./target/release/qwen_tts --caps
./target/release/qwen_tts -d models/qwen3-tts-0.6b-base -j 4 --seed 4 -l English --text "Hello, world!" -o output.wav
```

> **Nota**: sin `--load-voice` ni `--ref-audio`, el motor habla con la voz `default` incorporada (la CLI imprime `Voz: default` en `stderr`). Sin `-l`, el idioma es el de la voz cargada o, si no tiene, español. Los valores por defecto de muestreo son `temperature = 0.325` y `seed = 4`.

### 2. Clonación de voz

El perfil guarda el timbre (x-vector) y, si das `-l`, el idioma. El audio de referencia debe ser **WAV PCM mono de 16 bits a 24 kHz**; con cualquier otro formato, el error da el comando `ffmpeg` que lo convierte (por ejemplo, `ffmpeg -i entrada.mp3 -ar 24000 -ac 1 ref_24k.wav`).

```bash
# 1. Extraer perfil .qvoice reutilizable y sintetizar el primer audio
./target/release/qwen_tts -d models/qwen3-tts-0.6b-base -j 4 \
  --ref-audio ref_24k.wav \
  --save-voice mi_voz.qvoice -l Spanish --text "Hola, esta es mi voz clonada." -o clon.wav

# 2. Reutilizar el perfil guardado (sin volver a codificar la referencia)
./target/release/qwen_tts -d models/qwen3-tts-0.6b-base -j 4 \
  --load-voice mi_voz.qvoice -l Spanish --text "Segunda frase con la misma voz." -o clon2.wav
```

### 3. Servidor HTTP (`--serve`)

Inicia el servidor en `127.0.0.1:8080` con 4 hilos:

```bash
./target/release/qwen_tts -d models/qwen3-tts-0.6b-base -j 4 --serve 8080
```

El servidor sin `--load-voice` usa la voz `default`; con `--load-voice mi_voz.qvoice` usa esa voz. El campo `voice` de las peticiones es opcional y, si viene, debe coincidir con el nombre de la voz cargada (si no, responde `400`); `GET /v1/voices` devuelve la voz cargada.

Envía peticiones síncronas (WAV completo) o en *streaming* (PCM `s16le` 24 kHz mono):

```bash
# WAV completo
curl -X POST http://127.0.0.1:8080/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello from Qwen TTS Engine!","voice":"default","language":"English"}' \
  --output speech.wav

# Streaming PCM s16le 24 kHz mono (Transfer-Encoding: chunked)
curl -X POST http://127.0.0.1:8080/v1/tts/stream \
  -H "Content-Type: application/json" \
  -d '{"text":"Streaming audio frame by frame.","language":"English"}' \
  --output stream.pcm
```

### 4. Biblioteca embebida en Rust (`qwen-engine`)

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
use qwen_core::Script;
use qwen_engine::{
    write_wav, AbortHandle, Delivery, Model06B, SessionScratch,
    SynthesisRequest, Synthesizer, Voice,
};
use qwen_kernels::ComputeRuntime;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = ComputeRuntime::new(Some(4));
    let model = Model06B::load(Path::new("models/qwen3-tts-0.6b-base"), &runtime, true)?; // `true`: sin trazas por stderr

    let voice = Voice::builtin_default(); // voz incorporada, idioma español
    let synth = Synthesizer { runtime: &runtime, model: &model, voice: &voice };
    let mut scratch = SessionScratch::new();
    let abort = AbortHandle::new();
    let req = SynthesisRequest::default(); // idioma por defecto: español

    let out = synth.synthesize(
        &mut scratch,
        &req,
        &abort,
        &Script::Plain("Hola desde Rust.".to_string()),
        Delivery::Buffer,
    )?;
    write_wav(Path::new("out.wav"), &out.samples.unwrap())?;
    Ok(())
}
```

El idioma de la síntesis es `SynthesisRequest::language_id`, ya resuelto por el llamador. La regla del motor (el pedido; si no hay, el de la voz; y si tampoco, español) la aplica `qwen_core::config::effective_language(pedido, voice.language_id)`. Para *streaming* (`Delivery::Stream` sobre un `AudioSink`, por ejemplo `EdgeRampSink`), cancelación cooperativa (`AbortHandle`), guiones con marcado (`Script::Spans`) y carga de voces clonadas, consulta [`docs/INTERFACE_CONTRACT.md`](docs/INTERFACE_CONTRACT.md).

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
$env:OPENBLAS_LIB_DIR = "C:/openblas/lib"

.\target\x86_64-pc-windows-gnu\release\qwen_tts.exe --self-test
cargo test --workspace --release --target x86_64-pc-windows-gnu
```

Consulta el detalle de suites, pruebas de integración con pesos reales y matriz de CI en [`docs/VERIFICATION_PLAN.md`](docs/VERIFICATION_PLAN.md).

---

## Documentación y arquitectura

La documentación sigue el modelo **Diátaxis** para separar guías prácticas, contratos de referencia, arquitectura y decisiones de diseño:

| Documento | Rol Diátaxis | Audiencia principal | Qué contiene |
|---|---|---|---|
| [`docs/INTERFACE_CONTRACT.md`](docs/INTERFACE_CONTRACT.md) | Referencia | Usuarios avanzados, integradores HTTP y Rust | Flags CLI (`qwen_tts`), endpoints HTTP (`--serve`), API Rust embebida, variables de entorno y líneas de log de referencia. |
| [`docs/SPEC_06B.md`](docs/SPEC_06B.md) | Referencia | Ingenieros del motor y mantenedores | Geometría fija `0.6B`, mapa de tensores `safetensors`, formato `.qvoice` v6, vocabularios, *prefill* y marcado en línea. |
| [`docs/NUMERIC_CONTRACT.md`](docs/NUMERIC_CONTRACT.md) | Referencia | Ingenieros de kernels y optimización | Cuantización `INT8`/`Q4_0`, `FTZ`/`DAZ`, reglas IEEE-754/`FMA`, determinismo de *streaming* y tolerancias. |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Explicación | Desarrolladores y contribuidores | Grafo de *crates* (DAG), política `unsafe`, tiempos de vida del estado, `ComputeRuntime`, camino único de síntesis y servidor. |
| [`docs/VERIFICATION_PLAN.md`](docs/VERIFICATION_PLAN.md) | Guía operativa | QA, contribuidores y mantenimiento de CI | Suites `cargo test`, verificación numérica de kernels, CI en CircleCI (`.circleci/config.yml`) y protocolo de medición de rendimiento. |
| [`docs/BACKLOG.md`](docs/BACKLOG.md) | Planificación | Mantenedores y contribuidores | Funcionalidades aplazadas: qué son, por qué esperan, cuándo retomarlas, lo decidido y lo pendiente. |
| [`docs/adr/001-only-06b.md`](docs/adr/001-only-06b.md) | ADR | Mantenedores | Especialización del motor para Qwen3-TTS `0.6B`. |
| [`docs/adr/002-blas-strategy.md`](docs/adr/002-blas-strategy.md) | ADR | Mantenedores | Estrategia BLAS (OpenBLAS propio y Accelerate). |
| [`docs/adr/003-concurrency-pool.md`](docs/adr/003-concurrency-pool.md) | ADR | Mantenedores | Pool de hilos `ComputePool` y control de sobresuscripción. |
| [`docs/adr/004-simd-baseline.md`](docs/adr/004-simd-baseline.md) | ADR | Mantenedores | Despacho SIMD en runtime sobre Rust estable (AVX2, NEON y escalar). |
| [`docs/adr/005-cpu-inference-optimizations.md`](docs/adr/005-cpu-inference-optimizations.md) | ADR | Mantenedores | Optimizaciones de la inferencia en CPU adoptadas y descartadas, con su efecto medido, y rendimiento por plataforma. |
| [`docs/adr/006-single-synthesis-path.md`](docs/adr/006-single-synthesis-path.md) | ADR | Mantenedores | Un único camino de síntesis (`Synthesizer`, `Script` × `Delivery`), una regla de idioma y un servidor de un solo trabajador. |
| [`docs/adr/007-natural-voice-stepwise-text-int8-talker.md`](docs/adr/007-natural-voice-stepwise-text-int8-talker.md) | ADR | Mantenedores | Texto por pasos, Talker `INT8` en el decodificado, clonación solo por x-vector y `.qvoice` v6, con la evidencia de cada decisión. |
| [`docs/adr/008-end-of-synthesis.md`](docs/adr/008-end-of-synthesis.md) | ADR | Mantenedores | Final de la síntesis: retención de 2 frames tras el primer EOS y punto final del texto, con la evidencia de cada decisión. |
| [`docs/adr/009-leading-silence.md`](docs/adr/009-leading-silence.md) | ADR | Mantenedores | Inicio de la voz: el motor no entrega el audio de la cadena de silencio inicial, con la evidencia de cada decisión. |

---

## Contribución y atribuciones

- **Issues y Pull Requests**: reporta errores o propón cambios mediante GitHub Issues y Pull Requests verificando previamente `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace --release` y la alineación con los contratos de `docs/`.
- **Atribución del proyecto original**: implementación independiente en Rust inspirada y portada a partir del motor en C `gabriele-mastrapasqua/qwen3-tts` (licencia MIT). Consulta [`NOTICE`](NOTICE) para el detalle completo.

---

## Licencia

Este proyecto está licenciado bajo **Mozilla Public License Version 2.0 (`MPL-2.0`)**.

Copyright (c) 2026 Cristián Rojas Arredondo y colaboradores de qwen-tts-engine.

- [`LICENSE`](LICENSE): texto completo de la MPL-2.0.
- [`NOTICE`](NOTICE): titular, dónde obtener el código fuente, atribución al motor original en C y avisos del proyecto.
- [`THIRD-PARTY-LICENSES.md`](THIRD-PARTY-LICENSES.md): avisos y licencias de los componentes de terceros que entran en el binario.

La voz `default` incorporada en el binario es un x-vector calculado de una grabación propia del autor, cedido bajo la licencia del proyecto (MPL-2.0). Los pesos del modelo Qwen3-TTS 0.6B Base no forman parte del repositorio ni del binario y tienen su propia licencia (Apache-2.0).
