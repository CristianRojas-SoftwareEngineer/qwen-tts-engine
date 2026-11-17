# Qwen TTS Engine (`qwen-tts-engine`)

> Motor de inferencia TTS en CPU escrito en Rust estable para **Qwen3-TTS 0.6B** (`CustomVoice` y `Base`), sin dependencias de Python, LibTorch ni Candle. Pensado para usuarios finales, integradores de servidor HTTP y desarrolladores Rust.

---

## Qué resuelve

- **Síntesis local en CPU**: ejecuta **Qwen3-TTS 0.6B** con cuantización `--quant-mixed` (`Q4_0` + `INT8`), kernels `AVX2 + FMA` en x86-64 (y *fallback* escalar en ARM64) y soporte de *streaming* progresivo.
- **Clonación de voz y expresividad**: soporta perfiles `.qvoice`/`.bin`, extracción `ECAPA-TDNN` + RVQ desde audio de referencia, activos `emovoice`, `--emo-ref`, `--roughness` y marcado inline.
- **Tres formas de integración**: binario CLI (`qwen_tts`), servidor HTTP compatible con OpenAI (`/v1/audio/speech`, `/v1/tts`, `/v1/tts/stream`) y librería Rust embebida (`qwen-engine`).

---

## Requisitos de entorno

| Plataforma | Toolchain y librerías | Notas |
|---|---|---|
| **Windows (`x86_64-pc-windows-gnu`)** | MSYS2 UCRT64 (`mingw-w64-ucrt-x86_64-gcc`, `mingw-w64-ucrt-x86_64-openblas`, `mingw-w64-ucrt-x86_64-binutils`) en `PATH` (`C:\msys64\ucrt64\bin`) | Si OpenBLAS está en otra ruta, define `OPENBLAS_LIB_DIR` (default `C:/msys64/ucrt64/lib`). |
| **Linux** | `libopenblas-dev` | Opcionalmente define `OPENBLAS_LIB_DIR` apuntando a `libopenblas.a` para enlace estático. |
| **macOS** | Xcode Command Line Tools | Usa el framework del sistema `Accelerate`. |
| **Rust puro (opcional)** | Feature `--features faer-blas` | Compila `PureRustBlas` (`faer`) y se selecciona en ejecución con `QWEN_BLAS=faer`. |

---

## Descarga de modelos `0.6B`

Descarga los pesos oficiales desde Hugging Face bajo `models/`:

- `models/qwen3-tts-0.6b` ← `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`
- `models/qwen3-tts-0.6b-base` ← `Qwen/Qwen3-TTS-12Hz-0.6B-Base` *(requerido para `--ref-audio` / clonación de voz)*

```bash
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --local-dir models/qwen3-tts-0.6b
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base --local-dir models/qwen3-tts-0.6b-base
```

---

## Compilación y primer audio por CLI

### Windows (MSYS2 UCRT64)

```powershell
$env:PATH = "C:\msys64\ucrt64\bin;$env:PATH"
$env:OMP_NUM_THREADS = "4"
Remove-Item Env:OPENBLAS_NUM_THREADS -ErrorAction SilentlyContinue

cargo build -p qwen-app --release --target x86_64-pc-windows-gnu
.\target\x86_64-pc-windows-gnu\release\qwen_tts.exe --caps
.\target\x86_64-pc-windows-gnu\release\qwen_tts.exe -d models\qwen3-tts-0.6b -j 4 --seed 42 -s ryan -l English --quant-mixed --text "Hello, world!" -o output.wav
```

### Linux / macOS

```bash
cargo build -p qwen-app --release
./target/release/qwen_tts --caps
./target/release/qwen_tts -d models/qwen3-tts-0.6b -j 4 --seed 42 -s ryan -l English --quant-mixed --text "Hello, world!" -o output.wav
```

> **Consejo**: usa `--quant-mixed` como perfil recomendado de producción para `0.6B` y `--stream` si necesitas escribir audio progresivamente conforme se genera.

---

## Clonación de voz básica (`0.6B-Base`)

La referencia debe ser un archivo WAV PCM de **24 kHz mono** (si parte de MP3/MP4/OGG/FLAC, conviértelo antes con `ffmpeg -i entrada.mp3 -ar 24000 -ac 1 ref_24k.wav`).

```bash
# 1. Crear un perfil .qvoice reutilizable y generar audio en el mismo paso
./target/release/qwen_tts -d models/qwen3-tts-0.6b-base -j 4 --quant-mixed \
  --ref-audio ref_24k.wav --ref-text "Texto exacto del audio de referencia." \
  --save-voice mi_voz.qvoice -l Spanish --text "Hola, esta es mi voz clonada." -o clon.wav

# 2. Reutilizar el perfil guardado (sin volver a codificar la referencia)
./target/release/qwen_tts -d models/qwen3-tts-0.6b-base -j 4 --quant-mixed \
  --load-voice mi_voz.qvoice -l Spanish --text "Segunda frase con la misma voz." -o clon2.wav
```

---

## Servidor HTTP compatible con OpenAI

Arranca el servidor con *continuous batching* (`--batch-size 4`) y un pool de 4 hilos:

```bash
./target/release/qwen_tts -d models/qwen3-tts-0.6b -j 4 --quant-mixed --serve 8080 --batch-size 4
```

Prueba el *endpoint* compatible con OpenAI o el *streaming* PCM crudo:

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

---

## Uso mínimo embebido en Rust

Agrega `qwen-engine`, `qwen-core` y `qwen-kernels` desde el workspace en tu `Cargo.toml` y sintetiza directamente en memoria:

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

Consulta la guía completa de la API embebida (incluidos `synthesize_to_sink` y cancelación con `AbortHandle`) en [`docs/INTERFACE_CONTRACT.md`](docs/INTERFACE_CONTRACT.md).

---

## Documentación y arquitectura

| Documento | Qué contiene |
|---|---|
| [`docs/INTERFACE_CONTRACT.md`](docs/INTERFACE_CONTRACT.md) | Referencia completa de CLI (`qwen_tts`), servidor HTTP (`--serve`), API Rust embebida, variables de entorno y logs operativos. |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Grafo de *crates*, política `unsafe`, cuatro tiempos de vida del estado, `ComputeRuntime`, pipeline y *schedulers*. |
| [`docs/SPEC_06B.md`](docs/SPEC_06B.md) | Geometría fija `0.6B`, mapa de tensores `safetensors`, formato `.qvoice` v3 / `.bin`, vocabularios, *prefill* y marcado inline. |
| [`docs/NUMERIC_CONTRACT.md`](docs/NUMERIC_CONTRACT.md) | Cuantización `INT8`/`Q4_0`/`Q2_0`, FTZ/DAZ, reglas IEEE-754/`FMA`, determinismo de *streaming* y tolerancias. |
| [`docs/VERIFICATION_PLAN.md`](docs/VERIFICATION_PLAN.md) | Suites de `cargo test`, verificación numérica de kernels y matriz de CI multiplataforma. |
| [`docs/adr/001-only-06b.md`](docs/adr/001-only-06b.md) | ADR-001: Especialización del motor para Qwen3-TTS `0.6B`. |
| [`docs/adr/002-blas-strategy.md`](docs/adr/002-blas-strategy.md) | ADR-002: Estrategia BLAS (`FfiBlas` por defecto y `faer-blas` opcional). |
| [`docs/adr/003-concurrency-pool.md`](docs/adr/003-concurrency-pool.md) | ADR-003: Pool de hilos `ComputePool` y control de sobresuscripción. |
| [`docs/adr/004-simd-baseline.md`](docs/adr/004-simd-baseline.md) | ADR-004: Despacho SIMD en runtime sobre Rust estable. |

---

## Licencia

Este proyecto está licenciado bajo **Mozilla Public License Version 2.0 (`MPL-2.0`)**. Consulta [`LICENSE`](LICENSE) para el texto completo de la licencia y [`NOTICE`](NOTICE) para la atribución upstream.
