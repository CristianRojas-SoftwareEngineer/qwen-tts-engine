# Contrato de Interfaz Pública (`INTERFACE_CONTRACT.md`)

> Referencia normativa de las tres interfaces públicas de **Qwen TTS Engine** (`qwen-tts-engine`): binario CLI (`qwen_tts`), servidor HTTP (`--serve`) y API Rust embebida (`qwen-engine`, `qwen-core`, `qwen-kernels`), además de variables de entorno y logs operativos.

> **Alcance**: Exclusivo para **Qwen3-TTS 0.6B Base**. Las opciones de `1.7B` (`--voice-design`, `-I`/`--instruct`, `--expr`, `--expr-weight`, `--ml-steer`) son rechazadas explícitamente.

---

## 1. Interfaz CLI (`qwen_tts`)

### 1.1 Modos auxiliares y subcomando posicional

| Comando / Flag | Comportamiento | Código de salida |
|---|---|---:|
| `-h`, `--help` | Imprime la ayuda de uso por `stderr` y termina. | `0` |
| `--caps` | Inicializa el runtime con `-j`, imprime capacidades SIMD/hilos/BLAS a `stdout` y termina. | `0` |
| `--self-test` | Ejecuta la batería numérica de kernels hacia `stdout`. | `0` si pasa, `1` si falla |
| `--matmat-bench` | Mide `matmat` vs `B × matvec` hacia `stdout` con el `-j` activo. | `0` |
| `--list-voices <dir>` | Lista los `.qvoice` de `<dir>` (o un archivo `.qvoice` puntual): por voz, archivo, nombre, idioma (`-` si no tiene), ICL (fotogramas y segundos, o `no`) y tamaño. Un archivo que el lector rechaza (por ejemplo, una versión distinta de la 4) aparece con `unsupported: <mensaje>`. La voz `default` no se lista: no es un archivo. | `0` / `1` |
| `--delete-voice <path>` | Valida la extensión `.qvoice` y la cabecera `QVCE` antes de eliminar el archivo. | `0` / `1` |
| `decode-dump <codes.txt> <model_dir> <out.wav>` | Subcomando posicional (sin otras flags): carga `<model_dir>` y decodifica códigos (múltiplo de 16) a WAV 24 kHz s16le. | `0` / `1` |

### 1.2 Flags de modelo, voz, muestreo y cuantización

| Categoría | Flag(s) | Default | Contrato y validación |
|---|---|---:|---|
| **Modelo y texto** | `-d, --model-dir <path>` | *(obligatorio con modelo)* | Directorio del modelo Qwen3-TTS `0.6B` Base. Rechaza `1.7B`, `VoiceDesign` y cualquier checkpoint que no sea Base (`exit 1`, con un mensaje que indica que el motor solo carga el modelo Base). |
| | `-t, --text <string>` | `None` | Obligatorio salvo `--serve`, `--compose`, `--save-voice` sin texto (con `--ref-audio`). |
| | `-o, --output <path>` | `output.wav` | Ruta de salida WAV (24 kHz, PCM s16le mono). Lleva una rampa de entrada de 5 ms y otra de salida de 40 ms con y sin `--stream`. El PCM es idéntico en los dos modos (`NUMERIC_CONTRACT.md` §4). |
| **Voz e idioma** | `-l, --language <name>` | idioma de la voz cargada o, si no tiene, `English` (`2050`) | Acepta 10 idiomas *case-insensitive*: `english`, `german`, `spanish`, `chinese`, `japanese`, `french`, `korean`, `russian`, `italian`, `portuguese`. Con `--load-voice`, si se omite usa el idioma guardado en el `.qvoice` si es distinto de `0`. Desconocido → `exit 1`. |
| **Muestreo** | `-T, --temperature <f>` | `0.325` | Rango `[0.0, 2.0]`; `0.0` = greedy. Si hay sustitución paralingüística inline sin `-T`, usa la `T` de la etiqueta (`1.1`, o `1.0` en `[scoff]`). |
| | `-k, --top-k <n>` | `50` | Límite de candidatos *top-k* en `[0, 3072]`. |
| | `-p, --top-p <f>` | `1.0` | Umbral acumulado *nucleus sampling* en `[0.0, 1.0]`. |
| | `-r, --rep-penalty <f>` | `1.05` | Penalización de repetición sobre tokens previos en `[0.5, 2.0]`. |
| | `-m, --max-tokens <n>` | `8192` | Tope de fotogramas generados (`>= 1`). |
| | `--max-duration <secs>` | `None` | Si es `> 0`, sobrescribe `--max-tokens` con `max(1, floor(secs * 12.5))`. |
| | `--seed <n>` | `4` (`< 0` reloj) | Entero `0..=u32::MAX` fija semilla; omitido usa `DEFAULT_SEED = 4` (salvo sustitución paralingüística inline); negativo usa reloj del sistema; `> u32::MAX` → `exit 1`. |
| | `--greedy-warmup <n>` | `0` | Primeros `n` fotogramas con `T=0.0, top_k=1`. |
| **Cuantización** | `--int8` | off | Talker + Code Predictor en `INT8`. |
| | `--int4` | off | Talker + Code Predictor en `Q4_0`. |
| | `--quant-mixed` | off | **Perfil recomendado 0.6B**: Talker en `Q4_0`; Code Predictor con atención y `lm_head` en `INT8` y FFN en `Q4_0` (`[INT8+Q4_0-FFN]`). |

### 1.3 Streaming, audio y post-procesado

| Flag(s) | Default | Contrato y validación |
|---|---|---|
| `--stream` | off | Escribe el WAV progresivamente conforme se decodifican chunks. Retiene los últimos 40 ms para aplicar la rampa de salida al cerrar. Incompatible con `--rate != 1.0` (`exit 1`). |
| `--stdout` | off | Implica `--stream` y `--silent`; emite PCM crudo s16le 24 kHz mono a `stdout`. |
| `--stream-chunk <n>` | `10` | Fotogramas máximos por chunk (`clamp(2, 250)`). Rampa: `2` en el primer fragmento y `min(n, 4 + 2 * (k / 8))` para `k = 0, 1, 2, ...` en los siguientes. |
| `--volume <f>` | `1.0` | Ganancia lineal en `[0.0, 10.0]` aplicada sobre las muestras PCM con *clamp* a `[-1.0, 1.0]`. |
| `--rate <f>` | `1.0` | Cambio de velocidad WSOLA en `[0.25, 4.0]` preservando pitch (incompatible con `--stream`). |
| `--onset-fade <ms>` | `0` | Rampa de entrada sobre el ataque real detectado (solo modo buffer / `--compose`). |
| `--tail-trim` | off | Recorta cola metálica degenerada detectada al final (solo modo buffer / `--compose`). |
| `--seed-audition <N>` | `0` | Evalúa hasta `N` (`1..64`) semillas consecutivas y elige la mejor (solo modo no-streaming). |
| `--audition-keep` | off | Conserva cada toma de `--seed-audition` como `<stem>.seed<S>.wav`. |

### 1.4 Voces, clonación y marcado `0.6B`

| Categoría | Flag(s) | Default | Contrato y validación |
|---|---|---|---|
| **Clonación** | `--ref-audio <path>` | `None` | Valida cabecera `RIFF`/WAV 24 kHz mono antes de cargar el modelo; otros formatos fallan con comando `ffmpeg`. |
| | `--ref-text <string>` | `None` | Transcripción de la referencia para activar modo ICL (o sobrescribir `ref_text` de `.qvoice`). Sin ella, la voz guarda solo el timbre (x-vector). |
| | `--xvector-only` | off | Usa únicamente el *x-vector* (`1024 × f32`), omitiendo los códigos ICL. |
| | `--save-voice <path>` | `None` | Guarda `.qvoice` v4 (la extensión debe ser `.qvoice`; otra da `voice files must be .qvoice (got <ruta>)`). Con idioma `-l` lo guarda; sin `-l` guarda «sin idioma». Sin `--text` ni `--serve`, crea el perfil y sale (`0`). |
| | `--load-voice <path>` | voz `default` incorporada | Carga `.qvoice` v4. Otra extensión o una versión distinta de la 4 → `exit 1`. Sin este flag ni `--ref-audio`, se usa la voz `default`. |
| | `--icl-frames <n>` | `0` *(todos)* | Tope máximo de fotogramas de referencia en modo ICL. |
| | `--voice-name <name>` | `None` | Nombre guardado en el `.qvoice`; sin él, la voz clonada al vuelo se llama como el WAV (sin extensión). Al cargar, el nombre es el guardado en el archivo o, si está vacío, el nombre del archivo. |
| | `--max-ref-duration <s>` | `30.0` | Duración máxima en segundos leída de `--ref-audio` (`0` = audio completo). |
| **Marcado y estilo** | `--roughness <0..1>` | `0.0` | Mezcla pesos `Q2_0` en `down_proj` del Code Predictor (`[0.0, 1.0]`). |
| | `--compose <spec>` | `None` | Síntesis multi-span no streaming con pausas `[pause:<s>]`, `[break:<s>]`, `[0.5]` o `\|`, y tags paralingüísticos inline (`[laugh]`, `[sigh]`, `[yawn]`, `[wow]`, `[giggle]`, `[scoff]`). |
| | `--compose-pause <s>` | `0.12` | Pausa por defecto entre spans hablados en `--compose`. |
| | `--no-compose` | off | Desactiva la sustitución de `[tags]` y el auto-compose en `--text`. |

### 1.5 Servidor HTTP y sistema

| Flag(s) | Default | Contrato y validación |
|---|---|---|
| `--serve <port>` | `None` | Inicia el servidor HTTP en `<port>` (`> 0`). |
| `--host <ipv4>` | `127.0.0.1` | Dirección `Ipv4Addr` validada antes de cargar el modelo. Inválida → `exit 1`. |
| `--watch-stdin` | off | Vigila `stdin` en un hilo dedicado; al recibir EOF termina el proceso (`exit 0`). |
| `-j, --threads <n>` | `0` *(auto)* | Hilos de cómputo (`0` = `min(ncpus, 4)`). |
| `-S, --silent` | off | Suprime logs en `stderr`; no afecta el audio en `--stdout`. |
| `-D, --debug` | off | Imprime trazas internas de prefill y generación en `stderr`. |

---

## 2. Interfaz HTTP (`--serve`)

### 2.1 Seguridad de red, límites y CORS

- **Bind por defecto**: `127.0.0.1:<port>` (solo expone otra interfaz con `--host`).
- **Límite de cuerpo**: máximo `1 MiB` (`1_048_576` bytes) por petición HTTP.
- **CORS**: todas las respuestas incluyen `Access-Control-Allow-Origin: *`; `OPTIONS` responde `204 No Content`.
- **Formato de error**: `{"error":"<causa>"}` con `Content-Type: application/json`.
- **Calentamiento al arrancar**: el servidor sintetiza y descarta 12 fotogramas de «Hello.» antes de aceptar conexiones (~1 s), para que la primera petición no pague el arranque en frío. El commit en reposo queda en el nivel que antes alcanzaba tras la primera petición; el audio de las peticiones no cambia. Log: `[HTTP] Calentamiento: <ms> ms`.

### 2.2 Endpoints y esquema JSON

| Método y Ruta | Entrada | Respuesta exitosa | Reglas específicas |
|---|---|---|---|
| `GET /v1/health` | — | `200` `{"status":"ok"}` | Comprobación de disponibilidad. |
| `GET /v1/voices` | — | `200` `{"voices":[{"name":"<nombre>","language":"<nombre o null>","icl":<bool>}]}` | Devuelve la voz cargada. |
| `POST /v1/tts` | JSON TTS | `200` `audio/wav` (24 kHz s16le mono) | WAV completo con rampas de 5 ms (entrada) y 40 ms (salida); mismo PCM que `/v1/tts/stream` y que la CLI. Usa `DEFAULT_CHUNK_FRAMES = 10`; `max_tokens = 8192`. |
| `POST /v1/audio/speech` | JSON OpenAI (`input`, `voice`, ...) | `200` `audio/wav` | Alias completo de `POST /v1/tts` (`input` → `text`). |
| `POST /v1/tts/stream` | JSON TTS | `200` `audio/pcm` (`Transfer-Encoding: chunked`) | Cabeceras: `X-Sample-Rate: 24000`, `X-Sample-Format: s16le`, `X-Channels: 1`. Rechaza `rate != 1.0` (`400`); `max_tokens = 8192`. |

**Campos JSON aceptados en `POST`**:

| Campo | Tipo | Default | Notas |
|---|---|---|---|
| `text` o `input` | `string` | *(obligatorio)* | No vacío ni solo espacios/escapes `\n`, `\t`, `\r`; `<= 8192` bytes. |
| `voice` | `string` | *(opcional)* | Si viene, debe coincidir (sin distinguir mayúsculas) con el nombre de la voz cargada; si no, `400` con `voice '<x>' is not loaded (loaded: '<nombre>')`. |
| `language` | `string` | idioma de la voz cargada (`None` si no tiene) | Debe existir en `lookup_language`. Si se omite y la voz no tiene idioma, no fuerza idioma (`CODEC_NO_THINK`). |
| `temperature` | `number` | `0.325` | Rango válido `[0.0, 2.0]`. |
| `top_k` | `integer` | `50` | Rango válido `[0, 3072]`. |
| `top_p` | `number` | `1.0` | Rango válido `[0.0, 1.0]`. |
| `rep_penalty` | `number` | `1.05` | Rango válido `[0.5, 2.0]`. |
| `seed` | `integer` | `4` (`< 0` reloj) | Entero `0..=u32::MAX` fija semilla; omitido usa `DEFAULT_SEED = 4` (salvo sustitución paralingüística inline); negativo usa reloj; `> u32::MAX` devuelve `400`. |
| `volume` | `number` | `1.0` | Ganancia PCM lineal; rango válido `[0.0, 10.0]`. |
| `rate` | `number` | `1.0` | WSOLA pitch-preserving en `/v1/tts` (`[0.25, 4.0]`); prohibido `!= 1.0` en `/v1/tts/stream`. |
| `chunk_frames` | `integer` | `10` | Solo en `/v1/tts/stream`; se acota con `clamp(2, 250)`. |
| `voice_design` | `bool \| string \| number` | `false` | `true`, `"true"`, `"1"` o `1` devuelven `400` (`0.6B` no soporta `VoiceDesign`). |

### 2.3 Validación HTTP `400` y valores acotados

Toda petición se valida en `ValidatedTtsRequest::parse` antes de planificar o encolar síntesis:

- Cuerpo JSON inválido, no objeto, `Content-Length` inválido o tamaño mayor que `1 MiB` → `400`.
- `text`/`input` ausente, vacío, solo whitespace (o `\n`/`\t`/`\r` escapados) o `> 8192` bytes → `400`.
- `language` desconocido, o `voice` distinto de la voz cargada → `400`.
- Los campos desconocidos se ignoran (no dan `400`).
- `temperature` fuera de `[0.0, 2.0]`, `top_k` fuera de `[0, 3072]`, `top_p` fuera de `[0.0, 1.0]`, `rep_penalty` fuera de `[0.5, 2.0]`, `volume` fuera de `[0.0, 10.0]`, `rate` fuera de `[0.25, 4.0]` o `seed > u32::MAX` → `400`.
- `rate != 1.0` en `/v1/tts/stream` → `400`.
- `voice_design` activo → `400`.
- `chunk_frames` en `/v1/tts/stream` se acota silenciosamente a `[2, 250]`.

### 2.4 Streaming *chunked*, cancelación y cierre

1. **Rampa de chunks (`/v1/tts/stream`)**: primer fragmento de `2` fotogramas (`160 ms`); los siguientes (`k = 0, 1, 2, ...`) siguen `min(chunk_frames, 4 + 2 * (k / 8))`. El decoder usa el mismo calendario también en `/v1/tts` y en la CLI.
   - **Rampa de salida de 40 ms**: el servidor retiene los últimos 40 ms de audio y los envía con el fragmento siguiente o al cerrar. Por eso el primer fragmento entrega 120 ms en lugar de 160 ms.
2. **Señal de cierre vs fallo**: el terminador `0\r\n\r\n` solo se envía si la generación termina sin error y con `samples > 0`. Si falla o se corta la conexión, el servidor cierra el socket **sin** enviar `0\r\n\r\n` para que el cliente detecte transferencia truncada.
3. **Cancelación por socket**: el motor sondea desconexión del cliente antes de cada fotograma. El cliente **no** debe ejecutar `shutdown(SHUT_WR)` tras enviar el request HTTP, o se interpretará como desconexión.

---

## 3. API Rust embebida (`qwen-engine`, `qwen-core`, `qwen-kernels`)

La librería está diseñada para compartir `Arc<ComputeRuntime>`, `Arc<Model06B>` y `Arc<VoiceBindings>` entre hilos mientras cada petición usa un `SessionScratch` mutable exclusivo.

### 3.1 Inicialización de `ComputeRuntime` y carga de `Model06B`

```rust
use std::path::Path;
use std::sync::Arc;
use qwen_engine::{Model06B, ModelLoadOptions};
use qwen_kernels::ComputeRuntime;

let runtime = Arc::new(ComputeRuntime::new(Some(4))); // `None` -> min(ncpus, 4)
let load_opts = ModelLoadOptions {
    quant_mixed: true, // Talker Q4_0 + CP [INT8+Q4_0-FFN]
    silent: false,
    ..Default::default()
};
let model = Arc::new(Model06B::load(
    Path::new("models/qwen3-tts-0.6b-base"),
    &runtime,
    load_opts,
)?);
```

- `Model06B::load` valida `config.json` para `0.6B`, abre `safetensors` vía `MmapStore` y cuantiza matrices según `ModelLoadOptions` (o `QWEN_TALKER_PREC` / `QWEN_CP_PREC`).

### 3.2 Preparación de `VoiceBindings` y `SessionScratch`

- **Voz `default` incorporada** (la voz por defecto del motor):
  ```rust
  use qwen_engine::{SessionScratch, VoiceBindings};

  let voice = Arc::new(VoiceBindings::default_voice()); // nombre: "default"
  let mut scratch = SessionScratch::new(); // reutilizable entre peticiones secuenciales
  ```
- **Identidad de voz**: `VoiceIdentity::{XVector(Vec<f32>), Icl { speaker_embedding, ref_text, ref_codes }}`. `VoiceBindings` lleva `identity`, `name` (`String`) e `icl_frames_cap`.
- **Voz clonada desde WAV o archivo `.qvoice`**:
  - `build_voice_from_ref_audio(&runtime, &model, ref_wav_path, save_path_opt, &SaveVoiceOptions::default())` extrae el x-vector (y el ICL si hay `ref_text`); `save_path_opt` debe terminar en `.qvoice` (v4).
  - `load_voice_from_file(voice_path, ref_text_override, &VoiceLoadOptions::default())` carga un `.qvoice` v4 sin tocar el modelo y devuelve `(VoiceBindings, Option<i32>)` con el id de idioma de la voz.

### 3.3 Síntesis en memoria (`synthesize_to_samples`) y streaming (`synthesize_to_sink`)

```rust
use std::path::Path;
use qwen_engine::{
    synthesize_to_samples, synthesize_to_sink, write_wav, RequestConfig,
    RequestExecution, WavStreamSink,
};

// A) Síntesis completa a Vec<f32> (24 kHz mono)
let exec = RequestExecution::new(RequestConfig {
    language_id: Some(2050), // English
    temperature: 0.325,
    seed: 4,
    ..Default::default()
});
let (samples, timing) = synthesize_to_samples(
    &runtime, &model, &voice, &mut scratch, &exec, "Hello from Rust!",
)?;
write_wav(Path::new("buffer.wav"), &samples, 24_000)?;

// B) Streaming progresivo hacia cualquier impl AudioSink + Send
let mut stream_cfg = exec.config.clone();
stream_cfg.stream = true;
stream_cfg.stream_chunk_frames = 10;
let stream_exec = RequestExecution::new(stream_cfg);

let mut sink = WavStreamSink::create(Path::new("stream.wav"), 24_000, 1.0)?;
synthesize_to_sink(
    &runtime, &model, &voice, &mut scratch, &stream_exec, "Streaming audio!", &mut sink,
)?;
```

Sinks incluidos en `qwen_engine`: `MemorySink`, `WavStreamSink`, `StdoutRawSink`, o cualquier tipo propio que implemente `AudioSink` (`write_samples(&mut self, &[f32])` y `finish(&mut self)`).

### 3.4 Cancelación cooperativa (`AbortHandle`) y síntesis multi-span

- **Cancelación manual o por sondeo**:
  ```rust
  use qwen_engine::{AbortHandle, EngineError, RequestConfig, RequestExecution};

  let abort = AbortHandle::new(); // o AbortHandle::with_poll(|| false)
  let exec = RequestExecution { config: RequestConfig::default(), abort: abort.clone() };

  // Desde otro hilo:
  abort.cancel();
  // `synthesize_to_samples` / `synthesize_to_sink` retornan `Err(EngineError::Aborted)`.
  ```
- **Marcado y multi-span (`qwen_core::MarkupPlanner` + `qwen_engine::compose`)**: usa `prepare_markup_06b(...)` para detectar `[tags]`, seguido de `MarkupPlanner::parse_spans(&prep.text)` y `render_spans_buffer(...)` o `render_spans_stream(...)` (que delega la ganancia base `exec.config.volume` al `AudioSink` del llamador, igual que `synthesize_to_sink`).

---

## 4. Variables de entorno, ciclo de vida y logs operativos

### 4.1 Variables de entorno soportadas

| Variable | Scope | Default | Efecto |
|---|---|---|---|
| `OMP_NUM_THREADS` | OpenBLAS / Windows | `-j` *(lo fija el binario)* | Opcional. Si se define, manda: OpenBLAS reserva un búfer de 128 MiB de commit por hilo indicado, y los hilos de BLAS no superan ese valor aunque `-j` sea mayor. |
| `OPENBLAS_NUM_THREADS` | OpenBLAS | *(no definir)* | Si está definida, `FfiBlas::set_threads` no reconfigura hilos entre *prefill* y generación. |
| `OPENBLAS_LIB_DIR` | Build (`qwen-kernels`) | `C:/msys64/ucrt64/lib` *(Win)* | Ruta de búsqueda para `openblas` / `libopenblas.a`. En Linux, si contiene `libopenblas.a`, OpenBLAS se enlaza estático (`scripts/build-openblas.sh` lo compila), y `libgfortran.a`/`libquadmath.a` solo si están en el mismo directorio. Sin la variable, la `libopenblas` dinámica del sistema. |
| `WINDRES` | Build (`qwen-app`) | auto | Compilador de recursos COFF para incrustar el manifiesto UTF-8 en Windows GNU. |
| `QWEN_BLAS` | Runtime | `ffi` | Con `--features faer-blas`, `QWEN_BLAS=faer` selecciona `PureRustBlas`. |
| `QWEN_KERNELS` | Runtime | `auto` | `auto`: los kernels más rápidos que soporte la CPU (AVX2 + FMA en x86-64; en aarch64, NEON en todos los kernels, con `SDOT` en Q4_0 e INT8 si la CPU tiene `dotprod`; `--caps` muestra la ruta en la línea `kernels:`). `scalar`: fuerza la ruta escalar, como referencia para comparar kernels en la misma máquina; `--caps` lo indica con `kernels: scalar (forced by QWEN_KERNELS=scalar)`. Un valor no válido se avisa por stderr y se trata como `auto`. Cambiar de ruta cambia el redondeo y, por tanto, el audio. |
| `QWEN_POOL_SPIN` | Runtime | `4096`, o `0` si los hilos del productor ocupan todos los núcleos físicos | Iteraciones de *spin-wait* en `ComputePool` (`0` = espera directa en `Condvar`). Sin la variable, la CLI y el servidor de una petición usan `0` cuando `-j` llega a los núcleos físicos (con 4 núcleos y `-j 4`) y `4096` si no; el servidor con varios trabajadores aplica la misma regla a sus hilos (`spin_budget_for`). Con la variable, manda su valor. |
| `QWEN_SD_THREADS` | Runtime | `-j` | Número de hilos asignados a `sd_pool` (`SpeechDecoder`). |
| `QWEN_BLAS_GEN_THREADS` | Runtime | `max(1, nt - 1)` | Hilos BLAS durante la fase de generación paralela con `SpeechDecoder`. |
| `QWEN_TALKER_PREC` | Carga `Model06B` | según flags | Fuerza precisión del Talker: `bf16`, `int8` o `int4`. |
| `QWEN_CP_PREC` | Carga `Model06B` | según flags | Fuerza precisión del Code Predictor: `bf16`, `int8`, `int4` o `mixed`. |
| `QWEN_SPK_SCALE` | Prefill | `1.0` | Factor multiplicador sobre la norma L2 del *speaker embedding* clonado. |
| `QWEN_ICL_FRAMES` | Prefill ICL | `voice.icl_frames_cap` | Sobrescribe el límite de fotogramas de referencia ICL. |
| `QWEN_ICL_TRIM_FRAMES` | Pipeline ICL | `2` | Fotogramas recortados al inicio del audio generado en modo ICL (`2 × 1920` muestras). |
| `QWEN_NO_REF_TRIM` | Audio ref | off | Si está definida, desactiva `trim_trailing_silence` sobre `--ref-audio`. |
| `QWEN_DUMP_CODES` | Pipeline | `None` | Ruta de archivo donde volcar los 16 codebooks por fotograma (`decode-dump`). |
| `QWEN_BATCH_B` | `--matmat-bench` | `8` | Tamaño de lote `B` en microbenchmarks (`1..=64`; valores fuera de rango usan `8`). |

### 4.2 Afinidad SMT en Windows y gestión de OpenBLAS

- **Afinidad Win32 (`affinity.rs`)**: al arrancar en Windows con 1 grupo de procesadores, `qwen_tts` fija `SetProcessAffinityMask` a **1 hilo lógico por núcleo físico** para evitar contención SMT del planificador.
- **OpenBLAS + OpenMP**: en Windows con `x86_64-pc-windows-gnu`, OpenBLAS está enlazado estático y se inicializa antes de `main`, reservando 128 MiB de commit por hilo. Si no hay `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` ni `GOTO_NUM_THREADS`, el binario fija `OMP_NUM_THREADS` al `-j` efectivo antes de esa inicialización (`.CRT$XCT`); no hace falta definir nada. `set_threads` aplica los hilos en cada hilo que llama a `sgemm` y nunca pasa de los iniciales (OpenBLAS 0.3.34 se cuelga si se le piden más). Mantén **sin definir** `OPENBLAS_NUM_THREADS` para que el motor pueda ajustar los hilos entre *prefill* (`nt`) y generación (`gen_blas`).

### 4.3 Líneas de log contractuales

Los tests y scripts de operación esperan las siguientes líneas exactas en `stderr`:

- **Carga de Code Predictor** (`model.rs`):
  `  Code Predictor: 5 layers loaded, q_dim=2048 kv_dim=1024 [BF16|INT8|Q4_0|INT8+Q4_0-FFN]`
- **Voz cargada** (`main.rs`, salvo con `-S`): `Voice: <nombre>` (`default`, el nombre guardado en el `.qvoice`, el nombre del archivo o el del WAV clonado).
- **Semilla resuelta** (`main.rs`):
  `seed: <u32>` *(o `seed: <u32> (auto/time-based)`)*
- **Generación en pipeline** (`pipeline.rs`):
  `  Generated <N> frames (<S>s audio)`
- **Servidor HTTP al arrancar** (`http.rs`): `[HTTP] Calentamiento: <ms> ms` (o `[HTTP] Calentamiento fallido (sin efecto en las peticiones): <causa>`).
- **Servidor HTTP en petición exitosa** (`http.rs` / `scheduler.rs`):
  ```text
  [HTTP] Sent <bytes> bytes WAV (<dur>s audio) in <sec>s (RTF <rtf>)
  [HTTP] Streamed <samples> samples (<dur>s audio) in <sec>s (RTF <rtf>)
  [HTTP] Desglose: prefill <ms> ms, Talker <ms> ms (<ms/f> ms/f), CP <ms> ms (<ms/f> ms/f), decodificador <ms> ms, drenaje <ms> ms, <N> fotogramas en <M> fragmentos, cola máxima <K> fotogramas
  ```
- **Cancelación HTTP** (`http.rs` / `scheduler.rs`):
  `[HTTP] TTS cancelado: el cliente se desconectó tras <sec> s`
  `[HTTP] TTS stream cancelado: el cliente se desconectó tras <sec> s`
