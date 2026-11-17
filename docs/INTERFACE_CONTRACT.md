# Contrato de Interfaz Pública (`INTERFACE_CONTRACT.md`)

> Referencia normativa de las tres interfaces públicas de **Qwen TTS Engine** (`qwen-tts-engine`): binario CLI (`qwen_tts`), servidor HTTP (`--serve`) y API Rust embebida (`qwen-engine`, `qwen-core`, `qwen-kernels`), además de variables de entorno y logs operativos.

> **Alcance**: Exclusivo para **Qwen3-TTS 0.6B** (`CustomVoice` y `Base`). Las opciones de `1.7B` (`--voice-design`, `-I`/`--instruct`, `--expr`, `--expr-weight`, `--ml-steer`) son rechazadas explícitamente.

---

## 1. Interfaz CLI (`qwen_tts`)

### 1.1 Modos auxiliares y subcomando posicional

| Comando / Flag | Comportamiento | Código de salida |
|---|---|---:|
| `-h`, `--help` | Imprime la ayuda de uso por `stderr` y termina. | `0` |
| `--caps` | Inicializa el runtime con `-j`, imprime capacidades SIMD/hilos/BLAS a `stdout` y termina. | `0` |
| `--self-test` | Ejecuta la batería numérica de kernels hacia `stdout`. | `0` si pasa, `1` si falla |
| `--matmat-bench` | Mide `matmat` vs `B × matvec` hacia `stdout` con el `-j` activo. | `0` |
| `--list-voices <dir>` | Lista los perfiles `.qvoice` válidos en `<dir>` (o un archivo `.qvoice` puntual). | `0` / `1` |
| `--delete-voice <path>` | Valida la extensión `.qvoice` y la cabecera `QVCE` antes de eliminar el archivo. | `0` / `1` |
| `decode-dump <codes.txt> <model_dir> <out.wav>` | Subcomando posicional (sin otras flags): carga `<model_dir>` y decodifica códigos (múltiplo de 16) a WAV 24 kHz s16le. | `0` / `1` |

### 1.2 Flags de modelo, voz, muestreo y cuantización

| Categoría | Flag(s) | Default | Contrato y validación |
|---|---|---:|---|
| **Modelo y texto** | `-d, --model-dir <path>` | *(obligatorio con modelo)* | Directorio del modelo `0.6B` (`CustomVoice` o `Base`). Rechaza `1.7B` y `VoiceDesign` (`exit 1`). |
| | `-t, --text <string>` | `None` | Obligatorio salvo `--serve`, `--compose`, `--save-voice` sin texto (con `--ref-audio`), `--batch-test` o `--batch-bench`. |
| | `-o, --output <path>` | `output.wav` | Ruta de salida WAV (24 kHz, PCM s16le mono). |
| **Voz e idioma** | `-s, --speaker <name>` | `ryan` (`3061`) | Presets: `ryan`, `vivian`, `serena`, `aiden`, `eric`, `dylan`, `uncle_fu`, `ono_anna`, `sohee`. Nombre desconocido → `exit 1`. |
| | `-l, --language <name>` | `English` (`2050`) *(o `META` en `--load-voice`)* | Acepta 12 idiomas *case-insensitive*: `english`, `german`, `spanish`, `chinese`, `japanese`, `french`, `sichuan_dialect`, `korean`, `russian`, `italian`, `portuguese`, `beijing_dialect`. Con `--load-voice`, si se omite usa `META.lang_id` si es `> 0`. Desconocido → `exit 1`. |
| **Muestreo** | `-T, --temperature <f>` | `0.9` *(CLI)* / `0.5` *(HTTP)* | Rango `[0.0, 2.0]`; `0.0` = greedy. Si hay sustitución paralingüística inline sin `-T`, usa la `T` de la etiqueta (`1.1`, o `1.0` en `[scoff]`). |
| | `-k, --top-k <n>` | `50` | Límite de candidatos *top-k* en `[0, 3072]`. |
| | `-p, --top-p <f>` | `1.0` | Umbral acumulado *nucleus sampling* en `[0.0, 1.0]`. |
| | `-r, --rep-penalty <f>` | `1.05` | Penalización de repetición sobre tokens previos en `[0.5, 2.0]`. |
| | `-m, --max-tokens <n>` | `8192` | Tope de fotogramas generados (`>= 1`). |
| | `--max-duration <secs>` | `None` | Si es `> 0`, sobrescribe `--max-tokens` con `max(1, floor(secs * 12.5))`. |
| | `--seed <n>` | tiempo (`< 0`) | Entero `0..=u32::MAX` fija semilla; negativo usa reloj del sistema; `> u32::MAX` → `exit 1`. |
| | `--greedy-warmup <n>` | `0` | Primeros `n` fotogramas con `T=0.0, top_k=1`. |
| **Cuantización** | `--int8` | off | Talker + Code Predictor en `INT8`. |
| | `--int4` | off | Talker + Code Predictor en `Q4_0`. |
| | `--quant-mixed` | off | **Perfil recomendado 0.6B**: Talker en `Q4_0`; Code Predictor con atención y `lm_head` en `INT8` y FFN en `Q4_0` (`[INT8+Q4_0-FFN]`). |

### 1.3 Streaming, audio y post-procesado

| Flag(s) | Default | Contrato y validación |
|---|---|---|
| `--stream` | off | Escribe el WAV progresivamente conforme se decodifican chunks. Incompatible con `--rate != 1.0` (`exit 1`). |
| `--stdout` | off | Implica `--stream` y `--silent`; emite PCM crudo s16le 24 kHz mono a `stdout`. |
| `--stream-chunk <n>` | `10` | Fotogramas máximos por chunk (`clamp(2, 250)`). Rampa: `2` en el primer fragmento y `min(n, 4 + 2 * (k / 8))` para `k = 0, 1, 2, ...` en los siguientes. |
| `--volume <f>` | `1.0` | Ganancia lineal en `[0.0, 10.0]` aplicada sobre las muestras PCM con *clamp* a `[-1.0, 1.0]`. |
| `--rate <f>` | `1.0` | Cambio de velocidad WSOLA en `[0.25, 4.0]` preservando pitch (incompatible con `--stream`). |
| `--onset-fade <ms>` | `0` | Rampa de entrada sobre el ataque real detectado (solo modo buffer no-batch / `--compose`). |
| `--tail-trim` | off | Recorta cola metálica degenerada detectada al final (solo modo buffer no-batch / `--compose`). |
| `--seed-audition <N>` | `0` | Evalúa hasta `N` (`1..64`) semillas consecutivas y elige la mejor (solo modo no-streaming). |
| `--audition-keep` | off | Conserva cada toma de `--seed-audition` como `<stem>.seed<S>.wav`. |

### 1.4 Clonación de voz y expresividad `0.6B`

| Categoría | Flag(s) | Default | Contrato y validación |
|---|---|---|---|
| **Clonación** | `--ref-audio <path>` | `None` | Solo `0.6B-Base`. Valida cabecera `RIFF`/WAV 24 kHz mono antes de cargar el modelo; otros formatos fallan con comando `ffmpeg`. |
| | `--ref-text <string>` | `None` | Transcripción de la referencia para activar modo ICL (o sobrescribir `ref_text` de `.qvoice`). |
| | `--xvector-only` | off | Usa únicamente el *x-vector* (`1024 × f32`), omitiendo los códigos ICL. |
| | `--save-voice <path>` | `None` | Guarda `.qvoice` v3 (con `--target-cv` incluye `WDLT`) o `.bin` (embedding raw). Sin `--text` ni `--serve`, crea el perfil y sale (`0`). |
| | `--load-voice <path>` | `None` | Carga `.qvoice` v3 o `.bin`. Si el `.qvoice` incluye `WDLT` sobre pesos `Q4_0` sin `--icl-only`, falla (`exit 1`). |
| | `--icl-only` | off | Omite la sección `WDLT` al cargar un `.qvoice` (conserva `WOVR` salvo `QWEN_GRAFT_NO_WOVR`). |
| | `--graft` | off | Implica `--icl-only`; ignora `ref_codes` y clona vía *x-vector* + `WOVR`. |
| | `--icl-frames <n>` | `0` *(todos)* | Tope máximo de fotogramas de referencia en modo ICL. |
| | `--voice-name <name>` | `None` | Nombre guardado en la sección `META` del `.qvoice`. |
| | `--max-ref-duration <s>` | `30.0` | Duración máxima en segundos leída de `--ref-audio` (`0` = audio completo). |
| | `--target-cv <dir>` | `None` | Directorio `0.6B-CustomVoice` usado para calcular deltas `WDLT` al guardar `.qvoice`. |
| | `--voice-strength <a>` | `1.0` | Factor de mezcla aplicado sobre proyecciones de `WDLT` en las capas `vs_layers`. |
| | `--vs-layers <a-b>` | `14-27` | Rango de capas del Talker moduladas por `--voice-strength` (`0 <= a <= b`). |
| **Expresividad** | `--emotion <name>` | `None` | Emociones válidas: `sad`/`sadness`, `joy`/`happy`/`joyful`, `anger`/`angry`/`rage`, `fear`/`afraid`, `disgust`/`disgusted`, `surprise`/`surprised`, `contempt`/`scorn`, `awe`/`wonder`, `nostalgia`/`wistful`, `disapproval`, `remorse`/`regret`, `outrage`, `despair`. Busca `<key>_<tok>.{qvoice,bin}` junto a `--load-voice` o en `presets/emovoice/`, o `dir_<tok>.bin` sobre voz clonada (ignorado si se pasa `--compose`). |
| | `--emotion-strength <f>` | `0.25` | Dosis de desplazamiento `dir_<tok>.bin` preservando norma L2. |
| | `--emo-ref <path>` | `None` | Audio WAV 24 kHz mono como ancla de prosodia ICL; exige `--emo-ref-text`. |
| | `--emo-ref-text <str>` | `None` | Transcripción de `--emo-ref`. |
| | `--roughness <0..1>` | `0.0` | Mezcla pesos `Q2_0` en `down_proj` del Code Predictor (`[0.0, 1.0]`). |
| | `--compose <spec>` | `None` | Síntesis multi-span no streaming con pausas `[pause:<s>]`, `[break:<s>]`, `[0.5]` o `\|`, y tags paralingüísticos inline (`[laugh]`, `[sigh]`, `[yawn]`, `[wow]`, `[giggle]`, `[scoff]`). |
| | `--compose-pause <s>` | `0.12` | Pausa por defecto entre spans hablados en `--compose` / `--batch`. |
| | `--no-compose` | off | Desactiva la sustitución de `[tags]` y el auto-compose en `--text`. |

### 1.5 Servidor HTTP, lotes y sistema

| Flag(s) | Default | Contrato y validación |
|---|---|---|
| `--serve <port>` | `None` | Inicia el servidor HTTP en `<port>` (`> 0`). |
| `--host <ipv4>` | `127.0.0.1` | Dirección `Ipv4Addr` validada antes de cargar el modelo. Inválida → `exit 1`. |
| `--workers <n>` | `1` | Número de *workers* concurrentes (`WorkerPoolScheduler`) cuando `--batch-size < 2`. |
| `--batch-size <n>` | `1` | Con `n >= 2` activa `ContinuousBatchScheduler` en servidor HTTP (acotado a `64` ranuras). |
| `--watch-stdin` | off | Vigila `stdin` en un hilo dedicado; al recibir EOF termina el proceso (`exit 0`). |
| `-j, --threads <n>` | `0` *(auto)* | Hilos de cómputo (`0` = `min(ncpus, 4)`). |
| `-S, --silent` | off | Suprime logs en `stderr`; no afecta el audio en `--stdout`. |
| `-D, --debug` | off | Imprime trazas internas de prefill y generación en `stderr`. |
| `--batch` | off | Divide `--text` por frases hacia lotes de `generate_batch_longform`. |
| `--batch-words <n>` | `16` *(min 4)* | Palabras objetivo por chunk en `--batch` (`min_words = max(n / 3, 6)`). |
| `--batch-dry` | off | Implica `--batch`; muestra la partición de chunks en `stderr` y sale (`0`). |
| `--batch-test`, `--batch-bench` | off | Autoprueba de equivalencia o microbenchmark entre ruta *batched* y *single-stream*. |
| `--batch-multi-test <N>` | `None` | Sintetiza `N` copias en lote a `bm_0.wav .. bm_<N-1>.wav`. |

---

## 2. Interfaz HTTP (`--serve`)

### 2.1 Seguridad de red, límites y CORS

- **Bind por defecto**: `127.0.0.1:<port>` (solo expone otra interfaz con `--host`).
- **Límite de cuerpo**: máximo `1 MiB` (`1_048_576` bytes) por petición HTTP.
- **CORS**: todas las respuestas incluyen `Access-Control-Allow-Origin: *`; `OPTIONS` responde `204 No Content`.
- **Formato de error**: `{"error":"<causa>"}` con `Content-Type: application/json`.

### 2.2 Endpoints y esquema JSON

| Método y Ruta | Entrada | Respuesta exitosa | Reglas específicas |
|---|---|---|---|
| `GET /v1/health` | — | `200` `{"status":"ok"}` | Comprobación de disponibilidad. |
| `GET /v1/speakers` | — | `200` `{"speakers":[...]}` | Lista los 9 presets `{name, language, gender}`. |
| `POST /v1/tts` | JSON TTS | `200` `audio/wav` (24 kHz s16le mono) | Usa `DEFAULT_CHUNK_FRAMES = 10`; `max_tokens = 8192` en `WorkerPoolScheduler`/fallback, o `600` (y prompt `<= 512`) en `ContinuousBatchScheduler`. |
| `POST /v1/audio/speech` | JSON OpenAI (`input`, `voice`, ...) | `200` `audio/wav` | Alias completo de `POST /v1/tts` (`input` → `text`, `voice` → `speaker`). |
| `POST /v1/tts/stream` | JSON TTS | `200` `audio/pcm` (`Transfer-Encoding: chunked`) | Cabeceras: `X-Sample-Rate: 24000`, `X-Sample-Format: s16le`, `X-Channels: 1`. Rechaza `rate != 1.0` (`400`); `max_tokens = 8192` / `600` en *continuous batching*. |

**Campos JSON aceptados en `POST`**:

| Campo | Tipo | Default | Notas |
|---|---|---|---|
| `text` o `input` | `string` | *(obligatorio)* | No vacío ni solo espacios/escapes `\n`, `\t`, `\r`; `<= 8192` bytes. |
| `speaker` o `voice` | `string` | `ryan` (o voz clonada) | Debe existir en `PRESET_SPEAKERS` si se envía (con voz clonada se mantiene la identidad clonada). |
| `language` | `string` | `English` (`None` en clon HTTP) | Debe existir en `lookup_language`. En HTTP con voz clonada, si se omite no fuerza idioma (`CODEC_NO_THINK`). |
| `temperature` | `number` | `0.5` | Rango válido `[0.0, 2.0]`. |
| `top_k` | `integer` | `50` | Rango válido `[0, 3072]`. |
| `top_p` | `number` | `1.0` | Rango válido `[0.0, 1.0]`. |
| `rep_penalty` | `number` | `1.05` | Rango válido `[0.5, 2.0]`. |
| `seed` | `integer` | reloj (`< 0`) | Entero `0..=u32::MAX` fija semilla; negativo usa reloj; `> u32::MAX` devuelve `400`. |
| `emotion` | `string` | `None` | Se valida el tipo `string` si se envía; no aplica steering en `0.6B` HTTP. |
| `volume` | `number` | `1.0` | Ganancia PCM lineal; rango válido `[0.0, 10.0]`. |
| `rate` | `number` | `1.0` | WSOLA pitch-preserving en `/v1/tts` (`[0.25, 4.0]`); prohibido `!= 1.0` en `/v1/tts/stream`. |
| `chunk_frames` | `integer` | `10` | Solo en `/v1/tts/stream`; se acota con `clamp(2, 250)`. |
| `voice_design` | `bool \| string \| number` | `false` | `true`, `"true"`, `"1"` o `1` devuelven `400` (`0.6B` no soporta `VoiceDesign`). |

### 2.3 Validación HTTP `400` y valores acotados

Toda petición se valida en `ValidatedTtsRequest::parse` antes de planificar o encolar síntesis:

- Cuerpo JSON inválido, no objeto, `Content-Length` inválido o tamaño mayor que `1 MiB` → `400`.
- `text`/`input` ausente, vacío, solo whitespace (o `\n`/`\t`/`\r` escapados) o `> 8192` bytes → `400`.
- `language` o `speaker`/`voice` desconocidos → `400`.
- `temperature` fuera de `[0.0, 2.0]`, `top_k` fuera de `[0, 3072]`, `top_p` fuera de `[0.0, 1.0]`, `rep_penalty` fuera de `[0.5, 2.0]`, `volume` fuera de `[0.0, 10.0]`, `rate` fuera de `[0.25, 4.0]` o `seed > u32::MAX` → `400`.
- `rate != 1.0` en `/v1/tts/stream` → `400`.
- `voice_design` activo → `400`.
- `chunk_frames` en `/v1/tts/stream` se acota silenciosamente a `[2, 250]`.

### 2.4 Streaming *chunked*, cancelación y cierre

1. **Rampa de chunks (`/v1/tts/stream`)**: primer fragmento de `2` fotogramas (`160 ms`); los siguientes (`k = 0, 1, 2, ...`) siguen `min(chunk_frames, 4 + 2 * (k / 8))`.
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
    Path::new("models/qwen3-tts-0.6b"),
    &runtime,
    load_opts,
)?);
```

- `ComputeRuntime::with_threads_and_spin(threads, spin_budget, sd_threads)` permite controlar explícitamente el presupuesto de *spin-wait* y los hilos de `sd_pool`.
- `Model06B::load` valida `config.json` para `0.6B`, abre `safetensors` vía `MmapStore` y cuantiza matrices según `ModelLoadOptions` (o `QWEN_TALKER_PREC` / `QWEN_CP_PREC`).

### 3.2 Preparación de `VoiceBindings` y `SessionScratch`

- **Voz preset**:
  ```rust
  use qwen_core::lookup_speaker;
  use qwen_engine::{SessionScratch, VoiceBindings};

  let speaker_id = lookup_speaker("ryan").map(|s| s.id).unwrap_or(3061);
  let voice = Arc::new(VoiceBindings::preset(speaker_id));
  let mut scratch = SessionScratch::new(); // reutilizable entre peticiones secuenciales
  ```
- **Voz clonada desde WAV o archivo `.qvoice`/`.bin`**:
  - `build_voice_from_ref_audio(&runtime, &model, ref_wav_path, save_path_opt, &SaveVoiceOptions::default())` requiere `model.config.is_base_model == true`.
  - `load_voice_from_file(&runtime, &mut model_mut, voice_path, ref_text_override, &VoiceLoadOptions::default())` carga `.qvoice` v3 o `.bin`; si el archivo contiene `WDLT`, debe llamarse sobre `&mut Model06B` antes de envolverlo en `Arc`.

### 3.3 Síntesis en memoria (`synthesize_to_samples`) y streaming (`synthesize_to_sink`)

```rust
use std::path::Path;
use qwen_engine::{
    synthesize_to_samples, synthesize_to_sink, write_wav, RequestConfig,
    RequestExecution, WavStreamSink,
};

// A) Síntesis completa a Vec<f32> (24 kHz mono)
let exec = RequestExecution::new(RequestConfig {
    speaker_id: 3061,
    language_id: Some(2050), // English
    temperature: 0.9,
    seed: 42,
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
| `OMP_NUM_THREADS` | OpenBLAS / Windows | — | Debe igualar `-j` cuando OpenBLAS usa OpenMP. |
| `OPENBLAS_NUM_THREADS` | OpenBLAS | *(no definir)* | Si está definida, `FfiBlas::set_threads` no reconfigura hilos entre *prefill* y generación. |
| `OPENBLAS_LIB_DIR` | Build (`qwen-kernels`) | `C:/msys64/ucrt64/lib` *(Win)* | Ruta de búsqueda para `openblas` / `libopenblas.a`. |
| `WINDRES` | Build (`qwen-app`) | auto | Compilador de recursos COFF para incrustar el manifiesto UTF-8 en Windows GNU. |
| `QWEN_BLAS` | Runtime | `ffi` | Con `--features faer-blas`, `QWEN_BLAS=faer` selecciona `PureRustBlas`. |
| `QWEN_POOL_SPIN` | Runtime | `4096` | Iteraciones de *spin-wait* en `ComputePool` (`0` = espera directa en `Condvar`). |
| `QWEN_SD_THREADS` | Runtime | `-j` | Número de hilos asignados a `sd_pool` (`SpeechDecoder`). |
| `QWEN_BLAS_GEN_THREADS` | Runtime | `max(1, nt - 1)` | Hilos BLAS durante la fase de generación paralela con `SpeechDecoder`. |
| `QWEN_TALKER_PREC` | Carga `Model06B` | según flags | Fuerza precisión del Talker: `bf16`, `int8` o `int4`. |
| `QWEN_CP_PREC` | Carga `Model06B` | según flags | Fuerza precisión del Code Predictor: `bf16`, `int8`, `int4` o `mixed`. |
| `QWEN_SPK_SCALE` | Prefill | `1.0` | Factor multiplicador sobre la norma L2 del *speaker embedding* clonado. |
| `QWEN_ICL_FRAMES` | Prefill ICL | `voice.icl_frames_cap` | Sobrescribe el límite de fotogramas de referencia ICL. |
| `QWEN_ICL_TRIM_FRAMES` | Pipeline ICL | `2` | Fotogramas recortados al inicio del audio generado en modo ICL (`2 × 1920` muestras). |
| `QWEN_NO_REF_TRIM` | Audio ref | off | Si está definida, desactiva `trim_trailing_silence` sobre `--ref-audio` / `--emo-ref`. |
| `QWEN_GRAFT_NO_WOVR` | Carga `.qvoice` | off | Si está definida, omite la sección `WOVR` al cargar `.qvoice`. |
| `QWEN_DUMP_CODES` | Pipeline | `None` | Ruta de archivo donde volcar los 16 codebooks por fotograma (`decode-dump`). |
| `QWEN_BATCH_SEQ` | CLI `--batch` | off | Valor distinto de `"0"` fuerza ejecución secuencial por spans en `--batch`. |
| `QWEN_BATCH_NOMATMUL` | Batching | off | Valor distinto de `"0"` desactiva la ruta `matmat` en `batch.rs`. |
| `QWEN_BATCH_FORCE_MATVEC` | Batching | off | Valor distinto de `"0"` fuerza `matvec` incluso con `b_dim > 1`. |
| `QWEN_BATCH_B` | `--batch-test` / `--batch-bench` / `--matmat-bench` | `8` | Tamaño de lote `B` en microbenchmarks (`1..=64`; valores fuera de rango usan `8`). |

### 4.2 Afinidad SMT en Windows y gestión de OpenBLAS

- **Afinidad Win32 (`affinity.rs`)**: al arrancar en Windows con 1 grupo de procesadores, `qwen_tts` fija `SetProcessAffinityMask` a **1 hilo lógico por núcleo físico** para evitar contención SMT del planificador.
- **OpenBLAS + OpenMP**: en Windows con `x86_64-pc-windows-gnu`, define `OMP_NUM_THREADS` igual a `-j` y mantén **sin definir** `OPENBLAS_NUM_THREADS` para que el motor pueda ajustar los hilos entre *prefill* (`nt`) y generación (`gen_blas`).

### 4.3 Líneas de log contractuales

Los tests y scripts de operación esperan las siguientes líneas exactas en `stderr`:

- **Carga de Code Predictor** (`model.rs`):
  `  Code Predictor: 5 layers loaded, q_dim=2048 kv_dim=1024 [BF16|INT8|Q4_0|INT8+Q4_0-FFN]`
- **Semilla resuelta** (`main.rs`):
  `seed: <u32>` *(o `seed: <u32> (auto/time-based)`)*
- **Generación en pipeline** (`pipeline.rs`):
  `  Generated <N> frames (<S>s audio)`
- **Servidor HTTP en petición exitosa** (`http.rs` / `scheduler.rs`):
  ```text
  [HTTP] Sent <bytes> bytes WAV (<dur>s audio) in <sec>s (RTF <rtf>)
  [HTTP] Streamed <samples> samples (<dur>s audio) in <sec>s (RTF <rtf>)
  [HTTP] Desglose: prefill <ms> ms, Talker <ms> ms (<ms/f> ms/f), CP <ms> ms (<ms/f> ms/f), decodificador <ms> ms, drenaje <ms> ms, <N> fotogramas en <M> fragmentos, cola máxima <K> fotogramas  (solo modo --workers / fallback)
  ```
- **Cancelación HTTP** (`http.rs` / `scheduler.rs`):
  `[HTTP] TTS cancelado: el cliente se desconectó tras <sec> s`
  `[HTTP] TTS stream cancelado: el cliente se desconectó tras <sec> s`
