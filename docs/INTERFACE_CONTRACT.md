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
| **Voz e idioma** | `-l, --language <name>` | el de la voz cargada o, si no tiene, `Spanish` (`2054`) | Acepta 10 idiomas *case-insensitive*: `english`, `german`, `spanish`, `chinese`, `japanese`, `french`, `korean`, `russian`, `italian`, `portuguese`. Una sola regla en todos los modos: el pedido (`-l`); si no hay, el idioma guardado en la voz (si es distinto de `0`; la voz `default` declara español); si tampoco, español. Desconocido → `exit 1`. |
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
| `--stream` | off | Escribe el WAV progresivamente conforme se decodifican chunks. Retiene los últimos 40 ms para aplicar la rampa de salida al cerrar. Incompatible con `--rate != 1.0` (`exit 1`). También con marcado y `--compose`: va por el mismo destino y da el mismo PCM que la entrega completa. |
| `--stdout` | off | Implica `--stream` y `--silent`; emite PCM crudo s16le 24 kHz mono a `stdout`, también con marcado y `--compose`. |
| `--stream-chunk <n>` | `10` | Fotogramas máximos por chunk (`clamp(2, 250)`). Rampa: `2` en el primer fragmento y `min(n, 4 + 2 * (k / 8))` para `k = 0, 1, 2, ...` en los siguientes. |
| `--volume <f>` | `1.0` | Ganancia lineal en `[0.0, 10.0]` aplicada sobre las muestras PCM con *clamp* a `[-1.0, 1.0]`. |
| `--rate <f>` | `1.0` | Cambio de velocidad WSOLA en `[0.25, 4.0]` preservando pitch (incompatible con `--stream`). |
| `--onset-fade <ms>` | `0` | Rampa de entrada sobre el ataque real detectado (solo entrega completa). |
| `--tail-trim` | off | Recorta cola metálica degenerada detectada al final (solo entrega completa). |
| `--seed-audition <N>` | `0` | Evalúa hasta `N` (`1..64`) semillas consecutivas y elige la mejor (solo entrega completa). |
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
| | `--compose <spec>` | `None` | Síntesis multi-span con pausas `[pause:<s>]`, `[break:<s>]`, `[0.5]` o `\|`, y tags paralingüísticos inline (`[laugh]`, `[sigh]`, `[yawn]`, `[wow]`, `[giggle]`, `[scoff]`), que se tratan igual que en `--text`. Sirve para la entrega completa y para `--stream`/`--stdout` (mismo PCM, `NUMERIC_CONTRACT.md` §4, invariante 5). Una composición solo de pausas, sin nada que hablar, es un error (`nothing to synthesize`) en las dos entregas. |
| | `--compose-pause <s>` | `0.12` | Pausa por defecto entre spans hablados en `--compose`. |
| | `--no-compose` | off | Desactiva la sustitución de `[tags]` y el auto-compose en `--text`. El marcado en `--text` (`[hmm]`, `[pause:0.4]`...) va por el mismo camino que `--compose`, con `--stream` y `--stdout` incluidos. |

### 1.5 Servidor HTTP y sistema

| Flag(s) | Default | Contrato y validación |
|---|---|---|
| `--serve <port>` | `None` | Inicia el servidor HTTP en `<port>` (`> 0`). |
| `--host <ipv4>` | `127.0.0.1` | Dirección `Ipv4Addr` validada antes de cargar el modelo. Inválida → `exit 1`. |
| `--watch-stdin` | off | Vigila `stdin` en un hilo dedicado; al recibir EOF termina el proceso (`exit 0`). |
| `-j, --threads <n>` | `0` *(auto)* | Hilos de cómputo (`0` = `min(ncpus, 4)`). |
| `-S, --silent` | off | Suprime logs en `stderr`; no afecta el audio en `--stdout`. Si se combina con `-D`, gana `-S`. |
| `-D, --debug` | off | Imprime trazas internas de prefill y generación en `stderr`. |

---

## 2. Interfaz HTTP (`--serve`)

### 2.1 Seguridad de red, límites y CORS

- **Bind por defecto**: `127.0.0.1:<port>` (solo expone otra interfaz con `--host`).
- **Límite de cuerpo**: máximo `1 MiB` (`1_048_576` bytes) por petición HTTP.
- **CORS**: todas las respuestas incluyen `Access-Control-Allow-Origin: *`; `OPTIONS` responde `204 No Content`.
- **Formato de error**: `{"error":"<causa>"}` con `Content-Type: application/json`.
- **Una conexión a la vez**: el servidor atiende cada conexión en línea, con un único trabajador y sin cola.
- **Calentamiento al arrancar**: el servidor sintetiza y descarta 12 fotogramas de «Hello.» antes de aceptar conexiones (~1 s), para que la primera petición no pague el arranque en frío. El commit en reposo queda en el nivel que antes alcanzaba tras la primera petición; el audio de las peticiones no cambia. Log: `[HTTP] Calentamiento: <ms> ms`.

### 2.2 Endpoints y esquema JSON

| Método y Ruta | Entrada | Respuesta exitosa | Reglas específicas |
|---|---|---|---|
| `GET /v1/health` | — | `200` `{"status":"ok"}` | Comprobación de disponibilidad. |
| `GET /v1/voices` | — | `200` `{"voices":[{"name":"<nombre>","language":"<nombre o null>","icl":<bool>}]}` | Devuelve la voz cargada. `"language"` es el idioma declarado por la voz (la voz `default` declara `"Spanish"`; `null` si una voz cargada no declara ninguno). |
| `POST /v1/tts` | JSON TTS | `200` `audio/wav` (24 kHz s16le mono) | WAV completo con rampas de 5 ms (entrada) y 40 ms (salida); mismo PCM que `/v1/tts/stream` y que la CLI. Usa `DEFAULT_CHUNK_FRAMES = 10`; `max_tokens = 8192`. |
| `POST /v1/audio/speech` | JSON OpenAI (`input`, `voice`, ...) | `200` `audio/wav` | Alias completo de `POST /v1/tts` (`input` → `text`). |
| `POST /v1/tts/stream` | JSON TTS | `200` `audio/pcm` (`Transfer-Encoding: chunked`) | Cabeceras: `X-Sample-Rate: 24000`, `X-Sample-Format: s16le`, `X-Channels: 1`. Rechaza `rate != 1.0` (`400`); `max_tokens = 8192`. |

**Campos JSON aceptados en `POST`**:

| Campo | Tipo | Default | Notas |
|---|---|---|---|
| `text` o `input` | `string` | *(obligatorio)* | No vacío ni solo espacios/escapes `\n`, `\t`, `\r`; `<= 8192` bytes. |
| `voice` | `string` | *(opcional)* | Si viene, debe coincidir (sin distinguir mayúsculas) con el nombre de la voz cargada; si no, `400` con `voice '<x>' is not loaded (loaded: '<nombre>')`. |
| `language` | `string` | idioma de la voz cargada o, si no tiene, `Spanish` | Debe existir en `lookup_language`. Es el idioma efectivo: el pedido, el de la voz o español, la misma regla que la CLI. El log `lang=` de las peticiones muestra el idioma efectivo. |
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

La librería comparte `ComputeRuntime`, `Model06B` y `Voice` entre hilos (todos inmutables durante la síntesis) mientras cada síntesis usa un `SessionScratch` mutable exclusivo. Una única entrada pública sintetiza: `Synthesizer::synthesize(scratch, &req, &abort, &script, delivery)`.

Los tipos de la petición viven en `qwen-core` (`Script`, `SamplingParams`, `plan_script`, `resolve_sampling`, `effective_language`) y los del motor en `qwen-engine` (`Voice`, `SynthesisRequest`, `Synthesizer`, `Delivery`, `AudioSink`, `EdgeRampSink`).

### 3.1 Inicialización de `ComputeRuntime` y carga de `Model06B`

```rust
use std::path::Path;
use qwen_engine::{Model06B, ModelLoadOptions};
use qwen_kernels::ComputeRuntime;

let runtime = ComputeRuntime::new(Some(4)); // `None` -> min(ncpus, 4)
let load_opts = ModelLoadOptions {
    quant_mixed: true, // Talker Q4_0 + CP [INT8+Q4_0-FFN]
    silent: false,
    ..Default::default()
};
let model = Model06B::load(Path::new("models/qwen3-tts-0.6b-base"), &runtime, load_opts)?;
```

- `Model06B::load` valida `config.json` para `0.6B`, abre `safetensors` vía `MmapStore` y cuantiza matrices según `ModelLoadOptions` (o `QWEN_TALKER_PREC` / `QWEN_CP_PREC`). Los codificadores del locutor y de voz son obligatorios: si faltan, es un error de carga.

### 3.2 `Voice` y `SessionScratch`

`Voice` es la única representación de una voz: `name`, `language_id: Option<i32>` (idioma declarado; `None` si no declara ninguno), `xvector: Vec<f32>` e `icl: Option<IclPrompt>`. `IclPrompt { ref_text, ref_codes }` ya está recortado al tope de fotogramas al construir la voz (`VoiceLoadOptions::icl_frames_cap` o, si está definida, `QWEN_ICL_FRAMES`).

- **Voz `default` incorporada**, que declara español (`2054`):
  ```rust
  use qwen_engine::{SessionScratch, Voice};

  let voice = Voice::builtin_default(); // nombre: "default"
  let mut scratch = SessionScratch::new(); // reutilizable entre síntesis secuenciales
  ```
- **Voz desde un `.qvoice` v4** (no toca el modelo): `Voice::load(path, &VoiceLoadOptions { ref_text_override, use_icl, icl_frames_cap, log })` o `Voice::from_qvoice(QVoiceFile, &opts)`. El nombre es el guardado en el archivo o, si está vacío, el del archivo.
- **Voz clonada desde un WAV**: `clone_voice(&runtime, &model, ref_wav, &CloneOptions { ref_text, max_ref_seconds, name, language_id, icl_frames_cap, log })` devuelve `(QVoiceFile, Voice)`. El `QVoiceFile` es el archivo completo (`ref_codes` sin recortar), listo para guardarse; el `Voice` es el ya construido y recortado. `Voice` no guarda archivos: `QVoiceFile` (`qwen-core`) es quien lee y escribe `.qvoice`.

### 3.3 Síntesis: `Synthesizer`, `Script` y `Delivery`

```rust
use std::path::Path;
use qwen_core::Script;
use qwen_engine::{
    write_wav, AbortHandle, Delivery, EdgeRampSink, SynthesisRequest, Synthesizer,
};

let synth = Synthesizer { runtime: &runtime, model: &model, voice: &voice };
let abort = AbortHandle::new();
let req = SynthesisRequest { language_id: 2050 /* English */, ..Default::default() };

// A) Entrega completa a Vec<f32> (24 kHz mono)
let out = synth.synthesize(
    &mut scratch, &req, &abort, &Script::Plain("Hello from Rust!".into()), Delivery::Buffer,
)?;
write_wav(Path::new("buffer.wav"), &out.samples.unwrap(), 24_000)?;

// B) Streaming progresivo hacia cualquier AudioSink + Send; aquí, PCM s16le en memoria
let mut sink = EdgeRampSink::new(Vec::<u8>::new(), 24_000);
synth.synthesize(
    &mut scratch, &req, &abort, &Script::Plain("Streaming audio!".into()),
    Delivery::Stream(&mut sink),
)?;
let pcm: Vec<u8> = sink.into_writer();
```

- **`Script`**: `Plain(String)` (texto sin marcado) o `Spans(Vec<SpanSpec>)` (tramos hablados y pausas). `qwen_core::plan_script(text, MarkupMode::{Auto, Literal, Compose})` es el único sitio que decide la sustitución paralingüística y los tramos.
- **`Delivery`**: `Buffer` devuelve todo el audio en `SynthesisOutput::samples`, con `time_stretch` (`rate`), `tail_trim` y `onset_fade`; `Stream(&mut sink)` entrega fragmentos al destino sin acabado y rechaza `rate != 1.0`. Los dos dan el mismo PCM (`NUMERIC_CONTRACT.md` §4, invariante 5).
- **`SynthesisRequest`**: `language_id` (ya resuelto con `effective_language(pedido, voz)`: pedido, voz o español), `sampling: SamplingParams` (`temperature`, `top_k`, `top_p`, `rep_penalty`, `greedy_warmup`, `seed`), `max_frames`, `chunk_frames`, `cp_roughness`, `volume`, `rate`, `compose_pause_s`, `tail_trim`, `onset_fade_ms` y `log: LogLevel`. Los valores por defecto (0,325 / 50 / 1,0 / 1,05 / 0 / 4, 8192 fotogramas, fragmentos de 10) están en un solo sitio.
- **Destinos de audio**: el trait `AudioSink` (`write_samples(&mut self, &[f32])` y `finish(&mut self)`) es el único destino del camino. `MemorySink` acumula `f32`; `EdgeRampSink<W: PcmWrite>` aplica las rampas de borde (5 ms de entrada, 40 ms de salida con retención) y convierte a PCM `s16le`, y no aplica ganancia (la aplica `synthesize` antes de escribir). `PcmWrite` (`write_pcm` y `finish`) tiene escritores incluidos —`WavFileWriter` (WAV progresivo), `StdoutWriter` y `Vec<u8>`— o uno propio.

### 3.4 Cancelación cooperativa (`AbortHandle`) y marcado

- **Cancelación manual o por sondeo**:
  ```rust
  use qwen_engine::{AbortHandle, EngineError};

  let abort = AbortHandle::new(); // o AbortHandle::with_poll(|| false)

  // Desde otro hilo:
  abort.cancel();
  // `synthesize` retorna `Err(EngineError::Aborted)`.
  ```
- **Marcado y multi-span**: `plan_script(text, mode)` devuelve un `PlannedScript` con el `Script` y la etiqueta paralingüística (semilla y temperatura propias), que `resolve_sampling(base, seed, temperature, para)` combina con lo que pidió el usuario (primero lo explícito, después la etiqueta, por último el valor por defecto). `Synthesizer::synthesize` compone los tramos con `compose_pause_s` entre ellos. Una composición sin nada que hablar (solo pausas) es un error.

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
| `QWEN_ICL_FRAMES` | Construcción de la voz | `icl_frames_cap` | Si está definida, sobrescribe el límite de fotogramas de referencia ICL al construir la `Voice`. |
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
- **Voz cargada** (`modes/voice_cmd.rs`, salvo con `-S`): `Voice: <nombre>` (`default`, el nombre guardado en el `.qvoice`, el nombre del archivo o el del WAV clonado).
- **Semilla resuelta** (`request.rs`):
  `seed: <u32>` *(o `seed: <u32> (auto/time-based)`)*
- **Generación en pipeline** (`pipeline.rs`):
  `  Generated <N> frames (<S>s audio)`
- **Servidor HTTP al arrancar** (`http.rs`): `[HTTP] Calentamiento: <ms> ms` (o `[HTTP] Calentamiento fallido (sin efecto en las peticiones): <causa>`).
- **Servidor HTTP en petición exitosa** (`http.rs`):
  ```text
  [HTTP] Sent <bytes> bytes WAV (<dur>s audio) in <sec>s (RTF <rtf>)
  [HTTP] Streamed <samples> samples (<dur>s audio) in <sec>s (RTF <rtf>)
  [HTTP] Desglose: prefill <ms> ms, Talker <ms> ms (<ms/f> ms/f), CP <ms> ms (<ms/f> ms/f), decodificador <ms> ms, drenaje <ms> ms, <N> fotogramas en <M> fragmentos, cola máxima <K> fotogramas
  ```
- **Cancelación HTTP** (`http.rs`):
  `[HTTP] TTS cancelado: el cliente se desconectó tras <sec> s`
  `[HTTP] TTS stream cancelado: el cliente se desconectó tras <sec> s`
