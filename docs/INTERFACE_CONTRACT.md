# Contrato de Interfaz Pública (`INTERFACE_CONTRACT.md`)

> Referencia normativa de las tres interfaces públicas de **Qwen TTS Engine** (`qwen-tts-engine`): binario CLI (`qwen_tts`), servidor HTTP (`--serve`) y API Rust embebida (`qwen-engine`, `qwen-core`, `qwen-kernels`), además de variables de entorno y logs operativos.

> **Alcance**: Exclusivo para **Qwen3-TTS 0.6B Base**. Cualquier otro checkpoint se rechaza al cargar; una opción que el motor no define es un error de uso.

---

## 1. Interfaz CLI (`qwen_tts`)

### 1.1 Modos auxiliares y subcomando posicional

| Comando / Flag | Comportamiento | Código de salida |
|---|---|---:|
| `-h`, `--help` | Imprime la ayuda de uso por `stderr` y termina. | `0` |
| `--caps` | Inicializa el runtime con `-j`, imprime capacidades SIMD/hilos/BLAS a `stdout` y termina. | `0` |
| `--self-test` | Ejecuta la batería numérica de kernels hacia `stdout`. | `0` si pasa, `1` si falla |
| `--list-voices <dir>` | Lista los `.qvoice` de `<dir>` (o un archivo `.qvoice` puntual): por voz, archivo, idioma (`-` si no tiene) y tamaño. Un archivo que el lector rechaza (por ejemplo, una versión distinta de la 6) aparece con `no compatible: <mensaje>`. La voz `default` no se lista: no es un archivo. | `0` / `1` |
| `--delete-voice <path>` | Valida la extensión `.qvoice` y la cabecera `QVCE` antes de eliminar el archivo. | `0` / `1` |

### 1.2 Flags de modelo, voz y muestreo

| Categoría | Flag(s) | Por defecto | Contrato y validación |
|---|---|---:|---|
| **Modelo y texto** | `-d, --model-dir <path>` | *(obligatorio con modelo)* | Directorio del modelo Qwen3-TTS `0.6B` Base. Rechaza cualquier checkpoint que no sea Base (`exit 1`, con el mensaje `Este motor solo carga Qwen3-TTS 0.6B Base: <detalle>`). |
| | `-t, --text <string>` | `None` | Obligatorio salvo `--serve` y `--save-voice` sin texto. |
| | `-o, --output <path>` | `output.wav` | Ruta de salida WAV (24 kHz, PCM s16le mono). Lleva una rampa de entrada de 5 ms y otra de salida de 40 ms con y sin `--stream`. El PCM es idéntico en los dos modos (`NUMERIC_CONTRACT.md` §4). |
| **Voz e idioma** | `-l, --language <name>` | el de la voz cargada o, si no tiene, `Spanish` (`2054`) | Acepta 10 idiomas *sin distinguir mayúsculas*: `english`, `german`, `spanish`, `chinese`, `japanese`, `french`, `korean`, `russian`, `italian`, `portuguese`. Una sola regla en todos los modos: el pedido (`-l`); si no hay, el idioma guardado en la voz (si es distinto de `0`; la voz `default` declara español); si tampoco, español. Desconocido → `exit 1`. |
| **Muestreo** | `-T, --temperature <f>` | `0.325` | Rango `[0.0, 2.0]`; `0.0` = voraz (siempre el código más probable). Si hay sustitución paralingüística en línea sin `-T`, usa la `T` del tag (`1.1`, o `1.0` en `[scoff]`). |
| | `-k, --top-k <n>` | `50` | Número de candidatos más probables entre los que se muestrea, en `[0, 3072]`. |
| | `-p, --top-p <f>` | `1.0` | Probabilidad acumulada máxima de esos candidatos, en `[0.0, 1.0]`. |
| | `-r, --rep-penalty <f>` | `1.05` | Penalización de repetición sobre tokens previos en `[0.5, 2.0]`. |
| | `--max-duration <secs>` | `None` | Debe ser `> 0`; el tope es `max(1, floor(secs · 12,5))` frames; sin el flag, `8192`; `≤ 0` → `exit 1`. |
| | `--seed <n>` | `4` (`< 0` reloj) | Entero `0..=u32::MAX` fija semilla; omitido usa `DEFAULT_SEED = 4` (salvo sustitución paralingüística en línea); negativo usa reloj del sistema; `> u32::MAX` → `exit 1`. |

**Cuantización**: no hay flags. El motor usa siempre el mismo perfil: Talker con el prefill en `BF16` y el decodificado paso a paso en `INT8`; Code Predictor con atención y `lm_head` en `INT8` y FFN en `Q4_0` (`[INT8+Q4_0-FFN]`).

### 1.3 Streaming, audio y post-procesado

| Flag(s) | Por defecto | Contrato y validación |
|---|---|---|
| `--stream` | off | Escribe el WAV progresivamente conforme se decodifican chunks. Retiene los últimos 40 ms para aplicar la rampa de salida al cerrar. Incompatible con `--rate != 1.0` (`exit 1`). También con marcado: va por el mismo destino y da el mismo PCM que la entrega completa. |
| `--stdout` | off | Implica `--stream` y `--silent`; emite PCM crudo s16le 24 kHz mono a `stdout`, también con marcado. |
| `--stream-chunk <n>` | `10` | Frames máximos por chunk (`clamp(2, 250)`). Rampa: `2` en el primer chunk y `min(n, 4 + 2 * (k / 8))` para `k = 0, 1, 2, ...` en los siguientes. |
| `--volume <f>` | `1.0` | Ganancia lineal en `[0.0, 10.0]` aplicada sobre las muestras PCM con *clamp* a `[-1.0, 1.0]`. |
| `--rate <f>` | `1.0` | Cambio de velocidad WSOLA en `[0.25, 4.0]` preservando el tono (incompatible con `--stream`). |

### 1.4 Voces, clonación y marcado `0.6B`

| Categoría | Flag(s) | Por defecto | Contrato y validación |
|---|---|---|---|
| **Clonación** | `--ref-audio <path>` | `None` | Valida cabecera `RIFF`/WAV 24 kHz mono antes de cargar el modelo; otros formatos fallan con comando `ffmpeg`. La referencia se usa entera. |
| | `--save-voice <path>` | `None` | Guarda `.qvoice` v6; la voz se llama como el archivo (la extensión debe ser `.qvoice`; otra da `los archivos de voz deben ser .qvoice (se obtuvo <ruta>)`). Con idioma `-l` lo guarda; sin `-l` guarda «sin idioma». Sin `--text` ni `--serve`, crea el perfil y sale (`0`). Requiere `--ref-audio`. |
| | `--load-voice <path>` | voz `default` incorporada | Carga `.qvoice` v6; la voz se llama como el archivo (sin extensión). Otra extensión o una versión distinta de la 6 → `exit 1`. Sin este flag ni `--ref-audio`, se usa la voz `default`. |
| | Reglas de las opciones de voz (valen en todos los modos; la primera que falle da `exit 1`) | | 1) `--ref-audio` con `--load-voice`; 2) `--save-voice` sin `--ref-audio`. Una voz clonada se llama como el archivo de `--save-voice` (sin extensión) o, si no se guarda, como el WAV de `--ref-audio`. |
| **Marcado y estilo** | Marcado en `--text` | sin flag propio | Síntesis multitramo con pausas `[pause:<s>]`, `[break:<s>]`, `[0.5]` o `\|`, y tags paralingüísticos en línea (`[laugh]`, `[sigh]`, `[yawn]`, `[wow]`, `[giggle]`, `[scoff]`). Sirve para la entrega completa y para `--stream`/`--stdout` (mismo PCM, `NUMERIC_CONTRACT.md` §4, invariante 5). Un texto solo de pausas, sin nada que hablar, es un error (`Composición: nada que sintetizar`) en las dos entregas. |
| | `--compose-pause <s>` | `0.12` | Pausa entre tramos hablados seguidos del texto con marcado (no se aplica junto a las muletillas, que se funden). |
| | `--no-compose` | off | Desactiva la sustitución de `[tags]` y la composición automática en `--text`. El marcado en `--text` (`[hmm]`, `[pause:0.4]`...) va por el mismo camino con `--stream` y `--stdout`. |

### 1.5 Servidor HTTP y sistema

| Flag(s) | Por defecto | Contrato y validación |
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

- **Enlace por defecto**: `127.0.0.1:<port>` (solo expone otra interfaz con `--host`).
- **Límite de cuerpo**: máximo `1 MiB` (`1_048_576` bytes) por petición HTTP.
- **CORS**: todas las respuestas incluyen `Access-Control-Allow-Origin: *`; `OPTIONS` responde `204 No Content`.
- **Formato de error**: `{"error":"<causa>"}` con `Content-Type: application/json`.
- **Una conexión a la vez**: el servidor atiende cada conexión en línea, con un único trabajador y sin cola.
- **Calentamiento al arrancar**: el servidor sintetiza y descarta 12 frames de «Hello.» antes de aceptar conexiones (~1 s), para que la primera petición no pague el arranque en frío. La memoria comprometida (*commit*) en reposo queda en el nivel que antes alcanzaba tras la primera petición; el audio de las peticiones no cambia. Log: `[HTTP] Calentamiento: <ms> ms`.

### 2.2 Endpoints y esquema JSON

| Método y Ruta | Entrada | Respuesta exitosa | Reglas específicas |
|---|---|---|---|
| `GET /v1/health` | — | `200` `{"status":"ok"}` | Comprobación de disponibilidad. |
| `GET /v1/voices` | — | `200` `{"voices":[{"name":"<nombre>","language":"<nombre o null>"}]}` | Devuelve la voz cargada. `"language"` es el idioma declarado por la voz (la voz `default` declara `"Spanish"`; `null` si una voz cargada no declara ninguno). |
| `POST /v1/tts` | JSON TTS | `200` `audio/wav` (24 kHz s16le mono) | WAV completo con rampas de 5 ms (entrada) y 40 ms (salida); mismo PCM que `/v1/tts/stream` y que la CLI. Usa `DEFAULT_CHUNK_FRAMES = 10`; `max_frames = 8192`. |
| `POST /v1/tts/stream` | JSON TTS | `200` `audio/pcm` (`Transfer-Encoding: chunked`) | Cabeceras: `X-Sample-Rate: 24000`, `X-Sample-Format: s16le`, `X-Channels: 1`. No admite el campo `rate` (`400`); `max_frames = 8192`. |

**Campos JSON aceptados en `POST`**:

| Campo | Tipo | Por defecto | Notas |
|---|---|---|---|
| `text` | `string` | *(obligatorio)* | No vacío ni solo espacios/escapes `\n`, `\t`, `\r`; `<= 8192` bytes. |
| `voice` | `string` | *(opcional)* | Si viene, debe coincidir (sin distinguir mayúsculas) con el nombre de la voz cargada; si no, `400` con `la voz '<x>' no está cargada (cargada: '<nombre>')`. |
| `language` | `string` | idioma de la voz cargada o, si no tiene, `Spanish` | Debe existir en `lookup_language`. Es el idioma efectivo: el pedido, el de la voz o español, la misma regla que la CLI. El log `lang=` de las peticiones muestra el idioma efectivo. |
| `temperature` | `number` | `0.325` | Rango válido `[0.0, 2.0]`. |
| `top_k` | `integer` | `50` | Rango válido `[0, 3072]`. |
| `top_p` | `number` | `1.0` | Rango válido `[0.0, 1.0]`. |
| `rep_penalty` | `number` | `1.05` | Rango válido `[0.5, 2.0]`. |
| `seed` | `integer` | `4` (`< 0` reloj) | Entero `0..=u32::MAX` fija semilla; omitido usa `DEFAULT_SEED = 4` (salvo sustitución paralingüística en línea); negativo usa reloj; `> u32::MAX` devuelve `400`. |
| `volume` | `number` | `1.0` | Ganancia PCM lineal; rango válido `[0.0, 10.0]`. |
| `rate` | `number` | `1.0` | Solo en `/v1/tts`: WSOLA que preserva el tono (`[0.25, 4.0]`). En `/v1/tts/stream` es un campo desconocido (`400`). |
| `chunk_frames` | `integer` | `10` | Solo en `/v1/tts/stream`; se acota con `clamp(2, 250)`. En `/v1/tts` es un campo desconocido (`400`). |

Un `null` cuenta como campo ausente. Cualquier otra clave da `400` con `campo desconocido '<clave>'`.

### 2.3 Validación HTTP `400` y valores acotados

Toda petición se valida en `ValidatedTtsRequest::parse` antes de sintetizar:

- Cuerpo JSON inválido, no objeto, `Content-Length` inválido o tamaño mayor que `1 MiB` → `400`.
- `text` ausente, vacío, solo espacios en blanco (o `\n`/`\t`/`\r` escapados) o `> 8192` bytes → `400`.
- `language` desconocido, o `voice` distinto de la voz cargada → `400`.
- Campo desconocido, o que la ruta no admite (`rate` en `/v1/tts/stream`, `chunk_frames` en `/v1/tts`) → `400` con `campo desconocido '<clave>'`.
- `temperature` fuera de `[0.0, 2.0]`, `top_k` fuera de `[0, 3072]`, `top_p` fuera de `[0.0, 1.0]`, `rep_penalty` fuera de `[0.5, 2.0]`, `volume` fuera de `[0.0, 10.0]`, `rate` fuera de `[0.25, 4.0]` o `seed > u32::MAX` → `400`.
- `chunk_frames` en `/v1/tts/stream` se acota silenciosamente a `[2, 250]`.

### 2.4 Streaming *chunked*, cancelación y cierre

1. **Rampa de chunks (`/v1/tts/stream`)**: primer chunk de `2` frames (`160 ms`); los siguientes (`k = 0, 1, 2, ...`) siguen `min(chunk_frames, 4 + 2 * (k / 8))`. El decoder usa el mismo calendario también en `/v1/tts` y en la CLI.
   - **Rampa de salida de 40 ms**: el servidor retiene los últimos 40 ms de audio y los envía con el chunk siguiente o al cerrar. Por eso el primer chunk entrega 120 ms en lugar de 160 ms.
2. **Señal de cierre vs fallo**: el terminador `0\r\n\r\n` solo se envía si la generación termina sin error y con `samples > 0`. Si falla o se corta la conexión, el servidor cierra el socket **sin** enviar `0\r\n\r\n` para que el cliente detecte transferencia truncada.
3. **Cancelación por socket**: el motor sondea desconexión del cliente antes de cada frame. El cliente **no** debe ejecutar `shutdown(SHUT_WR)` tras enviar la petición HTTP, o se interpretará como desconexión.

---

## 3. API Rust embebida (`qwen-engine`, `qwen-core`, `qwen-kernels`)

La biblioteca comparte `ComputeRuntime`, `Model06B` y `Voice` entre hilos (todos inmutables durante la síntesis) mientras cada síntesis usa un `SessionScratch` mutable exclusivo. Una única entrada pública sintetiza: `Synthesizer::synthesize(scratch, &req, &abort, &script, delivery)`.

Los tipos de la petición viven en `qwen-core` (`Script`, `SamplingParams`, `plan_script`, `resolve_sampling`, `effective_language`) y los del motor en `qwen-engine` (`Voice`, `SynthesisRequest`, `Synthesizer`, `Delivery`, `AudioSink`, `EdgeRampSink`).

### 3.1 Inicialización de `ComputeRuntime` y carga de `Model06B`

```rust
use std::path::Path;
use qwen_engine::Model06B;
use qwen_kernels::ComputeRuntime;

let runtime = ComputeRuntime::new(Some(4)); // `None` -> min(ncpus, 4)
// Último argumento: `silent` (`true` suprime las trazas de carga por `stderr`).
let model = Model06B::load(Path::new("models/qwen3-tts-0.6b-base"), &runtime, false)?;
```

- `Model06B::load` valida `config.json` para `0.6B`, abre `safetensors` vía `MmapStore` y cuantiza las matrices con el perfil único (Talker con el prefill en `BF16` y el decodificado paso a paso en `INT8`; Code Predictor `INT8` y `Q4_0`). El encoder del locutor es obligatorio: si falta, es un error de carga.

### 3.2 `Voice` y `SessionScratch`

`Voice` es la única representación de una voz: `name`, `language_id: Option<i32>` (idioma declarado; `None` si no declara ninguno) y `xvector: Vec<f32>`.

- **Voz `default` incorporada**, que declara español (`2054`):
  ```rust
  use qwen_engine::{SessionScratch, Voice};

  let voice = Voice::builtin_default(); // nombre: "default"
  let mut scratch = SessionScratch::new(); // reutilizable entre síntesis secuenciales
  ```
- **Voz desde un `.qvoice` v6** (no toca el modelo): `Voice::load(path, &VoiceLoadOptions { log })` o `Voice::from_qvoice(QVoiceFile, name)`. `load` llama a la voz como el archivo (sin extensión); `from_qvoice` usa el nombre dado.
- **Voz clonada desde un WAV**: `clone_voice(&runtime, &model, ref_wav, &CloneOptions { language_id, log })` devuelve el `QVoiceFile`, listo para guardarse; la voz se construye con `Voice::from_qvoice`. Antes de calcular el x-vector se recorta el silencio final de la referencia con el criterio de `trim_trailing_silence`. La implementación oficial no recorta; aquí se conserva porque, de oído, el clon de una grabación con silencio al final suena mejor recortado. `Voice` no guarda archivos: `QVoiceFile` (`qwen-core`) es quien lee y escribe `.qvoice`.

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

// B) Streaming progresivo hacia cualquier AudioSink + Send; aquí, PCM s16le hacia un
// destino propio `mi_destino` que implementa `PcmWrite`
let mut sink = EdgeRampSink::new(mi_destino, 24_000);
synth.synthesize(
    &mut scratch, &req, &abort, &Script::Plain("Streaming audio!".into()),
    Delivery::Stream(&mut sink),
)?;
```

- **`Script`**: `Plain(String)` (texto sin marcado) o `Spans(Vec<SpanSpec>)` (tramos hablados y pausas). `qwen_core::plan_script(text, MarkupMode::{Auto, Literal})` es el único sitio que decide la sustitución paralingüística y los tramos.
- **`Delivery`**: `Buffer` devuelve todo el audio en `SynthesisOutput::samples`, con `time_stretch` (`rate`); `Stream(&mut sink)` entrega chunks al destino y rechaza `rate != 1.0`. Los dos dan el mismo PCM (`NUMERIC_CONTRACT.md` §4, invariante 5).
- **`SynthesisRequest`**: `language_id` (ya resuelto con `effective_language(pedido, voz)`: pedido, voz o español), `sampling: SamplingParams` (`temperature`, `top_k`, `top_p`, `rep_penalty`, `seed`), `max_frames`, `chunk_frames`, `volume`, `rate`, `compose_pause_s` y `log: LogLevel`. Los valores por defecto (0,325 / 50 / 1,0 / 1,05 / 4, 8192 frames, chunks de 10) están en un solo sitio.
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
- **Marcado y multitramo**: `plan_script(text, mode)` devuelve un `PlannedScript` con el `Script` y el tag paralingüístico (semilla y temperatura propias), que `resolve_sampling(base, seed, temperature, paralinguistic)` combina con lo que pidió el usuario (primero lo explícito, después el tag, por último el valor por defecto). `Synthesizer::synthesize` compone los tramos con `compose_pause_s` entre ellos. Una composición sin nada que hablar (solo pausas) es un error.

---

## 4. Variables de entorno, ciclo de vida y logs operativos

### 4.1 Variables de entorno soportadas

| Variable | Ámbito | Por defecto | Efecto |
|---|---|---|---|
| `OPENBLAS_NUM_THREADS` (o `GOTO_NUM_THREADS`, `OMP_NUM_THREADS`) | OpenBLAS / Windows | `-j` *(lo fija el binario en Windows)* | El motor fija los hilos con `-j` e ignora estas variables. En Windows escribe `OPENBLAS_NUM_THREADS` al `-j` efectivo antes de que OpenBLAS se inicialice. En Linux, OpenBLAS las lee antes de `main`; si alguna está por debajo de `-j`, limita los hilos de BLAS (es la inicialización de OpenBLAS, no un ajuste del motor). |
| `OPENBLAS_LIB_DIR` | Compilación (`qwen-kernels`) | *(obligatoria en Linux y Windows)* | Directorio del `libopenblas.a` propio, que el motor enlaza estático (`scripts/build-openblas.sh` lo compila). Es obligatoria, sin valor por defecto, y debe contener `libopenblas.a`; si falta, la compilación falla con un mensaje que explica cómo compilarlo. En la documentación: `C:/openblas/lib` en Windows (`x86_64-pc-windows-gnu`) y `$HOME/openblas/lib` en Linux. En macOS no se usa (Accelerate). |
| `QWEN_KERNELS` | Runtime | `auto` | `auto`: los kernels más rápidos que soporte la CPU (AVX2 + FMA en x86-64; en aarch64, NEON en todos los kernels, con `SDOT` en Q4_0 e INT8 si la CPU tiene `dotprod`; `--caps` muestra la ruta en la línea `kernels:`). `scalar`: fuerza la ruta escalar, como referencia para comparar kernels en la misma máquina; `--caps` lo indica con `kernels: scalar (forzado por QWEN_KERNELS=scalar)`. Un valor no válido es un error al arrancar (`exit 1`). Cambiar de ruta cambia el redondeo y, por tanto, el audio. |

### 4.2 Afinidad SMT en Windows y gestión de OpenBLAS

- **Afinidad Win32 (`affinity.rs`)**: al arrancar en Windows con 1 grupo de procesadores, `qwen_tts` fija `SetProcessAffinityMask` a **1 hilo lógico por núcleo físico** para evitar contención SMT del planificador. El mensaje `Afinidad: …` no sale con `-S`.
- **OpenBLAS propio**: en Windows con `x86_64-pc-windows-gnu`, el OpenBLAS propio (0.3.34, sin OpenMP) está enlazado estático y se inicializa antes de `main`, reservando 128 MiB de memoria comprometida (*commit*) por hilo. El binario fija `OPENBLAS_NUM_THREADS` al `-j` efectivo antes de esa inicialización (`.CRT$XCT`); no hace falta definir nada. `set_threads` llama una vez a `openblas_set_num_threads`, que es global al proceso, y nunca pasa de los hilos iniciales (OpenBLAS 0.3.34 se cuelga si se le piden más).

### 4.3 Líneas de log de referencia

Formato actual de las líneas de diagnóstico que el motor escribe en `stderr`. Son texto para personas y no forman parte del contrato: pueden cambiar, y los scripts y las pruebas del repositorio que las leen se actualizan con ellas.

- **Carga del Talker** (`model.rs`):
  `  Talker: 28 capas cargadas, caché KV de <N> slots [INT8]`
- **Carga de Code Predictor** (`model.rs`):
  `  Code Predictor: 5 capas cargadas, q_dim=2048 kv_dim=1024 [INT8+Q4_0-FFN]`
- **Voz cargada** (`modes/voice_cmd.rs`, salvo con `-S`): `Voz: <nombre>` (`default`, el nombre del archivo `.qvoice` o el del WAV clonado).
- **Semilla resuelta** (`request.rs`):
  `semilla: <u32>` *(o `semilla: <u32> (automática/temporal)`)*
- **Generación en pipeline** (`pipeline.rs`):
  `  Generados <N> frames (<S>s de audio)`
- **Servidor HTTP al arrancar** (`http.rs`): `[HTTP] Calentamiento: <ms> ms` (o `[HTTP] Calentamiento fallido (sin efecto en las peticiones): <causa>`).
- **Servidor HTTP en petición exitosa** (`http.rs`):
  ```text
  [HTTP] Enviados <bytes> bytes de WAV (<dur>s de audio) en <sec>s (RTF <rtf>)
  [HTTP] Transmitidas <samples> muestras (<dur>s de audio) en <sec>s (RTF <rtf>)
  [HTTP] Desglose: prefill <ms> ms, Talker <ms> ms (<ms/f> ms/f), CP <ms> ms (<ms/f> ms/f), decoder <ms> ms, drenaje <ms> ms, <N> frames en <M> chunks, cola máxima <K> frames
  ```
- **Cancelación HTTP** (`http.rs`):
  `[HTTP] TTS cancelado: el cliente se desconectó tras <sec> s`
  `[HTTP] TTS stream cancelado: el cliente se desconectó tras <sec> s`
