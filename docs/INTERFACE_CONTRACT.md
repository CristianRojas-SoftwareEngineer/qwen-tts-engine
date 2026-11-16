# Contrato de Interfaz Externa (`0.6B`)

Define la superficie pública que el binario Rust (`qwen-app`) debe respetar para ser un reemplazo compatible del motor C (`qwen3-tts/`) en CLI, servidor HTTP y ciclo de vida del proceso.

> **Alcance**: Exclusivo para **Qwen3-TTS 0.6B** (`CustomVoice` y `Base`). Las opciones exclusivas del `1.7B` (`--voice-design`, `-I`/`--instruct`, `--expr`, `--expr-weight`, `--ml-steer`) no forman parte de este motor.

---

## 1. Interfaz CLI

### 1.1 Modos sin modelo (salen inmediatamente)
| Flag | Comportamiento | Código de salida |
|---|---|---:|
| `-h`, `--help` | Imprime uso por `stderr` y termina. | `0` |
| `--caps` | Inicializa hilos, imprime capacidades SIMD/hilos/BLAS detectadas a `stdout` y termina. | `0` |
| `--self-test` | Ejecuta la batería numérica de kernels (`--self-test`) hacia `stdout`. | `0` si pasa, `>0` si falla |
| `--matmat-bench` | Mide `matmat` vs `B × matvec` hacia `stdout` con `-j` actual. | `0` |
| `--list-voices <dir>` | Lista perfiles `.qvoice` válidos en `<dir>`. | `0` / `1` |
| `--delete-voice <path>` | Valida cabecera `QVCE` y elimina el archivo `.qvoice`. | `0` / `1` |

### 1.2 Flags soportadas en `0.6B`

| Categoría | Flag(s) | Default | Contrato / Validación |
|---|---|---:|---|
| **Modelo y texto** | `-d, --model-dir <path>` | *(obligatorio)* | Error `exit 1` si falta o si el modelo no es `0.6B`. |
| | `-t, --text <string>` | `NULL` | Obligatorio salvo `--serve`, `--compose` o `--save-voice` sin texto. |
| | `-o, --output <path>` | `output.wav` | WAV 24 kHz, PCM s16le mono. |
| **Voz e idioma** | `-s, --speaker <name>` | `ryan` (`3061`) | Rechaza nombres desconocidos (`exit 1`). Presets: `ryan`, `vivian`, `serena`, `aiden`, `eric`, `dylan`, `uncle_fu`, `ono_anna`, `sohee`. |
| | `-l, --language <name>` | `English` (`2050`) | Rechaza idiomas desconocidos (`exit 1`). Ver tabla en `SPEC_06B.md`. |
| **Muestreo** | `-T, --temperature <f>` | `0.9` *(CLI)* / `0.5` *(server)* | `0.0` = greedy. |
| | `-k, --top-k <n>` | `50` | — |
| | `-p, --top-p <f>` | `1.0` | — |
| | `-r, --rep-penalty <f>` | `1.05` | — |
| | `-m, --max-tokens <n>` | `8192` | Sobrescrito por `--max-duration <secs>` (`secs * 12.5`). |
| | `--seed <n>` | tiempo (`-1`) | Imprime `seed: <u32> ...` en `stderr` si no está `--silent`. |
| | `--greedy-warmup <n>` | `0` | Primeros `n` fotogramas en greedy (`T=0, k=1`). |
| **Cuantización** | `--int8` | off | Talker + Code Predictor en INT8. |
| | `--int4` | off | Talker + Code Predictor en `Q4_0`. |
| | `--quant-mixed` | off | **Perfil recomendado 0.6B**: Talker `Q4_0` + CP atención/heads `INT8` y FFN `Q4_0` (`[INT8+Q4_0-FFN]`). |
| **Streaming / Audio** | `--stream` | off | Escribe WAV progresivamente. Incompatible con `--rate != 1.0` (`exit 1`). |
| | `--stdout` | off | Implica `--stream`; emite PCM s16le 24 kHz mono crudo a `stdout`. |
| | `--stream-chunk <n>` | `10` | Máximo de fotogramas por chunk (`2..250`). Rampa: `2, 4×8, 6×8, 8×8, ...`. |
| | `--volume <f>` | `1.0` | Ganancia lineal PCM. |
| | `--rate <f>` | `1.0` | Tempo WSOLA preservando pitch (incompatible con `--stream`). |
| **Clonación de voz** | `--ref-audio <path>` | `NULL` | Solo `0.6B-Base`. Valida cabecera `RIFF`/WAV 24 kHz mono antes de cargar el modelo; si es MP3/MP4/OGG/FLAC falla con guía `ffmpeg`. |
| | `--ref-text <string>` | `NULL` | Transcripción de referencia para modo ICL. |
| | `--xvector-only` | off | Solo speaker embedding (sin ICL). |
| | `--save-voice <path>` | `NULL` | `.qvoice` (graft `QVCE+META+TPAD+WOVR`, o `WDLT` con `--target-cv`) o `.bin` (embedding raw). Sin `--text` crea y sale (`0`). |
| | `--load-voice <path>` | `NULL` | Carga `.qvoice` o `.bin`. Con `--icl-only` omite `WDELTA`/`WFULL`. |
| | `--icl-only`, `--graft`, `--icl-frames <n>`, `--voice-name <s>`, `--max-ref-duration <s>`, `--target-cv <dir>`, `--voice-strength <a>`, `--vs-layers <a-b>` | `max_ref=30`, `strength=1.0` | Ver formato en `SPEC_06B.md`. Si `WDELTA` intenta sobrescribir pesos `Q4_0`, falla (`exit 1`). |
| **Expresividad 0.6B** | `--emotion <name>` | `NULL` | Busca `presets/emovoice/<key>_<tok>.{qvoice,bin}` o `dir_<tok>.bin` sobre clon. Falla explícitamente si no existe activo/clon. |
| | `--emotion-strength <f>` | `0.25` | Dosis de `dir_<tok>.bin` preservando norma L2. |
| | `--emo-ref <path>`, `--emo-ref-text <str>` | `NULL` | Emoción por ejemplo (ancla códec ICL). Exige `--emo-ref-text`. |
| | `--roughness <0..1>` | `0.0` | Mezcla `Q2_0` en `down_proj` del Code Predictor. |
| | `--compose <spec>`, `--compose-pause <s>`, `--no-compose` | `pause=0.12` | Marcado inline auto-detectado en `--text` salvo `--no-compose`. |
| **Servidor / Proceso** | `--serve <port>` | `0` (off) | Inicia servidor HTTP. |
| | `--host <ipv4>` | `127.0.0.1` | Validado con `inet_pton` (o parser IPv4 estricto) antes de cargar el modelo. Inválido → `exit 1`. |
| | `--workers <n>` | `1` | Número de workers concurrentes en servidor. |
| | `--batch-size <n>` | `1` | Continuous batching tipo vLLM en servidor (`n >= 2`). |
| | `--watch-stdin` | off | Vigila `stdin`; al detectar EOF (muerte del lanzador) termina inmediatamente (`_exit(0)`). Un fallo al iniciar el vigía aborta (`exit 1`). |
| **Sistema / Diagnóstico** | `-j, --threads <n>` | `0` (auto: `min(ncpus, 4)`) | Hilos de cómputo del pool. |
| | `-S, --silent`, `-D, --debug` | off | `--silent` suprime logs en `stderr`, **nunca** el WAV de salida. |
| | `--batch`, `--batch-words`, `--batch-dry`, `--batch-test`, `--batch-bench`, `--batch-multi-test`, `--onset-fade`, `--tail-trim`, `--seed-audition`, `--audition-keep` | — | Modos auxiliares de lote/audición/benchmark. |

---

## 2. Interfaz HTTP (`--serve`)

### 2.1 Seguridad de red y transporte
- **Bind por defecto**: `127.0.0.1:<port>` (nunca `0.0.0.0` salvo `--host 0.0.0.0` explícito).
- **Límite de cuerpo**: máximo `1 MiB` (`1024 * 1024` bytes) por petición HTTP; `OPTIONS` responde `204 No Content` con cabeceras CORS (`Access-Control-Allow-Origin: *`).
- **Errores**: `{"error":"<causa>"}` con `Content-Type: application/json`.

### 2.2 Endpoints
| Método y Ruta | Entrada | Respuesta exitosa | Reglas específicas |
|---|---|---|---|
| `GET /v1/health` | — | `200` `{"status":"ok"}` | — |
| `GET /v1/speakers` | — | `200` `{"speakers":[...]}` | Lista los 9 presets (`name`, `language`, `gender`). |
| `POST /v1/tts` | JSON TTS | `200` `audio/wav` (24 kHz s16le mono) | Usa tamaño de fragmento constante `10` (`DEFAULT_CHUNK_FRAMES`). |
| `POST /v1/audio/speech` | JSON OpenAI (`input`, `voice`, ...) | `200` `audio/wav` | Equivalente a `POST /v1/tts` (`input`→`text`, `voice`→`speaker`). |
| `POST /v1/tts/stream` | JSON TTS (`chunk_frames` opcional `2..250`, default `10`) | `200` `audio/pcm` (`Transfer-Encoding: chunked`) | Cabeceras obligatorias: `X-Sample-Rate: 24000`, `X-Sample-Format: s16le`, `X-Channels: 1`. Rechaza `rate != 1.0` con `400`. |

### 2.3 Validación única HTTP `400`
Toda petición en modo normal, worker o batch se valida antes de sintetizar. **Nada se recorta ni se sustituye en silencio**:
- `text` (o `input`) ausente, vacío, solo espacios (incluyendo `\n`, `\t`, `\r`) o `> 8192` bytes (`MAX_TTS_TEXT`) → `400`.
- `language` desconocido → `400`.
- `speaker` (o `voice`) desconocido → `400`.
- `temperature` fuera de `[0.0, 2.0]` → `400`.
- `top_k` fuera de `[0, 3072]` (`codec_vocab_size`) → `400`.
- `top_p` fuera de `[0.0, 1.0]` → `400`.
- `rep_penalty` fuera de `[0.5, 2.0]` → `400`.
- `rate != 1.0` en `/v1/tts/stream` → `400`.
- `voice_design` activado en `0.6B` → `400`.

### 2.4 Streaming, cancelación y cierre de stream
1. **Protocolo de chunked stream (`/v1/tts/stream`)**:
   - Primer chunk: `2` fotogramas (`160 ms`). Siguientes chunks siguen la rampa `min(chunk_frames, 4 + 2 * (k / 8))`.
   - **Señal de éxito vs fallo**: el chunk terminal `0\r\n\r\n` **solo se envía si la generación finalizó con éxito y produjo muestras (`> 0`)**. Si falla o el cliente corta la conexión, el socket se cierra **sin** enviar el chunk `0` para que el cliente detecte respuesta truncada.
2. **Cancelación por desconexión**:
   - El motor comprueba desconexión del cliente antes de cada fotograma (y en cada escritura fallida de streaming).
   - **Requisito para clientes**: no cerrar la mitad de escritura del socket (`shutdown(SHUT_WR)`) tras enviar el request HTTP, o se interpretará como abandono del cliente.

---

## 3. Ciclo de Vida del Proceso y Logs Operativos

- **Windows (afinidades y OpenBLAS)**:
  - Al arranque (`_WIN32`), si hay 1 grupo de procesadores, fijar máscara de afinidad a **1 hilo lógico por núcleo físico** (`SetProcessAffinityMask`) para evitar contención SMT.
  - Cuando se enlace con OpenBLAS compilado con OpenMP, el lanzador/usuario debe fijar `OMP_NUM_THREADS` igual a `-j` y **nunca** `OPENBLAS_NUM_THREADS` (porque impediría a `openblas_set_num_threads` ajustar los hilos entre prefill y generación).
- **Líneas de log contractuales (observadas por scripts/tests y diagnóstico)**:
  - Carga de Code Predictor: `Code Predictor: 5 layers loaded ... [BF16|INT8|Q4_0|INT8+Q4_0-FFN]`.
  - Semilla resuelta: `seed: <u32> ...`.
  - Generación CLI: `Generated <N> frames (...)` (comprobado por tests).
  - Servidor tras cada `/v1/tts` o `/v1/tts/stream` exitoso:
    ```text
    [HTTP] Sent <bytes> bytes (<dur>s audio) in <sec>s (RTF <rtf>)
    [HTTP] Desglose: prefill <ms> ms, Talker <ms> ms (<ms/f> ms/f), CP <ms> ms (<ms/f> ms/f), decodificador <ms> ms, drenaje <ms> ms, <N> fotogramas en <M> fragmentos, cola máxima <K> fotogramas
    ```
  - Cancelación en servidor: `[HTTP] TTS cancelado...` / `[HTTP] TTS stream cancelado...`.
