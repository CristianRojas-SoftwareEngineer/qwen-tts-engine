# Contrato Numérico, Cuantización y Determinismo (`NUMERIC_CONTRACT.md`)

> Fija las leyes aritméticas, de cuantización, alineación de memoria, punto flotante y *streaming* que **Qwen TTS Engine** (`qwen-tts-engine`) cumple para preservar el determinismo, el SNR y el RTF en `0.6B`. Pensado para ingenieros de kernels y optimización.

---

## 1. Cuantización y aritmética entera

### 1.1 Cuantizador único de activaciones (`quantize_act_int8`)

Toda multiplicación cuantizada (`int8` y `q4_0`, tanto `matvec` como cada columna de `matmat` y `argmax`) cuantiza el vector `x[0..n]` con idéntica numérica:

1. `amax = max(|x_i|)`. Si `amax == 0.0`, `qx = [0; n]` y `scale = 0.0`.
2. `inv = 127.0 / amax`, `scale = amax / 127.0`.
3. `qx[i] = clamp(round_half_even(x[i] * inv), -128, 127)` (`round_ties_even` / `_mm256_cvtps_epi32`).
4. Límite de pila y lote: `QWEN_QACT_MAX = 8192`. Toda matriz cuantizada debe cumplir `cols > 0 && cols <= 8192 && cols % 32 == 0`, y toda llamada `matmat_*` exige explícitamente `b_dim <= 64`.

### 1.2 Formatos `INT8` y `Q4_0`

| Formato | Layout por bloque / fila | Regla de cuantización | Uso en `0.6B` |
|---|---|---|---|
| **INT8** | `i8[rows, cols]` + `f32 scale[rows]` | Por fila: `s = absmax / 127.0`, `round_ties_even(w * 127.0 / absmax)` saturado a `[-128, 127]`. | Decodificado paso a paso del Talker, y atención y 15 `lm_head` del Code Predictor. |
| **Q4_0** (`18 B / 32 pesos`) | `scale_f16` (`u16`) + `qs[16]` | **Mínimos cuadrados ponderados (`q4_0s1`)**: toma el elemento de mayor magnitud `vmax` (con signo), fija `isc = -8.0 / vmax`, `q[i] = clamp(round(v[i] * isc), -8, 7)` y reescala `s* = sum(v^2 * v * q) / sum(v^2 * q^2)`. Empaque intercalado: `qs[i] = (q[2i]+8) \| ((q[2i+1]+8) << 4)`. | FFN del Code Predictor. |

- **Ninguna ruta de reserva en coma flotante**: todas las rutas (`AVX2` `maddubs`, NEON `SDOT` o `SMULL` y escalar `i32`) acumulan en producto escalar entero.
- **Matriz `gate_up` intercalada**: la fila `2r` corresponde a `gate[r]` y la `2r+1` a `up[r]`. En el Code Predictor (`Q4_0`) se cuantiza directamente desde `gate_bf16` y `up_bf16`, sin materializar la matriz intercalada en BF16; en el Talker (`INT8`), `quantize_gate_up_int8` cuantiza cada fila de `gate` y de `up` directamente en su fila entrelazada (pares `gate`, impares `up`), con una escala por fila, sin materializar la matriz intercalada en BF16.

### 1.3 Perfil de precisión único (Talker `INT8`; Code Predictor `[INT8+Q4_0-FFN]`)

- **Talker (28 capas)**: `wq`, `wk`, `wv`, `wo`, `gate_up` (filas entrelazadas) y `down` en `INT8` para el decodificado paso a paso (el prefill lee los pesos `BF16`); `text_embedding`, `text_projection`, `codec_embedding`, `codec_head` en `BF16`; norms en `F32`; caché KV en `BF16`.
- **Code Predictor (5 capas)**: `wq`, `wk`, `wv`, `wo` y 15 `lm_head` en `INT8` *(mantiene salida por `argmax_matvec_int8`, nunca en `Q4_0`)*; `gate_up` y `down` en `Q4_0`; `codec_embedding` en `BF16`; caché KV en `BF16`.

---

## 2. Hardware, alineación y kernels SIMD

1. **FTZ / DAZ por hilo (`ftz_on`)**: las activaciones cuantizadas entran con frecuencia en rango subnormal (penalización ~100×). El hilo principal (`ComputeRuntime`), cada hilo trabajador de `ComputePool` y los puntos de entrada cuantizados de `KernelDispatch` activan:
   - **x86-64**: `MXCSR |= 0x8040` (`FTZ` + `DAZ`).
   - **ARM64**: `FPCR |= (1 << 24)` (`FZ`).
2. **Alineación a línea de caché**: activaciones, caché KV, pesos cuantizados, `W_k` y workspaces del decoder usan `AlignedBuf<64>`.
3. **Patrones `AVX2` (`qwen-kernels/src/simd/avx2.rs`)**:
   - **Prefetch**: en `int8`, anticipa `5120` bytes (`QWEN_INT8_PF_DIST = 5120`) por fila procesada; en `q4_0`, 16 bloques (288 bytes).
   - **Filas largas**: en `int8_matvec_slice` y `q4_0_matvec_slice`, procesa **1 fila por iteración** cuando `cols >= 2048` (`QWEN_LONG_ROW_COLS`), manteniendo acumulación idéntica bit a bit frente a 2 filas (`matmat` mantiene 2 filas).
4. **Kernels NEON (`qwen-kernels/src/simd/neon.rs`, aarch64)**: todas las entradas de la tabla, igual que AVX2. Q4_0 e INT8 (`matvec`, QKV fusionado y `matmat`) en dos variantes; el resto (cuantización de activaciones, normas RMS, BF16, atención, RoPE, conversiones, matvec `f32` del decoder y *snake*) en NEON base, con las mismas operaciones que AVX2 en el mismo orden: cada registro AVX2 de 8 `f32` es un par de registros NEON y la reducción final es la de `hsum256_ps`.
   - **Dos variantes con el mismo código**: `neon-dotprod` usa `SDOT` (emitido con `asm!`, porque `vdotq_s32` no es estable en Rust 1.96) y `neon`, multiplicación ampliada `SMULL` con sumas por pares, para núcleos ARMv8.0 sin `dotprod`. `KernelDispatch::for_caps` elige `neon-dotprod` si la CPU tiene `dotprod`; si no, `neon`. Las dos dan los mismos bits.
   - **Orden de suma de AVX2 en Q4_0**: por bloque, 8 sumas enteras de 4 productos (pesos sin desplazar, `0..15`) menos la suma de las 32 activaciones, conversión a `f32` y FMA en 8 carriles; reducción final en el orden de `hsum256_ps`, `((c0 + c4) + (c2 + c6)) + ((c1 + c5) + (c3 + c7))`. Mismas filas largas (`QWEN_LONG_ROW_COLS`) y mismos microkernels `matmat` de 8 columnas. Así NEON y AVX2 dan los mismos bits en estos kernels (prueba `test_simd_q4_0_matches_avx2_summation_order`, con el hash de la ruta AVX2).
   - **Mismos bits que AVX2** en todos los kernels que no usan `libm`: prueba `test_simd_float_kernels_match_avx2_summation_order`, con el hash de AVX2. La atención (`exp`) y el *snake* (`exp` de sus parámetros y `sin` de respaldo) dependen de la `libm` de cada plataforma.
   - **El audio de ARM sigue sin ser el de x86**: el prefill y las convoluciones del decoder usan BLAS (OpenBLAS o Accelerate), que redondea distinto en cada plataforma, y `exp`/`sin` vienen de su `libm`. En ARM el audio se compara con la ruta escalar de la misma máquina (`QWEN_KERNELS=scalar`), no por hash con x86.
5. **Reparto de filas entre hilos (`parallel_rows`)**: las `matvec`/`matmat` de 256 filas o más se reparten en `PARALLEL_ROWS_CHUNKS_PER_THREAD = 4` chunks por hilo con bordes múltiplos de 64 filas, que los hilos toman de un contador compartido. Cada fila se calcula con el mismo kernel y el mismo orden de suma, así que el resultado no depende del reparto: 1 hilo y 4 hilos dan los mismos bits (prueba `test_parallel_rows_partition_bit_exact`). Con la CPU ocupada, un hilo frenado no retrasa a los demás en la barrera.

---

## 3. Sensibilidades de punto flotante

| Componente | Regla crítica | Motivo |
|---|---|---|
| **Kernels SIMD** | FMA explícito (`_mm256_fmadd_ps`, `vfmaq_f32`) | Evita divergencia observada sin contracción FMA en frases cortas. |
| **`convnext_mlp` (`SpeechDecoder`)** | **LayerNorm de 2 pasadas** (`mean`, luego `var`) | La variante de una pasada altera la señal respecto al streaming validado. |
| **Activación Snake (`SpeechDecoder`)** | `y = x + exp(-log_beta) * sin^2(exp(log_alpha) * x)` en `F32` | Los parámetros `{alpha, beta}` del checkpoint están en escala logarítmica. |
| **`causal_conv1d` / `conv_transpose1d`** | `conv_transpose1d`: 1 `FfiBlas::sgemm` + dispersión en orden ascendente de `k`; `causal_conv1d` (`k>1`): `k` `sgemm` acumulados sobre `W_k` `[k][out][in]`, salvo la `conv1` k=7 de los bloques residuales que el *streaming* procesa por tiles (`b >= SD_TILE_FROM_BLOCK`, 192 y 96 canales): **im2col del tile y una sola `sgemm`** con `K = 7 · canales` sobre `conv1_w` `[out][k · in]` | Sin `im2col` se ahorran copias con canales anchos; con 96–192 canales, `K = canales` rinde 1,7× peor por FLOP y el im2col del tile (≤ 1,75 MiB) compensa. |
| **Atención con ventana deslizante (`SpeechDecoder`)** | Un único kernel `causal_attention_windowed` (*online softmax* + FMA) para el decodificador por partes; consultas en coordenadas de la caché compactable (`q_offset = kv_len - kv_base`), que conserva `window - 1` posiciones | Un solo kernel para todo el audio |
| **`Talker` prefill** | Pesos `BF16` → `F32` por bloques de filas de **4 MiB** (`PREFILL_STAGE_ELEMS`) y una `sgemm` por bloque (`gate_up` intercalado por parejas de filas) | OpenBLAS redondea según el ancho del bloque y el muestreo autorregresivo amplifica cualquier diferencia hasta cambiar los códigos: **cambiar el tamaño de bloque cambia el audio** y exige una línea base nueva. |
| **`Talker`: `codec_head`** | Solo se calculan las filas que sobreviven a `apply_talker_logit_rules`: los códigos de audio `0..2048` y `CODEC_EOS` = 2150, este junto a su pareja 2151 (`talker_codec_logits`, camino de una petición) | Las reglas fijan a `-1e30` el resto de `2048..3072` sin mirar su valor; calcular la pareja par conserva el emparejamiento de filas del kernel BF16 y da los mismos bits que el cálculo completo (pruebas `pipeline::tests`). |
| **`CodePredictor`: pasadas de 2 posiciones** | **`cp_step2(pos)`** (`matmat(B=2)`): posiciones `0`–`1` y, después, **verificación especulativa de 1 adivinanza**. Tras `c[g]` se adivina `ĉ = argmax(lm_heads[g+1](h[g+1]))` y una pasada calcula las posiciones `g+2` (entrada `c[g]`) y `g+3` (entrada `ĉ`); se acepta la segunda columna solo si el código real `c[g+1]` (`argmax` sobre la columna 0) es `ĉ`, y si no se descarta (su K/V se sobrescribe antes de leerse) | Cada matriz se lee una vez para las dos columnas; cada código sigue siendo el `argmax` de la misma cabeza sobre el mismo estado, y cada columna es bit-idéntica a un `cp_transformer_step` (pruebas `code_predictor::tests`). Pasadas de pesos por frame: 15 → ~11,9 (acierta el 28–31 %). |

---

## 4. Invariantes de determinismo en generación y *streaming*

1. **Prefill completo**: cada petición arranca con `kv_len = 0` desde la posición `0` (sin reutilización parcial entre peticiones).
2. **Hilos BLAS congelados durante generación**: prefill usa `nt` hilos; antes de lanzar el hilo del decoder se fija `gen_blas = max(1, nt - 1)` y no cambia hasta terminar el drenaje final.
3. **Calendario fijo de chunks (con y sin streaming)**: primer chunk de `2` frames (`160 ms`); siguientes `k = 0, 1, 2, ...` de `min(chunk_frames, 4 + 2 * (k / 8))`, con `chunk_frames = 10` (`DEFAULT_CHUNK_FRAMES`) salvo que la petición lo cambie. Los dos modos usan el mismo calendario porque las fronteras de chunk fijan el redondeo del decoder: así el PCM es idéntico en entrega completa y en *streaming*. La evidencia y el coste de esta regla están en [ADR-005](adr/005-cpu-inference-optimizations.md).
4. **Troceo por tiles y workspace del decoder**: `DecoderWorkspace` reutiliza 8 slots crecientes. Los bloques `up2` y `up3` (`SD_TILE_FROM_BLOCK = 2`: todos los bloques con índice 2 o mayor) procesan siempre por tiles de **`256 KB`** (`SD_TILE_BYTES = 262144`) durante toda la petición (incluido el drenaje final).
5. **Rampas de borde y composición.** Un mismo guion da el mismo PCM en la entrega completa y en *streaming*.
   - **Rampas de borde**: 5 ms de entrada y 40 ms de salida, cada una acotada a la mitad del audio, aplicadas después de la ganancia (`apply_gain`) y antes de la conversión a PCM16. El WAV completo las aplica en `encode_wav_bytes`. Las salidas en *streaming* (`--stdout` y `/v1/tts/stream`) las aplican con `EdgeRampSink`, que retiene los últimos 40 ms hasta el chunk siguiente o el cierre y no emite nada hasta reunir 80 ms. Ambos caminos usan las mismas operaciones en el mismo orden (prueba `audio::tests`). El descarte de la cadena de silencio inicial (los frames del arranque cuyo `code0` pertenece a `LEADING_SILENCE_CODES`; si la síntesis no sale de la cadena se entrega entera) ocurre antes, en el hilo del decodificador: precede a la ganancia, a `time_stretch`, a la composición de tramos y a las rampas de borde, de modo que la rampa de entrada cae sobre la primera muestra entregada.
   - **Ganancia**: la aplica siempre `synthesize`, elemento a elemento (`(s·g)` con *clamp* a `[-1, 1]`), justo antes de escribir en el destino; `EdgeRampSink` no aplica ganancia. En texto plano es `x·v`. Con `rate != 1`, la ganancia va antes de `time_stretch`.
   - **Composición multi-tramo** (`Script::Spans`, una sola implementación para las dos entregas): cada tramo hablado se sintetiza como texto plano con ganancia `x·(v·o)` (`v` el volumen de la petición, `o` el del tramo) y, si el tramo trae `rate`, `time_stretch` después de la ganancia. Entre tramos hablados va una pausa (`compose_pause_s`); junto a una muletilla (`[hmm]` y similares) no hay pausa y los tramos se unen con un **fundido cruzado de 45 ms**, para lo que se retienen las últimas 45 ms emitidas. Las pausas explícitas (`[pause:<s>]`) son silencio exacto.
   - Verificado por `qwen-engine/tests/synthesis_paths_tests.rs`: `Delivery::Buffer` y `Delivery::Stream(EdgeRampSink)` dan los mismos bytes PCM en texto plano, con voz clonada y con marcado.

---

## 5. Matriz de tolerancias

| Prueba / Nivel | Comparación | Umbral obligatorio |
|---|---|---|
| **Kernels `int8`** (`--self-test`) | `matmat_int8(B=8)` vs `B × matvec_int8` | **`L2_rel == 0.0` (bit-idéntico)** |
| | `matvec_int8` vs referencia `f32` | `rel_L2 < 3e-2` |
| | `argmax_matvec_int8` vs referencia | Índice exacto o empate `< 0.02 * (\|max\| + 1e-3)` |
| **Kernels `q4_0`** (`--self-test`) | `matmat_q4_0(B=8)` vs `B × matvec_q4_0` | **`L2_rel <= 1e-6`** (`0.0` en `AVX2` y escalar) |
| | `matvec_q4_0` vs dequant `f32` ref | `rel_L2 < 3e-2` |
| **QKV fusionado `int8`** (`--self-test`) | `qkv_int8` vs `3 × matvec_int8` | **Bit-idéntico** |
| **Rutas SIMD frente a la escalar** (`--self-test`, cada ruta que pueda ejecutar la CPU: `avx2`, `neon`, `neon-dotprod`) | `matvec_int8`, `matmat_int8(B=3, 8)`, `qkv_int8` | **Bit-idéntico** |
| | `matvec_q4_0`, `matmat_q4_0(B=3, 8)` | **`L2_rel <= 1e-6`** (otro orden de suma; AVX2 y NEON dan ≤ 3,5e-7) |
| | `quantize_act_int8`, `bf16_accum_f32`, `bf16_to_f32_vec`, `f32_to_bf16_vec`, suma residual de `rms_norm_residual` | **Bit-idéntico** |
| | normas RMS, `matvec_bf16`, atención `f32` (con y sin ventana) y BF16 (en serie y repartida), RoPE, `f32_matvec_accum`, *snake* | **`L2_rel <= 1e-6`** (AVX2 da ≤ 6,2e-7) |
| **Kernels `bf16`** (`--self-test`) | `matvec_bf16` vs `f32` ref | `max_rel < 1e-2` |
| **Pipeline E2E** (misma semilla/params) | 2 corridas independientes | **Bit-idéntico** entre corridas y entre los cuatro modos: CLI, CLI `--stdout`, `/v1/tts` y `/v1/tts/stream` dan el mismo PCM (mismo calendario de chunks y mismas rampas de borde, invariantes 3 y 5); también con marcado. Comprobado en las 4 combinaciones de idioma y longitud de la tabla de referencia. |

### Audio de referencia (x86-64)

Tras cualquier cambio que no deba tocar el audio, el PCM tiene que coincidir con esta tabla; lo comprueba `test_reference_audio_hashes` (`crates/qwen-engine/tests/engine_tests.rs`) con `QWEN_TTS_CHECK_REFERENCE_AUDIO=1 cargo test --workspace --release` en este equipo. Otra CPU x86-64 da otros hashes (las máquinas Windows y Linux de la CI los dan distintos en las cuatro frases), así que sin la variable la prueba se omite. Se mide en Windows x86_64 (Ryzen 7 5825U, ruta `avx2`, OpenBLAS 0.3.34 propio (`scripts/build-openblas.sh`)) con `-j 4 --seed 4` y la temperatura por defecto (0,325). El hash es el de `scripts/ci-common.sh` (`pcm_hash`): los 16 primeros dígitos hexadecimales del SHA-256 del PCM, sin la cabecera WAV de 44 bytes. Es el mismo en los cuatro modos (CLI, CLI `--stdout`, `/v1/tts` y `/v1/tts/stream`) y en todas las corridas. El idioma se indica con `-l` (`Spanish` o `English`).

- **Voz**: `default`, la voz incorporada en el binario (x-vector de 4096 bytes, SHA-256 `4c8a7679eb7eb6496db01f039b13f789f3a71e23bbcb58a5ec8a72bf4ed6e4e6`), sin `--load-voice`, con el modelo Base (`-d models/qwen3-tts-0.6b-base`).
- **Textos** (los de inglés son `TEXT_short` y `TEXT_long` de `scripts/ci-common.sh`):
  - inglés, corta: `Hello world, this is a quick baseline test.`
  - inglés, larga: `In this long synthesis benchmark, we measure sustained memory bandwidth, Talker latency, Code Predictor throughput, and streaming Speech Decoder stability across multiple seconds of continuous speech generation.`
  - español, corta: `Hola mundo, esta es una prueba rápida de referencia.`
  - español, larga: `En esta prueba de síntesis larga medimos el ancho de banda de memoria sostenido, la latencia del Talker, el rendimiento del Code Predictor y la estabilidad del decodificador de voz en streaming durante varios segundos de generación continua.`

| Voz | `-l` | Frase | Duración (s) | Hash del PCM |
|---|---|---|---|---|
| `default` | English | corta | 3,04 | `14f1f77c6c0e1435` |
| `default` | English | larga | 12,40 | `b04de85df9cfc3ed` |
| `default` | Spanish | corta | 3,04 | `9ffdf85cf9132e4a` |
| `default` | Spanish | larga | 14,64 | `04879f24f0c80827` |

Cambian esta tabla, y exigen regenerarla: el tamaño de bloque del prefill, cualquier cambio de redondeo en el Talker o el Code Predictor, el calendario de chunks, las rampas de borde, los frames que el Talker retiene tras el primer EOS, el descarte de la cadena de silencio inicial y el punto final que `plan_script` añade al texto. En ARM64 el audio se compara con la ruta escalar de la misma máquina (`QWEN_KERNELS=scalar`), no con esta tabla (§2.4).
