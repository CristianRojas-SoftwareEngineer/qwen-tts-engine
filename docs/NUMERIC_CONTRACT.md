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

### 1.2 Formatos `INT8`, `Q4_0` y `Q2_0`

| Formato | Layout por bloque / fila | Regla de cuantización | Uso en `0.6B` |
|---|---|---|---|
| **INT8** | `i8[rows, cols]` + `f32 scale[rows]` | Por fila: `s = absmax / 127.0`, `round_ties_even(w * 127.0 / absmax)` saturado a `[-128, 127]`. | `--int8`; atención y 15 `lm_head` en `--quant-mixed`. |
| **Q4_0** (`18 B / 32 pesos`) | `scale_f16` (`u16`) + `qs[16]` | **Mínimos cuadrados ponderados (`q4_0s1`)**: toma el elemento de mayor magnitud `vmax` (con signo), fija `isc = -8.0 / vmax`, `q[i] = clamp(round(v[i] * isc), -8, 7)` y reescala `s* = sum(v^2 * v * q) / sum(v^2 * q^2)`. Empaque intercalado: `qs[i] = (q[2i]+8) \| ((q[2i+1]+8) << 4)`. | `--int4`; Talker + FFN del CP en `--quant-mixed`. |
| **Q2_0** (`12 B / 32 pesos`) | `scale` (`f32`) + `qs[8]` | 4 niveles simétricos `{-1.5, -0.5, 0.5, 1.5} * scale`: `scale = absmax / 1.5`, `code = clamp(round_ties_even(v / scale + 1.5), 0, 3)`; 4 códigos/byte (`(i & 3) * 2`). | Solo `down_q2_rough` bajo `--roughness > 0`, mezclado en `down_proj` del CP: `proj = (1 - r) * proj + r * proj_q2`. |

- **Cero fallbacks en coma flotante**: todas las rutas (`AVX2` `maddubs`, NEON `SDOT` o `SMULL` y escalar `i32`) acumulan en producto escalar entero.
- **Matriz `gate_up` intercalada**: la fila `2r` corresponde a `gate[r]` y la `2r+1` a `up[r]`. En modos cuantizados se cuantiza directamente desde `gate_bf16` y `up_bf16` sin materializar `gate_up_fused_bf16`.

### 1.3 Perfil `--quant-mixed` (`[INT8+Q4_0-FFN]`)

- **Talker (28 capas)**: `wq`, `wk`, `wv`, `wo`, `gate_up`, `down` en `Q4_0`; `text_embedding`, `text_projection`, `codec_embedding`, `codec_head` en `BF16`; norms en `F32`; caché KV en `BF16`.
- **Code Predictor (5 capas)**: `wq`, `wk`, `wv`, `wo` y 15 `lm_head` en `INT8` *(mantiene salida por `argmax_matvec_int8`, nunca en `Q4_0`)*; `gate_up` y `down` en `Q4_0`; `codec_embedding` en `BF16`; caché KV en `BF16`.

---

## 2. Hardware, alineación y kernels SIMD

1. **FTZ / DAZ por hilo (`ftz_on`)**: las activaciones cuantizadas entran con frecuencia en rango subnormal (penalización ~100×). El hilo principal (`ComputeRuntime`), cada hilo trabajador de `ComputePool` y los puntos de entrada cuantizados de `KernelDispatch` activan:
   - **x86-64**: `MXCSR |= 0x8040` (`FTZ` + `DAZ`).
   - **ARM64**: `FPCR |= (1 << 24)` (`FZ`).
2. **Alineación a línea de caché**: activaciones, KV cache, pesos cuantizados, `W_k` y workspaces del decodificador usan `AlignedBuf<64>`.
3. **Patrones `AVX2` (`qwen-kernels/src/simd/avx2.rs`)**:
   - **Prefetch**: en `int8`, anticipa `5120` bytes (`QWEN_INT8_PF_DIST = 5120`) por fila procesada; en `q4_0`, 16 bloques (288 bytes). Anticipar 5 120 bytes también en `q4_0` se midió el 2026-11-18 (frase larga con carga): neutro o peor (RTF 0,94–1,00 → 0,97–1,01).
   - **Filas largas**: en `int8_matvec_slice` y `q4_0_matvec_slice`, procesa **1 fila por iteración** cuando `cols >= 2048` (`QWEN_LONG_ROW_COLS`), manteniendo acumulación idéntica bit a bit frente a 2 filas (`matmat` y `q4_0_matvec_qkv_slice` mantienen 2 filas).
4. **Kernels NEON (`qwen-kernels/src/simd/neon.rs`, aarch64)**: todas las entradas de la tabla, igual que AVX2. Q4_0 e INT8 (`matvec`, QKV fusionado y `matmat`) en dos variantes; el resto (cuantización de activaciones, normas RMS, BF16, atención, RoPE, conversiones, matvec `f32` del decoder y *snake*) en NEON base, con las mismas operaciones que AVX2 en el mismo orden: cada registro AVX2 de 8 `f32` es un par de registros NEON y la reducción final es la de `hsum256_ps`.
   - **Dos variantes con el mismo código**: `neon-dotprod` usa `SDOT` (emitido con `asm!`, porque `vdotq_s32` no es estable en Rust 1.96) y `neon`, multiplicación ampliada `SMULL` con sumas por pares, para núcleos ARMv8.0 sin `dotprod`. `KernelDispatch::for_caps` elige `neon-dotprod` si la CPU tiene `dotprod`; si no, `neon`. Las dos dan los mismos bits.
   - **Orden de suma de AVX2 en Q4_0**: por bloque, 8 sumas enteras de 4 productos (pesos sin desplazar, `0..15`) menos la suma de las 32 activaciones, conversión a `f32` y FMA en 8 carriles; reducción final en el orden de `hsum256_ps`, `((c0 + c4) + (c2 + c6)) + ((c1 + c5) + (c3 + c7))`. Mismas filas largas (`QWEN_LONG_ROW_COLS`) y mismos microkernels `matmat` de 8 columnas. Así NEON y AVX2 dan los mismos bits en estos kernels (prueba `test_simd_q4_0_matches_avx2_summation_order`, con la huella de la ruta AVX2).
   - **Mismos bits que AVX2** en todos los kernels que no usan `libm`: prueba `test_simd_float_kernels_match_avx2_summation_order`, con la huella de AVX2. La atención (`exp`) y el *snake* (`exp` de sus parámetros y `sin` de respaldo) dependen de la `libm` de cada plataforma.
   - **El audio de ARM sigue sin ser el de x86**: el prefill y las convoluciones del decoder usan BLAS (OpenBLAS o Accelerate), que redondea distinto en cada plataforma, y `exp`/`sin` vienen de su `libm`. En ARM el audio se compara con la ruta escalar de la misma máquina (`QWEN_KERNELS=scalar`), no por hash con x86.
5. **Reparto de filas entre hilos (`parallel_rows`)**: las `matvec`/`matmat` de 256 filas o más se reparten en `PARALLEL_ROWS_CHUNKS_PER_THREAD = 4` trozos por hilo con bordes múltiplos de 64 filas, que los hilos toman de un contador compartido. Cada fila se calcula con el mismo kernel y el mismo orden de suma, así que el resultado no depende del reparto: 1 hilo y 4 hilos dan los mismos bits (prueba `test_parallel_rows_partition_bit_exact`). Con la CPU ocupada, un hilo frenado ya no retrasa a los demás en la barrera.

---

## 3. Sensibilidades de punto flotante

| Componente | Regla crítica | Motivo |
|---|---|---|
| **Kernels SIMD** | FMA explícito (`_mm256_fmadd_ps`, `vfmaq_f32`) | Evita divergencia observada sin contracción FMA en frases cortas. |
| **`SpeechEncoder` (RVQ)** | Distancia L2 (`||x - e||^2`) en **IEEE-754 estricto** | Reordenamientos agresivos rompen la búsqueda de vecino más cercano en RVQ. |
| **`convnext_mlp` (`SpeechDecoder`)** | **LayerNorm de 2 pasadas** (`mean`, luego `var`) | La variante de una pasada altera la señal respecto al streaming validado. |
| **Activación Snake (`SpeechDecoder`)** | `y = x + exp(-log_beta) * sin^2(exp(log_alpha) * x)` en `F32` | Los parámetros `{alpha, beta}` del checkpoint están en escala logarítmica. |
| **`causal_conv1d` / `conv_transpose1d`** | `conv_transpose1d`: 1 `BlasBackend::sgemm` + dispersión en orden ascendente de `k`; `causal_conv1d` (`k>1`): `k` `sgemm` acumulados sobre `W_k` `[k][out][in]`, salvo la `conv1` k=7 de los bloques residuales que el *streaming* procesa por tramos (`b >= SD_TILE_FROM_BLOCK`, 192 y 96 canales): **im2col del tramo y una sola `sgemm`** con `K = 7 · canales` sobre `conv1_wcat` `[out][k · in]` | Sin `im2col` se ahorran copias con canales anchos; con 96–192 canales, `K = canales` rinde 1,7× peor por FLOP y el im2col del tramo (≤ 1,75 MiB) compensa. Cambio medido frente a la referencia vigente: SNR 104,7–105,8 dB, ±1 LSB. |
| **Atención con ventana deslizante (`SpeechDecoder`)** | Un único kernel `causal_attention_windowed` (*online softmax* + FMA) para streaming y decodificación completa; en streaming, consultas en coordenadas de la caché compactable (`q_offset = kv_len - kv_base`), que conserva `window - 1` posiciones | Streaming y completo comparten aritmética (`SNR >= 100 dB`); el bucle escalar previo difería en ±1 LSB de PCM16. |
| **`Talker` prefill** | Pesos `BF16` → `F32` por bloques de filas de **4 MiB** (`PREFILL_STAGE_ELEMS`) y una `sgemm` por bloque (`gate_up` intercalado por parejas de filas) | OpenBLAS redondea según el ancho del bloque y el muestreo autorregresivo amplifica cualquier diferencia hasta cambiar los códigos: **cambiar el tamaño de bloque cambia el audio** y exige una línea base nueva. |
| **`Talker`: `codec_head`** | Solo se calculan las filas que sobreviven a `apply_talker_logit_rules`: los códigos de audio `0..2048` y `CODEC_EOS` = 2150, este junto a su pareja 2151 (`talker_codec_logits`, camino de una petición) | Las reglas fijan a `-1e30` el resto de `2048..3072` sin mirar su valor; calcular la pareja par conserva el emparejamiento de filas del kernel BF16 y da los mismos bits que el cálculo completo (pruebas `pipeline::tests`). El camino por lotes sigue calculando las 3 072 filas. |
| **`CodePredictor`: pasadas de 2 posiciones** | **`cp_step2(pos)`** (`matmat(B=2)`) cuando las 5 capas están cuantizadas: posiciones `0`–`1` y, después, **verificación especulativa de 1 adivinanza**. Tras `c[g]` se adivina `ĉ = argmax(lm_heads[g+1](h[g+1]))` y una pasada calcula las posiciones `g+2` (entrada `c[g]`) y `g+3` (entrada `ĉ`); se acepta la segunda columna solo si el código real `c[g+1]` (`argmax` sobre la columna 0) es `ĉ`, y si no se descarta (su K/V se sobrescribe antes de leerse) | Cada matriz se lee una vez para las dos columnas; cada código sigue siendo el `argmax` de la misma cabeza sobre el mismo estado, y cada columna es bit-idéntica a un `cp_transformer_step` (pruebas `code_predictor::tests`, `--batch-test` y PCM idéntico al de antes de introducirla). Pasadas de pesos por fotograma: 15 → ~11,9 (acierta el 28–31 %). |

---

## 4. Invariantes de determinismo en generación y *streaming*

1. **Prefill completo**: cada petición arranca con `kv_len = 0` desde la posición `0` (sin reutilización parcial entre peticiones).
2. **Hilos BLAS congelados durante generación**: prefill usa `nt` hilos; antes de lanzar el hilo del decodificador se fija `gen_blas = max(1, nt - 1)` (o `QWEN_BLAS_GEN_THREADS`) y no cambia hasta terminar el drenaje final.
3. **Calendario fijo de fragmentos (con y sin streaming)**: primer chunk de `2` fotogramas (`160 ms`); siguientes `k = 0, 1, 2, ...` de `min(chunk_frames, 4 + 2 * (k / 8))`, con `chunk_frames = 10` (`DEFAULT_CHUNK_FRAMES`) salvo que la petición lo cambie. Los dos modos usan el mismo calendario porque las fronteras de fragmento fijan el redondeo del decoder: así el PCM es idéntico. **Decisión del 2026-11-19**:
- **Coste aceptado**: sin *streaming* se renuncia a los bloques fijos de 10 heredados del motor original en C. El decoder trabaja un 4–17 % más y el RTF sube de +0,00 a +0,01 (A/B intercalado).
- **Por qué no bloques de 10 en todos los modos**: retrasaría el primer audio del *streaming* (estimado: ~450 → ~850 ms sin carga). La subida gradual del calendario evita cortes con RTF cercano a 1.
- **Para recuperar esa eficiencia sin perder la identidad entre modos**: haría falta un decoder cuyo redondeo no dependa del tamaño del fragmento.
4. **Tiling y workspace del decodificador**: `DecoderWorkspace` reutiliza 8 ranuras crecientes por worker. En modo paralelo, los bloques residuales `up2` y `up3` (`SD_TILE_FROM_BLOCK = 2`) procesan por tramos de **`256 KB`** (`SD_TILE_BYTES = 262144`) durante toda la petición (incluido el drenaje final).
5. **Rampas de borde**: 5 ms de entrada y 40 ms de salida, cada una acotada a la mitad del audio, aplicadas después de la ganancia (`apply_gain`) y antes de la conversión a PCM16. El WAV completo las aplica en `encode_wav_bytes`. Las salidas en *streaming* (`--stream`, `--stdout`, `/v1/tts/stream`, también por lotes) las aplican con `StreamEdgeRamp`, que retiene los últimos 40 ms hasta el fragmento siguiente o el cierre y no emite nada hasta reunir 80 ms. Ambos caminos usan las mismas operaciones en el mismo orden (prueba `audio::tests`).

---

## 5. Matriz de tolerancias

| Prueba / Nivel | Comparación | Umbral obligatorio |
|---|---|---|
| **Kernels `int8`** (`--self-test`) | `matmat_int8(B=8)` vs `B × matvec_int8` | **`L2_rel == 0.0` (bit-idéntico)** |
| | `matvec_int8` vs referencia `f32` | `rel_L2 < 3e-2` |
| | `argmax_matvec_int8` vs referencia | Índice exacto o empate `< 0.02 * (\|max\| + 1e-3)` |
| **Kernels `q4_0`** (`--self-test`) | `matmat_q4_0(B=8)` vs `B × matvec_q4_0` | **`L2_rel <= 1e-6`** (`0.0` en `AVX2` y escalar) |
| | `matvec_q4_0` vs dequant `f32` ref | `rel_L2 < 3e-2` |
| **Rutas SIMD frente a la escalar** (`--self-test`, cada ruta que pueda ejecutar la CPU: `avx2`, `neon`, `neon-dotprod`) | `matvec_int8`, `matmat_int8(B=3, 8)` | **Bit-idéntico** |
| | `matvec_q4_0`, `qkv_q4_0`, `matmat_q4_0(B=3, 8)` | **`L2_rel <= 1e-6`** (otro orden de suma; AVX2 y NEON dan ≤ 3,5e-7) |
| | `quantize_act_int8`, `bf16_accum_f32`, `bf16_to_f32_vec`, `f32_to_bf16_vec`, suma residual de `rms_norm_residual` | **Bit-idéntico** |
| | normas RMS, `matvec_bf16`, `matmat_bf16(B=3, 8)`, atención `f32` (con y sin ventana) y BF16 (en serie y repartida), RoPE, `f32_matvec_accum`, *snake* | **`L2_rel <= 1e-6`** (AVX2 da ≤ 6,2e-7); `argmax_bf16`, índice igual o empate |
| **Kernels `bf16`** (`--self-test`) | `matvec_bf16` vs `f32` ref | `max_rel < 1e-2` |
| | `matmat_bf16(B=8)` vs `B × matvec_bf16` | `L2_rel < 1e-4` |
| **Speech Decoder aislado** | Streaming vs decodificación completa | **`SNR >= 100 dB`** |
| **Pipeline E2E** (misma semilla/params) | 2 corridas independientes | **Bit-idéntico** entre corridas y entre los cuatro modos: CLI, CLI `--stream`, `/v1/tts` y `/v1/tts/stream` dan el mismo PCM (mismo calendario de fragmentos y mismas rampas de borde, invariantes 3 y 5). Comprobado en 12 combinaciones de voz, idioma y longitud, y es igual con y sin carga. |

### Audio de referencia (x86-64)

Tras cualquier cambio que no deba tocar el audio, el PCM tiene que coincidir con esta tabla. Se mide en Windows x86_64 (Ryzen 7 5825U, ruta `avx2`, OpenBLAS de MSYS2 UCRT64) con `--quant-mixed -j 4 --seed 4` y la temperatura por defecto (0,325). El hash es el de `scripts/ci-comun.sh` (`pcm_hash`): los 16 primeros dígitos hexadecimales del SHA-256 del PCM, sin la cabecera WAV de 44 bytes. Es el mismo en los cuatro modos (CLI, CLI `--stream`, `/v1/tts` y `/v1/tts/stream`), en todas las corridas y con y sin carga.

- **Voces**: `ryan` y `vivian`, presets del modelo CustomVoice (`-s`); `default`, voz clonada del modelo Base a partir de `assets/Speech Sampler.wav` (perfil `default.qvoice` sin referencia ICL, SHA-256 `02488C75BBCFA749…`, cargado con `--load-voice … --graft`). El audio y el perfil no se versionan.
- **Textos** (los de inglés son `TEXT_short` y `TEXT_long` de `scripts/ci-comun.sh`):
  - inglés, corta: `Hello world, this is a quick baseline test.`
  - inglés, larga: `In this long synthesis benchmark, we measure sustained memory bandwidth, Talker latency, Code Predictor throughput, and streaming Speech Decoder stability across multiple seconds of continuous speech generation.`
  - español, corta: `Hola mundo, esta es una prueba rápida de referencia.`
  - español, larga: `En esta prueba de síntesis larga medimos el ancho de banda de memoria sostenido, la latencia del Talker, el rendimiento del Code Predictor y la estabilidad del decodificador de voz en streaming durante varios segundos de generación continua.`

| Voz | Idioma | Frase | Duración (s) | Hash del PCM |
|---|---|---|---|---|
| `ryan` | en | corta | 4,48 | `5483a06201f29d02` |
| `ryan` | en | larga | 15,60 | `e5be5a8e2d4990e8` |
| `ryan` | es | corta | 5,28 | `fcbd68e794d54409` |
| `ryan` | es | larga | 16,72 | `31f57166277afec4` |
| `vivian` | en | corta | 3,92 | `307ac8884c8f7dd0` |
| `vivian` | en | larga | 15,04 | `6a240edcbb34b2b8` |
| `vivian` | es | corta | 4,08 | `759f0ba666da7769` |
| `vivian` | es | larga | 16,64 | `1ebabf1a687e194b` |
| `default` | en | corta | 2,48 | `43731630d57d8880` |
| `default` | en | larga | 12,40 | `1ba4589f273942be` |
| `default` | es | corta | 4,48 | `848092132a0a9547` |
| `default` | es | larga | 14,64 | `b9adc5cf06585e61` |

Cambian esta tabla, y exigen regenerarla: el tamaño de bloque del prefill, cualquier cambio de redondeo en el Talker o el Code Predictor, el calendario de fragmentos y las rampas de borde. En ARM64 el audio se compara con la ruta escalar de la misma máquina (`QWEN_KERNELS=scalar`), no con esta tabla (§2.4).
