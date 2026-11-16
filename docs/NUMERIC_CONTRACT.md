# Contrato Numérico, Cuantización y Determinismo (`NUMERIC_CONTRACT.md`)

Fija las leyes aritméticas, de cuantización, de memoria y de streaming que el motor Rust debe cumplir para preservar la calidad de audio y el RTF del motor C (`qwen3-tts/`).

---

## 1. Cuantización y Aritmética Entera Única

### 1.1 Cuantizador único de activaciones (`quantize_act_int8`)
Toda multiplicación cuantizada (`int8` y `q4_0`, tanto `matvec` como cada columna de `matmat` y `argmax`) debe cuantizar la activación `x[0..n]` con la **misma numérica**:
1. `amax = max(|x_i|)`. Si `amax == 0.0`, `qx = [0; n]` y `scale = 0.0`.
2. `inv = 127.0 / amax`, `scale = amax / 127.0`.
3. `qx[i] = clamp(round_half_even(x[i] * inv), -128, 127)` (`lrintf` / `cvtps_epi32` / `vcvtnq_s32_f32`).
4. Tamaño máximo de vector en pila: `QWEN_QACT_MAX = 8192`. Toda matriz cuantizable debe cumplir en carga `cols > 0 && cols <= 8192 && cols % 32 == 0`.

### 1.2 Formatos de pesos
| Formato | Layout por bloque / fila | Regla de cuantización | Uso en `0.6B` |
|---|---|---|---|
| **INT8** | `int8_t[rows, cols]` + `float scale[rows]` | Por fila: `s = absmax / 127.0`, `round(w * 127.0 / absmax)` saturado a `[-128, 127]`. | `--int8`; atención + 15 `lm_head` en `--quant-mixed`. |
| **Q4_0** (`18 B / 32 pesos`) | `scale_f16` (IEEE fp16 `u16`) + `qs[16]` | **Mínimos cuadrados ponderados (`q4_0s1`)**: mapea el máximo con signo `vmax` a `-8`, redondea, reescala `s* = sum(v^2 * v * q) / sum(v^2 * q^2)`. Empaque **intercalado**: `qs[i] = (q[2i]+8) \| ((q[2i+1]+8) << 4)`. | `--int4`; Talker + FFN del CP en `--quant-mixed`. |
| **Q2_0** (`12 B / 32 pesos`) | `float scale` (`f32`) + `qs[8]` | 4 niveles simétricos `{-1.5, -0.5, 0.5, 1.5} * scale`, `scale = absmax / 1.5`; 4 códigos/byte (`(i%4)*2`). | Solo `down_q2_rough` bajo `--roughness > 0`. |

- **Prohibido usar fallbacks en coma flotante**: todas las ISA (AVX2 `maddubs`, VNNI `dpbusd`, ARM SDOT `vdotq_s32`, escalar `i32`) acumulan en producto escalar entero.
- **Matriz `gate_up` intercalada**: la fila `2r` corresponde a `gate[r]` y la `2r+1` a `up[r]`. En modos cuantizados se cuantiza directamente desde `gate_bf16` y `up_bf16` sin materializar `gate_up_fused_bf16`.

### 1.3 Perfil de producción `--quant-mixed` (`[INT8+Q4_0-FFN]`)
- **Talker (28 capas)**: `wq`, `wk`, `wv`, `wo`, `gate_up`, `down` en `Q4_0`; `text_embedding`, `text_projection`, `codec_embedding`, `codec_head` en `BF16`; norms en `F32`; caché KV en `BF16`.
- **Code Predictor (5 capas)**: `wq`, `wk`, `wv`, `wo` y 15 `lm_head` en `INT8` *(nota: `lm_head` argmax mantiene la activación/salida vía `qwen_matvec_int8`, nunca degradar a Q4_0)*; `gate_up` y `down` en `Q4_0`; `codec_embedding` en `BF16`; caché KV en `BF16`.

---

## 2. Requisitos de Hardware, Memoria y Kernels

1. **FTZ / DAZ obligatorio por hilo (`qwen_ftz_on`)**:
   - Las activaciones tras cuantización INT8 caen frecuentemente en rango subnormal (penalización ~100×).
   - Todo hilo trabajador del pool (y el hilo principal) debe activar **una vez al arrancar**:
     - **x86_64**: `MXCSR |= 0x8040` (bits FTZ=15 y DAZ=6).
     - **ARM64**: `FPCR |= (1 << 24)` (bit FZ).
2. **Alineación a línea de caché**: Todo buffer de activación, KV cache, pesos cuantizados, `W_k` y workspace del decodificador debe alinearse a **64 bytes** (`AlignedBuf<64>`).
3. **Patrones de lectura obligatorios en AVX2** (`DIVERGENCIAS.md`):
   - **Prefetch**: en `int8` y `q4_0`, anticipar `5120` bytes por delante (`QWEN_INT8_PF_DIST = 5120`) por fila procesada.
   - **Filas largas**: en `qwen_matvec_int8` y `qwen_matvec_q4_0` sobre AVX2, usar núcleo de **1 fila por iteración** cuando `cols >= 2048` (`QWEN_LONG_ROW_COLS`), conservando acumulación idéntica bit a bit frente a 2 filas (`matmat` y `QKV` siguen en 2 filas).

---

## 3. Sensibilidades de Punto Flotante (`Talker`, `Decoder`, `Encoder`)

| Componente | Regla crítica verificada en C | Por qué importa |
|---|---|---|
| **Global / Kernels** | Usar FMA explícito en kernels SIMD (`_mm256_fmadd_ps` / `vmlaq_f32`) | El `Makefile` documentó que `-ffp-contract=off -fno-fast-math` corrompía frases cortas (`dub`). |
| **`SpeechEncoder` (RVQ)** | Mantener cálculo de distancia L2 (`||x - e||^2`) en **IEEE-754 estricto** (sin reordenamientos agresivos) | En `Makefile:281`, `-ffast-math` rompió la búsqueda de vecino más cercano. |
| **`convnext_mlp` (`SpeechDecoder`)** | **LayerNorm de 2 pasadas** (`mean`, luego `var = sum((x - mean)^2) / C`) | Una pasada (`E[x^2] - mean^2`) altera el audio respecto al streaming validado (`DIVERGENCIAS.md`). |
| **`causal_conv1d` / `conv_transpose1d`** | `conv_transpose1d`: 1 único `cblas_sgemm` + dispersión en orden ascendente de `k`; `causal_conv1d` (`k>1`): `k` `cblas_sgemm` acumulados sobre entrada con relleno izquierdo sobre `W_k` pre-reordenado `[k][out][in]` | Elimina copias `im2col`; mantiene SNR ≥ 96–98 dB vs la implementación original. |
| **`CodePredictor` prefill** | Usar siempre **`PREFILL2`** (`matmat(B=2)`) cuando las 5 capas están en `int8` y/o `q4_0` en cualquier mezcla | Lee cada matriz 1 vez para posiciones 0 y 1 con resultado idéntico al secuencial. |

---

## 4. Invariantes de Determinismo en Streaming y Generación

1. **Prefill siempre completo**: cada síntesis arranca con `kv_len = 0` desde la posición 0 (sin reutilización parcial entre peticiones).
2. **Hilos BLAS congelados durante la generación**:
   - Prefill usa `blas_threads = -j`.
   - Antes de lanzar el hilo del decodificador se fija `gen_blas = (j > 1) ? j - 1 : 1` (o `QWEN_BLAS_GEN_THREADS`) y **no se modifica** hasta terminar el drenaje final (cambiar hilos BLAS a mitad altera el orden de reducción de `sgemm`).
3. **Calendario fijo de fragmentos crecientes (streaming)**:
   - Primer chunk: `2` fotogramas (`160 ms`).
   - Chunks siguientes `k = 0, 1, 2, ...`: `min(chunk_frames, 4 + 2 * (k / 8))` (por defecto `chunk_frames = 10` → `2`, luego `4×8`, `6×8`, `8×8`, `10...`).
   - En modo no-streaming (`/v1/tts` o CLI normal), tamaño constante = `10` (mismo `DEFAULT_CHUNK_FRAMES` reiniciado por petición).
4. **Tiling y Workspace del decodificador**:
   - `DecoderWorkspace`: 8 ranuras crecientes reutilizadas (`A, B, RES, EXT, FULL, CT, XP, PW1`), aisladas por worker.
   - Cuando el hilo del decodificador corre en paralelo con la generación, las 3 unidades residuales de `up2` y `up3` (`SD_TILE_FROM_BLOCK = 2`) procesan por tramos de **`256 KB`** (`SD_TILE_BYTES = 262144`) durante **toda** la petición (incluido el drenaje final) para no depender de la temporización.

---

## 5. Matriz de Tolerancias y Criterios de Validación

| Prueba / Nivel | Comparación | Métrica y umbral obligatorio |
|---|---|---|
| **Kernels `int8`** (`--self-test`) | `matmat_int8(B=8)` vs `B × matvec_int8` | **`L2_rel == 0.0` (exacto bit a bit)** |
| | `matvec_int8` vs referencia f32 | `rel_L2 < 3e-2` (~`7e-3` típico) |
| | `argmax_matvec_int8` vs referencia | Índice exacto o empate `< 0.02 * (\|max\| + 1e-3)` |
| **Kernels `q4_0`** (`--self-test`) | `matmat_q4_0(B=8)` vs `B × matvec_q4_0` | **`L2_rel <= 1e-6`** (`0.0` en AVX2 y escalar) |
| | `matvec_q4_0` vs dequant f32 ref | `rel_L2 < 3e-2` |
| **Kernels `bf16`** (`--self-test`) | `matvec_bf16` vs f32 ref | `max_rel < 1e-2` |
| | `matmat_bf16(B=8)` vs `B × matvec_bf16` | `L2_rel < 1e-4` (o `< 1e-2` si hardware trunca activación a bf16) |
| **Speech Decoder aislado** | `test-decoder-tool` con tiling vs sin tiling / C vs Rust | **`SNR >= 100 dB`** (diferencia máx. ±1 LSB en s16le) |
| **Pipeline E2E** (misma semilla y params) | Rust vs sí mismo (2 corridas) | **MD5 bit-idéntico** en CLI, `/v1/tts` y `/v1/tts/stream` |
| **Pipeline E2E** (C vs Rust) | Comparativa A/B (`0.6B`) | `WER` equivalente, audio coherente sin truncamientos y **`mel-corr >= 0.95`** (o ≥ `0.98` en misma ISA/BLAS) |
