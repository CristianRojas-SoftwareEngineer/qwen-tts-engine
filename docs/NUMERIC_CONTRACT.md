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

- **Cero fallbacks en coma flotante**: todas las rutas (`AVX2` `maddubs` y escalar `i32`) acumulan en producto escalar entero.
- **Matriz `gate_up` intercalada**: la fila `2r` corresponde a `gate[r]` y la `2r+1` a `up[r]`. En modos cuantizados se cuantiza directamente desde `gate_bf16` y `up_bf16` sin materializar `gate_up_fused_bf16`.

### 1.3 Perfil `--quant-mixed` (`[INT8+Q4_0-FFN]`)

- **Talker (28 capas)**: `wq`, `wk`, `wv`, `wo`, `gate_up`, `down` en `Q4_0`; `text_embedding`, `text_projection`, `codec_embedding`, `codec_head` en `BF16`; norms en `F32`; caché KV en `BF16`.
- **Code Predictor (5 capas)**: `wq`, `wk`, `wv`, `wo` y 15 `lm_head` en `INT8` *(mantiene salida por `argmax_matvec_int8`, nunca en `Q4_0`)*; `gate_up` y `down` en `Q4_0`; `codec_embedding` en `BF16`; caché KV en `BF16`.

---

## 2. Hardware, alineación y kernels `AVX2`

1. **FTZ / DAZ por hilo (`ftz_on`)**: las activaciones cuantizadas entran con frecuencia en rango subnormal (penalización ~100×). El hilo principal (`ComputeRuntime`), cada hilo trabajador de `ComputePool` y los puntos de entrada cuantizados de `KernelDispatch` activan:
   - **x86-64**: `MXCSR |= 0x8040` (`FTZ` + `DAZ`).
   - **ARM64**: `FPCR |= (1 << 24)` (`FZ`).
2. **Alineación a línea de caché**: activaciones, KV cache, pesos cuantizados, `W_k` y workspaces del decodificador usan `AlignedBuf<64>`.
3. **Patrones `AVX2` (`qwen-kernels/src/simd/avx2.rs`)**:
   - **Prefetch**: en `int8` y `q4_0`, anticipa `5120` bytes (`QWEN_INT8_PF_DIST = 5120`) por fila procesada.
   - **Filas largas**: en `int8_matvec_slice` y `q4_0_matvec_slice`, procesa **1 fila por iteración** cuando `cols >= 2048` (`QWEN_LONG_ROW_COLS`), manteniendo acumulación idéntica bit a bit frente a 2 filas (`matmat` y `q4_0_matvec_qkv_slice` mantienen 2 filas).

---

## 3. Sensibilidades de punto flotante

| Componente | Regla crítica | Motivo |
|---|---|---|
| **Kernels SIMD** | FMA explícito (`_mm256_fmadd_ps`) | Evita divergencia observada sin contracción FMA en frases cortas. |
| **`SpeechEncoder` (RVQ)** | Distancia L2 (`||x - e||^2`) en **IEEE-754 estricto** | Reordenamientos agresivos rompen la búsqueda de vecino más cercano en RVQ. |
| **`convnext_mlp` (`SpeechDecoder`)** | **LayerNorm de 2 pasadas** (`mean`, luego `var`) | La variante de una pasada altera la señal respecto al streaming validado. |
| **Activación Snake (`SpeechDecoder`)** | `y = x + exp(-log_beta) * sin^2(exp(log_alpha) * x)` en `F32` | Los parámetros `{alpha, beta}` del checkpoint están en escala logarítmica. |
| **`causal_conv1d` / `conv_transpose1d`** | `conv_transpose1d`: 1 `BlasBackend::sgemm` + dispersión en orden ascendente de `k`; `causal_conv1d` (`k>1`): `k` `sgemm` acumulados sobre `W_k` `[k][out][in]` | Elimina copias `im2col` manteniendo `SNR >= 96–98 dB`. |
| **Atención con ventana deslizante (`SpeechDecoder`)** | Un único kernel `causal_attention_windowed` (*online softmax* + FMA) para streaming y decodificación completa; en streaming, consultas en coordenadas de la caché compactable (`q_offset = kv_len - kv_base`), que conserva `window - 1` posiciones | Streaming y completo comparten aritmética (`SNR >= 100 dB`); el bucle escalar previo difería en ±1 LSB de PCM16. |
| **`CodePredictor` prefill** | **`PREFILL2`** (`matmat(B=2)`) cuando las 5 capas están cuantizadas | Lee cada matriz una vez para posiciones `0` y `1` con resultado idéntico al secuencial. |

---

## 4. Invariantes de determinismo en generación y *streaming*

1. **Prefill completo**: cada petición arranca con `kv_len = 0` desde la posición `0` (sin reutilización parcial entre peticiones).
2. **Hilos BLAS congelados durante generación**: prefill usa `nt` hilos; antes de lanzar el hilo del decodificador se fija `gen_blas = max(1, nt - 1)` (o `QWEN_BLAS_GEN_THREADS`) y no cambia hasta terminar el drenaje final.
3. **Calendario fijo de fragmentos (streaming)**: primer chunk de `2` fotogramas (`160 ms`); siguientes `k = 0, 1, 2, ...` de `min(chunk_frames, 4 + 2 * (k / 8))`. En modo no streaming, tamaño fijo `10` (`DEFAULT_CHUNK_FRAMES`).
4. **Tiling y workspace del decodificador**: `DecoderWorkspace` reutiliza 8 ranuras crecientes por worker. En modo paralelo, los bloques residuales `up2` y `up3` (`SD_TILE_FROM_BLOCK = 2`) procesan por tramos de **`256 KB`** (`SD_TILE_BYTES = 262144`) durante toda la petición (incluido el drenaje final).

---

## 5. Matriz de tolerancias

| Prueba / Nivel | Comparación | Umbral obligatorio |
|---|---|---|
| **Kernels `int8`** (`--self-test`) | `matmat_int8(B=8)` vs `B × matvec_int8` | **`L2_rel == 0.0` (bit-idéntico)** |
| | `matvec_int8` vs referencia `f32` | `rel_L2 < 3e-2` |
| | `argmax_matvec_int8` vs referencia | Índice exacto o empate `< 0.02 * (\|max\| + 1e-3)` |
| **Kernels `q4_0`** (`--self-test`) | `matmat_q4_0(B=8)` vs `B × matvec_q4_0` | **`L2_rel <= 1e-6`** (`0.0` en `AVX2` y escalar) |
| | `matvec_q4_0` vs dequant `f32` ref | `rel_L2 < 3e-2` |
| **Kernels `bf16`** (`--self-test`) | `matvec_bf16` vs `f32` ref | `max_rel < 1e-2` |
| | `matmat_bf16(B=8)` vs `B × matvec_bf16` | `L2_rel < 1e-4` |
| **Speech Decoder aislado** | Streaming vs decodificación completa | **`SNR >= 100 dB`** |
| **Pipeline E2E** (misma semilla/params) | 2 corridas independientes | **Bit-idéntico** en CLI, `/v1/tts` y `/v1/tts/stream` |
