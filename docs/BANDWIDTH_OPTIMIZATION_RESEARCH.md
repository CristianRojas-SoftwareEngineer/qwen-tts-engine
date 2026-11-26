# Investigación de optimización del ancho de banda de memoria

> Dossier de investigación. No es un registro de decisión: recoge la contabilidad de bytes del
> motor, el diagnóstico de los kernels, el catálogo de oportunidades y el plan de hipótesis y
> experimentos con el que se construirá la propuesta de implementación. Las decisiones que se
> adopten a partir de aquí van a `docs/adr/`.

## Tabla de contenidos

- [0. Alcance y método](#0-alcance-y-método)
- [1. Contabilidad de bytes por frame](#1-contabilidad-de-bytes-por-frame)
- [2. Mecanismos que explican el consumo](#2-mecanismos-que-explican-el-consumo)
- [3. La causa raíz: el kernel Q4_0 está limitado por ALU](#3-la-causa-raíz-el-kernel-q4_0-está-limitado-por-alu)
- [4. Duplicaciones y reutilización](#4-duplicaciones-y-reutilización)
- [5. El bloque grande: oportunidades sobre el 91 %](#5-el-bloque-grande-oportunidades-sobre-el-91-)
- [6. El resto: decoder, KV, cabezas y prefill](#6-el-resto-decoder-kv-cabezas-y-prefill)
- [7. Escalado con la longitud de la locución](#7-escalado-con-la-longitud-de-la-locución)
- [8. Catálogo consolidado de oportunidades](#8-catálogo-consolidado-de-oportunidades)
- [9. Hipótesis y experimentos](#9-hipótesis-y-experimentos)
- [10. Protocolo de medición](#10-protocolo-de-medición)
- [11. Consolidación: propuesta por olas](#11-consolidación-propuesta-por-olas)
- [12. Riesgos y contrato numérico](#12-riesgos-y-contrato-numérico)
- [13. Registro de experimentos (2026-11-25)](#13-registro-de-experimentos-2026-11-25)
  - [13.8 Perfil interno del decoder por etapas](#138-perfil-interno-del-decoder-por-etapas)
  - [13.9 H4 — tiles más grandes para el GEMM de `conv1` (refutado)](#139-h4--tiles-más-grandes-para-el-gemm-de-conv1-refutado)
  - [13.10 Hipótesis descartadas sin medir (por el contrato numérico)](#1310-hipótesis-descartadas-sin-medir-por-el-contrato-numérico)
  - [13.11 Viabilidad de INT8 en `conv1` (refutada por velocidad; numéricamente correcta)](#1311-viabilidad-de-int8-en-conv1-refutada-por-velocidad-numéricamente-correcta)
  - [13.12 Verificación del *batching* intra-síntesis en el CP](#1312-verificación-del-batching-intra-síntesis-en-el-cp)
  - [13.13 Revisión bibliográfica: especulación RVQ en Qwen3-TTS](#1313-revisión-bibliográfica-especulación-rvq-en-qwen3-tts)
  - [13.14 Factibilidad de entrenamiento en CPU](#1314-factibilidad-de-entrenamiento-en-cpu)
- [Anexo A. Aritmética de los tamaños](#anexo-a-aritmética-de-los-tamaños)

---

## 0. Alcance y método

**Restricciones del trabajo**: sin cambios de hardware, sin reentrenar el modelo, sin procesamiento
en batch. Solo optimizaciones programáticas sobre operaciones, precisión de datos y complejidad
computacional.

**Máquina de referencia**: 2 × 16 GiB DDR4-3200 en `CHANNEL A` y `CHANNEL B` (dual channel, 32 GiB;
ancho de banda teórico 51,2 GB/s, efectivo ~30-35 GB/s), APU de 8 núcleos / 16 hilos. La huella de
RAM del motor (pico ~1,35 GiB comprometidos) es irrelevante: aquí el recurso escaso es el bus, no
la capacidad.

**Fuentes**: código de `crates/qwen-engine`, `crates/qwen-kernels` y `crates/qwen-core`; tamaños
exactos de tensor extraídos de las cabeceras de los `safetensors`; medidas publicadas en
`docs/adr/005-cpu-inference-optimizations.md` y `docs/NUMERIC_CONTRACT.md`.

**Convención de cifras**: MiB = 2²⁰ bytes. Todas las cifras de bytes por frame son cálculos
derivados del código, no medidas de contador de hardware. Las de tiempo y RTF citadas como
"medidas" provienen de ADR-005. Las proyecciones de RTF son estimaciones del modelo bytes/tiempo y
se marcan como tales.

**Corrección respecto al cálculo inicial**: el bloque de pesos del Talker + Code Predictor +
cabezas es **1 055 MiB de 1 159 MiB = 91,0 %**, no el 87 % que se citó en el análisis previo (fue un
redondeo). El resto, **104 MiB = 9,0 %**, es lo que se analiza en §6.

---

## 1. Contabilidad de bytes por frame

Locución de referencia: ~400 frames (32 s de audio), posición media 200.

### 1.1 Camino productor (serial: Talker → CP por frame)

| # | Concepto | Cálculo | MiB/frame |
|---|---|---|---|
| 1 | Talker, matrices INT8, 28 capas | `wq 2 + wk 1 + wv 1 + wo 1 + gate_up 6 + down 3 = 14 MiB` × 28 | **392,0** |
| 2 | Talker `codec_head`, solo filas válidas | `2048 × 1024 × 2 B` (BF16) | 4,0 |
| 3 | Talker KV (atención) | `28 capas × 4 KiB × pos` con pos = 200 | 22,4 |
| 4 | CP transformer, 5 capas | `10,06 MiB/pasada × 11,9 pasadas` | **614,0** |
| 5 | CP `lm_heads` | `~25 lecturas × 2 MiB` | 50,0 |
| | **Subtotal productor** | | **1 082** |

Composición de una pasada del CP (5 capas, por capa):

| Matriz | Formato | MiB |
|---|---|---|
| `wq` (2048 × 1024) | INT8 | 2,000 |
| `wk`, `wv` (1024 × 1024) | INT8 | 1,000 c/u |
| `wo` (1024 × 1024) | INT8 | 1,000 |
| `gate_up` (6144 × 1024) | Q4_0 (18 B / 32 pesos) | 3,375 |
| `down` (1024 × 3072) | Q4_0 | 1,688 |
| | **por capa** | **10,06** |

### 1.2 Camino decodificador (hilo paralelo)

| # | Concepto | Cálculo | MiB/frame |
|---|---|---|---|
| 6 | Pesos no tiled | `419 MiB / chunk_frames (10)` | 41,9 |
| 7 | Pesos tiled (bloques de upsample ≥ 2) | `13,5 MiB × 19 tiles + 3,4 MiB × 29 tiles`, por chunk de 10 | 35,4 |
| | **Subtotal decoder** | | **77,3** |

### 1.3 Coste amortizado por petición

| Concepto | Bytes | Equivalente en una locución de 37 frames | En una de 400 |
|---|---|---|---|
| Prefill (28 capas en BF16) | 1 400 MiB | 37,8 MiB/frame | 3,5 MiB/frame |
| Codificador de locutor (voz x-vector) | 16,9 MiB | 0,5 MiB/frame | 0,04 MiB/frame |

El prefill es, en textos cortos, **mayor que todo el camino decodificador junto**.

### 1.4 Total

```
1 082 (productor) + 77 (decoder) = 1 159 MiB/frame
```

Reparto: Talker 34 %, CP 57 %, decoder 6,7 %, KV 1,9 %, cabezas 4,3 %.

---

## 2. Mecanismos que explican el consumo

**M1. Streaming de pesos sin reutilización posible (Talker).** El Talker emite un token por frame y
cada frame tiene una activación distinta. Cada matriz INT8 se lee exactamente una vez por frame
(`talker.rs:296-390`) y no hay ninguna lectura repetida. Los 392 MiB/frame son, por tanto,
irreducibles sin cambiar bytes o sin batchear frames (bloqueado por la serialización
`Talker(t) → CP(t) → Talker(t+1)`).

**M2. Relectura del CP (×11,9).** El CP predice 15 codebooks de forma autorregresiva: cada codebook
necesita una pasada completa de 5 capas. La verificación especulativa de 1 adivinanza empaqueta 2
posiciones por pasada (`cp_step2`, `matmat(B=2)`) y reduce de 15 a ~11,9 pasadas
(`NUMERIC_CONTRACT.md:64`), con una tasa de acierto del 28-31 %. Es la mejor optimización de
reutilización que el motor ya tiene.

**M3. KV creciente.** Cada frame lee `112 KiB × pos` de K/V en BF16. El coste por frame crece
linealmente con la posición y el total de la locución crece cuadráticamente (§7).

**M4. Relectura del decoder por chunk y por tile.** El decoder procesa el chunk completo en una
pasada (`conv_decoder_forward_streaming`, `speech_decoder.rs:615-624`), pero los bloques residuales
de upsample ≥ 2 se parten en tiles y **cada tile relee los pesos del bloque**
(`cs_res_tiled`, `:539-602`).

**M5. Prefill íntegro en BF16.** `talker_prefill` recorre las 28 capas en BF16
(`talker.rs:141-143`), 1,4 GiB por petición, mientras las copias INT8 (422 MiB) están residentes y
ociosas.

---

## 3. La causa raíz: el kernel Q4_0 está limitado por ALU

> **Corrección del 2026-11-25 (ver §13).** Esta sección partía de comparar GB/s entre formatos de
> densidad distinta, lo que no es una comparación de velocidad. La medición directa por kernel
> muestra que Q4_0 y INT8 rinden **casi lo mismo por peso** (~25 Gpesos/s), y que el motor entero
> está en ese techo. Las proyecciones de §5.1 y §5.2 quedan corregidas en §13.6.

### 3.1 El dato que ordena todo

| Fase | Bytes/frame | Tiempo (ADR-005, con carga) | GB/s efectivos | Formato |
|---|---|---|---|---|
| Talker | 392 MiB | ~24 ms | **~16,3** | 100 % INT8 |
| CP | 613 MiB | 45,6 ms | **~13,4** | 50 % INT8 + 50 % Q4_0 |

Misma máquina, mismo bus, misma pool. La única diferencia estructural en el bucle caliente es que
el CP tiene la mitad de sus bytes en Q4_0 (5,06 de 10,06 MiB por capa: `gate_up` 3,375 + `down`
1,688 frente a `wq` 2 + `wk`/`wv`/`wo` 1 cada una).

Esto explica un resultado que parecía contradictorio y que ya está en el repo: *todo el Talker en
Q4_0 reduce 44 % los bytes pero el RTF solo baja de 0,65 a 0,62*. No es que 4 bits no sirvan: es que
**el decodificador Q4_0 está limitado por ALU, no por memoria**, así que el ahorro de bytes se paga
en cómputo.

### 3.2 Anatomía del bloque

`q4_0_block` (`avx2.rs:541-554`), por bloque de 32 pesos (18 B):

| Operación | µops |
|---|---|
| carga 16 B | 1 |
| `and` ×2 + `srli` | 3 |
| `unpacklo/hi` + `inserti128` | 3 |
| `maddubs` + `madd` + `sub` | 3 |
| `cvtepi32_ps` | 1 |
| `f16_to_f32` escalar, con ramas (`quant.rs:38-58`) | 6-8 |
| `set1_ps` | 1-2 |
| `fmadd` | 1 |
| prefetch + bucle | 2-3 |
| **Total** | **~21-24 por 18 B** |

`int8_matvec_1row` (`avx2.rs:354-365`), por 32 pesos (32 B): ~9-10 µops.

| | µops/B | B/ciclo a 4 µops/ciclo | GB/s a 3,5 GHz |
|---|---|---|---|
| INT8 | 0,30 | 13,3 | **~46** |
| Q4_0 | **1,25** | 3,2 | **~11** |

Los ~11 GB/s predichos para Q4_0 coinciden con los 13,4 GB/s medidos del CP. La hipótesis cuadra.

> **Refutado en §13.3.** La coincidencia era una coincidencia: 13,4 GB/s de Q4_0 y 26 Gpesos/s son
> la misma cifra expresada en unidades distintas. El modelo de µops por byte no sirve para comparar
> formatos con densidad distinta.

### 3.3 El prefetch del Q4_0 es 18× más corto

```rust
// avx2.rs:582, :609-610, :674
_mm_prefetch::<_MM_HINT_T0>(r0.add(b + 16) as *const i8);
```

`b + 16` bloques × 18 B = **288 bytes** de adelanto, frente a `QWEN_INT8_PF_DIST = 5120` del INT8
(`quant.rs:10`). A 13 GB/s, 288 B son 22 ns: por debajo de la latencia de DRAM (~60-90 ns). Ese
prefetch no cubre latencia.

Además usa `_MM_HINT_T0` sobre un stream de 392-613 MiB que nunca se reutiliza, mientras el camino
BF16 ya usa `_MM_HINT_NTA` (`avx2.rs:263-266`). La inconsistencia es una oportunidad gratuita.

> **Refutado en §13.4.** Llevar el adelanto a 4 608 B no cambia nada (razón 0,998 en 6 pares A/B):
> el *streamer* de hardware ya cubre un recorrido perfectamente secuencial.

---

## 4. Duplicaciones y reutilización

### 4.1 Duplicaciones confirmadas

| ID | Qué se duplica | Dónde | Coste |
|---|---|---|---|
| D1 | Pesos de tramos residuales del decoder, una vez por tile | `speech_decoder.rs:539-602` | ~35 MiB/frame |
| D2 | Pesos no tiled del decoder, una vez por chunk | `speech_decoder.rs:615-624` | 41,9 MiB/frame |
| D3 | Cada cabeza KV, una vez por cada cabeza de consulta que la comparte (GQA, 16/8) | `avx2.rs:1021-1034` | 11 MiB/frame a 400 frames; 56 a 2 000 |
| D4 | Cada `lm_head` del CP, dos veces por paso (adivinanza y verificación) | `code_predictor.rs:400`, `:412` | ~18 MiB/frame redundantes |

D4 **no es fusionable**: la adivinanza usa `h[g+1]` y la verificación `h[g+2]`, y son secuenciales.
Sí es sustituible la fuente de la adivinanza (ver H15).

### 4.2 Reutilización ya explotada (no tocar)

| Mecanismo | Dónde | Ahorro ya obtenido |
|---|---|---|
| `matmat(B=2)` con verificación especulativa | `cp_step2` | 15 → 11,9 pasadas (~125 MiB/frame) |
| QKV fusionado en un barrido | `int8_qkv` (`model.rs:67`) | −2 despachos y −1 lectura de activación por capa |
| `gate_up` entrelazado | `TalkerLayerWeights.gate_up` | gate y up en una lectura |
| `codec_head` solo con filas válidas | `pipeline.rs:83-96` | −2 MiB/frame |
| Softmax online de una pasada | `avx2.rs:1032-1046` | K/V se lee una sola vez |
| Prefill por bloques de 4 MiB en L3 | `talker.rs:20` | los pesos convertidos no vuelven a DRAM |
| Decoder batcheado por chunk | `speech_decoder.rs:624` | 10 frames por lectura de pesos |

**Propiedad clave de la especulación**: la adivinanza no afecta a la salida. La verificación decide
con `argmax` sobre el estado real (`code_predictor.rs:412`), así que una adivinanza equivocada solo
cuesta tiempo, nunca bits. Esto abre la puerta a adivinanzas gratuitas (H15).

---

## 5. El bloque grande: oportunidades sobre el 91 %

### 5.1 Plan A — arreglar el kernel Q4_0 (bit-exacto, sin tocar precisión)

| # | Cambio | Dónde | Efecto |
|---|---|---|---|
| A1 | Escala con `_cvtsh_ss` (F16C) en vez de `f16_to_f32` | `avx2.rs:551`, `:679` | −6 µops/bloque |
| A2 | Dos bloques por iteración (una carga de 32 B, un `and`/`srli`, un par de `unpack`) | `avx2.rs:568-621`, `:660-690` | −3 µops/bloque |
| A3 | `QWEN_Q4_0_PF_DIST ≈ 4608` B en vez de 288 B | `avx2.rs:582`, `:609-610`, `:674` | cubre latencia |
| A4 | `_MM_HINT_NTA` en el stream | idem | menos presión de L3 |
| A5 | Añadir `f16c` al `target_feature` | cabeceras de los kernels | habilita A1 |

Con A1 + A2 el bloque baja de ~22 a ~13 µops para los mismos 32 pesos (1,7×). Q4_0 pasaría de
~11 a ~19 GB/s.

**Seguridad**: el formato no cambia, los valores no cambian y `_cvtsh_ss` es la conversión IEEE
correcta para todo valor finito. `--self-test` verifica `matmat` vs `matvec` y SIMD vs escalar.

**Impacto estimado en el CP**: los 184 MiB de FFN por frame pasan de 10,2 a ~19 GB/s.

| | Hoy | Con Plan A |
|---|---|---|
| CP FFN (184 MiB) | 18,0 ms | 9,7 ms |
| CP atención + QKV (429 MiB) | 26,3 ms | 26,3 ms |
| **CP por frame** | **45,6 ms** | **~36 ms** |

→ **−9,6 ms/frame → RTF 0,87 → ~0,74** (estimación).

> **Corregido en §13.6.** A1 (F16C) no solo no ayuda: **empeora un 13 %**. Y el techo real del CP no
> son 19 GB/s de Q4_0 sino ~26 Gpesos/s, que el CP ya alcanza. El margen real de A2 es −1,5 a −3 ms,
> no −9,6.

### 5.2 Plan B — Talker FFN a Q4_0 (habilitado por A)

El Talker son 14 MiB/capa: atención 5 MiB (wq 2 + wk 1 + wv 1 + wo 1) y **FFN 9 MiB
(gate_up 6 + down 3)**. El FFN es el **64 % de los bytes** y es la parte tolerante a cuantización.

Hay precedente medido en el propio repo: el CP ya tiene su FFN en Q4_0 y su atención en INT8,
porque *"Q4_0 en la atención aplana la voz"*. La asimetría FFN-sí / atención-no está medida.

| | Bytes Talker | Tiempo |
|---|---|---|
| Hoy (todo INT8) | 392 MiB | ~24,0 ms |
| FFN Q4_0 + kernel arreglado | 282 MiB | ~15,8 ms |

→ **−8,2 ms/frame.** Con A + B el camino productor baja de ~69,6 a ~52 ms → **RTF ~0,64-0,70**
(estimación; el rango depende de si el bus real sostiene 18 GB/s o se queda en ~16).

> **Refutado en §13.6.** El Talker ya está en el techo del bus (25,5 GB/s ≈ 27 GB/s medidos) y Q4_0
> rinde lo mismo **por peso** que INT8. Pasar el FFN a 4 bits no ahorra tiempo: reduce bytes de un
> recurso que no es el que limita. La cifra de ADR-005 (−4,6 % de RTF) es todo lo que hay.

**Riesgo acotado**: el diff de logits de 2,75 que descartó el Q4_0 era del Talker *completo*
(atención incluida). FFN-only debe medirse con el harness existente.

### 5.3 Lo que no va a funcionar, con números

| Idea | Por qué no |
|---|---|
| Micro-optimizar el `vpsignb` del kernel INT8 | Ahorra 2 µops de 9, pero el kernel ya va a ~46 GB/s de capacidad sobre una fase a 16 GB/s. No es el cuello |
| Overhead de la pool | ~400 despachos/frame × ~2 µs ≈ 0,8 ms = 1,1 % |
| Páginas grandes / TLB | Con 1 GiB de stream cíclico cada página falla, pero los page walks se solapan; esperable <5 %. Experimento barato, no titular |
| Profundizar la especulación a 3 adivinanzas | 1,30 → 1,19 códigos/pasada (con `c(B=4)=1,19`) |
| Batchear frames del Talker | Bloqueado por la serialización `Talker(t) → CP(t) → Talker(t+1)` |
| Fusionar las dos lecturas del `lm_head` | Entradas secuenciales distintas; 18 MiB = 1,6 % |

---

## 6. El resto: decoder, KV, cabezas y prefill

Total del resto: **104 MiB/frame (9,0 %)** en régimen estacionario, más el prefill amortizado.
Es pequeño en bytes, pero tiene dos propiedades que lo hacen valioso: **crece con la longitud**
(KV) y **compite con el productor por núcleos y bus** (decoder).

### 6.1 Decoder — 77,3 MiB/frame

Geometría por chunk de `m` frames (`speech_decoder.rs:615-661`):

```
cur_len = m                    (10)
ConvNeXt b=0:  k=2, s=2   →    cur_len = 20
ConvNeXt b=1:  k=2, s=2   →    cur_len = 40
initial_conv:  1024→1536, k=7 → cur_len = 40
upsample b=0:  rate 8, 768 ch  → cur_len = 320   (no tiled)
upsample b=1:  rate 5, 384 ch  → cur_len = 1600  (no tiled)
upsample b=2:  rate 4, 192 ch  → cur_len = 6400  (tiled: tile_len = 341 → 19 tiles)
upsample b=3:  rate 3,  96 ch  → cur_len = 19200 (tiled: tile_len = 682 → 29 tiles)
```

| Parte | MiB | Veces por chunk | MiB/frame |
|---|---|---|---|
| No tiled (ConvNeXt, initial_conv, upsample 0-1, pre_transformer) | 419 | 1 | 41,9 |
| Tiled (res_blocks de upsample 2 y 3) | 13,5 / 3,4 | 19 / 29 | 35,4 |

**D1 — tiles más grandes (bit-exacto).** `tile_len = SD_TILE_BYTES / (4 × cur_ch)`.

| `SD_TILE_BYTES` | Tiles b2 / b3 | MiB/frame tiled |
|---|---|---|
| 256 KiB (hoy) | 19 / 29 | 35,4 |
| 512 KiB | 10 / 15 | 18,0 |
| 1 MiB | 5 / 8 | 9,5 |

El recorte de convolución ya usa `cs_res_tail` como estado de continuidad (`:576`), así que el
resultado no depende del tamaño del tile. El coste es memoria de trabajo (de 256 KiB a 1 MiB) y que
el working set de activaciones deja de caber en L2, lo que puede encarecer el cómputo. **Es la
mejor relación beneficio/riesgo de todo este bloque.**

**D2 — chunks más grandes (cambia el redondeo).** `DEFAULT_CHUNK_FRAMES = 10`
(`request.rs:14`). Subir a 40 reduce la parte no tiled de 41,9 a 10,5 MiB/frame.

Dato relevante: **subir `chunk_frames` no empeora el TTFA**, porque el primer chunk es siempre de 2
frames (`next_stream_chunk_target`, `pipeline.rs:195-197`) y la rampa arranca en 4. Pero
`NUMERIC_CONTRACT.md:72` es explícito: *"las fronteras de chunk fijan el redondeo del decoder"*, y el
contrato exige PCM idéntico entre los cuatro modos. Cambiarlo obliga a decisión de contrato y a
re-basar los hashes de referencia.

**D3 — F32 → BF16.** Los pesos del decoder son F32 (`SdPreLayerWeights`, `SdConvNeXtWeights`,
`SdResBlockWeights`). Pasar a BF16 lleva 436 → 218 MiB y el bloque entero de 77 a ~39 MiB/frame.
No es bit-exacto. El decoder es una red convolucional, el caso más tolerante.

**D4 — tráfico interno del decoder (menor).** El `im2col` de `cs_convt` escribe y relee ~12 MiB por
chunk (~2,5 MiB/frame); las copias `cur → pw1 → cur` de `cs_res_tiled` añaden ~1 MiB/frame. Es el
precio conocido de im2col frente a convolución directa. `resize_zeroed` **no** es un coste: solo
inicializa los elementos nuevos (`aligned.rs:63-71`).

### 6.2 KV del Talker — 22,4 MiB/frame a 400 frames, cuadrático

- Por capa y posición: K + V = `2 × 1024 × 2 B` = 4 KiB.
- 28 capas → **112 KiB por posición**.
- En el frame `p` se leen `p + 1` posiciones.

| Oportunidad | Factor | Bit-exacto | Riesgo |
|---|---|---|---|
| K1 — GQA: una pasada de K/V por par de cabezas | ÷2 | Sí | Bajo |
| K2 — KV en INT8 | ÷2 | No | Medio |
| K3 — ventana deslizante + sumidero de prompt | O(N) en vez de O(N²) | No | Alto |

**K1** (`avx2.rs:1021-1034`): con `heads_per_kv = NUM_HEADS / NUM_KV_HEADS = 2`, dos cabezas de
consulta recorren la misma K/V. Reordenar a `for kv_h { for h in sus 2 cabezas }` mantiene cada
producto punto idéntico → bit-exacto. Baja de 16 a 8 unidades paralelas; conviene repartir por
`kv_h × posición`. Buena parte de la segunda lectura probablemente ya impacta en L2/L3, así que el
ahorro en DRAM es menor que los bytes nominales: **medir antes de comprometer**.

**K2**: K/V en INT8 en vez de BF16. Es una palanca distinta de la cuantización de pesos: aquí se
cuantiza el estado por posición. Riesgo medio; el CP ya demostró que la atención es sensible
(Q4_0 en la atención del CP aplana la voz).

**K3**: la palanca que cambia la *ley de escalado*, no la constante. Mantener el prefijo del prompt
(los tokens de texto, que el modelo necesita siempre) más una ventana local de las últimas W
posiciones convierte el coste por frame de `O(pos)` en `O(W)`. Es la receta de *attention sinks*.
Alto riesgo de calidad, pero es la única forma de que una locución de 11 minutos no se vuelva
cuadrática.

### 6.3 `codec_head` — 4 MiB/frame

Ya se limita a las 2 048 filas válidas (`pipeline.rs:83-96`). Pasarlo a INT8 lo deja en 2 MiB/frame
(−0,2 %). Riesgo bajo: el `argmax` sobre 2 048 logits es robusto a perturbaciones pequeñas.

### 6.4 Prefill — 1 400 MiB por petición

`talker_prefill` recorre las 28 capas en BF16 con un bloque de conversión de 4 MiB en L3
(`talker.rs:20`). La estructura ya es óptima: cada peso se lee una vez y no vuelve a DRAM.

La oportunidad es de **sustitución, no de reutilización**: usar las copias INT8 que ya están
residentes (422 MiB) en vez de las BF16. Con `int8_matmat_slice` (B = 8) el prefill baja a ~422 MiB
y además es más rápido (el producto INT8 es más barato por elemento en AVX2).

Impacto: 37,8 MiB/frame en una locución de 37 frames; 3,5 en una de 400. **Es la mayor palanca para
textos cortos y para el TTFA.**

---

## 7. Escalado con la longitud de la locución

Con `ROPE_MAX_POS = 8192` y `DEFAULT_MAX_FRAMES = 8192` (`request.rs:12`), el motor admite
locuciones de hasta ~11 minutos.

| N (frames) | Pesos (MiB totales) | KV (MiB totales) | KV / total |
|---|---|---|---|
| 400 | 422 000 | 8 960 | 2,1 % |
| 2 000 | 2 110 000 | 224 000 | 9,6 % |
| 4 000 | 4 220 000 | 896 000 | 17,5 % |
| 8 192 | 8 642 000 | 3 758 000 | 30,3 % |

Fórmulas: pesos = `1 055 MiB × N`; KV = `112 KiB × N²/2`.

**Consecuencia de diseño**: para locuciones cortas y medias, el bloque de pesos manda y las
palancas son el kernel (§5) y la precisión. Para locuciones largas, **el KV pasa a ser el primer
consumidor** y las únicas palancas son K1, K2 y K3. Un plan que ignore esto se rompe en el caso de
uso de 10 minutos.

---

## 8. Catálogo consolidado de oportunidades

| ID | Oportunidad | Bloque | Ahorro | Bit-exacto | Riesgo |
|---|---|---|---|---|---|
| A1 | Escala Q4_0 con F16C | 91 % | −6 µops/bloque | Sí | Nulo |
| A2 | Dos bloques por iteración | 91 % | −3 µops/bloque | Sí | Bajo |
| A3 | Prefetch Q4_0 a ~4,6 KiB | 91 % | cubre latencia | Sí | Nulo |
| A4 | `NTA` en el stream | 91 % | <5 % | Sí | Nulo |
| B | Talker FFN a Q4_0 | 91 % | −110 MiB/frame | No | Medio |
| D1 | Tiles del decoder a 1 MiB | 9 % | −26 MiB/frame | Sí | Bajo |
| D2 | `chunk_frames` a 40 | 9 % | −31 MiB/frame | No (contrato) | Medio |
| D3 | Decoder en BF16 | 9 % | −38 MiB/frame | No | Medio |
| K1 | GQA en una pasada | 9 % | −11 a −56 MiB/frame | Sí | Bajo |
| K2 | KV en INT8 | 9 % | −11 a −56 MiB/frame | No | Medio |
| K3 | Ventana + sumidero | 9 % | O(N²) → O(N) | No | Alto |
| C1 | `codec_head` en INT8 | 91 % | −2 MiB/frame | No | Bajo |
| P1 | Prefill con INT8 residente | amortizado | −980 MiB/petición | No | Medio |
| T1 | Adivinanza temporal gratuita | 91 % | −25 MiB/frame si ≥25 % acierto | Sí | Nulo |

---

## 9. Hipótesis y experimentos

Cada experimento cambia **una sola cosa**. Métrica principal: `SynthesisTiming`
(`talker_ms`, `cp_ms`, `decoder_ms`, RTF), que ya existe (`pipeline.rs:279-281`). Métrica de
corrección: `--self-test`, `test_reference_audio_hashes`, `synthesis_paths_tests` (paridad de
modos), y acuerdo de `code0` por frame. Regla de decisión explícita en cada uno.

### Ola 0 — Medición, riesgo cero

**E1 · H1: el kernel Q4_0 mueve menos GB/s que el INT8 con la misma forma.**
Sin código nuevo: derivar GB/s por fase de `SynthesisTiming` dividiendo los bytes conocidos entre el
tiempo medido. El Talker es 100 % INT8; el CP tiene 30 % Q4_0.
*Regla*: si la razón por byte (Talker / CP) es ≥1,25×, H1 se sostiene y el Plan A pasa a prioridad
máxima. Si es <1,15×, H1 se refuta y hay que buscar la causa en otra parte (pool, paralelismo,
tamaño de matriz).
*Nota*: un microbenchmark por kernel daría más precisión, pero exige código nuevo; no se propone
hasta que E1 lo justifique.

**E15 · H15: el código del frame anterior es una adivinanza rentable.**
Medición offline sobre audio ya sintetizado: `P(code_t[k] == code_{t-1}[k])` para k = 1..15.
*Regla*: ≥25 % → entra como adivinanza gratuita (ahorra ~25 MiB/frame y el trabajo de las cabezas);
<25 % → se mantiene la adivinanza actual. Cero riesgo: no cambia la salida.

### Ola 1 — Kernel, bit-exacto

**E2 · H2: el prefetch corto del Q4_0 es la causa principal.**
Cambiar solo `b + 16` por una distancia de ~4 608 B.
*Regla*: `cp_ms` −5 % o más → aceptar; <2 % → el prefetcher de hardware ya lo cubría y H2 se
refuta (pero A1/A2 siguen en pie).
*Corrección*: `--self-test` debe seguir en verde (el prefetch no altera resultados).

**E3 · H3: la conversión `f16_to_f32` por bloque cuesta ≥10 %.**
A5 + A1 (`_cvtsh_ss`).
*Regla*: `cp_ms` −8 % o más → aceptar. Verificar igualdad bit a bit de `matvec_q4_0` y
`matmat_q4_0` contra la versión anterior.

**E4 · H4: procesar dos bloques por iteración da ≥10 %.**
A2 sobre E3.
*Regla*: −8 % o más → aceptar. **E2+E3+E4 juntos deben dar ≥1,5× en Q4_0**; si no, H1 está mal
cuantificada y hay que revisar el modelo de µops.

**E5 · H5: `NTA` en vez de `T0` mejora ≥3 %.**
Solo el hint.
*Regla*: ≥3 % → aceptar; si empeora, descartar (NTA puede saturar los buffers de write-combining).

### Ola 2 — Bit-exacto, fuera del kernel

**E7 · H7: los tiles del decoder a 1 MiB son bit-exactos y bajan `decoder_ms` ≥25 %.**
Solo `SD_TILE_BYTES` 256 KiB → 1 MiB.
*Regla*: **PCM byte-idéntico obligatorio** (si no lo es, el carry no es exacto y H7 se refuta); y
`decoder_ms` −25 % o más. Probar también 512 KiB: el óptimo puede estar ahí por L2.

**E10 · H10: la GQA fusionada reduce `talker_ms` ≥3 % en locución larga.**
Una pasada de K/V por par de cabezas (`avx2.rs:1021-1034`).
*Regla*: medir a 400 y a 2 000 frames. A 2 000, ≥3 % → aceptar; <2 % → descartar (el acierto ya
estaba en L2/L3). Bit-exacto obligatorio.

### Ola 3 — Precisión, con prueba de calidad

**E6 · H6: el Talker FFN en Q4_0 es aceptable y rentable con el kernel rápido.**
`gate_up` + `down` en Q4_0, atención en INT8, sobre E2-E4.
*Regla*: acuerdo de `code0` ≥99,5 % por frame sobre el corpus de referencia, y ninguna regresión
audible. Si falla, reintentar solo `gate_up` (43 % de los bytes del Talker). Si falla también, H6 se
refuta.
*Métrica de tiempo*: `talker_ms` −25 % o más.

**E8 · H8: el decoder en BF16 baja `decoder_ms` ≥30 % sin daño.**
*Regla*: escucha A/B y diff espectral. Sibilancias o pérdida de aire → descartar.

**E11 · H11: el KV en INT8 reduce `talker_ms` ≥10 % a 2 000 frames sin daño.**
*Regla*: hashes + escucha a 400 y 2 000 frames. Si hay degradación en locución larga, descartar.

**E13 · H13: el prefill en INT8 baja el TTFA ≥40 % con acuerdo de `code0` ≥99 %.**
*Regla*: `prefill_ms` −60 % y acuerdo ≥99 %. Re-basar hashes de referencia tras la escucha.

**E14 · H14: el `codec_head` en INT8 es indistinguible.**
*Regla*: acuerdo de `code0` ≥99,9 % → aceptar (−2 MiB/frame, barato).

### Ola 4 — Contrato y arquitectura

**E9 · H9: `chunk_frames` a 40 no cambia el TTFA y baja `decoder_ms` ≥30 %.**
*Regla*: TTFA idéntico (debe serlo: primer chunk = 2 frames); PCM cambia → **decisión de contrato**
antes de implementar. Opciones: (a) subir el valor por defecto en ambos modos y re-basar hashes;
(b) valor grande solo en `Delivery::Buffer` y aceptar PCM distinto por modo; (c) no hacerlo.

**E12 · H12: ventana deslizante + sumidero de prompt convierte el KV en O(N) sin daño.**
*Regla*: a 2 000 frames, `talker_ms` −20 % o más; escucha de locuciones largas. Cualquier
degradación → descartar. Es el experimento de mayor riesgo y solo se aborda si el caso de uso de
locuciones largas es real.

---

## 10. Protocolo de medición

1. **Estado del sistema**: misma carga de fondo en A y B. El repo documenta que el RTF con carga
   (0,87) y sin carga (0,65) difieren mucho: medir en los dos escenarios y reportar ambos.
2. **Alternancia**: A/B/A/B intercalado, no bloques. Es lo que ya hace ADR-005 y evita deriva
   térmica (APU de 15 W).
3. **Repeticiones**: mínimo 5 pares por condición; reportar mediana, no media.
4. **Fijado**: misma semilla, mismos parámetros, misma voz, mismos textos.
5. **Desglose por fase**: `talker_ms`, `cp_ms`, `decoder_ms` y RTF total. Un cambio que mejora el
   total pero empeora una fase indica contención, no mejora.
6. **Corrección**: `--self-test` en verde y hashes de referencia cuando el cambio sea bit-exacto;
   acuerdo de `code0` cuando no lo sea.
7. **Hilos**: registrar `-j` y verificar que no se cae en `spin_budget = 0`
   (`pool.rs:39-45`: ocurre cuando `-j` ≥ núcleos físicos).

---

## 11. Consolidación: propuesta por olas

La propuesta final se redacta **después** de cada ola, con los deltas medidos, no antes. Estructura:

> **Invalidado por §13 (2026-11-25).** Las olas 1-3 y la proyección acumulada de abajo se
> construyeron sobre el modelo GB/s, que resultó ser una ilusión de unidades (§13.2). Ninguna de
> las hipótesis A1-A3, B, D1, D2 o D3 sobrevivió: A1 empeora un 13 %, A3 no hace nada, B ya está en
> el techo, D1 no cambia el RTF y D2/D3 no están en la ruta crítica. Además la ruta crítica es el
> **productor** (`talker + cp` = 48,6 ms/frame) y el decoder va 42,8 ms/frame **en paralelo**, así
> que cualquier mejora del decoder por debajo de esa diferencia es invisible en RTF. La única
> optimización viva es el paso 4 del kernel Q4_0 (§13.7), con un techo realista de −1,5 a −2,5
> ms/frame y RTF ~0,62-0,63. K3 (ventana + sumidero en el KV) sigue siendo la única palanca con
> cambio de ley de escalado, pero no está medida y es de alto riesgo.

| Ola | Experimentos | Criterio de cierre | Resultado esperado |
|---|---|---|---|
| 0 | E1, E15 | H1 decidida | saber si el Plan A aplica |
| 1 | E2, E3, E4, E5 | Q4_0 ≥1,5× o H1 refutada | RTF 0,87 → ~0,74 |
| 2 | E7, E10 | bit-exactitud + `decoder_ms` | −26 MiB/frame, menos contención |
| 3 | E6, E8, E11, E13, E14 | calidad aceptada | RTF ~0,64-0,70 |
| 4 | E9, E12 | decisión de contrato | escalado lineal en locuciones largas |

**Regla de consolidación**: una optimización entra en la propuesta solo si (a) su experimento pasó
su regla de decisión, (b) no rompe el contrato numérico o la ruptura está aprobada, y (c) su efecto
se mantiene al combinarse con las anteriores (medido, no supuesto). Las optimizaciones de la misma
fase pueden canibalizarse: si el CP ya está en el techo del bus, arreglar el kernel y bajar bytes a
la vez no suma.

**Proyección acumulada** (estimación del modelo bytes/tiempo, a confirmar):

| Escenario | Bytes 91 %/frame | RTF estimado |
|---|---|---|
| Hoy (ADR-005, con carga) | 1 055 MiB | **0,87** |
| Ola 1 (kernel Q4_0) | 1 055 MiB | **0,74** |
| Ola 1 + E6 (Talker FFN Q4_0) | 945 MiB | **0,64-0,70** |
| + Ola 3 (KV, decoder, cabezas) | ~880 MiB | **0,60-0,66** |
| + Ola 4 (locuciones largas) | igual, pero O(N) | dominio del caso de 11 min |

---

## 12. Riesgos y contrato numérico

| Riesgo | Dónde | Mitigación |
|---|---|---|
| Romper la bit-identidad de `matmat`/`matvec` | A1, A2 | `--self-test` compara contra la referencia escalar |
| Romper la paridad de modos (Buffer ↔ Stream) | D2, E9 | `synthesis_paths_tests`; decisión de contrato explícita |
| Degradar la voz con 4 bits en el Talker | B, E6 | acuerdo de `code0` + escucha; fallback a solo `gate_up` |
| Degradar la atención con KV INT8 | K2, E11 | medir a 400 y 2 000 frames, no solo en frases cortas |
| Cambiar el escalado sin darse cuenta | K3, E12 | comparar curvas RTF vs N, no un solo punto |
| Medir con el sistema cargado y atribuirlo al cambio | todos | alternancia A/B y dos escenarios de carga |

**Invariantes que no se tocan sin decisión explícita**: `NUMERIC_CONTRACT.md:64` (pasadas del CP y
bit-identidad de la verificación especulativa), `:72` (calendario fijo de chunks), `:86`
(`matmat_int8(B=8)` ≡ B `matvec_int8`), `:49` (reparto de filas entre hilos).

---

## 13. Registro de experimentos (2026-11-25)

Máquina: Ryzen 7 5825U (8 núcleos / 16 hilos), 2 × 16 GiB DDR4-3200, plan «Equilibrado», sin carga
de fondo. Texto de prueba: 279 frames (21,8 s de audio), `-j 4`.

**Aviso de método**: este portátil tiene dos regímenes de rendimiento separados por ~1,6×. Las
primeras medidas (RTF ≈ 1,00) se tomaron en el régimen lento, probablemente por calor residual de
las compilaciones; las válidas (RTF ≈ 0,65) se tomaron con la máquina estabilizada y se repitieron
en pares A/B contrabalanceados. Toda medida de este registro pertenece al régimen estable.

### 13.1 E1 — línea base y puerta de decisión

| Magnitud | Valor |
|---|---|
| `talker_ms` | 16,0 ms/frame |
| `cp_ms` | 32,6 ms/frame |
| Decoder (en paralelo) | 42,8 ms/frame |
| RTF | 0,65 |

Razón `cp_ms / talker_ms` = **2,04**, por encima de la puerta de 1,56. La puerta se abre, pero la
interpretación que la acompañaba era incorrecta: ver §13.2.

### 13.2 E1′ — GB/s por kernel, medida directa

Banco aislado (proyecto de usar y tirar, fuera del repo) con 300 MB de copias por forma para forzar
DRAM y no caché. Mejor de dos pasadas por celda, 4 hilos:

| Forma | INT8 GB/s | Q4_0 GB/s | Razón GB/s | INT8 Gpesos/s | Q4_0 Gpesos/s |
|---|---|---|---|---|---|
| `gate_up` 6144×1024 matvec | 25,0 | 14,8 | 1,64 | 25,0 | 26,3 |
| `down` 1024×3072 matvec | 26,7 | 15,2 | 1,75 | 26,7 | 27,0 |
| `wq` 2048×1024 matvec | 29,2 | 16,7 | 1,75 | 29,2 | 29,7 |
| `wo` 1024×1024 matvec | 23,7 | 14,3 | 1,66 | 23,7 | 25,4 |
| `gate_up` matmat B=2 | 26,3 | 13,6 | 1,94 | 26,3 | 24,2 |
| `down` matmat B=2 | 33,8 | 13,6 | 2,49 | 33,8 | 24,2 |

**Conclusión que cambia el diagnóstico**: en GB/s Q4_0 parece 1,6-2,5× peor, pero en **pesos por
segundo** rinde lo mismo o más que INT8 (24-30 Gpesos/s ambos). Q4_0 mueve menos bytes para hacer
el mismo trabajo: no está «limitado por ALU», está *más cerca* de su límite de cómputo porque su
formato es 1,78× más denso. Comparar GB/s entre formatos distintos no mide velocidad.

Escalado con hilos (forma `gate_up`, matvec): INT8 11,3 → 24,2 → 26,2 GB/s con 1, 4 y 8 hilos; Q4_0
7,6 → 14,8 → 30,6. A 8 hilos ambos convergen en ~25 GB/s: **ese es el techo de DRAM del equipo**
(≈50 % del teórico de 51,2 GB/s, lo normal en APU).

### 13.3 E3 — escala F16C en lugar de `f16_to_f32` (refutado)

El código generado para `f16_to_f32` es un blob de ~20 instrucciones enteras con dos ramas y un
bucle de subnormales (verificado en el `.s`). Sustituirlo por `_mm_cvtph_ps` debería haber ahorrado
~10 µops por bloque. Medido, contrabalanceado (A B B A A B B A):

| Variante | `cp_ms` medio | `talker_ms` medio |
|---|---|---|
| A: `f16_to_f32` escalar | **31,88** | 15,97 |
| B: F16C | **36,05** | 15,40 |

**+13 % de `cp_ms`**: la conversión vectorial usa el mismo puerto que los `vpunpck`/`vinserti128` del
desempaquetado, que es el puerto saturado; la versión escalar corre en puertos enteros que están
libres y además se solapa con el trabajo vectorial. Conclusión: el cuello del kernel es el **puerto
de *shuffle***, no la anchura de emisión. Cambio revertido.

### 13.4 E2 — adelanto del prefetch 288 B → 4 608 B (refutado)

6 pares A/B contrabalanceados: razón mediana `cp_ms` = **0,998** (sin efecto). El recorrido es
perfectamente secuencial (bloques de 18 B dentro de filas contiguas), así que el *streamer* de
hardware ya cubre la latencia y el prefetch manual con 288 B no era el problema. Cambio revertido.

### 13.5 Barrido de `-j` (2, 3, 4, 6, 8)

| `-j` | talker ms/f | cp ms/f | RTF |
|---|---|---|---|
| 2 | 17,8 | 39,2 | 0,82 |
| 3 | 16,3 | 34,3 | 0,68 |
| **4** | **16,1** | **32,1** | **0,64** |
| 6 | 17,7 | 34,1 | 0,86 |
| 8 | 20,5 | 44,9 | 0,86 |

`-j 4` es el óptimo y coincide con el reparto 4 + 4 (productor + decoder) sobre 8 núcleos físicos.
A partir de 6 la sobre-suscripción y `spin_budget_for = 0` cuestan 30 %.

### 13.6 Correcciones a las proyecciones del dossier

| Afirmación original | Estado | Dato que la corrige |
|---|---|---|
| El CP mueve 13,4 GB/s porque Q4_0 está limitado por ALU | **refutada** | 26,9 Gpesos/s: el CP ya está en el techo |
| Plan A (kernel Q4_0) = −9,6 ms/frame | **corregida a −1,5 a −3 ms** | Q4_0 matmat ya rinde 24,2 Gpesos/s frente a 26,3 de INT8: solo queda un 8 % |
| Plan B (Talker FFN Q4_0) = −8,2 ms/frame | **refutada** | El Talker ya está en el techo del bus (25,5 GB/s); Q4_0 no sube el ritmo por peso |
| El prefetch corto era la causa | **refutada** | 0,998 |
| `f16_to_f32` costaba ≥10 % | **refutada** | F16C empeora un 13 % |
| El decoder es el 9 % del coste | **corregida** | En tiempo de pared es el 47 %: 42,8 ms/frente a 48,6 del productor |

**Rendimiento por peso, que es la métrica correcta**:

| Fase | Pesos/frame | Tiempo | Gpesos/s |
|---|---|---|---|
| Talker | 411 M | 16,0 ms | 25,7 |
| CP | 875 M | 32,6 ms | 26,8 |
| Decoder (conv, no comparable) | 45,7 MB de pesos | 42,8 ms | — (limitado por FLOP) |

Ambas fases del productor están en el mismo techo, ~26 Gpesos/s. **No hay margen en el productor
sin reducir el número de pesos que se tocan**, y eso ya no es una cuestión de formato ni de kernel.

### 13.7 Qué queda vivo

1. **El decoder es ahora el primer objetivo.** 42,8 ms/frame de pared, co-crítico con el productor
   (48,6), y mueve sus pesos a 1,07 GB/s: está 25× por debajo del ritmo del *matvec*, así que su
   límite son los FLOP de la convolución, no el bus. Cualquier mejora del productor se topará con él
   en ~0,55 de RTF.
2. **Reescribir el desempaquetado de Q4_0** sigue siendo lo único que puede acelerar el CP, y ahora
   se sabe dónde: hay que **quitar `vinserti128`** procesando dos bloques por iteración. El techo
   estimado aquí (32,6 → 26,7 ms, RTF 0,57) usaba el modelo *bytes-bound* ya refutado en §13.2.
   Con los Gpesos/s medidos (Q4_0 24,2 frente a INT8 26,3 en `matmat B=2`, §13.2) y el 50 % de los
   bytes del CP en Q4_0, el margen realista es **−1,5 a −2,5 ms/frame**, RTF 0,65 → ~0,62-0,63.
   Además el decoder (39-43 ms/frame) ya manda, así que el efecto en RTF es menor aún.
3. **Lo demás del catálogo (§8) no gana tiempo en el productor**: bajar bytes no baja tiempo si el
   recurso saturado son los pesos por segundo. Solo el KV en locuciones largas (§7) cambia la ley
   de escalado y sigue siendo válido.

### 13.8 Perfil interno del decoder por etapas

Instrumentación temporal (ya revertida) con `Instant` en cada etapa de
`speech_decoder_decode_streaming` y `conv_decoder_forward_streaming` (`speech_decoder.rs`). Se
imprimió una línea por chunk y se agregaron los 39 chunks de los 279 frames. Tres ejecuciones,
`-j 4`, régimen estable: 39,43 / 38,71 / 39,26 ms/frame de decoder; las proporciones se mantienen
entre la primera y la segunda con menos del 2 % de diferencia.

| Etapa | ms/frame | % del decoder |
|---|---|---|
| VQ + pre-conv + proyección de entrada | 0,26 | 0,6 % |
| Pre-transformer (5 capas, caché KV) | 2,45 | 6,2 % |
| Norma final + proyección de salida | 0,04 | 0,1 % |
| ConvNeXt ×2 | 2,19 | 5,6 % |
| `initial_conv` (1024→1536, k=7) | 0,87 | 2,2 % |
| Upsample b0 (rate 8, 768 ch) | 7,99 | 20,3 % |
| Upsample b1 (rate 5, 384 ch) | 7,15 | 18,1 % |
| Upsample b2 (rate 4, 192 ch, tiled) | 9,29 | 23,6 % |
| Upsample b3 (rate 3, 96 ch, tiled) | 8,68 | 22,0 % |
| Snake final + `final_conv` | 0,52 | 1,3 % |

Los cuatro bloques de upsample son el **84 % del decoder**. Dentro de ellos:

| Parte | ms/frame | % del decoder |
|---|---|---|
| Bloques residuales (3 por upsample) | 24,95 | 64,8 % |
| Convoluciones transpuestas (`cs_convt`) | 7,49 | 19,4 % |
| Snake de entrada al upsample | 0,06 | 0,2 % |

Y dentro de cada bloque residual (medición de segundo nivel, 12 bloques × 39 chunks):

| Operación | ms/frame | % del residual |
|---|---|---|
| `snake1` | 0,91 | 3,6 % |
| **`conv1` completa dilatada k=7** | **19,67** | **78,8 %** |
| `snake2` | 0,61 | 2,4 % |
| `conv2` pointwise | 3,47 | 13,9 % |
| Suma residual | 0,29 | 1,2 % |

> **Corrección**: `conv1` **no es depthwise**. Los pesos del safetensors tienen forma
> `[ch, ch, 7]` (p. ej. `[768, 768, 7]`) y el 100 % de las posiciones fuera de la diagonal son
> no nulas (verificado leyendo el tensor). Es una convolución completa y densa; no hay un
> desperdicio de `ch×` por ceros.

**Conclusión**: el coste no está repartido. Una sola operación —la convolución completa dilatada
de 7 taps dentro de los bloques residuales— consume **19,67 de 39 ms/frame, el ~50 % del decoder
completo**. La confirma el perfil histórico de ADR-005 (convoluciones k=7 dilatadas: 58 %), ahora
con la medición directa del código actual. `cs_conv1d_im2col` (bloques 2-3, *tiled*) y `cs_conv1d`
(bloques 0-1) son el objetivo natural; las transpuestas son el segundo (19,4 %) y el
pre-transformer solo el 6,2 %, así que optimizar KV o BLAS no mueve la aguja aquí.

**Contabilidad de FLOP por frame de audio** (chunk de 10 frames; longitudes internas 320 / 1 600 /
6 400 / 19 200):

| Operación | GMAC/chunk | GMAC/frame | Tiempo | Rendimiento efectivo |
|---|---:|---:|---:|---:|
| `conv1` (7 taps, `ch×ch`) | 17,59 | 1,76 | 19,67 ms | **179 GFLOP/s** |
| `conv2` (pointwise `ch×ch`) | 2,51 | 0,25 | 3,47 ms | 145 GFLOP/s |
| `cs_convt` | 3,35 | 0,34 | 7,49 ms | 89 GFLOP/s |

`conv1` es de lejos la operación con más FLOP (7× el resto) y, sin embargo, es la **más eficiente
por FLOP** de las tres: ya se ejecuta como `sgemm` denso. El coste es trabajo real, no una mala
formulación del kernel. El techo de ~448 GFLOP/s (4 hilos OpenBLAS sobre Zen 3) está a ~2,5×, pero
los 4 hilos del productor comparten los 8 núcleos físicos, así que el margen real es menor.

Nota sobre el método: el hilo del decoder corre en paralelo con los 4 hilos del productor sobre 8
núcleos físicos, así que cada `lap` mide tiempo de pared incluyendo la espera a los 4 hilos de la
pool del decoder. Es la magnitud correcta para RTF. La instrumentación en sí añade ~2 340 lecturas
de reloj en total (decenas de µs frente a 11 s), despreciable.

### 13.9 H4 — tiles más grandes para el GEMM de `conv1` (refutado)

**Hipótesis**: en los bloques 2 y 3, `SD_TILE_BYTES = 256 KiB` da tiles de `N = 341` (b2) y
`N = 682` (b3) columnas, demasiado estrechos para OpenBLAS. Cuadruplicar el tile a 1 MiB
(`N = 1365` y `2730`), manteniendo el total de FLOP, debería subir el rendimiento del GEMM un
10-20 %.

**Microbanco aislado** (3 hilos OpenBLAS, mismo `gen_blas = nt - 1` que la generación, formas
reales del safetensors, mejor de 3 pasadas):

| Forma | ms por chunk | GFLOP/s |
|---|---:|---:|
| b2 real, `N=341` | 40,93 | 245,1 |
| b2 `N=640` | 37,93 | 261,3 |
| b2 `N=1600` | **35,51** | **279,1** |
| b2 `N=3200` | 35,72 | 277,4 |
| b3 real, `N=682` | 31,82 | 240,6 |
| b3 `N=1366` | 29,72 | 249,1 |
| b3 `N=2732` | **29,33** | **252,3** |

La hipótesis se sostenía **en el GEMM aislado**: −13 % en b2 y −8 % en b3.

**A/B en el motor real** (`SD_TILE_BYTES` 256 KiB → 1 MiB, 4 pares contrabalanceados A B B A A B B A,
279 frames, `-j 4`):

| Par | RTF A/B (>1 = B mejor) |
|---|---:|
| 1 | 0,952 |
| 2 | 1,049 |
| 3 | 0,956 |
| 4 | 1,076 |

**Mediana 1,003: sin efecto.** `decoder_ms` medio 11 647 ms (A) frente a 12 040 ms (B), es decir,
B fue un 3,4 % *más lento*. El patrón alterno de las razones es firma de deriva térmica de la
máquina, que el contrabalanceo cancela en la mediana. Cambio revertido.

**Por qué el microbanco engañó**: el buffer im2col pasa de 1,75 MiB (256 KiB de tile × 7) a 7 MiB,
por encima de la L2 por núcleo (512 KiB). Lo que gana el GEMM al ensanchar `N` se pierde en los
accesos a memoria del buffer. El valor de 256 KiB del contrato ya estaba eligiendo ese punto.

**Efecto colateral**: el hash del PCM cambió (`900060bb…` → `b733f864…`). Aunque la convolución es
matemáticamente la misma, OpenBLAS bloquea distinto con otro `N` y el redondeo cambia. Cualquier
cambio de tile exige regenerar la tabla de §5 de `NUMERIC_CONTRACT.md`.

### 13.10 Hipótesis descartadas sin medir (por el contrato numérico)

- **Contención de hilos BLAS**: `pipeline.rs:265` fija `gen_blas = nt - 1 = 3` antes de lanzar el
  decoder. Con `-j 4`: 3 BLAS + 4 productores + 1 hilo decoder = 8 hilos sobre 8 núcleos físicos.
  No hay sobre-suscripción.
- **Unir las 7 `sgemm` de b0/b1 en una im2col**: el microbanco de §13.9 da 47,54 ms/chunk
  (166,7 GFLOP/s) frente a 28,23 ms (280,8 GFLOP/s) del camino actual. `NUMERIC_CONTRACT.md:60`
  ya lo documenta: con 768/384 canales, `K = ch` rinde mejor; el im2col solo compensa con 96-192.

### 13.11 Viabilidad de INT8 en `conv1` (refutada por velocidad; numéricamente correcta)

**Método**: se capturaron con instrumentación temporal (inmediatamente revertida) los pesos
`wcat`, la matriz im2col `cols` y la salida F32 de una llamada real de `cs_conv1d_im2col` para
`ch = 192` y `ch = 96`. Con esos datos, el banco desechable cuantiza los pesos a `i8` con escala por
fila (idéntico a `quantize_bf16_to_int8`), llama a `matmat_int8` por bloques de `b_dim` columnas y
compara contra la salida F32. La instrumentación no quedó en el repo.

**Error numérico** (3 hilos, datos reales):

| Bloque | L2 relativo | Error máx. | SNR |
|---|---:|---:|---:|
| b2 (`ch=192`, `K=1344`) | **1,76 %** | 0,0202 | 70,2 dB |
| b3 (`ch=96`, `K=672`) | **0,32 %** | 0,0139 | 99,6 dB |

Dentro del umbral del contrato para kernels INT8 (<3 %). El error no depende de `b_dim`.

**Velocidad** (ms por chunk de 10 frames, 3 hilos; `sgemm` aislado de §13.9):

| Forma | `sgemm` | `matmat_int8` (3 hilos) | `matmat_int8` (4 hilos) |
|---|---:|---:|---:|
| b0 `K=ch`, `N=320` | 28,23 | **24,3** (0,86×) | 21,1 (0,75×) |
| b1 `K=ch`, `N=1600` | 69,87 | 79,5 (1,14×) | 80,9 (1,16×) |
| b2 `K=7ch`, `N=341` | 40,93 | 86,0 (2,10×) | 87,2 (2,13×) |
| b3 `K=7ch`, `N=682` | 31,82 | 86,1 (2,70×) | 86,5 (2,72×) |
| **Total** | **170,9** | **275,9 (1,61× más lento)** | **275,7 (1,61×)** |

Con los datos reales (b2/b3) el resultado coincide: 91-112 ms/chunk frente a 41/32 de `sgemm`.

**Causa**: `matmat_int8 reparte filas` solo si `rows >= 256` (`dispatch.rs:597`). Con `ch = 192`
y `96`, **se ejecuta en un solo hilo**, mientras `sgemm` usa 3. Además `b_dim <= 64` obliga a 57 y
87 llamadas por chunk; la cuantización de activaciones cuesta 9,3 µs por bloque de 64 columnas
(~0,5 ms por chunk, no es lo dominante). Solo b0 (`ch=768`) sale ganando, y un 14 %.

**Veredicto**: se descarta la fase de PCM (no tiene sentido validar audio de una opción que ya es
1,6× más lenta). INT8 no es una mejora con el kernel actual. La única forma de que cambiara el
resultado sería envolver `matmat_int8` con reparto manual de columnas entre hilos para b2/b3 —eso
ya es implementación, no viabilidad— y aun así b1 sigue perdiendo.

Nota: la comparación es aislada, con el productor parado. En el motor, `sgemm` cae de 245-281 a
179 GFLOP/s por contención con el productor; el kernel INT8 usaría la pool del decoder, que
también compite. No es plausible que una diferencia de 1,6× se invierta por ahí.

### 13.12 Verificación del *batching* intra-síntesis en el CP

Afirmación a verificar: *«para una síntesis individual, la única forma de batch que sirve es la
especulación dentro del CP, y ya está casi exprimida»*.

**Medición directa**: la prueba `cp_predict_matches_sequential_reference`
(`code_predictor.rs:677-702`) imprime `K1TEST aciertos posibles: 67 de 208` → **tasa de acierto
32,2 %** (el contrato decía 28-31 %; medido en este equipo, 67/208).

**Modelo exacto de pasadas** (leyendo el bucle `code_predictor.rs:396-427`): una pasada inicial
`cp_step2(0)` y luego `n` iteraciones que avanzan `g` en 2 si aciertan y en 1 si fallan, hasta
`g = 14`. Por tanto `n·(1 + h) = 14` y:

```
pasadas/frame = 1 + 14 / (1 + h)
```

| Tasa `h` | Pasadas/frame |
|---:|---:|
| 0 (sin especulación) | 15,0 |
| **0,322 (medido)** | **11,6** (el contrato registra 11,9) |
| 0,65 | 9,5 |
| 1,0 (oráculo perfecto) | 8,0 |

Estamos a 3,6 pasadas del suelo absoluto: quitarlas todas exige `h = 1`, imposible sin cambiar el
predictor.

**Por qué no se puede especular más profundo**: la adivinanza sale de `lm_heads[g+1]` sobre el
estado `normed` que ya existe (`code_predictor.rs:400`) — es gratis. Para adivinar dos códigos por
delante habría que alimentar el código adivinado al transformador y calcular un estado futuro, es
decir, **una pasada completa adicional**. Con `h = 0,322`, la probabilidad de acertar dos seguidos
es ~10 % y tres ~3 %: la pasada extra casi nunca se amortiza. La especulación de profundidad 1 es
óptima para esta arquitectura.

**Cuánto habría que ganar para que el CP dejara de mandar**: con 29,5 ms/frame y 11,9 pasadas
(2,48 ms/pasada), el CP tendría que bajar a 23,6 ms/frame para igualar al decoder (39,3 ms,
productor = 15,7 + CP). Eso son 9,5 pasadas, es decir **h ≈ 0,65**, el doble del acierto medido.
No es alcanzable con el predictor actual.

**Corrección a la afirmación**: «la única forma de batch que sirve» es cierta solo si se refiere a
la **ruta crítica**. El decoder también hace batch (chunks de 10 frames en GEMM) y el Talker lo
hace en el prefill (10 tokens), pero ninguno baja el RTF: el decoder ya va por debajo del
productor y el prefill se amortiza una sola vez por petición. En la ruta crítica
(`talker + cp`), la especulación del CP es efectivamente el único batching, y está cerca de su
techo práctico.

**Confirmación independiente** (simulación offline de 4 688 frames, 2026-11-19, registrada en
Engram #2214): verificar los `top-m` candidatos del código siguiente da 11,83 pasadas/frame con
`top1` (31,6 % de acierto, coincide con el motor), 11,53 con `top2` (**−2,5 % del CP, ~1 ms/frame
con carga**), 11,75 con `top3` y 12,68 con `top4`; el coste relativo por pasada sube de 0,986
(B=2) a 1,047 (B=3) y 1,126 (B=4). El código del frame anterior solo se repite el 7,3 % de las
veces. Es decir, **ampliar la especulación ya se simuló y se descartó por marginal**, y además
exigiría una máscara de atención B=3.

### 13.13 Revisión bibliográfica (2026-11-25): especulación RVQ en Qwen3-TTS

Tres fuentes directamente aplicables a este modelo:

1. **NVIDIA TensorRT-Edge-LLM, `CodePredictor Speculative Decoding`** (docs del repo, 2026).
   Qwen3-TTS / Qwen3-Omni tienen exactamente nuestro esquema: un `lm_head` por profundidad RVQ
   sobre un *residual stream* compartido. El *draft* es gratis: aplicar `head_{s+t}` al estado
   `h_s` da una propuesta válida a `t` posiciones; una GEMV por profundidad, todas en un lote.
   La verificación es **speculative sampling exacto** (misma distribución). Ventana `N = 2..8`,
  atención saturada a partir de 4 profundidades, desactivada por defecto. Documentan que en
   **greedy el resultado difiere** del camino autorregresivo por redondeo distinto; bajo muestreo
   la distribución no cambia. Es exactamente lo que ya hace nuestro `cp_step2`, pero con
   ventana 2 en vez de N.

2. **Argmax, `RVQ Position Aware Speculative Decoding for On Device TTS`** (arXiv 2609.37007,
   2026-09). Sobre **Qwen3-TTS-0.6B**, el mismo modelo:
   | Drafter | Tokens/paso | Parámetros nuevos |
   |---|---:|---:|
   | Cabezas vecinas reutilizadas | 1,70 | 0 |
   | Cabezas Medusa | 2,60 | ~94 M (+16 %) |
   | **Offsets posicionales** | **2,47** | **3 072 (5×10⁻⁴ %)** |
   El problema que resuelven es el **desajuste posicional**: cada `lm_head_{k+j}` fue entrenado
   para el estado oculto de su propia posición, así que aplicarlo a `h_k` está fuera de
   distribución. Añaden vectores de sesgo `b_j` al estado oculto solo en el camino del *draft*
   (el verificador nunca los consume → la distribución de salida queda intacta). Árbol disperso
   con *tree attention*, N=16-32, overhead 10-20 %, **2-2,2× speedup real**, WER par en 6 idiomas.
   Nuestro motor: 15 códigos / 11,9 pasadas = **1,26 tokens/paso**; los 1,70 de reutilización
   simple ya son +35 %.

3. **Relajación de aceptación**:
   - SSD (Interspeech 2025): factor de tolerancia `β`, draft = modelo poco profundo inicializado
     con capas altas del target, 1,4× con calidad perceptual preservada.
   - PCG (ICASSP 2026): agrupa tokens por similitud acústica y acepta a nivel de grupo, 1,4× con
     mejor WER que SSD.
   - VADUSA (ICASSP 2025): Medusa + tolerancia sobre VALL-E.
   Todas **cambian la distribución o el token exacto**; incompatibles con nuestro requisito de PCM
   bit-idéntico salvo decisión explícita.

**Ya se intentó aquí (Engram #2187, 2026-11-18, con entrenamiento autorizado por el usuario)**:
- **Offsets posicionales**: 40 textos propios, 4 688 frames, dos versiones (por distancia y por
  posición), Adam y cross-entropy con cabezas congeladas. Resultado 1,35-1,37 códigos/pasada
  frente a 1,34 del control: **inútil**. El error solo baja de 4,06 a 3,88 nats: un offset aditivo
  no puede imitar lo que la red hace al procesar el código anterior. El 2,47 del artículo
  depende de su **aceptación probabilística**, más permisiva que la coincidencia exacta de nuestro
  `argmax`.
- **Borrador EAGLE** (MLP residual sobre `concat(h_g, E_g[c_g])`): descartado, 1,30 frente a 1,30
  del control, con umbral fijado antes de medir.
- **Árbol con sorteo** (top-k 50, T = 0,325): mejor RTF estimado 0,898 con carga, **marginal**;
  la aceptación baja al subir la temperatura.
- **K=1 con cabeza reutilizada**: **integrado** (commits b89c93e y d783c81), audio bit-idéntico,
  CP con carga 51,6 → 45,6 ms/frame y RTF 0,95 → 0,87.

Aciertos medidos por profundidad desde `h_g`: j=1 **27-41 %**, j=2 16-30 %, j=3 11-24 %,
j=4 8-19 %; la cadena da 1,26-1,36 códigos/pasada y K>2 no aporta.

**Conclusión para este proyecto**: el esquema actual (depth-1, match exacto, cero parámetros
nuevos) es el óptimo para nuestro contrato (greedy + PCM bit-idéntico) y coincide con la línea
base de NVIDIA/Argmax. Las tres vías que podrían subir la aceptación ya están exploradas y
descartadas: offsets entrenados, borradores EAGLE y sorteo con árbol. Lookahead / n-gram son
débiles (el código anterior repite solo 7,3 %). La relajación de aceptación (SSD, PCG, VADUSA)
cambia el token exacto y por tanto el PCM. Quedaría sin probar la self-speculación por salto de
capas (SWIFT, sin entrenamiento), pero el CP tiene solo 5 capas y en CPU la verificación tiene
que repetir el modelo completo: es poco probable que supere el overhead.

### 13.14 Factibilidad de entrenamiento/fine-tuning en CPU

**Viabilidad general** (revisión bibliográfica, 2026-11-25):
- LoRA sobre modelos de 0,5 B en CPU: 5-10 minutos con conjuntos pequeños (NUC Core Ultra 5,
  64 GB RAM, 2,6 M parámetros entrenables, solo PEFT sin GPU).
- Qwen3-TTS 0,6 B con LoRA en Apple M4 Pro vía `mlx-tune`: 2,2 minutos para 200 pasos con
  12 minutos de audio; el rango útil de loss es estrecho (0,48-0,53; por debajo de 0,45
  overfitting y el modelo deja de generar audio).
- El fine-tuning oficial de Qwen3-TTS usa CUDA (`flash-attn`, `bf16`) y en CPU completo sería
  de **días**, no «tranquilo». La guía comunitaria lo resume: «se puede, pero extremadamente
  lento (posiblemente días)».

**Lo ya entrenado aquí** (Engram #2187): offsets posicionales (3 072 parámetros, minutos) →
1,35-1,37 vs 1,34 del control; borrador EAGLE con cabeza congelada → 1,30 vs 1,30. Ambos
descartados con criterio fijado antes de medir.

**Opciones no probadas que sí caben en CPU**:

| Opción | Parámetros | Coste CPU estimado | Riesgo |
|---|---:|---|---|
| Cabezas Medusa con **proyección propia entrenada** (no compartida) | 3 × (1024² + 1024×2048) ≈ 9 M | Minutos-decenas de min con 4 688 frames | Media: autodistilación y verificación en árbol |
| Draft pequeño alineado estilo DistillSpec | ~1-4 M | Horas | Medio: datos on-policy y divergencia a ajustar |
| LoRA sobre el CP para hacer los códigos más predecibles | ~1-8 M | Minutos-horas | Alto: cambia la salida y exige re-basar hashes |

Medusa es la diferencia clave respecto al EAGLE ya probado: allí se usaba la cabeza real
congelada; Medusa entrena una proyección de vocabulario propia por cabeza, inicializada desde
la del modelo, justo porque predecir la posición `t+k` sin condicionar en las intermedias es un
problema distinto.

**Techo realista**: con verificación greedy de coincidencia exacta, la aceptación está limitada
por la entropía intrínseca del siguiente codebook dado `h_g` (medido aquí: j=1 27-41 %, j=2
16-30 %, j=3 11-24 %). Unos cabezales bien entrenados podrían llevar la cadena de 1,26 a
~1,4-1,5 tokens/paso: CP −10-15 % (3-4 ms/frame), RTF ~0,01-0,02. Con el decoder en 39 ms/frame,
no es una mejora significativa.

**Decisión del usuario (2026-11-25)**: se descartan **todas** las opciones de entrenamiento
(LoRA sobre el CP, Medusa con proyecciones propias, DistillSpec y offsets). Ninguna compensa su
coste con beneficio suficiente: el techo es ~0,02 de RTF y el decoder ya marca el suelo. No se
ejecutó ninguna de ellas.

### 13.15 Implementación y medida del kernel sin `vinserti128`

**Experimento aislado** (`kbench/src/bin/vinsert.rs` y `vinsert_full.rs`, fuera del repo). Se
implementaron dos caminos para dos filas Q4_0 con las mismas activaciones:

- **A (actual)**: carga de 16 B por bloque y fila, `vpunpck` de 128 bits + **`vinserti128`**.
- **B (candidata)**: layout emparejado `[s0][s1][qs0(16)][qs1(16)]` = 36 B, una sola carga de
  32 B, `vpand`/`vpsrlw`/`vpunpck` de 256 bits, **sin inserción entre carriles**. Las activaciones
  se duplican por carril con `vbroadcasti128`, y cada carril del acumulador guarda una fila.

**Corrección numérica**: el sesgo Q4_0 se resta **por dword** (`p -= sumq` en los 8 dwords); al
agrupar dos filas quedan 4 dwords por fila, así que el valor pasa de 16 a 32. Verificado contra
referencia escalar: A y B dan exactamente la misma suma (una fila 485 852, otra 371 448).

| Medida | A (actual) | B (candidata) | B/A |
|---|---:|---:|---:|
| Desempaquetado aislado | 3,02 ns/par | 1,90 ns/par | **0,63** |
| Kernel completo, 64 bloques (~2 KiB, L1) | 0,781 ns/bloque | 0,781 ns/bloque | 1,00 |
| Kernel completo, 4 096 bloques (~147 KiB) | 0,891 ns/bloque | 0,488 ns/bloque | **0,55** |
| Kernel completo, 65 536 bloques (~2,4 MiB) | 0,909 ns/bloque | 0,593 ns/bloque | **0,65** |

**Conclusión**: la hipótesis se sostiene en el kernel —el camino sin `vinserti128` es 1,5-1,8×
más rápido cuando el bucle es suficientemente largo—, pero **no se integró en el motor**. Integrar
exige cambiar el layout de `Q4_0Block` en `quant.rs`, los kernels AVX2 y escalar, `dispatch.rs`,
los tests y la ruta NEON (esta última no verificable en esta máquina x86). Con un techo de −1,5 a
−2,5 ms/frame y el decoder ya limitando, el riesgo de regresión no compensa. El repo sigue limpio.

---

**Talker, por capa (INT8 salvo donde se indica)**

```
wq        2048 × 1024 × 1 B = 2 097 152 B = 2,000 MiB
wk, wv    1024 × 1024 × 1 B = 1 048 576 B = 1,000 MiB c/u
wo        1024 × 1024 × 1 B = 1 048 576 B = 1,000 MiB
gate_up   6144 × 1024      → 196 608 bloques × 18 B = 3 538 944 B = 3,375 MiB
down      1024 × 3072      →  98 304 bloques × 18 B = 1 769 472 B = 1,688 MiB
                                                       por capa = 10,06 MiB (Q4_0 en FFN)
                                                       por capa = 14,00 MiB (todo INT8)
```

**Talker completo**: `14 MiB × 28 = 392 MiB` (todo INT8) o `10,06 × 28 = 282 MiB` (FFN Q4_0).

**KV**: `2 (K,V) × 1024 × 2 B = 4 KiB` por posición y capa; `× 28 = 112 KiB` por posición.

**CP, por capa**: `2 + 1 + 1 + 1 (INT8) + 3,375 + 1,688 (Q4_0) = 10,06 MiB`; `× 5 = 50,3 MiB`
por pasada; `× 11,9 = 599 MiB` por frame.

**Decoder**: `436,1 MiB` de pesos F32; tiled `13,5 × 19 + 3,4 × 29 = 354 MiB` por chunk de 10
frames.

**Q4_0**: `18 B / 32 pesos = 0,5625 B/peso` (56,25 % de INT8).
