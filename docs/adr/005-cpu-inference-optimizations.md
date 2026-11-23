# ADR-005: Optimización de la inferencia en CPU para uso local

> Registro de las optimizaciones del pipeline de inferencia que se adoptaron, las que se descartaron con su evidencia y el estado de rendimiento resultante en cada plataforma.

- **Estado**: Aceptado
- **Fecha**: 2026-11-19

---

## Contexto

El uso objetivo es **local**: una síntesis a la vez en un equipo modesto (laptop, solo CPU, Windows, Linux y macOS), por ejemplo un servidor residente que narra los mensajes de un agente por `POST /v1/tts/stream`. Las optimizaciones que solo rinden con varios clientes concurrentes quedan fuera de alcance ([ADR-006](006-single-synthesis-path.md)).

El tiempo de una síntesis sigue el modelo:

$$\text{Tiempo total} \approx \text{prefill} + \max(\text{productor}, \text{consumidor}) + \text{drenaje final}$$

- **Productor** (Talker + Code Predictor): genera un frame de 16 códigos, 80 ms de audio. Para tiempo real debe tardar menos de 80 ms por frame.
- **Consumidor** (`SpeechDecoder`): decodifica en paralelo en `sd_pool`, casi todo en `sgemm` de BLAS.
- **Primer audio (TTFA)**: lo gobierna el prefill, no el Code Predictor.

Las cifras «antes → después» de las tablas parten del motor recién portado: en el equipo de referencia (Ryzen 7 5825U, 8 núcleos), RTF 0,82 en frase corta y 0,72 en larga sin carga, por encima de 1 con 1 hilo copiando 256 MiB en bucle; en ARM64, la ruta escalar (RTF ~5,4 en Linux, Neoverse N1 de 4 vCPU, y 1,16–1,23 en macOS, Apple M4 Pro de 6 núcleos).

---

## Método

- Perfilar antes de optimizar algo cuyo coste no se conoce; predecir el efecto antes de medirlo.
- Verificar cada cambio con la herramienta mínima suficiente (`--self-test`, hashes de audio, microbanco o 3 corridas) y hacer un A/B intercalado de extremo a extremo para la decisión.
- Repetir el A/B con carga (1 hilo copiando 256 MiB, `scripts/SyntheticLoad.ps1`): ahí se verifica el tiempo real.
- Una mejora consistente se integra aunque sea pequeña; un cambio que no mejora de forma consistente se revierte.

El protocolo de medición está en [`VERIFICATION_PLAN.md`](../VERIFICATION_PLAN.md) §4.

---

## Decisión

### Adoptadas

| Área | Cambio | Efecto medido |
|---|---|---|
| RAM | Caché KV del Talker con 256 posiciones iniciales que crece bajo demanda (`KV_INITIAL_POSITIONS`) | −84 MiB por sesión |
| RAM | *Staging* F32 de pesos del prefill local y transitorio de `talker_prefill` | Pico de memoria comprometida (*commit*): CLI −60 MiB |
| Hilos | El binario fija `OPENBLAS_NUM_THREADS = -j` antes de que se inicialice OpenBLAS (inicializador en `.CRT$XCT` en Windows, con el OpenBLAS propio 0.3.34, [ADR-002](002-blas-strategy.md)); `set_threads` no pide más hilos que los iniciales | Servidor arrancado sin la variable: −1 539 MiB de memoria comprometida (*commit*), RTF 0,74–0,77 → 0,66–0,69. Pedir más hilos que los iniciales colgaba la primera `sgemm` |
| Primer audio | Prefill con conversión `BF16→F32` por bloques de 4 MiB y una `sgemm` por bloque | TTFA 618 → 493 ms (corta) y 702 → 590 ms (larga) sin carga. Cambia el audio (contrato en [`NUMERIC_CONTRACT.md`](../NUMERIC_CONTRACT.md) §3) |
| Decoder | im2col en la `conv1` k=7 de los bloques residuales 2–3 (una `sgemm` con `K = 7 · canales`) | Decoder −3 / −7 %, drenaje hasta −38 %, SNR 104,7–105,8 dB, ±1 LSB |
| Decoder | Atención del *streaming* con un único kernel, `causal_attention_windowed` | Rendimiento neutro; un único camino aritmético |
| Productor | Microkernel `matmat` INT8/Q4_0 que lee cada fila de pesos una vez para hasta 8 columnas | `matmat` Q4_0 con B=2 −40 a −50 %; RTF corta 0,72 → 0,69 |
| Productor | Verificación especulativa de 1 adivinanza en el Code Predictor (`cp_step2`) | Pasadas de pesos por frame 15 → ~11,9; con carga, CP 51,6 → 45,6 ms/f y RTF 0,95 → 0,87; audio bit-idéntico |
| Productor | Reparto dinámico de filas entre hilos (`parallel_rows`, 4 chunks por hilo) y `codec_head` solo con las filas válidas | Con carga, productor −2,0 a −3,7 ms/f y RTF 0,87–0,88 → 0,83–0,85; audio bit-idéntico |
| Audio | Mismo calendario de chunks y mismas rampas de borde en los cuatro modos | PCM idéntico en CLI, CLI `--stdout`, `/v1/tts` y `/v1/tts/stream`; RTF +0,00 a +0,01 sin *streaming* |
| ARM64 | Todos los kernels en NEON, en dos variantes (`neon-dotprod` con `SDOT` y `neon`), con el orden de suma de AVX2 ([ADR-004](004-simd-baseline.md)) | macOS 1,36–1,56 → 0,25–0,26; Linux N1 5,3 → 1,61–1,69 |
| Hilos | Sin spin-wait en los pools cuando los hilos del productor ocupan todos los núcleos físicos (`spin_budget_for`, [ADR-003](003-concurrency-pool.md)) | 4 núcleos: Linux ARM64 1,61–1,69 → 1,54–1,62, Windows 1,53–2,16 → 1,40–1,83. Con 6 núcleos empeora (0,25 → 0,43), por eso la regla |
| Hilos | Afinidad Win32: un hilo lógico por núcleo físico (`affinity.rs`) | Ryzen 7 5825U, 8 núcleos y 16 hilos, `-j 4`, servidor, frase larga en español, ABBA ×3: sin carga, primer audio 472 → 370 ms y RTF 0,815 → 0,666; con carga, 579 → 489 ms y 1,095 → 0,967; audio idéntico |
| Carga | Cuantización al cargar el modelo sin llamadas a `rintf`/`roundf` por peso (redondeo exacto en el propio código), capas del Talker y del Code Predictor y sus 15 `lm_head` en paralelo con el pool de `-j` hilos, y `gate_up` del Talker cuantizado sin copia intermedia. Importa en la CLI, que carga el modelo en cada síntesis (el servidor lo carga una vez) | Ryzen 7 5825U, `-j 4`, CLI: carga del modelo 2 205 → 624 ms; audio idéntico |
| BLAS | OpenBLAS 0.3.34 propio, estático y con `DYNAMIC_ARCH` en Linux y Windows (`scripts/build-openblas.sh`, [ADR-002](002-blas-strategy.md)) | Neoverse V1 con Ubuntu 22.04 (kernel genérico `armv8`): RTF 1,20–1,33 → 1,12–1,21 y decoder 13,5 → 9,5 s; neutro donde la distribución ya trae `DYNAMIC_ARCH` |

Calendario de chunks único (fila «Mismo calendario de chunks…»): primer chunk de 2 frames (160 ms) y siguientes de `min(chunk_frames, 4 + 2 * (k / 8))`, con los mismos valores con y sin *streaming* porque las fronteras de chunk fijan el redondeo del decoder y así el PCM es idéntico.
- **Coste aceptado**: sin *streaming* se renuncia a los bloques fijos de 10 frames del motor original en C. El decoder trabaja un 4–17 % más y el RTF sube de +0,00 a +0,01 (A/B intercalado).
- **Por qué no bloques de 10 en todos los modos**: retrasaría el primer audio del *streaming* (estimado: ~450 → ~850 ms sin carga). La subida gradual del calendario evita cortes con RTF cercano a 1.
- **Para recuperar esa eficiencia sin perder la identidad entre modos**: haría falta un decoder cuyo redondeo no dependa del tamaño del chunk.

### Descartadas

| Alternativa | Evidencia |
|---|---|
| SwiGLU/SiLU vectorizados y su fusión con la cuantización de `down_proj` | SwiGLU es el 1,7 % del Code Predictor; repartirlo entre hilos ahorró 0,68 ms/f. El coste está en `gate_up` (37 %) y QKV (27 %) |
| `codec_head` en INT8 | 3 MiB por frame, el 0,6 % del tráfico, y cambia la precisión de los logits de `code0` |
| Prefill cuantizado (`matmat` Q4_0 en lugar de BF16→F32) | +4,7 semitonos de tono en una de las voces medidas, WER +1,8 / +2,7 puntos y prefill largo +41 % |
| `sd_pool` sin spin-wait con núcleos libres | Con `-j 4` en 8 núcleos: decoder +44 / +110 ms. Dormir el pool cuesta más que el spin-wait acotado (`4096`) |
| Retirar `ftz_on()` de la entrada de los kernels | ~2 µs por frame frente a ~45 ms; es la única garantía de FTZ/DAZ en los hilos que no la activan al nacer |
| Fusionar despachos del pool en el Code Predictor | Los 4 despachos por capa dependen cada uno del anterior; la espera en la barrera se recupera con el reparto dinámico |
| Prefetch de 5 120 bytes también en Q4_0 | Neutro o peor (RTF 0,94–1,00 → 0,97–1,01 con carga) |
| `matmat` BF16 | 0,76–0,97× frente a `B × matvec` |
| Decoder en BF16; un hilo menos para el decoder | Quitar el decoder entero no acelera el productor de forma consistente |
| Prioridad `AboveNormal` del proceso | La carga compite por la memoria, no por turnos de CPU |
| Atención del Code Predictor en Q4_0 calibrada (tipo GPTQ) | −1,6 ms/f con carga (RTF 0,83 → 0,82), pero de oído cambia la prosodia y el tono; INT8 suena más natural |
| Sorteo en el Code Predictor, borrador entrenado tipo EAGLE, árboles de candidatos | El sorteo cambia la voz; los otros dos, ≤ 0,01 de RTF sobre la verificación especulativa (simulación con datos reales) |
| Reparto dinámico con 2 chunks por hilo | Mejora no consistente |
| `OPENBLAS_CORETYPE` en Linux ARM64 | Sin efecto |
| `-j 2` con 4 núcleos | RTF 0,92 sin carga y 1,31 con carga frente a 1,01 y 1,49 con `-j 4` (una corrida); con carga tampoco llega a tiempo real |

---

## Consecuencias

### Rendimiento resultante

Todas las filas usan `-j 4` y `--seed 4`. El equipo de referencia se midió con voz clonada en español, por la CLI y por el servidor; las máquinas de CircleCI, con la voz `default` en inglés (`scripts/ci-measure-rtf.sh`), solo sin carga. Las dos clases de equipo no se comparan entre sí:

- **Equipo de referencia**: la laptop de desarrollo, el uso real del motor. Es el único medido también con carga (1 hilo copiando 256 MiB en bucle).
- **Máquinas de CircleCI**: máquinas virtuales para cubrir las demás arquitecturas. Los núcleos son los de la mayor máquina disponible en el plan del proyecto: 6 en macOS (`m4pro.medium`, el único Mac), 4 en Linux ARM64 (`arm.large`; `arm.xlarge`, de 8 vCPU, no está en el plan) y 4 en Windows (`large`; `medium`, de 2, no llega a sintetizar). Medidas sin carga.

| Equipo | Núcleos | Qué se midió | RTF sin carga | RTF con carga | Tiempo real |
|---|---|---|---|---|---|
| **Referencia**: laptop Windows x86_64, AMD Ryzen 7 5825U | 8 | Frase larga en español con voz clonada, CLI y servidor (completa y *stream*); mediana de 3 intercalada | 0,66 | 0,93–0,95 | Sí |
| CircleCI macOS, Apple M4 Pro (virtual) | 6 | Frase corta y larga, voz `default` en inglés; 3 corridas de cada una | 0,23–0,26 | sin medir | Sí |
| CircleCI Linux ARM64, Arm Neoverse N1 (Docker `arm.large`) | 4 vCPU | Igual que macOS | 1,46–2,09 | sin medir | No |
| CircleCI Windows x86_64 (`large`) | 4 | Igual que macOS | 0,89–0,98 | sin medir | Sí |

Cómo leer los rangos:
- En la referencia, cada extremo es la mediana de 3 de un modo (CLI, servidor completo o servidor en *stream*).
- En CircleCI, cada extremo es una corrida: va de la más rápida a la más lenta, mezclando la frase corta y la larga.
- La tanda por núcleos (`cores.py`, CLI) es aparte: con 8 núcleos dio 0,68 sin carga y 0,98 con carga.

Detalles del equipo de referencia:
- **Frase corta** (~2 s de audio): 0,76–0,77 sin carga y 1,07–1,09 con carga. Pesan el prefill y el primer chunk: la frase corta pasa de 1 con carga.
- **Primer audio** en el servidor por `/v1/tts/stream`: 354–362 ms sin carga y 452–469 ms con carga.
- **Carga del modelo**: 624 ms.
- **Memoria**: pico de 1 353 MiB de memoria comprometida privada y 2 362 MiB de *working set*.
- **Por núcleos** (afinidad limitada, frase larga, RTF sin carga / con carga): 6 núcleos 0,93 / 1,30; 4 núcleos 1,01 / 1,49; 4 núcleos con `-j 2` 0,92 / 1,31.
- **Ruta escalar** (`QWEN_KERNELS=scalar`): RTF 3,59 sin carga.

### Positivas

- Tiempo real en el equipo objetivo de 8 núcleos con carga moderada y en macOS ARM64 con margen.
- Audio determinista e idéntico entre los cuatro modos de salida; los kernels NEON dan los mismos bits que AVX2 salvo donde interviene `libm`.

### Riesgos y límites

- **Con 4 núcleos no hay tiempo real con carga**; sin carga, x86_64 queda al límite (RTF 0,89–1,01) y Linux ARM64 está por encima de 1 (1,46–2,09). El productor y el decoder se reparten las mismas CPU. La palanca pendiente es ese reparto.
- **Linux ARM64 con 8 vCPU sin verificar**: CircleCI no ofrece a este proyecto máquinas ARM más grandes.
- **El audio depende de la plataforma**: BLAS y `libm` redondean distinto, y el muestreo autorregresivo amplifica la diferencia. En ARM se compara con la ruta escalar de la misma máquina, no por hash con x86.
