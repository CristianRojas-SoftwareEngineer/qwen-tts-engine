# ADR-005: Optimización de la inferencia en CPU para uso local

> Registro de las optimizaciones del pipeline de inferencia que se adoptaron, las que se descartaron con su evidencia y el estado de rendimiento resultante en cada plataforma.

- **Estado**: Aceptado
- **Fecha**: 2026-11-19

---

## Contexto

El uso objetivo es **local**: una síntesis a la vez en un equipo modesto (laptop, solo CPU, Windows, Linux y macOS), por ejemplo un servidor residente que narra los mensajes de un agente por `POST /v1/tts/stream`. Las optimizaciones que solo rinden con varios clientes concurrentes están en [`ROADMAP.md`](../../ROADMAP.md).

El tiempo de una síntesis sigue el modelo:

$$\text{Tiempo total} \approx \text{prefill} + \max(\text{productor}, \text{consumidor}) + \text{drenaje final}$$

- **Productor** (Talker + Code Predictor): genera un fotograma de 16 códigos, 80 ms de audio. Para tiempo real debe tardar menos de 80 ms por fotograma.
- **Consumidor** (`SpeechDecoder`): decodifica en paralelo en `sd_pool`, casi todo en `sgemm` de BLAS.
- **Primer audio (TTFA)**: lo gobierna el prefill, no el Code Predictor.

Punto de partida, Ryzen 7 5825U (8 núcleos), `--quant-mixed -j 4 --seed 4 --stream`: RTF 0,82 en frase corta y 0,72 en larga sin carga, por encima de 1 con 1 hilo copiando 256 MiB en bucle. En ARM64 todo pasaba por la ruta escalar: RTF ~5,4 en Linux (Neoverse N1, 4 vCPU) y 1,16–1,23 en macOS (Apple M4 Pro, 6 núcleos).

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
| RAM | *Staging* F32 de pesos del prefill local y transitorio de `talker_prefill` | Pico de commit: CLI −60 MiB, `--workers 4` −241 MiB |
| RAM | KV de `BatchWorkspace` bajo demanda (`kv_cap` hasta `kv_max`) y copia compacta del KV en `generate_batch_multi` | `--batch-size 4` −377 MiB; `--batch-multi-test` −170 a −206 MiB |
| Hilos | El binario fija `OMP_NUM_THREADS = -j` antes de que se inicialice OpenBLAS (inicializador en `.CRT$XCT` en Windows); `set_threads` no pide más hilos que los iniciales | Servidor arrancado sin la variable: −1 539 MiB de commit, RTF 0,74–0,77 → 0,66–0,69. Pedir más hilos que los iniciales colgaba la primera `sgemm` |
| Primer audio | Prefill con conversión `BF16→F32` por bloques de 4 MiB y una `sgemm` por bloque | TTFA 618 → 493 ms (corta) y 702 → 590 ms (larga) sin carga. Cambia el audio (contrato en [`NUMERIC_CONTRACT.md`](../NUMERIC_CONTRACT.md) §3) |
| Decoder | im2col en la `conv1` k=7 de los bloques residuales 2–3 (una `sgemm` con `K = 7 · canales`) | Decoder −3 / −7 %, drenaje hasta −38 %, SNR ≥ 104 dB |
| Decoder | Atención del *streaming* por el mismo kernel que la decodificación completa | Rendimiento neutro; un único camino aritmético |
| Productor | Microkernel `matmat` INT8/Q4_0 que lee cada fila de pesos una vez para hasta 8 columnas | `matmat` Q4_0 con B=2 −40 a −50 %; RTF corta 0,72 → 0,69 |
| Productor | Verificación especulativa de 1 adivinanza en el Code Predictor (`cp_step2`) | Pasadas de pesos por fotograma 15 → ~11,9; con carga, CP 51,6 → 45,6 ms/f y RTF 0,95 → 0,87; audio bit-idéntico |
| Productor | Reparto dinámico de filas entre hilos (`parallel_rows`, 4 trozos por hilo) y `codec_head` solo con las filas válidas | Con carga, productor −2,0 a −3,7 ms/f y RTF 0,87–0,88 → 0,83–0,85; audio bit-idéntico |
| Audio | Mismo calendario de fragmentos y mismas rampas de borde en los cuatro modos | PCM idéntico en CLI, CLI `--stream`, `/v1/tts` y `/v1/tts/stream`; RTF +0,00 a +0,01 sin *streaming* |
| ARM64 | Todos los kernels en NEON, en dos variantes (`neon-dotprod` con `SDOT` y `neon`), con el orden de suma de AVX2 ([ADR-004](004-simd-baseline.md)) | macOS 1,36–1,56 → 0,25–0,26; Linux N1 5,3 → 1,61–1,69 |
| Hilos | Sin espera activa en los pools cuando los hilos del productor ocupan todos los núcleos físicos (`spin_budget_for`, [ADR-003](003-concurrency-pool.md)) | 4 núcleos: Linux ARM64 1,61–1,69 → 1,54–1,62, Windows 1,53–2,16 → 1,40–1,83. Con 6 núcleos empeora (0,25 → 0,43), por eso la regla |
| BLAS | OpenBLAS 0.3.30 propio, estático y con `DYNAMIC_ARCH` en Linux (`scripts/build-openblas.sh`, [ADR-002](002-blas-strategy.md)) | Neoverse V1 con Ubuntu 22.04 (kernel genérico `armv8`): RTF 1,20–1,33 → 1,12–1,21 y decoder 13,5 → 9,5 s; neutro donde la distribución ya trae `DYNAMIC_ARCH` |

### Descartadas

| Alternativa | Evidencia |
|---|---|
| SwiGLU/SiLU vectorizados y su fusión con la cuantización de `down_proj` | SwiGLU es el 1,7 % del Code Predictor; repartirlo entre hilos ahorró 0,68 ms/f. El coste está en `gate_up` (37 %) y QKV (27 %) |
| `codec_head` en INT8 | 3 MiB por fotograma, el 0,6 % del tráfico, y cambia la precisión de los logits de `code0` |
| Prefill cuantizado (`matmat` Q4_0 en lugar de BF16→F32) | +4,7 semitonos en `ryan`, WER +1,8 / +2,7 puntos y prefill largo +41 % |
| Cuantizar en paralelo al cargar el modelo | El servidor residente carga el modelo una vez |
| `sd_pool` sin espera activa con núcleos libres | Con `-j 4` en 8 núcleos: decoder +44 / +110 ms. Dormir el pool cuesta más que la espera acotada (`4096`) |
| Retirar `ftz_on()` de la entrada de los kernels | ~2 µs por fotograma frente a ~45 ms; es la única garantía de FTZ/DAZ en los hilos que no la activan al nacer |
| Fusionar despachos del pool en el Code Predictor | Los 4 despachos por capa dependen cada uno del anterior; la espera en la barrera se recupera con el reparto dinámico |
| Prefetch de 5 120 bytes también en Q4_0 | Neutro o peor (RTF 0,94–1,00 → 0,97–1,01 con carga) |
| `matmat` BF16 | 0,76–0,97× frente a `B × matvec` |
| Decoder en BF16; un hilo menos para el decoder | Quitar el decoder entero no acelera el productor de forma consistente |
| Prioridad `AboveNormal` del proceso | La carga compite por la memoria, no por turnos de CPU |
| Atención del Code Predictor en Q4_0 calibrada (tipo GPTQ) | −1,6 ms/f con carga (RTF 0,83 → 0,82), pero de oído cambia la prosodia y el tono; INT8 suena más natural |
| Sorteo en el Code Predictor, borrador entrenado tipo EAGLE, árboles de candidatos | El sorteo cambia la voz; los otros dos, ≤ 0,01 de RTF sobre la verificación especulativa (simulación con datos reales) |
| Reparto dinámico con 2 trozos por hilo | Mejora no consistente |
| `QWEN_BLAS=faer` en Linux ARM64 | RTF 1,67–1,81 frente a 1,54–1,62 con OpenBLAS, decoder más lento |
| `OPENBLAS_CORETYPE`, `QWEN_SD_THREADS=2` en Linux ARM64; `-j 2` con 4 núcleos | Sin efecto o neutros; `-j 2` no baja de RTF 1 |

---

## Consecuencias

### Rendimiento resultante (`--quant-mixed -j 4`, sin carga salvo que se indique)

| Plataforma | RTF | Tiempo real |
|---|---|---|
| Windows x86_64, Ryzen 7 5825U (8 núcleos) | Frase larga 0,58–0,67 sin carga y 0,83–0,92 con carga; frase corta 0,63–0,80 y 0,90–1,14 | Sí, salvo frases muy cortas con carga (pesan prefill y primer fragmento) |
| macOS, Apple M4 Pro virtual (6 núcleos) | 0,25–0,26 | Sí |
| Linux ARM64, Neoverse V1 (4 vCPU) | 1,12–1,21 | No: el productor tarda ~85 ms/f |
| Linux ARM64, Neoverse N1 (4 vCPU) | 1,54–1,62 | No |
| Windows x86_64 con 4 núcleos (CircleCI) | 1,40–1,83 | No |

TTFA en el servidor por `/v1/tts/stream`: 442–644 ms sin carga y 548–816 ms con carga (8 núcleos).

### Positivas

- Tiempo real en el equipo objetivo de 8 núcleos con carga moderada y en macOS ARM64 con margen.
- Audio determinista e idéntico entre los cuatro modos de salida; los kernels NEON dan los mismos bits que AVX2 salvo donde interviene `libm`.

### Riesgos y límites

- **Con 4 núcleos no hay tiempo real** en ninguna arquitectura: el productor y el decoder se reparten las mismas CPU. La palanca pendiente es ese reparto.
- **Linux ARM64 con 8 vCPU sin verificar**: CircleCI no ofrece a este proyecto máquinas ARM más grandes.
- **El audio depende de la plataforma**: BLAS y `libm` redondean distinto, y el muestreo autorregresivo amplifica la diferencia. En ARM se compara con la ruta escalar de la misma máquina, no por hash con x86.
