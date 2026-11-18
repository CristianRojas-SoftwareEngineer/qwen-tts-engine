# Roadmap — Servidor con solicitudes de síntesis concurrentes

> **Fuera del alcance del plan de optimización vigente.** El plan ([`docs/reviews/001-inference-pipeline-optimization.md`](docs/reviews/001-inference-pipeline-optimization.md)) optimiza el uso local: **una síntesis a la vez** en un equipo modesto (laptop, solo CPU, Windows/Linux/macOS), por ejemplo narrar los mensajes de un agente desde un *hook*. Este documento reúne las optimizaciones que solo aportan cuando el motor sirve a **varios clientes concurrentes** (`--serve` con `--batch-size N` o `--workers N`, o el modo `--batch` del CLI), para retomarlas si ese escenario pasa a ser objetivo.

## Punto de partida (cierre de la Fase 1, 2026-11-18)

| Métrica | Valor |
|---|---|
| `--batch-bench` (B=8, K=32) | 9 659.1 → 7 407.6 ms (**1.31×**) |
| `--matmat-bench` B=8 (mediana) | int8 1.07× · int4 1.03× · bf16 0.94× |
| Servidor `--workers 4` (4 peticiones, texto largo) | pico de commit 1 637.7 MiB · RTF/petición 1.89 |
| Servidor `--batch-size 4` (4 peticiones, texto largo) | pico de commit 2 506.0 MiB (RTF pendiente de medir) |

Evidencia: `.work/plans/optimizations/phase-1/paso_cierre/` y `paso_cierre_v2/`.

## Optimizaciones

### R1. Escalado batch de los microkernels `matmat` (de `O1`)
- **Problema**: con `B` peticiones en lote, `int8_matmat_slice` y `q4_0_matmat_slice` (`crates/qwen-kernels/src/simd/avx2.rs`) releen cada fila de pesos `B` veces (un `matvec` por columna de lote): el tráfico de DRAM crece casi ×`B`.
- **Propuesta**: el microkernel de `O1` (una lectura de pesos por grupo de hasta 8 columnas, acumuladores por columna en registros) ya generaliza a `B = 1…64`. Aquí se trata de validar y aprovechar `B = 4, 8` en `run_continuous_batch`, más el equivalente NEON en ARM64.
- **Contrato**: bit-idéntico (`L2_rel == 0` en INT8 y Q4_0 frente a `B × matvec`).
- **Verificación**: `--matmat-bench` (`QWEN_BATCH_B=2,4,8`), `--batch-bench` y servidor `--batch-size 4`.

### R2. Eliminar la doble transposición en `batch_proj_q` (antes `O2`)
- **Problema**: `BatchWorkspace` guarda activaciones en `[B, cols]`. `batch_proj_q` las transpone a `[cols, B]` (`batch_gather`), `quantize_act_int8_cols` las retranspone en `QXT_TLS` y `batch_scatter` transpone la salida (`crates/qwen-engine/src/batch.rs`, `crates/qwen-kernels/src/dispatch.rs`).
- **Propuesta**: cuantizar directamente cada fila contigua `[B, cols]` con `quantize_act_int8` y alinear la salida con el microkernel de R1.
- **Efecto esperado**: bajo. La transposición mueve `B × cols × 4` B frente a `rows × cols` B de pesos (< 3 % del tráfico con `rows ≥ 1024`).
- **Contrato**: bit-idéntico. **Depende de**: R1.

### R3. Compactar slots activos y una sola barrera de atención por capa (antes `O7` batch)
- **Problema**: `batch_talker_step_impl` invoca `causal_attention_bf16kv_heads_parallel` dentro de `for b in 0..b_dim` (`b_dim` barreras por capa) y proyecta también los slots con `active[b] == false`.
- **Propuesta**: compactar los slots activos (`n_act`) antes de `batch_proj_q` y paralelizar `(slot activo, cabeza)` en una sola barrera por capa.
- **Contrato**: bit-idéntico. **Verificación**: `--batch-test` y servidor `--batch-size 4` con peticiones de longitudes distintas.

### R4. `openblas_set_num_threads` una sola vez con `--workers N > 1` (parte servidor de `O9`)
- **Problema**: `FfiBlas::set_threads` (`crates/qwen-kernels/src/blas.rs`) cambia estado global del proceso; con varios workers (`crates/qwen-app/src/server/scheduler.rs`), estos se pisan entre prefill y generación.
- **Propuesta**: fijar los hilos de OpenBLAS una vez al iniciar cuando `n_workers > 1`, sin reconfigurar por petición. `--batch-size N` (o `faer-blas`) sigue siendo la opción preferente en concurrencia.
- **Contrato**: bit-idéntico por petición con el mismo número efectivo de hilos.

### R5. KV de lote sin *staging* y paginado (derivado de `O11`/`O10c`)
- **Problema**: `generate_batch_multi` necesita un *staging* compacto del KV por slot porque `kv_max` depende del prompt más largo del grupo; `BatchWorkspace` usa el mismo stride (`kv_cap`) para todos los slots.
- **Propuesta**: bloques KV paginados con stride por slot, para cargar directamente desde el scratch y crecer cada slot por separado. Interesa a partir de `B ≥ 16` o con longitudes muy dispares.
- **Contrato**: bit-idéntico.

## Criterio para retomarlo

Retomar este roadmap solo si el servidor con clientes concurrentes pasa a ser un objetivo del producto. Hacerlo con su propia línea base (servidor `--batch-size N` y `--workers N`, RTF agregado y pico de commit) y sin regresión del uso local.
