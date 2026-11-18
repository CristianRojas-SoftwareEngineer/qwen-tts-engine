# Roadmap — Servidor con solicitudes de síntesis concurrentes

El objetivo principal del motor es el **uso local**: una síntesis a la vez en un equipo modesto (laptop, solo CPU, Windows, Linux y macOS), por ejemplo para narrar los mensajes de un agente desde un *hook*. Este documento reúne las optimizaciones que solo aportan cuando el motor sirve a **varios clientes concurrentes**: servidor HTTP con `--batch-size N` (lotes continuos) o `--workers N` (workers independientes), o el modo `--batch` del CLI. No forman parte del trabajo de optimización actual; se retomarán si ese escenario pasa a ser un objetivo.

## Punto de partida medido (2026-11-18)

Ryzen 7 5825U, modelo 0.6B, `--quant-mixed -j 4`, texto largo de referencia (~15 s de audio), 4 peticiones concurrentes.

| Métrica | Valor |
|---|---|
| `--batch-bench` (B=8, 32 pasos) | 9 659.1 ms secuencial → 7 407.6 ms en lote (**1.31×**) |
| `--matmat-bench` (B=8, aceleración en mediana) | INT8 1.07× · Q4_0 1.03× · BF16 0.94× |
| `--serve --workers 4` | pico de memoria comprometida 1 637.7 MiB · RTF por petición 1.89 |
| `--serve --batch-size 4` | pico de memoria comprometida 2 506.0 MiB · RTF por petición sin medir |

La aceleración en lote es baja (1.31× con 8 peticiones) porque cada petición del lote vuelve a leer los pesos del modelo desde memoria.

## Optimizaciones

### Microkernels `matmat` que lean los pesos una sola vez por lote
- **Problema**: con `B` peticiones en lote, `int8_matmat_slice` y `q4_0_matmat_slice` (`crates/qwen-kernels/src/simd/avx2.rs`) ejecutan un producto matriz-vector por cada columna del lote, de modo que cada fila de pesos se lee `B` veces y el tráfico de DRAM crece casi ×`B`.
- **Propuesta**: un microkernel AVX2 que cargue cada bloque de pesos una sola vez (decodificando Q4_0 una vez por bloque) y acumule las `B` columnas en registros independientes, en grupos de hasta 8. Equivalente NEON para ARM64.
- **Contrato numérico**: bit-idéntico al producto matriz-vector (`L2_rel == 0` en INT8 y Q4_0).
- **Verificación**: `--self-test`, `--matmat-bench` con `QWEN_BATCH_B=2,4,8`, `--batch-bench` y servidor `--batch-size 4`.

### Eliminar la doble transposición de activaciones en `batch_proj_q`
- **Problema**: `BatchWorkspace` guarda las activaciones como `[B, cols]`. `batch_proj_q` (`crates/qwen-engine/src/batch.rs`) las transpone a `[cols, B]` (`batch_gather`), `quantize_act_int8_cols` (`crates/qwen-kernels/src/dispatch.rs`) las vuelve a transponer al cuantizar y `batch_scatter` transpone la salida.
- **Propuesta**: cuantizar directamente cada fila contigua `[B, cols]` con `quantize_act_int8` y producir la salida en el layout que consume el microkernel de lote.
- **Efecto esperado**: bajo. La transposición mueve `B × cols × 4` bytes frente a `rows × cols` bytes de pesos, menos del 3 % del tráfico con `rows ≥ 1024`.
- **Contrato numérico**: bit-idéntico. **Depende de**: los microkernels de lote.

### Compactar los slots activos y usar una sola barrera de atención por capa
- **Problema**: `batch_talker_step_impl` (`crates/qwen-engine/src/batch.rs`) invoca `causal_attention_bf16kv_heads_parallel` dentro de `for b in 0..b_dim`, con una barrera del pool por slot y capa, y proyecta también los slots inactivos (`active[b] == false`).
- **Propuesta**: compactar los slots activos antes de las proyecciones y paralelizar los pares `(slot activo, cabeza)` en una sola barrera por capa.
- **Contrato numérico**: bit-idéntico. **Verificación**: `--batch-test` y servidor `--batch-size 4` con peticiones de longitudes distintas.

### Configurar los hilos de OpenBLAS una sola vez con varios workers
- **Problema**: `FfiBlas::set_threads` (`crates/qwen-kernels/src/blas.rs`) modifica estado global del proceso. Con `--workers N > 1` (`crates/qwen-app/src/server/scheduler.rs`), los workers se lo pisan entre prefill y generación.
- **Propuesta**: fijar los hilos de OpenBLAS una vez al arrancar cuando hay varios workers, sin reconfigurarlos por petición. En concurrencia, `--batch-size N` sigue siendo la opción preferente.
- **Contrato numérico**: bit-idéntico por petición con el mismo número efectivo de hilos.

### Caché KV de lote paginada y sin copia intermedia
- **Problema**: `generate_batch_multi` necesita una copia intermedia compacta del KV de cada slot, porque la capacidad del lote depende del prompt más largo del grupo y solo se conoce tras prefilar todos los slots. `BatchWorkspace` usa además la misma capacidad (`kv_cap`) para todos los slots.
- **Propuesta**: bloques de KV paginados, con capacidad propia por slot, para cargar directamente desde la caché de sesión y crecer cada slot por separado. Interesa a partir de `B ≥ 16` o con peticiones de longitudes muy distintas.
- **Contrato numérico**: bit-idéntico.

## Criterio para retomarlo

Solo si el servidor con clientes concurrentes pasa a ser un objetivo del producto. Debe medirse contra su propia línea base (servidor `--batch-size N` y `--workers N`: RTF agregado y pico de memoria comprometida) y sin regresión del uso local.
