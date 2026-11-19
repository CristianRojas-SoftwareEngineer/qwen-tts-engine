# Roadmap — Servidor con solicitudes de síntesis concurrentes

El objetivo principal del motor es el **uso local**: una síntesis a la vez en un equipo modesto (laptop, solo CPU, Windows, Linux y macOS), por ejemplo para narrar los mensajes de un agente desde un *hook*. Este documento reúne las optimizaciones que solo aportan cuando el motor sirve a **varios clientes concurrentes**: servidor HTTP con `--batch-size N` (lotes continuos) o `--workers N` (workers independientes), o el modo `--batch` del CLI. Quedan fuera del alcance actual ([ADR-005](docs/adr/005-cpu-inference-optimizations.md)); se retomarán si ese escenario pasa a ser un objetivo.

## Ya resuelto en el camino de lote

- **Microkernels `matmat` INT8/Q4_0**: el lote ya usa los de `cp_step2` (AVX2 y NEON), que leen cada fila de pesos una vez por grupo de hasta 8 columnas (`MM_BATCH_CHUNK`). Con `B > 8`, cada fila se lee `⌈B / 8⌉` veces.
- **Caché KV de lote bajo demanda**: `kv_cap` crece por slot desde `KV_INITIAL_POSITIONS` hasta `kv_max`.

Las mediciones de lote anteriores a los microkernels ya no valen; al retomarlo hay que medir de nuevo el punto de partida (`--batch-bench`, `--matmat-bench` con `QWEN_BATCH_B=2,4,8`, servidor `--batch-size 4` y `--workers 4`).

## Optimizaciones

### Eliminar la doble transposición de activaciones en `batch_proj_q`
- **Problema**: `BatchWorkspace` guarda las activaciones como `[B, cols]`. `batch_proj_q` (`crates/qwen-engine/src/batch.rs`) las transpone a `[cols, B]` (`batch_gather`), `quantize_act_int8_cols` (`crates/qwen-kernels/src/dispatch.rs`) las vuelve a transponer al cuantizar y `batch_scatter` transpone la salida.
- **Propuesta**: cuantizar directamente cada fila contigua `[B, cols]` con `quantize_act_int8` y producir la salida en el layout que consume el microkernel de lote.
- **Efecto esperado**: bajo. La transposición mueve `B × cols × 4` bytes frente a `rows × cols` bytes de pesos, menos del 3 % del tráfico con `rows ≥ 1024`.
- **Contrato numérico**: bit-idéntico.

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
