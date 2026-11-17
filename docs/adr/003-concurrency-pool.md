# ADR-003: Pool de Hilos Propio (`ComputePool`) y Control de Sobresuscripción

- **Estado**: Aceptado
- **Fecha**: 2026-11-16

## Contexto
Por cada fotograma de 80 ms, el `Code Predictor` y el `Talker` lanzan cientos de `matvec` diminutos (microsegundos por llamada). Un pool basado en `condvar` pura o `rayon` global introduce latencia de despertar hilos o contención entre sesiones concurrentes, penalizando el RTF y obligando a serializar `--workers` si existe estado global compartido.

## Decisión
1. Implementar **`ComputePool`** instanciable en `qwen-kernels` (sin estado global estático) usando `AtomicUsize`, espera activa corta (`QWEN_POOL_SPIN = 4096` iteraciones con `std::hint::spin_loop()`), `Condvar` y llamada única a `ftz_on()` al crear cada hilo trabajador. No usar `rayon` en el bucle caliente de `matvec`.
2. Para concurrencia en servidor HTTP:
   - Priorizar **`ContinuousBatchScheduler` (`--batch-size N`)** sobre **1 único `ComputePool` compartido** (amortiza el ancho de banda de memoria).
   - En **`WorkerPoolScheduler` (`--workers N`)**, repartir el presupuesto de hilos entre workers (`workers * (j_worker + gen_blas) <= n_physical_cores`) o reducir/desactivar el *spin-wait* si hay sobresuscripción, evitando contención destructiva en DRAM y caché.

## Consecuencias
- **Positivas**: Baja latencia de despacho en bucles `matvec` cortos, portabilidad nativa en Windows/Linux/macOS sin bifurcaciones de plataforma (`GCD`/`Win32`/`pthread`), y seguridad frente a carreras de datos en Rust.
- **Negativas**: ~150–200 líneas de código de sincronización propia en `qwen-kernels` que deben cubrirse con pruebas de estrés concurrente.
