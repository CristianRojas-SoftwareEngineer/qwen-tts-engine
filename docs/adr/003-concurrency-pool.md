# ADR-003: Pool de hilos `ComputePool` y control de sobresuscripción

> Registro de decisión arquitectónica sobre el paralelismo de grano fino para `matvec`/`matmat` cuantizados y la planificación de concurrencia en el servidor HTTP.

- **Estado**: Aceptado
- **Fecha**: 2026-11-16

---

## Contexto

Por cada fotograma de `80 ms`, el `Code Predictor` y el `Talker` ejecutan cientos de `matvec` cortos (microsegundos por llamada). Un pool basado solo en `Condvar` o un runtime global tipo `rayon` introduce latencia de despertar o contención entre peticiones simultáneas.

---

## Decisión

1. Implementar **`ComputePool`** instanciable en `qwen-kernels` (sin estado global estático) usando `AtomicUsize`, espera activa corta (`QWEN_POOL_SPIN=4096` con `std::hint::spin_loop()`), `Condvar` y llamada única a `ftz_on()` al arrancar cada hilo trabajador.
2. En el servidor HTTP (`qwen-app`):
   - Priorizar **`ContinuousBatchScheduler` (`--batch-size N`)** sobre un único `ComputePool` compartido para amortizar lecturas de pesos en DRAM.
   - En **`WorkerPoolScheduler` (`--workers N`)**, dividir los hilos por *worker* (`total_threads / n_workers`) y desactivar el *spin-wait* (`spin_budget = 0`) cuando `workers * worker_threads >= physical_cores`.

---

## Consecuencias

### Positivas

- Baja latencia de despacho en bucles `matvec` cortos.
- Portabilidad nativa en Windows, Linux y macOS sin bifurcaciones de plataforma (`GCD`/`Win32`/`pthread`).
- Aislamiento completo entre instancias de `ComputeRuntime`.

### Riesgos y mitigación

- Requiere mantener lógica propia de sincronización (~250 líneas en `pool.rs`), cubierta por las pruebas de estrés concurrente en `kernels_tests.rs`.
