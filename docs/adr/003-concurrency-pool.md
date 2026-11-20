# ADR-003: Pool de hilos `ComputePool` y control de sobresuscripción

> Registro de decisión arquitectónica sobre el paralelismo de grano fino para `matvec`/`matmat` cuantizados y la planificación de concurrencia en el servidor HTTP.

- **Estado**: Aceptado
- **Nota**: Retirado el 2026-11-20: el servidor tiene un único trabajador; los puntos de la decisión sobre `ContinuousBatchScheduler` y `WorkerPoolScheduler` quedan solo como historia.
- **Fecha**: 2026-11-16 (revisado el 2026-11-19: espera activa según núcleos y pánicos de los trabajadores)

---

## Contexto

Por cada fotograma de `80 ms`, el `Code Predictor` y el `Talker` ejecutan cientos de `matvec` cortos (microsegundos por llamada). Un pool basado solo en `Condvar` o un runtime global tipo `rayon` introduce latencia de despertar o contención entre peticiones simultáneas.

---

## Decisión

1. Implementar **`ComputePool`** instanciable en `qwen-kernels` (sin estado global estático) usando `AtomicUsize`, espera activa corta (`QWEN_POOL_SPIN=4096` con `std::hint::spin_loop()`), `Condvar` y llamada única a `ftz_on()` al arrancar cada hilo trabajador.
2. En el servidor HTTP (`qwen-app`):
   - Priorizar **`ContinuousBatchScheduler` (`--batch-size N`, retirado)** sobre un único `ComputePool` compartido para amortizar lecturas de pesos en DRAM.
   - En **`WorkerPoolScheduler` (`--workers N`, retirado)**, dividir los hilos por *worker* (`total_threads / n_workers`).
3. Desactivar la espera activa (`spin_budget = 0`) cuando los hilos de cálculo ocupan todos los núcleos físicos (`spin_budget_for(threads, cores)`), tanto en la CLI como en el servidor: con 4 núcleos, los hilos que giran quitan CPU a los que calculan (RTF 1,61–1,69 → 1,54–1,62 en Linux ARM64 y 1,53–2,16 → 1,40–1,83 en Windows); con 6 núcleos, quitarla empeora (0,25 → 0,43 en un Apple M4 Pro). `QWEN_POOL_SPIN` fija el valor a mano.
4. Si una tarea entra en pánico en un hilo trabajador, el trabajador guarda el pánico y sigue vivo, y `ComputePool::parallel` lo relanza en quien llama al terminar la tarea.

---

## Consecuencias

### Positivas

- Baja latencia de despacho en bucles `matvec` cortos.
- Portabilidad nativa en Windows, Linux y macOS sobre `std::thread` y `std::sync` con una única implementación compartida.
- Aislamiento completo entre instancias de `ComputeRuntime`.

### Riesgos y mitigación

- Requiere mantener lógica propia de sincronización (`pool.rs`), cubierta por las pruebas de estrés concurrente, de pánicos (`test_compute_pool_worker_panic_propagates`) y de la regla de espera (`test_spin_budget_for_cores`) en `kernels_tests.rs`.
