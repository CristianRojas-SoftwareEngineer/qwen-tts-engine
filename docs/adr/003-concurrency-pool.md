# ADR-003: Pool de hilos `ComputePool` y control de sobresuscripción

> Registro de decisión arquitectónica sobre el paralelismo de grano fino para `matvec`/`matmat` cuantizados.

- **Estado**: Aceptado
- **Nota**: el 2026-11-20 se retiró la planificación de concurrencia del servidor, que tiene un único trabajador; la antigua decisión sobre `ContinuousBatchScheduler` y `WorkerPoolScheduler` queda como historia (abajo).
- **Fecha**: 2026-11-16 (revisado el 2026-11-19: spin-wait según núcleos y pánicos de los trabajadores)

---

## Contexto

Por cada frame de `80 ms`, el `Code Predictor` y el `Talker` ejecutan cientos de `matvec` cortos (microsegundos por llamada). Un pool basado solo en `Condvar` o un runtime global tipo `rayon` introduce latencia de despertar o contención entre peticiones simultáneas.

---

## Decisión

1. Implementar **`ComputePool`** instanciable en `qwen-kernels` (sin estado global estático) usando `AtomicUsize`, spin-wait corto (4096 iteraciones con `std::hint::spin_loop()`), `Condvar` y llamada única a `ftz_on()` al arrancar cada hilo trabajador. También reparte la cuantización de las capas al cargar el modelo.
2. Historial (retirado el 2026-11-20): el servidor HTTP priorizaba un `ContinuousBatchScheduler` (`--batch-size N`) sobre un único `ComputePool` compartido, para amortizar lecturas de pesos en DRAM, y un `WorkerPoolScheduler` (`--workers N`) que dividía los hilos por *trabajador* (`total_threads / n_workers`). Hoy el servidor tiene un único trabajador y ninguno de los dos existe.
3. Desactivar el spin-wait (`spin_budget = 0`) cuando los hilos de cálculo ocupan todos los núcleos físicos (`spin_budget_for(threads, cores)`), tanto en la CLI como en el servidor: con 4 núcleos, los hilos que giran quitan CPU a los que calculan (RTF 1,61–1,69 → 1,54–1,62 en Linux ARM64 y 1,53–2,16 → 1,40–1,83 en Windows); con 6 núcleos, quitarla empeora (0,25 → 0,43 en un Apple M4 Pro).
4. Si una tarea entra en pánico en un hilo trabajador, el trabajador guarda el pánico y sigue vivo, y `ComputePool::parallel` lo relanza en quien llama al terminar la tarea.

---

## Consecuencias

### Positivas

- Baja latencia de despacho en bucles `matvec` cortos.
- Portabilidad nativa en Windows, Linux y macOS sobre `std::thread` y `std::sync` con una única implementación compartida.
- Aislamiento completo entre instancias de `ComputeRuntime`.

### Riesgos y mitigación

- Requiere mantener lógica propia de sincronización (`pool.rs`), cubierta por las pruebas de estrés concurrente, de pánicos (`test_compute_pool_worker_panic_propagates`) y de la regla de espera (`test_spin_budget_for_cores`) en `kernels_tests.rs`.
- `spin_budget_for` recibe como núcleos los físicos en Windows (los de la afinidad) y los hilos lógicos fuera de Windows, que coinciden con los físicos en ARM64. En Linux x86_64 con SMT, el resultado solo cambia si `-j` es mayor o igual que los núcleos físicos y menor que los hilos lógicos.
