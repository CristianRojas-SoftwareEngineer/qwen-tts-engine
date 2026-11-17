# ADR-002: Estrategia BLAS (`FfiBlas` por defecto y `faer-blas` opcional)

- **Estado**: Aceptado
- **Fecha**: 2026-11-16

## Contexto
Las convoluciones del `SpeechDecoder`, `SpeechEncoder` y el prefill del `Talker` dependen intensivamente de `sgemm`. Además, el número de hilos de BLAS afecta al orden de suma en punto flotante y a la contención con `ComputePool`. En Windows UCRT64 están disponibles `libopenblas.a` y el toolchain GNU (`x86_64-pc-windows-gnu`).

## Decisión
1. Abstraer BLAS detrás del trait `BlasBackend` en `qwen-kernels` (`sgemm` y `set_threads`).
2. Usar **`FfiBlas`** como backend por defecto, enlazado a **OpenBLAS** (Linux/Windows) o **Accelerate** (macOS), invocando `openblas_set_num_threads` en fronteras de fase (prefill vs generación) y respetando `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS`.
3. En Windows, soportar compilación con `x86_64-pc-windows-gnu` para enlace estático con `libopenblas.a` de UCRT64 (o librería de importación DLL en `msvc`).
4. Mantener **`FfiBlas` como backend por defecto** frente a `PureRustBlas` (`faer`, disponible bajo `--features faer-blas` / `QWEN_BLAS=faer`): en `0.6B --quant-mixed -j 4`, OpenBLAS logra `RTF 0.70–0.71` vs `0.72–0.75` de `faer` en texto largo al evitar la contención entre `Par::rayon` y `ComputePool`.

## Consecuencias
- **Positivas**: Máximo rendimiento en generación larga sin contención de `rayon`, determinismo validado en `SpeechDecoder` y prefill, y posibilidad de compilar 100 % Rust mediante `--features faer-blas`.
- **Negativas**: El perfil por defecto mantiene una dependencia de enlace del sistema (`OpenBLAS`/`Accelerate`).
