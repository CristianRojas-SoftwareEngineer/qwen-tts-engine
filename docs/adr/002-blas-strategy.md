# ADR-002: Estrategia BLAS (`FfiBlas` inicial vs evaluación futura de `faer`)

- **Estado**: Aceptado
- **Fecha**: 2026-11-16

## Contexto
El motor C eliminó todas las convoluciones y multiplicaciones sin BLAS (`DIVERGENCIAS.md`). `cblas_sgemm` se usa intensivamente en el prefill del `Talker`, en el `SpeechDecoder` (22 llamadas) y en el `SpeechEncoder` (9 llamadas). Además, el número de hilos de BLAS afecta ligeramente el orden de suma en punto flotante y la contención con el pool de cómputo. En este entorno Windows están disponibles `C:\msys64\ucrt64\lib\libopenblas.a` y `cblas.h` (target `x86_64-pc-windows-gnu`), además del toolchain `msvc`.

## Decisión
1. Abstraer BLAS detrás del trait `BlasBackend` en `qwen-kernels` (`sgemm` y `set_threads`).
2. En las Fases 1–5, usar **`FfiBlas`** enlazado a **OpenBLAS** (Linux/Windows) o **Accelerate** (macOS), invocando `openblas_set_num_threads` en fronteras de fase (prefill vs generación) y respetando `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS`.
3. En Windows, soportar compilación con `x86_64-pc-windows-gnu` para enlace estático con `libopenblas.a` de UCRT64 (o librería de importación DLL si se usa `msvc`).
4. Dejar la evaluación de un backend 100 % Rust (`faer`) para la Fase 6, una vez verificada la paridad de audio y RTF contra el C.

## Consecuencias
- **Positivas**: Cero riesgo de regresión en `SpeechDecoder` y prefill durante las fases críticas de paridad; permite aislar errores de kernels vs errores de GEMM.
- **Negativas**: Mantiene una dependencia nativa de enlace (`OpenBLAS`/`Accelerate`) hasta la Fase 6.
