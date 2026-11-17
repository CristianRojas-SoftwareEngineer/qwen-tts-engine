# ADR-002: Estrategia BLAS (`FfiBlas` por defecto y `faer-blas` opcional)

> Registro de decisión arquitectónica sobre el backend de álgebra lineal densa (`sgemm`) para *prefill* y convoluciones del decodificador/codificador de audio.

- **Estado**: Aceptado
- **Fecha**: 2026-11-16

---

## Contexto

Las convoluciones de `SpeechDecoder`, `SpeechEncoder` y el *prefill* del `Talker` dependen intensivamente de `sgemm`. El número de hilos de BLAS afecta tanto al orden de reducción en coma flotante como a la contención de CPU frente a `ComputePool`.

---

## Decisión

1. Abstraer BLAS detrás del trait `BlasBackend` en `qwen-kernels` (`sgemm` y `set_threads`).
2. Usar **`FfiBlas`** como backend por defecto: **OpenBLAS** en Linux/Windows y **Accelerate** en macOS, invocando `openblas_set_num_threads` en las fronteras entre *prefill* y generación (siempre que `OPENBLAS_NUM_THREADS` no esté fijada).
3. En Windows (`x86_64-pc-windows-gnu`), enlazar estáticamente con `libopenblas.a` de MSYS2 UCRT64 (y en Linux preferir `libopenblas.a` estático si existe en `OPENBLAS_LIB_DIR`).
4. Mantener `PureRustBlas` (`faer`) como backend opcional bajo `--features faer-blas` (`QWEN_BLAS=faer`): en `0.6B --quant-mixed -j 4`, OpenBLAS alcanza `RTF 0.70–0.71` frente a `0.72–0.75` de `faer` en texto largo al evitar la contención entre `Par::rayon` y `ComputePool`.

---

## Consecuencias

### Positivas

- Máximo rendimiento en generación larga sin contención de `rayon`.
- Determinismo validado en `SpeechDecoder` y *prefill*.
- Opción de compilación 100 % Rust mediante `--features faer-blas`.

### Riesgos y mitigación

- El perfil por defecto requiere `OpenBLAS` en Linux/Windows (en macOS usa `Accelerate`); se mitiga con el script de build `qwen-kernels/build.rs` y la alternativa opcional `faer-blas`.
