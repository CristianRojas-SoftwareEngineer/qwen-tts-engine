# ADR-002: Estrategia BLAS (`FfiBlas` por defecto y `faer-blas` opcional)

> Registro de decisión arquitectónica sobre el backend de álgebra lineal densa (`sgemm`) para *prefill* y convoluciones del decodificador/codificador de audio.

- **Estado**: Aceptado
- **Fecha**: 2026-11-16 (revisado el 2026-11-19: hilos de OpenMP y OpenBLAS propio en Linux)

---

## Contexto

Las convoluciones de `SpeechDecoder`, `SpeechEncoder` y el *prefill* del `Talker` dependen intensivamente de `sgemm`. El número de hilos de BLAS afecta tanto al orden de reducción en coma flotante como a la contención de CPU frente a `ComputePool`.

---

## Decisión

1. Abstraer BLAS detrás del trait `BlasBackend` en `qwen-kernels` (`sgemm` y `set_threads`).
2. Usar **`FfiBlas`** como backend por defecto: **OpenBLAS** en Linux/Windows y **Accelerate** en macOS, invocando `openblas_set_num_threads` en las fronteras entre *prefill* y generación (siempre que `OPENBLAS_NUM_THREADS` no esté fijada).
   - En Windows, el binario fija `OMP_NUM_THREADS = -j` antes de que se inicialice OpenBLAS (inicializador en `.CRT$XCT`): con el OpenBLAS de MSYS2, que usa OpenMP, OpenBLAS reserva 128 MiB de commit por hilo inicial (sin la variable, uno por CPU lógica: +1 539 MiB en el servidor y RTF un 12 % peor).
   - `set_threads` no pide más hilos que los iniciales (pedir más colgaba la primera `sgemm`) y los aplica en cada hilo que llama a `sgemm`, porque con OpenMP el valor es por hilo.
3. En Windows (`x86_64-pc-windows-gnu`), enlazar estáticamente con `libopenblas.a` de MSYS2 UCRT64. En Linux, enlazar estático si `OPENBLAS_LIB_DIR` contiene `libopenblas.a`, y recomendar un OpenBLAS propio (`scripts/build-openblas.sh`: 0.3.30, solo BLAS, hilos POSIX, `DYNAMIC_ARCH`), que elige el kernel de cada CPU al arrancar. El de la distribución puede traer solo el kernel genérico: en un Neoverse V1 con Ubuntu 22.04 (`armv8`), el propio baja el decoder de 13,5 a 9,5 s y el RTF de 1,20–1,33 a 1,12–1,21. `--caps` muestra el kernel elegido (`openblas_get_corename`).
4. Mantener `PureRustBlas` (`faer`) como backend opcional bajo `--features faer-blas` (`QWEN_BLAS=faer`): en `0.6B --quant-mixed -j 4`, OpenBLAS alcanza `RTF 0.70–0.71` frente a `0.72–0.75` de `faer` en texto largo al evitar la contención entre `Par::rayon` y `ComputePool`; en Linux ARM64 (Neoverse N1, 4 vCPU), `faer` da 1,67–1,81 frente a 1,54–1,62.

---

## Consecuencias

### Positivas

- Máximo rendimiento en generación larga sin contención de `rayon`.
- Determinismo validado en `SpeechDecoder` y *prefill*.
- Opción de compilación 100 % Rust mediante `--features faer-blas`.

### Riesgos y mitigación

- El perfil por defecto requiere `OpenBLAS` en Linux/Windows (en macOS usa `Accelerate`); se mitiga con el script de build `qwen-kernels/build.rs` y la alternativa opcional `faer-blas`.
