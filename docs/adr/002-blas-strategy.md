# ADR-002: Estrategia BLAS (OpenBLAS propio y Accelerate)

> Registro de decisión arquitectónica sobre el backend de álgebra lineal densa (`sgemm`) para *prefill* y convoluciones del decoder de audio y del codificador del locutor.

- **Estado**: Aceptado
- **Fecha**: 2026-11-16

---

## Contexto

Las convoluciones de `SpeechDecoder`, el codificador del locutor (ECAPA-TDNN, `voice_clone.rs`) y el *prefill* del `Talker` dependen intensivamente de `sgemm`. El número de hilos de BLAS afecta tanto al orden de reducción en coma flotante como a la contención de CPU frente a `ComputePool`.

---

## Decisión

1. El runtime usa `FfiBlas` directamente, sin abstraer BLAS detrás de un trait (`sgemm` y `set_threads` son métodos de `FfiBlas`).
2. **`FfiBlas`** usa **OpenBLAS** en Linux/Windows y **Accelerate** en macOS, e invoca `openblas_set_num_threads` en las fronteras entre *prefill* y generación.
   - En Windows, el binario fija `OPENBLAS_NUM_THREADS = -j` antes de que se inicialice OpenBLAS (inicializador en `.CRT$XCT`): el OpenBLAS propio también reserva 128 MiB de memoria comprometida (*commit*) por hilo inicial (sin la variable, uno por CPU lógica: +1 539 MiB en el servidor y RTF un 12 % peor).
   - `-j` es la única fuente del número de hilos de BLAS. Cada `set_threads` hace una sola llamada, global al proceso, a `openblas_set_num_threads`, sin pedir más hilos que los iniciales (pedir más colgaba la primera `sgemm`).
3. Linux y Windows (`x86_64-pc-windows-gnu`) enlazan **solo** el OpenBLAS propio estático (`scripts/build-openblas.sh`: 0.3.34, solo BLAS, el servidor de hilos propio de OpenBLAS (pthreads en Linux, hilos Win32 en Windows), `DYNAMIC_ARCH`, sin OpenMP ni Fortran; base `CORE2` en x86_64 y `ARMV8` en ARM64), con `OPENBLAS_LIB_DIR` obligatoria y sin enlace alternativo con el OpenBLAS de la distribución o de MSYS2. `DYNAMIC_ARCH` elige el kernel de cada CPU al arrancar. El de la distribución puede traer solo el kernel genérico, lo que justifica el propio: en un Neoverse V1 con Ubuntu 22.04 (`armv8`), el propio baja el decoder de 13,5 a 9,5 s y el RTF de 1,20–1,33 a 1,12–1,21. `--caps` muestra el kernel elegido (`openblas_get_corename`).
   - Spike en Windows (en la CI, con un solo núcleo en `DYNAMIC_LIST`, para que A y B usaran el mismo kernel): audio idéntico, −129 MiB de memoria comprometida y −31 % de ejecutable; en 4 núcleos, RTF de la frase corta de 1,34 a 1,30 y de la larga de 1,21 a 1,20, y decoder un 7 % más rápido sin carga y un 3 % más lento con carga.
   - Spike en ARM con la 0.3.34 frente a la 0.3.30: audio idéntico, mismo núcleo (`neoversen1`) y RTF sin diferencia consistente.

---

## Consecuencias

### Positivas

- Máximo rendimiento en generación larga, con los hilos de BLAS fijados por `-j`.
- Determinismo validado en `SpeechDecoder` y *prefill*.

### Riesgos y mitigación

- Linux y Windows requieren compilar el OpenBLAS propio una vez (en macOS se usa `Accelerate`); se mitiga con `scripts/build-openblas.sh`, la caché de la CI y un mensaje de `qwen-kernels/build.rs` que explica cómo compilarlo si falta.
