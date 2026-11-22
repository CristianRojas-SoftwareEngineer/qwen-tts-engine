# ADR-002: Estrategia BLAS (OpenBLAS propio y Accelerate)

> Registro de decisión arquitectónica sobre el backend de álgebra lineal densa (`sgemm`) para *prefill* y convoluciones del decoder/encoder de audio.

- **Estado**: Aceptado
- **Fecha**: 2026-11-16 (revisado el 2026-11-21: OpenBLAS propio 0.3.34 en Linux y Windows)

---

## Contexto

Las convoluciones de `SpeechDecoder`, `SpeechEncoder` y el *prefill* del `Talker` dependen intensivamente de `sgemm`. El número de hilos de BLAS afecta tanto al orden de reducción en coma flotante como a la contención de CPU frente a `ComputePool`.

---

## Decisión

1. El runtime usa `FfiBlas` directamente, sin abstraer BLAS detrás de un trait (`sgemm` y `set_threads` son métodos de `FfiBlas`).
2. **`FfiBlas`** usa **OpenBLAS** en Linux/Windows y **Accelerate** en macOS, e invoca `openblas_set_num_threads` en las fronteras entre *prefill* y generación.
   - En Windows, el binario fija `OPENBLAS_NUM_THREADS = -j` antes de que se inicialice OpenBLAS (inicializador en `.CRT$XCT`): el OpenBLAS propio también reserva 128 MiB de memoria comprometida (*commit*) por hilo inicial (sin la variable, uno por CPU lógica: +1 539 MiB en el servidor y RTF un 12 % peor). Hasta el 2026-11-21 se usaba el OpenBLAS de MSYS2, con OpenMP, y el binario fijaba `OMP_NUM_THREADS`.
   - Hasta el 2026-11-21, si el usuario fijaba alguna de las tres variables de hilos, el binario no tocaba los hilos (`user_fixed_threads`); se retiró para que `-j` sea la única fuente.
   - `set_threads` llama una vez a `openblas_set_num_threads`, que es global al proceso, sin pedir más hilos que los iniciales (pedir más colgaba la primera `sgemm`). Hasta el 2026-11-21 se reaplicaba en cada hilo que llamaba a `sgemm`, porque con el OpenMP de MSYS2 el valor era por hilo.
3. Linux y Windows (`x86_64-pc-windows-gnu`) enlazan **solo** el OpenBLAS propio estático (`scripts/build-openblas.sh`: 0.3.34, solo BLAS, el servidor de hilos propio de OpenBLAS (pthreads en Linux, hilos Win32 en Windows), `DYNAMIC_ARCH`, sin OpenMP ni Fortran; base `CORE2` en x86_64 y `ARMV8` en ARM64; en Windows, más `winpthread` del toolchain de MSYS2), con `OPENBLAS_LIB_DIR` obligatoria y sin enlace alternativo con el OpenBLAS de la distribución o de MSYS2. `DYNAMIC_ARCH` elige el kernel de cada CPU al arrancar. Hasta el 2026-11-21 Windows enlazaba el `libopenblas.a` de MSYS2 UCRT64 (con OpenMP y el runtime de Fortran). El de la distribución puede traer solo el kernel genérico, lo que justifica el propio: en un Neoverse V1 con Ubuntu 22.04 (`armv8`), el propio baja el decoder de 13,5 a 9,5 s y el RTF de 1,20–1,33 a 1,12–1,21. `--caps` muestra el kernel elegido (`openblas_get_corename`).
   - Spike en Windows (en la CI, con un solo núcleo en `DYNAMIC_LIST`, para que A y B usaran el mismo kernel): audio idéntico, −129 MiB de memoria comprometida y −31 % de ejecutable; en 4 núcleos, RTF de la frase corta de 1,34 a 1,30 y de la larga de 1,21 a 1,20, y decoder un 7 % más rápido sin carga y un 3 % más lento con carga.
   - Spike en ARM con la 0.3.34 frente a la 0.3.30: audio idéntico, mismo núcleo (`neoversen1`) y RTF sin diferencia consistente.
4. Hasta el 2026-11-21 existió `PureRustBlas` (`faer`) como opción (`--features faer-blas`, `QWEN_BLAS=faer`). Se retiró porque era más lento (en `0.6B --quant-mixed -j 4`, `RTF 0.72–0.75` frente a `0.70–0.71` de OpenBLAS en texto largo, por la contención entre `Par::rayon` y `ComputePool`; en Linux ARM64, Neoverse N1 de 4 vCPU, 1,67–1,81 frente a 1,54–1,62), no se probaba en la CI y no tenía usuarios.

---

## Consecuencias

### Positivas

- Máximo rendimiento en generación larga sin contención de `rayon`.
- Determinismo validado en `SpeechDecoder` y *prefill*.

### Riesgos y mitigación

- Linux y Windows requieren compilar el OpenBLAS propio una vez (en macOS se usa `Accelerate`); se mitiga con `scripts/build-openblas.sh`, la caché de la CI y un mensaje de `qwen-kernels/build.rs` que explica cómo compilarlo si falta.
