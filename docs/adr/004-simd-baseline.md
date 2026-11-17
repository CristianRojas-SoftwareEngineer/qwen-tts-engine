# ADR-004: Despacho SIMD en runtime sobre Rust estable

> Registro de decisión arquitectónica sobre la selección dinámica de kernels SIMD y la portabilidad multiplataforma sin `nightly` ni binarios separados.

- **Estado**: Aceptado
- **Fecha**: 2026-11-16

---

## Contexto

Compilar un binario atado a una única ISA SIMD obliga a distribuir variantes por microarquitectura y expone a fallos `SIGILL` en CPUs incompatibles. Extensiones recientes (`AVX-512 BF16`, `ARM i8mm`) además presentan soporte heterogéneo en Rust estable.

---

## Decisión

1. Compilar **Qwen TTS Engine** (`qwen-tts-engine`) en **Rust estable** con despacho en runtime (`KernelDispatch`) resuelto al inicializar `ComputeRuntime` mediante `CpuCaps::detect()`.
2. Mantener dos rutas activas en Rust estable:
   - **x86-64 (`AVX2 + FMA`)**: `_mm256_maddubs_epi16`, `_mm256_sign_epi8`, `_mm256_fmadd_ps`, *prefetch* `5120 B` y núcleo de 1 fila para `cols >= 2048`.
   - **Universal (ARM64 y fallback x86)**: núcleo escalar entero (`simd/scalar.rs`) con `ftz_on()` (`FPCR` en `aarch64`) y BLAS nativo (`Accelerate` / `OpenBLAS`).
3. Dejar extensiones avanzadas (`AVX-512 VNNI/BF16`, `NEON+SDOT`) fuera del núcleo base hasta incorporarse como módulos opt-in aislados.

---

## Consecuencias

### Positivas

- Un único binario portable por sistema operativo, compilable en Rust estable y verificable kernel a kernel con `--self-test`.

### Riesgos y mitigación

- En ARM64 los `matvec` cuantizados ejecutan el núcleo escalar entero mientras las GEMM de `SpeechDecoder` y el *prefill* usan `Accelerate`/`OpenBLAS`; la tabla `KernelDispatch` permite añadir nuevos backends SIMD sin tocar `qwen-engine`.
