# ADR-004: Despacho SIMD en runtime sobre Rust estable

> Registro de decisión arquitectónica sobre la selección dinámica de kernels SIMD y la portabilidad multiplataforma mediante un único binario compilado en Rust estable.

- **Estado**: Aceptado
- **Fecha**: 2026-11-16

---

## Contexto

Compilar un binario atado a una única ISA SIMD obliga a distribuir variantes por microarquitectura y expone a fallos `SIGILL` en CPUs incompatibles. Extensiones recientes (`AVX-512 BF16`, `ARM i8mm`) además presentan soporte heterogéneo en Rust estable.

---

## Decisión

1. Compilar **Qwen TTS Engine** (`qwen-tts-engine`) en **Rust estable** con despacho en runtime (`KernelDispatch`) resuelto al inicializar `ComputeRuntime` mediante `CpuCaps::detect()`.
2. Mantener estas rutas en Rust estable, todas con la tabla completa de `KernelDispatch`:
   - **x86-64 (`avx2`)**: `_mm256_maddubs_epi16`, `_mm256_sign_epi8`, `_mm256_fmadd_ps`, *prefetch* `5120 B` y kernel de 1 fila para `cols >= 2048`.
   - **aarch64 (`neon-dotprod` y `neon`)**: `simd/neon.rs`, con el mismo orden de suma que AVX2 (cada registro de 8 `f32` es un par de registros NEON), así que da los mismos bits salvo donde interviene `libm`. Q4_0 e INT8 usan `SDOT` si la CPU tiene `dotprod` (emitido con `asm!`, porque `vdotq_s32` no es estable en Rust 1.96) y multiplicación ampliada `SMULL` si no (ARMv8.0). El resto de kernels es NEON base, común a las dos.
   - **Escalar (`scalar`)**: `simd/scalar.rs`, para x86-64 sin AVX2 y como referencia numérica. `QWEN_KERNELS=scalar` la fuerza en cualquier CPU, como control.
3. `KernelDispatch::available` lista las rutas que puede ejecutar la CPU (la escalar primero) y `for_caps` elige la última. `--self-test` compara cada ruta SIMD con la escalar.
4. Dejar fuera `AVX-512 VNNI/BF16` y `ARM i8mm`, de soporte heterogéneo en Rust estable, hasta incorporarse como tablas nuevas.

---

## Consecuencias

### Positivas

- Un único binario portable por sistema operativo, compilable en Rust estable y verificable kernel a kernel con `--self-test`.

### Riesgos y mitigación

- Cada kernel existe tres veces (AVX2, NEON y escalar). Las pruebas `test_simd_q4_0_matches_avx2_summation_order` y `test_simd_float_kernels_match_avx2_summation_order` fijan el hash de AVX2, así que un cambio en una ruta que no se replique en la otra falla en la CI de ARM64 o de x86.
- Las GEMM del prefill y del `SpeechDecoder` están en BLAS (`Accelerate`/`OpenBLAS`), que redondea distinto en cada plataforma: el audio de ARM no es el de x86.
