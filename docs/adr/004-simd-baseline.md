# ADR-004: Despacho SIMD en Runtime sobre Rust Estable

- **Estado**: Aceptado
- **Fecha**: 2026-11-16

## Contexto
El motor C compila una sola ISA por binario (`Makefile`: `-mavx2 -mfma` por defecto en x86, `SIMD=avx512vnni`/`avx512bf16` opt-in, `-march=armv8.2-a+dotprod` en ARM64). Además, varias rutas avanzadas (`AVX-512 BF16`, `ARM i8mm`) usan intrínsecos recientes cuya disponibilidad en Rust estable varía o exige `nightly`/`asm!`.

## Decisión
1. Compilar `qwen-tts-rs` en **Rust estable** (validado con `rustc 1.96.0`) con **despacho en tiempo de ejecución** (`KernelDispatch`) resuelto al arranque mediante `is_x86_feature_detected!` / `CpuCaps`.
2. **Línea base implementada en Rust estable**:
   - **x86-64**: `AVX2 + FMA` (`_mm256_maddubs_epi16`, `_mm256_sign_epi8`, `_mm256_fmadd_ps`, prefetch `5120B`, núcleo de 1 fila para `cols >= 2048`).
   - **Universal (ARM64 y resto de plataformas)**: fallback entero `scalar` (`simd/scalar.rs`) con `ftz_on()` (`FPCR` en `aarch64`) y BLAS nativo (`Accelerate` / `OpenBLAS`), garantizando portabilidad total sin `SIGILL`.
3. **Extensiones avanzadas (`AVX-512 VNNI/BF16`, `NEON+SDOT`, `ARM i8mm`)**: no se incluyen en el núcleo base estable; quedan fuera del camino crítico hasta requerirse como módulos aislados opt-in.

## Consecuencias
- **Positivas**: Un único binario portable por sistema operativo, compilable en Rust estable y validable kernel a kernel con `--self-test`.
- **Negativas**: En ARM64 los matvec cuantizados corren sobre el núcleo escalar entero mientras las GEMM de `SpeechDecoder`/`Talker` prefill usan `Accelerate`/`OpenBLAS`.
