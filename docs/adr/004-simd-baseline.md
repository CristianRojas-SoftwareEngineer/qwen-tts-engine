# ADR-004: Despacho SIMD en Runtime sobre Rust Estable

- **Estado**: Aceptado
- **Fecha**: 2026-11-16

## Contexto
El motor C compila una sola ISA por binario (`Makefile`: `-mavx2 -mfma` por defecto en x86, `SIMD=avx512vnni`/`avx512bf16` opt-in, `-march=armv8.2-a+dotprod` en ARM64). Además, varias rutas avanzadas (`AVX-512 BF16`, `ARM i8mm`) usan intrínsecos recientes cuya disponibilidad en Rust estable varía o exige `nightly`/`asm!`.

## Decisión
1. Compilar `qwen3-tts-rs` en **Rust estable** (actualmente `rustc 1.96.0`) con **despacho en tiempo de ejecución** (`KernelDispatch`) resuelto al arranque mediante `is_x86_feature_detected!` / detección ARM.
2. **Línea base obligatoria en Rust estable**:
   - **x86-64**: `AVX2 + FMA` (`_mm256_maddubs_epi16`, `_mm256_sign_epi8`, `_mm256_fmadd_ps`, prefetch `5120B`, núcleo de 1 fila para `cols >= 2048`).
   - **ARM64**: `NEON + SDOT` (`vdotq_s32`).
   - **Universal**: Fallback escalar entero en todas las plataformas (sin fallar nunca con `SIGILL`).
3. **Extensiones avanzadas (`AVX-512 VNNI/BF16`, `ARM i8mm`)**: quedan detrás de features opcionales (solo si compilan limpiamente en el Rust estable vigente o mediante módulos opt-in aislados), nunca como requisito para compilar el motor.

## Consecuencias
- **Positivas**: Un único binario portable por sistema operativo, compilable en Rust estable y validable kernel a kernel con `--self-test`.
- **Negativas**: Máquinas con `AVX-512 VNNI/BF16` o `i8mm` correrán inicialmente con `AVX2+FMA` o `NEON+SDOT` hasta habilitar la ruta opt-in y validarla en silicio real.
