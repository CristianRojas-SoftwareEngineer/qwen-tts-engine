# Migración del motor Qwen3-TTS (0.6B) de C a Rust

Workspace de migración y rediseño del motor de inferencia TTS **Qwen3-TTS 0.6B** (`CustomVoice` y `Base`) desde C puro hacia Rust.

## Estructura del directorio

- `docs/` — Especificación técnica pre-migración (`INTERFACE_CONTRACT.md`, `SPEC_06B.md`, `NUMERIC_CONTRACT.md`, `ARCHITECTURE.md`, `VERIFICATION_PLAN.md`, `MIGRATION_PLAN.md`) y `docs/adr/` (decisiones de arquitectura).
- `qwen3-tts/` — Snapshot congelado del motor C de referencia (oráculo para paridad numérica, audio y rendimiento).
- `qwen3-tts-rs/` — *(Fases de implementación)* Workspace Cargo del nuevo motor en Rust (`qwen-kernels`, `qwen-core`, `qwen-engine`, `qwen-app`).
- `parity/` — Arnés de pruebas cruzadas C vs Rust y artefactos de línea base.
- `models/` — *(Opcional, ignorado en Git)* Pesos `0.6B` y perfiles `.qvoice`/`.bin` compartidos por ambos motores.

## Entorno local auditado (Fase D0)

- **Git**: `2.53.0.windows.1`
- **Rust**: `rustc 1.96.0` / `cargo 1.96.0` (host `x86_64-pc-windows-msvc`; targets instalados: `x86_64-pc-windows-msvc`, `x86_64-pc-windows-gnu`, `x86_64-unknown-linux-gnu`)
- **Compilador C / OpenBLAS (MSYS2 UCRT64)**: `C:\msys64\ucrt64\bin\gcc.exe` (`16.2.0`), `C:\msys64\ucrt64\include\openblas\cblas.h`, `C:\msys64\ucrt64\lib\libopenblas.a`
- **Modelos locales en este workspace**: ninguno descargado actualmente (solo `qwen3-tts/config.json` y `qwen3-tts/speech_tokenizer_config.json` de referencia).
