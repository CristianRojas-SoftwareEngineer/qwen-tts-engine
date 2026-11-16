# Plan Maestro de Migración y Trazabilidad (`MIGRATION_PLAN.md`)

Hoja de ruta ejecutiva para la implementación de `qwen3-tts-rs/` (`0.6B`) y matriz completa de trazabilidad contra `qwen3-tts/` y `DIVERGENCIAS.md`.

---

## 1. Fases de Implementación y Criterios de Cierre

1. **Fase 0 — Baseline C (`parity/`)**: Compilar oráculo C y capturar métricas/dumps dorados (`Gate G0`).
2. **Fase 1 — `qwen-kernels`**: `AlignedBuf<64>`, `CpuCaps`, `ComputePool` instanciable (`spin-wait` + `FTZ/DAZ`), `BlasBackend` (`FfiBlas`), cuantización (`int8`/`q4_0`/`q2_0`) y kernels SIMD estables (`AVX2+FMA`, `NEON+SDOT`, escalar) + `--caps`, `--self-test`, `--matmat-bench` (`Gate G1`).
3. **Fase 2 — `qwen-core`**: Constantes `0.6B`, `MmapStore` (`safetensors` + `config.json`), `BpeTokenizer`, `Sampler`/`LcgRng`, `.qvoice`/`.bin`, `MarkupPlanner` (`small_model=1`) (`Gate G2`).
4. **Fase 3 — `qwen-engine` (Inferencia y Streaming)**: `Model06B`, `VoiceBindings` (`WOVR`), `SessionScratch`, `Talker`, `CodePredictor` (`PREFILL2`, `--quant-mixed`), `SpeechDecoder` (`conv_decoder_forward`, `DecoderWorkspace`, `W_k`, tiling `up2`/`up3`), `AudioSink` y pipeline productor-consumidor (`Gate G3`).
5. **Fase 4 — `qwen-engine` (Clonación y Emoción 0.6B)**: `ECAPA-TDNN`, `SpeechEncoder` RVQ (sin singleton), flujo `.qvoice`, `emovoice` (`dir_*.bin`), `--emo-ref`, `--roughness`, composición inline (`Gate G4`).
6. **Fase 5 — `qwen-app` (CLI + HTTP)**: CLI completa, `--watch-stdin`, afinidad SMT Win32, manifiesto UTF-8, servidor HTTP único (`ValidatedTtsRequest`, `/v1/tts`, `/v1/tts/stream`, `/v1/audio/speech`, `/v1/health`, `/v1/speakers`), cancelación RAII, `ContinuousBatchScheduler` y `WorkerPoolScheduler` (`Gate G5`).
7. **Fase 6 — Endurecimiento y BLAS 100 % Rust (Opcional)**: Benchmark `faer` vs `OpenBLAS`/`Accelerate`, CI multiplataforma y retiro del código C de producción (`Gate G6`).

---

## 2. Matriz de Trazabilidad: Archivos C → Crates Rust

| Archivo(s) C en `qwen3-tts/` | Destino en `qwen3-tts-rs/` | Estado |
|---|---|---|
| `main.c` (CLI, `.qvoice`, `emovoice`, Win32 affinity, `--watch-stdin`) | `qwen-app/src/main.rs` + `qwen-core/src/qvoice.rs` + `qwen-engine/src/voice_clone.rs` | Portar (podando flags `1.7B`) |
| `qwen_tts.h`, `qwen_tts.c` (contexto, prompt, bucle gen, batching, clone) | `qwen-engine/src/{model.rs, pipeline.rs}` | Rediseñado (4 tiempos de vida) |
| `qwen_tts_talker.c` | `qwen-engine/src/talker.rs` | Portar (`H=1024`, `INTER=3072`) |
| `qwen_tts_code_predictor.c` | `qwen-engine/src/code_predictor.rs` | Portar (sin `cp_mtp_proj`) |
| `qwen_tts_speech_decoder.c` | `qwen-engine/src/speech_decoder.rs` | Portar (workspace + tiling + `W_k`) |
| `qwen_tts_speech_encoder.c` | `qwen-engine/src/speech_encoder.rs` | Portar (sin `g_encoder` global) |
| `qwen_tts_kernels.h`, `qwen_tts_kernels.c` | `qwen-kernels/src/{aligned,caps,ftz,blas,quant,selftest,simd/*}` | Portar (despacho runtime estable) |
| `qwen_tts_thread.h`, `qwen_tts_thread.c` | `qwen-kernels/src/pool.rs` | Portar (`ComputePool` instanciable) |
| `qwen_tts_audio.h`, `qwen_tts_audio.c` | `qwen-engine/src/audio.rs` | Portar (WAV, PCM, WSOLA, ganancia) |
| `qwen_tts_sampling.c` | `qwen-core/src/sampling.rs` | Portar (`LcgRng` sin `__thread`) |
| `qwen_tts_tokenizer.h`, `qwen_tts_tokenizer.c` | `qwen-core/src/tokenizer.rs` | Portar (UTF-8 + guarda progreso) |
| `qwen_tts_voice_clone.h`, `qwen_tts_voice_clone.c` | `qwen-engine/src/voice_clone.rs` | Portar (WAV 24kHz, Mel, ECAPA-TDNN) |
| `qwen_tts_server.h`, `qwen_tts_server.c`, `qwen_tts_batch.h` | `qwen-app/src/server/{mod,validation,http,scheduler}.rs` | Rediseñado (frontend único) |
| `qwen_tts_compose.h`, `qwen_tts_compose.c` | `qwen-core/src/markup.rs` + `qwen-engine/src/pipeline.rs` | Portar (`small_model=1` solo `0.6B`) |
| `qwen_tts_emotion.h`, `qwen_tts_emotion.c` | `qwen-core/src/markup.rs` (tokens) + `qwen-engine/src/voice_clone.rs` | Podado `.qlsteer` `1.7B`; conservado `0.6B` |
| `qwen_tts.exe.manifest`, `qwen_tts_manifest.rc` | `qwen-app/{build.rs, windows/qwen_tts.exe.manifest}` | Portar (soporte Windows UTF-8) |
| `bench_scalar.c`, `bench_simd.c`, `test_decoder_standalone.c` | `qwen-kernels` benches / `qwen-app` subcomando + `parity/` | Reemplazado por tooling Rust/`parity/` |
| `third_party/ingot/**`, `vendor/lz4.{c,h}` | Crates `safetensors`, `memmap2`, `serde_json`, `lz4_flex` | **Reemplazado** (no portar) |

---

## 3. Matriz de Trazabilidad: `DIVERGENCIAS.md` → Rust

| Divergencia en `DIVERGENCIAS.md` | Módulo Rust responsable | Verificación (`VERIFICATION_PLAN.md`) |
|---|---|---|
| Bind `--host` + loopback `127.0.0.1` por defecto | `qwen-app/src/main.rs`, `server/http.rs` | Gate G5 (test CLI `--host` inválido y bind) |
| Vigía `--watch-stdin` (`_exit(0)` en EOF) | `qwen-app/src/main.rs` | Gate G5 (test cierre de pipe padre) |
| `WSAStartup` + shims POSIX + ramas MinGW/ARM | Nativo en Rust `std` + `memmap2` + `ADR-002` | Compilación nativa Windows/Linux/macOS |
| Manifiesto UTF-8 incrustado (`activeCodePage`) | `qwen-app/build.rs` + `windows/qwen_tts.exe.manifest` | Gate G5 (rutas con tildes/ñ en Windows) |
| Cancelación por desconexión del cliente | `qwen-app/src/server/http.rs` (`AbortGuard` RAII) + `pipeline.rs` | Gate G5 (`C7_server` corte de curl) |
| Guarda UTF-8/progreso en tokenizer + EOS corto | `qwen-core/src/tokenizer.rs` + `qwen-engine/src/pipeline.rs` | Gate G2 + Gate G3 (`C1_short_es`) |
| Poda tooling/GPU CUDA/Metal/rutas sin BLAS | Excluidos desde diseño (`ARCHITECTURE.md`) | Gate G3 |
| Red convolucional unificada + LayerNorm 2 pasadas | `qwen-engine/src/speech_decoder.rs` | Gate G3 (`decode-dump` `SNR >= 100 dB`) |
| `DecoderWorkspace` persistente (8 ranuras) | `qwen-engine/src/speech_decoder.rs` (`SessionScratch`) | Gate G3 (`C3_long_it` / `C4_memload`) |
| Convolución transpuesta 1 GEMM + causal sin `im2col` (`W_k`) | `qwen-engine/src/speech_decoder.rs` | Gate G3 (`decode-dump` `SNR >= 100 dB`) |
| Afinidad 1 hilo lógico/núcleo físico en Win32 | `qwen-app/src/main.rs` | Gate G5 (verificación de máscara en Win32) |
| Herramienta standalone del decoder | Subcomando/binario `decode-dump` en `qwen-app` | Gate G3 |
| `spin-wait` 4096 en pool de hilos (`QWEN_POOL_SPIN`) | `qwen-kernels/src/pool.rs` | Gate G1 + Gate G3 (`C3_long_it`) |
| Producto escalar entero AVX2 (`maddubs`) + `matmat==B×matvec` | `qwen-kernels/src/simd/avx2.rs` + `quant.rs` | Gate G1 (`--self-test` error `0.0`) |
| `PREFILL2` único en CP cuantizado + log `[INT8+Q4_0-FFN]` | `qwen-engine/src/code_predictor.rs` | Gate G3 (`C2_short_en` + `C3_long_it`) |
| Exigencia `dotprod` ARM64 + `quant_shape_check` (`%32`, `<=8192`) | `qwen-kernels/src/{caps.rs, quant.rs}` + `qwen-core/src/config.rs` | Gate G1 + Gate G2 |
| Eliminación de fallbacks float/env switches + `WDELTA` sin `Q4_0` | `qwen-kernels` + `qwen-engine/src/model.rs` | Gate G1 + Gate G4 |
| Prefetch `5120B` en `int8` + 1-fila AVX2 en `cols >= 2048` | `qwen-kernels/src/simd/avx2.rs` | Gate G1 (`--self-test`) + Gate G3 (`C4_memload`) |
| Perfil `mixed` del CP (`[INT8+Q4_0-FFN]`) | `qwen-engine/src/code_predictor.rs` | Gate G3 (`C2_short_en`, `C3_long_it`) |
| Tiling `256 KB` en `up2`/`up3` + calendario `2,4,6,8,10` + hilos BLAS fijos | `qwen-engine/src/{speech_decoder.rs, pipeline.rs}` | Gate G3 + Gate G5 (reproducibilidad 3 rondas) |
| Prefill siempre completo (`kv_len=0`) + reset `chunk_frames=10` | `qwen-engine/src/talker.rs` + `qwen-app/src/server/` | Gate G5 (`C7_server`) |
| Validación única HTTP `400` + chunk `0` solo en éxito + `[HTTP] Desglose:` | `qwen-app/src/server/{validation.rs, http.rs}` | Gate G5 (`C7_server`) |
| Atención Talker paralela por cabezas + `gate_up` fusionado solo en `bf16` | `qwen-kernels` + `qwen-engine/src/talker.rs` | Gate G3 (`C3_long_it` + RSS) |
| Caché KV inicial `1024` + conversión `bf16->f32` prefill paralelizada | `qwen-engine/src/talker.rs` | Gate G3 (`TTFA` + RSS) |
| Uso de `OMP_NUM_THREADS = -j` (sin `OPENBLAS_NUM_THREADS`) | `docs/INTERFACE_CONTRACT.md` + `qwen-kernels/src/blas.rs` | Gate G3 + Gate G5 |
