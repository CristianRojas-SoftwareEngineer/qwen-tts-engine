# Plan Maestro de Migración y Trazabilidad (`MIGRATION_PLAN.md`)

Registro histórico cerrado de la migración del motor `Qwen3-TTS 0.6B` hacia `qwen3-tts-rs/` (Gates **G0–G6** completados). Para el diseño y contrato vivos actuales, véanse [ARCHITECTURE.md](ARCHITECTURE.md), [INTERFACE_CONTRACT.md](INTERFACE_CONTRACT.md), [SPEC_06B.md](SPEC_06B.md) y [NUMERIC_CONTRACT.md](NUMERIC_CONTRACT.md).

---

## 1. Fases de Implementación y Criterios de Cierre

1. **Fase 0 — Baseline C (`parity/`)**: Compilar oráculo C y capturar métricas/dumps dorados (`Gate G0`) — **Completado (`aeb466f`)**.
2. **Fase 1 — `qwen-kernels`**: `AlignedBuf<64>`, `CpuCaps`, `ComputePool` instanciable (`spin-wait` + `FTZ/DAZ`), `BlasBackend` (`FfiBlas`), cuantización (`int8`/`q4_0`/`q2_0`) y kernels SIMD estables (`AVX2+FMA` en x86-64, fallback entero escalar portable) + `--caps`, `--self-test`, `--matmat-bench` (`Gate G1`) — **Completado (`d7b3efd`)**.
3. **Fase 2 — `qwen-core`**: Constantes `0.6B`, `MmapStore` (`safetensors` + `config.json`), `BpeTokenizer`, `Sampler`/`LcgRng`, `.qvoice`/`.bin`, `MarkupPlanner` (`small_model=1`) (`Gate G2`) — **Completado (`675460c`)**.
4. **Fase 3 — `qwen-engine` (Inferencia y Streaming)**: `Model06B`, `VoiceBindings` (`WOVR`), `SessionScratch`, `Talker`, `CodePredictor` (`PREFILL2`, `--quant-mixed`), `SpeechDecoder` (`conv_decoder_forward`, `DecoderWorkspace`, `W_k`, tiling `up2`/`up3`), `AudioSink` y pipeline productor-consumidor (`Gate G3`) — **Completado (`44631c0`)**.
5. **Fase 4 — `qwen-engine` (Clonación y Emoción 0.6B)**: `ECAPA-TDNN`, `SpeechEncoder` RVQ (sin singleton), flujo `.qvoice`, `emovoice` (`dir_*.bin`), `--emo-ref`, `--roughness`, composición inline (`Gate G4`) — **Completado (`a391aaf`)**.
6. **Fase 5 — `qwen-app` (CLI + HTTP)**: CLI completa, `--watch-stdin`, afinidad SMT Win32, manifiesto UTF-8, servidor HTTP único (`ValidatedTtsRequest`, `/v1/tts`, `/v1/tts/stream`, `/v1/audio/speech`, `/v1/health`, `/v1/speakers`), cancelación RAII, `ContinuousBatchScheduler` y `WorkerPoolScheduler` (`Gate G5`) — **Completado (`ec50b49`)**.
7. **Fase 6 — Endurecimiento, `faer` vs `FfiBlas` y Cierre**:
   - **Benchmark `faer` (0.24.4) vs `FfiBlas` (OpenBLAS) en `0.6B --quant-mixed -j 4`**:
     - `C3_long_it_ryan_normal`: OpenBLAS `RTF 0.71` vs C `0.74` vs `faer` `0.75`.
     - `C3_long_it_ryan_stream`: OpenBLAS `RTF 0.70` (`TTFA 1116 ms`) vs C `0.78` (`TTFA 1226 ms`) vs `faer` `0.72` (`TTFA 1125 ms`).
     - `decode-dump` SNR vs C: OpenBLAS `96.2 dB` (`mel_corr 1.0`) vs `faer` `93.3 dB` (`mel_corr 1.0`).
     - **Decisión (`VERIFICATION_PLAN.md` Gate G6)**: se mantiene **`FfiBlas` como backend por defecto** (mejor RTF en generación larga por ausencia de contención entre el pool global de `rayon` en `faer` y `ComputePool`); `PureRustBlas` (`faer`) queda disponible como feature opcional `--features faer-blas` (`QWEN_BLAS=faer`).
   - CI multiplataforma añadido en `.github/workflows/ci.yml` (Windows GNU, Linux x86_64/aarch64, macOS arm64) y retiro de `qwen3-tts/` y `parity/` (`Gate G6`) — **Completado**.

---

## 2. Matriz de Trazabilidad: Archivos C → Crates Rust

| Archivo(s) C en `qwen3-tts/` *(retirado en G6)* | Destino en `qwen3-tts-rs/` | Estado |
|---|---|---|
| `main.c` (CLI, `.qvoice`, `emovoice`, Win32 affinity, `--watch-stdin`) | `qwen-app/src/{main.rs,cli.rs,affinity.rs}` + `qwen-core/src/qvoice.rs` + `qwen-engine/src/{voice_clone.rs,emotion.rs}` | **Completado (G2/G4/G5)** |
| `qwen_tts.h`, `qwen_tts.c` (contexto, prompt, bucle gen, batching, clone) | `qwen-engine/src/{model.rs,pipeline.rs,batch.rs}` | **Completado (G3/G5)** |
| `qwen_tts_talker.c` | `qwen-engine/src/{talker.rs,batch.rs}` | **Completado (G3/G5)** |
| `qwen_tts_code_predictor.c` | `qwen-engine/src/{code_predictor.rs,batch.rs}` | **Completado (G3/G5)** |
| `qwen_tts_speech_decoder.c` | `qwen-engine/src/speech_decoder.rs` | **Completado (G3)** |
| `qwen_tts_speech_encoder.c` | `qwen-engine/src/speech_encoder.rs` | **Completado (G4)** |
| `qwen_tts_kernels.h`, `qwen_tts_kernels.c` | `qwen-kernels/src/{aligned,caps,ftz,blas,quant,selftest,bench,dispatch,simd/{scalar,avx2}}` | **Completado (G1/G6)** |
| `qwen_tts_thread.h`, `qwen_tts_thread.c` | `qwen-kernels/src/pool.rs` | **Completado (G1)** |
| `qwen_tts_audio.h`, `qwen_tts_audio.c` | `qwen-engine/src/audio.rs` | **Completado (G3/G4)** |
| `qwen_tts_sampling.c` | `qwen-core/src/sampling.rs` | **Completado (G2)** |
| `qwen_tts_tokenizer.h`, `qwen_tts_tokenizer.c` | `qwen-core/src/tokenizer.rs` | **Completado (G2)** |
| `qwen_tts_voice_clone.h`, `qwen_tts_voice_clone.c` | `qwen-engine/src/voice_clone.rs` | **Completado (G4)** |
| `qwen_tts_server.h`, `qwen_tts_server.c`, `qwen_tts_batch.h` | `qwen-app/src/server/{mod,validation,http,scheduler}.rs` | **Completado (G5)** |
| `qwen_tts_compose.h`, `qwen_tts_compose.c` | `qwen-core/src/markup.rs` + `qwen-engine/src/compose.rs` | **Completado (G2/G4)** |
| `qwen_tts_emotion.h`, `qwen_tts_emotion.c` | `qwen-core/src/markup.rs` + `qwen-engine/src/emotion.rs` | **Completado (G2/G4)** |
| `qwen_tts.exe.manifest`, `qwen_tts_manifest.rc` | `qwen-app/{build.rs,qwen_app.exe.manifest,qwen_app_manifest.rc}` | **Completado (G5)** |
| `bench_scalar.c`, `bench_simd.c`, `test_decoder_standalone.c` | `qwen-kernels` (`selftest`, `bench`) + `qwen-app` (`decode-dump`, `--self-test`, `--matmat-bench`) | **Completado (G1/G6)** |
| `third_party/ingot/**`, `vendor/lz4.{c,h}` | Crates `safetensors`, `memmap2`, `serde_json`, `lz4_flex` | **Reemplazado (G2)** |

---

## 3. Matriz de Trazabilidad: `DIVERGENCIAS.md` → Rust

| Divergencia | Módulo Rust responsable | Verificación |
|---|---|---|
| Bind `--host` + loopback `127.0.0.1` por defecto | `qwen-app/src/cli.rs`, `server/scheduler.rs` | **Gate G5** (`--host` inválido y bind) |
| Vigía `--watch-stdin` (`exit(0)` en EOF) | `qwen-app/src/main.rs` | **Gate G5** (cierre de pipe padre) |
| `WSAStartup` + shims POSIX + ramas MinGW/ARM | Nativo en Rust `std` + `memmap2` + `ADR-002` | **Gate G6** (CI Win/Linux/macOS) |
| Manifiesto UTF-8 incrustado (`activeCodePage`) | `qwen-app/{build.rs,qwen_app.exe.manifest}` | **Gate G5** (verificado en PE Win32) |
| Cancelación por desconexión del cliente | `qwen-app/src/server/http.rs` (`AbortGuard`) + `model.rs` (`AbortHandle::with_poll`) | **Gate G5** (`C7_server` corte de socket) |
| Guarda UTF-8/progreso en tokenizer + EOS corto | `qwen-core/src/tokenizer.rs` + `qwen-engine/src/pipeline.rs` | **Gate G2 + G3** (`C1_short_es`) |
| Poda tooling/GPU CUDA/Metal/rutas sin BLAS | Excluidos desde diseño (`ARCHITECTURE.md`) | **Gate G3 + G6** |
| Red convolucional unificada + LayerNorm 2 pasadas | `qwen-engine/src/speech_decoder.rs` | **Gate G3** (`decode-dump` `SNR=96.2 dB`, `<=1 LSB`, `mel_corr=1.0`) |
| `DecoderWorkspace` persistente (8 ranuras) | `qwen-engine/src/model.rs` + `speech_decoder.rs` | **Gate G3** (`C3_long_it` / `C4_memload`) |
| Convolución transpuesta 1 GEMM + causal sin `im2col` (`W_k`) | `qwen-engine/src/speech_decoder.rs` | **Gate G3** |
| Afinidad 1 hilo lógico/núcleo físico en Win32 | `qwen-app/src/affinity.rs` | **Gate G5** (máscara `0x5555` en Win32) |
| Herramienta standalone del decoder | Subcomando `decode-dump` en `qwen-app/src/main.rs` | **Gate G3 + G6** |
| `spin-wait` 4096 en pool de hilos (`QWEN_POOL_SPIN`) | `qwen-kernels/src/pool.rs` | **Gate G1 + G3** |
| Producto escalar entero AVX2 (`maddubs`) + `matmat==B×matvec` | `qwen-kernels/src/simd/avx2.rs` + `quant.rs` | **Gate G1** (`--self-test` error `0.00e0`) |
| `PREFILL2` único en CP cuantizado + log `[INT8+Q4_0-FFN]` | `qwen-engine/src/code_predictor.rs` | **Gate G3** (`C2_short_en` + `C3_long_it`) |
| `quant_shape_check` (`%32`, `<=8192`) + detección `CpuCaps` | `qwen-kernels/src/{caps.rs,quant.rs}` + `qwen-core/src/config.rs` | **Gate G1 + G2** |
| Eliminación de fallbacks float/env switches + `WDELTA` sin `Q4_0` | `qwen-kernels` + `qwen-engine/src/model.rs` | **Gate G1 + G4** |
| Prefetch `5120B` en `int8` + 1-fila AVX2 en `cols >= 2048` | `qwen-kernels/src/simd/avx2.rs` | **Gate G1 + G3** (`C4_memload`) |
| Perfil `mixed` del CP (`[INT8+Q4_0-FFN]`) | `qwen-engine/src/code_predictor.rs` | **Gate G3** |
| Tiling `256 KB` en `up2`/`up3` + calendario `2,4,6,8,10` + hilos BLAS fijos | `qwen-engine/src/{speech_decoder.rs,pipeline.rs}` | **Gate G3 + G5** |
| Prefill siempre completo (`kv_len=0`) + reset `chunk_frames=10` | `qwen-engine/src/talker.rs` + `qwen-app/src/server/` | **Gate G5** (`C7_server`) |
| Validación única HTTP `400` + chunk `0` solo en éxito + `[HTTP] Desglose:` | `qwen-app/src/server/{validation.rs,http.rs}` | **Gate G5** (`C7_server`) |
| Atención Talker paralela por cabezas + `gate_up` fusionado solo en `bf16` | `qwen-kernels` + `qwen-engine/src/talker.rs` | **Gate G3** |
| Caché KV inicial `1024` + conversión `bf16->f32` prefill paralelizada | `qwen-engine/src/talker.rs` | **Gate G3** |
| Uso de `OMP_NUM_THREADS = -j` (sin `OPENBLAS_NUM_THREADS`) | `qwen-kernels/src/blas.rs` | **Gate G3 + G5 + G6** |
