# Plan de Verificación y Gates (`VERIFICATION_PLAN.md`)

Define los criterios objetivos de validación de **Qwen TTS Engine** (`qwen-tts-engine`, actualmente `0.6B`) en Rust estable. Todos los Gates de migración (**G0–G6**) fueron superados y el código C histórico (`qwen3-tts/` y `parity/`) fue retirado al cerrar el Gate **G6**.

---

## 1. Suites Automatizadas Vigentes (`cargo test`)

| Suite / Comando | Qué valida |
|---|---|
| `qwen-kernels/tests/kernels_tests.rs` | `AlignedBuf<64>`, `--self-test` (`0` fallos numéricos en `int8`/`q4_0`/`bf16`), estrés concurrente de `ComputePool` sin estado global, `BlasBackend::sgemm` y `quantize_bf16_to_q2_0`. |
| `qwen-core/tests/core_tests.rs` | Carga `0.6B` (`CustomVoice`/`Base`), rechazo negativo de `1.7B` y `VoiceDesign`, `MmapStore`, tokenización BPE golden, roundtrip `.bin` y `.qvoice` v3 (`META`/`TPAD`/`WOVR` + `WDLT` LZ4) y rechazo de `.qvoice` legacy (`v1`/`v2`), `MarkupPlanner` y determinismo `Sampler`/`LcgRng`. |
| `qwen-engine/tests/engine_tests.rs` | Paridad `SpeechDecoder` streaming (`2, 4×8, 6×8, 8×8, 10` con tiling 256 KB) vs decodificación completa (`SNR >= 100 dB`), determinismo E2E bit-idéntico y cancelación vía `AbortHandle`. |
| `qwen-engine/tests/f4_tests.rs` | Validación temprana de WAV 24 kHz y guías `ffmpeg`, `trim_trailing_silence`, `mel_spectrogram`, resolución `emovoice` + norma L2 `dir_*.bin`, flujo E2E `0.6B-Base` → `.qvoice` (`WDLT`) + error explícito sobre `Q4_0` + `--icl-only`, `--emo-ref`, `--roughness` (`Q2_0`) y composición multi-span. |
| `qwen-app` unit tests (`cli`, `validation`) | Rechazo de flags no soportadas (`--voice-design`, `--instruct`), `--host` inválido, `--graft` / `--stdout` y validación completa HTTP `400` en `ValidatedTtsRequest`. |
| `qwen_tts --caps`, `--self-test`, `--matmat-bench` | Capacidades SIMD/BLAS/hilos en runtime, batería numérica de kernels (`0` fallos) y microbenchmark `matmat` vs `B × matvec`. |
| `qwen_tts decode-dump <codes.txt> <model_dir> <out.wav>` | Decodificación aislada de tramas de 16 codebooks a WAV 24 kHz s16le. |

En `.github/workflows/ci.yml` se ejecutan en cada push/PR las pruebas del workspace, compilación release, `--caps`, `--self-test` y verificación del manifiesto UTF-8 Win32 sobre **Windows (`x86_64-pc-windows-gnu`)**, **Linux (`x86_64` y `aarch64`)** y **macOS (`arm64`)**, además de `cargo check -p qwen-app --features faer-blas`.

---

## 2. Histórico de Gates Superados (G0–G6)

| Gate | Fase | Criterio superado |
|---|---|---|
| **G0** | Baseline | Captura de línea base A/B en `C1..C7` y dumps dorados de paridad (`aeb466f`). |
| **G1** | `qwen-kernels` | `0` fallos en `--self-test` (`matmat_int8` error `0.0`, `q4_0 <= 1e-6`, `rel_L2 < 3e-2`) y rendimiento `>= 95 %` (`d7b3efd`). |
| **G2** | `qwen-core` | Tokenización BPE idéntica, validación negativa `1.7B`/`VoiceDesign` y roundtrip `.qvoice`/`.bin` (`675460c`). |
| **G3** | `qwen-engine` (1) | Decoder `SNR = 96.2 dB` (`<= 1 LSB`, `mel_corr = 1.0`), determinismo bit-idéntico y `RTF/TTFA <= C` (`44631c0`). |
| **G4** | `qwen-engine` (2) | `ECAPA-TDNN`, `SpeechEncoder` RVQ, `.qvoice` (`WOVR`/`WDLT`), error `WDLT` sobre `Q4_0`, `emovoice`, `--emo-ref`, `--roughness` y composición (`a391aaf`). |
| **G5** | `qwen-app` | CLI completa, `--watch-stdin`, afinidad SMT Win32, HTTP `400`, `/v1/tts == /v1/tts/stream`, cancelación RAII sin chunk `0`, `--workers` y `--batch-size` (`ec50b49`). |
| **G6** | Cierre / `faer` | Comparativa `FfiBlas` (`RTF 0.70–0.71`) vs `PureRustBlas` (`0.72–0.75`); `FfiBlas` mantenido como default (`faer-blas` opt-in), CI multiplataforma y retiro de `qwen3-tts/` y `parity/` (`fecc6e5`). |
