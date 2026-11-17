# Plan de Verificación (`VERIFICATION_PLAN.md`)

Define los criterios objetivos de validación de **Qwen TTS Engine** (`qwen-tts-engine`, actualmente `0.6B`) en Rust estable.

---

## 1. Suites Automatizadas Vigentes (`cargo test`)

| Suite / Comando | Qué valida |
|---|---|
| `qwen-kernels/tests/kernels_tests.rs` | `AlignedBuf<64>`, `--self-test` (`0` fallos numéricos en `int8`/`q4_0`/`bf16`), estrés concurrente de `ComputePool` sin estado global, `BlasBackend::sgemm` y `quantize_bf16_to_q2_0`. |
| `qwen-core/tests/core_tests.rs` | Carga `0.6B` (`CustomVoice`/`Base`), rechazo negativo de `1.7B` y `VoiceDesign`, `MmapStore`, tokenización BPE golden, roundtrip `.bin` y `.qvoice` v3 (`META`/`TPAD`/`WOVR` + `WDLT` LZ4) y rechazo de `.qvoice` v1/v2, `MarkupPlanner` y determinismo `Sampler`/`LcgRng`. |
| `qwen-engine/tests/engine_tests.rs` | Equivalencia de `SpeechDecoder` streaming (`2, 4×8, 6×8, 8×8, 10` con tiling 256 KB) frente a decodificación completa (`SNR >= 100 dB`), determinismo E2E bit-idéntico y cancelación vía `AbortHandle`. |
| `qwen-engine/tests/voice_clone_emotion_tests.rs` | Validación temprana de WAV 24 kHz y guías `ffmpeg`, `trim_trailing_silence`, `mel_spectrogram`, resolución `emovoice` + norma L2 `dir_*.bin`, flujo E2E `0.6B-Base` → `.qvoice` (`WDLT`) + error explícito sobre `Q4_0` + `--icl-only`, `--emo-ref`, `--roughness` (`Q2_0`) y composición multi-span. |
| `qwen-app` unit tests (`cli`, `validation`) | Rechazo de flags no soportadas (`--voice-design`, `--instruct`), `--host` inválido, `--graft` / `--stdout` y validación completa HTTP `400` en `ValidatedTtsRequest`. |
| `qwen_tts --caps`, `--self-test`, `--matmat-bench` | Capacidades SIMD/BLAS/hilos en runtime, batería numérica de kernels (`0` fallos) y microbenchmark `matmat` vs `B × matvec`. |
| `qwen_tts decode-dump <codes.txt> <model_dir> <out.wav>` | Decodificación aislada de tramas de 16 codebooks a WAV 24 kHz s16le. |

En `.github/workflows/ci.yml` se ejecutan en cada push/PR sobre `main` las pruebas del workspace, compilación release, `--caps`, `--self-test` y verificación del manifiesto UTF-8 Win32 en **Windows (`x86_64-pc-windows-gnu`)**, **Linux (`x86_64` y `aarch64`)** y **macOS (`arm64`)**, además de `cargo check -p qwen-app --features faer-blas`.
