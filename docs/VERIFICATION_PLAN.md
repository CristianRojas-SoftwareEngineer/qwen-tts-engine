# Plan de Verificación, Oráculo C y Gates (`VERIFICATION_PLAN.md`)

Define cómo validar de forma objetiva cada fase de la migración de `qwen3-tts` (`0.6B`) de C a Rust frente al oráculo congelado en `qwen3-tts/`.

---

## 1. Corpus Dorado y Entorno de Medición (Fase 0)

### 1.1 Regla de comparación A/B (`qwen3-tts/AGENTS.md`)
Toda comparación entre C y Rust debe fijar parámetros explícitos idénticos: `--seed <N>`, `-s <speaker>`, `-l <language>`, mismo `--text`, mismo `-j <threads>`, mismo `-d <model_dir>` y (en Windows con OpenBLAS OpenMP) `OMP_NUM_THREADS = -j` sin `OPENBLAS_NUM_THREADS`.

### 1.2 Casos del corpus (`parity/corpus.json`)
| ID | Caso | Modelo / Perfil | Qué valida |
|---|---|---|---|
| `C1_short_es` | Frase corta español (`"Hola, ¿cómo estás?"`, `ryan`) | `0.6B` (`bf16`, `--int8`, `--quant-mixed`) | EOS boosting corto, coherencia y ausencia de bucle/ruido. |
| `C2_short_en` | Frase golden inglés (`GOLDEN_EN` del `Makefile`, `ryan`) | `0.6B` (`bf16`, `--int8`, `--int4`, `--quant-mixed`) | Paridad de modos (`test-modes`) y etiqueta `[INT8+Q4_0-FFN]`. |
| `C3_long_it` | Texto largo (~15–25 s, `ryan`/`vivian`, italiano) | `0.6B` (`--quant-mixed`, normal y `--stream`) | `RTF`, `TTFA`, tiling `up2`/`up3`, atención paralela por cabezas y caché KV > 1024 si aplica. |
| `C4_memload` | Igual a `C3_long_it` con 1 hilo lector/copiador de 256 MB en paralelo | `0.6B` (`--quant-mixed --stream`) | Eficacia real de prefetch `5120B`, 1-fila `>= 2048` cols y tiling de 256 KB. |
| `C5_clone` | Clonación `0.6B-Base` (`--ref-audio`, `--save-voice .qvoice`/`.bin`) → `0.6B` (`--icl-only`, `--graft`) | `0.6B-Base` + `0.6B` | `ECAPA-TDNN`, `SpeechEncoder` RVQ, `WOVR`/`TPAD` y timbre. |
| `C6_para` | Texto con `[laugh]`, `[sigh]`, `[pause:400ms]` (`small_model=1`) | `0.6B` (`--quant-mixed`) | Sustitución inline `small_model=1` y composición sin fugas. |
| `C7_server` | Suite equivalente a `test-serve-stream` + `--workers` + `--batch-size` | `0.6B` (`--quant-mixed`) | Reproducibilidad, HTTP `400`, chunk `0`, cancelación por socket y concurrencia. |

---

## 2. Protocolo de Captura del Baseline C (`parity/`)

1. Compilar el motor C de referencia (en MSYS2 UCRT64: `make blas` + `make test-decoder-tool`).
2. Ejecutar `parity/run_baseline_c.ps1` para guardar en `parity/golden/`:
   - Salida completa de `--caps`, `--self-test` y `--matmat-bench`.
   - Dump fijo de códigos (`QWEN_DUMP_CODES`) para probar el `SpeechDecoder` aislado.
   - Métricas base por caso (`C1..C7`): `RTF`, `TTFA`, desglose `[HTTP] Desglose:`, memoria RSS pico y transcripción/WER (si hay ASR local disponible) o verificación de `mel-corr`.

---

## 3. Gates de Aceptación por Fase de Implementación

| Gate | Fase | Pruebas obligatorias | Criterio de paso |
|---|---|---|---|
| **G0** | Baseline C | `test-selftest`, `test-errors`, `test-caps` + captura `C1..C7` | Artefactos dorados guardados en `parity/golden/`. |
| **G1** | `qwen-kernels` | `--caps`, `--self-test`, `--matmat-bench`, test estrés `ComputePool` | **0 fallos** en `--self-test` (`matmat_int8` error `0.0`, `q4_0 <= 1e-6`, `rel_L2 < 3e-2`) y `matmat-bench >= 95 %` del C. |
| **G2** | `qwen-core` | Carga real `0.6B` (`CustomVoice`/`Base`), BPE corpus, `.qvoice`/`.bin` | Tokenización **100 % idéntica** a C; validación negativa de `1.7B` y shapes inválidos; roundtrip `.qvoice` exacto. |
| **G3** | `qwen-engine` (1) | `decode-dump` sobre `QWEN_DUMP_CODES` + `C1..C4` | Decoder **`SNR >= 100 dB`** vs C; reproducibilidad **bit-idéntica** entre corridas Rust; `WER` equivalente, `mel-corr >= 0.95` y **`RTF <= baseline C`**, **`TTFA <= baseline C`**. |
| **G4** | `qwen-engine` (2) | `C5_clone`, `C6_para`, `emovoice`, `--emo-ref`, `--roughness` | Error explícito ante WAV != 24 kHz o `WDELTA` sobre `Q4_0`; paridad auditiva/WER en clonación y paralingüística `0.6B`. |
| **G5** | `qwen-app` | `C7_server` (`test-errors` + `test-serve*` + `--watch-stdin` + Win32 affinity) | Todos los `400` exactos; `/v1/tts == /v1/tts/stream` en bytes; cancelación inmediata sin bloqueo; cierre limpio al cerrar `stdin`; cero fugas/carreras. |
| **G6** | Cierre / `faer` | Comparativa `FfiBlas` vs `PureRustBlas` (`C1..C7`) + CI Win/Linux/Mac | Se adopta `faer` solo si `RTF <= FfiBlas` y `mel-corr/WER` sin regresión; si no, `FfiBlas` queda como default. |
