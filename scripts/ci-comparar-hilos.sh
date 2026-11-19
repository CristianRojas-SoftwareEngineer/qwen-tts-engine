#!/usr/bin/env bash
# Compara configuraciones de hilos en la máquina actual, intercaladas por corrida para que la
# deriva de la máquina no favorezca a ninguna:
#   j4      -j 4 (productor 4, decoder 4, BLAS de generación 3)
#   j2      -j 2 (productor 2, decoder 2, BLAS de generación 1)
#   j4-sd2  -j 4 con QWEN_SD_THREADS=2 (productor 4, decoder 2)
# Uso: scripts/ci-comparar-hilos.sh <carpeta de salida>   (BIN = ruta del binario)
set -euo pipefail
source "$(dirname "$0")/ci-comun.sh"

OUT="${1:-hilos}"
BIN="${BIN:-target/release/qwen_tts}"
mkdir -p "$OUT"

"$BIN" --caps | grep -i -E "Afinidad|runtime cpu|matvec threads" || true
echo "== Síntesis (3 corridas intercaladas)"
cabecera_tabla
for rep in 1 2 3; do
    for key in short long; do
        medir_una "$BIN" 4 "$key" "$rep" j4 "$OUT"
        medir_una "$BIN" 2 "$key" "$rep" j2 "$OUT"
        ( export QWEN_SD_THREADS=2; medir_una "$BIN" 4 "$key" "$rep" j4-sd2 "$OUT" )
    done
done
