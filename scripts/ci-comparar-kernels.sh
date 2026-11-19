#!/usr/bin/env bash
# Compara la ruta de kernels por defecto (QWEN_KERNELS=auto) con la escalar (QWEN_KERNELS=scalar) en
# la máquina actual, intercaladas por corrida para que la deriva de la máquina no favorezca a ninguna.
# Frase corta y larga de la línea base, 3 corridas, -j 4.
# Uso: scripts/ci-comparar-kernels.sh <carpeta de salida>   (BIN = ruta del binario)
set -euo pipefail
source "$(dirname "$0")/ci-comun.sh"

OUT="${1:-kernels}"
BIN="${BIN:-target/release/qwen_tts}"
mkdir -p "$OUT"

for k in auto scalar; do
    echo "== --caps con QWEN_KERNELS=$k"
    QWEN_KERNELS=$k "$BIN" --caps | grep -E "kernels:|matvec \+ attn|runtime cpu" || true
    QWEN_KERNELS=$k "$BIN" --self-test > "$OUT/self-test-$k.log" 2>&1 || { tail -20 "$OUT/self-test-$k.log"; exit 1; }
    tail -1 "$OUT/self-test-$k.log"
done

echo "== Síntesis (3 corridas intercaladas)"
cabecera_tabla
for rep in 1 2 3; do
    for key in short long; do
        for k in auto scalar; do
            ( export QWEN_KERNELS=$k; medir_una "$BIN" 4 "$key" "$rep" "$k" "$OUT" )
        done
    done
done
