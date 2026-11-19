#!/usr/bin/env bash
# Mide el motor en la máquina actual (pensado para CircleCI): --caps, --self-test y síntesis de la
# frase corta y la larga de referencia (ryan, inglés, --quant-mixed, --seed 4), 3 corridas cada una.
# Uso: scripts/ci-medir-rtf.sh <hilos> <carpeta de salida>   (BIN = ruta del binario)
set -euo pipefail
source "$(dirname "$0")/ci-comun.sh"

THREADS="${1:-4}"
OUT="${2:-medicion}"
BIN="${BIN:-target/release/qwen_tts}"
mkdir -p "$OUT"

echo "== Máquina"
uname -a
if [ -r /sys/fs/cgroup/cpu.max ]; then echo "cgroup cpu.max: $(cat /sys/fs/cgroup/cpu.max)"; fi
echo "hilos del motor: $THREADS"

echo "== --caps"
"$BIN" --caps
echo "== --self-test"
"$BIN" --self-test > "$OUT/self-test.log" 2>&1 || { tail -20 "$OUT/self-test.log"; exit 1; }
tail -1 "$OUT/self-test.log"

echo "== Síntesis"
cabecera_tabla
for rep in 1 2 3; do
    for key in short long; do
        medir_una "$BIN" "$THREADS" "$key" "$rep" "-j$THREADS" "$OUT"
    done
done
