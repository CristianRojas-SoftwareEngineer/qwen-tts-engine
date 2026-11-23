#!/usr/bin/env bash
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Compara la ruta de kernels por defecto (QWEN_KERNELS=auto) con la escalar (QWEN_KERNELS=scalar) en
# la máquina actual, intercaladas por corrida para que la deriva de la máquina no favorezca a ninguna.
# Frase corta y larga de referencia (ci-common.sh), 3 corridas, -j 4.
# Uso: scripts/ci-compare-kernels.sh <carpeta de salida>   (BIN = ruta del binario)
set -euo pipefail
source "$(dirname "$0")/ci-common.sh"

OUT="${1:-kernels}"
BIN="${BIN:-target/release/qwen_tts}"
mkdir -p "$OUT"

for k in auto scalar; do
    echo "== --caps con QWEN_KERNELS=$k"
    QWEN_KERNELS=$k "$BIN" --caps | grep -E "kernels:|matvec \+ attn|q4_0 \+ int8|runtime cpu" || true
    QWEN_KERNELS=$k "$BIN" --self-test > "$OUT/self-test-$k.log" 2>&1 || { grep FALLO "$OUT/self-test-$k.log" | head -20; tail -5 "$OUT/self-test-$k.log"; exit 1; }
    grep -c " frente a escalar: .*OK" "$OUT/self-test-$k.log" | sed 's/^/  comparaciones SIMD frente a escalar en OK: /'
    tail -1 "$OUT/self-test-$k.log"
done

echo "== Síntesis (3 corridas intercaladas)"
print_table_header
for rep in 1 2 3; do
    for key in short long; do
        for k in auto scalar; do
            ( export QWEN_KERNELS=$k; measure_one "$BIN" 4 "$key" "$rep" "$k" "$OUT" )
        done
    done
done
