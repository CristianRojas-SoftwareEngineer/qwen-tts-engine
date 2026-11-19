#!/usr/bin/env bash
# Mide el motor en la máquina actual (pensado para CircleCI): --caps, --self-test y síntesis de la
# frase corta y la larga de la línea base (ryan, inglés, --quant-mixed, --seed 4), 3 corridas cada una.
# Uso: scripts/ci-medir-rtf.sh <hilos> <carpeta de salida>
set -euo pipefail

THREADS="${1:-4}"
OUT="${2:-medicion}"
BIN="target/release/qwen_tts"
MODEL="models/qwen3-tts-0.6b"
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

TEXT_short="Hello world, this is a quick baseline test."
TEXT_long="In this long synthesis benchmark, we measure sustained memory bandwidth, Talker latency, Code Predictor throughput, and streaming Speech Decoder stability across multiple seconds of continuous speech generation."

pcm_hash() {
    if command -v sha256sum > /dev/null; then tail -c +45 "$1" | sha256sum | cut -c1-16
    else tail -c +45 "$1" | shasum -a 256 | cut -c1-16; fi
}

echo "== Síntesis"
printf '%-6s %-4s %-6s %-9s %-11s %-11s %-12s %-8s %s\n' texto rep RTF fotogramas "talker ms/f" "CP ms/f" "decoder ms" "TTFA ms" "hash PCM"
for rep in 1 2 3; do
    for key in short long; do
        text_var="TEXT_$key"
        log="$OUT/${key}_${rep}.log"
        wav="$OUT/${key}_${rep}.wav"
        "$BIN" -d "$MODEL" -s ryan -l English --quant-mixed -j "$THREADS" --seed 4 -t "${!text_var}" -o "$wav" > "$log" 2>&1
        rtf=$(grep -o 'RTF [0-9.]*' "$log" | tail -1 | cut -d' ' -f2)
        frames=$(grep -o 'Generated [0-9]* frames' "$log" | cut -d' ' -f2)
        tk=$(grep -o 'Talker step: [0-9]* ms ([0-9.]* ms/f)' "$log" | grep -o '([0-9.]*' | tr -d '(')
        cp=$(grep -o 'Code Predictor: [0-9]* ms ([0-9.]* ms/f)' "$log" | grep -o '([0-9.]*' | tr -d '(')
        dec=$(grep -o 'Speech decoder: *[0-9]* ms' "$log" | grep -o '[0-9]*' | head -1)
        ttfa=$(grep -o 'TTFA: [0-9]* ms' "$log" | grep -o '[0-9]*' | head -1)
        printf '%-6s %-4s %-6s %-9s %-11s %-11s %-12s %-8s %s\n' "$key" "$rep" "$rtf" "$frames" "$tk" "$cp" "$dec" "${ttfa:--}" "$(pcm_hash "$wav")"
    done
done
