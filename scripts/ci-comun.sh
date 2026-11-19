#!/usr/bin/env bash
# Funciones compartidas por los scripts de medición de CI (se cargan con `source`).

TEXT_short="Hello world, this is a quick baseline test."
TEXT_long="In this long synthesis benchmark, we measure sustained memory bandwidth, Talker latency, Code Predictor throughput, and streaming Speech Decoder stability across multiple seconds of continuous speech generation."

pcm_hash() {
    if command -v sha256sum > /dev/null; then tail -c +45 "$1" | sha256sum | cut -c1-16
    else tail -c +45 "$1" | shasum -a 256 | cut -c1-16; fi
}

cabecera_tabla() {
    printf '%-10s %-6s %-4s %-6s %-9s %-11s %-11s %-12s %-8s %s\n' variante texto rep RTF fotogramas "talker ms/f" "CP ms/f" "decoder ms" "TTFA ms" "hash PCM"
}

# medir_una <bin> <hilos> <short|long> <rep> <variante> <carpeta>: una síntesis (ryan, inglés,
# --quant-mixed, --seed 4) y una fila de la tabla. Hereda el entorno (p. ej. QWEN_SD_THREADS).
medir_una() {
    local bin="$1" threads="$2" key="$3" rep="$4" variant="$5" out="$6"
    local text_var="TEXT_$key" log="$out/${variant}_${key}_${rep}.log" wav="$out/${variant}_${key}_${rep}.wav"
    "$bin" -d models/qwen3-tts-0.6b -s ryan -l English --quant-mixed -j "$threads" --seed 4 -t "${!text_var}" -o "$wav" > "$log" 2>&1
    local rtf frames tk cp dec ttfa
    rtf=$(grep -o 'RTF [0-9.]*' "$log" | tail -1 | cut -d' ' -f2)
    frames=$(grep -o 'Generated [0-9]* frames' "$log" | cut -d' ' -f2)
    tk=$(grep -o 'Talker step: [0-9]* ms ([0-9.]* ms/f)' "$log" | grep -o '([0-9.]*' | tr -d '(')
    cp=$(grep -o 'Code Predictor: [0-9]* ms ([0-9.]* ms/f)' "$log" | grep -o '([0-9.]*' | tr -d '(')
    dec=$(grep -o 'Speech decoder: *[0-9]* ms' "$log" | grep -o '[0-9]*' | head -1)
    ttfa=$(grep -o 'TTFA: [0-9]* ms' "$log" | grep -o '[0-9]*' | head -1)
    printf '%-10s %-6s %-4s %-6s %-9s %-11s %-11s %-12s %-8s %s\n' "$variant" "$key" "$rep" "$rtf" "$frames" "$tk" "$cp" "$dec" "${ttfa:--}" "$(pcm_hash "$wav")"
}
