#!/usr/bin/env bash
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Funciones compartidas por los scripts de medición de CI (se cargan con `source`).

TEXT_short="Hello world, this is a quick baseline test."
TEXT_long="In this long synthesis benchmark, we measure sustained memory bandwidth, Talker latency, Code Predictor throughput, and streaming Speech Decoder stability across multiple seconds of continuous speech generation."

pcm_hash() {
    if command -v sha256sum > /dev/null; then tail -c +45 "$1" | sha256sum | cut -c1-16
    else tail -c +45 "$1" | shasum -a 256 | cut -c1-16; fi
}

print_table_header() {
    printf '%-10s %-6s %-4s %-6s %-9s %-11s %-11s %-12s %-8s %s\n' variante texto rep RTF frames "talker ms/f" "CP ms/f" "decoder ms" "TTFA ms" "hash PCM"
}

# measure_one <bin> <hilos> <short|long> <rep> <variante> <carpeta>: una síntesis (voz default, inglés,
# --seed 4) y una fila de la tabla.
measure_one() {
    local bin="$1" threads="$2" key="$3" rep="$4" variant="$5" out="$6"
    local text_var="TEXT_$key" log="$out/${variant}_${key}_${rep}.log" wav="$out/${variant}_${key}_${rep}.wav"
    "$bin" -d models/qwen3-tts-0.6b-base -l English -j "$threads" --seed 4 -t "${!text_var}" -o "$wav" > "$log" 2>&1
    local rtf frames tk cp dec ttfa
    rtf=$(grep -o 'RTF [0-9.]*' "$log" | tail -1 | cut -d' ' -f2)
    frames=$(grep -o 'Generados [0-9]* frames' "$log" | cut -d' ' -f2)
    tk=$(grep -o 'Paso del Talker: [0-9]* ms ([0-9.]* ms/f)' "$log" | grep -o '([0-9.]*' | tr -d '(')
    cp=$(grep -o 'Code Predictor: [0-9]* ms ([0-9.]* ms/f)' "$log" | grep -o '([0-9.]*' | tr -d '(')
    dec=$(grep -o 'Speech decoder: *[0-9]* ms' "$log" | grep -o '[0-9]*' | head -1)
    ttfa=$(grep -o 'TTFA: [0-9]* ms' "$log" | grep -o '[0-9]*' | head -1)
    printf '%-10s %-6s %-4s %-6s %-9s %-11s %-11s %-12s %-8s %s\n' "$variant" "$key" "$rep" "$rtf" "$frames" "$tk" "$cp" "$dec" "${ttfa:--}" "$(pcm_hash "$wav")"
}

# release_version: imprime la versión de la release (`v` + la versión de [workspace.package] de
# Cargo.toml, por ejemplo v1.0.0). Se ejecuta desde la raíz del repositorio.
release_version() {
    sed -n '/^\[workspace\.package\]/,/^\[/s/^version *= *"\([^"]*\)".*/v\1/p' Cargo.toml | head -n1
}
