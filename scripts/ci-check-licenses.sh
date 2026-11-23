#!/usr/bin/env bash
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Comprueba que la tabla «Crates que entran en el binario» de THIRD-PARTY-LICENSES.md
# (o del archivo indicado en THIRD_PARTY_FILE) coincide con los crates de Cargo que entran
# en el binario de los cuatro targets que se distribuyen: sale 1 si falta o sobra alguno.
set -euo pipefail

cd "$(dirname "$0")/.."
doc="${THIRD_PARTY_FILE:-THIRD-PARTY-LICENSES.md}"
targets=(x86_64-pc-windows-gnu x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu aarch64-apple-darwin)

# Recorre el grafo de dependencias desde `qwen-app` y escribe «nombre versión» por crate.
# Una arista cuenta si alguna de sus clases de dependencia es normal (`kind` nulo). Los
# crates de macros procedurales solo se usan al compilar: no cuentan ni se desciende por
# ellos. Los paquetes del workspace tampoco cuentan.
read -r -d '' jq_program <<'JQ' || true
. as $m
| ($m.packages | map({key: .id, value: .}) | from_entries) as $pkg
| ($m.resolve.nodes | map({key: .id, value: .deps}) | from_entries) as $deps
| $m.workspace_members as $ws
| ($m.packages | map(select(.name == "qwen-app"))[0].id) as $root
| def next_ids($id):
    ($deps[$id] // [])
    | map(select(any(.dep_kinds[]; .kind == null)))
    | map(.pkg)
    | map(select(any($pkg[.].targets[]; any(.kind[]; . == "proc-macro")) | not));
  {seen: [$root], front: [$root]}
| until(.front | length == 0;
    . as $s
    | ($s.front | map(next_ids(.)[]) | unique
       | map(select(. as $x | ($s.seen | index($x)) == null))) as $new
    | {seen: ($s.seen + $new), front: $new})
| .seen[]
| select(. as $x | ($ws | index($x)) == null)
| $pkg[.] | "\(.name) \(.version)"
JQ

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

for target in "${targets[@]}"; do
    cargo metadata --format-version 1 --filter-platform "$target" \
        | jq -r "$jq_program" | tr -d '\r' > "$tmp/$target.txt"
done
sort -u "$tmp"/*.txt > "$tmp/cargo.txt"

# Solo cuentan las filas de esa tabla cuya primera celda es exactamente `nombre`. La tabla de
# la biblioteca estándar de Rust va en su propia sección: sus crates no salen de `cargo metadata`.
awk -F'|' '
    /^## / { in_table = ($0 == "## Crates que entran en el binario"); next }
    in_table && NF > 3 {
        name = $2; version = $3
        gsub(/^ +| +$/, "", name); gsub(/^ +| +$/, "", version)
        if (name ~ /^`[^`]+`$/) { gsub(/`/, "", name); print name " " version }
    }' "$doc" | tr -d '\r' | sort -u > "$tmp/doc.txt"

missing="$(comm -23 "$tmp/cargo.txt" "$tmp/doc.txt")"
extra="$(comm -13 "$tmp/cargo.txt" "$tmp/doc.txt")"

if [[ -n "$missing" || -n "$extra" ]]; then
    if [[ -n "$missing" ]]; then
        echo "Crates que entran en el binario y no tienen fila en $doc:" >&2
        sed 's/^/  /' <<< "$missing" >&2
    fi
    if [[ -n "$extra" ]]; then
        echo "Filas de $doc que no corresponden a ningún crate de los cuatro targets:" >&2
        sed 's/^/  /' <<< "$extra" >&2
    fi
    exit 1
fi
echo "THIRD-PARTY-LICENSES.md coincide con los $(wc -l < "$tmp/cargo.txt") crates de los cuatro targets."
