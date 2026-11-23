#!/usr/bin/env bash
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Audita que un binario sea portable: solo depende de bibliotecas del sistema y no exige una
# versión de sistema más reciente que la prometida.
#
# Uso: ci-audit-binary.sh <binario> <triple>
#   *-unknown-linux-gnu      bibliotecas dinámicas permitidas, símbolos GLIBC_* <= 2.28 y sin GLIBCXX_*.
#   aarch64-apple-darwin     minos <= 26.0 y solo bibliotecas de /usr/lib/ o /System/Library/.
#   x86_64-pc-windows-gnu    solo DLL del sistema (KERNEL32, ntdll, WS2_32, bcryptprimitives, api-ms-win-*).
# Imprime lo que encuentra y sale con 1, explicando el motivo, ante el primer incumplimiento.

set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Uso: $0 <binario> <triple>" >&2
    exit 2
fi
BIN="$1"
TRIPLE="$2"
[ -f "$BIN" ] || { echo "ERROR: no existe el binario $BIN" >&2; exit 2; }

FAILED=0
fail() {
    echo "INCUMPLIMIENTO: $*" >&2
    FAILED=1
}

# version_le <a> <b>: verdadero si la versión a (numérica, con puntos) es menor o igual que b.
version_le() {
    [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -n1)" = "$1" ]
}

audit_linux() {
    local allowed=" libc.so.6 libm.so.6 libpthread.so.0 libdl.so.2 libgcc_s.so.1 ld-linux-x86-64.so.2 ld-linux-aarch64.so.1 "
    local needed lib
    needed=$(readelf -d "$BIN" | sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p')
    echo "Bibliotecas dinámicas (NEEDED):"
    echo "${needed:-  (ninguna)}" | sed 's/^/  /'
    for lib in $needed; do
        case "$allowed" in
            *" $lib "*) ;;
            *) fail "la biblioteca $lib no está en la lista permitida" ;;
        esac
    done

    local max
    max=$(objdump -T "$BIN" | grep -o 'GLIBC_[0-9][0-9.]*' | sed 's/^GLIBC_//' | sort -uV | tail -n1 || true)
    echo "Símbolo GLIBC_* máximo: ${max:-(ninguno)}"
    if [ -n "$max" ] && ! version_le "$max" "2.28"; then
        fail "el binario exige GLIBC_$max y el máximo permitido es GLIBC_2.28"
    fi

    if objdump -T "$BIN" | grep -q 'GLIBCXX_'; then
        fail "el binario usa símbolos GLIBCXX_* (depende de libstdc++)"
    fi
}

audit_macos() {
    local minos libs lib
    minos=$(otool -l "$BIN" | awk '/cmd LC_BUILD_VERSION/{f=1} f && $1=="minos"{print $2; exit}')
    echo "minos: ${minos:-(no encontrado)}"
    if [ -z "$minos" ]; then
        fail "no hay LC_BUILD_VERSION con minos en el binario"
    elif ! version_le "$minos" "26.0"; then
        fail "el binario exige macOS $minos y el máximo permitido es 26.0"
    fi

    # La primera línea de otool -L es el propio binario.
    libs=$(otool -L "$BIN" | tail -n +2 | sed 's/^[[:space:]]*//; s/ (.*//')
    echo "Bibliotecas enlazadas:"
    echo "$libs" | sed 's/^/  /'
    while IFS= read -r lib; do
        [ -n "$lib" ] || continue
        case "$lib" in
            /usr/lib/*|/System/Library/*) ;;
            *) fail "la biblioteca $lib no es del sistema" ;;
        esac
    done <<< "$libs"
}

audit_windows() {
    local dlls dll lower
    dlls=$(objdump -p "$BIN" | sed -n 's/.*DLL Name: *//p' | tr -d '\r')
    echo "DLL importadas:"
    echo "${dlls:-(ninguna)}" | sed 's/^/  /'
    for dll in $dlls; do
        lower=$(printf '%s' "$dll" | tr '[:upper:]' '[:lower:]')
        case "$lower" in
            kernel32.dll|ntdll.dll|ws2_32.dll|bcryptprimitives.dll|api-ms-win-*) ;;
            *) fail "la DLL $dll no es una DLL del sistema permitida" ;;
        esac
    done
}

case "$TRIPLE" in
    *-unknown-linux-gnu) audit_linux ;;
    aarch64-apple-darwin) audit_macos ;;
    x86_64-pc-windows-gnu) audit_windows ;;
    *) echo "ERROR: triple no soportado: $TRIPLE" >&2; exit 2 ;;
esac

if [ "$FAILED" -ne 0 ]; then
    echo "Auditoría fallida: $BIN ($TRIPLE)" >&2
    exit 1
fi
echo "Auditoría correcta: $BIN ($TRIPLE)"
