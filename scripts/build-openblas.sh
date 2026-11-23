#!/usr/bin/env bash
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Compila OpenBLAS estático para enlazarlo en el binario, en Linux y en Windows: solo BLAS (sin
# LAPACK ni Fortran), el servidor de hilos propio de OpenBLAS (sin OpenMP) y DYNAMIC_ARCH, que
# elige al arrancar el kernel de la CPU (en aarch64, Neoverse N1/V1/N2, Cortex-A57/A72/A76,
# Apple, etc.).
# La versión por defecto es la 0.3.34, la misma en todas las plataformas.
#
# Base del código común (BASE_TARGET): en x86_64 es CORE2, es decir, el código común sin AVX2
# ni FMA, para que el binario arranque en cualquier CPU x86_64, como el resto del motor (que
# detecta las capacidades de la CPU en tiempo de ejecución); con una base con AVX2 el código
# común llevaría AVX2 y FMA y el binario se caería en CPUs que no los tienen. En aarch64 es ARMV8.
#
# Uso en Linux:
#   scripts/build-openblas.sh <prefijo> [versión]
#   OPENBLAS_LIB_DIR=<prefijo>/lib cargo build --release -p qwen-app
# Uso en Windows (x86_64-pc-windows-gnu), en el bash de MSYS2 UCRT64, con make instalado
# (pacman -S --needed make):
#   scripts/build-openblas.sh /c/openblas
#   OPENBLAS_LIB_DIR=C:/openblas/lib cargo build --release -p qwen-app --target x86_64-pc-windows-gnu
set -euo pipefail

PREFIX="${1:?uso: scripts/build-openblas.sh <prefijo> [versión]}"
VERSION="${2:-0.3.34}"
SRC="$(mktemp -d)"
trap 'rm -rf "$SRC"' EXIT

curl -fsSL --retry 3 -o "$SRC/openblas.tar.gz" \
    "https://github.com/OpenMathLib/OpenBLAS/releases/download/v${VERSION}/OpenBLAS-${VERSION}.tar.gz"
tar -xzf "$SRC/openblas.tar.gz" -C "$SRC"
cd "$SRC/OpenBLAS-${VERSION}"

case "$(uname -m)" in
    aarch64 | arm64) BASE_TARGET=ARMV8 ;;
    x86_64) BASE_TARGET=CORE2 ;;
    *) echo "arquitectura no soportada: $(uname -m)" >&2; exit 1 ;;
esac

FLAGS=(DYNAMIC_ARCH=1 TARGET="$BASE_TARGET" USE_OPENMP=0 USE_THREAD=1 NUM_THREADS=64
       NO_LAPACK=1 NOFORTRAN=1 NO_SHARED=1 BUILD_BFLOAT16=0)
make -j"$(nproc)" "${FLAGS[@]}" libs > build.log 2>&1 || { tail -40 build.log; exit 1; }
make "${FLAGS[@]}" PREFIX="$PREFIX" install > install.log 2>&1 || { tail -40 install.log; exit 1; }
ls -l "$PREFIX/lib/libopenblas.a"
