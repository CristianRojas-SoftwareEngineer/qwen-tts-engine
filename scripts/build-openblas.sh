#!/usr/bin/env bash
# Compila OpenBLAS estático para enlazarlo en el binario (Linux): solo BLAS (sin LAPACK ni
# Fortran), hilos POSIX y DYNAMIC_ARCH, que elige al arrancar el kernel de la CPU (en aarch64,
# Neoverse N1/V1/N2, Cortex-A57/A72/A76, Apple, etc.). Después:
#   OPENBLAS_LIB_DIR=<prefijo>/lib cargo build --release -p qwen-app
# Uso: scripts/build-openblas.sh <prefijo> [versión]   (por defecto 0.3.30)
set -euo pipefail

PREFIX="${1:?uso: scripts/build-openblas.sh <prefijo> [versión]}"
VERSION="${2:-0.3.30}"
SRC="$(mktemp -d)"
trap 'rm -rf "$SRC"' EXIT

curl -fsSL --retry 3 -o "$SRC/openblas.tar.gz" \
    "https://github.com/OpenMathLib/OpenBLAS/releases/download/v${VERSION}/OpenBLAS-${VERSION}.tar.gz"
tar -xzf "$SRC/openblas.tar.gz" -C "$SRC"
cd "$SRC/OpenBLAS-${VERSION}"

case "$(uname -m)" in
    aarch64 | arm64) BASE_TARGET=ARMV8 ;;
    x86_64) BASE_TARGET=HASWELL ;;
    *) echo "arquitectura no soportada: $(uname -m)" >&2; exit 1 ;;
esac

FLAGS=(DYNAMIC_ARCH=1 TARGET="$BASE_TARGET" USE_OPENMP=0 USE_THREAD=1 NUM_THREADS=64
       NO_LAPACK=1 NOFORTRAN=1 NO_SHARED=1 BUILD_BFLOAT16=0)
make -j"$(nproc)" "${FLAGS[@]}" libs > build.log 2>&1 || { tail -40 build.log; exit 1; }
make "${FLAGS[@]}" PREFIX="$PREFIX" install > install.log 2>&1 || { tail -40 install.log; exit 1; }
ls -l "$PREFIX/lib/libopenblas.a"
