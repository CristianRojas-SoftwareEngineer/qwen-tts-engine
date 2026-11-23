#!/usr/bin/env bash
# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Publica la release de GitHub del tag CIRCLE_TAG con los seis archivos de la carpeta indicada.
# Uso: GH_TOKEN=... CIRCLE_TAG=v1.0.0 bash scripts/ci-publish-release.sh <carpeta con los seis archivos>
#
# La publicación es atómica: la release se crea como borrador, se suben los seis archivos, se
# comprueba que la release los lista y solo entonces se publica. Si algo falla antes, queda en
# borrador y el script sale con 1. Una release (publicada o en borrador) que ya tenga ese tag
# detiene el script sin tocar nada: una release publicada no cambia y una corrección es otra versión.
set -euo pipefail

REPO="CristianRojas-SoftwareEngineer/qwen-tts-engine"
API="https://api.github.com/repos/$REPO"
DIR="${1:?Uso: ci-publish-release.sh <carpeta con los seis archivos>}"
: "${GH_TOKEN:?Falta GH_TOKEN}"
: "${CIRCLE_TAG:?Falta CIRCLE_TAG: el job solo corre con un tag}"
TAG="$CIRCLE_TAG"
V="${TAG#v}"

# Archivos que debe contener la release, en el orden en que se suben.
EXPECTED=(
    "qwen-tts-engine-$TAG-x86_64-pc-windows-gnu.zip"
    "qwen-tts-engine-$TAG-x86_64-unknown-linux-gnu.tar.gz"
    "qwen-tts-engine-$TAG-aarch64-unknown-linux-gnu.tar.gz"
    "qwen-tts-engine-$TAG-aarch64-apple-darwin.tar.gz"
    "qwen-tts-engine-$TAG-src.tar.gz"
    "SHA256SUMS"
)

# Llamada a la API de GitHub: api <método> <url> [cuerpo JSON]. Falla si el estado no es 2xx.
api() {
    local method="$1" url="$2" body="${3:-}"
    local args=(-sS -X "$method" -H "Authorization: Bearer $GH_TOKEN" -H "Accept: application/vnd.github+json" -H "X-GitHub-Api-Version: 2022-11-28" -w '\n%{http_code}')
    if [ -n "$body" ]; then args+=(-H "Content-Type: application/json" --data "$body"); fi
    local out status
    out=$(curl "${args[@]}" "$url")
    status="${out##*$'\n'}"
    out="${out%$'\n'*}"
    if [ "${status:0:1}" != "2" ]; then
        echo "ERROR: $method $url devolvió HTTP $status: $out" >&2
        return 1
    fi
    printf '%s\n' "$out"
}

# Los seis archivos deben existir antes de tocar GitHub.
for f in "${EXPECTED[@]}"; do
    if [ ! -f "$DIR/$f" ]; then
        echo "ERROR: falta $DIR/$f" >&2
        exit 1
    fi
done

# Una release que ya tenga este tag, publicada o en borrador, detiene el job sin tocar nada.
EXISTING=$(api GET "$API/releases?per_page=100" | jq -r --arg tag "$TAG" '[.[] | select(.tag_name == $tag)] | length')
if [ "$EXISTING" != "0" ]; then
    echo "ERROR: la release $TAG ya existe (publicada o en borrador); no se toca nada." >&2
    exit 1
fi

BODY_TEXT=$(cat <<TEXT
Versión $V del motor de síntesis de voz para Qwen3-TTS 0.6B Base. Las versiones siguen el versionado semántico sobre el contrato de uso (docs/INTERFACE_CONTRACT.md, sección «Versionado»).

## Archivos

- \`qwen-tts-engine-$TAG-x86_64-pc-windows-gnu.zip\`: binario \`qwen_tts.exe\` para Windows x86_64.
- \`qwen-tts-engine-$TAG-x86_64-unknown-linux-gnu.tar.gz\`: binario \`qwen_tts\` para Linux x86_64 (glibc 2.28 o posterior).
- \`qwen-tts-engine-$TAG-aarch64-unknown-linux-gnu.tar.gz\`: binario \`qwen_tts\` para Linux ARM64 (glibc 2.28 o posterior).
- \`qwen-tts-engine-$TAG-aarch64-apple-darwin.tar.gz\`: binario \`qwen_tts\` para macOS ARM64 (macOS 26.0 o posterior).
- \`qwen-tts-engine-$TAG-src.tar.gz\`: código fuente de esta versión.
- \`SHA256SUMS\`: hash SHA-256 de los otros cinco archivos.

Cada paquete contiene una carpeta \`qwen-tts-engine-$TAG-<target>/\` con el binario, \`LICENSE\`, \`NOTICE\` y \`THIRD-PARTY-LICENSES.md\`.

## Verificación

Descarga todos los archivos en una misma carpeta y ejecuta en ella:

\`\`\`
sha256sum -c SHA256SUMS
\`\`\`

## Modelo

Los pesos del modelo no van en los paquetes: se descargan aparte, como explica la sección «Descarga de modelos» del README.

## Contrato

Lo que el motor garantiza a los programas que lo usan como proceso o por HTTP está en docs/INTERFACE_CONTRACT.md.
TEXT
)

# Cuerpo de la petición de creación: borrador, no prerelease.
CREATE_BODY=$(jq -n --arg tag "$TAG" --arg body "$BODY_TEXT" '{tag_name: $tag, name: $tag, draft: true, prerelease: false, body: $body}')
RELEASE=$(api POST "$API/releases" "$CREATE_BODY")
RELEASE_ID=$(printf '%s' "$RELEASE" | jq -r '.id')
# La upload_url trae el sufijo de plantilla {?name,label}, que se quita antes de añadir el nombre.
UPLOAD_URL=$(printf '%s' "$RELEASE" | jq -r '.upload_url' | sed 's/{?name,label}$//')
echo "Borrador creado: id $RELEASE_ID"

for f in "${EXPECTED[@]}"; do
    echo "Subiendo $f"
    curl -sS -f -X POST -H "Authorization: Bearer $GH_TOKEN" -H "Accept: application/vnd.github+json" \
        -H "Content-Type: application/octet-stream" --data-binary "@$DIR/$f" \
        "$UPLOAD_URL?name=$f" > /dev/null
done

# La release debe listar exactamente los seis nombres esperados antes de publicarse. Las dos
# listas se ordenan con el mismo `sort` en orden de bytes (LC_ALL=C): con el locale del sistema,
# `sort` pone `SHA256SUMS` en otro lugar que el orden por código de `jq` y la comparación falla.
LISTED=$(api GET "$API/releases/$RELEASE_ID" | jq -r '.assets[].name' | LC_ALL=C sort)
WANTED=$(printf '%s\n' "${EXPECTED[@]}" | LC_ALL=C sort)
if [ "$LISTED" != "$WANTED" ]; then
    echo "ERROR: los archivos de la release no son los esperados." >&2
    echo "Esperados:" >&2; printf '%s\n' "$WANTED" >&2
    echo "Listados:" >&2; printf '%s\n' "$LISTED" >&2
    echo "La release $TAG queda en borrador." >&2
    exit 1
fi

api PATCH "$API/releases/$RELEASE_ID" '{"draft": false}' | jq -r '"Release publicada: " + .html_url'
