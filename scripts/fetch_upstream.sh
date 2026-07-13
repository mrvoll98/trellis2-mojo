#!/bin/sh
# Henter oppstrøms TRELLIS.2 (paritets-fasiten) inn i repo-rota:
# trellis2/ (originalkoden testene sammenligner mot), o-voxel/ og
# configs/. Pinnet til commiten porten ble verifisert mot.
set -eu
UPSTREAM_SHA=75fbf0183001ed9876c8dbb35de6b68552ee08bd
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
git clone --filter=blob:none https://github.com/microsoft/TRELLIS.2 "$TMP/upstream"
git -C "$TMP/upstream" checkout -q "$UPSTREAM_SHA"
for d in trellis2 o-voxel configs; do
    rm -rf "${ROOT:?}/$d"
    cp -R "$TMP/upstream/$d" "$ROOT/$d"
done
# Mac-kompatibilitets-patchene (9 filer: CPU-conv/attention-backends,
# ingen harde .cuda()-kall, rembg-guard, stub-hook for mesh-
# ekstraksjon) — paritetstestene er verifisert mot NØYAKTIG denne
# tilstanden (hentet + patchet == fasiten suiten var grønn mot)
(cd "$ROOT" && patch -p1 -s < scripts/upstream_mac_compat.patch)
echo "Hentet trellis2/, o-voxel/ og configs/ @ ${UPSTREAM_SHA} (+ mac-compat-patch)"
