#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf"
PREVIEW="DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/ds4-install-test.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

bash -n \
  "$ROOT/install-gb10.sh" \
  "$ROOT/run-dspark-server.sh" \
  "$ROOT/run-dspark-server-1m.sh" \
  "$ROOT/upgrade-target-0731.sh"
sh -n "$ROOT/download_model.sh"

install_out="$("$ROOT/install-gb10.sh" --dspark q2 --model-dir "$TMP/install" --dry-run)"
grep -Fq "Target release:   DeepSeek-V4-Flash-0731" <<<"$install_out"
grep -Fq "Target file:      $TARGET" <<<"$install_out"

upgrade_out="$("$ROOT/upgrade-target-0731.sh" --model-dir "$TMP/upgrade" --dry-run)"
grep -Fq "$TARGET" <<<"$upgrade_out"
grep -Fq "validate 86720111488 bytes" <<<"$upgrade_out"

help_out="$("$ROOT/download_model.sh" --help)"
grep -Fq "q2-imatrix-0731" <<<"$help_out"
grep -Fq "q2-imatrix-preview" <<<"$help_out"

mkdir -p "$TMP/missing"
set +e
missing_out="$(DS4_MODEL_DIR="$TMP/missing" "$ROOT/run-dspark-server.sh" 2>&1)"
missing_rc=$?
set -e
[[ "$missing_rc" -eq 2 ]]
grep -Fq "upgrade-target-0731.sh --model-dir $TMP/missing" <<<"$missing_out"

mkdir -p "$TMP/guard"
printf 'preview' > "$TMP/guard/$PREVIEW"
printf 'incomplete' > "$TMP/guard/$TARGET"
ln -s "$TMP/guard/$PREVIEW" "$TMP/guard/ds4flash.gguf"
set +e
guard_out="$("$ROOT/upgrade-target-0731.sh" --model-dir "$TMP/guard" 2>&1)"
guard_rc=$?
set -e
[[ "$guard_rc" -ne 0 ]]
grep -Fq "Invalid 0731 target size" <<<"$guard_out"
[[ "$(readlink "$TMP/guard/ds4flash.gguf")" == "$TMP/guard/$PREVIEW" ]]

echo "gb10 install scripts: OK"
