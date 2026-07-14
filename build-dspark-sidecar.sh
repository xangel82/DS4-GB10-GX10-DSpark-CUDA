#!/usr/bin/env bash
# GB10/GX10 DSpark CUDA modifications:
# Copyright (c) 2026 Marco Palaferri. Licensed under the MIT License.
set -euo pipefail

HF_DIR="${DS4_DSPARK_HF_DIR:-/home/athena/ds4/dspark-v4flash-hf}"
OUT="${DS4_DSPARK_GGUF:-/home/athena/ds4/DeepSeek-V4-Flash-DSpark-Q4K-Q8.gguf}"
TMP_OUT="${OUT}.partial"

for shard in 46 47 48; do
  file="$HF_DIR/model-000${shard}-of-00048.safetensors"
  if [[ ! -s "$file" ]]; then
    echo "Missing DSpark shard: $file" >&2
    exit 2
  fi
done
if [[ ! -s "$HF_DIR/model.safetensors.index.json" ]]; then
  echo "Missing tensor index: $HF_DIR/model.safetensors.index.json" >&2
  exit 2
fi

# The lab may have been synchronized from macOS, including a Mach-O copy of
# this utility whose timestamp makes GNU make consider it current on Linux.
# Force a native rebuild so the converter always matches the host running it.
make -B -C gguf-tools deepseek4-quantize

echo "DSpark HF:   $HF_DIR"
echo "Sidecar out: $OUT"
echo "Expected size: about 10.7 GiB"

./gguf-tools/deepseek4-quantize \
  --hf "$HF_DIR" \
  --dspark-sidecar \
  --out "$TMP_OUT" \
  --overwrite

mv -f "$TMP_OUT" "$OUT"
echo "DSpark sidecar complete: $OUT"
