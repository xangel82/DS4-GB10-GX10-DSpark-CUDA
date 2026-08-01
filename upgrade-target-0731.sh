#!/usr/bin/env bash
# GB10/GX10 DSpark CUDA modifications:
# Copyright (c) 2026 Marco Palaferri. Licensed under the MIT License.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${DS4_MODEL_DIR:-${DS4_GGUF_DIR:-$HOME/ds4}}"
HF_TOKEN_VALUE="${HF_TOKEN:-}"
DRY_RUN=0

TARGET_FILE="DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf"
PREVIEW_FILE="DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
TARGET_SIZE=86720111488

usage() {
  cat <<'EOF'
Download and activate the Antirez DeepSeek-V4-Flash-0731 Q2/imatrix target.

Usage:
  ./upgrade-target-0731.sh [options]

Options:
  --model-dir DIR   Persistent model directory. Default: $HOME/ds4
  --hf-token TOKEN  Hugging Face token for the resumable download
  --dry-run         Show the resolved upgrade without downloading or relinking
  -h, --help        Show this help

The previous preview GGUF is preserved. The ds4flash.gguf link is changed only
after the complete 0731 file exists and its byte size has been validated.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir)
      [[ $# -ge 2 ]] || { echo "Missing value after --model-dir" >&2; exit 2; }
      MODEL_DIR="$2"
      shift 2
      ;;
    --hf-token)
      [[ $# -ge 2 ]] || { echo "Missing value after --hf-token" >&2; exit 2; }
      HF_TOKEN_VALUE="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$MODEL_DIR" != /* ]]; then
  MODEL_DIR="$PWD/$MODEL_DIR"
fi

target="$MODEL_DIR/$TARGET_FILE"
preview="$MODEL_DIR/$PREVIEW_FILE"
link="$MODEL_DIR/ds4flash.gguf"

echo "DS4 source:      $ROOT"
echo "Model directory: $MODEL_DIR"
echo "0731 target:     $target"
if [[ -e "$preview" ]]; then
  echo "Preview target:  $preview (kept for rollback)"
else
  echo "Preview target:  not present"
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Action: download the 0731 target if missing, validate $TARGET_SIZE bytes, then update $link"
  exit 0
fi

mkdir -p "$MODEL_DIR"
if [[ ! -s "$target" ]]; then
  echo
  echo "==> Downloading the complete 0731 target (86.72 GB decimal)"
  echo "This is a full model download, not a binary delta from the preview weights."
  download_args=(q2-imatrix-0731)
  if [[ -n "$HF_TOKEN_VALUE" ]]; then
    download_args+=(--token "$HF_TOKEN_VALUE")
  fi
  DS4_GGUF_DIR="$MODEL_DIR" "$ROOT/download_model.sh" "${download_args[@]}"
else
  echo "==> Reusing existing 0731 target: $target"
fi

actual_size="$(wc -c < "$target" | tr -d '[:space:]')"
if [[ "$actual_size" != "$TARGET_SIZE" ]]; then
  echo "Invalid 0731 target size: $actual_size bytes (expected $TARGET_SIZE)" >&2
  echo "The active model link was not changed. Move the incomplete file aside and rerun this script." >&2
  exit 1
fi

ln -sfn "$target" "$link"

echo
echo "0731 target ready."
echo "Active link: $link -> $target"
echo "The preview GGUF and existing DSpark sidecars were not modified."
echo "Start the server with:"
echo "  cd $ROOT && DS4_MODEL_DIR=$MODEL_DIR ./run-dspark-server.sh"
