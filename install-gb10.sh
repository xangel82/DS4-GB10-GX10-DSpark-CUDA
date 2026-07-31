#!/usr/bin/env bash
# GB10/GX10 DSpark CUDA modifications:
# Copyright (c) 2026 Marco Palaferri. Licensed under the MIT License.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${DS4_MODEL_DIR:-${DS4_GGUF_DIR:-$HOME/ds4}}"
HF_DIR="${DS4_DSPARK_HF_DIR:-$MODEL_DIR/dspark-v4flash-hf}"
DSPARK_VARIANTS="q2"
INSTALL_DEPS=0
FORCE_SIDECAR=0
SKIP_TARGET=0
SKIP_REGRESSION=0
DRY_RUN=0
HF_TOKEN_VALUE="${HF_TOKEN:-}"

TARGET_FILE="DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
Q4_FILE="DeepSeek-V4-Flash-DSpark-Q4K-Q8.gguf"
Q2_FILE="DeepSeek-V4-Flash-DSpark-IQ2XXS-Q2K-Q8.gguf"
HF_REPO="deepseek-ai/DeepSeek-V4-Flash-DSpark"
HF_FILES=(
  config.json
  model.safetensors.index.json
  model-00046-of-00048.safetensors
  model-00047-of-00048.safetensors
  model-00048-of-00048.safetensors
)

usage() {
  cat <<'EOF'
Install DS4 GB10/GX10 with DeepSeek-V4-Flash and DSpark.

Usage:
  ./install-gb10.sh [options]

Options:
  --dspark q4|q2|both   Sidecar variant to build. Default: q2
  --model-dir DIR       Persistent model directory. Default: $HOME/ds4
  --hf-token TOKEN      Hugging Face token for model downloads
  --install-deps        Install apt build/download dependencies with sudo
  --force-sidecar       Rebuild an existing sidecar
  --skip-target         Do not download the main DeepSeek-V4-Flash GGUF
  --skip-regression     Build without running cuda-regression
  --dry-run             Print resolved paths and planned work, then exit
  -h, --help            Show this help

Examples:
  ./install-gb10.sh --install-deps --dspark q4
  ./install-gb10.sh --install-deps --dspark q2
  ./install-gb10.sh --install-deps --dspark both

Downloads are resumable. Existing target and sidecar GGUF files are reused
unless --force-sidecar is supplied.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dspark)
      [[ $# -ge 2 ]] || { echo "Missing value after --dspark" >&2; exit 2; }
      DSPARK_VARIANTS="$2"
      shift 2
      ;;
    --model-dir)
      [[ $# -ge 2 ]] || { echo "Missing value after --model-dir" >&2; exit 2; }
      MODEL_DIR="$2"
      HF_DIR="$MODEL_DIR/dspark-v4flash-hf"
      shift 2
      ;;
    --hf-token)
      [[ $# -ge 2 ]] || { echo "Missing value after --hf-token" >&2; exit 2; }
      HF_TOKEN_VALUE="$2"
      shift 2
      ;;
    --install-deps)
      INSTALL_DEPS=1
      shift
      ;;
    --force-sidecar)
      FORCE_SIDECAR=1
      shift
      ;;
    --skip-target)
      SKIP_TARGET=1
      shift
      ;;
    --skip-regression)
      SKIP_REGRESSION=1
      shift
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

case "$DSPARK_VARIANTS" in
  q4) VARIANTS=(q4) ;;
  q2) VARIANTS=(q2) ;;
  both) VARIANTS=(q4 q2) ;;
  *)
    echo "Invalid --dspark value: $DSPARK_VARIANTS (expected q4, q2 or both)" >&2
    exit 2
    ;;
esac

if [[ "$MODEL_DIR" != /* ]]; then
  MODEL_DIR="$PWD/$MODEL_DIR"
fi
if [[ "$HF_DIR" != /* ]]; then
  HF_DIR="$PWD/$HF_DIR"
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DS4 source:       $ROOT"
  echo "Model directory:  $MODEL_DIR"
  echo "DSpark HF shards: $HF_DIR"
  echo "DSpark variants:  ${VARIANTS[*]}"
  echo "Download target:  $([[ "$SKIP_TARGET" == "1" ]] && echo no || echo yes)"
  echo "CUDA regression:  $([[ "$SKIP_REGRESSION" == "1" ]] && echo no || echo yes)"
  echo "Force sidecar:    $([[ "$FORCE_SIDECAR" == "1" ]] && echo yes || echo no)"
  exit 0
fi

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This installer targets a Linux GB10/GX10 host." >&2
  exit 2
fi

if [[ "$INSTALL_DEPS" == "1" ]]; then
  command -v apt-get >/dev/null 2>&1 || {
    echo "--install-deps requires apt-get" >&2
    exit 2
  }
  sudo apt-get update
  sudo apt-get install -y build-essential git curl wget rsync python3
fi

for cmd in make curl wget ln; do
  command -v "$cmd" >/dev/null 2>&1 || {
    echo "Missing required command: $cmd" >&2
    echo "Run again with --install-deps or install the missing package." >&2
    exit 2
  }
done

NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
if [[ ! -x "$NVCC" ]]; then
  echo "CUDA compiler not found: $NVCC" >&2
  exit 2
fi

if [[ "$SKIP_REGRESSION" == "0" ]] && pgrep -x ds4-server >/dev/null 2>&1; then
  echo "A ds4-server process is running. Stop it before the CUDA regression" >&2
  echo "so the test has enough unified-memory headroom, or rerun with" >&2
  echo "--skip-regression to download, convert and build without the test." >&2
  exit 2
fi

mkdir -p "$MODEL_DIR" "$HF_DIR"

echo "DS4 source:       $ROOT"
echo "Model directory:  $MODEL_DIR"
echo "DSpark HF shards: $HF_DIR"
echo "DSpark variants:  ${VARIANTS[*]}"
echo

if [[ "$SKIP_TARGET" == "0" ]]; then
  echo "==> Downloading the DeepSeek-V4-Flash Q2/imatrix target"
  download_args=(q2-imatrix)
  if [[ -n "$HF_TOKEN_VALUE" ]]; then
    download_args+=(--token "$HF_TOKEN_VALUE")
  fi
  DS4_GGUF_DIR="$MODEL_DIR" "$ROOT/download_model.sh" "${download_args[@]}"
  ln -sfn "$MODEL_DIR/$TARGET_FILE" "$MODEL_DIR/ds4flash.gguf"
else
  echo "==> Skipping target download"
fi

if [[ ! -s "$MODEL_DIR/ds4flash.gguf" ]]; then
  echo "Target model is missing: $MODEL_DIR/ds4flash.gguf" >&2
  echo "Remove --skip-target or provide the expected symlink/file." >&2
  exit 2
fi

echo
echo "==> Downloading official DeepSeek-V4-Flash-DSpark shards"
for file in "${HF_FILES[@]}"; do
  out="$HF_DIR/$file"
  if [[ -s "$out" ]]; then
    echo "Already downloaded: $out"
    continue
  fi
  wget_args=(-c)
  if [[ -n "$HF_TOKEN_VALUE" ]]; then
    wget_args+=(--header="Authorization: Bearer $HF_TOKEN_VALUE")
  fi
  wget "${wget_args[@]}" \
    "https://huggingface.co/$HF_REPO/resolve/main/$file" \
    -O "$out"
done

for variant in "${VARIANTS[@]}"; do
  if [[ "$variant" == "q4" ]]; then
    sidecar="$MODEL_DIR/$Q4_FILE"
  else
    sidecar="$MODEL_DIR/$Q2_FILE"
  fi

  echo
  if [[ -s "$sidecar" && "$FORCE_SIDECAR" == "0" ]]; then
    echo "==> Reusing existing $variant sidecar: $sidecar"
  else
    echo "==> Building $variant sidecar: $sidecar"
    DS4_DSPARK_VARIANT="$variant" \
    DS4_DSPARK_HF_DIR="$HF_DIR" \
    DS4_DSPARK_GGUF="$sidecar" \
      "$ROOT/build-dspark-sidecar.sh"
  fi
done

echo
if [[ "$SKIP_REGRESSION" == "0" ]]; then
  echo "==> Running the mandatory SM121a CUDA regression"
  make -C "$ROOT" -B cuda-regression CUDA_ARCH=sm_121a
else
  echo "==> Skipping cuda-regression"
fi

echo
echo "==> Building the native SM121a CUDA Graph server"
make -C "$ROOT" -B cuda-spark-graph-sm121

echo
echo "Installation complete."
echo "Target: $MODEL_DIR/ds4flash.gguf"
for variant in "${VARIANTS[@]}"; do
  if [[ "$variant" == "q4" ]]; then
    echo "Q4 sidecar: $MODEL_DIR/$Q4_FILE"
  else
    echo "Q2 sidecar: $MODEL_DIR/$Q2_FILE"
  fi
done
echo
echo "Start Q2 (default):"
echo "  cd $ROOT && DS4_MODEL_DIR=$MODEL_DIR ./run-dspark-server.sh"
echo "Start Q4 (optional):"
echo "  cd $ROOT && DS4_MODEL_DIR=$MODEL_DIR DS4_DSPARK_VARIANT=q4 ./run-dspark-server.sh"
