#!/usr/bin/env bash
# GB10/GX10 DSpark CUDA modifications:
# Copyright (c) 2026 Marco Palaferri. Licensed under the MIT License.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
ATHENA_HOST="${ATHENA_HOST:-}"
ATHENA_USER="${ATHENA_USER:-athena}"
ATHENA_DEST="${ATHENA_DEST:-/home/athena/DS4-GB10-GX10-DSpark-CUDA/}"
SSH_KEY="${SSH_KEY:-}"

if [[ -z "$ATHENA_HOST" ]]; then
  echo "ATHENA_HOST is required, for example:" >&2
  echo "  ATHENA_HOST=<gb10-host> SSH_KEY=~/.ssh/<your-key> ./deploy-athena.sh" >&2
  exit 2
fi

if [[ -z "$SSH_KEY" ]]; then
  echo "SSH_KEY is required, for example:" >&2
  echo "  ATHENA_HOST=<gb10-host> SSH_KEY=~/.ssh/<your-key> ./deploy-athena.sh" >&2
  exit 2
fi

if [[ ! -f "$SSH_KEY" ]]; then
  echo "SSH key not found: $SSH_KEY" >&2
  exit 2
fi

rsync -az --progress \
  -e "ssh -i $SSH_KEY -o ConnectTimeout=8" \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='*.o' \
  --exclude='ds4' \
  --exclude='ds4-server' \
  --exclude='ds4-bench' \
  --exclude='ds4-eval' \
  --exclude='ds4-agent' \
  --exclude='ds4_test' \
  --exclude='ds4_agent_test' \
  --exclude='tests/test_q4k_dot' \
  --exclude='tests/cuda_long_context_smoke' \
  --exclude='gguf-tools/deepseek4-quantize' \
  --exclude='benchmark-results' \
  "$ROOT/" \
  "$ATHENA_USER@$ATHENA_HOST:$ATHENA_DEST"

echo
echo "Deploy complete: $ATHENA_USER@$ATHENA_HOST:$ATHENA_DEST"
echo "Next on Athena:"
echo "  cd $ATHENA_DEST && make -B cuda-regression CUDA_ARCH=sm_121a"
echo "  cd $ATHENA_DEST && make -B cuda-spark-graph-sm121"
echo "  cd $ATHENA_DEST && ./run-dspark-server.sh 2>&1 | tee /tmp/ds4-dspark-server.log"
