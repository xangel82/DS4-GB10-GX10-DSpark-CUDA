#!/usr/bin/env bash
# GB10/GX10 DSpark CUDA modifications:
# Copyright (c) 2026 Marco Palaferri. Licensed under the MIT License.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# Capacity-first profile validated on GB10.  Keep these values fixed so a stale
# shell environment cannot accidentally select the 8192-token chunk that OOMed
# with a 1M physical context.
export DS4_CTX=1048576
export DS4_PREFILL_CHUNK=4096
export DS4_KV_CACHE_COLD_MAX_TOKENS=1048576
export DS4_KV_LONG_COLD_ANCHOR_MIN_TOKENS=524288
export DS4_KV_LONG_COLD_ANCHOR_TRIM_TOKENS=65536
export DS4_ADVERTISE_CONTEXT_PCT=85
export DS4_MEMORY_PROFILE=balanced

# Keep experimental checkpoints separate from the recommended 256K profile.
export DS4_EXPERIMENT_KV_DIR="${DS4_EXPERIMENT_KV_DIR:-/tmp/ds4-gb10-dspark-1m-kv}"
export DS4_TELEMETRY="${DS4_TELEMETRY:-1}"

echo "Experimental 1M profile: stop any active ds4-server before continuing."
echo "KV isolation: $DS4_EXPERIMENT_KV_DIR"

exec "$ROOT/run-dspark-server.sh"
