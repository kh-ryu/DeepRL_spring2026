#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MODE="${MODE:-local}"

if [[ "$MODE" == "modal" ]]; then
  uv run modal run src/scripts/modal_run_sac.py -- -cfg experiments/sac/halfcheetah_autotune.yaml "$@"
else
  uv run src/scripts/run_sac.py -cfg experiments/sac/halfcheetah_autotune.yaml "$@"
fi
