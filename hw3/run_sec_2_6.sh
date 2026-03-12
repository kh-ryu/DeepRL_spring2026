#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MODE="${MODE:-local}"
CONFIGS=(
  experiments/dqn/lunarlander.yaml
  experiments/dqn/lunarlander_lr_0p00025.yaml
  experiments/dqn/lunarlander_lr_0p0005.yaml
  experiments/dqn/lunarlander_lr_0p002.yaml
)

for cfg in "${CONFIGS[@]}"; do
  if [[ "$MODE" == "modal" ]]; then
    uv run modal run src/scripts/modal_run_dqn.py -- -cfg "$cfg" "$@"
  else
    uv run src/scripts/run_dqn.py -cfg "$cfg" "$@"
  fi
done
