#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MODE="${MODE:-local}"

if [[ "$MODE" == "modal" ]]; then
  uv run modal run src/scripts/modal_run_dqn.py -- -cfg experiments/dqn/cartpole.yaml --eval_interval 2500 "$@"
else
  uv run src/scripts/run_dqn.py -cfg experiments/dqn/cartpole.yaml --eval_interval 2500 "$@"
fi
