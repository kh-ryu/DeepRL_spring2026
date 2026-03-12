#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MODE="${MODE:-local}"

if [[ "$MODE" == "modal" ]]; then
  uv run modal run src/scripts/modal_run_dqn.py -- -cfg experiments/dqn/lunarlander.yaml "$@"
  uv run modal run src/scripts/modal_run_dqn.py -- -cfg experiments/dqn/mspacman.yaml "$@"
else
  uv run src/scripts/run_dqn.py -cfg experiments/dqn/lunarlander.yaml "$@"
  uv run src/scripts/run_dqn.py -cfg experiments/dqn/mspacman.yaml "$@"
fi
