#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

SEED="${SEED:-0}"

# download_modal_logs() {
#   local idx=1
#   local dest
#   while [[ -e "part3_exp_${idx}" ]]; do
#     idx=$((idx + 1))
#   done
#   dest="part3_exp_${idx}"
#   uv run modal volume get hw5-offline-rl-volume / "$dest"
# }

# uv run modal run src/scripts/modal_run.py --njobs=4 \
#   "JOB --run_group=q3 --base_config=fql --env_name=antmaze-medium-navigate-singletask-task1-v0 --seed=$SEED --alpha=1" \
#   "JOB --run_group=q3 --base_config=fql --env_name=antmaze-medium-navigate-singletask-task1-v0 --seed=$SEED --alpha=3" \
#   "JOB --run_group=q3 --base_config=fql --env_name=antmaze-medium-navigate-singletask-task1-v0 --seed=$SEED --alpha=10" \
#   "JOB --run_group=q3 --base_config=fql --env_name=antmaze-medium-navigate-singletask-task1-v0 --seed=$SEED --alpha=30"
# download_modal_logs

uv run modal run src/scripts/modal_run.py --njobs=4 \
  "JOB --run_group=q3 --base_config=fql --env_name=cube-single-play-singletask-task1-v0 --seed=$SEED --alpha=30" \
  "JOB --run_group=q3 --base_config=fql --env_name=cube-single-play-singletask-task1-v0 --seed=$SEED --alpha=100" \
  "JOB --run_group=q3 --base_config=fql --env_name=cube-single-play-singletask-task1-v0 --seed=$SEED --alpha=300" \
  "JOB --run_group=q3 --base_config=fql --env_name=cube-single-play-singletask-task1-v0 --seed=$SEED --alpha=1000"
# download_modal_logs

uv run modal run src/scripts/modal_run.py --njobs=4 \
  "JOB --run_group=q3 --base_config=fql --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=$SEED --alpha=1" \
  "JOB --run_group=q3 --base_config=fql --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=$SEED --alpha=3" \
  "JOB --run_group=q3 --base_config=fql --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=$SEED --alpha=10" \
  "JOB --run_group=q3 --base_config=fql --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=$SEED --alpha=30"
# download_modal_logs
