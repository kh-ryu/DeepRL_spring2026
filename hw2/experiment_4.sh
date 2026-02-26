#!/usr/bin/env bash
set -euo pipefail

# Run from hw2 directory:
#   bash experiment_4.sh
#
# Goal: reach InvertedPendulum-v4 return ~= 1000 with fewer than 100K env steps.
# The first command is the default baseline; the following commands are tuned
# candidates ordered from strongest prior to weaker alternatives.

GPU_ID="1"

# 1) Default baseline (for comparison)
MUJOCO_GL=egl uv run src/scripts/run.py \
  --env_name InvertedPendulum-v4 \
  -n 100 -b 5000 -eb 1000 \
  --exp_name pendulum_default \
  --which_gpu "${GPU_ID}"

# 2) Best-first tuned run:
#    - smaller batch for finer-grained policy updates (better sample efficiency)
#    - baseline + RTG + advantage normalization + GAE for lower variance
#    - moderately larger network for smoother control value fit
MUJOCO_GL=egl uv run src/scripts/run.py \
  --env_name InvertedPendulum-v4 \
  -n 120 -b 1000 -eb 1000 \
  --discount 0.99 -lr 3e-3 \
  --use_reward_to_go --use_baseline \
  -blr 1e-3 -bgs 10 \
  -na --gae_lambda 0.95 \
  -l 2 -s 128 \
  --exp_name pendulum_tuned_best \
  --which_gpu "${GPU_ID}"

# 3) Alternative: slightly larger batch and higher lambda
MUJOCO_GL=egl uv run src/scripts/run.py \
  --env_name InvertedPendulum-v4 \
  -n 120 -b 2000 -eb 1000 \
  --discount 0.99 -lr 3e-3 \
  --use_reward_to_go --use_baseline \
  -blr 1e-3 -bgs 10 \
  -na --gae_lambda 0.98 \
  -l 2 -s 128 \
  --exp_name pendulum_tuned_b2000_lam098 \
  --which_gpu "${GPU_ID}"

# 4) Alternative: stronger discount and slightly lower actor lr
MUJOCO_GL=egl uv run src/scripts/run.py \
  --env_name InvertedPendulum-v4 \
  -n 120 -b 1000 -eb 1000 \
  --discount 0.995 -lr 1e-3 \
  --use_reward_to_go --use_baseline \
  -blr 1e-3 -bgs 10 \
  -na --gae_lambda 0.95 \
  -l 2 -s 128 \
  --exp_name pendulum_tuned_lr1e3_gamma0995 \
  --which_gpu "${GPU_ID}"
