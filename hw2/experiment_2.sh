# MUJOCO_GL=egl uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg \
# --discount 0.95 -lr 0.01 --exp_name cheetah --video_log_freq 10 --which_gpu 2

# MUJOCO_GL=egl uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg \
# --discount 0.95 -lr 0.01 --use_baseline -blr 0.01 -bgs 5 --exp_name cheetah_baseline --video_log_freq 10 --which_gpu 2

MUJOCO_GL=egl uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg \
--discount 0.95 -lr 0.01 --use_baseline -blr 0.001 -bgs 5 --exp_name cheetah_baseline_small_lr --video_log_freq 10 --which_gpu 2 \

MUJOCO_GL=egl uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg \
--discount 0.95 -lr 0.01 --use_baseline -blr 0.01 -bgs 5 --exp_name cheetah_baseline_na --video_log_freq 10 --which_gpu 2 -na
