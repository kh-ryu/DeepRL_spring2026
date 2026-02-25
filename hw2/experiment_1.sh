uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 \
--exp_name cartpole --video_log_freq 10 --which_gpu 2

uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 \
-rtg --exp_name cartpole_rtg --video_log_freq 10 --which_gpu 2

uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 \
-na --exp_name cartpole_na --video_log_freq 10 --which_gpu 2

uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 \
-rtg -na --exp_name cartpole_rtg_na --video_log_freq 10 --which_gpu 2

uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 4000 \
--exp_name cartpole_lb --video_log_freq 10 --which_gpu 2

uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 4000 \
-rtg --exp_name cartpole_lb_rtg --video_log_freq 10 --which_gpu 2

uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 4000 \
-na --exp_name cartpole_lb_na --video_log_freq 10 --which_gpu 2

uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 4000 \
-rtg -na --exp_name cartpole_lb_rtg_na --video_log_freq 10 --which_gpu 2