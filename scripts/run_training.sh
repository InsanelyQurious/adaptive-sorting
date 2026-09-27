#!/usr/bin/env bash
# Starts the PPO training process (independent of the backend). Extra args are passed through,
# e.g. scripts/run_training.sh --resume latest   |   scripts/run_training.sh --set total_timesteps=500000
cd "$(dirname "$0")/.."
source .venv/bin/activate
export MUJOCO_GL=${MUJOCO_GL:-egl} PYTHONUNBUFFERED=1
exec python -m training.train "$@"
