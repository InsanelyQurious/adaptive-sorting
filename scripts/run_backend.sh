#!/usr/bin/env bash
# Starts the FastAPI backend on port 8000 (override with BACKEND_PORT).
cd "$(dirname "$0")/.."
source .venv/bin/activate
export MUJOCO_GL=${MUJOCO_GL:-egl}
exec python -m uvicorn backend.main:app --host 0.0.0.0 --port "${BACKEND_PORT:-8000}" --log-level info
