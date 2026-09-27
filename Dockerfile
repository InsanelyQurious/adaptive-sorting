# Live backend for cloud hosting (Render / any Docker host). CPU-only: MuJoCo renders with OSMesa software OpenGL,
# PyTorch is the CPU build. The React dashboard is built in a separate stage and served by the backend at "/".
# ---------------------------------------------------------------- frontend build
FROM node:22-slim AS web
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
# same-origin API: the dashboard is served by the backend itself, so no VITE_BACKEND_URL is needed
RUN npm run build

# ---------------------------------------------------------------- backend
FROM python:3.12-slim AS app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PORT=8000
RUN apt-get update && apt-get install -y --no-install-recommends \
        libosmesa6 libgl1 libglu1-mesa libegl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
COPY backend ./backend
COPY simulation ./simulation
COPY training ./training
COPY lerobot_adapter ./lerobot_adapter
COPY configs ./configs
COPY scripts ./scripts
COPY checkpoints ./checkpoints
COPY datasets/demos ./datasets/demos
COPY logs ./logs
COPY README.md ASSUMPTIONS.md DEBUGGING.md PROGRESS.md pytest.ini ./
COPY --from=web /app/frontend/dist ./frontend/dist
RUN mkdir -p datasets/live_recordings logs/eval_reels && python -c "import mujoco, torch, stable_baselines3; print('mujoco', mujoco.__version__, 'torch', torch.__version__)"
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s CMD curl -fsS http://localhost:${PORT}/health || exit 1
CMD ["sh", "-c", "exec python -m uvicorn backend.main:app --host 0.0.0.0 --port ${PORT} --log-level info"]
