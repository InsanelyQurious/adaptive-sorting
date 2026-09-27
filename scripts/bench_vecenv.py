"""Throughput benchmark for the vectorized environment (sizing the training run)."""
import os, sys, time, numpy as np
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stable_baselines3.common.vec_env import SubprocVecEnv
from simulation.environments.sorting_env import make_env

def bench(n_envs, image, steps=300):
    venv = SubprocVecEnv([lambda: make_env(include_image=image) for _ in range(n_envs)], start_method="forkserver")
    venv.reset()
    t0 = time.time()
    for _ in range(steps):
        venv.step(np.random.uniform(-1, 1, (n_envs, 7)).astype(np.float32))
    dt = time.time() - t0; venv.close()
    print(f"n_envs={n_envs:2d} image={image}: {n_envs*steps/dt:6.0f} env steps/s", flush=True)

if __name__ == "__main__":
    for n in (4, 8, 12):
        bench(n, True)
    bench(8, False)
