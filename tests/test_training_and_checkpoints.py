"""Checkpoint save/load, policy inference, LeRobot dataset adapter, training smoke test."""
import json, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_checkpoint_save_and_load(tmp_path):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from simulation.environments.sorting_env import make_env
    from training.checkpointing import save_checkpoint, list_checkpoints, latest_checkpoint
    from backend.policy_loader import LoadedPolicy
    venv = VecNormalize(DummyVecEnv([lambda: make_env(include_image=False)]), norm_obs_keys=["state"])
    model = PPO("MultiInputPolicy", venv, n_steps=16, batch_size=16, n_epochs=1, verbose=0, device="cpu")
    model.learn(32)
    path = save_checkpoint(model, venv, tmp_path, model.num_timesteps, {"training": {"a": 1}, "environment": {}, "simulation": {}},
                           {"algorithm": "PPO", "episodes": 0})
    assert (path / "model.zip").exists() and (path / "vecnormalize.pkl").exists() and (path / "metadata.json").exists()
    assert (path / "training_config.yaml").exists()
    assert latest_checkpoint(tmp_path) == path
    assert list_checkpoints(tmp_path)[0]["timesteps"] == model.num_timesteps
    pol = LoadedPolicy(path, device="cpu")
    e = make_env(sb3=False, include_image=False)
    obs, _ = e.reset(seed=0)
    a = pol.select_action(obs)
    assert a.shape == (7,) and np.all(np.abs(a) <= 1.0) and pol.latency_ms > 0
    e.close(); venv.close()


def test_lerobot_dataset_adapter(tmp_path):
    from simulation.environments.sorting_env import make_env, build_state_layout
    from lerobot_adapter.dataset_writer import LeRobotEpisodeRecorder, load_dataset
    env = make_env(sb3=False)
    rec = LeRobotEpisodeRecorder(root=tmp_path / "ds", state_layout=build_state_layout(["bracket", "bolt"]), fps=10)
    obs, info = env.reset(seed=0)
    for t in range(5):
        a = env.action_space.sample()
        rec.add_step(obs, a, None, False, info)
        obs, r, term, trunc, info = env.step(a)
        rec.set_last_reward(r, term or trunc)
    rec.end_episode({"success": False, "reward": 0.0, "length": 5})
    rec.close(); env.close()
    ds = load_dataset(rec.root)
    assert ds.num_episodes == 1 and ds.num_frames == 5
    item = ds[0]
    assert "observation.images.overhead" in item and "observation.state" in item and "action" in item
    assert tuple(item["observation.state"].shape) == (44,)


@pytest.mark.slow
def test_training_smoke_run(tmp_path):
    """Runs the real trainer briefly: must produce metrics, status, a checkpoint and exit 0."""
    env = dict(os.environ, MUJOCO_GL="egl")
    overrides = {"total_timesteps": 1200, "n_envs": 2, "n_steps": 64, "batch_size": 64, "n_epochs": 1,
                 "checkpoint_every_steps": 600, "eval_every_steps": 0, "run_name": "pytest_smoke",
                 "log_dir": str(tmp_path / "logs"), "checkpoint_dir": str(tmp_path / "ckpt"),
                 "metrics_file": str(tmp_path / "logs/metrics.jsonl"), "status_file": str(tmp_path / "logs/status.json"),
                 "control_file": str(tmp_path / "logs/control.json"), "dataset": {"record_eval_episodes": False}}
    of = tmp_path / "ov.json"; of.write_text(json.dumps(overrides))
    res = subprocess.run([sys.executable, "-m", "training.train", "--overrides", str(of)], cwd=ROOT, env=env,
                         capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    metrics = [json.loads(l) for l in (tmp_path / "logs/metrics.jsonl").read_text().splitlines()]
    assert metrics and "train/value_loss" in metrics[-1] and "rollout/ep_rew_mean" in metrics[-1]
    status = json.loads((tmp_path / "logs/status.json").read_text())
    assert status["state"] == "completed" and status["timesteps"] >= 1200
    from training.checkpointing import list_checkpoints
    assert len(list_checkpoints(tmp_path / "ckpt")) >= 1
