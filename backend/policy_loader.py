"""Loads a checkpoint into an SB3 PPO policy and exposes a LeRobot-style select_action()."""
from __future__ import annotations
import pickle
import time
from pathlib import Path
from typing import Any
import numpy as np
import torch

from simulation.environments.sorting_env import IMAGE_KEY, STATE_KEY, SB3KeyAlias
from training.checkpointing import load_metadata, latest_checkpoint, best_checkpoint
from simulation.config import resolve_path


class LoadedPolicy:
    def __init__(self, ckpt_path: Path, device: str = "cpu"):
        from stable_baselines3 import PPO
        self.path = Path(ckpt_path)
        self.name = self.path.name
        self.device = device
        self.model = PPO.load(str(self.path / "model.zip"), device=device, print_system_info=False)
        self.metadata = load_metadata(self.path)
        self.obs_rms = None
        self.clip_obs = 10.0
        vn = self.path / "vecnormalize.pkl"
        if vn.exists():
            with open(vn, "rb") as f:
                vec = pickle.load(f)
            # VecNormalize pickles the whole wrapper; take the state RMS only
            rms = getattr(vec, "obs_rms", None)
            if isinstance(rms, dict) and "state" in rms:
                self.obs_rms = rms["state"]
            self.clip_obs = float(getattr(vec, "clip_obs", 10.0))
            self.epsilon = float(getattr(vec, "epsilon", 1e-8))
        self.latency_ms = float("nan")
        self.n_calls = 0
        self.uses_image = "image" in self.model.observation_space.spaces

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        if self.obs_rms is None:
            return state
        return np.clip((state - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon),
                       -self.clip_obs, self.clip_obs).astype(np.float32)

    def select_action(self, obs: dict[str, Any], deterministic: bool = True) -> np.ndarray:
        """LeRobot-shaped observation dict -> action in [-1, 1]^7."""
        sb3_obs = {"state": self._normalize_state(np.asarray(obs[STATE_KEY], dtype=np.float32))}
        if self.uses_image:
            sb3_obs["image"] = np.asarray(obs[IMAGE_KEY], dtype=np.uint8)
        t0 = time.perf_counter()
        action, _ = self.model.predict(sb3_obs, deterministic=deterministic)
        self.latency_ms = (time.perf_counter() - t0) * 1000.0
        self.n_calls += 1
        return np.asarray(action, dtype=np.float32)

    def value(self, obs: dict[str, Any]) -> float | None:
        try:
            sb3_obs = {"state": self._normalize_state(np.asarray(obs[STATE_KEY], dtype=np.float32))}
            if self.uses_image:
                sb3_obs["image"] = np.asarray(obs[IMAGE_KEY], dtype=np.uint8)
            t, _ = self.model.policy.obs_to_tensor(sb3_obs)
            with torch.no_grad():
                return float(self.model.policy.predict_values(t).item())
        except Exception:
            return None

    def info(self) -> dict:
        md = self.metadata
        return {
            "name": md.get("run_name") or "PPO-sorting",
            "checkpoint": self.name,
            "algorithm": md.get("algorithm", "PPO"),
            "timesteps": md.get("timesteps"),
            "episodes": md.get("episodes"),
            "device": self.device,
            "trained_on_device": md.get("device"),
            "eval_success_rate": md.get("eval_success_rate"),
            "eval_mean_reward": md.get("eval_mean_reward"),
            "eval_placement_rate": md.get("eval_placement_rate"),
            "eval_grasp_rate": md.get("eval_grasp_rate"),
            "eval_at_timesteps": md.get("eval_at_timesteps"),
            "observation_type": "RGB (overhead 64x64) + proprioception/state" if self.uses_image else "state only",
            "action_type": "end-effector Cartesian delta (dX dY dZ dRx dRy dRz) + gripper",
            "action_dim": int(self.model.action_space.shape[0]),
            "state_dim": int(self.model.observation_space["state"].shape[0]),
            "image_shape": list(self.model.observation_space["image"].shape) if self.uses_image else None,
            "normalized_state": self.obs_rms is not None,
            "created_iso": md.get("created_iso"),
        }


def resolve_checkpoint(name: str | None, ckpt_dir: str = "checkpoints") -> Path | None:
    """'best' (default) = highest eval success (ties: placement rate, eval reward); 'latest' = most recent."""
    if name in (None, "", "best"):
        return best_checkpoint(ckpt_dir)
    if name == "latest":
        return latest_checkpoint(ckpt_dir)
    p = Path(name)
    if not p.is_absolute():
        cand = resolve_path(name)                      # project-relative path (e.g. checkpoints/archive_x/checkpoint_...)
        p = cand if (cand / "model.zip").exists() else resolve_path(ckpt_dir) / name
    return p if (p / "model.zip").exists() else None
