"""LeRobot-style policy interface over the SB3 PPO checkpoint.

LeRobot policies consume a batch dict keyed `observation.*` and return an `action` tensor
from `select_action(batch)`. This adapter gives the RL policy the same surface so it can be
dropped into LeRobot-style rollout code; the RL training itself is stable-baselines3 PPO
(LeRobot 0.6's training loop is imitation-learning only — see ASSUMPTIONS.md #7).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch

from backend.policy_loader import LoadedPolicy, resolve_checkpoint
from simulation.environments.sorting_env import IMAGE_KEY, STATE_KEY


class SortingRLPolicy:
    name = "ppo_sorting"

    def __init__(self, checkpoint: str | Path = "latest", device: str = "cpu"):
        path = resolve_checkpoint(str(checkpoint))
        if path is None:
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
        self.inner = LoadedPolicy(path, device=device)
        self.config = {"input_features": {STATE_KEY: {"shape": [self.inner.info()["state_dim"]]},
                                          **({IMAGE_KEY: {"shape": self.inner.info()["image_shape"]}} if self.inner.uses_image else {})},
                       "output_features": {"action": {"shape": [self.inner.info()["action_dim"]]}}}

    def reset(self):
        pass

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """batch: {observation.state: (B, D), observation.images.overhead: (B, 3, H, W) float in [0,1] or (B,H,W,3) uint8}"""
        states = batch[STATE_KEY]
        B = states.shape[0]
        actions = []
        for i in range(B):
            obs = {STATE_KEY: states[i].detach().cpu().numpy()}
            if self.inner.uses_image:
                img = batch[IMAGE_KEY][i].detach().cpu()
                if img.dtype != torch.uint8:
                    img = (img.clamp(0, 1) * 255).to(torch.uint8)
                if img.shape[0] == 3:
                    img = img.permute(1, 2, 0)
                obs[IMAGE_KEY] = img.numpy()
            actions.append(self.inner.select_action(obs, deterministic=True))
        return torch.as_tensor(np.stack(actions), dtype=torch.float32)

    def info(self) -> dict:
        return self.inner.info()
