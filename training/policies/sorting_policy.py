"""Policy architecture for PPO (stable-baselines3 MultiInputPolicy).

  image (3x64x64 uint8) --> NatureCNN (3 conv + linear, features_dim) --+
                                                                        +--> concat --> MLP pi / vf
  state (44-d, VecNormalize'd) --> Flatten -----------------------------+

Why PPO (see README): on-policy, robust to reward scale with normalization, trivially
parallel across CPU processes (our bottleneck is env throughput, not sample-efficiency
per se), and stable with the mixed image+state observation and clipped continuous actions.
"""
from __future__ import annotations
import torch.nn as nn


def build_policy_kwargs(train_cfg: dict) -> dict:
    return dict(
        features_extractor_kwargs=dict(cnn_output_dim=int(train_cfg.get("features_dim", 128)),
                                       normalized_image=False),
        net_arch=dict(pi=list(train_cfg.get("net_arch", [256, 256])), vf=list(train_cfg.get("net_arch", [256, 256]))),
        activation_fn=nn.Tanh,
        ortho_init=True,
        log_std_init=float(train_cfg.get("log_std_init", -0.5)),
    )
