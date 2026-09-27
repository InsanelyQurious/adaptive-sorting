"""Behaviour-cloning bootstrap for the PPO policy (Section 43).

The PPO policy (SB3 MultiInputPolicy: NatureCNN on the overhead image + MLP on the state) is trained
here by supervised regression on recorded demonstrations, then handed to PPO unchanged - a direct
weight transfer, no separate BC network. The value head is regressed on the demonstrations'
discounted returns (in VecNormalize's normalised reward scale) so PPO's first advantages are not garbage.

`DemoBuffer` also serves the interleaved BC auxiliary steps that TrainerControlCallback runs after
every PPO update while `bootstrap.bc_aux_coef` is > 0 (see training/callbacks/trainer_callbacks.py).
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Callable
import numpy as np
import torch
import torch.nn.functional as F

from lerobot_adapter.demo_recorder import list_demos

IMAGE_KEY = "observation.images.overhead"
STATE_KEY = "observation.state"


def select_demo_dirs(demos_dir: Path, demo_types: list[str] | None, successful_only: bool = True) -> list[dict]:
    demos = list_demos(Path(demos_dir))
    out = []
    for d in demos:
        if demo_types and d.get("episode_type") not in demo_types:
            continue
        if successful_only and not d.get("outcome", {}).get("success", False):
            continue
        if d.get("aborted"):
            continue
        out.append(d)
    return out


class DemoBuffer:
    """All demonstration transitions in memory (a few thousand 64x64 frames is small)."""

    def __init__(self, states: np.ndarray, actions: np.ndarray, images: np.ndarray | None, rewards: np.ndarray,
                 episode_ids: np.ndarray, dones: np.ndarray, sources: list[dict]):
        self.states, self.actions, self.images = states, actions, images
        self.rewards, self.episode_ids, self.dones = rewards, episode_ids, dones
        self.sources = sources
        self.n = len(actions)

    @classmethod
    def from_dirs(cls, demos: list[dict], include_image: bool = True) -> "DemoBuffer":
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        S, A, I, R, E, D = [], [], [], [], [], []
        for ei, dm in enumerate(demos):
            root = Path(dm["path"])
            ds = LeRobotDataset(f"volvo-physical-ai/{dm.get('episode_type', 'demo')}", root=str(root))
            for i in range(ds.num_frames):
                item = ds[i]
                S.append(item[STATE_KEY].numpy().astype(np.float32))
                A.append(item["action"].numpy().astype(np.float32))
                R.append(float(item["next.reward"].reshape(-1)[0]))
                D.append(bool(item["next.done"].reshape(-1)[0]))
                E.append(ei)
                if include_image:
                    img = item[IMAGE_KEY]
                    if img.dtype != torch.uint8:
                        img = (img.clamp(0, 1) * 255).round().to(torch.uint8)
                    if img.shape[0] == 3:
                        img = img.permute(1, 2, 0)
                    I.append(np.ascontiguousarray(img.numpy()))
        if not A:
            raise RuntimeError("no demonstration frames found")
        return cls(np.stack(S), np.stack(A), np.stack(I) if include_image else None, np.asarray(R, np.float32),
                   np.asarray(E), np.asarray(D), demos)

    # ------------------------------------------------------------------ returns / normalisation
    def discounted_returns(self, gamma: float, reward_scale: float = 1.0) -> np.ndarray:
        """Return-to-go per frame, episode-wise, with rewards divided by reward_scale."""
        G = np.zeros(self.n, np.float32)
        running = 0.0
        for i in range(self.n - 1, -1, -1):
            if self.dones[i] or (i + 1 < self.n and self.episode_ids[i + 1] != self.episode_ids[i]):
                running = 0.0
            running = self.rewards[i] / reward_scale + gamma * running
            G[i] = running
        return G

    def vecnormalize_returns_accumulator(self, gamma: float) -> np.ndarray:
        """The forward accumulator VecNormalize tracks (ret = ret*gamma + r), for seeding ret_rms."""
        acc = np.zeros(self.n, np.float32)
        running = 0.0
        prev_ep = None
        for i in range(self.n):
            if prev_ep is not None and self.episode_ids[i] != prev_ep:
                running = 0.0
            running = running * gamma + self.rewards[i]
            acc[i] = running
            prev_ep = self.episode_ids[i]
            if self.dones[i]:
                running = 0.0
        return acc

    def seed_vecnormalize(self, venv, gamma: float) -> dict:
        """Initialise VecNormalize's observation (state) and return statistics from the demos so the
        policy sees the same normalisation during BC and during PPO."""
        from stable_baselines3.common.vec_env import VecNormalize
        if not isinstance(venv, VecNormalize):
            return {}
        info = {}
        rms = venv.obs_rms["state"] if isinstance(venv.obs_rms, dict) else venv.obs_rms
        rms.mean[:] = self.states.mean(0)
        rms.var[:] = self.states.var(0) + 1e-8
        rms.count = float(self.n)
        info["state_rms_count"] = self.n
        if venv.norm_reward:
            acc = self.vecnormalize_returns_accumulator(gamma)
            venv.ret_rms.mean = float(acc.mean())
            venv.ret_rms.var = float(acc.var() + 1e-8)
            venv.ret_rms.count = float(self.n)
            info["reward_scale"] = float(np.sqrt(venv.ret_rms.var + venv.epsilon))
        return info

    def normalized_states(self, venv) -> np.ndarray:
        from stable_baselines3.common.vec_env import VecNormalize
        if isinstance(venv, VecNormalize):
            rms = venv.obs_rms["state"] if isinstance(venv.obs_rms, dict) else venv.obs_rms
            return np.clip((self.states - rms.mean) / np.sqrt(rms.var + venv.epsilon), -venv.clip_obs, venv.clip_obs).astype(np.float32)
        return self.states


# ---------------------------------------------------------------------- policy forward (mirrors SB3 ActorCriticPolicy.forward)
def _policy_mean_and_value(policy, obs_t):
    features = policy.extract_features(obs_t)
    if policy.share_features_extractor:
        latent_pi, latent_vf = policy.mlp_extractor(features)
    else:
        pi_f, vf_f = features
        latent_pi = policy.mlp_extractor.forward_actor(pi_f)
        latent_vf = policy.mlp_extractor.forward_critic(vf_f)
    mean = policy.action_net(latent_pi)
    values = policy.value_net(latent_vf).flatten()
    return mean, values


def _batch_obs(buf: DemoBuffer, norm_states: np.ndarray, idx: np.ndarray, use_image: bool) -> dict:
    obs = {"state": norm_states[idx]}
    if use_image and buf.images is not None:
        obs["image"] = buf.images[idx]
    return obs


class BCTrainer:
    """Supervised steps on the PPO policy. Used for the pretraining stage and for aux steps."""

    def __init__(self, model, venv, buf: DemoBuffer, lr: float, vf_coef: float = 0.5, value_regression: bool = True,
                 gamma: float = 0.99, holdout_fraction: float = 0.1, seed: int = 0):
        self.model, self.venv, self.buf = model, venv, buf
        self.policy = model.policy
        self.use_image = "image" in model.observation_space.spaces
        self.vf_coef = float(vf_coef) if value_regression else 0.0
        self.params = [p for n, p in self.policy.named_parameters() if n != "log_std"]
        self.opt = torch.optim.Adam(self.params, lr=float(lr))
        self.base_lr = float(lr)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(buf.n)
        n_hold = int(round(holdout_fraction * buf.n)) if holdout_fraction > 0 else 0
        self.hold_idx, self.train_idx = perm[:n_hold], perm[n_hold:]
        self.rng = rng
        self.refresh_normalization(gamma)

    def refresh_normalization(self, gamma: float):
        """Recompute normalised states / return targets with the venv's CURRENT statistics."""
        from stable_baselines3.common.vec_env import VecNormalize
        self.norm_states = self.buf.normalized_states(self.venv)
        scale = float(np.sqrt(self.venv.ret_rms.var + self.venv.epsilon)) if isinstance(self.venv, VecNormalize) and self.venv.norm_reward else 1.0
        self.returns = self.buf.discounted_returns(gamma, reward_scale=scale)

    def set_lr(self, lr: float):
        for g in self.opt.param_groups:
            g["lr"] = lr

    def step(self, idx: np.ndarray) -> dict:
        obs_t, _ = self.policy.obs_to_tensor(_batch_obs(self.buf, self.norm_states, idx, self.use_image))
        a = torch.as_tensor(self.buf.actions[idx], device=self.policy.device)
        ret = torch.as_tensor(self.returns[idx], device=self.policy.device)
        mean, values = _policy_mean_and_value(self.policy, obs_t)
        loss_a = F.mse_loss(mean, a)
        loss_v = F.mse_loss(values, ret) if self.vf_coef > 0 else torch.zeros((), device=self.policy.device)
        loss = loss_a + self.vf_coef * loss_v
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, 1.0)
        self.opt.step()
        return {"loss": float(loss.item()), "action_mse": float(loss_a.item()), "value_mse": float(loss_v.item())}

    @torch.no_grad()
    def evaluate(self, idx: np.ndarray) -> dict:
        if len(idx) == 0:
            return {}
        out = {"action_mse": 0.0, "gripper_sign_acc": 0.0, "value_mse": 0.0}
        n = 0
        for s in range(0, len(idx), 512):
            b = idx[s:s + 512]
            obs_t, _ = self.policy.obs_to_tensor(_batch_obs(self.buf, self.norm_states, b, self.use_image))
            mean, values = _policy_mean_and_value(self.policy, obs_t)
            a = torch.as_tensor(self.buf.actions[b], device=self.policy.device)
            ret = torch.as_tensor(self.returns[b], device=self.policy.device)
            out["action_mse"] += float(F.mse_loss(mean, a, reduction="sum").item()) / a.shape[1]
            out["gripper_sign_acc"] += float((torch.sign(mean[:, 6]) == torch.sign(a[:, 6])).float().sum().item())
            out["value_mse"] += float(F.mse_loss(values, ret, reduction="sum").item())
            n += len(b)
        return {k: v / n for k, v in out.items()}

    def sample(self, batch_size: int) -> np.ndarray:
        return self.rng.choice(self.train_idx, size=min(batch_size, len(self.train_idx)), replace=False)


def bc_pretrain(model, venv, buf: DemoBuffer, *, epochs: int, batch_size: int, lr: float, gamma: float,
                vf_coef: float = 0.5, value_regression: bool = True, holdout_fraction: float = 0.1, seed: int = 0,
                log: Callable[[str], None] = print, progress: Callable[[dict], None] | None = None) -> tuple[BCTrainer, dict]:
    t0 = time.time()
    norm_info = buf.seed_vecnormalize(venv, gamma)
    bc = BCTrainer(model, venv, buf, lr=lr, vf_coef=vf_coef, value_regression=value_regression, gamma=gamma,
                   holdout_fraction=holdout_fraction, seed=seed)
    before = bc.evaluate(bc.hold_idx if len(bc.hold_idx) else bc.train_idx)
    log(f"BC pretraining: {buf.n} frames from {len(buf.sources)} demos "
        f"({', '.join(sorted(set(d.get('episode_type', '?') for d in buf.sources)))}), train {len(bc.train_idx)} / holdout {len(bc.hold_idx)}, "
        f"epochs={epochs} batch={batch_size} lr={lr}; before: holdout action MSE {before.get('action_mse', float('nan')):.4f}")
    history = []
    for ep in range(int(epochs)):
        perm = bc.rng.permutation(bc.train_idx)
        agg = {"loss": 0.0, "action_mse": 0.0, "value_mse": 0.0}; nb = 0
        for s in range(0, len(perm), batch_size):
            r = bc.step(perm[s:s + batch_size])
            for k in agg: agg[k] += r[k]
            nb += 1
        agg = {k: v / max(nb, 1) for k, v in agg.items()}
        hold = bc.evaluate(bc.hold_idx) if len(bc.hold_idx) else {}
        row = {"epoch": ep + 1, **{f"train_{k}": v for k, v in agg.items()}, **{f"holdout_{k}": v for k, v in hold.items()}}
        history.append(row)
        if progress:
            progress({"bc_epoch": ep + 1, "bc_epochs": int(epochs), **row})
        if (ep + 1) % max(1, epochs // 8) == 0 or ep == epochs - 1:
            log(f"BC epoch {ep + 1}/{epochs}: train action MSE {agg['action_mse']:.4f} value MSE {agg['value_mse']:.3f}"
                + (f" | holdout action MSE {hold['action_mse']:.4f} gripper-sign acc {hold['gripper_sign_acc']:.3f}" if hold else ""))
    after = bc.evaluate(bc.hold_idx if len(bc.hold_idx) else bc.train_idx)
    info = {"enabled": True, "demo_episodes": len(buf.sources), "demo_frames": int(buf.n),
            "demo_types": sorted(set(d.get("episode_type", "?") for d in buf.sources)),
            "demo_type_counts": {t: sum(1 for d in buf.sources if d.get("episode_type") == t) for t in sorted(set(d.get("episode_type", "?") for d in buf.sources))},
            "demo_names": [d["name"] for d in buf.sources],
            "epochs": int(epochs), "batch_size": int(batch_size), "learning_rate": float(lr),
            "holdout_frames": int(len(bc.hold_idx)), "before": before, "after": after,
            "final_train": history[-1] if history else None, "seconds": round(time.time() - t0, 1), **norm_info}
    log(f"BC done in {info['seconds']} s: holdout action MSE {before.get('action_mse', float('nan')):.4f} -> {after.get('action_mse', float('nan')):.4f}, "
        f"gripper-sign acc {before.get('gripper_sign_acc', float('nan')):.3f} -> {after.get('gripper_sign_acc', float('nan')):.3f}")
    return bc, info
