"""PPO with an in-loss behaviour-cloning anchor (Incremental Update 2, fix #1).

`PPOWithBC.train()` is SB3 2.9's `PPO.train()` plus one term per minibatch:

    loss = PPO_clip + ent_coef * entropy + vf_coef * value_loss + bc_coef(t) * BC(demo minibatch)

where BC is either the MSE between the policy mean and the demonstrated action (default) or the negative
log-likelihood of the demonstrated action under the current Gaussian, computed on a fresh minibatch of
demonstration frames every gradient step. bc_coef(t) decays linearly from `bc_loss_coef` to
`bc_loss_coef_floor` over `bc_loss_decay_timesteps` and never goes below the floor, so the policy stays
anchored to the demonstrations throughout fine-tuning while PPO improves on them.

The demonstration observations are normalised with the SAME VecNormalize statistics as the rollout
observations (the buffer is asked for normalised states each update), so both losses see one input space.
Also logs `bc/loss`, `bc/coef`, `bc/demo_action_mse` (drift from the demonstrations, the quantity that
grew monotonically in the regressing run) and honours `target_kl` early stopping like stock PPO.
"""
from __future__ import annotations
import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance


class PPOWithBC(PPO):
    def __init__(self, *args, bc_loss_coef: float = 1.0, bc_loss_coef_floor: float = 0.2, bc_loss_decay_timesteps: int = 1_000_000,
                 bc_loss_type: str = "mse", bc_batch_size: int = 256, **kwargs):
        super().__init__(*args, **kwargs)
        self.bc_loss_coef0 = float(bc_loss_coef)
        self.bc_loss_coef_floor = float(bc_loss_coef_floor)
        self.bc_loss_decay_timesteps = int(bc_loss_decay_timesteps)
        self.bc_loss_type = str(bc_loss_type)
        self.bc_batch_size = int(bc_batch_size)
        self.demo_sampler = None          # callable(batch_size) -> (obs_dict_normalised, actions) as numpy
        self.last_bc: dict | None = None

    # SB3 pickles __init__ kwargs from `_get_constructor_parameters`; the extra ones are safe to keep
    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["demo_sampler", "last_bc"]

    def set_demo_sampler(self, sampler):
        self.demo_sampler = sampler

    def bc_coef(self) -> float:
        if self.demo_sampler is None or self.bc_loss_coef0 <= 0:
            return 0.0
        if self.bc_loss_decay_timesteps <= 0:
            return self.bc_loss_coef0
        frac = min(1.0, self.num_timesteps / self.bc_loss_decay_timesteps)
        return float(self.bc_loss_coef0 + (self.bc_loss_coef_floor - self.bc_loss_coef0) * frac)

    def _bc_loss(self) -> tuple[th.Tensor, float]:
        obs_np, act_np = self.demo_sampler(self.bc_batch_size)
        obs_t, _ = self.policy.obs_to_tensor(obs_np)
        actions = th.as_tensor(act_np, device=self.device)
        dist = self.policy.get_distribution(obs_t)
        mean = dist.distribution.mean
        mse = F.mse_loss(mean, actions)
        if self.bc_loss_type == "nll":
            loss = -dist.log_prob(actions).mean()
        else:
            loss = mse
        return loss, float(mse.item())

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        clip_range_vf = self.clip_range_vf(self._current_progress_remaining) if self.clip_range_vf is not None else None  # type: ignore[operator]
        coef = self.bc_coef()
        entropy_losses, pg_losses, value_losses, clip_fractions, bc_losses, bc_mses = [], [], [], [], [], []
        continue_training = True
        approx_kl_divs = []
        loss = th.zeros(())
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()
                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(policy_loss.item())
                clip_fractions.append(th.mean((th.abs(ratio - 1) > clip_range).float()).item())
                if clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(values - rollout_data.old_values, -clip_range_vf, clip_range_vf)
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())
                entropy_loss = -th.mean(-log_prob) if entropy is None else -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                if coef > 0:
                    bc_loss, bc_mse = self._bc_loss()
                    loss = loss + coef * bc_loss
                    bc_losses.append(float(bc_loss.item())); bc_mses.append(bc_mse)
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)
                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    break
                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
            self._n_updates += 1
            if not continue_training:
                break
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs) if approx_kl_divs else 0.0)
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", float(loss.item()))
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/kl_early_stop", float(not continue_training))
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        self.logger.record("bc/coef", coef)
        if bc_losses:
            self.last_bc = {"coef": coef, "loss": float(np.mean(bc_losses)), "demo_action_mse": float(np.mean(bc_mses)), "timesteps": int(self.num_timesteps)}
            self.logger.record("bc/loss", self.last_bc["loss"])
            self.logger.record("bc/demo_action_mse", self.last_bc["demo_action_mse"])
