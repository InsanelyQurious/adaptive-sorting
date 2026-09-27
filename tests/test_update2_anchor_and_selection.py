"""Incremental Update 2: BC anchor inside PPO, LR gate, best-by-eval checkpoint selection."""
import json
from pathlib import Path
import numpy as np
import pytest


def _fake_ckpt(root: Path, steps: int, succ, place, reward, eval_at, reason="periodic"):
    d = root / f"checkpoint_{steps:09d}"; d.mkdir(parents=True)
    (d / "model.zip").write_bytes(b"x")
    (d / "metadata.json").write_text(json.dumps({"timesteps": steps, "eval_success_rate": succ, "eval_placement_rate": place,
                                                 "eval_mean_reward": reward, "eval_at_timesteps": eval_at, "eval_episodes": 20, "reason": reason, "run_name": "t"}))


def test_best_checkpoint_prefers_eval_success_then_placement_then_reward(tmp_path):
    from training.checkpointing import best_checkpoint, latest_checkpoint
    _fake_ckpt(tmp_path, 100, 0.0, 0.1, -5.0, 100)
    _fake_ckpt(tmp_path, 200, 0.1, 0.3, 4.0, 200)      # best: highest success
    _fake_ckpt(tmp_path, 250, 0.1, 0.2, 9.0, 250)      # same success, lower placement -> loses
    _fake_ckpt(tmp_path, 300, 0.0, 0.5, 20.0, 300)     # most recent, worse eval
    (tmp_path / "latest.json").write_text(json.dumps({"name": "checkpoint_000000300"}))
    assert latest_checkpoint(tmp_path).name == "checkpoint_000000300"
    assert best_checkpoint(tmp_path).name == "checkpoint_000000200"


def test_best_checkpoint_ignores_evals_from_other_steps(tmp_path):
    from training.checkpointing import best_checkpoint
    _fake_ckpt(tmp_path, 100032, 0.0, 0.0, -9.0, 100032)
    _fake_ckpt(tmp_path, 275088, 0.5, 0.5, 30.0, 250020)   # eval belongs to step 250020, not these weights -> not eligible
    assert best_checkpoint(tmp_path).name == "checkpoint_000100032"


def test_resolve_checkpoint_best_default(tmp_path):
    from backend.policy_loader import resolve_checkpoint
    _fake_ckpt(tmp_path, 100, 0.2, 0.2, 1.0, 100)
    _fake_ckpt(tmp_path, 200, 0.0, 0.0, 1.0, 200)
    (tmp_path / "latest.json").write_text(json.dumps({"name": "checkpoint_000000200"}))
    assert resolve_checkpoint("best", str(tmp_path)).name == "checkpoint_000000100"
    assert resolve_checkpoint(None, str(tmp_path)).name == "checkpoint_000000100"
    assert resolve_checkpoint("latest", str(tmp_path)).name == "checkpoint_000000200"


def test_ppo_with_bc_anchor_coefficient_and_train_step():
    """The BC term decays to a floor (never 0) and a PPO update with the anchor runs and logs drift."""
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.logger import Logger
    from simulation.environments.sorting_env import make_env
    from training.policies.ppo_bc import PPOWithBC
    venv = VecNormalize(DummyVecEnv([lambda: make_env(include_image=False)]), norm_obs_keys=["state"])
    model = PPOWithBC("MultiInputPolicy", venv, n_steps=16, batch_size=16, n_epochs=1, verbose=0, device="cpu", target_kl=0.05,
                      bc_loss_coef=1.0, bc_loss_coef_floor=0.25, bc_loss_decay_timesteps=1000)
    assert model.bc_coef() == 0.0            # no sampler yet -> anchor inactive
    rng = np.random.default_rng(0)
    states = rng.normal(size=(64, venv.observation_space["state"].shape[0])).astype(np.float32)
    acts = np.clip(rng.normal(size=(64, 7)), -1, 1).astype(np.float32)
    model.set_demo_sampler(lambda n: ({"state": states[:n]}, acts[:n]))
    assert model.bc_coef() == 1.0
    model.num_timesteps = 500; assert abs(model.bc_coef() - 0.625) < 1e-6
    model.num_timesteps = 5000; assert model.bc_coef() == 0.25   # floor, never 0
    model.num_timesteps = 0
    model.set_logger(Logger(folder=None, output_formats=[]))
    model.learn(16)
    assert model.last_bc is not None and model.last_bc["demo_action_mse"] > 0
    assert "bc/demo_action_mse" in model.logger.name_to_value
    venv.close()


def test_lr_gate_ramps_up_and_backs_off():
    from training.callbacks.trainer_callbacks import TrainerControlCallback
    import types
    cb = TrainerControlCallback.__new__(TrainerControlCallback)
    cb.lr_gate_enabled = True; cb.base_lr = 3e-4; cb.lr_scale = 0.1; cb.lr_min, cb.lr_max = 0.05, 1.0
    cb.lr_up, cb.lr_down, cb.lr_tol = 2.0, 0.5, 0.05; cb.prev_eval_score = None; cb.lr_events = []
    cb.model = None; cb.num_timesteps = 0; cb.log = lambda m: None
    cb._apply_lr_scale = lambda: None
    cb._gate_lr_on_eval({"success_rate": 0.0, "placement_rate": 0.1, "grasp_rate": 0.4})   # first eval: ramp
    assert cb.lr_scale == 0.2
    cb._gate_lr_on_eval({"success_rate": 0.1, "placement_rate": 0.2, "grasp_rate": 0.5})   # improved: ramp
    assert cb.lr_scale == 0.4
    cb._gate_lr_on_eval({"success_rate": 0.0, "placement_rate": 0.0, "grasp_rate": 0.0})   # regressed: back off
    assert cb.lr_scale == 0.2
