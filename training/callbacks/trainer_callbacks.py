"""SB3 callbacks: file-based control (pause/resume/stop/save/evaluate), status + metrics
persistence, periodic checkpoints and periodic evaluation. All numbers written are taken
from the running training process."""
from __future__ import annotations
import json
import os
import signal
import sys
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import KVWriter, Logger, HumanOutputFormat
from stable_baselines3.common.vec_env import VecNormalize, sync_envs_normalization

from training.checkpointing import save_checkpoint


class JSONLWriter(KVWriter):
    """Appends every SB3 logger dump as one JSON line to logs/metrics.jsonl."""

    def __init__(self, path: Path, extra_provider=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.extra_provider = extra_provider

    def write(self, key_values: dict[str, Any], key_excluded: dict[str, tuple[str, ...]], step: int = 0) -> None:
        row: dict[str, Any] = {"timesteps": int(step), "wall_time": time.time()}
        for k, v in key_values.items():
            if isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(v):
                row[k] = float(v)
            elif isinstance(v, str):
                row[k] = v
        if self.extra_provider is not None:
            try:
                row.update(self.extra_provider())
            except Exception:
                pass
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")

    def close(self) -> None:
        pass


class TrainerControlCallback(BaseCallback):
    EPISODE_KEYS = ("is_success", "placed_count", "grasp_achieved", "lift_achieved", "collisions", "dropped",
                    "wrong_bin", "unstable")

    def __init__(self, *, configs: dict, status_file: Path, control_file: Path, log_file: Path,
                 ckpt_dir: Path, checkpoint_every: int, eval_every: int, eval_episodes: int,
                 eval_env_factory, device: str, run_name: str, total_timesteps: int,
                 episode_cap: int | None = None, dataset_recorder_factory=None, verbose: int = 1,
                 bc_trainer=None, bootstrap_info: dict | None = None, lr_gate: dict | None = None, base_lr: float | None = None):
        super().__init__(verbose)
        # learning-rate warm-up + evaluation-gated ramp (fix #2)
        g = dict(lr_gate or {})
        self.lr_gate_enabled = bool(g.get("enabled", False)) and base_lr is not None
        self.base_lr = float(base_lr) if base_lr is not None else None
        self.lr_scale = float(g.get("initial_scale", 1.0)) if self.lr_gate_enabled else 1.0
        self.lr_min, self.lr_max = float(g.get("min_scale", 0.05)), float(g.get("max_scale", 1.0))
        self.lr_up, self.lr_down = float(g.get("up_factor", 2.0)), float(g.get("down_factor", 0.5))
        self.lr_tol = float(g.get("regress_tolerance", 0.05))
        self.prev_eval_score: float | None = None
        self.lr_events: list[dict] = []
        # imitation bootstrap: interleaved BC auxiliary steps (see training/bc_pretrain.py)
        self.bc_trainer = bc_trainer
        self.bootstrap_info = bootstrap_info
        b = bootstrap_info or {}
        self.bc_aux_coef0 = float(b.get("bc_aux_coef", 0.0)) if bc_trainer is not None else 0.0
        self.bc_aux_timesteps = int(b.get("bc_aux_timesteps", 0))
        self.bc_aux_steps = int(b.get("bc_aux_steps_per_iter", 0))
        self.bc_aux_batch = int(b.get("bc_batch_size", 256))
        self.last_bc_aux: dict | None = None
        self.configs = configs
        self.status_file = Path(status_file)
        self.control_file = Path(control_file)
        self.log_file = Path(log_file)
        self.ckpt_dir = Path(ckpt_dir)
        self.checkpoint_every = int(checkpoint_every)
        self.eval_every = int(eval_every)
        self.eval_episodes = int(eval_episodes)
        self.eval_env_factory = eval_env_factory
        self.eval_env = None
        self.device = device
        self.run_name = run_name
        self.total_timesteps = int(total_timesteps)
        self.episode_cap = episode_cap
        self.dataset_recorder_factory = dataset_recorder_factory
        self.dataset_recorder = None
        self.state = "starting"
        self.paused = False
        self.stop_requested = False
        self.pending_save = False
        self.pending_eval: int | None = None
        self.start_time = time.time()
        self.episodes = 0
        self.recent: deque = deque(maxlen=100)     # recent episode dicts
        self.last_ckpt_at = 0
        self.last_eval_at = 0
        self.last_checkpoint: str | None = None
        self.last_eval: dict | None = None
        self.last_status_write = 0.0
        self.last_control_mtime = 0.0
        self.error: str | None = None
        self.iteration = 0
        self._fps_window: deque = deque(maxlen=20)
        self._last_fps_t = time.time()
        self._last_fps_steps = 0
        signal.signal(signal.SIGTERM, self._on_signal)
        signal.signal(signal.SIGINT, self._on_signal)

    # ------------------------------------------------------------------ util
    def log(self, msg: str):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
        print(line, flush=True)
        with open(self.log_file, "a") as f:
            f.write(line + "\n")

    def _on_signal(self, signum, frame):
        self.log(f"signal {signum} received -> graceful stop (checkpoint will be saved)")
        self.stop_requested = True
        self.paused = False

    def _fps(self) -> float | None:
        now = time.time()
        if now - self._last_fps_t >= 2.0:
            fps = (self.num_timesteps - self._last_fps_steps) / (now - self._last_fps_t)
            self._fps_window.append(fps)
            self._last_fps_t, self._last_fps_steps = now, self.num_timesteps
        return float(np.mean(self._fps_window)) if self._fps_window else None

    def recent_stats(self) -> dict:
        n = len(self.recent)
        if n == 0:
            return {"episodes": self.episodes}
        r = list(self.recent)
        return {
            "episodes": self.episodes,
            "recent_success_rate": float(np.mean([e["is_success"] for e in r])),
            "recent_grasp_rate": float(np.mean([e["grasp_achieved"] for e in r])),
            "recent_lift_rate": float(np.mean([e["lift_achieved"] for e in r])),
            "recent_placement_rate": float(np.mean([e["placed_count"] > 0 for e in r])),
            "recent_mean_placed": float(np.mean([e["placed_count"] for e in r])),
            "recent_drop_rate": float(np.mean([e["dropped"] for e in r])),
            "recent_wrong_bin_rate": float(np.mean([e["wrong_bin"] for e in r])),
            "recent_unstable_rate": float(np.mean([e["unstable"] for e in r])),
            "recent_mean_collisions": float(np.mean([e["collisions"] for e in r])),
            "recent_mean_reward": float(np.mean([e["r"] for e in r])),
            "recent_mean_length": float(np.mean([e["l"] for e in r])),
        }

    def status_dict(self) -> dict:
        elapsed = time.time() - self.start_time
        d = {
            "state": self.state, "pid": os.getpid(), "run_name": self.run_name,
            "algorithm": self.configs["training"].get("algorithm", "PPO"),
            "policy": "MultiInputPolicy (NatureCNN image + MLP state)",
            "env_id": self.configs["environment"].get("env_id"),
            "device": self.device,
            "timesteps": int(self.num_timesteps), "total_timesteps": self.total_timesteps,
            "progress": min(1.0, self.num_timesteps / max(self.total_timesteps, 1)),
            "iteration": self.iteration, "elapsed_s": round(elapsed, 1),
            "fps": self._fps(), "started": self.start_time, "updated": time.time(),
            "latest_checkpoint": self.last_checkpoint, "last_eval": self.last_eval,
            "n_envs": self.configs["training"].get("n_envs"),
            "learning_rate": (self.base_lr * self.lr_scale) if self.lr_gate_enabled else (float(self.model.learning_rate) if self.model is not None and not callable(self.model.learning_rate) else None),
            "error": self.error,
            "training_config": self.configs["training"],
            "bootstrap": ({**self.bootstrap_info, "aux_coef_now": self._bc_aux_coef(), "last_aux": self.last_bc_aux,
                           "bc_anchor": getattr(self.model, "last_bc", None), "bc_coef_now": (self.model.bc_coef() if hasattr(self.model, "bc_coef") else None)}
                          if self.bootstrap_info else {"enabled": False}),
            "lr_gate": ({"enabled": True, "scale": self.lr_scale, "lr": self.base_lr * self.lr_scale, "events": self.lr_events[-8:]}
                        if self.lr_gate_enabled else {"enabled": False}),
        }
        d.update(self.recent_stats())
        return d

    def _bc_aux_coef(self) -> float:
        if self.bc_trainer is None or self.bc_aux_coef0 <= 0 or self.bc_aux_steps <= 0:
            return 0.0
        if self.bc_aux_timesteps <= 0:
            return self.bc_aux_coef0
        return float(self.bc_aux_coef0 * max(0.0, 1.0 - self.num_timesteps / self.bc_aux_timesteps))

    def _bc_aux_update(self):
        """A few supervised steps on the demonstration buffer after each PPO update (lr scaled by a
        linearly decaying coefficient). Runs BEFORE the next rollout so PPO's importance ratios are unaffected."""
        coef = self._bc_aux_coef()
        if coef <= 0:
            return
        bc = self.bc_trainer
        try:
            bc.refresh_normalization(float(self.configs["training"]["gamma"]))
            bc.set_lr(bc.base_lr * coef)
            agg = {"loss": 0.0, "action_mse": 0.0, "value_mse": 0.0}
            for _ in range(self.bc_aux_steps):
                r = bc.step(bc.sample(self.bc_aux_batch))
                for k in agg:
                    agg[k] += r[k] / self.bc_aux_steps
            self.last_bc_aux = {"coef": coef, **agg, "timesteps": int(self.num_timesteps)}
            self.logger.record("bc/aux_coef", coef)
            self.logger.record("bc/aux_action_mse", agg["action_mse"])
            self.logger.record("bc/aux_value_mse", agg["value_mse"])
        except Exception as e:
            self.log(f"BC aux step failed (disabled): {e}")
            self.bc_trainer = None

    def write_status(self, force: bool = False):
        if not force and time.time() - self.last_status_write < 2.0:
            return
        tmp = self.status_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.status_dict(), default=str))
        os.replace(tmp, self.status_file)
        self.last_status_write = time.time()

    def _poll_control(self):
        if not self.control_file.exists():
            return
        try:
            mtime = self.control_file.stat().st_mtime
            if mtime <= self.last_control_mtime:
                return
            self.last_control_mtime = mtime
            cmd = json.loads(self.control_file.read_text())
        except Exception:
            return
        c = cmd.get("command")
        self.log(f"control command: {c}")
        if c == "pause":
            self.paused = True
        elif c == "resume":
            self.paused = False
        elif c == "stop":
            self.stop_requested = True
            self.paused = False
        elif c == "save":
            self.pending_save = True
        elif c == "evaluate":
            self.pending_eval = int(cmd.get("episodes", self.eval_episodes))

    # ------------------------------------------------------------------ SB3 hooks
    @staticmethod
    def eval_score(res: dict) -> float:
        """Gate metric: success dominates, partial progress breaks ties (success alone is ~0 early on)."""
        return float(res.get("success_rate", 0.0)) + 0.5 * float(res.get("placement_rate", 0.0)) + 0.25 * float(res.get("grasp_rate", 0.0))

    def _apply_lr_scale(self):
        if not self.lr_gate_enabled or self.model is None:
            return
        lr = self.base_lr * self.lr_scale
        self.model.lr_schedule = (lambda _progress, _lr=lr: _lr)
        self.model.learning_rate = lr
        for grp in self.model.policy.optimizer.param_groups:
            grp["lr"] = lr

    def _gate_lr_on_eval(self, res: dict):
        if not self.lr_gate_enabled:
            return
        score = self.eval_score(res)
        old = self.lr_scale
        if self.prev_eval_score is None or score >= self.prev_eval_score - self.lr_tol:
            self.lr_scale = min(self.lr_max, self.lr_scale * self.lr_up)      # holding steady / improving -> ramp up
            why = "eval held/improved"
        else:
            self.lr_scale = max(self.lr_min, self.lr_scale * self.lr_down)    # regressed -> back off
            why = "eval regressed"
        self.prev_eval_score = max(score, self.prev_eval_score or 0.0) if score >= (self.prev_eval_score or 0.0) - self.lr_tol else self.prev_eval_score
        self._apply_lr_scale()
        self.lr_events.append({"timesteps": int(self.num_timesteps), "score": round(score, 3), "scale": self.lr_scale, "why": why})
        self.log(f"lr gate: eval score {score:.3f} ({why}) -> lr scale {old:.3f} -> {self.lr_scale:.3f} (lr {self.base_lr * self.lr_scale:.2e})")

    def _on_training_start(self) -> None:
        self.state = "running"
        self._apply_lr_scale()
        if self.lr_gate_enabled:
            self.log(f"lr gate: warm-up scale {self.lr_scale} (lr {self.base_lr * self.lr_scale:.2e}); ramps x{self.lr_up} per non-regressing eval, x{self.lr_down} on regression")
        self.log(f"training started: run={self.run_name} device={self.device} total_timesteps={self.total_timesteps} "
                 f"n_envs={self.configs['training'].get('n_envs')} start_timesteps={self.num_timesteps}")
        if self.dataset_recorder_factory is not None:
            try:
                self.dataset_recorder = self.dataset_recorder_factory()
            except Exception as e:
                self.log(f"dataset recorder disabled: {e}")
        if self.bootstrap_info and self.num_timesteps == 0 and self.eval_every > 0:
            # Evaluate the BC initialisation itself and save it WITH that evaluation, so the 0-step checkpoint
            # competes for "best" like every other one (if PPO never beats it, best stays the BC init) and the
            # LR gate's first comparison is against the BC-init score.
            self.evaluate(self.eval_episodes)
            self.save("bc_init")
        self.write_status(force=True)

    def _on_rollout_start(self) -> None:
        self.iteration += 1
        if self.iteration > 1:      # after the first PPO update (the pretraining stage precedes iteration 1)
            self._bc_aux_update()

    def _on_step(self) -> bool:
        for info, done in zip(self.locals.get("infos", []), self.locals.get("dones", [])):
            if done and "episode" in info:
                ep = {"r": float(info["episode"]["r"]), "l": int(info["episode"]["l"])}
                for k in self.EPISODE_KEYS:
                    v = info.get(k, 0)
                    ep[k] = float(v) if isinstance(v, (bool, int, float, np.bool_, np.integer, np.floating)) else 0.0
                self.recent.append(ep)
                self.episodes += 1
        self._poll_control()
        self.write_status()
        if self.paused:
            self.state = "paused"
            self.write_status(force=True)
            self.log("paused")
            while self.paused and not self.stop_requested:
                time.sleep(0.5)
                self._poll_control()
                self.write_status()
            self.state = "running"
            self.log("resumed")
        if self.pending_save:
            self.pending_save = False
            self.save("manual")
        if self.pending_eval is not None:
            n, self.pending_eval = self.pending_eval, None
            self.evaluate(n)
        if self.eval_every > 0 and self.num_timesteps - self.last_eval_at >= self.eval_every:
            self.evaluate(self.eval_episodes)
            self.save("eval")            # checkpoint saved at the eval step -> its metadata IS its evaluation
        elif self.num_timesteps - self.last_ckpt_at >= self.checkpoint_every:
            self.save("periodic")
        if self.episode_cap is not None and self.episodes >= self.episode_cap:
            self.log(f"episode cap {self.episode_cap} reached -> stopping")
            self.stop_requested = True
        if self.stop_requested:
            self.state = "stopping"
            self.write_status(force=True)
            return False
        return True

    def _on_rollout_end(self) -> None:
        st = self.recent_stats()
        for k, v in st.items():
            if k != "episodes":
                self.logger.record(f"rollout/{k}", v)
        self.logger.record("rollout/episodes_total", self.episodes)
        self.logger.record("time/elapsed_s", time.time() - self.start_time)
        fps = self._fps()
        if fps is not None:
            self.logger.record("time/env_fps", fps)
        if self.last_eval:
            for k in ("success_rate", "mean_reward", "mean_length", "grasp_rate", "placement_rate"):
                if self.last_eval.get(k) is not None:
                    self.logger.record(f"eval/{k}", self.last_eval[k])
        self.logger.record("train/policy_log_std_mean", float(self.model.policy.log_std.mean().item()))

    def _on_training_end(self) -> None:
        if self.dataset_recorder is not None:
            try:
                self.dataset_recorder.close()
            except Exception as e:
                self.log(f"dataset recorder close failed: {e}")

    # ------------------------------------------------------------------ actions
    def save(self, reason: str) -> Path | None:
        prev = self.state
        self.state = "saving"
        try:
            vec = self.model.get_vec_normalize_env()
            meta = {
                "run_name": self.run_name, "algorithm": self.configs["training"].get("algorithm", "PPO"),
                "episodes": self.episodes, "device": self.device, "reason": reason,
                "eval_success_rate": (self.last_eval or {}).get("success_rate"),
                "eval_mean_reward": (self.last_eval or {}).get("mean_reward"),
                "eval_placement_rate": (self.last_eval or {}).get("placement_rate"),
                "eval_grasp_rate": (self.last_eval or {}).get("grasp_rate"),
                "eval_episodes": (self.last_eval or {}).get("episodes"),
                "eval_at_timesteps": (self.last_eval or {}).get("timesteps"),
                "lr_scale": self.lr_scale if self.lr_gate_enabled else None,
                "state_layout": self.configs.get("state_layout"),
                "observation_keys": ["observation.images.overhead", "observation.state"],
                "action_names": ["dX", "dY", "dZ", "dRx", "dRy", "dRz", "gripper"],
                "elapsed_s": round(time.time() - self.start_time, 1),
                "bootstrap": self.bootstrap_info or {"enabled": False},
            }
            meta.update({k: v for k, v in self.recent_stats().items() if k != "episodes"})
            path = save_checkpoint(self.model, vec, self.ckpt_dir, self.num_timesteps, self.configs, meta)
            self.last_checkpoint = path.name
            self.last_ckpt_at = self.num_timesteps
            self.log(f"checkpoint saved ({reason}): {path.name}")
            return path
        except Exception as e:
            self.error = f"checkpoint save failed: {e}"
            self.log(self.error + "\n" + traceback.format_exc())
            return None
        finally:
            self.state = prev if prev != "saving" else "running"
            self.write_status(force=True)

    def evaluate(self, n_episodes: int):
        prev = self.state
        self.state = "evaluating"
        self.write_status(force=True)
        t0 = time.time()
        try:
            if self.eval_env is None:
                self.eval_env = self.eval_env_factory()
            train_vec = self.model.get_vec_normalize_env()
            if train_vec is not None and isinstance(self.eval_env, VecNormalize):
                sync_envs_normalization(train_vec, self.eval_env)
            results = run_evaluation(self.model, self.eval_env, n_episodes, recorder=self.dataset_recorder,
                                     timesteps=self.num_timesteps)
            results["timesteps"] = int(self.num_timesteps)
            results["duration_s"] = round(time.time() - t0, 1)
            self.last_eval = results
            self.last_eval_at = self.num_timesteps
            self.log(f"eval @ {self.num_timesteps}: success {results['success_rate']:.2f} "
                     f"grasp {results['grasp_rate']:.2f} place {results['placement_rate']:.2f} "
                     f"reward {results['mean_reward']:.2f} len {results['mean_length']:.1f} ({n_episodes} eps)")
            for k in ("success_rate", "mean_reward", "mean_length", "grasp_rate", "placement_rate"):
                self.logger.record(f"eval/{k}", results[k])
            self.logger.record("eval/score", self.eval_score(results))
            self._gate_lr_on_eval(results)
        except Exception as e:
            self.error = f"evaluation failed: {e}"
            self.log(self.error + "\n" + traceback.format_exc())
        finally:
            self.state = prev if prev != "evaluating" else "running"
            self.write_status(force=True)


def run_evaluation(model, eval_env, n_episodes: int, recorder=None, timesteps: int | None = None) -> dict:
    """Deterministic rollouts on a (VecNormalize-wrapped, training=False) single env.
    Returns aggregate metrics computed from the actual episodes."""
    episodes = []
    for ep_i in range(n_episodes):
        obs = eval_env.reset()
        done = False
        ep_r = 0.0
        info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            if recorder is not None:
                try:
                    raw = eval_env.get_original_obs() if isinstance(eval_env, VecNormalize) else obs
                    recorder.add_step({"state": raw["state"][0], "image": raw["image"][0] if "image" in raw else None},
                                      action[0], None, False, {})
                except Exception:
                    recorder = None
            obs, rewards, dones, infos = eval_env.step(action)
            r = float(eval_env.get_original_reward()[0]) if isinstance(eval_env, VecNormalize) else float(rewards[0])
            ep_r += r
            if recorder is not None:
                try:
                    recorder.set_last_reward(r, bool(dones[0]))
                except Exception:
                    pass
            done = bool(dones[0])
            info = infos[0]
        ep = {"success": bool(info.get("is_success")), "reward": ep_r, "length": int(info.get("step", 0)),
              "grasp": bool(info.get("grasp_achieved")), "placed": int(info.get("placed_count", 0)),
              "collisions": int(info.get("collisions", 0)), "dropped": bool(info.get("dropped")),
              "wrong_bin": bool(info.get("wrong_bin")), "seed": info.get("seed"),
              "object_configuration": info.get("episode_config")}
        episodes.append(ep)
        if recorder is not None:
            try:
                recorder.end_episode({"success": ep["success"], "reward": ep_r, "length": ep["length"],
                                      "timesteps": timesteps})
            except Exception:
                recorder = None
    n = len(episodes)
    return {
        "episodes": n,
        "success_rate": sum(e["success"] for e in episodes) / n,
        "grasp_rate": sum(e["grasp"] for e in episodes) / n,
        "placement_rate": sum(e["placed"] > 0 for e in episodes) / n,
        "mean_reward": sum(e["reward"] for e in episodes) / n,
        "mean_length": sum(e["length"] for e in episodes) / n,
        "mean_collisions": sum(e["collisions"] for e in episodes) / n,
        "drops": sum(e["dropped"] for e in episodes),
        "per_episode": episodes,
    }
