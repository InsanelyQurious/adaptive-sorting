"""PPO training process (run as `python -m training.train`).

Independent of the backend and the browser: it writes
  logs/training.log, logs/metrics.jsonl, logs/training_status.json
and periodic checkpoints under checkpoints/. Control via logs/training_control.json.
Usage:
  python -m training.train                       # config from configs/training.yaml
  python -m training.train --overrides o.json    # UI-provided overrides
  python -m training.train --resume latest       # resume from the latest checkpoint
  python -m training.train --smoke               # short smoke-test run
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from simulation.config import load_configs, deep_update, resolve_path
from simulation.environments.sorting_env import make_env, build_state_layout
from backend.device import detect_device, device_label
from training.checkpointing import latest_checkpoint, list_checkpoints
from training.policies.sorting_policy import build_policy_kwargs
from training.callbacks.trainer_callbacks import TrainerControlCallback, JSONLWriter

ENV_LEVEL_OVERRIDES = ("randomization_level",)


def build_vec_env(configs: dict, n_envs: int, seed: int, env_overrides: dict, include_image: bool):
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor, VecNormalize

    def thunk(rank: int):
        def _init():
            env = make_env(sim_cfg=configs["simulation"], env_cfg=configs["environment"], overrides=env_overrides,
                           sb3=True, include_image=include_image, training=True)
            env.reset(seed=seed + rank)
            env.action_space.seed(seed + rank)
            return env
        return _init

    if n_envs == 1:
        venv = DummyVecEnv([thunk(0)])
    else:
        venv = SubprocVecEnv([thunk(i) for i in range(n_envs)], start_method="forkserver")
    venv = VecMonitor(venv, info_keywords=TrainerControlCallback.EPISODE_KEYS)
    if configs["training"].get("normalize_state", True):
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, norm_obs_keys=["state"], clip_obs=10.0,
                            gamma=float(configs["training"]["gamma"]))
    return venv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overrides", type=str, default=None, help="JSON file with training overrides")
    ap.add_argument("--resume", type=str, default=None, help="checkpoint name or 'latest'")
    ap.add_argument("--smoke", action="store_true", help="short smoke-test configuration")
    ap.add_argument("--set", action="append", default=[], help="key=value training override (repeatable)")
    args = ap.parse_args()

    configs = load_configs()
    tcfg = configs["training"]
    overrides: dict = {}
    if args.overrides:
        overrides.update(json.loads(Path(args.overrides).read_text()))
    for kv in args.set:
        k, v = kv.split("=", 1)
        try:
            overrides[k] = json.loads(v)
        except json.JSONDecodeError:
            overrides[k] = v
    if args.smoke:
        overrides = {"total_timesteps": 6000, "n_envs": 4, "n_steps": 128, "batch_size": 256, "n_epochs": 2,
                     "checkpoint_every_steps": 3000, "eval_every_steps": 3000, "eval_episodes": 2,
                     "run_name": "smoke", "log_dir": "logs/smoke", "checkpoint_dir": "checkpoints_smoke",
                     "metrics_file": "logs/smoke/metrics.jsonl", "status_file": "logs/smoke/training_status.json",
                     "control_file": "logs/smoke/training_control.json", **overrides}
    env_overrides: dict = {}
    if "randomization_level" in overrides:
        env_overrides = {"randomization": {"level": float(overrides.pop("randomization_level"))}}
        configs["environment"] = deep_update(configs["environment"], env_overrides)
    _boot_pre = deep_update(tcfg.get("bootstrap", {}) or {}, overrides.get("bootstrap") or {})
    if _boot_pre.get("init_from_demos", False) and float(_boot_pre.get("curriculum_scale", 1.0)) != 1.0:
        cs = float(_boot_pre.get("curriculum_scale", 1.0))
        cur = dict(configs["environment"]["task"].get("curriculum", {}) or {})
        scaled = {k: (float(v) * cs if k.endswith("_prob") else v) for k, v in cur.items()}
        env_overrides = deep_update(env_overrides, {"task": {"curriculum": scaled}})
        configs["environment"] = deep_update(configs["environment"], {"task": {"curriculum": scaled}})
    episode_cap = overrides.pop("training_episodes", None)
    tcfg = deep_update(tcfg, overrides)
    configs["training"] = tcfg
    configs["state_layout"] = build_state_layout(list(configs["environment"]["task"]["objects"]))

    log_dir = resolve_path(tcfg.get("log_dir", "logs")); log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = resolve_path(tcfg.get("checkpoint_dir", "checkpoints")); ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = resolve_path(tcfg.get("metrics_file", "logs/metrics.jsonl"))
    status_file = resolve_path(tcfg.get("status_file", "logs/training_status.json"))
    control_file = resolve_path(tcfg.get("control_file", "logs/training_control.json"))
    log_file = log_dir / "training.log"
    run_name = tcfg.get("run_name") or f"ppo_sorting_{time.strftime('%Y%m%d_%H%M%S')}"

    dev, dev_name = detect_device(tcfg.get("device", "auto"))
    device_str = device_label(dev, dev_name)
    seed = int(tcfg.get("seed", 0))
    np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() // 2)))
    n_envs = int(tcfg.get("n_envs", 8))
    include_image = bool(configs["environment"]["observation"].get("include_image", True))

    def write_status(state: str, error: str | None = None, **extra):
        payload = {"state": state, "pid": os.getpid(), "run_name": run_name, "device": device_str,
                   "timesteps": 0, "total_timesteps": int(tcfg["total_timesteps"]), "error": error,
                   "updated": time.time(), **extra}
        tmp = status_file.with_suffix(".tmp"); tmp.write_text(json.dumps(payload, default=str)); os.replace(tmp, status_file)

    write_status("starting")
    from stable_baselines3 import PPO
    from stable_baselines3.common.logger import Logger, HumanOutputFormat
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, VecMonitor

    venv = build_vec_env(configs, n_envs, seed, env_overrides, include_image)

    resume_path = None
    if not args.resume and not args.smoke:
        # A fresh run must not overwrite an earlier run's checkpoints (names are step counts only):
        # move whatever is in the checkpoint dir into checkpoints/archive_<previous run>_<timestamp>/.
        from training.checkpointing import archive_existing_checkpoints
        archived = archive_existing_checkpoints(ckpt_dir)
        if archived:
            print(f"archived {archived['count']} checkpoint(s) of run '{archived['run_name']}' to {archived['dir']}")
    if args.resume:
        resume_path = latest_checkpoint(ckpt_dir) if args.resume == "latest" else (ckpt_dir / args.resume)
        if resume_path is None or not (resume_path / "model.zip").exists():
            print(f"resume checkpoint not found: {args.resume}", file=sys.stderr)
            write_status("crashed", error=f"resume checkpoint not found: {args.resume}")
            sys.exit(2)

    boot_cfg = dict(tcfg.get("bootstrap", {}) or {})
    use_bc_anchor = bool(boot_cfg.get("init_from_demos", False)) and float(boot_cfg.get("bc_loss_coef", 0.0)) > 0
    from training.policies.ppo_bc import PPOWithBC
    Algo = PPOWithBC if use_bc_anchor else PPO
    bc_kwargs = dict(bc_loss_coef=float(boot_cfg.get("bc_loss_coef", 1.0)), bc_loss_coef_floor=float(boot_cfg.get("bc_loss_coef_floor", 0.2)),
                     bc_loss_decay_timesteps=int(boot_cfg.get("bc_loss_decay_timesteps", 1_000_000)),
                     bc_loss_type=str(boot_cfg.get("bc_loss_type", "mse")), bc_batch_size=int(boot_cfg.get("bc_batch_size", 256))) if use_bc_anchor else {}
    target_kl = tcfg.get("target_kl", None)
    target_kl = float(target_kl) if target_kl not in (None, 0, "none") else None
    if resume_path is not None:
        vn = resume_path / "vecnormalize.pkl"
        if vn.exists() and isinstance(venv, VecNormalize):
            inner = venv.venv
            venv = VecNormalize.load(str(vn), inner)
            venv.training = True
        model = Algo.load(str(resume_path / "model.zip"), env=venv, device=dev, print_system_info=False,
                          custom_objects={"learning_rate": float(tcfg["learning_rate"]), "target_kl": target_kl,
                                          "clip_range": float(tcfg["clip_range"]), "ent_coef": float(tcfg["ent_coef"]), **bc_kwargs})
        start_steps = model.num_timesteps
        if use_bc_anchor:   # a plain-PPO checkpoint carries no anchor settings; apply the configured ones explicitly
            model.bc_loss_coef0 = bc_kwargs["bc_loss_coef"]; model.bc_loss_coef_floor = bc_kwargs["bc_loss_coef_floor"]
            model.bc_loss_decay_timesteps = bc_kwargs["bc_loss_decay_timesteps"]; model.bc_loss_type = bc_kwargs["bc_loss_type"]
            model.bc_batch_size = bc_kwargs["bc_batch_size"]
        model.target_kl = target_kl
        print(f"resumed from {resume_path.name} at {start_steps} timesteps")
    else:
        model = Algo(tcfg.get("policy", "MultiInputPolicy"), venv, learning_rate=float(tcfg["learning_rate"]),
                     n_steps=int(tcfg["n_steps"]), batch_size=int(tcfg["batch_size"]), n_epochs=int(tcfg["n_epochs"]),
                     gamma=float(tcfg["gamma"]), gae_lambda=float(tcfg["gae_lambda"]), clip_range=float(tcfg["clip_range"]),
                     ent_coef=float(tcfg["ent_coef"]), vf_coef=float(tcfg["vf_coef"]), max_grad_norm=float(tcfg["max_grad_norm"]),
                     target_kl=target_kl, policy_kwargs=build_policy_kwargs(tcfg), seed=seed, device=dev, verbose=0, **bc_kwargs)
        start_steps = 0

    def eval_env_factory():
        e = DummyVecEnv([lambda: make_env(sim_cfg=configs["simulation"], env_cfg=configs["environment"], sb3=True,
                                          include_image=include_image)])
        e = VecMonitor(e)
        if isinstance(venv, VecNormalize):
            e = VecNormalize(e, norm_obs=True, norm_reward=False, norm_obs_keys=["state"], clip_obs=10.0, training=False)
        e.seed(seed + 10_000)
        return e

    # ---------------- imitation bootstrap (Section 43): BC pretraining of the PPO policy itself ----------------
    boot = dict(tcfg.get("bootstrap", {}) or {})
    bc_trainer = None
    bootstrap_info: dict | None = None
    if boot.get("init_from_demos", False):
        from training.bc_pretrain import select_demo_dirs, DemoBuffer, bc_pretrain
        from training.checkpointing import save_checkpoint
        demos = select_demo_dirs(resolve_path(boot.get("demos_dir", "datasets/demos")), boot.get("demo_types"),
                                 bool(boot.get("successful_only", True)))
        if not demos:
            write_status("crashed", error="init_from_demos requested but no (successful) demonstration episodes found")
            print("no demonstrations found", file=sys.stderr)
            sys.exit(3)
        if resume_path is not None:
            print("resuming from a checkpoint: BC pretraining skipped (weights come from the checkpoint); aux BC steps stay active")
        def _blog(msg):
            line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
            print(line, flush=True)
            with open(log_file, "a") as f:
                f.write(line + "\n")
        write_status("bc_pretraining", bootstrap={"enabled": True, "stage": "loading demos", "demo_episodes": len(demos)})
        buf = DemoBuffer.from_dirs(demos, include_image=include_image)
        gamma = float(tcfg["gamma"])
        if resume_path is None:
            bc_trainer, bootstrap_info = bc_pretrain(
                model, venv, buf, epochs=int(boot.get("bc_epochs", 40)), batch_size=int(boot.get("bc_batch_size", 256)),
                lr=float(boot.get("bc_learning_rate", 5e-4)), gamma=gamma, vf_coef=float(tcfg.get("vf_coef", 0.5)),
                value_regression=bool(boot.get("bc_value_regression", True)), holdout_fraction=float(boot.get("holdout_fraction", 0.1)),
                seed=seed, log=_blog,
                progress=lambda row: write_status("bc_pretraining", bootstrap={"enabled": True, "stage": "bc_pretraining", "demo_episodes": len(demos), "demo_frames": int(buf.n), **row}))
            bootstrap_info["mechanism"] = "BC pretraining of the PPO network (direct weight transfer) + interleaved BC aux steps"
            try:
                save_checkpoint(model, venv, ckpt_dir, 0, configs, {"run_name": run_name, "algorithm": "PPO", "episodes": 0, "device": device_str,
                                                                    "reason": "bc_init", "bootstrap": bootstrap_info,
                                                                    "state_layout": configs["state_layout"]})
                _blog("checkpoint saved (bc_init): checkpoint_000000000")
            except Exception as e:
                _blog(f"bc_init checkpoint save failed: {e}")
        else:
            from training.bc_pretrain import BCTrainer
            bc_trainer = BCTrainer(model, venv, buf, lr=float(boot.get("bc_learning_rate", 5e-4)), vf_coef=float(tcfg.get("vf_coef", 0.5)),
                                   value_regression=bool(boot.get("bc_value_regression", True)), gamma=gamma,
                                   holdout_fraction=float(boot.get("holdout_fraction", 0.1)), seed=seed)
            bootstrap_info = {"enabled": True, "demo_episodes": len(demos), "demo_frames": int(buf.n), "mechanism": "interleaved BC aux steps only (resumed run)"}
        bootstrap_info.update({"bc_aux_coef": float(boot.get("bc_aux_coef", 0.0)), "bc_aux_timesteps": int(boot.get("bc_aux_timesteps", 0)),
                               "bc_aux_steps_per_iter": int(boot.get("bc_aux_steps_per_iter", 0)), "bc_batch_size": int(boot.get("bc_batch_size", 256)),
                               "bc_loss_coef": float(boot.get("bc_loss_coef", 0.0)), "bc_loss_coef_floor": float(boot.get("bc_loss_coef_floor", 0.0)),
                               "bc_loss_decay_timesteps": int(boot.get("bc_loss_decay_timesteps", 0)), "bc_loss_type": str(boot.get("bc_loss_type", "mse")),
                               "freeze_obs_norm": bool(boot.get("freeze_obs_norm", False)), "curriculum_scale": float(boot.get("curriculum_scale", 1.0)),
                               "target_kl": target_kl, "log_std_init": float(tcfg.get("log_std_init", -0.5))})
        if bool(boot.get("freeze_obs_norm", False)) and isinstance(venv, VecNormalize):
            # keep the input space the BC solution was learned in (statistics were seeded from the demos)
            venv.training = False
            _blog("VecNormalize statistics frozen (seeded from demonstrations)")
        if use_bc_anchor:
            # in-loss BC anchor: demo minibatches normalised exactly like the rollout observations
            _norm_states = bc_trainer.norm_states
            _use_img = "image" in model.observation_space.spaces
            def _demo_sampler(n, _buf=buf, _bc=bc_trainer):
                idx = _bc.sample(n)
                obs = {"state": _bc.norm_states[idx]}
                if _use_img and _buf.images is not None:
                    obs["image"] = _buf.images[idx]
                return obs, _buf.actions[idx]
            model.set_demo_sampler(_demo_sampler)
            _blog(f"in-loss BC anchor active: coef {bc_kwargs['bc_loss_coef']} -> floor {bc_kwargs['bc_loss_coef_floor']} over {bc_kwargs['bc_loss_decay_timesteps']} steps ({bc_kwargs['bc_loss_type']})")

    dataset_factory = None
    dcfg = tcfg.get("dataset", {})
    if dcfg.get("record_eval_episodes", False):
        def dataset_factory():
            from lerobot_adapter.dataset_writer import LeRobotEpisodeRecorder
            return LeRobotEpisodeRecorder(root=resolve_path(dcfg.get("dir", "datasets/eval_rollouts")) / run_name,
                                          state_layout=configs["state_layout"], fps=int(round(1.0 / configs["simulation"]["control"]["control_dt"])),
                                          image_shape=(configs["simulation"]["camera"]["overhead"]["height"], configs["simulation"]["camera"]["overhead"]["width"], 3) if include_image else None,
                                          robot_type=configs["simulation"]["robot"]["name"])

    cb = TrainerControlCallback(configs=configs, status_file=status_file, control_file=control_file, log_file=log_file,
                                ckpt_dir=ckpt_dir, checkpoint_every=int(tcfg["checkpoint_every_steps"]),
                                eval_every=int(tcfg.get("eval_every_steps", 0)), eval_episodes=int(tcfg.get("eval_episodes", 5)),
                                eval_env_factory=eval_env_factory, device=device_str, run_name=run_name,
                                total_timesteps=int(tcfg["total_timesteps"]), episode_cap=episode_cap,
                                dataset_recorder_factory=dataset_factory, bc_trainer=bc_trainer, bootstrap_info=bootstrap_info,
                                lr_gate=tcfg.get("lr_gate"), base_lr=float(tcfg["learning_rate"]))
    cb.last_ckpt_at = start_steps
    cb.last_eval_at = start_steps
    logger = Logger(folder=str(log_dir), output_formats=[HumanOutputFormat(sys.stdout),
                                                          JSONLWriter(metrics_file, extra_provider=lambda: {"run_name": run_name, "device": device_str})])
    model.set_logger(logger)

    exit_code = 0
    try:
        model.learn(total_timesteps=int(tcfg["total_timesteps"]), callback=cb, reset_num_timesteps=(resume_path is None),
                    log_interval=1, progress_bar=False)
        final_state = "stopped" if cb.stop_requested else "completed"
        cb.save("final")
        cb.state = final_state
        cb.log(f"training {final_state} at {model.num_timesteps} timesteps")
    except Exception as e:  # crash: save latest valid weights, surface error
        cb.error = f"{type(e).__name__}: {e}"
        cb.log("TRAINING CRASHED\n" + traceback.format_exc())
        try:
            cb.save("crash")
        except Exception:
            pass
        cb.state = "crashed"
        exit_code = 1
    finally:
        cb.write_status(force=True)
        try:
            venv.close()
        except Exception:
            pass
        if cb.eval_env is not None:
            try:
                cb.eval_env.close()
            except Exception:
                pass
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
