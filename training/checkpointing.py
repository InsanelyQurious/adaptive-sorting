"""Checkpoint layout (Section 17):

checkpoints/checkpoint_<timesteps:09d>/
    model.zip                 SB3 PPO archive: policy weights, optimizer state, hyperparameters
    vecnormalize.pkl          observation normalization statistics (VecNormalize), if used
    training_config.yaml      the training config in effect
    environment_config.yaml   task / reward / randomization config in effect
    simulation_config.yaml    robot / control / camera config in effect
    metadata.json             algorithm, timesteps, episodes, device, eval results, obs/action spec
checkpoints/latest.json       {"name": "<most recent checkpoint dir>"}
"""
from __future__ import annotations
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any
import yaml

from simulation.config import resolve_path

CHECKPOINT_PREFIX = "checkpoint_"


def checkpoint_name(timesteps: int) -> str:
    return f"{CHECKPOINT_PREFIX}{int(timesteps):09d}"


def _atomic_write_text(path: Path, text: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def save_checkpoint(model, vecnormalize, ckpt_dir: Path, timesteps: int, configs: dict[str, dict],
                    metadata: dict[str, Any]) -> Path:
    """Write the checkpoint to a temporary directory first, then rename it into place so a
    crash mid-save can never leave a half-written checkpoint that looks valid."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    name = checkpoint_name(timesteps)
    final = ckpt_dir / name
    tmp = ckpt_dir / f".{name}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    model.save(str(tmp / "model.zip"))
    if vecnormalize is not None:
        vecnormalize.save(str(tmp / "vecnormalize.pkl"))
    for key, fname in (("training", "training_config.yaml"), ("environment", "environment_config.yaml"),
                       ("simulation", "simulation_config.yaml")):
        (tmp / fname).write_text(yaml.safe_dump(configs.get(key, {}), sort_keys=False))
    meta = dict(metadata)
    meta.update({"name": name, "timesteps": int(timesteps), "created": time.time(),
                 "created_iso": time.strftime("%Y-%m-%dT%H:%M:%S")})
    (tmp / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    if final.exists():
        shutil.rmtree(final)
    os.replace(tmp, final)
    _atomic_write_text(ckpt_dir / "latest.json", json.dumps({"name": name}))
    return final


def list_checkpoints(ckpt_dir: Path | str = "checkpoints") -> list[dict]:
    ckpt_dir = resolve_path(ckpt_dir)
    out = []
    if not ckpt_dir.exists():
        return out
    for p in sorted(ckpt_dir.iterdir()):
        if not p.is_dir() or not p.name.startswith(CHECKPOINT_PREFIX) or not (p / "model.zip").exists():
            continue
        meta = {}
        mp = p / "metadata.json"
        if mp.exists():
            try:
                meta = json.loads(mp.read_text())
            except Exception:
                meta = {"error": "unreadable metadata"}
        eval_at = meta.get("eval_at_timesteps")
        out.append({"name": p.name, "path": str(p), "timesteps": meta.get("timesteps"),
                    "created_iso": meta.get("created_iso"), "algorithm": meta.get("algorithm"),
                    "success_rate": meta.get("eval_success_rate"), "mean_reward": meta.get("eval_mean_reward"),
                    "placement_rate": meta.get("eval_placement_rate"), "grasp_rate": meta.get("eval_grasp_rate"),
                    "eval_episodes": meta.get("eval_episodes"), "eval_at_timesteps": eval_at,
                    "eval_matches": bool(eval_at is not None and meta.get("timesteps") is not None and abs(int(eval_at) - int(meta["timesteps"])) <= 1024),
                    "run_name": meta.get("run_name"), "reason": meta.get("reason"),
                    "episodes": meta.get("episodes"), "device": meta.get("device"),
                    "size_mb": round(sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6, 2)})
    return out


def latest_checkpoint(ckpt_dir: Path | str = "checkpoints") -> Path | None:
    ckpt_dir = resolve_path(ckpt_dir)
    lj = ckpt_dir / "latest.json"
    if lj.exists():
        try:
            name = json.loads(lj.read_text())["name"]
            p = ckpt_dir / name
            if (p / "model.zip").exists():
                return p
        except Exception:
            pass
    cks = list_checkpoints(ckpt_dir)
    return Path(cks[-1]["path"]) if cks else None


def load_metadata(ckpt_path: Path | str) -> dict:
    p = resolve_path(ckpt_path) / "metadata.json"
    return json.loads(p.read_text()) if p.exists() else {}


def archive_existing_checkpoints(ckpt_dir: Path | str = "checkpoints") -> dict | None:
    """Move all checkpoint_* directories (and latest.json) into an archive_<run_name>_<timestamp>/ sub-directory.
    Called by a fresh training run so step-count-named checkpoints of the previous run are never overwritten.
    Archived checkpoints stay loadable by absolute path (see checkpoints/ARCHIVES.md)."""
    ckpt_dir = resolve_path(ckpt_dir)
    if not ckpt_dir.exists():
        return None
    cks = [p for p in sorted(ckpt_dir.iterdir()) if p.is_dir() and p.name.startswith(CHECKPOINT_PREFIX)]
    if not cks:
        return None
    run_name = "unknown"
    for p in reversed(cks):
        try:
            run_name = json.loads((p / "metadata.json").read_text()).get("run_name") or run_name
            break
        except Exception:
            continue
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(run_name))
    dest = ckpt_dir / f"archive_{safe}_{time.strftime('%Y%m%d_%H%M%S')}"
    dest.mkdir(parents=True, exist_ok=True)
    for p in cks:
        shutil.move(str(p), str(dest / p.name))
    lj = ckpt_dir / "latest.json"
    if lj.exists():
        shutil.move(str(lj), str(dest / "latest.json"))
    return {"count": len(cks), "run_name": run_name, "dir": str(dest)}


def eval_score(ck: dict) -> tuple:
    """Ranking key for checkpoint selection: eval success rate, then placement rate, then eval reward."""
    return (float(ck.get("success_rate") or 0.0), float(ck.get("placement_rate") or 0.0), float(ck.get("mean_reward") if ck.get("mean_reward") is not None else -1e9))


def best_checkpoint(ckpt_dir: Path | str = "checkpoints") -> Path | None:
    """The checkpoint with the best evaluation (highest eval success, ties -> placement rate -> eval reward).
    Only checkpoints whose evaluation was run at (about) their own step count are eligible, so a checkpoint
    is never credited with an evaluation of different weights; if none qualifies, fall back to any checkpoint
    with an eval, then to the latest."""
    cks = list_checkpoints(ckpt_dir)
    if not cks:
        return None
    eligible = [c for c in cks if c.get("eval_matches") and c.get("success_rate") is not None]
    if not eligible:
        eligible = [c for c in cks if c.get("success_rate") is not None]
    if not eligible:
        return latest_checkpoint(ckpt_dir)
    best = max(eligible, key=lambda c: (eval_score(c), c.get("timesteps") or 0))   # newest wins exact ties
    return Path(best["path"])
