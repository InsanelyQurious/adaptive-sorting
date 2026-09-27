"""Config loading helpers (YAML -> plain dicts / dot-access)."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    with open(p) as f:
        return yaml.safe_load(f) or {}


def load_configs(sim_path="configs/simulation.yaml", env_path="configs/environment.yaml",
                 train_path="configs/training.yaml") -> dict[str, dict]:
    return {
        "simulation": load_yaml(sim_path),
        "environment": load_yaml(env_path),
        "training": load_yaml(train_path),
    }


def deep_update(base: dict, override: dict | None) -> dict:
    """Recursively merge override into a copy of base."""
    import copy
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p
