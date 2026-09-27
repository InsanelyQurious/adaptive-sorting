"""Per-episode demonstration recorder (Section 43).

Every recorded episode becomes its OWN LeRobotDataset directory:

    datasets/demos/<episode_type>_<YYYYmmdd_HHMMSS_ffffff>/
        meta/ data/ images/ ...      (genuine LeRobotDataset v3.0, written by LeRobotEpisodeRecorder)
        demo.json                    sidecar: episode_type, outcome, duration, frames, seed, timestamp

One directory per episode makes "discard a bad demonstration" a plain directory removal and lets
the BC loader concatenate any subset. `episode_type` is one of
    teleop_demo     human keyboard/mouse teleoperation through the Teleop tab
    scripted_demo   hand-coded controller using ground-truth object/bin poses (NOT a human, NOT the policy)
    policy_rollout  the trained policy acting (live recordings from the ● Rec button)
    random_policy   uniform random actions
LeRobotDataset has no string feature type besides `task`, so the tag lives in the sidecar (and the
directory name); the task text itself stays the natural-language task.
"""
from __future__ import annotations
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
import numpy as np

from lerobot_adapter.dataset_writer import LeRobotEpisodeRecorder, TASK_TEXT

DEMO_TYPES = ("teleop_demo", "assisted_demo", "scripted_demo", "policy_rollout", "random_policy", "manual")
DEMO_TYPE_LABELS = {"teleop_demo": "human teleop (keyboard/mouse)", "assisted_demo": "human click-to-pick (scripted execution)",
                    "scripted_demo": "scripted ground-truth batch (not human, not policy)", "policy_rollout": "policy rollout",
                    "random_policy": "random actions", "manual": "manual"}


class DemoRecorder:
    """Same add_step / set_last_reward / end_episode / close surface as LeRobotEpisodeRecorder,
    but opens a fresh single-episode dataset directory for every episode."""

    def __init__(self, root: Path, episode_type: str, state_layout: list[str], fps: int,
                 image_shape: tuple | None, robot_type: str, extra_meta: dict | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.episode_type = episode_type
        self.state_layout = list(state_layout)
        self.fps = fps
        self.image_shape = image_shape
        self.robot_type = robot_type
        self.extra_meta = dict(extra_meta or {})
        self._rec: LeRobotEpisodeRecorder | None = None
        self._ep_dir: Path | None = None
        self._ep_started = 0.0
        self.episodes = 0                 # finished episodes in this recording session
        self.frames_in_episode = 0
        self.finished: list[dict] = []    # sidecars of finished episodes (this session)
        self.pending_frames_meta: dict = {}
        self._initial_state: dict | None = None
        self._episode_type_override: str | None = None

    def set_initial_state(self, state: dict):
        """Snapshot of the simulator at the start of the episode being recorded (for exact replay)."""
        self._initial_state = state

    def retag_current(self, episode_type: str):
        """Change the type of the episode currently being recorded (e.g. teleop -> assisted once click-to-pick is used)."""
        self._episode_type_override = episode_type

    # ------------------------------------------------------------------ lifecycle
    def _open_episode(self):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._ep_dir = self.root / f"{self.episode_type}_{stamp}"
        self._rec = LeRobotEpisodeRecorder(root=self._ep_dir, state_layout=self.state_layout, fps=self.fps,
                                           image_shape=self.image_shape, robot_type=self.robot_type,
                                           repo_id=f"volvo-physical-ai/{self.episode_type}", task=TASK_TEXT)
        self._ep_dir = self._rec.root   # recorder may have suffixed the name
        self._ep_started = time.time()
        self.frames_in_episode = 0

    def add_step(self, obs: dict, action: np.ndarray, reward: float | None, done: bool, info: dict):
        if self._rec is None:
            self._open_episode()
        self._rec.add_step(obs, action, reward, done, info)
        self.frames_in_episode = self._rec.frames_in_episode + 1

    def set_last_reward(self, reward: float, done: bool, success: bool = False):
        if self._rec is not None:
            self._rec.set_last_reward(reward, done, success)

    def end_episode(self, result: dict | None = None, aborted: bool = False) -> dict | None:
        """Finalize the current episode directory and write demo.json. Returns the sidecar."""
        if self._rec is None:
            return None
        rec, ep_dir = self._rec, self._ep_dir
        self._rec = None
        self._ep_dir = None
        rec.close()
        frames = rec.episode_meta[0]["frames"] if rec.episode_meta else 0
        if frames == 0:
            shutil.rmtree(ep_dir, ignore_errors=True)
            return None
        r = result or {}
        etype = self._episode_type_override or self.episode_type
        if etype != self.episode_type:
            new_dir = ep_dir.parent / (etype + ep_dir.name[len(self.episode_type):])
            shutil.move(str(ep_dir), str(new_dir)); ep_dir = new_dir
        side = {
            "name": ep_dir.name,
            "episode_type": etype,
            "initial_state": self._initial_state,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "frames": frames,
            "duration_s": round(frames / self.fps, 1),
            "wall_seconds": round(time.time() - self._ep_started, 1),
            "aborted": bool(aborted),
            "outcome": {
                "success": bool(r.get("success", False)),
                "placed_count": int(r.get("placed_count") or 0),
                "reward": r.get("reward"),
                "length": r.get("length"),
                "grasp_success": bool(r.get("grasp_success", False)),
                "dropped": bool(r.get("dropped", False)),
                "wrong_bin": bool(r.get("wrong_bin", False)),
                "truncated": bool(r.get("truncated", False)),
            },
            "seed": r.get("seed"),
            "object_configuration": r.get("object_configuration"),
            "robot": self.robot_type,
            "fps": self.fps,
            "task": TASK_TEXT,
            **self.extra_meta,
        }
        (ep_dir / "demo.json").write_text(json.dumps(side, indent=1, default=_json_default))
        self._initial_state = None
        self._episode_type_override = None
        self.episodes += 1
        self.finished.append(side)
        return side

    def close(self):
        self.end_episode(aborted=True)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


def load_demo_episode(path: Path) -> dict:
    """Actions, observation.state and sidecar of one recorded episode (read straight from the parquet files)."""
    import pyarrow.parquet as pq
    path = Path(path)
    side = json.loads((path / "demo.json").read_text())
    files = sorted((path / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet data in {path}")
    tables = [pq.read_table(f, columns=["action", "observation.state", "frame_index", "next.reward"]) for f in files]
    import pyarrow as pa
    t = pa.concat_tables(tables).to_pandas().sort_values("frame_index")
    actions = np.stack([np.asarray(a, dtype=np.float32) for a in t["action"]])
    states = np.stack([np.asarray(a, dtype=np.float32) for a in t["observation.state"]])
    rewards = np.asarray([float(np.asarray(r).reshape(-1)[0]) for r in t["next.reward"]], dtype=np.float32)
    return {"name": path.name, "sidecar": side, "actions": actions, "states": states, "rewards": rewards}


# ---------------------------------------------------------------------- catalogue helpers
def list_demos(root: Path) -> list[dict]:
    root = Path(root)
    out = []
    if not root.exists():
        return out
    for d in sorted(root.iterdir()):
        sj = d / "demo.json"
        if d.is_dir() and sj.exists() and (d / "meta" / "info.json").exists():
            try:
                side = json.loads(sj.read_text())
                side.pop("initial_state", None)          # bulky; fetched on demand for replay
                side["path"] = str(d)
                side["size_mb"] = round(sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e6, 2)
                out.append(side)
            except Exception:
                continue
    return out


def demo_summary(demos: list[dict]) -> dict:
    by_type: dict[str, dict] = {}
    for dm in demos:
        t = dm.get("episode_type", "unknown")
        b = by_type.setdefault(t, {"episodes": 0, "frames": 0, "successes": 0, "placed": 0})
        b["episodes"] += 1
        b["frames"] += int(dm.get("frames") or 0)
        b["successes"] += int(bool(dm.get("outcome", {}).get("success")))
        b["placed"] += int(dm.get("outcome", {}).get("placed_count") or 0)
    return {"total_episodes": len(demos), "total_frames": sum(int(d.get("frames") or 0) for d in demos), "by_type": by_type}


def discard_demo(root: Path, name: str) -> bool:
    """Delete one recorded episode directory. `name` must be a direct child of root."""
    root = Path(root).resolve()
    if not name or "/" in name or name in (".", ".."):
        raise ValueError("invalid demo name")
    target = (root / name).resolve()
    if target.parent != root or not (target / "demo.json").exists():
        raise FileNotFoundError(f"no such demonstration: {name}")
    shutil.rmtree(target)
    return True
