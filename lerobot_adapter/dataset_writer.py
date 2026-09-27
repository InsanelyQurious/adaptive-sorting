"""LeRobotDataset (v3.0) writer for recorded rollouts (Section 18).

Features written per frame:
  observation.images.overhead  uint8 image (the real policy input)
  observation.state            float32 vector (named by state_layout)
  action                       float32 [-1,1]^7
  next.reward, next.done, next.success
plus the LeRobot-managed columns (timestamp, frame_index, episode_index, index, task_index).
The result is a genuine `LeRobotDataset` on disk (meta/info.json, data/*.parquet, images),
loadable with `LeRobotDataset(repo_id, root=...)`.
"""
from __future__ import annotations
import json
import shutil
import time
from pathlib import Path
import numpy as np

ACTION_NAMES = ["dX", "dY", "dZ", "dRx", "dRy", "dRz", "gripper"]
TASK_TEXT = "Pick the bracket into Bin A and the bolt into Bin B."


class LeRobotEpisodeRecorder:
    def __init__(self, root: Path, state_layout: list[str], fps: int = 10, image_shape: tuple | None = (64, 64, 3),
                 robot_type: str = "ur10e", repo_id: str = "volvo-physical-ai/sorting_rollouts", task: str = TASK_TEXT,
                 use_videos: bool = False):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        self.root = Path(root)
        if self.root.exists():
            # never append into an unknown directory; keep runs separate
            self.root = self.root.parent / f"{self.root.name}_{time.strftime('%Y%m%d_%H%M%S')}"
        self.task = task
        self.fps = fps
        self.image_shape = tuple(image_shape) if image_shape else None
        features = {
            "observation.state": {"dtype": "float32", "shape": (len(state_layout),), "names": list(state_layout)},
            "action": {"dtype": "float32", "shape": (len(ACTION_NAMES),), "names": ACTION_NAMES},
            "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
            "next.done": {"dtype": "bool", "shape": (1,), "names": None},
            "next.success": {"dtype": "bool", "shape": (1,), "names": None},
        }
        if self.image_shape:
            features["observation.images.overhead"] = {"dtype": "video" if use_videos else "image",
                                                       "shape": self.image_shape, "names": ["height", "width", "channels"]}
        self.ds = LeRobotDataset.create(repo_id=repo_id, fps=fps, features=features, root=self.root,
                                        robot_type=robot_type, use_videos=use_videos)
        self.episodes = 0
        self.frames_in_episode = 0
        self._pending: dict | None = None
        self.episode_meta: list[dict] = []

    # frames are written one step late so next.reward / next.done are known
    def add_step(self, obs: dict, action: np.ndarray, reward: float | None, done: bool, info: dict):
        self._flush_pending()
        state = obs.get("observation.state", obs.get("state"))
        img = obs.get("observation.images.overhead", obs.get("image"))
        frame = {"observation.state": np.asarray(state, dtype=np.float32),
                 "action": np.asarray(action, dtype=np.float32), "task": self.task,
                 "next.reward": np.array([float(reward) if reward is not None else 0.0], dtype=np.float32),
                 "next.done": np.array([bool(done)]), "next.success": np.array([bool(info.get("is_success", False))])}
        if self.image_shape and img is not None:
            frame["observation.images.overhead"] = np.ascontiguousarray(np.asarray(img, dtype=np.uint8))
        self._pending = frame

    def set_last_reward(self, reward: float, done: bool, success: bool = False):
        if self._pending is not None:
            self._pending["next.reward"] = np.array([float(reward)], dtype=np.float32)
            self._pending["next.done"] = np.array([bool(done)])
            self._pending["next.success"] = np.array([bool(success)])

    def _flush_pending(self):
        if self._pending is not None:
            self.ds.add_frame(self._pending)
            self.frames_in_episode += 1
            self._pending = None

    def end_episode(self, result: dict | None = None):
        self._flush_pending()
        if self.frames_in_episode == 0:
            return
        self.ds.save_episode()
        self.episodes += 1
        self.episode_meta.append({"episode_index": self.episodes - 1, "frames": self.frames_in_episode,
                                  **{k: v for k, v in (result or {}).items() if k in ("success", "reward", "length", "seed", "timesteps")}})
        (self.root / "episode_results.json").write_text(json.dumps(self.episode_meta, indent=1, default=str))
        self.frames_in_episode = 0

    def close(self):
        try:
            self.end_episode()
        finally:
            try:
                self.ds.finalize()
            except Exception:
                pass


def load_dataset(root: Path, repo_id: str = "volvo-physical-ai/sorting_rollouts"):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset(repo_id, root=str(root))
