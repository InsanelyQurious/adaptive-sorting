"""Launches / controls the training process and reads its persisted state.

The trainer (training/train.py) is a separate OS process in its own session, so it
survives backend restarts and browser disconnects. Control is file-based:
  logs/training_control.json   commands written by the backend, polled by the trainer
  logs/training_status.json    status written by the trainer
  logs/metrics.jsonl           one JSON line per PPO update (real SB3 logger values)
"""
from __future__ import annotations
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from simulation.config import resolve_path, load_yaml, PROJECT_ROOT


class TrainingManager:
    def __init__(self, train_cfg: dict | None = None):
        self.cfg = train_cfg or load_yaml("configs/training.yaml")
        self.status_file = resolve_path(self.cfg.get("status_file", "logs/training_status.json"))
        self.control_file = resolve_path(self.cfg.get("control_file", "logs/training_control.json"))
        self.metrics_file = resolve_path(self.cfg.get("metrics_file", "logs/metrics.jsonl"))
        self.log_dir = resolve_path(self.cfg.get("log_dir", "logs"))
        self.log_dir.mkdir(exist_ok=True)
        self.proc: subprocess.Popen | None = None

    # ------------------------------------------------------------------ status
    def read_status(self) -> dict[str, Any]:
        st: dict[str, Any] = {"state": "not_started"}
        if self.status_file.exists():
            try:
                st = json.loads(self.status_file.read_text())
            except Exception:
                st = {"state": "unknown", "error": "status file unreadable"}
        pid = st.get("pid")
        alive = self._pid_alive(pid) if pid else False
        st["process_alive"] = alive
        if st.get("state") in ("running", "paused", "starting", "evaluating", "saving") and not alive:
            st["state"] = "crashed" if st.get("state") != "stopping" else "stopped"
            st.setdefault("error", "trainer process is not running (no clean shutdown recorded)")
        return st

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(int(pid), 0)
        except (ProcessLookupError, ValueError, TypeError):
            return False
        except PermissionError:
            return True
        # zombie check
        try:
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("State:"):
                        return "Z" not in line.split()[1]
        except FileNotFoundError:
            return False
        return True

    def is_running(self) -> bool:
        st = self.read_status()
        return bool(st.get("process_alive")) and st.get("state") in ("running", "paused", "starting", "evaluating", "saving", "stopping")

    def read_metrics(self, last_n: int = 500) -> list[dict]:
        if not self.metrics_file.exists():
            return []
        try:
            with open(self.metrics_file, "rb") as f:
                lines = f.readlines()[-last_n:]
        except Exception:
            return []
        out = []
        for ln in lines:
            try:
                out.append(json.loads(ln))
            except Exception:
                continue
        return out

    def metrics_size(self) -> int:
        return self.metrics_file.stat().st_size if self.metrics_file.exists() else 0

    def read_new_metrics(self, offset: int) -> tuple[list[dict], int]:
        if not self.metrics_file.exists():
            return [], 0
        size = self.metrics_file.stat().st_size
        if size < offset:
            offset = 0
        out = []
        with open(self.metrics_file, "rb") as f:
            f.seek(offset)
            data = f.read()
        if not data:
            return [], offset
        # only consume complete lines
        last_nl = data.rfind(b"\n")
        if last_nl < 0:
            return [], offset
        for ln in data[:last_nl].split(b"\n"):
            if ln.strip():
                try:
                    out.append(json.loads(ln))
                except Exception:
                    pass
        return out, offset + last_nl + 1

    # ------------------------------------------------------------------ control
    def _write_control(self, command: str, **extra):
        payload = {"command": command, "ts": time.time(), **extra}
        tmp = self.control_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, self.control_file)

    def start(self, overrides: dict[str, Any], resume_from: str | None = None) -> dict:
        if self.is_running():
            raise RuntimeError("training is already running")
        if self.control_file.exists():
            self.control_file.unlink()
        overrides_file = self.log_dir / "training_overrides.json"
        overrides_file.write_text(json.dumps(overrides or {}, indent=2))
        cmd = [sys.executable, "-m", "training.train", "--overrides", str(overrides_file)]
        if resume_from:
            cmd += ["--resume", resume_from]
        stdout = open(self.log_dir / "training_stdout.log", "ab")
        env = dict(os.environ, MUJOCO_GL=os.environ.get("MUJOCO_GL", "egl"), PYTHONUNBUFFERED="1")
        # detach: new session so it is independent of the backend and of the terminal
        self.proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=stdout, stderr=subprocess.STDOUT,
                                     start_new_session=True, env=env)
        # provisional status until the trainer writes its own
        self.status_file.write_text(json.dumps({"state": "starting", "pid": self.proc.pid, "started": time.time(),
                                                "overrides": overrides, "resume_from": resume_from}))
        return {"pid": self.proc.pid, "command": " ".join(cmd)}

    def pause(self):
        self._require_running(); self._write_control("pause")

    def resume(self):
        self._require_running(); self._write_control("resume")

    def save(self):
        self._require_running(); self._write_control("save")

    def evaluate(self, episodes: int = 10):
        self._require_running(); self._write_control("evaluate", episodes=int(episodes))

    def stop(self, timeout: float = 60.0) -> dict:
        st = self.read_status()
        pid = st.get("pid")
        if not st.get("process_alive"):
            return {"stopped": True, "note": "not running"}
        self._write_control("stop")
        t0 = time.time()
        while time.time() - t0 < timeout:
            if not self._pid_alive(pid):
                return {"stopped": True, "graceful": True, "seconds": round(time.time() - t0, 1)}
            time.sleep(0.25)
        try:
            os.kill(int(pid), signal.SIGTERM)  # trainer handles SIGTERM by saving a checkpoint
        except ProcessLookupError:
            return {"stopped": True, "graceful": True}
        t1 = time.time()
        while time.time() - t1 < 30 and self._pid_alive(pid):
            time.sleep(0.25)
        return {"stopped": not self._pid_alive(pid), "graceful": False}

    def _require_running(self):
        if not self.is_running():
            raise RuntimeError("training is not running")
