"""Live simulation service.

One MuJoCo environment runs in a dedicated thread. Modes:
  idle           physics is not advanced; frames are still rendered/streamed
  random         uniform random actions (debug / walking skeleton)
  inference      the loaded RL policy chooses actions; episodes auto-reset with randomization
  evaluate       like inference but runs a fixed number of episodes and records results
  teleop         a human drives the end-effector (keyboard / mouse via /ws/teleop) through the SAME
                 action space and controller limits as the policy; demos can be recorded. Click-to-pick
                 ("assist") hands one object to the scripted controller, then returns to manual control
  scripted_demo  the hand-coded ground-truth controller runs N episodes (pipeline validation only)
  replay         a recorded episode's exact action sequence is replayed open-loop from its stored initial state
Everything published to the browser comes from this live environment.
"""
from __future__ import annotations
import base64
import json
import queue
import threading
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any, Callable, Optional
import numpy as np
import cv2

from simulation.config import load_yaml, resolve_path
from simulation.environments.sorting_env import VolvoSortingEnv, IMAGE_KEY, STATE_KEY
from backend.policy_loader import LoadedPolicy, resolve_checkpoint
from backend.device import detect_device, device_label

ACTION_NAMES = ["dX", "dY", "dZ", "dRx", "dRy", "dRz", "gripper"]


def jpeg_b64(img: np.ndarray, quality: int = 80) -> str:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""


class SimulationService:
    def __init__(self, sim_cfg: dict | None = None, env_cfg: dict | None = None, train_cfg: dict | None = None):
        self.sim_cfg = sim_cfg or load_yaml("configs/simulation.yaml")
        self.env_cfg = env_cfg or load_yaml("configs/environment.yaml")
        self.train_cfg = train_cfg or load_yaml("configs/training.yaml")
        self.device, self.device_name = detect_device(self.train_cfg.get("device", "auto"))
        self.env: VolvoSortingEnv | None = None
        self.mode = "idle"
        self.paused = False
        self.speed = 1.0
        self.policy: LoadedPolicy | None = None
        self.deterministic = True
        self._cmd_q: queue.Queue[tuple[Callable, dict, Optional[queue.Queue]]] = queue.Queue()
        self._subscribers: list[queue.Queue] = []
        self._sub_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.ready = threading.Event()
        self.error: str | None = None
        vp = self.sim_cfg["camera"]["viewport"]
        self.view_cam = vp["name"]
        self.view_w, self.view_h, self.jpeg_q = int(vp["width"]), int(vp["height"]), int(vp.get("jpeg_quality", 80))
        pv = self.sim_cfg["camera"]["preview"]
        self.prev_w, self.prev_h = int(pv["width"]), int(pv["height"])
        self.max_render_fps = 15.0
        self.scripted_noise = 0.0
        # live episode / inference bookkeeping
        self.obs: dict | None = None
        self.last_action = np.zeros(7, dtype=np.float32)
        self.last_reward = 0.0
        self.last_info: dict = {}
        self.episode_index = 0
        self.episode_results: deque = deque(maxlen=500)
        self.inference_stats = {"episodes": 0, "successes": 0, "placements": 0, "grasps": 0, "parts_placed": 0, "sim_seconds": 0.0, "collision_steps": 0, "episodes_with_collision": 0, "control_effort": 0.0}
        self._ep_effort: list = []
        self._ep_false_pick = False
        self.step_times: deque = deque(maxlen=50)
        self.frame_times: deque = deque(maxlen=30)
        self.total_steps = 0
        self.start_time = time.time()
        # evaluation
        self.eval_state: dict | None = None
        self.eval_results: list[dict] = []
        self.last_eval_summary: dict | None = None
        self._episode_frames_for_dataset: list = []
        self.dataset_recorder = None
        self.recording_type: str | None = None
        self.demos_dir = resolve_path("datasets/demos")
        # teleop state (written by the /ws/teleop handler, read by the sim thread)
        self.teleop_axes = np.zeros(6, dtype=np.float32)
        self.teleop_gripper = -1.0            # -1 open, +1 closed; persists between key presses
        self.teleop_last_msg = 0.0
        self.teleop_timeout_s = 0.5           # no message for this long -> motion stops (gripper state kept)
        self.teleop_clients = 0
        # scripted demo generation
        self.scripted: Any = None
        self.scripted_remaining = 0
        self.show_frustum = False
        self.camera_info: dict | None = None
        # batch evaluations (generalization scorecard / difficulty sweep / checkpoint comparison)
        self.batch: dict | None = None
        self.reels_dir = resolve_path("logs/eval_reels")
        self._reel_frames: list = []
        self._reel_trace: list = []
        # click-to-pick assisted teleop
        self.assist: dict | None = None
        self.scene_builder = None
        # simulated emergency stop: halts the control loop between control steps and holds the arm at its
        # current joint targets; it cannot interrupt a physics sub-step (see README "Safety indicator")
        self.estop = False
        self.estop_events: list[dict] = []
        self.safety_last: dict = {}
        # deterministic open-loop replay of a recorded episode
        self.replay: dict | None = None

    # ------------------------------------------------------------------ lifecycle
    def start(self):
        """Start (or restart after stop()) the simulation thread."""
        self._stop.clear()
        self.ready.clear()
        self.error = None
        self._thread = threading.Thread(target=self._run, name="sim-thread", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def call(self, fn: Callable, timeout: float = 30.0, **kwargs):
        """Run fn(**kwargs) inside the sim thread and return its result."""
        rq: queue.Queue = queue.Queue(maxsize=1)
        self._cmd_q.put((fn, kwargs, rq))
        res = rq.get(timeout=timeout)
        if isinstance(res, Exception):
            raise res
        return res

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=4)
        with self._sub_lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue):
        with self._sub_lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def _publish(self, msg: dict):
        with self._sub_lock:
            subs = list(self._subscribers)
        for q in subs:
            if q.full():
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
            try:
                q.put_nowait(msg)
            except queue.Full:
                pass

    # ------------------------------------------------------------------ main loop
    def _run(self):
        try:
            self.env = VolvoSortingEnv(self.sim_cfg, self.env_cfg)
            self.obs, self.last_info = self.env.reset(seed=0)
            self.camera_info = self.env.sim.camera_info()
            self.camera_info["viewport"] = self.env.sim.viewport_basis(self.view_cam)
            try:
                from backend.scene_builder import SceneBuilder
                self.scene_builder = SceneBuilder(self)
                self.env.sim.scene_builder = self.scene_builder
            except Exception as e:  # pragma: no cover
                self.error = f"scene builder init failed: {e}\n{traceback.format_exc()}"
                self.scene_builder = None
            self.ready.set()
            # Warm up the LeRobot dataset stack now (torch/av imports take seconds) so the first recorded
            # frame can never stall the simulation thread mid-teleop.
            try:
                import lerobot.datasets.lerobot_dataset  # noqa: F401
                import lerobot_adapter.demo_recorder  # noqa: F401
            except Exception as e:  # pragma: no cover
                self.error = f"lerobot import failed: {e}"
        except Exception as e:  # pragma: no cover
            self.error = f"{e}\n{traceback.format_exc()}"
            self.ready.set()
            return
        last_render = 0.0
        next_step_time = time.perf_counter()
        while not self._stop.is_set():
            # commands
            try:
                while True:
                    fn, kwargs, rq = self._cmd_q.get_nowait()
                    try:
                        res = fn(**kwargs)
                    except Exception as e:
                        res = e
                        self.error = f"{e}"
                    if rq is not None:
                        rq.put(res)
            except queue.Empty:
                pass
            now = time.perf_counter()
            stepping = (self.mode in ("random", "inference", "evaluate", "teleop", "scripted_demo") or
                        (self.mode == "replay" and self.replay is not None and self.replay["playing"])) and not self.paused and not self.estop
            if stepping and now >= next_step_time:
                t0 = time.perf_counter()
                self._step_once()
                self.step_times.append(time.perf_counter() - t0)
                period = self.env.sim.control_dt / max(self.speed, 1e-3)
                if self.mode == "evaluate":
                    period = 0.0
                next_step_time = max(now, next_step_time) + period
            # render + publish
            if now - last_render >= 1.0 / self.max_render_fps:
                self._publish(self._make_frame_message())
                self.frame_times.append(now)
                last_render = now
            # pacing
            sleep_for = 0.002 if stepping else 0.02
            if stepping and self.mode != "evaluate":
                sleep_for = min(max(next_step_time - time.perf_counter(), 0.0), 0.05)
            if sleep_for > 0:
                time.sleep(sleep_for)
        self.env.close()

    def _choose_action(self) -> np.ndarray:
        if self.mode == "random":
            return self.env.action_space.sample()
        if self.mode == "teleop":
            if self.assist is not None:
                a = self._assist_action()
                if a is not None:
                    return a
            a = np.zeros(7, dtype=np.float32)
            if time.time() - self.teleop_last_msg < self.teleop_timeout_s:
                a[:6] = np.clip(self.teleop_axes, -1.0, 1.0)
            a[6] = 1.0 if self.teleop_gripper > 0 else -1.0
            return a
        if self.mode == "replay":
            rp = self.replay
            if rp is None or rp["index"] >= rp["n"]:
                return np.zeros(7, dtype=np.float32)
            return np.asarray(rp["actions"][rp["index"]], dtype=np.float32)
        if self.mode == "scripted_demo":
            if self.scripted is None:
                from training.demos.scripted_controller import ScriptedSortingController
                self.scripted = ScriptedSortingController(self.env, noise_std=self.scripted_noise)
            return self.scripted.act()
        if self.policy is None:
            return np.zeros(7, dtype=np.float32)
        return self.policy.select_action(self.obs, deterministic=self.deterministic)

    def teleop_update(self, axes, gripper: float | None = None):
        """Called from the /ws/teleop handler (any thread). axes: 6 floats in [-1, 1]."""
        arr = np.zeros(6, dtype=np.float32)
        vals = list(axes or [])[:6]
        arr[:len(vals)] = np.asarray(vals, dtype=np.float32)
        self.teleop_axes = np.clip(arr, -1.0, 1.0)
        if gripper is not None:
            self.teleop_gripper = 1.0 if float(gripper) > 0 else -1.0
        self.teleop_last_msg = time.time()

    def _step_once(self):
        if self.mode == "replay":
            self._replay_step()
            return
        action = self._choose_action()
        prev_obs = self.obs
        if self.dataset_recorder is not None and self.dataset_recorder.frames_in_episode == 0:
            try:
                self.dataset_recorder.set_initial_state(self._snapshot_state())
            except Exception as e:
                self.error = f"initial state snapshot: {e}"
        obs, reward, term, trunc, info = self.env.step(action)
        self.obs, self.last_action, self.last_reward, self.last_info = obs, action, reward, info
        self.total_steps += 1
        self._ep_effort.append(float(np.sum(np.square(action))))
        if self.mode == "evaluate":
            self._reel_capture(action, reward, info)
        # rejection-case metric: did the gripper grasp a part that does not belong to the task?
        if not self._ep_false_pick and self.scene_builder is not None and any(l["part"] == "reject_cap" for l in self.scene_builder.layout):
            if self.env.sim.pads_grasping("reject_cap"):
                self._ep_false_pick = True
        if self.assist is not None:
            self._assist_after_step(info, term or trunc)
        if self.dataset_recorder is not None and self.mode != "replay":
            try:
                # frame = observation BEFORE the action, with the action taken and the resulting reward/done
                self.dataset_recorder.add_step(prev_obs, action, reward, term or trunc, info)
                self.dataset_recorder.set_last_reward(reward, term or trunc, bool(info.get("is_success", False)))
            except Exception as e:
                self.error = f"dataset recorder: {e}"
        if term or trunc:
            self._on_episode_end(info, term, trunc)

    def _on_episode_end(self, info: dict, term: bool, trunc: bool):
        if self.assist is not None:
            self._finish_assist("episode ended")
        result = {
            "episode": self.episode_index,
            "seed": info.get("seed"),
            "success": bool(info.get("is_success")),
            "placed_count": info.get("placed_count"),
            "reward": round(float(info.get("episode_reward", 0.0)), 3),
            "length": info.get("step"),
            "grasp_success": bool(info.get("grasp_achieved")),
            "lift_success": bool(info.get("lift_achieved")),
            "placement_success": (info.get("placed_count") or 0) > 0,
            "collisions": info.get("collisions"),
            "dropped": bool(info.get("dropped")),
            "wrong_bin": bool(info.get("wrong_bin")),
            "unstable": bool(info.get("unstable")),
            "terminated": bool(term), "truncated": bool(trunc),
            "object_configuration": info.get("episode_config"),
            "mode": self.mode,
            "time": time.time(),
            "sim_seconds": round(float(info.get("step", 0)) * self.env.sim.control_dt, 2),
            "collision_steps": int(info.get("collisions") or 0),
            "control_effort": round(float(np.sum(self._ep_effort)) if len(self._ep_effort) else 0.0, 3),   # sum over steps of |a|^2
            "collision_penalty_total": round(float((info.get("episode_components") or {}).get("collision_penalty", 0.0)), 3),
            "control_effort_penalty_total": round(float((info.get("episode_components") or {}).get("control_effort_penalty", 0.0)), 3),
            "false_pick": bool(getattr(self, "_ep_false_pick", False)),
        }
        self._ep_effort = []
        self._ep_false_pick = False
        self.episode_results.append(result)
        if self.mode in ("inference", "evaluate"):
            st = self.inference_stats
            st["episodes"] += 1
            st["successes"] += int(result["success"])
            st["placements"] += int(result["placement_success"])
            st["grasps"] += int(result["grasp_success"])
            st["parts_placed"] += int(result["placed_count"] or 0)
            st["sim_seconds"] += float(result["sim_seconds"])
            st["collision_steps"] += int(result["collision_steps"])
            st["episodes_with_collision"] += int(result["collision_steps"] > 0)
            st["control_effort"] += float(result["control_effort"])
        if self.mode == "evaluate" and self.eval_state is not None:
            self.eval_results.append(result)
            self.eval_state["completed"] = len(self.eval_results)
            self._save_reel(result)
            self._publish({"type": "evaluation_progress", **self.eval_progress()})
            if len(self.eval_results) >= self.eval_state["episodes"]:
                self._finish_evaluation()
        if self.dataset_recorder is not None:
            try:
                side = self.dataset_recorder.end_episode(result)
                if side is not None:
                    self._publish({"type": "demo_recorded", "demo": side})
            except Exception as e:
                self.error = f"dataset recorder: {e}"
        self._publish({"type": "episode_end", "result": result, "stats": dict(self.inference_stats)})
        self.episode_index += 1
        if self.mode == "scripted_demo":
            self.scripted_remaining -= 1
            if self.scripted is not None:
                self.scripted.reset()
            if self.scripted_remaining <= 0:
                self.mode = "idle"
                self.scripted = None
                if self.recording_type == "scripted_demo":
                    self._stop_recording()
                self._publish({"type": "scripted_demos_done"})
        seed = None
        if self.eval_state is not None and self.eval_state.get("seed") is not None:
            seed = int(self.eval_state["seed"]) + self.episode_index
        opts = {"randomization_level": self.eval_state.get("randomization_level") if self.eval_state else None,
                "fixed_poses": self.eval_state.get("fixed_poses") if self.eval_state else None}
        self.obs, self.last_info = self.env.reset(seed=seed, options=opts)
        self._after_reset_hook()

    # ------------------------------------------------------------------ simulated e-stop / safety envelope
    def cmd_estop(self, reason: str = "operator"):
        """Engage the simulated e-stop: stepping halts after the current control step; joint position targets are
        frozen at their present values so the arm holds; the gripper command is left as is. Modes are not changed
        so the operator can see what was running; nothing can start until the e-stop is reset."""
        sim = self.env.sim
        self.estop = True
        sim.q_target = sim.arm_qpos().copy()
        sim.data.ctrl[sim.arm_act_ids] = sim.q_target
        sim.target_pos = sim.ee_pos().copy(); sim.target_yaw = sim.ee_yaw()
        if self.assist is not None:
            self._finish_assist("e-stop")
        ev = {"t": time.time(), "iso": time.strftime("%H:%M:%S"), "event": "engaged", "reason": reason, "mode": self.mode, "step": self.env.t}
        self.estop_events.append(ev)
        self._publish({"type": "estop", **ev})
        return self.safety_status()

    def cmd_estop_reset(self):
        self.estop = False
        ev = {"t": time.time(), "iso": time.strftime("%H:%M:%S"), "event": "reset", "mode": self.mode, "step": self.env.t}
        self.estop_events.append(ev)
        self._publish({"type": "estop", **ev})
        return self.safety_status()

    def _require_not_estopped(self):
        if self.estop:
            raise RuntimeError("E-STOP is engaged - reset it before starting any motion")

    def safety_status(self) -> dict:
        """Live safety-envelope figures from the simulator: distance of the TCP to the workspace box faces, margin to
        joint limits, joint-velocity utilisation vs. the UR10e limits, and whether an arm/static collision is active now."""
        sim = self.env.sim
        ee = sim.ee_pos(); q = sim.arm_qpos(); qd = np.abs(sim.arm_qvel())
        ws_margin = float(min(np.min(ee - sim.ws_lo), np.min(sim.ws_hi - ee)))
        jl_margin = float(min(np.min(q - sim.joint_lo), np.min(sim.joint_hi - q)))
        vel_util = float(np.max(qd / sim.vel_limits)) if len(qd) else 0.0
        cs = sim.contacts(self.env.target)
        st = {"estop": self.estop, "workspace_margin_m": round(ws_margin, 3), "joint_limit_margin_rad": round(jl_margin, 3),
              "velocity_utilization": round(vel_util, 3), "collision_now": bool(cs.arm_collision),
              "envelope_ok": bool(ws_margin > -0.01 and jl_margin > 0.0 and vel_util <= 1.05 and not cs.arm_collision),
              "limits": {"workspace_x": [float(sim.ws_lo[0]), float(sim.ws_hi[0])], "workspace_y": [float(sim.ws_lo[1]), float(sim.ws_hi[1])], "workspace_z": [float(sim.ws_lo[2]), float(sim.ws_hi[2])],
                         "joint_velocity_rad_s": sim.vel_limits.round(3).tolist(), "max_tcp_step_m": sim.max_dpos},
              "events": self.estop_events[-5:]}
        self.safety_last = st
        return st

    # ------------------------------------------------------------------ evaluation reels (failure review)
    def _reel_capture(self, action, reward, info):
        sim = self.env.sim
        t = self.env.t
        self._reel_trace.append({"t": t, "action": np.asarray(action).round(3).tolist(), "reward": round(float(reward), 3),
                                 "ee": sim.ee_pos().round(3).tolist(), "grasped": bool(info.get("grasped_now")), "lifted": bool(info.get("lifted_now")),
                                 "target": info.get("target"), "placed": info.get("placed_count"),
                                 "objects": {o: sim.obj_pos(o).round(3).tolist() for o in self.env.objects}})
        if (t - 1) % 10 == 0 or t == 1:
            self._reel_frames.append((t, sim.render_camera(self.view_cam, 320, 240), self.obs[IMAGE_KEY].copy() if self.obs is not None and IMAGE_KEY in self.obs else None))

    def _save_reel(self, result: dict):
        """Persist frames + trace of an evaluation episode: every failed episode, plus the first success as a reference."""
        frames, trace = self._reel_frames, self._reel_trace
        self._reel_frames, self._reel_trace = [], []
        st = self.eval_state or {}
        eval_id = st.get("id") or "eval"
        keep = (not result["success"]) or not st.get("_success_saved")
        if not frames or not keep:
            return
        try:
            # add the final frame
            frames.append((self.env.t, self.env.sim.render_camera(self.view_cam, 320, 240), self.obs[IMAGE_KEY].copy() if self.obs is not None and IMAGE_KEY in self.obs else None))
            d = self.reels_dir / eval_id / (f"ep{result['episode']:03d}_" + ("success" if result["success"] else "fail"))
            d.mkdir(parents=True, exist_ok=True)
            pick = frames if len(frames) <= 12 else [frames[int(round(i * (len(frames) - 1) / 11))] for i in range(12)]
            strip = np.concatenate([f[1] for f in pick], axis=1)
            cv2.imwrite(str(d / "filmstrip.jpg"), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            for name, fr in (("first", frames[0]), ("middle", frames[len(frames) // 2]), ("last", frames[-1])):
                cv2.imwrite(str(d / f"{name}.jpg"), cv2.cvtColor(fr[1], cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if fr[2] is not None:
                    cv2.imwrite(str(d / f"{name}_overhead.png"), cv2.cvtColor(cv2.resize(fr[2], (128, 128), interpolation=cv2.INTER_NEAREST), cv2.COLOR_RGB2BGR))
            failure = None
            if not result["success"]:
                failure = ("dropped part" if result["dropped"] else "wrong bin" if result["wrong_bin"] else "sim unstable" if result["unstable"]
                           else "timed out holding a part" if trace and trace[-1]["grasped"] else "timed out: never grasped" if not result["grasp_success"]
                           else "timed out after grasp (lost or never placed)")
            meta = {**{k: v for k, v in result.items() if k != "object_configuration"}, "object_configuration": result.get("object_configuration"),
                    "frame_steps": [f[0] for f in pick], "failure_reason": failure, "eval_id": eval_id, "label": st.get("label"),
                    "checkpoint": st.get("checkpoint"), "saved_iso": time.strftime("%Y-%m-%dT%H:%M:%S")}
            (d / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
            (d / "trace.json").write_text(json.dumps(trace, default=str))
            if result["success"]:
                st["_success_saved"] = True
        except Exception as e:
            self.error = f"reel save: {e}"

    def _after_reset_hook(self):
        """Called after every env reset in the sim thread (Phase 3 re-applies authored clutter here)."""
        cfg = self.env.sim.episode_config
        max_off = max((abs(v) for o in cfg.bin_offsets.values() for v in o), default=0.0)
        self.last_reset = {"t": time.time(), "episode": self.episode_index, "reset_count": int(getattr(self.env.sim, "reset_count", 0)),
                           "bins": "fixed" if cfg.bins_fixed else f"randomized ±{max_off * 100:.1f} cm", "bins_fixed": bool(cfg.bins_fixed),
                           "level": cfg.level, "mode": self.mode}
        self._reel_frames, self._reel_trace = [], []
        self._ep_effort = []
        self._ep_false_pick = False
        hook = getattr(self, "scene_after_reset", None)
        if hook is not None:
            try:
                hook()
            except Exception as e:
                self.error = f"scene hook: {e}"

    # ------------------------------------------------------------------ batch evaluations
    def cmd_start_batch(self, kind: str, items: list[dict], episodes: int, seed: int | None = None, randomization_level=None):
        """Run several evaluations back to back on the live sim. items: [{label, fixed_poses?, randomization_level?, checkpoint?}].
        kind: scorecard | sweep | compare. Results are persisted to logs/<kind>_latest.json when the batch completes."""
        self._require_not_estopped()
        if self.policy is None and not any(it.get("checkpoint") for it in items):
            raise RuntimeError("no checkpoint loaded")
        if self.eval_state is not None:
            self._abort_evaluation()
        self.batch = {"kind": kind, "items": list(items), "i": 0, "results": [], "episodes": int(episodes), "seed": seed,
                      "randomization_level": randomization_level, "started": time.time(), "id": time.strftime("%Y%m%d_%H%M%S"),
                      "restore_checkpoint": self.policy.name if self.policy else None, "status": "running"}
        self._batch_next()
        return self.batch_status()

    def _batch_next(self):
        b = self.batch
        it = b["items"][b["i"]]
        if it.get("checkpoint"):
            self.cmd_load_checkpoint(it["checkpoint"])
        level = it.get("randomization_level", b["randomization_level"])
        self.cmd_start_evaluation(b["episodes"], randomization_level=level, seed=b["seed"], fixed_poses=it.get("fixed_poses"),
                                  label=it.get("label"), batch_id=f"{b['kind']}_{b['id']}")

    def _batch_on_eval_done(self, summary: dict):
        b = self.batch
        if b is None:
            return
        it = b["items"][b["i"]]
        row = {k: v for k, v in summary.items() if k != "episodes"}
        row.update({"label": it.get("label"), "item": {k: v for k, v in it.items() if k != "fixed_poses"}, "has_fixed_poses": bool(it.get("fixed_poses"))})
        b["results"].append(row)
        b["i"] += 1
        if b["i"] < len(b["items"]):
            self._batch_next()
            return
        b["status"] = "completed"; b["finished"] = time.time(); b["duration_s"] = round(b["finished"] - b["started"], 1)
        out = {k: v for k, v in b.items() if k != "items"}
        out["items"] = [{k: v for k, v in it.items() if k != "fixed_poses"} for it in b["items"]]
        out["finished_iso"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        out_dir = resolve_path("logs"); out_dir.mkdir(exist_ok=True)
        (out_dir / f"{b['kind']}_{b['id']}.json").write_text(json.dumps(out, indent=1, default=str))
        (out_dir / f"{b['kind']}_latest.json").write_text(json.dumps(out, indent=1, default=str))
        if b.get("restore_checkpoint") and self.policy and self.policy.name != b["restore_checkpoint"]:
            try:
                self.cmd_load_checkpoint(b["restore_checkpoint"])
            except Exception:
                pass
        self.batch = None
        self._publish({"type": "batch_complete", "kind": b["kind"], "summary": out})

    def cmd_abort_batch(self):
        if self.batch is not None:
            self.batch["status"] = "aborted"
            self.batch = None
        if self.eval_state is not None:
            self._abort_evaluation()
        return {"aborted": True}

    def batch_status(self) -> dict | None:
        b = self.batch
        if b is None:
            return None
        return {"kind": b["kind"], "id": b["id"], "i": b["i"], "n": len(b["items"]), "status": b["status"],
                "current": b["items"][b["i"]].get("label") if b["i"] < len(b["items"]) else None,
                "results": b["results"], "episodes": b["episodes"]}

    # ------------------------------------------------------------------ state snapshot (exact replay)
    def _snapshot_state(self) -> dict:
        sim, env = self.env.sim, self.env
        return {"qpos": sim.data.qpos.copy().tolist(), "qvel": sim.data.qvel.copy().tolist(), "ctrl": sim.data.ctrl.copy().tolist(),
                "target_pos": sim.target_pos.tolist(), "target_yaw": float(sim.target_yaw), "q_target": sim.q_target.tolist(),
                "grip_cmd": float(sim.grip_cmd), "seed": int(env._episode_seed), "episode_config": sim.episode_config.to_dict(),
                "target": env.target, "time": float(sim.data.time)}

    def _restore_state(self, state: dict | None, seed: int | None, episode_config: dict | None):
        """Reset the env to a recorded episode's start: seed + fixed poses reproduce the randomisation (sizes,
        friction, bins, light); the exact qpos/qvel/controller snapshot (when stored) makes it bit-exact."""
        cfg = episode_config or (state or {}).get("episode_config") or {}
        opts = {"fixed_poses": cfg.get("object_poses"), "randomization_level": cfg.get("level")}
        self.obs, self.last_info = self.env.reset(seed=seed, options=opts)
        sim, env = self.env.sim, self.env
        if state and state.get("qpos") and len(state["qpos"]) == sim.model.nq:
            import mujoco
            sim.data.qpos[:] = np.asarray(state["qpos"]); sim.data.qvel[:] = np.asarray(state["qvel"]); sim.data.ctrl[:] = np.asarray(state["ctrl"])
            sim.target_pos = np.asarray(state["target_pos"], dtype=float); sim.target_yaw = float(state["target_yaw"])
            sim.q_target = np.asarray(state["q_target"], dtype=float); sim.grip_cmd = float(state["grip_cmd"])
            mujoco.mj_forward(sim.model, sim.data)
            if state.get("target") in env.objects:
                env.target = state["target"]; sim.track_target = env.target
            self.obs = env._get_obs()
        self.last_action[:] = 0
        self.last_reward = 0.0

    # ------------------------------------------------------------------ click-to-pick assisted teleop
    def cmd_pick_at(self, u: float, v: float, bin_override: str | None = None):
        """Resolve a viewport click (u, v in [0,1]) with a segmentation pass of the viewport camera, then hand the
        clicked object to the scripted controller (or re-route the active pick when a bin is clicked)."""
        if self.mode != "teleop":
            raise RuntimeError("click-to-pick is only available in teleop mode")
        hit = self.env.sim.segment_at(self.view_cam, self.view_w, self.view_h, float(u), float(v))
        world = self.env.sim.pixel_to_table(self.view_cam, self.view_w, self.view_h, float(u), float(v))
        res = {"hit": hit, "world_on_table": world.round(3).tolist() if world is not None else None, "assist": None}
        if hit["kind"] == "object":
            self._start_assist(hit["name"], bin_override)
        elif hit["kind"] == "bin" and self.assist is not None and not self.assist.get("finished"):
            self.assist["bin"] = hit["name"]; self.assist["controller"].bin_override[self.assist["object"]] = hit["name"]
            self.assist["events"].append({"t": self.env.t, "event": f"bin override -> {hit['name']}"})
        res["assist"] = self.assist_status()
        return res

    def _start_assist(self, obj: str, bin_name: str | None = None):
        from training.demos.scripted_controller import ScriptedSortingController
        env = self.env
        if env.stage[obj].placed:
            raise RuntimeError(f"{obj} is already placed")
        bin_name = bin_name or env.bin_mapping[obj]
        if bin_name not in env.sim.bins:
            raise ValueError(f"unknown bin {bin_name}")
        env.target = obj; env.sim.track_target = obj; env.reward_fn.reset(); env._grasp_streak = 0
        ctrl = ScriptedSortingController(env, fixed_target=obj, bin_override={obj: bin_name})
        self.assist = {"object": obj, "bin": bin_name, "controller": ctrl, "started_step": env.t, "finished": False,
                       "result": None, "events": [{"t": env.t, "event": f"assist start: {obj} -> {bin_name}"}]}
        if self.dataset_recorder is not None and self.recording_type == "teleop_demo":
            self.dataset_recorder.retag_current("assisted_demo")   # a human chose the target; the controller executed
        return self.assist_status()

    def _assist_action(self):
        a = self.assist["controller"].act()
        if self.assist["controller"].phase == "done" and not self.assist["finished"]:
            self._finish_assist("controller done")
            return None
        return a

    def _assist_after_step(self, info: dict, ended: bool):
        obj = self.assist["object"]; st = self.env.stage[obj]
        if st.placed:
            self._finish_assist("placed in " + (self.env.sim.which_bin(obj) or "?"))
        elif st.dropped or st.wrong_bin or ended:
            self._finish_assist("dropped" if st.dropped else "wrong bin" if st.wrong_bin else "episode ended")

    def _finish_assist(self, result: str):
        if self.assist is None:
            return
        self.assist["finished"] = True; self.assist["result"] = result
        self.assist["events"].append({"t": self.env.t, "event": result})
        self._publish({"type": "assist_done", "assist": self.assist_status()})
        self.last_assist = self.assist_status()
        self.assist = None
        self.teleop_gripper = 1.0 if self.env.sim.grip_cmd > 0.5 * self.env.sim.grip_stroke else -1.0

    def cmd_assist_cancel(self):
        if self.assist is None:
            return {"cancelled": False}
        self._finish_assist("cancelled by user")
        return {"cancelled": True}

    def assist_status(self) -> dict | None:
        a = self.assist
        if a is None:
            return getattr(self, "last_assist", None)
        c = a["controller"]
        return {"active": not a["finished"], "object": a["object"], "bin": a["bin"], "phase": c.phase, "retries": c.retries,
                "started_step": a["started_step"], "result": a["result"], "events": a["events"][-6:]}

    # ------------------------------------------------------------------ replay of a recorded episode
    def cmd_replay_start(self, name: str, autoplay: bool = True):
        self._require_not_estopped()
        from lerobot_adapter.demo_recorder import load_demo_episode
        demo_dir = None
        for root in (self.demos_dir, resolve_path("datasets/live_recordings")):
            if (root / name / "demo.json").exists():
                demo_dir = root / name
        if demo_dir is None:
            raise FileNotFoundError(f"recorded episode not found: {name}")
        ep = load_demo_episode(demo_dir)
        side = ep["sidecar"]
        if self.mode == "evaluate" and self.eval_state is not None:
            self._abort_evaluation()
        if self.recording_type in ("teleop_demo", "assisted_demo", "scripted_demo"):
            self._stop_recording()
        self.assist = None
        self.mode = "replay"; self.paused = False
        self.replay = {"name": name, "episode_type": side.get("episode_type"), "actions": ep["actions"], "states": ep["states"],
                       "rewards": ep["rewards"], "n": int(len(ep["actions"])), "index": 0, "playing": bool(autoplay),
                       "initial_state": side.get("initial_state"), "seed": side.get("seed"), "episode_config": side.get("object_configuration"),
                       "exact": bool(side.get("initial_state")), "max_state_dev": 0.0, "reward_dev": 0.0, "outcome": side.get("outcome")}
        self._restore_state(self.replay["initial_state"], self.replay["seed"], self.replay["episode_config"])
        return self.replay_status()

    def _replay_step(self):
        rp = self.replay
        if rp is None or rp["index"] >= rp["n"]:
            if rp is not None:
                rp["playing"] = False
            return
        i = rp["index"]
        action = np.asarray(rp["actions"][i], dtype=np.float32)
        # fidelity: how far does the replayed state drift from the recorded one (recorded state = obs BEFORE this action)
        if self.obs is not None:
            # fidelity over the continuous entries only (joints, TCP, object/bin poses); the trailing target /
            # grasp flags are binary and would turn a one-step timing difference into a deviation of 1.0
            k = len(self.env.state_layout) - 10
            dev = float(np.max(np.abs(np.asarray(self.obs[STATE_KEY])[:k] - rp["states"][i][:k])))
            rp["max_state_dev"] = max(rp["max_state_dev"], dev)
        obs, reward, term, trunc, info = self.env.step(action)
        rp["reward_dev"] = max(rp["reward_dev"], abs(float(reward) - float(rp["rewards"][i])))
        self.obs, self.last_action, self.last_reward, self.last_info = obs, action, reward, info
        rp["index"] = i + 1
        if rp["index"] >= rp["n"] or term or trunc:
            rp["playing"] = False
            rp["index"] = min(rp["index"], rp["n"])
            self._publish({"type": "replay_done", "replay": self.replay_status()})

    def cmd_replay_seek(self, index: int):
        rp = self.replay
        if rp is None:
            raise RuntimeError("no replay loaded")
        index = int(np.clip(index, 0, rp["n"]))
        was_playing = rp["playing"]
        rp["playing"] = False
        self._restore_state(rp["initial_state"], rp["seed"], rp["episode_config"])
        rp["index"] = 0; rp["max_state_dev"] = 0.0; rp["reward_dev"] = 0.0
        for _ in range(index):
            self._replay_step()
        rp["playing"] = was_playing and rp["index"] < rp["n"]
        return self.replay_status()

    def cmd_replay_set_playing(self, playing: bool):
        if self.replay is None:
            raise RuntimeError("no replay loaded")
        if playing and self.replay["index"] >= self.replay["n"]:
            self.cmd_replay_seek(0)
        self.replay["playing"] = bool(playing)
        return self.replay_status()

    def cmd_replay_stop(self):
        self.replay = None
        if self.mode == "replay":
            self.mode = "idle"
        return {"mode": self.mode}

    def replay_status(self) -> dict | None:
        rp = self.replay
        if rp is None:
            return None
        return {"name": rp["name"], "episode_type": rp["episode_type"], "index": rp["index"], "n": rp["n"], "playing": rp["playing"],
                "exact_initial_state": rp["exact"], "max_state_deviation": round(rp["max_state_dev"], 4), "max_reward_deviation": round(rp["reward_dev"], 4),
                "recorded_outcome": rp["outcome"]}

    # ------------------------------------------------------------------ dataset recording (live sim)
    def _default_recording_type(self) -> str:
        return {"teleop": "teleop_demo", "scripted_demo": "scripted_demo", "inference": "policy_rollout",
                "evaluate": "policy_rollout", "random": "random_policy"}.get(self.mode, "manual")

    def cmd_record_start(self, name: str | None = None, episode_type: str | None = None, reset: bool = True):
        """Start recording (observation, action, reward, done) per step into LeRobotDataset episodes.
        Demonstration types (teleop_demo / scripted_demo) go to datasets/demos/, one directory per
        episode (so single demos can be discarded); other recordings go to datasets/live_recordings/."""
        from lerobot_adapter.demo_recorder import DemoRecorder, DEMO_TYPES
        if self.dataset_recorder is not None:
            raise RuntimeError("already recording")
        episode_type = episode_type or self._default_recording_type()
        if episode_type not in DEMO_TYPES:
            raise ValueError(f"unknown episode_type {episode_type}; options: {DEMO_TYPES}")
        is_demo = episode_type in ("teleop_demo", "assisted_demo", "scripted_demo")
        root = self.demos_dir if is_demo else resolve_path("datasets/live_recordings")
        extra = {"controller": ("human_teleop" if episode_type == "teleop_demo" else "scripted_ground_truth" if episode_type == "scripted_demo"
                                else (self.policy.name if self.policy and episode_type == "policy_rollout" else self.mode)),
                 "checkpoint": self.policy.name if (self.policy and episode_type == "policy_rollout") else None,
                 "session": name}
        self.dataset_recorder = DemoRecorder(root=root, episode_type=episode_type, state_layout=self.env.state_layout,
                                             fps=int(round(1.0 / self.env.sim.control_dt)),
                                             image_shape=(self.env.img_h, self.env.img_w, 3) if self.env.include_image else None,
                                             robot_type=self.env.sim.robot_name, extra_meta=extra)
        self.recording_type = episode_type
        self._rec_started = time.time()
        if reset:
            # a demonstration should be a complete episode: start from a fresh randomized scene
            self.obs, self.last_info = self.env.reset()
            self._after_reset_hook()
            self.last_action[:] = 0
            self.last_reward = 0.0
            if self.scripted is not None:
                self.scripted.reset()
        return {"recording": True, "episode_type": episode_type, "path": str(root)}

    def _stop_recording(self) -> dict:
        rec = self.dataset_recorder
        if rec is None:
            return {"recording": False}
        self.dataset_recorder = None
        etype = self.recording_type
        self.recording_type = None
        # a partially recorded episode is kept but flagged (the user can discard it from the Demonstrations panel)
        side = rec.end_episode(self._partial_result(), aborted=True) if rec.frames_in_episode > 0 else None
        if side is not None:
            self._publish({"type": "demo_recorded", "demo": side})
        return {"recording": False, "episode_type": etype, "episodes": rec.episodes,
                "seconds": round(time.time() - getattr(self, "_rec_started", time.time()), 1)}

    def _partial_result(self) -> dict:
        info = self.last_info or {}
        return {"success": bool(info.get("is_success")), "placed_count": info.get("placed_count"),
                "reward": round(float(info.get("episode_reward", 0.0)), 3), "length": info.get("step"),
                "grasp_success": bool(info.get("grasp_achieved")), "dropped": bool(info.get("dropped")),
                "wrong_bin": bool(info.get("wrong_bin")), "truncated": False, "seed": info.get("seed"),
                "object_configuration": self.env.sim.episode_config.to_dict()}

    def cmd_record_stop(self):
        return self._stop_recording()

    def recording_status(self) -> dict:
        rec = self.dataset_recorder
        return {"recording": rec is not None, "episode_type": self.recording_type,
                "path": str(rec.root) if rec else None,
                "episodes": rec.episodes if rec else None, "frames_in_episode": rec.frames_in_episode if rec else None}

    # ------------------------------------------------------------------ teleop / scripted demos
    def cmd_teleop_start(self, reset: bool = True):
        self._require_not_estopped()
        if self.mode == "evaluate" and self.eval_state is not None:
            self._abort_evaluation()
        self.mode = "teleop"
        self.paused = False
        self.teleop_axes[:] = 0
        self.teleop_gripper = -1.0
        if reset:
            self.obs, self.last_info = self.env.reset()
            self._after_reset_hook()
            self.last_action[:] = 0
            self.last_reward = 0.0
        return {"mode": self.mode}

    def cmd_teleop_stop(self):
        if self.assist is not None:
            self._finish_assist("teleop stopped")
        if self.mode == "teleop":
            self.mode = "idle"
            self.teleop_axes[:] = 0
        if self.recording_type in ("teleop_demo", "assisted_demo"):
            self._stop_recording()
        return {"mode": self.mode}

    def cmd_scripted_demos(self, episodes: int = 5, noise_std: float = 0.0, record: bool = True):
        """Run the hand-coded ground-truth controller for N episodes on the live sim (visible in the
        dashboard) and record them as `scripted_demo` episodes."""
        self._require_not_estopped()
        from training.demos.scripted_controller import ScriptedSortingController
        if self.mode == "evaluate" and self.eval_state is not None:
            self._abort_evaluation()
        if self.dataset_recorder is not None and self.recording_type != "scripted_demo":
            self._stop_recording()
        self.scripted_noise = float(noise_std)
        self.scripted = ScriptedSortingController(self.env, noise_std=self.scripted_noise)
        self.scripted_remaining = int(episodes)
        self.mode = "scripted_demo"
        self.paused = False
        self.obs, self.last_info = self.env.reset()
        self._after_reset_hook()
        self.scripted.reset()
        if record and self.dataset_recorder is None:
            self.cmd_record_start(episode_type="scripted_demo", reset=False)
        return {"mode": self.mode, "episodes": int(episodes), "recording": self.dataset_recorder is not None}

    def cmd_set_frustum(self, enabled: bool):
        self.show_frustum = bool(enabled)
        return {"show_frustum": self.show_frustum}

    def cmd_set_wrist(self, enabled: bool):
        self.show_wrist = bool(enabled)
        return {"show_wrist": self.show_wrist}

    # ------------------------------------------------------------------ commands (run in sim thread)
    def cmd_reset(self, seed=None, randomize=True, randomization_level=None, fixed_poses=None):
        opts = {"randomize": randomize, "randomization_level": randomization_level, "fixed_poses": fixed_poses}
        self.obs, self.last_info = self.env.reset(seed=seed, options=opts)
        self._after_reset_hook()
        self.last_action[:] = 0
        self.last_reward = 0.0
        return self.state_payload()

    def cmd_set_mode(self, mode: str):
        self._require_not_estopped()
        if mode not in ("idle", "random", "inference", "teleop"):
            raise ValueError(f"unknown mode {mode}")
        if self.mode == "replay":
            self.replay = None
        if mode == "inference" and self.policy is None:
            raise RuntimeError("no checkpoint loaded")
        if self.mode == "evaluate" and self.eval_state is not None:
            self._abort_evaluation()
        if self.mode in ("teleop", "scripted_demo") and mode != self.mode:
            self.scripted = None
            self.scripted_remaining = 0
            if self.recording_type in ("teleop_demo", "assisted_demo", "scripted_demo"):
                self._stop_recording()
        if mode == "teleop":
            return self.cmd_teleop_start(reset=True)
        self.mode = mode
        return {"mode": self.mode}

    def cmd_set_paused(self, paused: bool):
        self.paused = bool(paused)
        return {"paused": self.paused}

    def cmd_step(self, steps: int = 1):
        if self.mode == "idle":
            # manual step uses the loaded policy if any, else zero action
            for _ in range(steps):
                action = self.policy.select_action(self.obs, self.deterministic) if self.policy else np.zeros(7, np.float32)
                obs, r, term, trunc, info = self.env.step(action)
                self.obs, self.last_action, self.last_reward, self.last_info = obs, action, r, info
                self.total_steps += 1
                if term or trunc:
                    self._on_episode_end(info, term, trunc)
                    break
        else:
            for _ in range(steps):
                self._step_once()
        return self.state_payload()

    def cmd_load_checkpoint(self, checkpoint: str | None):
        path = resolve_checkpoint(checkpoint, self.train_cfg.get("checkpoint_dir", "checkpoints"))
        if path is None:
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
        self.policy = LoadedPolicy(path, device=self.device)
        self.inference_stats = {"episodes": 0, "successes": 0, "placements": 0, "grasps": 0, "parts_placed": 0, "sim_seconds": 0.0, "collision_steps": 0, "episodes_with_collision": 0, "control_effort": 0.0}
        return self.policy.info()

    def cmd_start_evaluation(self, episodes: int, randomization_level=None, seed=None, fixed_poses=None, label=None, batch_id=None):
        self._require_not_estopped()
        if self.policy is None:
            raise RuntimeError("no checkpoint loaded")
        if self.mode in ("teleop", "replay", "scripted_demo"):
            self.assist = None; self.replay = None; self.scripted = None
        self.eval_results = []
        eid = (batch_id + "_" if batch_id else "") + time.strftime("%Y%m%d_%H%M%S") + (f"_{label}" if label else "")
        self.eval_state = {"episodes": int(episodes), "completed": 0, "started": time.time(),
                           "checkpoint": self.policy.name, "randomization_level": randomization_level,
                           "seed": seed, "status": "running", "fixed_poses": fixed_poses, "label": label,
                           "id": "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in eid)}
        self.mode = "evaluate"
        self.paused = False
        self.episode_index = 0
        self.obs, self.last_info = self.env.reset(seed=seed, options={"randomization_level": randomization_level, "fixed_poses": fixed_poses})
        self._after_reset_hook()
        return self.eval_progress()

    def _abort_evaluation(self):
        if self.eval_state is not None:
            self.eval_state["status"] = "aborted"
            self._finish_evaluation(aborted=True)

    def _finish_evaluation(self, aborted: bool = False):
        st = self.eval_state or {}
        res = self.eval_results
        n = len(res)
        summary = {
            "status": "aborted" if aborted else "completed",
            "checkpoint": st.get("checkpoint"),
            "policy": self.policy.info() if self.policy else None,
            "episodes_requested": st.get("episodes"),
            "episodes_run": n,
            "successes": sum(r["success"] for r in res),
            "success_rate": (sum(r["success"] for r in res) / n) if n else None,
            "grasp_rate": (sum(r["grasp_success"] for r in res) / n) if n else None,
            "placement_rate": (sum(r["placement_success"] for r in res) / n) if n else None,
            "average_reward": (sum(r["reward"] for r in res) / n) if n else None,
            "average_length": (sum(r["length"] for r in res) / n) if n else None,
            "collisions_total": sum(r["collisions"] or 0 for r in res),
            "episodes_with_collision": sum(1 for r in res if (r["collisions"] or 0) > 0),
            "control_effort_mean": (sum(r.get("control_effort", 0.0) for r in res) / n) if n else None,
            "control_effort_penalty_total": sum(r.get("control_effort_penalty_total", 0.0) for r in res),
            "collision_penalty_total": sum(r.get("collision_penalty_total", 0.0) for r in res),
            "parts_placed_total": sum(r.get("placed_count") or 0 for r in res),
            "sim_seconds_total": round(sum(r.get("sim_seconds", 0.0) for r in res), 1),
            "parts_per_minute": (round(60.0 * sum(r.get("placed_count") or 0 for r in res) / max(1e-6, sum(r.get("sim_seconds", 0.0) for r in res)), 2) if n else None),
            "false_picks": sum(1 for r in res if r.get("false_pick")),
            "unstable_episodes": sum(1 for r in res if r.get("unstable")),
            "drops": sum(r["dropped"] for r in res),
            "wrong_bin": sum(r["wrong_bin"] for r in res),
            "randomization_level": st.get("randomization_level"),
            "label": st.get("label"), "id": st.get("id"), "fixed_poses": st.get("fixed_poses"),
            "seed": st.get("seed"),
            "duration_s": round(time.time() - st.get("started", time.time()), 1),
            "finished_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "episodes": res,
        }
        self.last_eval_summary = summary
        out_dir = resolve_path("logs")
        out_dir.mkdir(exist_ok=True)
        fname = out_dir / f"evaluation_{time.strftime('%Y%m%d_%H%M%S')}.json"
        fname.write_text(json.dumps(summary, indent=2, default=str))
        (out_dir / "evaluation_latest.json").write_text(json.dumps(summary, indent=2, default=str))
        self.eval_state = None
        self.mode = "inference" if not aborted else self.mode
        self._publish({"type": "evaluation_complete", "summary": {k: v for k, v in summary.items() if k != "episodes"}})
        if not aborted and self.batch is not None:
            self._batch_on_eval_done(summary)

    def eval_progress(self) -> dict:
        st = self.eval_state
        if st is None:
            return {"running": False, "last": {k: v for k, v in (self.last_eval_summary or {}).items() if k != "episodes"} or None}
        res = self.eval_results
        n = len(res)
        return {"running": True, "episodes": st["episodes"], "completed": n, "label": st.get("label"), "id": st.get("id"),
                "successes": sum(r["success"] for r in res),
                "success_rate": (sum(r["success"] for r in res) / n) if n else None,
                "average_reward": (sum(r["reward"] for r in res) / n) if n else None,
                "average_length": (sum(r["length"] for r in res) / n) if n else None,
                "checkpoint": st["checkpoint"], "randomization_level": st.get("randomization_level")}

    # ------------------------------------------------------------------ payloads
    def state_payload(self) -> dict:
        env = self.env
        sim = env.sim
        info = self.last_info or {}
        st = self.inference_stats
        n_eps = st["episodes"]
        step_fps = (len(self.step_times) / sum(self.step_times)) if len(self.step_times) >= 2 and sum(self.step_times) > 0 else None
        frame_fps = None
        if len(self.frame_times) >= 2:
            span = self.frame_times[-1] - self.frame_times[0]
            frame_fps = (len(self.frame_times) - 1) / span if span > 0 else None
        target = env.target
        state_vec = self.obs[STATE_KEY] if self.obs is not None else None
        return {
            "mode": self.mode,
            "paused": self.paused,
            "speed": self.speed,
            "sim_time": round(float(sim.data.time), 3),
            "episode": self.episode_index,
            "step": env.t,
            "max_steps": env.max_steps,
            "device": device_label(self.device, self.device_name),
            "robot": sim.robot_name,
            "joint_positions": sim.arm_qpos().round(4).tolist(),
            "joint_velocities": sim.arm_qvel().round(4).tolist(),
            "ee_position": sim.ee_pos().round(4).tolist(),
            "ee_yaw": round(sim.ee_yaw(), 4),
            "ee_target": sim.target_pos.round(4).tolist(),
            "gripper_opening": round(sim.gripper_opening(), 4),
            "gripper_cmd_closed": round(sim.grip_cmd / sim.grip_stroke, 3),
            "objects": {o: {"position": sim.obj_pos(o).round(4).tolist(), "yaw": round(sim.obj_yaw(o), 4),
                            "in_bin": sim.which_bin(o), "placed": env.stage[o].placed if env.stage else False,
                            "grasped": env.stage[o].grasped if env.stage else False,
                            "lifted": env.stage[o].lifted if env.stage else False,
                            "target_bin": env.bin_mapping[o]} for o in env.objects},
            "bins": {b: sim.bin_pos(b).round(4).tolist() for b in sim.bins},
            "target": target,
            "target_bin": env.bin_mapping.get(target) if target else None,
            "present_count": sum(1 for o in env.objects if sim.present(o)),
            "grasped": bool(info.get("grasped_now", False)),
            "lifted": bool(info.get("lifted_now", False)),
            "placed_count": info.get("placed_count", 0),
            "collisions": env.collisions,
            "action": {"names": ACTION_NAMES, "values": np.asarray(self.last_action).round(4).tolist()},
            "reward": round(float(self.last_reward), 4),
            "episode_reward": round(float(env.ep_reward), 3),
            "reward_components": {k: round(v, 4) for k, v in (env.last_components or {}).items()},
            "state_vector": state_vec.round(4).tolist() if state_vec is not None else None,
            "state_layout": env.state_layout,
            "episode_config": sim.episode_config.to_dict(),
            "policy": self.policy.info() if self.policy else None,
            "inference": {
                "episodes": n_eps, "successes": st["successes"],
                "success_rate": (st["successes"] / n_eps) if n_eps else None,
                "placement_rate": (st["placements"] / n_eps) if n_eps else None,
                "grasp_rate": (st["grasps"] / n_eps) if n_eps else None,
                "action_latency_ms": (round(self.policy.latency_ms, 3) if self.policy and self.policy.n_calls else None),
                "parts_per_minute": (round(60.0 * st["parts_placed"] / st["sim_seconds"], 2) if st["sim_seconds"] > 0 else None),
                "parts_placed": st["parts_placed"], "sim_seconds": round(st["sim_seconds"], 1),
                "collision_steps": st["collision_steps"], "episodes_with_collision": st["episodes_with_collision"],
                "control_effort_mean": (round(st["control_effort"] / n_eps, 2) if n_eps else None),
                "episode_control_effort": round(float(np.sum(self._ep_effort)), 2) if len(self._ep_effort) else 0.0,
                "policy_calls": self.policy.n_calls if self.policy else 0,
                "control_fps": round(step_fps, 2) if step_fps else None,
                "stream_fps": round(frame_fps, 1) if frame_fps else None,
                "deterministic": self.deterministic,
            },
            "evaluation": self.eval_progress(),
            "batch": self.batch_status(),
            "recording": self.recording_status(),
            "teleop": {"active": self.mode == "teleop", "clients": self.teleop_clients,
                       "axes": np.asarray(self.teleop_axes).round(2).tolist(), "gripper_closed": self.teleop_gripper > 0,
                       "input_fresh": (time.time() - self.teleop_last_msg) < self.teleop_timeout_s},
            "scripted": ({"remaining": self.scripted_remaining, **(self.scripted.info() if self.scripted else {})}
                         if self.mode == "scripted_demo" else None),
            "camera": self.camera_info,
            "show_frustum": self.show_frustum,
            "assist": self.assist_status(),
            "replay": self.replay_status(),
            "safety": self.safety_status(),
            "last_reset": getattr(self, "last_reset", None),
            "bins_fixed": bool(sim.episode_config.bins_fixed),
            "show_wrist": bool(getattr(self, "show_wrist", False)),
            "scene": ({"variants": dict(sim.active_body), "layout": list(self.scene_builder.layout),
                       "reject_on_table": any(l["part"] == "reject_cap" for l in self.scene_builder.layout),
                       "false_pick_this_episode": bool(self._ep_false_pick)} if self.scene_builder else None),
            "last_episode": self.episode_results[-1] if self.episode_results else None,
            "total_steps": self.total_steps,
            "error": self.error,
        }

    def _make_frame_message(self) -> dict:
        sim = self.env.sim
        view = sim.render_camera(self.view_cam, self.view_w, self.view_h, show_frustum=self.show_frustum)
        overhead = None
        if self.obs is not None and IMAGE_KEY in self.obs:
            # the actual policy input, upscaled for display (nearest neighbour so pixels stay honest)
            small = self.obs[IMAGE_KEY]
            overhead = cv2.resize(small, (self.prev_w, self.prev_h), interpolation=cv2.INTER_NEAREST)
        wrist = None
        if getattr(self, "show_wrist", False):
            try:
                wrist = jpeg_b64(sim.render_camera("gripper_cam", 192, 144), 70)   # display only; NOT a policy input
            except Exception:
                wrist = None
        msg = {"type": "frame", "t": time.time(), "frame": jpeg_b64(view, self.jpeg_q), "wrist": wrist,
               "overhead": jpeg_b64(overhead, 85) if overhead is not None else None,
               "overhead_size": [sim.obs_w, sim.obs_h],
               "state": self.state_payload()}
        return msg
