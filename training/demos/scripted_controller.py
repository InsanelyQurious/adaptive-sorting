"""Hand-coded pick-and-place controller used ONLY to generate `scripted_demo` episodes.

It reads ground-truth object / bin poses straight from the simulator (which the learned policy does
not get in this form) and emits actions in the SAME continuous action space the policy uses
(dX dY dZ dRx dRy dRz gripper in [-1, 1]); every action goes through env.step(), i.e. the same IK,
joint/velocity/workspace limits and physics. It is a pipeline validator for recording -> storage ->
behaviour cloning, not a demonstration of learned behaviour and not a human.
"""
from __future__ import annotations
import numpy as np

from simulation.environments.workcell_sim import wrap_angle


class ScriptedSortingController:
    PHASES = ("hover", "descend", "close", "lift", "transport", "lower", "open", "retreat", "done")

    def __init__(self, env, noise_std: float = 0.0, rng: np.random.Generator | None = None,
                 hover_height: float = 0.14, carry_height: float = 1.03, release_height: float = 0.97,
                 max_retries: int = 3, fixed_target: str | None = None, bin_override: dict | None = None):
        """fixed_target / bin_override are used by click-to-pick assisted teleop: pick exactly this object and
        (optionally) put it in a chosen bin instead of its mapped one. Same controller, same action space."""
        self.fixed_target = fixed_target
        self.bin_override = dict(bin_override or {})
        self.env = env.unwrapped
        self.sim = self.env.sim
        self.noise_std = float(noise_std)
        self.rng = rng or np.random.default_rng(0)
        self.hover_height = hover_height
        self.carry_height = carry_height
        self.release_height = release_height
        self.max_retries = max_retries
        self.reset()

    def reset(self):
        self.phase = "hover"
        self.phase_t = 0
        self.retries = 0
        self.target = None
        self.grip = -1.0

    # ------------------------------------------------------------------ helpers
    def _goal_action(self, goal_xyz: np.ndarray, yaw: float | None, grip: float, gain: float = 1.0) -> np.ndarray:
        s = self.sim
        a = np.zeros(7, dtype=np.float32)
        d = np.asarray(goal_xyz) - s.target_pos
        a[:3] = np.clip(gain * d / s.max_dpos, -1, 1)
        if yaw is not None:
            dy = wrap_angle(yaw - s.target_yaw)
            a[5] = np.clip(dy / max(s.max_drot[2], 1e-6), -1, 1)
        a[6] = grip
        return a

    def _obj_yaw_for_grasp(self, obj: str) -> float:
        """Gripper x-axis parallel to the part's long axis (pads close along y)."""
        y = self.sim.obj_yaw(obj)
        # the arm cannot spin forever: pick the equivalent yaw (mod pi) closest to the current tool yaw
        cur = self.sim.target_yaw
        cands = [wrap_angle(y + k * np.pi) for k in (-1, 0, 1)]
        return min(cands, key=lambda c: abs(wrap_angle(c - cur)))

    # ------------------------------------------------------------------ policy
    def target_bin(self, tgt: str) -> str:
        return self.bin_override.get(tgt) or self.env.bin_mapping[tgt]

    def act(self) -> np.ndarray:
        env, s = self.env, self.sim
        tgt = self.fixed_target or env.target
        if tgt is None or (self.fixed_target and (env.stage[tgt].placed or env.stage[tgt].dropped or env.stage[tgt].wrong_bin)):
            self.phase = "done"
            return np.array([0, 0, 0, 0, 0, 0, -1.0], dtype=np.float32)
        if tgt != self.target:            # new target (first step or after a placement)
            self.target = tgt
            self.phase, self.phase_t, self.retries = "hover", 0, 0
            self.grip = -1.0
        p = s.obj_pos(tgt)
        fl = env.stage[tgt]
        yaw = self._obj_yaw_for_grasp(tgt)
        self.phase_t += 1
        if self.phase == "hover":
            goal = np.array([p[0], p[1], p[2] + self.hover_height])
            a = self._goal_action(goal, yaw, -1.0)
            if np.linalg.norm(goal - s.ee_pos()) < 0.012 and abs(wrap_angle(yaw - s.target_yaw)) < 0.05 or self.phase_t > 60:
                self.phase, self.phase_t = "descend", 0
        elif self.phase == "descend":
            goal = np.array([p[0], p[1], s.ws_lo[2]])
            a = self._goal_action(goal, yaw, -1.0)
            if s.ee_pos()[2] < s.ws_lo[2] + 0.006 or self.phase_t > 30:
                self.phase, self.phase_t = "close", 0
        elif self.phase == "close":
            a = self._goal_action(s.target_pos, None, 1.0)
            if self.phase_t >= 4:
                self.phase, self.phase_t = "lift", 0
        elif self.phase == "lift":
            goal = np.array([s.target_pos[0], s.target_pos[1], self.carry_height])
            a = self._goal_action(goal, None, 1.0, gain=0.8)
            if self.phase_t >= 4 and not fl.grasped:
                # grasp failed: open and retry from hover (or give up and let the episode time out)
                self.retries += 1
                self.phase, self.phase_t = ("hover" if self.retries <= self.max_retries else "done"), 0
                a[6] = -1.0
            elif s.ee_pos()[2] > self.carry_height - 0.02 or self.phase_t > 20:
                self.phase, self.phase_t = "transport", 0
        elif self.phase == "transport":
            b = s.bin_pos(self.target_bin(tgt))
            goal = np.array([b[0], b[1], self.carry_height])
            a = self._goal_action(goal, None, 1.0)
            if not fl.grasped and self.phase_t > 2:
                self.retries += 1
                self.phase, self.phase_t = ("hover" if self.retries <= self.max_retries else "done"), 0
                a[6] = -1.0
            elif np.linalg.norm((goal - s.ee_pos())[:2]) < 0.015 or self.phase_t > 40:
                self.phase, self.phase_t = "lower", 0
        elif self.phase == "lower":
            b = s.bin_pos(self.target_bin(tgt))
            goal = np.array([b[0], b[1], self.release_height])
            a = self._goal_action(goal, None, 1.0)
            if s.ee_pos()[2] < self.release_height + 0.015 or self.phase_t > 12:
                self.phase, self.phase_t = "open", 0
        elif self.phase == "open":
            a = self._goal_action(s.target_pos, None, -1.0)
            if self.phase_t >= 3:
                self.phase, self.phase_t = "retreat", 0
        elif self.phase == "retreat":
            goal = np.array([s.target_pos[0], s.target_pos[1], self.carry_height + 0.03])
            a = self._goal_action(goal, None, -1.0)
            if self.phase_t >= 4:
                self.phase, self.phase_t = ("done" if (fl.placed or self.fixed_target) else "hover"), 0
        else:  # done
            a = np.array([0, 0, 0, 0, 0, 0, -1.0], dtype=np.float32)
        if self.noise_std > 0:
            a[:6] = np.clip(a[:6] + self.rng.normal(0, self.noise_std, 6), -1, 1)
        return a.astype(np.float32)

    def info(self) -> dict:
        return {"controller": "scripted_ground_truth", "phase": self.phase, "target": self.target, "retries": self.retries,
                "fixed_target": self.fixed_target, "bin": self.target_bin(self.target) if self.target else None}
