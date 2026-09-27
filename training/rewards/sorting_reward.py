"""Reward for adaptive vision-guided component sorting.

Design (revised after the placement diagnosis, see DEBUGGING.md):
  * Shaping terms are POTENTIAL-BASED: reward = weight * (phi_t - phi_{t-1}). To be exploit-free the
    potentials are pure functions of the simulator state and are charged back when the state regresses
    (e.g. a dropped object loses the lift/transport progress it earned). Each potential switches on
    exactly once per target (approach: from episode start until the first grasp; lift/transport: from
    the first grasp until placement) so no gate can be cycled to pump reward.
  * Task events (first grasp, first lift, placement, task completion) pay sparse one-time bonuses.
  * Penalties: collision, pushing the un-grasped part, opening the gripper away from the correct bin,
    wrong bin, drop, control effort, time, timeout.
Every component is computed each step from real simulator state and returned individually (for
logging / dashboard). Weights come from configs/environment.yaml -> reward.
"""
from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass


@dataclass
class StageFlags:
    grasped: bool = False
    ever_grasped: bool = False
    lifted: bool = False
    ever_lifted: bool = False
    placed: bool = False
    wrong_bin: bool = False
    dropped: bool = False
    collision: bool = False


class SortingReward:
    COMPONENTS = [
        "approach_reward", "align_reward", "grasp_reward", "grasp_bonus_once", "lift_reward",
        "lift_shaping", "transport_reward", "placement_reward", "task_complete_bonus",
        "collision_penalty", "wrong_bin_penalty", "drop_penalty", "control_effort_penalty",
        "time_penalty", "timeout_penalty", "push_penalty", "release_penalty",
    ]

    def __init__(self, weights: dict):
        self.w = {k: float(weights.get(k, 0.0)) for k in self.COMPONENTS}
        self.approach_scale = float(weights.get("approach_scale", 8.0))
        self.transport_scale = float(weights.get("transport_scale", 4.0))
        # pre-grasp waypoint: while the TCP is horizontally misaligned, the approach target sits
        # `hover_height` above the object; it descends onto the object as the xy error shrinks
        # below `hover_radius`. This teaches "come from above", not "shove from the side".
        self.hover_height = float(weights.get("approach_hover_height", 0.12))
        self.hover_radius = float(weights.get("approach_hover_radius", 0.04))
        # yaw alignment of the gripper with the part's long axis is folded INTO the approach
        # potential (phi_a *= 1 - align_fraction * (1 - |cos|)), so it is conservative too.
        self.align_fraction = float(np.clip(weights.get("approach_align_fraction", 0.0), 0.0, 0.9))
        self.push_speed_scale = float(weights.get("push_speed_scale", 0.2))
        self.lift_cap = float(weights.get("lift_cap", 0.15))
        self.reset()

    def reset(self):
        """Forget potentials (call on episode reset and when the target object changes)."""
        self._phi = {"approach": None, "lift": None, "transport": None}

    def _progress(self, key: str, phi: float | None, pay: bool = True) -> float:
        """Potential difference for `key`. None deactivates the term; the first value after a None
        is stored without paying (no jump on activation). pay=False updates the stored potential
        without paying the difference (used only on the placement step)."""
        prev = self._phi[key]
        self._phi[key] = phi
        if phi is None or prev is None or not pay:
            return 0.0
        return phi - prev

    def compute(self, *, tcp_pos: np.ndarray, gripper_x_axis: np.ndarray, obj_pos: np.ndarray,
                obj_long_axis: np.ndarray, obj_height_above_rest: float, bin_pos: np.ndarray,
                flags: StageFlags, first_grasp: bool, first_lift: bool, placed_now: bool,
                task_complete: bool, wrong_bin_now: bool, dropped_now: bool, collision: bool,
                action: np.ndarray, truncated_now: bool, obj_xy_speed: float = 0.0,
                lost_grasp: bool = False, over_correct_bin: bool = False,
                gripper_open_cmd: bool = False) -> tuple[float, dict]:
        c = {k: 0.0 for k in self.COMPONENTS}
        dxy = float(np.linalg.norm((tcp_pos - obj_pos)[:2]))
        hover = self.hover_height * float(np.clip(dxy / max(self.hover_radius, 1e-6), 0.0, 1.0))
        target = obj_pos + np.array([0.0, 0.0, hover])
        d = float(np.linalg.norm(tcp_pos - target))
        # releasing over the correct bin is the intended final move: the object falling into the bin
        # must not be charged as "lost lift/transport progress" (the placement bonus follows).
        releasing_into_bin = over_correct_bin and gripper_open_cmd and flags.ever_grasped
        pay = not placed_now and not releasing_into_bin
        # ---- approach (potential, active from episode start until the first grasp) ----
        if not flags.ever_grasped:
            gx = gripper_x_axis[:2]; ox = obj_long_axis[:2]
            n = np.linalg.norm(gx) * np.linalg.norm(ox)
            cos = abs(float(np.dot(gx, ox)) / n) if n > 1e-6 else 0.0
            phi_a = (1.0 - math.tanh(self.approach_scale * d)) * (1.0 - self.align_fraction * (1.0 - cos))
            c["approach_reward"] = self.w["approach_reward"] * self._progress("approach", phi_a, pay)
        else:
            self._progress("approach", None)
        # ---- pushing an un-grasped part around is discouraged ----
        if not flags.grasped and not flags.placed:
            c["push_penalty"] = self.w["push_penalty"] * min(1.0, obj_xy_speed / max(self.push_speed_scale, 1e-6))
        # ---- holding: small per-step maintain term ----
        if flags.grasped:
            c["grasp_reward"] = self.w["grasp_reward"]
        # ---- lift / transport potentials: functions of the OBJECT state, active from the first grasp
        #      until placement, charged back if the object drops or moves away from its bin ----
        if flags.ever_grasped and not flags.placed:
            phi_l = float(np.clip(obj_height_above_rest / self.lift_cap, 0.0, 1.0))
            c["lift_shaping"] = self.w["lift_shaping"] * self._progress("lift", phi_l, pay)
            dxy_bin = float(np.linalg.norm(obj_pos[:2] - bin_pos[:2]))
            phi_t = 1.0 - math.tanh(self.transport_scale * dxy_bin)
            c["transport_reward"] = self.w["transport_reward"] * self._progress("transport", phi_t, pay)
        else:
            self._progress("lift", None)
            self._progress("transport", None)
        # ---- sparse events ----
        if first_grasp:
            c["grasp_bonus_once"] = self.w["grasp_bonus_once"]
        if first_lift:
            c["lift_reward"] = self.w["lift_reward"]
        if placed_now:
            c["placement_reward"] = self.w["placement_reward"]
        if task_complete:
            c["task_complete_bonus"] = self.w["task_complete_bonus"]
        # opening the gripper anywhere but over the correct bin (contact flicker is not penalised)
        if lost_grasp and gripper_open_cmd and not placed_now and not wrong_bin_now and not over_correct_bin:
            c["release_penalty"] = self.w["release_penalty"]
        # ---- penalties ----
        if collision:
            c["collision_penalty"] = self.w["collision_penalty"]
        if wrong_bin_now:
            c["wrong_bin_penalty"] = self.w["wrong_bin_penalty"]
        if dropped_now:
            c["drop_penalty"] = self.w["drop_penalty"]
        c["control_effort_penalty"] = self.w["control_effort_penalty"] * float(np.sum(np.square(action)))
        c["time_penalty"] = self.w["time_penalty"]
        if truncated_now:
            c["timeout_penalty"] = self.w["timeout_penalty"]
        total = float(sum(c.values()))
        return total, c
