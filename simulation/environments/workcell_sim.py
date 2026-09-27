"""MuJoCo workcell wrapper: UR10e + parallel-jaw gripper, two bins, bracket, bolt.

Provides:
  * Cartesian delta control of the tool-center-point with damped-least-squares IK
    (joint limits, joint velocity limits and a workspace box are enforced)
  * grasp / lift / bin-placement / drop / collision detection from real contacts
  * rendering of the overhead learning camera and of viewport cameras
  * domain randomization via DomainRandomizer
"""
from __future__ import annotations
import os
import math
from dataclasses import dataclass
from typing import Optional
import numpy as np
import mujoco

from simulation.config import load_yaml, resolve_path, deep_update
from simulation.environments.randomization import DomainRandomizer, EpisodeConfiguration

os.environ.setdefault("MUJOCO_GL", "egl")

# Renderers must be closed before the EGL context is torn down at interpreter exit,
# otherwise mujoco prints harmless-but-noisy EGLError tracebacks from __del__.
_LIVE_RENDERERS: list = []


def _close_renderer(r):
    try:
        r.close()
    except Exception:
        pass
    if r in _LIVE_RENDERERS:
        _LIVE_RENDERERS.remove(r)


def _close_all_renderers():
    for r in list(_LIVE_RENDERERS):
        _close_renderer(r)


import atexit
atexit.register(_close_all_renderers)


def yaw_from_quat(q: np.ndarray) -> float:
    """Yaw (rotation about world z) of a MuJoCo (w, x, y, z) quaternion."""
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def quat_from_yaw(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class ContactSummary:
    left_pad_target: bool = False
    right_pad_target: bool = False
    arm_collision: bool = False          # arm/gripper body vs table/bins/floor
    finger_bin_contact: bool = False
    n_contacts: int = 0


class WorkcellSim:
    ARM_DOF = 6

    def __init__(self, sim_cfg: dict | None = None, env_cfg: dict | None = None):
        self.sim_cfg = sim_cfg or load_yaml("configs/simulation.yaml")
        self.env_cfg = env_cfg or load_yaml("configs/environment.yaml")
        self.model = mujoco.MjModel.from_xml_path(str(resolve_path(self.sim_cfg["model_path"])))
        self.data = mujoco.MjData(self.model)
        self._scratch = mujoco.MjData(self.model)   # for IK iterations
        m = self.model
        rc = self.sim_cfg["robot"]
        cc = self.sim_cfg["control"]
        self.robot_name = rc["name"]
        self.arm_joint_ids = [m.joint(n).id for n in rc["arm_joints"]]
        self.arm_qpos_adr = np.array([m.jnt_qposadr[j] for j in self.arm_joint_ids])
        self.arm_dof_adr = np.array([m.jnt_dofadr[j] for j in self.arm_joint_ids])
        self.arm_act_ids = np.array([m.actuator(n).id for n in rc["arm_actuators"]])
        self.grip_act_id = m.actuator(rc["gripper_actuator"]).id
        self.grip_joint_qpos = m.jnt_qposadr[m.joint(rc["gripper_joint"]).id]
        self.grip_stroke = float(rc["gripper_stroke"])
        self.ee_site = m.site(rc["ee_site"]).id
        self.home_qpos = np.array(rc["home_qpos"], dtype=float)
        self.joint_lo = m.jnt_range[self.arm_joint_ids, 0] + rc.get("joint_limits_margin", 0.0)
        self.joint_hi = m.jnt_range[self.arm_joint_ids, 1] - rc.get("joint_limits_margin", 0.0)
        self.vel_limits = np.array(rc["joint_velocity_limits"], dtype=float)
        self.control_dt = float(cc["control_dt"])
        self.physics_dt = float(m.opt.timestep)
        self.n_substeps = max(1, int(round(self.control_dt / self.physics_dt)))
        self.max_dpos = float(cc["max_delta_pos"])
        self.max_drot = np.array(cc["max_delta_rot"], dtype=float)
        self.gripper_mode = str(self.env_cfg.get("action", {}).get("gripper_mode", "binary"))
        self.ik_iters = int(cc.get("ik_iterations", 3))
        self.ik_damping = float(cc.get("ik_damping", 0.01))
        self.ik_max_step = float(cc.get("ik_max_joint_step", 0.2))
        ws = cc["workspace"]
        self.ws_lo = np.array([ws["x"][0], ws["y"][0], ws["z"][0]])
        self.ws_hi = np.array([ws["x"][1], ws["y"][1], ws["z"][1]])

        task = self.env_cfg["task"]
        self.objects: list[str] = list(task["objects"])
        self.bins: list[str] = list(task["bins"])
        self.bin_mapping: dict[str, str] = dict(task["bin_mapping"])
        self.table_height = float(task["table_height"])
        # logical task objects (bracket, bolt) -> currently active body (variants can be swapped by the scene builder)
        self.active_body: dict[str, str] = {o: o for o in self.objects}
        self.absent: set[str] = set()          # task objects deselected in the scene builder (their body is parked)
        self.obj_body_id: dict[str, int] = {}
        self.obj_qpos_adr: dict[str, int] = {}
        self.obj_dof_adr: dict[str, int] = {}
        self.obj_geom_ids: dict[str, set] = {}
        for o in self.objects:
            self._bind_object(o, o)
        # parts pool (configs/parts_catalog.yaml): body -> parking pose; bodies not on the table are parked there
        self.catalog = load_yaml("configs/parts_catalog.yaml") if resolve_path("configs/parts_catalog.yaml").exists() else {"parts": {}}
        self.park_pose: dict[str, np.ndarray] = {}
        for i, (name, spec) in enumerate(self.catalog.get("parts", {}).items()):
            if name in self.objects:
                continue
            try:
                jid = m.joint(f"{name}_free").id
            except Exception:
                continue
            self.park_pose[name] = m.qpos0[m.jnt_qposadr[jid]: m.jnt_qposadr[jid] + 7].copy()
        self.rest_height_nominal: dict[str, float] = {n: float(sp.get("rest_height", 0.01)) for n, sp in self.catalog.get("parts", {}).items()}
        # per-DOF viscous damping on every free-body part (configs/simulation.yaml -> physics.object_damping)
        od = (self.sim_cfg.get("physics", {}) or {}).get("object_damping", {}) or {}
        lin, ang = float(od.get("linear", 0.0)), float(od.get("angular", 0.0))
        self.object_damping = (lin, ang)
        if lin > 0 or ang > 0:
            for name in list(self.objects) + list(self.park_pose.keys()):
                try:
                    jid = m.joint(f"{name}_free").id
                except Exception:
                    continue
                d0 = int(m.jnt_dofadr[jid])
                m.dof_damping[d0:d0 + 3] = lin
                m.dof_damping[d0 + 3:d0 + 6] = ang
        self.bin_body_id = {b: m.body(b).id for b in self.bins}
        self.bin_site_id = {b: m.site(f"{b}_center").id for b in self.bins}
        self.bin_geom_ids = {b: set(g for g in range(m.ngeom) if m.geom_bodyid[g] == self.bin_body_id[b])
                             for b in self.bins}
        self.left_pad = m.geom("left_pad").id
        self.right_pad = m.geom("right_pad").id
        self.finger_geoms = {m.geom("left_finger_geom").id, m.geom("right_finger_geom").id,
                             self.left_pad, self.right_pad}
        self.table_geom = m.geom("table_top").id
        self.floor_geom = m.geom("floor").id
        static_ids = {self.table_geom, self.floor_geom}
        for b in self.bins:
            static_ids |= self.bin_geom_ids[b]
        self.static_geoms = static_ids
        # arm + gripper body collision geoms (excluding fingers): any body in the robot chain
        robot_bodies = set()
        bid = m.body("robot_base").id
        for b in range(m.nbody):
            # walk up the tree
            cur = b
            while cur > 0:
                if cur == bid:
                    robot_bodies.add(b)
                    break
                cur = m.body_parentid[cur]
        self.robot_bodies = robot_bodies
        self.arm_geoms = {g for g in range(m.ngeom)
                          if m.geom_bodyid[g] in robot_bodies and g not in self.finger_geoms
                          and (m.geom_contype[g] or m.geom_conaffinity[g])}
        # cameras
        cam = self.sim_cfg["camera"]
        self.overhead_cam = cam["overhead"]["name"]
        self.obs_w, self.obs_h = int(cam["overhead"]["width"]), int(cam["overhead"]["height"])
        self._obs_renderer: Optional[mujoco.Renderer] = None
        self._view_renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        # scene options for viewport renders: default hides geom group 5; the frustum option shows it
        self._default_opt = mujoco.MjvOption()
        self._frustum_opt = mujoco.MjvOption()
        self._frustum_opt.geomgroup[5] = 1
        self._obs_opt = mujoco.MjvOption()
        self._obs_opt.geomgroup[5] = 0
        # randomization
        self.randomizer = DomainRandomizer(m, self.env_cfg["randomization"], self.objects, self.bins,
                                           self.table_height)
        for o in self.objects:
            self.randomizer.rebind(o, self.obj_body_id[o], self.rest_height_nominal.get(o))
        self.episode_config: EpisodeConfiguration = EpisodeConfiguration()
        # controller state
        self.target_pos = np.zeros(3)
        self.target_yaw = 0.0
        self.q_target = self.home_qpos.copy()
        self.grip_cmd = 0.0
        self.track_target: str | None = None      # object whose pad contacts are accumulated
        self.contact_check_every = 5
        self._acc_left = self._acc_right = False
        self.rng = np.random.default_rng(0)
        self.reset()

    def _bind_object(self, logical: str, body: str):
        """Point the logical task object at a (variant) body: ids, addresses, geom sets."""
        m = self.model
        bid = m.body(body).id
        jid = m.joint(f"{body}_free").id
        self.active_body[logical] = body
        self.obj_body_id[logical] = bid
        self.obj_qpos_adr[logical] = int(m.jnt_qposadr[jid])
        self.obj_dof_adr[logical] = int(m.jnt_dofadr[jid])
        self.obj_geom_ids[logical] = set(g for g in range(m.ngeom) if m.geom_bodyid[g] == bid)
        if hasattr(self, "randomizer"):
            self.randomizer.rebind(logical, bid, self.rest_height_nominal.get(body))

    def bind_variant(self, logical: str, body: str):
        """Re-point a task object at a body that is ALREADY on the table (no repositioning)."""
        spec = self.catalog.get("parts", {}).get(body)
        if body != logical and (spec is None or spec.get("role") != "variant" or spec.get("slot") != logical):
            raise ValueError(f"{body} is not a variant for {logical}")
        self._bind_object(logical, body)
        self.absent.discard(logical)

    def present(self, logical: str) -> bool:
        return logical not in self.absent

    def set_variant(self, logical: str, body: str):
        """Swap the geometry used for a task object (e.g. bracket -> bracket_flat). The previous body is parked."""
        if logical not in self.objects:
            raise ValueError(f"unknown task object {logical}")
        spec = self.catalog.get("parts", {}).get(body)
        if body != logical and (spec is None or spec.get("role") != "variant" or spec.get("slot") != logical):
            raise ValueError(f"{body} is not a variant for {logical}")
        if body == self.active_body[logical]:
            return
        old = self.active_body[logical]
        pos = self.obj_pos(logical).copy(); quat = self.obj_quat(logical).copy()
        self.park_body(old)
        self._bind_object(logical, body)
        self.place_body(body, pos[:2], yaw_from_quat(quat), settle_steps=0)

    def body_qpos_adr(self, body: str) -> int:
        return int(self.model.jnt_qposadr[self.model.joint(f"{body}_free").id])

    def park_body(self, body: str):
        """Move a pooled part (or an inactive variant) to its parking pose, at rest."""
        m, d = self.model, self.data
        adr = self.body_qpos_adr(body); dof = int(m.jnt_dofadr[m.joint(f"{body}_free").id])
        pose = self.park_pose.get(body)
        if pose is None:   # a task object body that has no pool slot: park it beside the pool
            pose = np.array([-6.0, -3.5 - 0.35 * self.objects.index(body) if body in self.objects else -3.0, 0.05, 1, 0, 0, 0])
        d.qpos[adr:adr + 7] = pose; d.qvel[dof:dof + 6] = 0
        mujoco.mj_forward(m, d)

    def place_body(self, body: str, xy, yaw: float = 0.0, settle_steps: int = 30):
        """Put a body on the table at xy with yaw, resting on its catalog rest height, then let physics settle."""
        m, d = self.model, self.data
        adr = self.body_qpos_adr(body); dof = int(m.jnt_dofadr[m.joint(f"{body}_free").id])
        z = self.table_height + self.rest_height_nominal.get(body, 0.01) + 0.004
        d.qpos[adr:adr + 3] = [float(xy[0]), float(xy[1]), z]; d.qpos[adr + 3:adr + 7] = quat_from_yaw(float(yaw))
        d.qvel[dof:dof + 6] = 0
        mujoco.mj_forward(m, d)
        if settle_steps > 0:
            mujoco.mj_step(m, d, nstep=settle_steps)

    def body_on_table(self, body: str) -> bool:
        p = self.data.xpos[self.model.body(body).id]
        return bool(p[2] > 0.5 and abs(p[0] + 0.1) < 0.9 and abs(p[1]) < 0.7)

    def pads_grasping(self, body: str) -> bool:
        """Both finger pads in contact with the given body while the gripper is commanded closed (any body, e.g. the rejection part)."""
        bid = self.model.body(body).id
        geoms = set(g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == bid)
        d = self.data; l = r = False
        for g1, g2 in zip(d.contact.geom1[:d.ncon].tolist(), d.contact.geom2[:d.ncon].tolist()):
            if g1 in geoms or g2 in geoms:
                other = g2 if g1 in geoms else g1
                l |= other == self.left_pad; r |= other == self.right_pad
        return l and r and bool(d.ctrl[self.grip_act_id] > 0.3 * self.grip_stroke)

    # ------------------------------------------------------------------ basics
    @property
    def m(self):
        return self.model

    @property
    def d(self):
        return self.data

    def seed(self, seed: int | None):
        self.rng = np.random.default_rng(seed)

    def ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.ee_site].copy()

    def ee_yaw(self) -> float:
        R = self.data.site_xmat[self.ee_site].reshape(3, 3)
        # gripper closing axis is the site's y-axis; report yaw of the site x-axis in the world xy-plane
        return math.atan2(R[1, 0], R[0, 0])

    def ee_rotmat(self) -> np.ndarray:
        return self.data.site_xmat[self.ee_site].reshape(3, 3).copy()

    def arm_qpos(self) -> np.ndarray:
        return self.data.qpos[self.arm_qpos_adr].copy()

    def arm_qvel(self) -> np.ndarray:
        return self.data.qvel[self.arm_dof_adr].copy()

    def gripper_opening(self) -> float:
        """0 = fully closed .. 1 = fully open"""
        return float(np.clip(1.0 - self.data.qpos[self.grip_joint_qpos] / self.grip_stroke, 0, 1))

    def obj_pos(self, name: str) -> np.ndarray:
        return self.data.xpos[self.obj_body_id[name]].copy()

    def obj_quat(self, name: str) -> np.ndarray:
        return self.data.xquat[self.obj_body_id[name]].copy()

    def obj_yaw(self, name: str) -> float:
        return yaw_from_quat(self.obj_quat(name))

    def obj_vel(self, name: str) -> np.ndarray:
        adr = self.obj_dof_adr[name]
        return self.data.qvel[adr:adr + 3].copy()

    def bin_pos(self, name: str) -> np.ndarray:
        return self.data.site_xpos[self.bin_site_id[name]].copy()

    def bin_half_extents(self, name: str) -> np.ndarray:
        return self.model.site_size[self.bin_site_id[name]].copy()

    # ------------------------------------------------------------------ reset
    def reset(self, seed: int | None = None, randomize: bool = True, level: float | None = None,
              fixed_poses: dict | None = None, fixed_bins: bool | None = None) -> EpisodeConfiguration:
        if seed is not None:
            self.seed(seed)
        if level is not None:
            self.randomizer.set_level(level)
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        # inactive variant bodies (e.g. the default bracket when bracket_flat is active) must not reappear on the table
        for o in self.objects:
            if self.active_body[o] != o:
                self.park_body(o)
        saved_level = self.randomizer.level
        if not randomize:
            self.randomizer.set_level(0.0)
        # scenario presets (fixed object poses) keep the bins at their nominal positions unless told otherwise
        if fixed_bins is None:
            fixed_bins = bool(fixed_poses)
        self.episode_config = self.randomizer.sample(self.rng, fixed_poses=fixed_poses, fixed_bins=fixed_bins)
        self.reset_count = getattr(self, "reset_count", 0) + 1
        self.randomizer.set_level(saved_level)
        # arm
        d.qpos[self.arm_qpos_adr] = self.home_qpos
        d.ctrl[self.arm_act_ids] = self.home_qpos
        d.ctrl[self.grip_act_id] = 0.0
        self.q_target = self.home_qpos.copy()
        self.grip_cmd = 0.0
        # objects
        for o in self.objects:
            p = self.episode_config.object_poses[o]
            adr = self.obj_qpos_adr[o]
            z = self.randomizer.object_rest_height(o) + 0.005
            d.qpos[adr:adr + 3] = [p["x"], p["y"], z]
            d.qpos[adr + 3:adr + 7] = quat_from_yaw(p["yaw"])
            dof = self.obj_dof_adr[o]
            d.qvel[dof:dof + 6] = 0
        # deselected task objects stay parked (their pose was sampled but is not applied)
        for o in self.absent:
            self.park_body(self.active_body[o])
        mujoco.mj_forward(m, d)
        for _ in range(int(self.env_cfg["task"].get("settle_steps", 40))):
            mujoco.mj_step(m, d)
        self.target_pos = self.ee_pos()
        self.target_yaw = self.ee_yaw()
        self._acc_left = self._acc_right = False
        return self.episode_config

    # ------------------------------------------------------------------ control
    def _desired_rotmat(self, yaw: float) -> np.ndarray:
        """Tool frame with z pointing down and x-axis at `yaw` in the xy-plane."""
        z = np.array([0.0, 0.0, -1.0])
        x = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        y = np.cross(z, x)
        return np.stack([x, y, z], axis=1)

    def solve_ik(self, target_pos: np.ndarray, target_yaw: float, q0: np.ndarray) -> np.ndarray:
        """Damped least-squares IK for the TCP site. Returns clipped joint targets."""
        m, s = self.model, self._scratch
        s.qpos[:] = self.data.qpos
        q = q0.copy()
        Rd = self._desired_rotmat(target_yaw)
        qd = np.zeros(4)
        mujoco.mju_mat2Quat(qd, Rd.flatten())
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        for _ in range(self.ik_iters):
            s.qpos[self.arm_qpos_adr] = q
            mujoco.mj_kinematics(m, s)
            mujoco.mj_comPos(m, s)
            pos = s.site_xpos[self.ee_site]
            qc = np.zeros(4)
            mujoco.mju_mat2Quat(qc, s.site_xmat[self.ee_site])
            err_pos = target_pos - pos
            err_rot = np.zeros(3)
            mujoco.mju_subQuat(err_rot, qd, qc)     # rotation vector taking qc -> qd (in local frame)
            # subQuat gives the error expressed in the qc frame; rotate to world
            Rc = s.site_xmat[self.ee_site].reshape(3, 3)
            err_rot = Rc @ err_rot
            err = np.concatenate([err_pos, err_rot])
            if np.linalg.norm(err_pos) < 1e-4 and np.linalg.norm(err_rot) < 1e-3:
                break
            mujoco.mj_jacSite(m, s, jacp, jacr, self.ee_site)
            J = np.vstack([jacp[:, self.arm_dof_adr], jacr[:, self.arm_dof_adr]])
            JJt = J @ J.T + (self.ik_damping ** 2) * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, err)
            step = np.max(np.abs(dq))
            if step > self.ik_max_step:
                dq *= self.ik_max_step / step
            q = np.clip(q + dq, self.joint_lo, self.joint_hi)
        return q

    def apply_action(self, action: np.ndarray) -> dict:
        """Cartesian delta action in [-1, 1]^7 -> IK -> position control targets, then
        advance physics by one control period. Returns controller diagnostics."""
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        dpos = a[:3] * self.max_dpos
        drot = a[3:6] * self.max_drot
        new_target = np.clip(self.target_pos + dpos, self.ws_lo, self.ws_hi)
        new_yaw = wrap_angle(self.target_yaw + drot[2])
        q_prev = self.q_target.copy()
        q_new = self.solve_ik(new_target, new_yaw, q_prev)
        # joint velocity limits (per control period)
        dq = q_new - q_prev
        max_dq = self.vel_limits * self.control_dt
        scale = np.max(np.abs(dq) / max_dq)
        if scale > 1.0:
            dq /= scale
            q_new = q_prev + dq
        self.q_target = q_new
        self.target_pos = new_target
        self.target_yaw = new_yaw
        # gripper: -1 open .. +1 closed (continuous) or sign-thresholded (binary, default)
        g = float(a[6]) if a.shape[0] > 6 else -1.0
        if self.gripper_mode == "binary":
            g = 1.0 if g > 0.0 else -1.0
        self.grip_cmd = (g + 1.0) / 2.0 * self.grip_stroke
        self.data.ctrl[self.arm_act_ids] = self.q_target
        self.data.ctrl[self.grip_act_id] = self.grip_cmd
        self.step_physics(self.n_substeps, track_target=self.track_target)
        return {"ik_pos_err": float(np.linalg.norm(self.target_pos - self.ee_pos())),
                "vel_scaled": bool(scale > 1.0)}

    def step_physics(self, n: int = 1, track_target: str | None = None):
        """Advance physics n substeps. If track_target is given, pad contacts with that
        object are accumulated every `contact_check_every` substeps so brief contact
        loss during a squeeze does not hide a real grasp."""
        if track_target is None:
            mujoco.mj_step(self.model, self.data, nstep=n)
            return
        self._acc_left = self._acc_right = False
        every = self.contact_check_every
        done = 0
        while done < n:
            k = min(every, n - done)
            mujoco.mj_step(self.model, self.data, nstep=k)
            done += k
            l, r = self._pad_contacts(track_target)
            self._acc_left |= l
            self._acc_right |= r

    def _pad_contacts(self, target: str) -> tuple[bool, bool]:
        d = self.data
        tgt = self.obj_geom_ids[target]
        l = r = False
        g1s = d.contact.geom1[:d.ncon]
        g2s = d.contact.geom2[:d.ncon]
        for g1, g2 in zip(g1s, g2s):
            if g1 in tgt or g2 in tgt:
                other = g2 if g1 in tgt else g1
                if other == self.left_pad:
                    l = True
                elif other == self.right_pad:
                    r = True
        return l, r

    def set_gripper(self, closed_fraction: float):
        self.grip_cmd = float(np.clip(closed_fraction, 0, 1)) * self.grip_stroke
        self.data.ctrl[self.grip_act_id] = self.grip_cmd

    def move_to(self, target_pos: np.ndarray, target_yaw: float | None = None, max_steps: int = 40,
                tol: float = 0.005) -> bool:
        """Scripted helper (tests / validation only, never used by the policy)."""
        yaw = self.target_yaw if target_yaw is None else target_yaw
        for _ in range(max_steps):
            delta = np.asarray(target_pos) - self.target_pos
            a = np.zeros(7)
            a[:3] = np.clip(delta / self.max_dpos, -1, 1)
            dy = wrap_angle(yaw - self.target_yaw)
            a[5] = np.clip(dy / max(self.max_drot[2], 1e-6), -1, 1)
            a[6] = self.grip_cmd / self.grip_stroke * 2 - 1
            self.apply_action(a)
            if np.linalg.norm(np.asarray(target_pos) - self.ee_pos()) < tol and abs(dy) < 0.02:
                return True
        return False

    # ------------------------------------------------------------------ detection
    def contacts(self, target: str | None) -> ContactSummary:
        d, m = self.data, self.model
        cs = ContactSummary(n_contacts=d.ncon)
        tgt = self.obj_geom_ids.get(target, set()) if target else set()
        all_obj = set().union(*self.obj_geom_ids.values())
        g1s = d.contact.geom1[:d.ncon]
        g2s = d.contact.geom2[:d.ncon]
        for g1, g2 in zip(g1s.tolist(), g2s.tolist()):
            if g1 in tgt or g2 in tgt:
                other = g2 if g1 in tgt else g1
                if other == self.left_pad:
                    cs.left_pad_target = True
                elif other == self.right_pad:
                    cs.right_pad_target = True
            pair = {g1, g2}
            if pair & self.arm_geoms and pair & self.static_geoms:
                cs.arm_collision = True
            if pair & self.finger_geoms and pair & self.static_geoms and not (pair & {self.table_geom}):
                cs.finger_bin_contact = True
        if target is not None and target == self.track_target:
            cs.left_pad_target |= self._acc_left
            cs.right_pad_target |= self._acc_right
        return cs

    def is_grasped(self, target: str, cs: ContactSummary | None = None) -> bool:
        cs = cs or self.contacts(target)
        closing = self.data.ctrl[self.grip_act_id] > 0.3 * self.grip_stroke
        return cs.left_pad_target and cs.right_pad_target and closing

    def in_bin(self, obj: str, bin_name: str) -> bool:
        p = self.obj_pos(obj)
        c = self.bin_pos(bin_name)
        h = self.bin_half_extents(bin_name)
        inside_xy = abs(p[0] - c[0]) <= h[0] and abs(p[1] - c[1]) <= h[1]
        inside_z = (c[2] - h[2] - 0.005) <= p[2] <= (c[2] + h[2] + 0.02)
        return bool(inside_xy and inside_z)

    def which_bin(self, obj: str) -> str | None:
        for b in self.bins:
            if self.in_bin(obj, b):
                return b
        return None

    def is_dropped(self, obj: str) -> bool:
        return bool(self.obj_pos(obj)[2] < float(self.env_cfg["task"]["drop_z_threshold"]))

    def is_lifted(self, obj: str) -> bool:
        rest = self.randomizer.object_rest_height(obj)
        return bool(self.obj_pos(obj)[2] > rest + float(self.env_cfg["task"]["lift_height"]))

    def object_height_above_rest(self, obj: str) -> float:
        return float(self.obj_pos(obj)[2] - self.randomizer.object_rest_height(obj))

    def is_stable(self) -> bool:
        return bool(np.all(np.isfinite(self.data.qpos)) and np.all(np.abs(self.data.qvel) < 200))

    # ------------------------------------------------------------------ render
    def render_overhead(self) -> np.ndarray:
        """Learning camera. Shadows/reflections/skybox are disabled for this small render:
        they are a fixed per-frame GPU cost that dominates at 64x64 on an iGPU."""
        if self._obs_renderer is None:
            self._obs_renderer = mujoco.Renderer(self.model, self.obs_h, self.obs_w)
            _LIVE_RENDERERS.append(self._obs_renderer)
            scn = self._obs_renderer.scene
            for f in (mujoco.mjtRndFlag.mjRND_SHADOW, mujoco.mjtRndFlag.mjRND_REFLECTION,
                      mujoco.mjtRndFlag.mjRND_SKYBOX, mujoco.mjtRndFlag.mjRND_HAZE):
                scn.flags[f] = 0
        self._obs_renderer.update_scene(self.data, camera=self.overhead_cam, scene_option=self._obs_opt)
        return self._obs_renderer.render()

    def render_camera(self, camera: str, width: int, height: int, show_frustum: bool = False) -> np.ndarray:
        """Viewport render. `show_frustum` reveals geom group 5 (the overhead camera's field-of-view
        overlay, see scene.xml). Group 5 is never enabled for the learning camera (render_overhead),
        so the policy input is unaffected by the overlay."""
        key = (width, height)
        r = self._view_renderers.get(key)
        if r is None:
            r = mujoco.Renderer(self.model, height, width)
            self._view_renderers[key] = r
            _LIVE_RENDERERS.append(r)
        opt = self._frustum_opt if show_frustum else self._default_opt
        r.update_scene(self.data, camera=camera, scene_option=opt)
        return r.render()

    def segment_at(self, camera: str, width: int, height: int, u: float, v: float) -> dict:
        """Segmentation pass (geom ids) with the given camera at the given resolution; returns what the pixel
        (u, v in [0, 1], image coordinates) belongs to: {"kind": object|bin|robot|table|none, "name", "geom", "body"}."""
        key = ("seg", width, height)
        r = self._view_renderers.get(key)
        if r is None:
            r = mujoco.Renderer(self.model, height, width)
            r.enable_segmentation_rendering()
            self._view_renderers[key] = r
            _LIVE_RENDERERS.append(r)
        r.update_scene(self.data, camera=camera, scene_option=self._default_opt)
        seg = r.render()
        x = int(np.clip(round(u * (width - 1)), 0, width - 1)); y = int(np.clip(round(v * (height - 1)), 0, height - 1))
        # search a small neighbourhood so thin parts (bolt shaft) are still clickable
        best = None
        for rad in range(0, 7):
            ys = slice(max(0, y - rad), min(height, y + rad + 1)); xs = slice(max(0, x - rad), min(width, x + rad + 1))
            patch = seg[ys, xs]
            ids = patch[..., 0][patch[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)]
            for gid in np.unique(ids):
                gid = int(gid)
                if gid < 0:
                    continue
                bid = int(self.model.geom_bodyid[gid])
                for o, ob in self.obj_body_id.items():
                    if bid == ob:
                        return {"kind": "object", "name": o, "geom": self.model.geom(gid).name, "body": self.model.body(bid).name, "radius_px": rad}
                for b, bb in self.bin_body_id.items():
                    if bid == bb:
                        best = best or {"kind": "bin", "name": b, "geom": self.model.geom(gid).name, "body": self.model.body(bid).name, "radius_px": rad}
                if best is None and rad == 0:
                    kind = "robot" if bid in self.robot_bodies else ("table" if gid == self.table_geom else "other")
                    best = {"kind": kind, "name": self.model.body(bid).name, "geom": self.model.geom(gid).name, "body": self.model.body(bid).name, "radius_px": 0}
            if best is not None and best["kind"] == "bin":
                return best
        return best or {"kind": "none", "name": None, "geom": None, "body": None}

    def pixel_ray(self, camera: str, width: int, height: int, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        """World-frame origin + direction of the camera ray through image pixel (u, v) in [0, 1]."""
        cid = self.model.camera(camera).id
        pos = self.data.cam_xpos[cid].copy(); R = self.data.cam_xmat[cid].reshape(3, 3)
        fovy = math.radians(float(self.model.cam_fovy[cid])); aspect = width / height
        ty = math.tan(fovy / 2.0)
        d_cam = np.array([(2 * u - 1) * ty * aspect, (1 - 2 * v) * ty, -1.0])
        d = R @ d_cam
        return pos, d / np.linalg.norm(d)

    def pixel_to_table(self, camera: str, width: int, height: int, u: float, v: float, z: float | None = None) -> np.ndarray | None:
        """Intersect the pixel ray with the horizontal plane z (default: table top). None if the ray misses."""
        z = self.table_height if z is None else z
        o, d = self.pixel_ray(camera, width, height, u, v)
        if abs(d[2]) < 1e-6:
            return None
        t = (z - o[2]) / d[2]
        if t <= 0:
            return None
        return o + t * d

    def viewport_basis(self, camera: str) -> dict:
        """Screen-right and horizontal-forward unit vectors (world frame) of a viewport camera, so a
        mouse drag on the streamed image can be mapped to table-plane motion."""
        cid = self.model.camera(camera).id
        R = self.data.cam_xmat[cid].reshape(3, 3)
        right = R[:, 0].copy(); right[2] = 0.0
        right = right / (np.linalg.norm(right) + 1e-9)
        fwd = -R[:, 2].copy(); fwd[2] = 0.0
        fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
        return {"name": camera, "right": right.round(4).tolist(), "forward": fwd.round(4).tolist(),
                "position": self.data.cam_xpos[cid].round(3).tolist()}

    def camera_info(self, name: str | None = None) -> dict:
        """Pose / intrinsics of the learning camera, derived from the compiled model (not hard-coded)."""
        name = name or self.overhead_cam
        cid = self.model.camera(name).id
        pos = self.data.cam_xpos[cid].copy()
        R = self.data.cam_xmat[cid].reshape(3, 3)
        view_dir = -R[:, 2]                       # MuJoCo cameras look along their local -z
        image_up = R[:, 1]                        # local +y = image up
        tilt = math.degrees(math.acos(float(np.clip(-view_dir[2], -1.0, 1.0))))   # 0 = straight down
        fovy = float(self.model.cam_fovy[cid])
        depth = float(pos[2] - self.table_height)
        half = depth * math.tan(math.radians(fovy / 2.0))
        axes = ["+x", "-x", "+y", "-y", "+z", "-z"]
        up_axis = axes[int(np.argmax([image_up[0], -image_up[0], image_up[1], -image_up[1], image_up[2], -image_up[2]]))]
        return {
            "name": name,
            "position": pos.round(3).tolist(),
            "quaternion_wxyz": self.model.cam_quat[cid].round(4).tolist(),
            "view_direction": view_dir.round(3).tolist(),
            "image_up_world": image_up.round(3).tolist(),
            "image_up_axis": up_axis,
            "tilt_from_vertical_deg": round(tilt, 2),
            "description": "looking straight down" if tilt < 0.5 else f"tilted {tilt:.1f}° off vertical",
            "fovy_deg": fovy,
            "image_size": [self.obs_w, self.obs_h],
            "height_above_table_m": round(depth, 3),
            "footprint_on_table_m": [round(2 * half, 3), round(2 * half, 3)],
        }

    def close(self):
        for r in [self._obs_renderer, *self._view_renderers.values()]:
            if r is not None:
                _close_renderer(r)
        self._obs_renderer = None
        self._view_renderers.clear()

    # ------------------------------------------------------------------ state export
    def state_dict(self, target: str | None = None) -> dict:
        """Compact JSON-able snapshot for the API / WebSocket."""
        cs = self.contacts(target)
        return {
            "time": float(self.data.time),
            "robot": self.robot_name,
            "joint_positions": self.arm_qpos().round(4).tolist(),
            "joint_velocities": self.arm_qvel().round(4).tolist(),
            "ee_position": self.ee_pos().round(4).tolist(),
            "ee_yaw": round(self.ee_yaw(), 4),
            "gripper_opening": round(self.gripper_opening(), 4),
            "gripper_cmd_closed": round(self.grip_cmd / self.grip_stroke, 3),
            "objects": {o: {"position": self.obj_pos(o).round(4).tolist(),
                            "yaw": round(self.obj_yaw(o), 4),
                            "in_bin": self.which_bin(o),
                            "lifted": self.is_lifted(o),
                            "dropped": self.is_dropped(o)} for o in self.objects},
            "bins": {b: self.bin_pos(b).round(4).tolist() for b in self.bins},
            "contacts": {"left_pad": cs.left_pad_target, "right_pad": cs.right_pad_target,
                         "arm_collision": cs.arm_collision, "n": cs.n_contacts},
            "grasped": self.is_grasped(target, cs) if target else False,
            "target": target,
            "episode_config": self.episode_config.to_dict(),
        }
