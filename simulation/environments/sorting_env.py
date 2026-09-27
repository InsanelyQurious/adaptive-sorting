"""Gymnasium environment: VolvoSorting-v0.

Observation (LeRobot-shaped dict):
  observation.images.overhead : uint8 (H, W, 3) RGB from the real MuJoCo overhead camera
  observation.state           : float32 vector (see STATE_LAYOUT)
Action: float32 [-1, 1]^7 = (dX, dY, dZ, dRx, dRy, dRz, gripper)
"""
from __future__ import annotations
import math
from typing import Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from simulation.config import load_yaml, deep_update
from simulation.environments.workcell_sim import WorkcellSim, wrap_angle
from training.rewards.sorting_reward import SortingReward, StageFlags

IMAGE_KEY = "observation.images.overhead"
STATE_KEY = "observation.state"


def build_state_layout(objects: list[str]) -> list[str]:
    names = [f"joint_pos_{i}" for i in range(6)] + [f"joint_vel_{i}" for i in range(6)]
    names += ["gripper_opening", "ee_x", "ee_y", "ee_z", "ee_yaw_sin", "ee_yaw_cos"]
    for o in objects:
        names += [f"{o}_x", f"{o}_y", f"{o}_z", f"{o}_yaw_sin", f"{o}_yaw_cos"]
    names += ["bin_a_x", "bin_a_y", "bin_a_z", "bin_b_x", "bin_b_y", "bin_b_z"]
    names += [f"target_is_{o}" for o in objects]
    names += ["tcp_to_target_x", "tcp_to_target_y", "tcp_to_target_z"]
    names += ["target_to_bin_x", "target_to_bin_y", "target_to_bin_z"]
    names += ["grasped_flag", "lifted_flag"]
    return names


class VolvoSortingEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, sim_cfg: dict | None = None, env_cfg: dict | None = None,
                 render_mode: str | None = "rgb_array", include_image: bool | None = None):
        super().__init__()
        self.sim_cfg = sim_cfg or load_yaml("configs/simulation.yaml")
        self.env_cfg = env_cfg or load_yaml("configs/environment.yaml")
        self.sim = WorkcellSim(self.sim_cfg, self.env_cfg)
        task = self.env_cfg["task"]
        obs_cfg = self.env_cfg["observation"]
        self.objects = list(task["objects"])
        self.bin_mapping = dict(task["bin_mapping"])
        self.mode = task.get("mode", "sequential")
        self.max_steps = int(task["max_episode_steps"])
        self.grasp_hold_steps = int(task.get("grasp_hold_steps", 2))
        cur = task.get("curriculum", {}) or {}
        self.curriculum_pregrasp = float(cur.get("pregrasp_prob", 0.0))
        self.curriculum_grasped = float(cur.get("grasped_prob", 0.0))
        self.curriculum_straddle = float(cur.get("straddle_prob", 0.0))
        self.curriculum_overbin = float(cur.get("overbin_prob", 0.0))
        self.curriculum_height = list(cur.get("pregrasp_height", [0.08, 0.14]))
        self.curriculum_enabled = False   # switched on by make_env(training=True) only
        self.last_reset_kind = "home"
        self.include_image = obs_cfg.get("include_image", True) if include_image is None else include_image
        self.img_h, self.img_w = self.sim.obs_h, self.sim.obs_w
        self.joint_vel_scale = float(obs_cfg.get("joint_vel_scale", 0.2))
        self.render_mode = render_mode
        self.reward_fn = SortingReward(self.env_cfg["reward"])
        self.state_layout = build_state_layout(self.objects)
        state_dim = len(self.state_layout)
        spaces_dict = {STATE_KEY: spaces.Box(-np.inf, np.inf, (state_dim,), np.float32)}
        if self.include_image:
            spaces_dict[IMAGE_KEY] = spaces.Box(0, 255, (self.img_h, self.img_w, 3), np.uint8)
        self.observation_space = spaces.Dict(spaces_dict)
        self.action_space = spaces.Box(-1.0, 1.0, (int(self.env_cfg["action"]["dim"]),), np.float32)
        self._rng = np.random.default_rng(0)
        self._episode_seed = 0
        self._reset_options: dict = {}
        self.stage: dict[str, StageFlags] = {}
        self.target: str | None = None
        self.t = 0
        self.collisions = 0
        self.ep_components: dict[str, float] = {}
        self.ep_reward = 0.0
        self._grasp_streak = 0
        self.last_action = np.zeros(self.action_space.shape[0], dtype=np.float32)
        self.last_components: dict[str, float] = {}

    # ------------------------------------------------------------------ helpers
    def _present(self) -> list[str]:
        return [o for o in self.objects if self.sim.present(o)]

    def _pick_target(self) -> str | None:
        remaining = [o for o in self._present() if not self.stage[o].placed]
        if not remaining:
            return None
        if self.mode == "single":
            return remaining[0]
        # sequential: nearest remaining object to the gripper
        tcp = self.sim.ee_pos()
        return min(remaining, key=lambda o: np.linalg.norm(self.sim.obj_pos(o) - tcp))

    def _over_bin(self, obj: str, bin_name: str) -> bool:
        p = self.sim.obj_pos(obj); c = self.sim.bin_pos(bin_name); h = self.sim.bin_half_extents(bin_name)
        return bool(abs(p[0] - c[0]) <= h[0] and abs(p[1] - c[1]) <= h[1])

    def _obj_long_axis(self, o: str) -> np.ndarray:
        R = np.zeros(9)
        import mujoco
        mujoco.mju_quat2Mat(R, self.sim.obj_quat(o))
        return R.reshape(3, 3)[:, 0]

    def _get_state(self) -> np.ndarray:
        s = self.sim
        tgt = self.target
        parts = [s.arm_qpos(), s.arm_qvel() * self.joint_vel_scale, [s.gripper_opening()],
                 s.ee_pos(), [math.sin(s.ee_yaw()), math.cos(s.ee_yaw())]]
        for o in self.objects:
            if not s.present(o):            # deselected in the scene builder: encoded as all-zero pose
                parts += [np.zeros(3), [0.0, 0.0]]
                continue
            y = s.obj_yaw(o)
            parts += [s.obj_pos(o), [math.sin(y), math.cos(y)]]
        parts += [s.bin_pos("bin_a"), s.bin_pos("bin_b")]
        parts += [[1.0 if tgt == o else 0.0 for o in self.objects]]
        if tgt is not None:
            tp = s.obj_pos(tgt); bp = s.bin_pos(self.bin_mapping[tgt])
            parts += [tp - s.ee_pos(), bp - tp,
                      [1.0 if self.stage[tgt].grasped else 0.0, 1.0 if self.stage[tgt].lifted else 0.0]]
        else:
            parts += [np.zeros(3), np.zeros(3), [0.0, 0.0]]
        state = np.concatenate([np.asarray(p, dtype=np.float32).ravel() for p in parts])
        noise = self.sim.episode_config.state_noise_std
        if noise > 0:
            state = state + self._rng.normal(0, noise, state.shape).astype(np.float32)
        return state.astype(np.float32)

    def _get_obs(self) -> dict[str, Any]:
        obs = {STATE_KEY: self._get_state()}
        if self.include_image:
            img = self.sim.render_overhead()
            noise = self.sim.episode_config.image_noise_std
            if noise > 0:
                img = np.clip(img.astype(np.float32) + self._rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
            obs[IMAGE_KEY] = np.ascontiguousarray(img)
        return obs

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        if seed is not None:
            self._episode_seed = int(seed)
            self._rng = np.random.default_rng(seed)
        else:
            self._episode_seed = int(self._rng.integers(0, 2**31 - 1))
        level = options.get("randomization_level")
        randomize = options.get("randomize", True)
        fixed_poses = options.get("fixed_poses")
        self.sim.reset(seed=self._episode_seed, randomize=randomize, level=level, fixed_poses=fixed_poses,
                       fixed_bins=options.get("fixed_bins"))
        self.stage = {o: StageFlags() for o in self.objects}
        self.t = 0
        self.collisions = 0
        self.ep_reward = 0.0
        self.ep_components = {k: 0.0 for k in SortingReward.COMPONENTS}
        self._grasp_streak = 0
        self.target = self._pick_target()
        self.sim.track_target = self.target
        self.reward_fn.reset()
        self.last_action[:] = 0
        self.last_reset_kind = "home"
        use_cur = options.get("curriculum", self.curriculum_enabled)
        if use_cur and self.target is not None:
            u = self._rng.random()
            if u < self.curriculum_overbin:
                self._scripted_pregrasp(self.target, close=True, transport=True)
                self.last_reset_kind = "overbin"
            elif u < self.curriculum_overbin + self.curriculum_grasped:
                self._scripted_pregrasp(self.target, close=True)
                self.last_reset_kind = "grasped"
            elif u < self.curriculum_overbin + self.curriculum_grasped + self.curriculum_straddle:
                self._scripted_pregrasp(self.target, close=False, descend=True)
                self.last_reset_kind = "straddle"
            elif u < self.curriculum_overbin + self.curriculum_grasped + self.curriculum_straddle + self.curriculum_pregrasp:
                self._scripted_pregrasp(self.target, close=False)
                self.last_reset_kind = "pregrasp"
        self._training_clutter()
        obs = self._get_obs()
        return obs, self._info(terminated=False, truncated=False)

    def _training_clutter(self):
        """Optional clutter exposure for TRAINING resets only (randomization.clutter.prob); parks the parts otherwise."""
        cc = (self.env_cfg.get("randomization", {}) or {}).get("clutter", {}) or {}
        parts = [p for p in cc.get("parts", []) if p in getattr(self.sim, "park_pose", {})]
        if not parts:
            return
        for p in parts:
            self.sim.park_body(p)
        if not self.curriculum_enabled or float(cc.get("prob", 0.0)) <= 0 or self._rng.random() >= float(cc.get("prob", 0.0)):
            return
        k = int(self._rng.integers(1, int(cc.get("max_parts", 2)) + 1))
        taken = [self.sim.obj_pos(o)[:2] for o in self.objects if self.sim.present(o)]
        zone = self.env_cfg["randomization"]["spawn_zone"]
        for p in self._rng.choice(parts, size=min(k, len(parts)), replace=False):
            for _ in range(30):
                xy = np.array([self._rng.uniform(*zone["x"]), self._rng.uniform(*zone["y"])])
                if all(np.linalg.norm(xy - t) >= 0.11 for t in taken):
                    break
            else:
                continue
            self.sim.place_body(str(p), xy, float(self._rng.uniform(-math.pi, math.pi)), settle_steps=0)
            taken.append(xy)
        import mujoco
        mujoco.mj_step(self.sim.model, self.sim.data, nstep=30)

    def _scripted_pregrasp(self, obj: str, close: bool, descend: bool = False, transport: bool = False):
        """Curriculum reset helper: drive the gripper (through the same IK controller and physics
        the policy uses) to hover above the target, optionally descend and close. No teleporting."""
        s = self.sim
        p = s.obj_pos(obj)
        yaw = wrap_angle(s.obj_yaw(obj) + self._rng.uniform(-0.25, 0.25))
        hover = p + np.array([self._rng.uniform(-0.01, 0.01), self._rng.uniform(-0.01, 0.01),
                              self._rng.uniform(*self.curriculum_height)])
        hover[2] = max(hover[2], s.ws_lo[2] - p[2] + 0.005)
        s.set_gripper(0.0)
        s.move_to(hover, yaw, max_steps=50)
        if close or descend:
            s.move_to(np.array([p[0], p[1], s.ws_lo[2]]), yaw, max_steps=40, tol=0.004)
        if close:
            for _ in range(8):
                s.apply_action(np.array([0, 0, 0, 0, 0, 0, 1.0]))
            # bookkeeping so the reward/flags see an established grasp
            cs = s.contacts(obj)
            if s.is_grasped(obj, cs):
                self._grasp_streak = self.grasp_hold_steps
                self.stage[obj].grasped = True
                self.stage[obj].ever_grasped = True
                if transport:
                    # lift, then carry above the correct bin (still through the controller + physics)
                    for _ in range(6):
                        s.apply_action(np.array([0, 0, 0.8, 0, 0, 0, 1.0]))
                    b = s.bin_pos(self.bin_mapping[obj])
                    target = np.array([b[0] + self._rng.uniform(-0.03, 0.03), b[1] + self._rng.uniform(-0.03, 0.03),
                                       self._rng.uniform(0.98, 1.06)])
                    for _ in range(40):
                        d = target - s.target_pos
                        a = np.zeros(7); a[:3] = np.clip(d / s.max_dpos, -1, 1); a[6] = 1.0
                        s.apply_action(a)
                        if np.linalg.norm(d) < 0.01:
                            break
                    if s.is_grasped(obj) and s.is_lifted(obj):
                        self.stage[obj].lifted = True
                        self.stage[obj].ever_lifted = True

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1, 1)
        self.last_action = action.copy()
        s = self.sim
        tgt = self.target
        ctrl_info = s.apply_action(action)
        self.t += 1
        unstable = not s.is_stable()
        first_grasp = first_lift = placed_now = wrong_bin_now = dropped_now = task_complete = False
        collision = False
        reward = 0.0
        comps = {k: 0.0 for k in SortingReward.COMPONENTS}
        terminated = False
        if tgt is not None and not unstable:
            fl = self.stage[tgt]
            cs = s.contacts(tgt)
            collision = cs.arm_collision
            if collision:
                self.collisions += 1
            grasp_contact = s.is_grasped(tgt, cs)
            self._grasp_streak = self._grasp_streak + 1 if grasp_contact else 0
            grasped = self._grasp_streak >= self.grasp_hold_steps
            lost_grasp = fl.grasped and not grasped
            if grasped and not fl.ever_grasped:
                first_grasp = True
                fl.ever_grasped = True
            fl.grasped = grasped
            lifted = grasped and s.is_lifted(tgt)
            if lifted and not fl.ever_lifted:
                first_lift = True
                fl.ever_lifted = True
            fl.lifted = lifted
            released = s.data.ctrl[s.grip_act_id] < 0.3 * s.grip_stroke or not grasp_contact
            in_bin = s.which_bin(tgt)
            if in_bin is not None and released and fl.ever_grasped:
                if in_bin == self.bin_mapping[tgt]:
                    placed_now = True
                    fl.placed = True
                else:
                    wrong_bin_now = True
                    fl.wrong_bin = True
                    terminated = True
            # any object falling off the table ends the episode
            for o in self._present():
                if s.is_dropped(o):
                    dropped_now = True
                    self.stage[o].dropped = True
                    terminated = True
            truncated = self.t >= self.max_steps
            if placed_now:
                self.target = self._pick_target()
                self.sim.track_target = self.target
                self._grasp_streak = 0
                if self.target is None:
                    task_complete = True
                    terminated = True
            reward, comps = self.reward_fn.compute(
                tcp_pos=s.ee_pos(), gripper_x_axis=s.ee_rotmat()[:, 0], obj_pos=s.obj_pos(tgt),
                obj_long_axis=self._obj_long_axis(tgt), obj_height_above_rest=s.object_height_above_rest(tgt),
                bin_pos=s.bin_pos(self.bin_mapping[tgt]), flags=fl, first_grasp=first_grasp,
                first_lift=first_lift, placed_now=placed_now, task_complete=task_complete,
                wrong_bin_now=wrong_bin_now, dropped_now=dropped_now, collision=collision,
                action=action, truncated_now=(truncated and not terminated and not task_complete),
                obj_xy_speed=float(np.linalg.norm(s.obj_vel(tgt)[:2])), lost_grasp=lost_grasp,
                over_correct_bin=self._over_bin(tgt, self.bin_mapping[tgt]),
                gripper_open_cmd=bool(s.grip_cmd < 0.3 * s.grip_stroke))
            if placed_now:
                self.reward_fn.reset()   # new target: start its potentials fresh
        else:
            truncated = self.t >= self.max_steps
            if unstable:
                terminated = True
                reward = self.reward_fn.w["drop_penalty"]
                comps["drop_penalty"] = reward
        for k, v in comps.items():
            self.ep_components[k] += v
        self.ep_reward += reward
        self.last_components = comps
        if terminated:
            truncated = False
        obs = self._get_obs()
        info = self._info(terminated, truncated, unstable=unstable, ctrl=ctrl_info)
        return obs, float(reward), bool(terminated), bool(truncated), info

    def _info(self, terminated: bool, truncated: bool, unstable: bool = False, ctrl: dict | None = None) -> dict:
        present = self._present()
        placed = sum(int(self.stage[o].placed) for o in present)
        success = len(present) > 0 and placed == len(present)
        info = {
            "target": self.target,
            "placed_count": placed,
            "is_success": bool(success),
            "grasp_achieved": any(self.stage[o].ever_grasped for o in present),
            "lift_achieved": any(self.stage[o].ever_lifted for o in present),
            "present_objects": present,
            "grasped_now": bool(self.target and self.stage[self.target].grasped),
            "lifted_now": bool(self.target and self.stage[self.target].lifted),
            "collisions": self.collisions,
            "dropped": any(f.dropped for f in self.stage.values()),
            "wrong_bin": any(f.wrong_bin for f in self.stage.values()),
            "unstable": unstable,
            "step": self.t,
            "reset_kind": self.last_reset_kind,
            "episode_reward": self.ep_reward,
            "reward_components": dict(self.last_components),
            "seed": self._episode_seed,
        }
        if terminated or truncated:
            info["episode_components"] = dict(self.ep_components)
            info["episode_config"] = self.sim.episode_config.to_dict()
        if ctrl:
            info["ik_pos_err"] = ctrl["ik_pos_err"]
        return info

    def render(self):
        return self.sim.render_camera(self.sim_cfg["camera"]["viewport"]["name"], 640, 480)

    def close(self):
        self.sim.close()


# --------------------------------------------------------------------------- SB3 adapter
class SB3KeyAlias(gym.ObservationWrapper):
    """torch.nn.ModuleDict forbids '.' in keys, so SB3's CombinedExtractor cannot consume
    LeRobot-style keys directly. This wrapper renames them for the RL library only:
        observation.images.overhead -> image
        observation.state           -> state
    The environment itself stays LeRobot-shaped."""
    ALIAS = {IMAGE_KEY: "image", STATE_KEY: "state"}

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Dict({self.ALIAS[k]: v for k, v in env.observation_space.spaces.items()})

    def observation(self, obs):
        return {self.ALIAS[k]: v for k, v in obs.items()}


def make_env(sim_cfg: dict | None = None, env_cfg: dict | None = None, overrides: dict | None = None,
             sb3: bool = True, include_image: bool | None = None, training: bool = False) -> gym.Env:
    """training=True enables the initial-state curriculum; evaluation/inference envs keep it off."""
    env_cfg = deep_update(env_cfg or load_yaml("configs/environment.yaml"), overrides)
    env = VolvoSortingEnv(sim_cfg=sim_cfg, env_cfg=env_cfg, include_image=include_image)
    env.curriculum_enabled = bool(training)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=env.max_steps + 1)  # env truncates itself first
    return SB3KeyAlias(env) if sb3 else env


gym.register(id="VolvoSorting-v0", entry_point="simulation.environments.sorting_env:VolvoSortingEnv")
