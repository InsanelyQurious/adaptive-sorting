"""Domain randomization for the sorting workcell.

A single, configurable system (not scattered random calls). Every randomizable
quantity is listed in configs/environment.yaml under `randomization`. The
`level` scalar in [0, 1] scales every range so a curriculum or a UI slider can
dial randomization up and down without touching source.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np
import mujoco


@dataclass
class EpisodeConfiguration:
    """What was sampled for one episode; logged with every evaluation result."""
    object_poses: dict[str, dict] = field(default_factory=dict)   # name -> {x, y, yaw}
    bin_offsets: dict[str, list[float]] = field(default_factory=dict)
    object_scales: dict[str, float] = field(default_factory=dict)
    frictions: dict[str, float] = field(default_factory=dict)
    state_noise_std: float = 0.0
    image_noise_std: float = 0.0
    light: dict = field(default_factory=dict)
    level: float = 1.0
    bins_fixed: bool = False

    def to_dict(self) -> dict:
        return {
            "object_poses": self.object_poses,
            "bin_offsets": self.bin_offsets,
            "object_scales": self.object_scales,
            "frictions": self.frictions,
            "state_noise_std": self.state_noise_std,
            "image_noise_std": self.image_noise_std,
            "light": self.light,
            "level": self.level,
            "bins_fixed": self.bins_fixed,
        }


class DomainRandomizer:
    def __init__(self, model: mujoco.MjModel, cfg: dict, objects: list[str], bins: list[str],
                 table_height: float):
        self.m = model
        self.cfg = cfg
        self.objects = objects
        self.bins = bins
        self.table_height = table_height
        self.level = float(cfg.get("level", 1.0))
        # nominal values to restore each reset
        self._nominal_bin_pos = {b: model.body(b).pos.copy() for b in bins}
        self._obj_geoms = {}
        self._nominal_geom_size = {}
        self._nominal_geom_pos = {}
        self._nominal_friction = {}
        self._nominal_body_mass = {}
        self._nominal_body_inertia = {}
        for o in objects:
            bid = model.body(o).id
            gids = [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid]
            self._obj_geoms[o] = gids
            self._nominal_body_mass[o] = float(model.body_mass[bid])
            self._nominal_body_inertia[o] = model.body_inertia[bid].copy()
            for g in gids:
                self._nominal_geom_size[g] = model.geom_size[g].copy()
                self._nominal_geom_pos[g] = model.geom_pos[g].copy()
                self._nominal_friction[g] = model.geom_friction[g].copy()
        self._rest_nominal: dict[str, float | None] = {o: None for o in objects}
        self.current_scale: dict[str, float] = {o: 1.0 for o in objects}
        self._key_light = model.light("key").id if "key" in [model.light(i).name for i in range(model.nlight)] else -1
        self._nominal_light_dir = model.light_dir[self._key_light].copy() if self._key_light >= 0 else None
        self._nominal_light_diffuse = model.light_diffuse[self._key_light].copy() if self._key_light >= 0 else None

    def rebind(self, logical: str, body_id: int, rest_height_nominal: float | None = None):
        """Re-point a logical object at another body (scene-builder variant swap): nominal geometry tables."""
        model = self.m
        gids = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body_id]
        self._obj_geoms[logical] = gids
        self._nominal_body_mass[logical] = float(model.body_mass[body_id])
        self._nominal_body_inertia[logical] = model.body_inertia[body_id].copy()
        for g in gids:
            self._nominal_geom_size.setdefault(g, model.geom_size[g].copy())
            self._nominal_geom_pos.setdefault(g, model.geom_pos[g].copy())
            self._nominal_friction.setdefault(g, model.geom_friction[g].copy())
        self._rest_nominal[logical] = rest_height_nominal
        self.current_scale[logical] = 1.0

    # ------------------------------------------------------------------
    def set_level(self, level: float):
        self.level = float(np.clip(level, 0.0, 1.0))

    def _scaled_range(self, lo: float, hi: float, center: float | None = None):
        """Shrink [lo, hi] toward its center by self.level."""
        if center is None:
            center = 0.5 * (lo + hi)
        return center + (lo - center) * self.level, center + (hi - center) * self.level

    def sample(self, rng: np.random.Generator, fixed_poses: dict | None = None, fixed_bins: bool = False) -> EpisodeConfiguration:
        """Sample an episode configuration and write it into the model (sizes, friction,
        bin positions, lights). Object poses are returned for the env to write into qpos.
        Bins move ONLY here (episode start), by at most bin_position_jitter * level; fixed_bins=True (scenario
        presets) keeps them at their nominal MJCF positions."""
        cfg = self.cfg
        ep = EpisodeConfiguration(level=self.level)
        # ----- bins -----
        jx, jy = cfg.get("bin_position_jitter", [0.0, 0.0])
        if not bool(cfg.get("randomize_bins", True)):
            fixed_bins = True
        ep.bins_fixed = bool(fixed_bins)
        for b in self.bins:
            off = np.zeros(3) if fixed_bins else np.array([rng.uniform(-jx, jx), rng.uniform(-jy, jy), 0.0]) * self.level
            self.m.body(b).pos[:] = self._nominal_bin_pos[b] + off
            ep.bin_offsets[b] = off[:2].round(4).tolist()
        # ----- object scale / friction / mass -----
        s_lo, s_hi = cfg.get("object_scale", [1.0, 1.0])
        f_lo, f_hi = cfg.get("friction", [1.0, 1.0])
        for o in self.objects:
            s_lo_l, s_hi_l = self._scaled_range(s_lo, s_hi, 1.0)
            scale = float(rng.uniform(s_lo_l, s_hi_l))
            f_lo_l, f_hi_l = self._scaled_range(f_lo, f_hi)
            fric = float(rng.uniform(f_lo_l, f_hi_l))
            bid = self.m.body(o).id
            for g in self._obj_geoms[o]:
                self.m.geom_size[g] = self._nominal_geom_size[g] * scale
                self.m.geom_pos[g] = self._nominal_geom_pos[g] * scale
                self.m.geom_friction[g, 0] = fric
            self.m.body_mass[bid] = self._nominal_body_mass[o] * scale ** 3
            self.m.body_inertia[bid] = self._nominal_body_inertia[o] * scale ** 5
            self.current_scale[o] = scale
            ep.object_scales[o] = round(scale, 4)
            ep.frictions[o] = round(fric, 4)
        # ----- object poses -----
        zone = cfg["spawn_zone"]
        x_lo, x_hi = self._scaled_range(*zone["x"])
        y_lo, y_hi = self._scaled_range(*zone["y"])
        yaw_lo, yaw_hi = cfg.get("object_yaw", [0.0, 0.0])
        yaw_lo, yaw_hi = yaw_lo * self.level, yaw_hi * self.level
        min_sep = zone.get("min_separation", 0.12)
        placed: list[np.ndarray] = []
        for o in self.objects:
            if fixed_poses and o in fixed_poses:
                fp = fixed_poses[o]
                pose = {"x": float(fp["x"]), "y": float(fp["y"]), "yaw": float(fp.get("yaw", 0.0))}
                placed.append(np.array([pose["x"], pose["y"]]))
                ep.object_poses[o] = pose
                continue
            for _ in range(200):
                xy = np.array([rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)])
                if all(np.linalg.norm(xy - p) >= min_sep for p in placed):
                    break
            placed.append(xy)
            ep.object_poses[o] = {"x": round(float(xy[0]), 4), "y": round(float(xy[1]), 4),
                                  "yaw": round(float(rng.uniform(yaw_lo, yaw_hi)), 4)}
        # ----- noise -----
        ep.state_noise_std = float(cfg.get("state_noise_std", 0.0)) * self.level
        ep.image_noise_std = float(cfg.get("image_noise_std", 0.0)) * self.level
        # ----- lighting -----
        if self._key_light >= 0 and cfg.get("light_jitter", False) and self.level > 0:
            d = self._nominal_light_dir + rng.normal(0, 0.12 * self.level, 3)
            d = d / np.linalg.norm(d)
            self.m.light_dir[self._key_light] = d
            gain = 1.0 + rng.uniform(-0.25, 0.25) * self.level
            self.m.light_diffuse[self._key_light] = np.clip(self._nominal_light_diffuse * gain, 0, 1)
            ep.light = {"dir": d.round(3).tolist(), "gain": round(gain, 3)}
        elif self._key_light >= 0:
            self.m.light_dir[self._key_light] = self._nominal_light_dir
            self.m.light_diffuse[self._key_light] = self._nominal_light_diffuse
        return ep

    def object_rest_height(self, name: str) -> float:
        """Approximate z of the body origin when the object rests flat on the table."""
        if self._rest_nominal.get(name) is not None:
            return self.table_height + float(self._rest_nominal[name]) * self.current_scale.get(name, 1.0) + 0.001
        scale = 1.0
        # bracket plate half-thickness / bolt radius (nominal), scaled
        gids = self._obj_geoms[name]
        g0 = gids[0]
        size = self.m.geom_size[g0]
        if name == "bracket":
            return self.table_height + float(size[2]) + 0.001
        if name == "bolt":
            # rests on the hex head (box half-height) and the shaft; head is the tallest point
            head = [g for g in gids if self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]
            if head:
                return self.table_height + float(self.m.geom_size[head[0]][2]) + 0.001
            return self.table_height + float(size[0]) + 0.001
        return self.table_height + 0.02
