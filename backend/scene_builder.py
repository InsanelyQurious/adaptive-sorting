"""Scene builder (Phase 3): parts catalog, drag-and-drop spawning on the live MuJoCo table, clutter layouts,
custom scenarios. No model recompilation: every catalog part is a pre-compiled free body parked off-cell.

Drop position: the viewport pixel is turned into a camera ray (same intrinsics/pose used by the click-to-pick
segmentation pass) and intersected with the table plane z = table_height.
"""
from __future__ import annotations
import base64
import json
import math
import time
from pathlib import Path
import numpy as np
import cv2

from simulation.config import resolve_path, load_yaml

CUSTOM_FILE = resolve_path("configs/custom_scenarios.json")
TABLE_X = (-0.85, 0.60)      # table top spans x in [-0.9, 0.7], y in [-0.6, 0.6]; keep parts inside with a margin
TABLE_Y = (-0.55, 0.55)


def load_custom_scenarios() -> dict:
    if CUSTOM_FILE.exists():
        try:
            return json.loads(CUSTOM_FILE.read_text())
        except Exception:
            return {}
    return {}


def save_custom_scenarios(d: dict):
    CUSTOM_FILE.write_text(json.dumps(d, indent=1))


class SceneBuilder:
    def __init__(self, svc):
        self.svc = svc
        self.sim = svc.env.sim
        self.catalog: dict = self.sim.catalog.get("parts", {})
        self.layout: list[dict] = []          # clutter/reject parts currently authored on the table: {part, x, y, yaw}
        self.thumbs: dict[str, str] = {}
        self._render_thumbnails()
        svc.scene_after_reset = self.after_reset

    # ------------------------------------------------------------------ catalog
    def _render_thumbnails(self):
        """Move each part to the photo booth, render it with catalog_cam, move it back. Runs once at start-up."""
        import mujoco
        sim = self.sim
        try:
            r = mujoco.Renderer(sim.model, 96, 96)
        except Exception:
            return
        saved_qpos = sim.data.qpos.copy(); saved_qvel = sim.data.qvel.copy()
        for name in self.catalog:
            try:
                adr = sim.body_qpos_adr(name)
            except Exception:
                continue
            q = sim.data.qpos.copy()
            sim.data.qpos[adr:adr + 3] = [-6.0, 0.0, 0.002 + sim.rest_height_nominal.get(name, 0.01)]
            sim.data.qpos[adr + 3:adr + 7] = [math.cos(0.35), 0, 0, math.sin(0.35)]
            mujoco.mj_forward(sim.model, sim.data)
            r.update_scene(sim.data, camera="catalog_cam", scene_option=sim._default_opt)
            img = r.render()
            ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            if ok:
                self.thumbs[name] = base64.b64encode(buf.tobytes()).decode("ascii")
            sim.data.qpos[:] = q
        sim.data.qpos[:] = saved_qpos; sim.data.qvel[:] = saved_qvel
        mujoco.mj_forward(sim.model, sim.data)
        r.close()

    def catalog_payload(self) -> dict:
        sim = self.sim
        parts = []
        for name, spec in self.catalog.items():
            bound = [o for o in sim.objects if sim.active_body[o] == name and sim.present(o)]
            in_layout = any(l["part"] == name for l in self.layout)
            parts.append({"name": name, **spec, "thumbnail_png_b64": self.thumbs.get(name), "on_table": bool(bound) or in_layout,
                          "slot_bound": bound[0] if bound else None, "removable": True})
        return {"parts": parts, "variants": {o: (sim.active_body[o] if sim.present(o) else None) for o in sim.objects},
                "absent_slots": sorted(sim.absent), "layout": list(self.layout), "table_bounds": {"x": TABLE_X, "y": TABLE_Y},
                "task_objects": list(sim.objects), "on_table": [p["name"] for p in parts if p["on_table"]]}

    # ------------------------------------------------------------------ authoring
    def drop_point(self, u: float, v: float) -> np.ndarray:
        svc = self.svc
        p = self.sim.pixel_to_table(svc.view_cam, svc.view_w, svc.view_h, float(u), float(v))
        if p is None:
            raise ValueError("drop point is not on the table plane")
        x = float(np.clip(p[0], *TABLE_X)); y = float(np.clip(p[1], *TABLE_Y))
        return np.array([x, y])

    def _random_spot(self) -> np.ndarray:
        rng = np.random.default_rng()
        return np.array([0.15 + rng.uniform(-0.14, 0.14), rng.uniform(-0.22, 0.22)])

    def _refresh_target(self):
        env = self.svc.env
        env.reward_fn.reset(); env._grasp_streak = 0
        if env.target is None or not self.sim.present(env.target) or env.stage[env.target].placed:
            env.target = env._pick_target(); self.sim.track_target = env.target
        self.svc.obs = env._get_obs()

    def spawn(self, part: str, u: float | None = None, v: float | None = None, x: float | None = None, y: float | None = None, yaw: float | None = None) -> dict:
        """Put a catalog part on the table (select it). Variants bind to their task slot if the slot is empty; any
        further variant of the same type is an extra (untracked) part. Already-selected parts are moved."""
        if part not in self.catalog:
            raise ValueError(f"unknown part {part}; options: {list(self.catalog)}")
        spec = self.catalog[part]
        xy = self.drop_point(u, v) if (u is not None and v is not None) else (np.array([float(np.clip(x, *TABLE_X)), float(np.clip(y, *TABLE_Y))]) if x is not None and y is not None else self._random_spot())
        yaw = float(yaw) if yaw is not None else float(np.random.default_rng().uniform(-math.pi, math.pi))
        sim, env = self.sim, self.svc.env
        kind = spec.get("role")
        taken = [sim.obj_pos(o)[:2].copy() for o in sim.objects if sim.present(o) and sim.active_body[o] != part]
        taken += [np.array([l["x"], l["y"]]) for l in self.layout if l["part"] != part]
        xy = self._free_spot(np.asarray(xy, dtype=float), taken)
        if spec.get("role") == "variant":
            slot = spec["slot"]
            if not sim.present(slot):                      # slot empty -> this part becomes the tracked task object
                sim.place_body(part, xy, yaw, settle_steps=40)
                sim.bind_variant(slot, part)
                env.stage[slot].__init__()
                kind = "task"
            elif sim.active_body[slot] == part:            # already the tracked object -> just move it
                sim.place_body(part, xy, yaw, settle_steps=40)
                env.stage[slot].__init__()
                kind = "task"
            else:                                          # another variant already tracked -> extra part
                sim.place_body(part, xy, yaw, settle_steps=40)
                self.layout = [l for l in self.layout if l["part"] != part] + [{"part": part, "x": round(float(xy[0]), 4), "y": round(float(xy[1]), 4), "yaw": round(yaw, 4)}]
                kind = "extra"
        else:
            sim.place_body(part, xy, yaw, settle_steps=40)
            self.layout = [l for l in self.layout if l["part"] != part] + [{"part": part, "x": round(float(xy[0]), 4), "y": round(float(xy[1]), 4), "yaw": round(yaw, 4)}]
        self._refresh_target()
        return {"part": part, "kind": kind, "x": round(float(xy[0]), 3), "y": round(float(xy[1]), 3), "yaw": round(yaw, 3),
                "resting_z": round(float(sim.data.xpos[sim.model.body(part).id][2]), 4)}

    def remove(self, part: str) -> dict:
        """Deselect a part (park it). If it was the tracked object of a slot, another on-table variant of that type
        takes over; otherwise the slot becomes empty (valid: a table with no bracket / no bolt)."""
        if part not in self.catalog:
            raise ValueError(f"unknown part {part}")
        sim, env = self.sim, self.svc.env
        spec = self.catalog[part]
        was_layout = any(l["part"] == part for l in self.layout)
        self.layout = [l for l in self.layout if l["part"] != part]
        sim.park_body(part)
        for slot in sim.objects:
            if sim.active_body[slot] == part and sim.present(slot):
                # hand the slot to another selected variant of the same type, else mark it absent
                cand = [l for l in self.layout if self.catalog.get(l["part"], {}).get("slot") == slot]
                if cand:
                    nxt = cand[0]["part"]
                    self.layout = [l for l in self.layout if l["part"] != nxt]
                    sim.bind_variant(slot, nxt)
                else:
                    sim.absent.add(slot)
                env.stage[slot].__init__()
        self._refresh_target()
        return {"removed": part, "was_extra": was_layout, "absent_slots": sorted(sim.absent)}

    def toggle(self, part: str, on: bool | None = None, **kw) -> dict:
        on_now = part in self.catalog_payload()["on_table"]
        want = (not on_now) if on is None else bool(on)
        if want:
            return {"on": True, **self.spawn(part, **kw)}
        if on_now:
            return {"on": False, **self.remove(part)}
        return {"on": False, "part": part}

    def clear(self) -> dict:
        """Clear the table completely: every extra part AND both task objects are parked (empty table is valid)."""
        removed = [l["part"] for l in self.layout]
        for l in self.layout:
            self.sim.park_body(l["part"])
        self.layout = []
        for slot in self.sim.objects:
            if self.sim.present(slot):
                removed.append(self.sim.active_body[slot])
                self.sim.park_body(self.sim.active_body[slot])
                self.sim.absent.add(slot)
                self.svc.env.stage[slot].__init__()
        self._refresh_target()
        return {"removed": removed, "absent_slots": sorted(self.sim.absent)}

    def ensure_defaults(self, only_if_empty: bool = False, keep_extras: bool = False) -> dict:
        """The safe demo state: default bracket + default bolt + the two bins and NOTHING else. Every extra /
        rejection part is parked (keep_extras=True is only used when a saved scenario restores its own layout)."""
        sim = self.sim
        if only_if_empty and any(sim.present(o) for o in sim.objects):
            return {"variants": dict(sim.active_body)}
        if not keep_extras:
            for l in self.layout:
                sim.park_body(l["part"])
            self.layout = []
        for slot in sim.objects:
            if sim.present(slot) and sim.active_body[slot] != slot:
                sim.park_body(sim.active_body[slot])
            if not sim.present(slot) or sim.active_body[slot] != slot:
                sim.place_body(slot, self._random_spot(), 0.0, settle_steps=0)
                sim.bind_variant(slot, slot)
        self.layout = [l for l in self.layout if self.catalog.get(l["part"], {}).get("role") != "variant"]
        return {"variants": dict(sim.active_body), "extras": [l["part"] for l in self.layout]}

    def reset_variants(self) -> dict:
        return self.ensure_defaults()

    MIN_CLEARANCE = 0.11    # m, centre-to-centre; the largest part is 120 mm long

    def _clear_of_bins(self, xy: np.ndarray) -> bool:
        """A part centre must stay outside every bin footprint + 7 cm (bin walls are where parts got wedged)."""
        for b in self.sim.bins:
            c = self.sim.bin_pos(b); h = self.sim.bin_half_extents(b)
            if abs(xy[0] - c[0]) < h[0] + 0.07 and abs(xy[1] - c[1]) < h[1] + 0.07:
                return False
        return True

    def _ok(self, xy: np.ndarray, taken: list[np.ndarray]) -> bool:
        return self._clear_of_bins(xy) and all(np.linalg.norm(xy - t) >= self.MIN_CLEARANCE for t in taken)

    def _free_spot(self, xy: np.ndarray, taken: list[np.ndarray]) -> np.ndarray:
        """Nearest spot to xy (on a growing ring) that is clear of the bins, MIN_CLEARANCE from every taken position
        and on the table. Falls back to the spawn-zone centre if nothing nearby is free."""
        if self._ok(xy, taken):
            return xy
        for r in (0.12, 0.16, 0.20, 0.25, 0.30, 0.36):
            for k in range(12):
                ang = 2 * math.pi * k / 12
                cand = np.array([float(np.clip(xy[0] + r * math.cos(ang), *TABLE_X)), float(np.clip(xy[1] + r * math.sin(ang), *TABLE_Y))])
                if self._ok(cand, taken):
                    return cand
        return np.array([0.15, 0.0])

    def after_reset(self):
        """Re-apply the authored extra parts after every env reset (mj_resetData parks everything). The task objects
        are randomized by the reset, so an extra part whose stored spot now overlaps one is nudged to a free spot -
        spawning parts inside each other was the source of exploding contacts."""
        sim = self.sim
        taken = [sim.obj_pos(o)[:2].copy() for o in sim.objects if sim.present(o)]
        for l in self.layout:
            xy = self._free_spot(np.array([l["x"], l["y"]]), taken)
            sim.place_body(l["part"], xy, l["yaw"], settle_steps=0)
            taken.append(xy)
        if self.layout:
            import mujoco
            mujoco.mj_step(sim.model, sim.data, nstep=30)

    # ------------------------------------------------------------------ scenarios
    def current_layout(self) -> dict:
        sim = self.sim
        poses = {o: {"x": round(float(sim.obj_pos(o)[0]), 4), "y": round(float(sim.obj_pos(o)[1]), 4), "yaw": round(float(sim.obj_yaw(o)), 4)} for o in sim.objects}
        return {"poses": poses, "variants": dict(sim.active_body), "layout": list(self.layout)}

    def save_scenario(self, name: str) -> dict:
        key = "".join(ch if ch.isalnum() or ch in "-_ " else "_" for ch in name).strip().replace(" ", "_").lower()
        if not key:
            raise ValueError("empty scenario name")
        d = load_custom_scenarios()
        d[key] = {**self.current_layout(), "saved_iso": time.strftime("%Y-%m-%dT%H:%M:%S"), "display_name": name}
        save_custom_scenarios(d)
        return {"name": key, "scenario": d[key]}

    def delete_scenario(self, name: str) -> dict:
        d = load_custom_scenarios()
        if name not in d:
            raise FileNotFoundError(name)
        d.pop(name); save_custom_scenarios(d)
        return {"deleted": name}

    def apply(self, scenario: dict):
        """Variants + clutter layout of a saved scenario (object poses are applied by the subsequent reset)."""
        for o, body in (scenario.get("variants") or {}).items():
            if o in self.sim.objects and body in self.catalog:
                if not self.sim.present(o):
                    self.sim.place_body(body, self._random_spot(), 0.0, settle_steps=0); self.sim.bind_variant(o, body)
                elif self.sim.active_body[o] != body:
                    self.sim.set_variant(o, body)
        for l in self.layout:
            self.sim.park_body(l["part"])
        self.layout = [dict(l) for l in (scenario.get("layout") or [])]
        self._scenario_keep_extras = True


def apply_scenario_layout(sim, scenario: dict):
    sb = getattr(sim, "scene_builder", None)
    if sb is not None:
        sb.apply(scenario)
