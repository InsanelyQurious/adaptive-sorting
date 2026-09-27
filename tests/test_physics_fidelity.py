"""Physics / randomization fidelity (Incremental Update 6).

* every catalog part (variants, clutter, rejection cap) rests on the table without falling through, has collision
  geometry that the gripper can actually grasp (pad contacts + is_grasped), and can be dropped into a bin and stay inside
* across a batch of episodes with random actions: no NaN / exploding velocities, no object below the table top,
  objects stay within the table bounds while resting
* bin positions never change within an episode (only at reset), and stay at nominal for scenario presets
"""
import numpy as np
import pytest
import mujoco

from simulation.environments.sorting_env import make_env
from simulation.environments.workcell_sim import wrap_angle

TABLE_TOP = 0.80
TABLE_X = (-0.9, 0.7)
TABLE_Y = (-0.6, 0.6)
CATALOG_PARTS = ["bracket_flat", "bracket_u", "bolt_socket", "bolt_m12", "bracket_extra", "bolt_extra", "reject_cap"]


@pytest.fixture(scope="module")
def penv():
    e = make_env(sb3=False)
    yield e
    e.close()


def _rest_check(sim, body):
    p = sim.data.xpos[sim.model.body(body).id]
    assert np.all(np.isfinite(sim.data.qpos)) and np.all(np.isfinite(sim.data.qvel)), "NaN in state"
    assert p[2] > TABLE_TOP - 0.003, f"{body} fell through the table (z={p[2]:.4f})"
    assert p[2] < TABLE_TOP + 0.15, f"{body} floating (z={p[2]:.4f})"
    assert TABLE_X[0] < p[0] < TABLE_X[1] and TABLE_Y[0] < p[1] < TABLE_Y[1], f"{body} left the table at {p[:2]}"


def test_every_catalog_part_rests_on_table_and_is_graspable(penv):
    u = penv.unwrapped; sim = u.sim
    for body in CATALOG_PARTS:
        penv.reset(seed=77)
        # move the two task objects to far corners so the part under test is alone in the grasp area
        sim.place_body(sim.active_body["bracket"], (-0.55, 0.45), 0.0, settle_steps=0)
        sim.place_body(sim.active_body["bolt"], (-0.55, -0.45), 0.0, settle_steps=30)
        sim.place_body(body, (0.20, 0.05), 0.4, settle_steps=200)
        _rest_check(sim, body)
        dof = int(sim.model.jnt_dofadr[sim.model.joint(f"{body}_free").id])
        assert np.abs(sim.data.qvel[dof:dof + 6]).max() < 0.05, f"{body} still moving after settling"
        # grasp with the parallel-jaw gripper through the normal controller (same as click-to-pick / scripted demos)
        p = sim.data.xpos[sim.model.body(body).id].copy()
        yaw = wrap_angle(float(np.arctan2(*sim.data.xmat[sim.model.body(body).id].reshape(3, 3)[[1, 0], 0])))
        sim.set_gripper(0.0)
        sim.move_to(p + [0, 0, 0.18], yaw, max_steps=60)
        sim.move_to([p[0], p[1], sim.ws_lo[2]], yaw, max_steps=40, tol=0.004)
        for _ in range(8):
            sim.apply_action(np.array([0, 0, 0, 0, 0, 0, 1.0]))
        assert sim.pads_grasping(body), f"gripper pads do not both contact {body}: collision geometry / grasp width problem"
        for _ in range(12):
            sim.apply_action(np.array([0, 0, 0.6, 0, 0, 0, 1.0]))
        z = sim.data.xpos[sim.model.body(body).id][2]
        assert z > TABLE_TOP + 0.06, f"{body} was not lifted by friction contacts (z={z:.3f})"
        # carry over bin A and release: the part must end up inside the bin (walls/floor collide) and not clip through
        b = sim.bin_pos("bin_a")
        for _ in range(60):
            d = np.array([b[0], b[1], 1.0]) - sim.target_pos
            a = np.zeros(7); a[:3] = np.clip(d / sim.max_dpos, -1, 1); a[6] = 1.0
            sim.apply_action(a)
            if np.linalg.norm(d[:2]) < 0.01:
                break
        for _ in range(25):
            sim.apply_action(np.array([0, 0, 0, 0, 0, 0, -1.0]))
        pb = sim.data.xpos[sim.model.body(body).id]; h = sim.bin_half_extents("bin_a")
        assert abs(pb[0] - b[0]) <= h[0] + 0.02 and abs(pb[1] - b[1]) <= h[1] + 0.02, f"{body} not inside bin A footprint after release: {pb}"
        assert TABLE_TOP - 0.002 < pb[2] < TABLE_TOP + 0.13, f"{body} clipped through the bin floor/walls (z={pb[2]:.3f})"
        sim.park_body(body)


def test_random_action_batch_no_explosions_no_fallthrough(penv):
    u = penv.unwrapped; sim = u.sim
    rng = np.random.default_rng(3)
    for ep in range(6):
        penv.reset(seed=500 + ep)
        clutter = CATALOG_PARTS[ep % len(CATALOG_PARTS)]
        sim.place_body(clutter, (0.05, -0.15 + 0.05 * ep), rng.uniform(-3, 3), settle_steps=30)
        bins0 = {b: sim.bin_pos(b).copy() for b in sim.bins}
        for t in range(120):
            a = np.clip(rng.normal(0, 0.6, 7), -1, 1)
            obs, r, term, trunc, info = penv.step(a)
            assert np.all(np.isfinite(sim.data.qpos)) and np.all(np.isfinite(sim.data.qvel)), f"NaN at ep {ep} t {t}"
            assert np.abs(sim.data.qvel).max() < 50, f"exploding velocity at ep {ep} t {t}: {np.abs(sim.data.qvel).max():.1f}"
            for body in list(u.objects) + [clutter]:
                bid = sim.obj_body_id[body] if body in u.objects else sim.model.body(body).id
                z = sim.data.xpos[bid][2]
                assert z > TABLE_TOP - 0.003 or z < 0.3 or info["dropped"], f"{body} penetrated the table at ep {ep} t {t} (z={z:.4f})"
            for b in sim.bins:
                assert np.allclose(sim.bin_pos(b), bins0[b], atol=1e-9), f"bin {b} moved mid-episode at t {t}"
            if term or trunc:
                break
        sim.park_body(clutter)


def test_bins_move_only_at_reset_and_stay_fixed_for_scenarios(penv):
    u = penv.unwrapped; sim = u.sim
    nominal = {b: sim.randomizer._nominal_bin_pos[b].copy() for b in sim.bins}
    # randomized resets: bins may move (within jitter) between episodes ...
    penv.reset(seed=1); p1 = {b: sim.bin_pos(b).copy() for b in sim.bins}
    for _ in range(50):
        penv.step(np.zeros(7))
        assert all(np.allclose(sim.bin_pos(b), p1[b], atol=1e-9) for b in sim.bins)
    assert sim.episode_config.bins_fixed is False
    jitter = np.array(u.env_cfg["randomization"]["bin_position_jitter"])
    for b in sim.bins:
        assert np.all(np.abs(p1[b][:2] - nominal[b][:2]) <= jitter + 1e-9)
    # ... scenario presets (fixed poses) keep the bins exactly at nominal on every reset
    poses = {"bracket": {"x": 0.12, "y": 0.10, "yaw": 0.0}, "bolt": {"x": 0.26, "y": -0.10, "yaw": 0.0}}
    for seed in (5, 6, 7):
        penv.reset(seed=seed, options={"fixed_poses": poses})
        assert sim.episode_config.bins_fixed is True
        for b in sim.bins:
            assert np.allclose(sim.bin_pos(b)[:2], nominal[b][:2], atol=1e-9), f"bin {b} moved in a fixed scenario"
    # explicit override: fixed poses but randomized bins on request
    penv.reset(seed=8, options={"fixed_poses": poses, "fixed_bins": False})
    assert sim.episode_config.bins_fixed is False


def test_clutter_next_to_bins_and_parts_stays_stable_with_scripted_controller(penv):
    """Extra / rejection parts placed in the transport paths (next to the bins) and beside a task object: the scripted
    controller sorts both task parts through them without a single unstable step (soft priority contacts on clutter)."""
    from training.demos.scripted_controller import ScriptedSortingController
    u = penv.unwrapped; sim = u.sim
    unstable = 0; successes = 0
    for ep in range(6):
        penv.reset(seed=700 + ep)
        b = sim.obj_pos("bracket")
        for part, xy, yaw in (("bracket_extra", (0.2, 0.30), 0.4), ("reject_cap", (0.2, -0.30), -1.2), ("bolt_extra", (float(np.clip(b[0] + 0.09, 0.0, 0.4)), float(b[1])), 1.0)):
            sim.place_body(part, xy, yaw, settle_steps=0)
        mujoco.mj_step(sim.model, sim.data, nstep=30)
        ctrl = ScriptedSortingController(penv)
        while True:
            obs, r, term, trunc, info = penv.step(ctrl.act())
            assert np.abs(sim.data.qvel).max() < 60, f"velocity spike with clutter at ep {ep} t {info['step']}"
            if term or trunc:
                break
        unstable += int(info["unstable"]); successes += int(info["is_success"])
        for part in ("bracket_extra", "reject_cap", "bolt_extra"):
            sim.park_body(part)
    assert unstable == 0
    assert successes >= 4, f"scripted controller should still sort with clutter present ({successes}/6)"


def test_clutter_spawned_through_a_bin_wall_does_not_explode(penv):
    """The live failure mode: an extra part re-spawned overlapping a bin wall at every reset. Soft contacts must let it
    settle instead of diverging (the scene builder also keeps parts clear of bins, this covers the physics side)."""
    u = penv.unwrapped; sim = u.sim
    for ep in range(4):
        penv.reset(seed=ep)
        c = sim.bin_pos("bin_a")
        sim.place_body("bracket_extra", (c[0], c[1] - 0.106), 0.3, settle_steps=0)
        mujoco.mj_step(sim.model, sim.data, nstep=30)
        for _ in range(40):
            obs, r, term, trunc, info = penv.step(np.zeros(7))
            assert not info["unstable"] and np.abs(sim.data.qvel).max() < 60
        sim.park_body("bracket_extra")


def test_scene_builder_free_spot_avoids_bins_and_parts(penv):
    from backend.scene_builder import SceneBuilder
    u = penv.unwrapped; sim = u.sim
    penv.reset(seed=3)
    sb = SceneBuilder.__new__(SceneBuilder); sb.sim = sim; sb.layout = []
    ca = sim.bin_pos("bin_a")
    spot = sb._free_spot(np.array([ca[0], ca[1] - 0.10]), [])          # on bin A's wall -> must move away
    assert sb._clear_of_bins(spot)
    t = sim.obj_pos("bracket")[:2]
    spot2 = sb._free_spot(t.copy(), [t])                                  # exactly on the bracket -> must move >= 11 cm
    assert np.linalg.norm(spot2 - t) >= SceneBuilder.MIN_CLEARANCE - 1e-9


def test_object_damping_is_configured_on_every_free_part(penv):
    """physics.object_damping (simulation.yaml) must be applied to task objects, variants and extra parts alike."""
    sim = penv.unwrapped.sim; m = sim.model
    lin, ang = sim.object_damping
    assert lin > 0 and ang > 0
    for name in ["bracket", "bolt"] + CATALOG_PARTS:
        d0 = int(m.jnt_dofadr[m.joint(f"{name}_free").id])
        assert np.allclose(m.dof_damping[d0:d0 + 3], lin) and np.allclose(m.dof_damping[d0 + 3:d0 + 6], ang), name
