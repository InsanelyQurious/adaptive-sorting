"""Regression tests for the placement diagnosis (DEBUGGING.md): the shaping must not be farmable by
grasp -> lift -> release cycles, and a real carry + release must still pay the placement bonus."""
import numpy as np
from simulation.environments.workcell_sim import wrap_angle


def _scripted_grasp(env, obj):
    sim = env.sim
    sim.track_target = obj
    p = sim.obj_pos(obj); yaw = wrap_angle(sim.obj_yaw(obj))
    sim.set_gripper(0.0)
    sim.move_to(p + [0, 0, 0.2], yaw, max_steps=60)
    sim.move_to([p[0], p[1], sim.ws_lo[2]], yaw, max_steps=40, tol=0.004)
    for _ in range(4):
        env.step(np.array([0, 0, 0, 0, 0, 0, 1.0]))


def _acc(env, a, comps, n=1):
    total = 0.0
    for _ in range(n):
        obs, r, term, trunc, info = env.step(np.asarray(a, dtype=np.float32)); total += r
        for k, v in info["reward_components"].items():
            comps[k] = comps.get(k, 0.0) + v
    return total, info


def test_grasp_release_cycles_do_not_pump_reward(env):
    """Each extra lift/drop/re-grasp cycle must be net negative (potentials are charged back on drop)."""
    env.reset(seed=11)
    tgt = env.target
    _scripted_grasp(env, tgt)
    comps = {}
    totals = []
    for cycle in range(3):
        cyc = 0.0
        cyc += _acc(env, [0, 0, 1.0, 0, 0, 0, 1.0], comps, n=4)[0]      # lift
        cyc += _acc(env, [0, 0, 0, 0, 0, 0, -1.0], comps, n=1)[0]       # release
        p = env.sim.obj_pos(tgt)
        for _ in range(6):                                              # descend + re-grasp
            d = np.array([p[0], p[1], env.sim.ws_lo[2]]) - env.sim.target_pos
            a = np.zeros(7); a[:3] = np.clip(d / env.sim.max_dpos, -1, 1); a[6] = -1.0
            cyc += _acc(env, a, comps)[0]
        cyc += _acc(env, [0, 0, 0, 0, 0, 0, 1.0], comps, n=3)[0]
        totals.append(cyc)
    # the first cycle may contain the one-time lift bonus; later cycles must lose money
    assert all(t < 0 for t in totals[1:]), f"reward pump still possible: cycle rewards {totals}"
    assert comps.get("lift_shaping", 0.0) <= 0.5, f"lift shaping accumulated over cycles: {comps.get('lift_shaping')}"
    assert comps.get("approach_reward", 0.0) <= 0.5, "approach potential re-earned after a release"


def test_release_over_correct_bin_is_not_penalised_and_places(env):
    env.reset(seed=12)
    sim = env.sim; tgt = env.target
    _scripted_grasp(env, tgt)
    comps = {}
    b = sim.bin_pos(env.bin_mapping[tgt])
    for _ in range(80):
        d = np.array([b[0], b[1], 1.02]) - sim.target_pos
        a = np.zeros(7); a[:3] = np.clip(d / sim.max_dpos, -1, 1); a[6] = 1.0
        total, info = _acc(env, a, comps)
        if np.linalg.norm(d[:2]) < 0.01:
            break
    assert info["grasped_now"] and info["lifted_now"]
    rel = 0.0
    for _ in range(12):
        t, info = _acc(env, [0, 0, 0, 0, 0, 0, -1.0], comps); rel += t
        if info["placed_count"] >= 1:
            break
    assert info["placed_count"] >= 1
    assert comps.get("release_penalty", 0.0) == 0.0, "releasing over the correct bin must not be penalised"
    assert rel > 0.9 * env.env_cfg["reward"]["placement_reward"], f"placement step paid only {rel}"


def test_release_away_from_bin_is_penalised_and_charges_back_lift(env):
    env.reset(seed=13)
    tgt = env.target
    _scripted_grasp(env, tgt)
    comps = {}
    _acc(env, [0, 0, 1.0, 0, 0, 0, 1.0], comps, n=4)
    lift_gain = comps.get("lift_shaping", 0.0)
    assert lift_gain > 1.0
    _acc(env, [0, 0, 0, 0, 0, 0, -1.0], comps, n=3)
    assert comps["release_penalty"] < 0
    assert comps["lift_shaping"] < 0.3 * lift_gain, "dropping the part must give the lift progress back"
