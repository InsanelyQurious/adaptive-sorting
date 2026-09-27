"""Probe: (1) scripted carry+release -> does placement detection fire and what reward does it pay?
(2) policy behaviour from curriculum starts (grasped / overbin) vs. from home.
(3) reward accounting of a grasp-lift-release-regrasp cycle (is the shaping exploitable?)."""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from simulation.environments.sorting_env import make_env
from backend.policy_loader import LoadedPolicy, resolve_checkpoint

pol = LoadedPolicy(resolve_checkpoint(sys.argv[1] if len(sys.argv) > 1 else "latest"))
env = make_env(sb3=False); u = env.unwrapped; sim = u.sim

def act(a):
    obs, r, term, trunc, info = env.step(np.asarray(a, dtype=np.float32)); return obs, r, term, trunc, info

# ---------- (1) scripted placement detection ----------
print("== (1) scripted carry + release ==")
placed_fired = 0; rewards = []
for ep in range(6):
    obs, info = env.reset(seed=500 + ep, options={"curriculum": True})
    # force a grasped start regardless of the curriculum dice
    u.stage = {o: type(u.stage[o])() for o in u.objects}; u.reward_fn.reset(); u._grasp_streak = 0
    u._scripted_pregrasp(u.target, close=True)
    tgt = u.target
    if not (sim.is_grasped(tgt)):
        print(f"  ep{ep}: scripted grasp failed"); continue
    total = 0.0; comps = {}
    # lift, carry above bin, release
    b = sim.bin_pos(u.bin_mapping[tgt])
    for t in range(60):
        goal = np.array([b[0], b[1], 1.02]); d = goal - sim.target_pos
        a = np.zeros(7); a[:3] = np.clip(d / sim.max_dpos, -1, 1); a[6] = 1.0
        if np.linalg.norm(d[:2]) < 0.02 and t > 8:
            a[6] = -1.0  # release
        obs, r, term, trunc, info = act(a); total += r
        for k, v in info["reward_components"].items(): comps[k] = comps.get(k, 0) + v
        if info["placed_count"] > 0:
            placed_fired += 1; break
    print(f"  ep{ep}: target={tgt} placed_count={info['placed_count']} which_bin={sim.which_bin(tgt)} obj_z={sim.obj_pos(tgt)[2]:.3f} steps={t+1} R={total:+.1f} "
          f"comps={{{', '.join(f'{k}:{v:+.1f}' for k, v in comps.items() if abs(v) > 0.05)}}}")
print(f"  placement fired in {placed_fired}/6 scripted carries")

# ---------- (2) policy from curriculum starts ----------
print("\n== (2) policy behaviour from curriculum starts (deterministic) ==")
for kind, kwargs in (("grasped", dict(close=True)), ("overbin", dict(close=True, transport=True)), ("straddle", dict(close=False, descend=True))):
    n_ok = 0; n_place = 0; n_drop_immediately = 0; first_open = []
    for ep in range(12):
        obs, info = env.reset(seed=700 + ep)
        u._scripted_pregrasp(u.target, **kwargs); tgt = u.target
        if kind != "straddle" and not sim.is_grasped(tgt):
            continue
        n_ok += 1
        obs = u._get_obs()
        opened_at = None
        for t in range(80):
            a = pol.select_action(obs)
            obs, r, term, trunc, info = act(a)
            if a[6] <= 0 and opened_at is None:
                opened_at = t
            if info["placed_count"] > 0 or term or trunc:
                break
        n_place += info["placed_count"] > 0
        first_open.append(opened_at if opened_at is not None else -1)
        if kind != "straddle" and opened_at is not None and opened_at <= 3:
            n_drop_immediately += 1
    print(f"  {kind:8s}: valid starts {n_ok}/12, placed {n_place}, opened gripper within 3 steps: {n_drop_immediately}, first-open step per ep: {first_open}")

# ---------- (3) reward exploit check: grasp-lift-release cycles ----------
print("\n== (3) scripted grasp -> lift -> release -> regrasp cycles: cumulative shaping ==")
obs, info = env.reset(seed=900)
u._scripted_pregrasp(u.target, close=True); tgt = u.target
comps = {}; total = 0.0
for cycle in range(4):
    for _ in range(4):   # lift
        obs, r, term, trunc, info = act([0, 0, 1.0, 0, 0, 0, 1.0]); total += r
        for k, v in info["reward_components"].items(): comps[k] = comps.get(k, 0) + v
    obs, r, term, trunc, info = act([0, 0, 0, 0, 0, 0, -1.0]); total += r   # release
    for k, v in info["reward_components"].items(): comps[k] = comps.get(k, 0) + v
    p = sim.obj_pos(tgt)
    for _ in range(6):   # descend and re-grasp
        d = np.array([p[0], p[1], sim.ws_lo[2]]) - sim.target_pos
        a = np.zeros(7); a[:3] = np.clip(d / sim.max_dpos, -1, 1); a[6] = -1.0
        obs, r, term, trunc, info = act(a); total += r
        for k, v in info["reward_components"].items(): comps[k] = comps.get(k, 0) + v
    for _ in range(3):
        obs, r, term, trunc, info = act([0, 0, 0, 0, 0, 0, 1.0]); total += r
        for k, v in info["reward_components"].items(): comps[k] = comps.get(k, 0) + v
    print(f"  after cycle {cycle+1}: grasped={info['grasped_now']} total={total:+.2f} comps={{{', '.join(f'{k}:{v:+.2f}' for k, v in comps.items() if abs(v) > 0.05)}}}")
env.close()
