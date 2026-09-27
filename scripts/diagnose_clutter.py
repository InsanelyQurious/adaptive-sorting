"""Clutter instability diagnosis: run batches with/without extra parts under random actions, the scripted controller
and the trained policy; count sim-unstable episodes, NaN, max |qvel|, collisions, success."""
import os, sys, argparse, json
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, mujoco
from simulation.environments.sorting_env import make_env
from backend.policy_loader import LoadedPolicy, resolve_checkpoint
from training.demos.scripted_controller import ScriptedSortingController

ap = argparse.ArgumentParser(); ap.add_argument("--episodes", type=int, default=12); ap.add_argument("--checkpoint", default="best"); args = ap.parse_args()
env = make_env(sb3=False); u = env.unwrapped; sim = u.sim
pol = LoadedPolicy(resolve_checkpoint(args.checkpoint))
CLUTTER = [("bracket_extra", (0.05, 0.20), 0.5), ("reject_cap", (0.24, -0.17), -2.4)]

def run(label, driver, clutter, n):
    rows = []; warn = []
    for ep in range(n):
        obs, info = env.reset(seed=9000 + ep)
        for part, xy, yaw in clutter:
            sim.place_body(part, xy, yaw, settle_steps=0)
        if clutter:
            mujoco.mj_step(sim.model, sim.data, nstep=30)
        ctrl = ScriptedSortingController(env) if driver == "scripted" else None
        rng = np.random.default_rng(ep); maxv = 0.0; coll = 0; unstable = False
        while True:
            if driver == "random": a = np.clip(rng.normal(0, 0.6, 7), -1, 1)
            elif driver == "scripted": a = ctrl.act()
            else: a = pol.select_action(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            maxv = max(maxv, float(np.abs(sim.data.qvel).max()))
            if info.get("unstable"): unstable = True
            if term or trunc: break
        # which bodies are moving fast / NaN at the end?
        rows.append({"success": info["is_success"], "placed": info["placed_count"], "collisions": info["collisions"], "unstable": unstable, "dropped": info["dropped"], "maxv": round(maxv, 1), "len": info["step"]})
        for part, _, _ in clutter: sim.park_body(part)
    n = len(rows)
    print(f"{label:38s} success {sum(r['success'] for r in rows)/n:.0%}  placed/ep {sum(r['placed'] for r in rows)/n:.2f}  unstable {sum(r['unstable'] for r in rows)}/{n}  dropped {sum(r['dropped'] for r in rows)}  coll/ep {sum(r['collisions'] for r in rows)/n:.1f}  max|qvel| {max(r['maxv'] for r in rows):.1f}  len {sum(r['len'] for r in rows)/n:.0f}")
    return rows

for driver in ("random", "scripted", "policy"):
    run(f"{driver:9s} | clean (bracket+bolt)", driver, [], args.episodes)
    run(f"{driver:9s} | + extra L-bracket + red cap", driver, CLUTTER, args.episodes)
    run(f"{driver:9s} | + red cap only", driver, CLUTTER[1:], args.episodes)
    run(f"{driver:9s} | + extra L-bracket only", driver, CLUTTER[:1], args.episodes)
env.close()
