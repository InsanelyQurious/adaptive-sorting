"""Baseline sim-instability rate of the trained policy on CLEAN scenes, and candidate physics fixes measured on the
same seeds. Reports unstable episodes, max |qvel|, success and which body diverged."""
import os, sys, argparse
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, mujoco
from simulation.environments.sorting_env import make_env
from backend.policy_loader import LoadedPolicy, resolve_checkpoint
ap = argparse.ArgumentParser(); ap.add_argument("--episodes", type=int, default=60); ap.add_argument("--fix", default="none"); ap.add_argument("--seed", type=int, default=20000); args = ap.parse_args()
env = make_env(sb3=False); u = env.unwrapped; sim = u.sim; m = sim.model
pol = LoadedPolicy(resolve_checkpoint("best"))
if "damp" in args.fix:
    lin, ang = (0.3, 0.002) if "light" in args.fix else (0.5, 0.005)
    for name in ("bracket", "bolt"):
        d0 = m.jnt_dofadr[m.joint(f"{name}_free").id]; m.dof_damping[d0:d0 + 6] = [lin, lin, lin, ang, ang, ang]
if "softbin" in args.fix:
    for g in range(m.ngeom):
        if m.geom(g).name.startswith("bin_") and (m.geom_contype[g] or m.geom_conaffinity[g]):
            m.geom_priority[g] = 1; m.geom_solref[g] = [0.03, 1.0]; m.geom_solimp[g] = [0.9, 0.95, 0.01, 0.5, 2]
if "thickbin" in args.fix:
    for g in range(m.ngeom):
        n = m.geom(g).name
        if n.startswith("bin_") and ("_w" in n) and (m.geom_contype[g] or m.geom_conaffinity[g]):
            m.geom_size[g] = [max(m.geom_size[g][0], 0.008) if m.geom_size[g][0] < 0.01 else m.geom_size[g][0], max(m.geom_size[g][1], 0.008) if m.geom_size[g][1] < 0.01 else m.geom_size[g][1], m.geom_size[g][2]]
if "softpad" in args.fix:
    for g in (m.geom("left_pad").id, m.geom("right_pad").id):
        m.geom_solref[g] = [0.03, 1.0]; m.geom_solimp[g] = [0.9, 0.95, 0.01, 0.5, 2]; m.geom_priority[g] = 1
if "gripforce" in args.fix:
    gid = m.actuator("gripper").id; F = 30.0
    m.actuator_forcerange[gid] = [-F, F]; m.actuator_forcelimited[gid] = 1
if "impratio" in args.fix:
    m.opt.impratio = 1.0
if "noslip" in args.fix:
    m.opt.noslip_iterations = 0
if "dt001" in args.fix:
    m.opt.timestep = 0.001; sim.physics_dt = 0.001; sim.n_substeps = int(round(sim.control_dt / 0.001))
if "padsoftish" in args.fix:
    for g in (m.geom("left_pad").id, m.geom("right_pad").id):
        m.geom_solref[g] = [0.02, 1.0]; m.geom_solimp[g] = [0.8, 0.9, 0.01, 0.5, 2]
if "inertia" in args.fix:
    b = m.body("bolt").id; m.body_inertia[b] = m.body_inertia[b] * 3.0
rows = []
for ep in range(args.episodes):
    obs, info = env.reset(seed=args.seed + ep)
    maxv = 0.0; fast = None; grasped_at_blow = None
    while True:
        a = pol.select_action(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        v = float(np.abs(sim.data.qvel).max())
        if v > maxv:
            maxv = v
            if v > 60 and fast is None:
                dof = int(np.argmax(np.abs(sim.data.qvel))); body = m.body(m.dof_bodyid[dof]).name
                d = sim.data; pairs = {}
                for i in range(d.ncon):
                    g1, g2 = d.contact.geom1[i], d.contact.geom2[i]
                    n1, n2 = m.geom(g1).name, m.geom(g2).name
                    if body in n1 or body in n2 or m.body(m.geom_bodyid[g1]).name == body or m.body(m.geom_bodyid[g2]).name == body:
                        pairs[f"{n1}~{n2}"] = round(float(d.contact.dist[i]), 4)
                fast = (body, info["step"], info["grasped_now"], info["target"], "pos", sim.data.xpos[m.body(body).id].round(3).tolist(), "contacts(dist)", pairs, "seed", args.seed + ep)
        if term or trunc: break
    rows.append({"unstable": bool(info["unstable"]), "maxv": maxv, "success": info["is_success"], "fast": fast})
n = len(rows)
print(f"fix={args.fix:12s} n={n}: unstable {sum(r['unstable'] for r in rows)}  |qvel|>60 in {sum(r['maxv']>60 for r in rows)}  success {sum(r['success'] for r in rows)/n:.0%}  max|qvel| {max(r['maxv'] for r in rows):.0f}")
for r in rows:
    if r["fast"]: print("   diverged:", r["fast"], "max|qvel|", round(r["maxv"]))
env.close()
