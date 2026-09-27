"""Find the mechanism of clutter-induced instability and measure candidate fixes.
Runs the trained policy with clutter deliberately placed in the transport paths (adversarial layouts), logs the
contact pairs at the first unstable step, and repeats with model tweaks applied in-memory."""
import os, sys, argparse
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, mujoco
from simulation.environments.sorting_env import make_env
from backend.policy_loader import LoadedPolicy, resolve_checkpoint

ap = argparse.ArgumentParser(); ap.add_argument("--episodes", type=int, default=30); ap.add_argument("--fix", default="none", choices=["none", "soft", "damp", "soft+damp", "soft+damp+margin"]); args = ap.parse_args()
env = make_env(sb3=False); u = env.unwrapped; sim = u.sim; m = sim.model
pol = LoadedPolicy(resolve_checkpoint("best"))
CLUTTER = ["bracket_extra", "reject_cap", "bolt_extra"]
if "soft" in args.fix:
    for name in CLUTTER:
        bid = m.body(name).id
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == bid and (m.geom_contype[g] or m.geom_conaffinity[g]):
                m.geom_priority[g] = 1                          # this geom's solref/solimp win in any pair
                m.geom_solref[g] = [0.02, 1.0]
                m.geom_solimp[g] = [0.9, 0.95, 0.01, 0.5, 2]
                m.geom_solref[g] = [0.03, 1.0]
if "damp" in args.fix:
    for name in ["bracket", "bolt", "bracket_flat", "bracket_u", "bolt_socket", "bolt_m12"] + CLUTTER:
        j = m.joint(f"{name}_free").id; d0 = m.jnt_dofadr[j]
        m.dof_damping[d0:d0 + 6] = [0.5, 0.5, 0.5, 0.005, 0.005, 0.005]   # light viscous damping on free bodies
if "margin" in args.fix:
    for name in CLUTTER:
        bid = m.body(name).id
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == bid: m.geom_margin[g] = 0.002
rng = np.random.default_rng(1)
rows = []
for ep in range(args.episodes):
    obs, info = env.reset(seed=12000 + ep)
    # adversarial: clutter between the spawn zone and the bins, and right next to a task object
    b = sim.obj_pos("bracket"); o = sim.obj_pos("bolt")
    layout = [("bracket_extra", (0.2, 0.30), rng.uniform(-3, 3)), ("reject_cap", (0.2, -0.30), rng.uniform(-3, 3)), ("bolt_extra", (float(np.clip(b[0] + 0.09, 0.0, 0.4)), float(b[1])), rng.uniform(-3, 3))]
    for part, xy, yaw in layout: sim.place_body(part, xy, yaw, settle_steps=0)
    mujoco.mj_step(m, sim.data, nstep=30)
    maxv = 0.0; first_pair = None; unstable = False
    while True:
        a = pol.select_action(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        v = float(np.abs(sim.data.qvel).max()); maxv = max(maxv, v)
        if v > 60 and first_pair is None:
            d = sim.data; pairs = {}
            for i in range(d.ncon):
                g1, g2 = d.contact.geom1[i], d.contact.geom2[i]
                pairs[f"{m.geom(g1).name}~{m.geom(g2).name}"] = round(float(np.linalg.norm(d.efc_force[d.contact.efc_address[i]] if d.contact.efc_address[i] >= 0 else 0)), 1)
            dof = int(np.argmax(np.abs(sim.data.qvel))); 
            body = m.body(m.dof_bodyid[dof]).name
            first_pair = (info["step"], body, dict(sorted(pairs.items(), key=lambda kv: -kv[1])[:5]))
        if info.get("unstable"): unstable = True
        if term or trunc: break
    rows.append({"unstable": unstable, "maxv": maxv, "success": info["is_success"], "placed": info["placed_count"], "coll": info["collisions"], "first": first_pair})
    for part, _, _ in layout: sim.park_body(part)
n = len(rows)
print(f"fix={args.fix:16s} episodes {n}: unstable {sum(r['unstable'] for r in rows)}  max|qvel|>60 in {sum(r['maxv']>60 for r in rows)}  success {sum(r['success'] for r in rows)/n:.0%}  placed/ep {sum(r['placed'] for r in rows)/n:.2f}  coll/ep {sum(r['coll'] for r in rows)/n:.1f}")
for r in rows:
    if r["first"]: print("   step", r["first"][0], "fast body:", r["first"][1], "contacts:", r["first"][2])
env.close()
