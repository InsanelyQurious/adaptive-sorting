"""Diagnose the grasp->placement gap (Section 42).

Rolls the checkpoint out from home (exactly like evaluation), and for every episode in which a
grasp is established, logs what happens between the grasp and the end of the episode:
object height / distance to the correct bin / gripper command / per-step reward components, and
saves frame strips (workcell + overhead camera) for inspection under logs/diagnosis/.
Also probes the curriculum reset helpers and the placement detector directly.
"""
import os, sys, json, argparse
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, cv2
from simulation.environments.sorting_env import make_env, STATE_KEY
from backend.policy_loader import LoadedPolicy, resolve_checkpoint

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default="latest")
ap.add_argument("--episodes", type=int, default=20)
ap.add_argument("--stochastic", action="store_true")
ap.add_argument("--out", default="logs/diagnosis")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

ck = resolve_checkpoint(args.checkpoint)
pol = LoadedPolicy(ck)
print("checkpoint", ck.name, "timesteps", pol.metadata.get("timesteps"))
env = make_env(sb3=False)
u = env.unwrapped
sim = u.sim

def strip(frames, path):
    if frames:
        cv2.imwrite(path, cv2.cvtColor(np.concatenate(frames, axis=1), cv2.COLOR_RGB2BGR))

summary = []
for ep in range(args.episodes):
    obs, info = env.reset(seed=1000 + ep)
    tgt0 = u.target
    log = []; frames = []; ov = []
    grasp_t = None; lift_t = None; place_t = None; release_events = []
    comps_after = {}
    for t in range(u.max_steps + 1):
        a = pol.select_action(obs, deterministic=not args.stochastic)
        tgt = u.target
        obs, r, term, trunc, info = env.step(a)
        rc = info["reward_components"]
        if tgt is not None:
            p = sim.obj_pos(tgt); b = sim.bin_pos(u.bin_mapping[tgt])
            row = dict(t=t, tgt=tgt, grasped=int(info["grasped_now"]), lifted=int(info["lifted_now"]),
                       obj_h=round(sim.object_height_above_rest(tgt), 3), obj_z=round(float(p[2]), 3),
                       dxy_bin=round(float(np.linalg.norm(p[:2] - b[:2])), 3),
                       tcp=[round(float(x), 3) for x in sim.ee_pos()], grip_raw=round(float(a[6]), 2),
                       grip_open=round(sim.gripper_opening(), 2), dz=round(float(a[2]), 2),
                       dx=round(float(a[0]), 2), dy=round(float(a[1]), 2), r=round(float(r), 3),
                       in_bin=sim.which_bin(tgt), placed=info["placed_count"],
                       comps={k: round(v, 3) for k, v in rc.items() if abs(v) > 1e-6})
            log.append(row)
            if info["grasped_now"] and grasp_t is None:
                grasp_t = t
            if info["lifted_now"] and lift_t is None:
                lift_t = t
            if grasp_t is not None:
                for k, v in rc.items():
                    comps_after[k] = comps_after.get(k, 0.0) + v
                if rc.get("release_penalty", 0) != 0:
                    release_events.append(t)
        if info["placed_count"] > 0 and place_t is None:
            place_t = t
        if grasp_t is not None and (t - grasp_t) % 5 == 0 and len(frames) < 24:
            frames.append(sim.render_camera("workcell", 320, 240)); ov.append(cv2.resize(obs["observation.images.overhead"], (240, 240), interpolation=cv2.INTER_NEAREST))
        if term or trunc:
            break
    res = dict(episode=ep, seed=1000 + ep, target0=tgt0, grasp_t=grasp_t, lift_t=lift_t, place_t=place_t,
               placed=info["placed_count"], success=info["is_success"], steps=info["step"], R=round(info["episode_reward"], 2),
               dropped=info["dropped"], wrong_bin=info["wrong_bin"], release_penalties=release_events,
               comps_after_grasp={k: round(v, 2) for k, v in comps_after.items() if abs(v) > 0.05},
               final_obj={o: [round(float(x), 3) for x in sim.obj_pos(o)] for o in u.objects},
               final_in_bin={o: sim.which_bin(o) for o in u.objects})
    summary.append(res)
    tag = "GRASP" if grasp_t is not None else "nograsp"
    print(f"ep {ep:2d} {tag:7s} grasp_t={grasp_t} lift_t={lift_t} place_t={place_t} placed={info['placed_count']} R={res['R']:7.2f} "
          f"rel_pen={len(release_events)} after-grasp comps={res['comps_after_grasp']}")
    if grasp_t is not None:
        with open(f"{args.out}/ep{ep:02d}_trace.json", "w") as f:
            json.dump(log[max(0, grasp_t - 3):], f, indent=0)
        strip(frames, f"{args.out}/ep{ep:02d}_workcell.png"); strip(ov, f"{args.out}/ep{ep:02d}_overhead.png")
        # condensed after-grasp trace
        print("    t   tgt      grasped lifted obj_h  dxy_bin grip_raw dz    r      in_bin")
        for row in log[grasp_t:]:
            if (row["t"] - grasp_t) % 10 == 0 or row["comps"].get("release_penalty") or row["in_bin"]:
                print(f"    {row['t']:3d} {row['tgt']:8s} {row['grasped']}       {row['lifted']}      {row['obj_h']:.3f}  {row['dxy_bin']:.3f}   {row['grip_raw']:+.2f}   {row['dz']:+.2f} {row['r']:+.3f}  {row['in_bin']}")
json.dump(summary, open(f"{args.out}/summary.json", "w"), indent=1)
n = len(summary)
print(f"\n== {n} episodes: grasp {sum(s['grasp_t'] is not None for s in summary)}, lift {sum(s['lift_t'] is not None for s in summary)}, "
      f"placed>0 {sum(s['placed'] > 0 for s in summary)}, success {sum(s['success'] for s in summary)}")
env.close()
