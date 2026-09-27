"""Episode-level diagnosis of the BC->PPO regression (Incremental Update 2).

(1) From-home deterministic eval episodes for several checkpoints of one run: what does the policy actually do
    (grasp? open early? carry? where does the part end up?) - per-episode behaviour table, not just aggregates.
(2) Drift from the demonstrations: policy-mean vs demonstrated action on demo frames (action MSE, gripper-sign
    accuracy) for every checkpoint -> direct evidence of forgetting.
(3) Train/eval gap: the SAME checkpoint rolled out the way training sees it (curriculum starts, stochastic
    actions) vs the way evaluation sees it (home start, deterministic); reward and placement per start type,
    and which reward components produce the training-time reward.
"""
import os, sys, json, argparse, math
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from simulation.environments.sorting_env import make_env, STATE_KEY, IMAGE_KEY
from backend.policy_loader import LoadedPolicy
from simulation.config import resolve_path
from training.bc_pretrain import select_demo_dirs, DemoBuffer

ap = argparse.ArgumentParser()
ap.add_argument("checkpoints", nargs="+", help="label=path")
ap.add_argument("--episodes", type=int, default=10)
ap.add_argument("--seed", type=int, default=777)
ap.add_argument("--train-episodes", type=int, default=60, help="curriculum/stochastic episodes for part (3), last checkpoint only")
ap.add_argument("--out", default="logs/diagnosis_bc_regression.json")
args = ap.parse_args()
out = {"checkpoints": {}, "drift": {}, "train_eval_gap": None}

env = make_env(sb3=False); u = env.unwrapped; sim = u.sim
demos = select_demo_dirs(resolve_path("datasets/demos"), ["teleop_demo", "scripted_demo"], True)
buf = DemoBuffer.from_dirs(demos, include_image=True)
rng = np.random.default_rng(0); hold = rng.permutation(buf.n)[:600]


def behaviour(pol, seed):
    obs, info = env.reset(seed=seed)
    tgt0 = u.target
    rec = {"seed": seed, "target0": tgt0, "grasp_t": None, "first_open_after_grasp": None, "opens_while_holding": 0,
           "max_lift_h": 0.0, "min_dxy_bin_lifted": None, "placed": 0, "success": False, "reward": 0.0, "steps": 0,
           "min_tcp_obj_dist": 9.0, "end_state": None}
    prev_grasped = False
    while True:
        a = pol.select_action(obs, deterministic=True)
        tgt = u.target
        obs, r, term, trunc, info = env.step(a)
        if tgt is not None:
            p = sim.obj_pos(tgt); b = sim.bin_pos(u.bin_mapping[tgt])
            rec["min_tcp_obj_dist"] = min(rec["min_tcp_obj_dist"], float(np.linalg.norm(sim.ee_pos() - p)))
            if info["grasped_now"]:
                if rec["grasp_t"] is None: rec["grasp_t"] = u.t
                rec["max_lift_h"] = max(rec["max_lift_h"], sim.object_height_above_rest(tgt))
                if info["lifted_now"]:
                    d = float(np.linalg.norm(p[:2] - b[:2])); rec["min_dxy_bin_lifted"] = d if rec["min_dxy_bin_lifted"] is None else min(rec["min_dxy_bin_lifted"], d)
            if prev_grasped and not info["grasped_now"] and a[6] <= 0 and not info["placed_count"] > rec["placed"]:
                rec["opens_while_holding"] += 1
                if rec["first_open_after_grasp"] is None: rec["first_open_after_grasp"] = u.t
            prev_grasped = info["grasped_now"]
            rec["placed"] = info["placed_count"]
        if term or trunc:
            break
    rec.update(success=bool(info["is_success"]), reward=round(float(info["episode_reward"]), 1), steps=info["step"],
               dropped=bool(info["dropped"]), wrong_bin=bool(info["wrong_bin"]))
    # phase reached
    if rec["placed"] >= 1: ph = "PLACED" if not rec["success"] else "SUCCESS"
    elif rec["min_dxy_bin_lifted"] is not None and rec["min_dxy_bin_lifted"] < 0.15: ph = "carried-to-bin"
    elif rec["max_lift_h"] > 0.06: ph = "lifted"
    elif rec["grasp_t"] is not None: ph = "grasped"
    elif rec["min_tcp_obj_dist"] < 0.05: ph = "reached-part"
    else: ph = "no-approach"
    rec["phase"] = ph
    rec["max_lift_h"] = round(rec["max_lift_h"], 3); rec["min_tcp_obj_dist"] = round(rec["min_tcp_obj_dist"], 3)
    if rec["min_dxy_bin_lifted"] is not None: rec["min_dxy_bin_lifted"] = round(rec["min_dxy_bin_lifted"], 3)
    return rec


@torch.no_grad()
def drift(pol):
    norm = pol._normalize_state(buf.states[hold])
    obs = {"state": norm, "image": buf.images[hold]}
    t, _ = pol.model.policy.obs_to_tensor(obs)
    mean = pol.model.policy.get_distribution(t).distribution.mean.cpu().numpy()
    a = buf.actions[hold]
    return {"action_mse": round(float(np.mean((mean - a) ** 2)), 4), "gripper_sign_acc": round(float(np.mean(np.sign(mean[:, 6]) == np.sign(a[:, 6]))), 3),
            "xyz_mse": round(float(np.mean((mean[:, :3] - a[:, :3]) ** 2)), 4), "policy_std": round(float(torch.exp(pol.model.policy.log_std).mean().item()), 3)}


last_pol = None
for spec in args.checkpoints:
    label, path = spec.split("=", 1)
    pol = LoadedPolicy(resolve_path(path)); last_pol = (label, pol)
    rows = [behaviour(pol, args.seed + i) for i in range(args.episodes)]
    phases = {}
    for r in rows: phases[r["phase"]] = phases.get(r["phase"], 0) + 1
    out["checkpoints"][label] = {"timesteps": pol.metadata.get("timesteps"), "episodes": rows, "phases": phases,
                                 "grasp_rate": sum(r["grasp_t"] is not None for r in rows) / len(rows),
                                 "placement_rate": sum(r["placed"] > 0 for r in rows) / len(rows), "success_rate": sum(r["success"] for r in rows) / len(rows),
                                 "mean_reward": round(sum(r["reward"] for r in rows) / len(rows), 2),
                                 "early_open_rate": sum(r["first_open_after_grasp"] is not None for r in rows) / len(rows)}
    out["drift"][label] = drift(pol)
    print(f"\n=== {label} ({pol.metadata.get('run_name')} @ {pol.metadata.get('timesteps')}) drift vs demos: {out['drift'][label]}")
    print(f"    phases: {phases}  grasp {out['checkpoints'][label]['grasp_rate']:.0%} place {out['checkpoints'][label]['placement_rate']:.0%} success {out['checkpoints'][label]['success_rate']:.0%} R {out['checkpoints'][label]['mean_reward']}")
    print("    seed  phase           grasp_t open@  opens maxLift minTCPobj minDxyBin placed R")
    for r in rows:
        print(f"    {r['seed']}  {r['phase']:15s} {str(r['grasp_t']):7s} {str(r['first_open_after_grasp']):6s} {r['opens_while_holding']:5d} {r['max_lift_h']:7.3f} {r['min_tcp_obj_dist']:9.3f} {str(r['min_dxy_bin_lifted']):9s} {r['placed']:6d} {r['reward']:+.1f}")

# ---------------- (3) train/eval gap for the last checkpoint ----------------
label, pol = last_pol
tenv = make_env(sb3=False, training=True); tu = tenv.unwrapped
by_kind = {}
for ep in range(args.train_episodes):
    obs, info = tenv.reset(seed=50_000 + ep, options={"curriculum": True})
    kind = tu.last_reset_kind
    while True:
        a = pol.select_action(obs, deterministic=False)
        obs, r, term, trunc, info = tenv.step(a)
        if term or trunc: break
    b = by_kind.setdefault(kind, {"n": 0, "reward": 0.0, "placed": 0, "success": 0, "comps": {}})
    b["n"] += 1; b["reward"] += info["episode_reward"]; b["placed"] += int(info["placed_count"] > 0); b["success"] += int(info["is_success"])
    for k, v in info["episode_components"].items(): b["comps"][k] = b["comps"].get(k, 0.0) + v
gap = {}
print(f"\n=== train/eval gap for {label}: stochastic actions + curriculum starts (how training sees it) ===")
print("    kind      n  meanR  placed  success  top reward components (per episode)")
for kind, b in sorted(by_kind.items()):
    n = b["n"]; comps = {k: round(v / n, 1) for k, v in b["comps"].items() if abs(v / n) > 0.3}
    gap[kind] = {"n": n, "mean_reward": round(b["reward"] / n, 2), "placement_rate": round(b["placed"] / n, 2), "success_rate": round(b["success"] / n, 2), "components": comps}
    print(f"    {kind:9s} {n:2d} {b['reward']/n:+6.1f}  {b['placed']/n:5.0%}  {b['success']/n:6.0%}  {dict(sorted(comps.items(), key=lambda kv: -abs(kv[1]))[:6])}")
tot = sum(b["n"] for b in by_kind.values()); wR = sum(b["reward"] for b in by_kind.values()) / tot
home = by_kind.get("home", {"n": 0, "reward": 0, "placed": 0})
print(f"    ALL       {tot:2d} {wR:+6.1f}  (training-episode mean)  | home-start share {home['n']/tot:.0%}, home meanR {home['reward']/max(home['n'],1):+.1f}")
out["train_eval_gap"] = {"checkpoint": label, "by_kind": gap, "all_mean_reward": round(wR, 2), "deterministic_home": out["checkpoints"][label]["mean_reward"]}
tenv.close(); env.close()
json.dump(out, open(resolve_path(args.out), "w"), indent=1, default=str)
print("written", args.out)
