"""Offline, like-for-like evaluation of several checkpoints: identical seeds, clean env (no curriculum),
deterministic actions, full randomization. Prints grasp / placement / success rates with 95 % Wilson CIs.

  python scripts/compare_checkpoints.py --episodes 30 --seed 777 \
      baseline=checkpoints_diag_baseline/checkpoint_000301056 bootstrap=checkpoints/checkpoint_000301056
"""
import os, sys, argparse, json, math, time
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from simulation.environments.sorting_env import make_env
from backend.policy_loader import LoadedPolicy
from simulation.config import resolve_path


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def evaluate(pol, env, seeds, deterministic=True):
    rows = []
    u = env.unwrapped
    for sd in seeds:
        obs, info = env.reset(seed=int(sd))
        while True:
            a = pol.select_action(obs, deterministic=deterministic)
            obs, r, term, trunc, info = env.step(a)
            if term or trunc:
                break
        rows.append({"seed": int(sd), "success": bool(info["is_success"]), "placed": int(info["placed_count"]), "grasp": bool(info["grasp_achieved"]),
                     "lift": bool(info["lift_achieved"]), "reward": round(float(info["episode_reward"]), 2), "length": int(info["step"]),
                     "dropped": bool(info["dropped"]), "wrong_bin": bool(info["wrong_bin"]), "collisions": int(info["collisions"])})
    return rows


def summarize(rows):
    n = len(rows)
    def rate(key):
        k = sum(1 for r in rows if r[key]); lo, hi = wilson(k, n); return {"count": k, "rate": k / n, "ci95": [round(lo, 3), round(hi, 3)]}
    return {"episodes": n, "grasp": rate("grasp"), "lift": rate("lift"), "placement": {"count": sum(r["placed"] > 0 for r in rows), "rate": sum(r["placed"] > 0 for r in rows) / n,
            "ci95": list(map(lambda v: round(v, 3), wilson(sum(r["placed"] > 0 for r in rows), n)))},
            "success": rate("success"), "mean_placed": sum(r["placed"] for r in rows) / n,
            "mean_reward": round(sum(r["reward"] for r in rows) / n, 2), "mean_length": round(sum(r["length"] for r in rows) / n, 1),
            "drops": sum(r["dropped"] for r in rows), "wrong_bin": sum(r["wrong_bin"] for r in rows)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="+", help="label=path/to/checkpoint_dir")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--out", default="logs/checkpoint_comparison.json")
    args = ap.parse_args()
    seeds = [args.seed + i for i in range(args.episodes)]
    env = make_env(sb3=False)
    results = {}
    for t in args.targets:
        label, path = t.split("=", 1)
        pol = LoadedPolicy(resolve_path(path))
        t0 = time.time()
        rows = evaluate(pol, env, seeds, deterministic=not args.stochastic)
        summ = summarize(rows)
        summ.update({"checkpoint": str(resolve_path(path)), "timesteps": pol.metadata.get("timesteps"), "run_name": pol.metadata.get("run_name"),
                     "bootstrap": (pol.metadata.get("bootstrap") or {}).get("enabled", False), "seconds": round(time.time() - t0, 1)})
        results[label] = {"summary": summ, "episodes": rows}
        print(f"{label:>10s} [{pol.metadata.get('run_name')} @ {pol.metadata.get('timesteps')}]: grasp {summ['grasp']['rate']:.0%} (CI {summ['grasp']['ci95']}) "
              f"lift {summ['lift']['rate']:.0%} placement {summ['placement']['rate']:.0%} (CI {summ['placement']['ci95']}) success {summ['success']['rate']:.0%} "
              f"mean placed {summ['mean_placed']:.2f} reward {summ['mean_reward']:+.1f} len {summ['mean_length']}")
    env.close()
    out = resolve_path(args.out); out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"seed": args.seed, "episodes": args.episodes, "deterministic": not args.stochastic, "results": results}, indent=1))
    print("written", out)
