"""Record `scripted_demo` episodes with the hand-coded ground-truth controller, straight through the
Gymnasium environment (no backend needed). Validates recording -> storage before any human demo exists.

  python scripts/record_scripted_demos.py --episodes 20 --noise 0.05
"""
import os, sys, argparse, time, json
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from simulation.environments.sorting_env import make_env
from simulation.config import resolve_path
from lerobot_adapter.demo_recorder import DemoRecorder, list_demos, demo_summary
from training.demos.scripted_controller import ScriptedSortingController

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=10)
ap.add_argument("--noise", type=float, default=0.0, help="gaussian noise std added to the motion axes (diversity)")
ap.add_argument("--seed", type=int, default=2000)
ap.add_argument("--out", default="datasets/demos")
ap.add_argument("--keep-failures", action="store_true", help="keep episodes that did not sort both parts")
args = ap.parse_args()

env = make_env(sb3=False); u = env.unwrapped
rec = DemoRecorder(root=resolve_path(args.out), episode_type="scripted_demo", state_layout=u.state_layout,
                   fps=int(round(1.0 / u.sim.control_dt)), image_shape=(u.img_h, u.img_w, 3) if u.include_image else None,
                   robot_type=u.sim.robot_name, extra_meta={"controller": "scripted_ground_truth", "noise_std": args.noise, "source": "scripts/record_scripted_demos.py"})
ctrl = ScriptedSortingController(env, noise_std=args.noise, rng=np.random.default_rng(args.seed))
kept = 0; t0 = time.time()
for ep in range(args.episodes):
    obs, info = env.reset(seed=args.seed + ep); ctrl.reset()
    while True:
        a = ctrl.act()
        rec.add_step(obs, a, None, False, info)
        obs, r, term, trunc, info = env.step(a)
        rec.set_last_reward(r, term or trunc, bool(info.get("is_success")))
        if term or trunc:
            break
    result = {"success": info["is_success"], "placed_count": info["placed_count"], "reward": round(info["episode_reward"], 3),
              "length": info["step"], "grasp_success": info["grasp_achieved"], "dropped": info["dropped"], "wrong_bin": info["wrong_bin"],
              "truncated": bool(trunc), "seed": info["seed"], "object_configuration": info.get("episode_config")}
    side = rec.end_episode(result)
    keep = info["is_success"] or args.keep_failures
    if side and not keep:
        import shutil; shutil.rmtree(resolve_path(args.out) / side["name"], ignore_errors=True)
    kept += int(bool(side) and keep)
    print(f"ep {ep:2d} seed={info['seed']} success={info['is_success']} placed={info['placed_count']} len={info['step']} R={info['episode_reward']:+.1f} retries={ctrl.retries} {'KEPT' if keep else 'discarded'}")
rec.close(); env.close()
demos = list_demos(resolve_path(args.out))
print(f"\nkept {kept}/{args.episodes} episodes in {time.time()-t0:.0f}s; catalogue now: {json.dumps(demo_summary(demos))}")
