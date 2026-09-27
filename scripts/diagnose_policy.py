"""Roll out a checkpoint deterministically and print per-step behaviour to understand what the
policy actually does (touch vs. grasp vs. lift)."""
import os, sys, numpy as np
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulation.environments.sorting_env import make_env, STATE_KEY
from backend.policy_loader import LoadedPolicy, resolve_checkpoint

ck = resolve_checkpoint(sys.argv[1] if len(sys.argv) > 1 else "latest")
pol = LoadedPolicy(ck); print("checkpoint", ck.name, "timesteps", pol.metadata.get("timesteps"))
env = make_env(sb3=False)
for ep in range(3):
    obs, info = env.reset(seed=100 + ep)
    sim = env.unwrapped.sim
    tgt = env.unwrapped.target
    print(f"\n--- episode {ep} target={tgt} obj={sim.obj_pos(tgt).round(3)} yaw={sim.obj_yaw(tgt):.2f}")
    ever = {"touch": 0, "grasp": 0, "lift": 0}; comps = {}
    for t in range(200):
        a = pol.select_action(obs)
        obs, r, term, trunc, info = env.step(a)
        cs = sim.contacts(tgt) if tgt else None
        touch = bool(cs and (cs.left_pad_target or cs.right_pad_target))
        ever["touch"] += touch; ever["grasp"] += info["grasped_now"]; ever["lift"] += info["lifted_now"]
        for k, v in info["reward_components"].items(): comps[k] = comps.get(k, 0) + v
        if t % 20 == 0 or info["grasped_now"]:
            d = np.linalg.norm(sim.ee_pos() - sim.obj_pos(tgt)) if tgt else -1
            print(f"t={t:3d} dist={d:.3f} ee_z={sim.ee_pos()[2]:.3f} obj_h={sim.object_height_above_rest(tgt):.3f} grip_cmd={a[6]:+.2f} open={sim.gripper_opening():.2f} pads=({int(cs.left_pad_target)},{int(cs.right_pad_target)}) grasped={int(info['grasped_now'])} dz={a[2]:+.2f} r={r:+.2f}")
        if term or trunc: break
        tgt = env.unwrapped.target
    print("steps touching/grasped/lifted:", ever, "success", info["is_success"], "R", round(info["episode_reward"], 1))
    print("reward components:", {k: round(v, 1) for k, v in comps.items() if abs(v) > 0.05})
env.close()
