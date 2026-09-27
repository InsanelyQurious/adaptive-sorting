"""Scripted validation of the MuJoCo model (Section 7): gravity, joint tracking,
gripper open/close, and a physical pick-and-place of each object using contacts only.
This is a model test, NOT the policy."""
import sys, os, time, numpy as np, mujoco
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulation.environments.workcell_sim import WorkcellSim
from simulation.environments.workcell_sim import wrap_angle
import PIL.Image as Image

def pick_and_place(sim: WorkcellSim, obj: str, save_prefix=None) -> dict:
    bin_name = sim.bin_mapping[obj]
    sim.track_target = obj
    p = sim.obj_pos(obj); yaw = sim.obj_yaw(obj)
    # gripper x-axis should align with object long axis so pads close across the short side
    g_yaw = wrap_angle(yaw)
    hover = p + np.array([0, 0, 0.20])
    sim.set_gripper(0.0)
    ok1 = sim.move_to(hover, g_yaw, max_steps=60)
    grasp_z = sim.ws_lo[2]
    ok2 = sim.move_to(np.array([p[0], p[1], grasp_z]), g_yaw, max_steps=40, tol=0.004)
    if save_prefix: Image.fromarray(sim.render_camera("workcell", 640, 480)).save(f"{save_prefix}_{obj}_pregrasp.png")
    # close
    for _ in range(8):
        a = np.zeros(7); a[6] = 1.0; sim.apply_action(a)
    cs = sim.contacts(obj); grasped = sim.is_grasped(obj, cs)
    # lift
    for _ in range(12):
        a = np.zeros(7); a[2] = 0.6; a[6] = 1.0; sim.apply_action(a)
    lifted = sim.is_lifted(obj); still_grasped = sim.is_grasped(obj)
    if save_prefix: Image.fromarray(sim.render_camera("workcell", 640, 480)).save(f"{save_prefix}_{obj}_lift.png")
    # transport
    b = sim.bin_pos(bin_name)
    for _ in range(60):
        delta = np.array([b[0], b[1], 1.02]) - sim.target_pos
        a = np.zeros(7); a[:3] = np.clip(delta / sim.max_dpos, -1, 1); a[6] = 1.0; sim.apply_action(a)
        if np.linalg.norm(delta[:2]) < 0.01: break
    over_bin = np.linalg.norm(sim.obj_pos(obj)[:2] - b[:2]) < 0.1
    # release
    for _ in range(15):
        a = np.zeros(7); a[6] = -1.0; sim.apply_action(a)
    in_bin = sim.in_bin(obj, bin_name)
    if save_prefix: Image.fromarray(sim.render_camera("workcell", 640, 480)).save(f"{save_prefix}_{obj}_placed.png")
    return dict(obj=obj, reach_hover=ok1, reach_grasp=ok2, pads=(cs.left_pad_target, cs.right_pad_target),
                grasped=grasped, lifted=lifted, held_after_lift=still_grasped, over_bin=over_bin,
                in_bin=in_bin, final_pos=sim.obj_pos(obj).round(3).tolist())

if __name__ == "__main__":
    sim = WorkcellSim()
    print("model:", sim.robot_name, "nq", sim.m.nq, "substeps", sim.n_substeps)
    # 1) gravity / stability
    sim.reset(seed=1)
    for _ in range(100): sim.step_physics(5)
    print("stable after 1s:", sim.is_stable(), "warnings:", int(sim.d.warning.number.sum()),
          "objects:", {o: sim.obj_pos(o).round(3).tolist() for o in sim.objects})
    # 2) joint tracking
    q0 = sim.arm_qpos(); a = np.zeros(7); a[0] = 1.0
    for _ in range(5): sim.apply_action(a)
    print("ee moved +x by", round(sim.ee_pos()[0] - sim.target_pos[0] + 5 * sim.max_dpos, 3), "target err",
          round(np.linalg.norm(sim.target_pos - sim.ee_pos()), 4))
    # 3) gripper
    sim.set_gripper(1.0); sim.step_physics(200); closed = sim.gripper_opening()
    sim.set_gripper(0.0); sim.step_physics(200); opened = sim.gripper_opening()
    print("gripper opening closed/open:", round(closed, 3), round(opened, 3))
    # 4) pick and place across randomized layouts
    results = []
    t0 = time.time(); n_steps = 0
    for seed in range(12):
        sim.reset(seed=seed)
        for obj in sim.objects:
            r = pick_and_place(sim, obj, save_prefix=f"logs/validate_s{seed}" if seed == 0 else None)
            results.append(r); print(r)
    succ = sum(r["in_bin"] for r in results)
    print(f"scripted pick&place: {succ}/{len(results)} in correct bin")
    sim.close()
