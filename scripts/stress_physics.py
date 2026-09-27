"""Random-action physics stress test: counts MuJoCo instability warnings."""
import os, sys, time, numpy as np, mujoco
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulation.environments.workcell_sim import WorkcellSim
from simulation.config import load_yaml

def run(sim_cfg=None, seeds=24, steps=150, verbose=True):
    sim = WorkcellSim(sim_cfg=sim_cfg)
    bad = 0; t0 = time.time(); n = 0
    for seed in range(seeds):
        sim.reset(seed=seed)
        sim.track_target = "bolt"
        rng = np.random.default_rng(seed + 1000)
        # biased random policy: descend a lot, close gripper often -> maximizes pinch situations
        for t in range(steps):
            a = rng.uniform(-1, 1, 7)
            a[2] -= 0.4
            if t % 30 < 15: a[6] = 1.0
            sim.apply_action(a); n += 1
            if not sim.is_stable() or sim.d.warning.number.sum() > 0:
                if verbose: print(f"seed {seed} step {t}: unstable; warnings={sim.d.warning.number.tolist()} ee={sim.ee_pos().round(3)}")
                bad += 1; sim.d.warning.number[:] = 0
                break
    dt = time.time() - t0
    print(f"unstable episodes: {bad}/{seeds}   ({n/dt:.0f} control steps/s incl. IK)")
    sim.close(); return bad

if __name__ == "__main__":
    run()
