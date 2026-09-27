"""Teleop API, demonstration recording/discard, scripted controller and BC bootstrap (Section 43)."""
import json, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from backend import main as m
    m.sim.demos_dir = tmp_path_factory.mktemp("demos")
    m.trainer.is_running = lambda: False        # a real trainer may be running on this machine; the test owns the sim
    with TestClient(m.app) as c:
        m.sim.ready.wait(60)
        yield c
    m.sim.stop()


def test_camera_readout_and_frustum(client):
    cam = client.get("/simulation/camera").json()
    assert cam["name"] == "overhead" and cam["description"] == "looking straight down" and abs(cam["tilt_from_vertical_deg"]) < 0.5
    assert client.post("/simulation/frustum", json={"enabled": True}).status_code == 200
    assert client.get("/simulation/state").json()["show_frustum"] is True
    client.post("/simulation/frustum", json={"enabled": False})


def test_teleop_moves_arm_through_action_space_and_records(client):
    from backend import main as m
    r = client.post("/teleop/start", json={"reset": True}); assert r.status_code == 200
    ee0 = client.get("/simulation/state").json()["ee_position"]
    assert client.post("/dataset/record/start", json={"episode_type": "teleop_demo", "reset": True}).status_code == 200
    with client.websocket_connect("/ws/teleop") as ws:
        for _ in range(12):
            ws.send_text(json.dumps({"type": "action", "axes": [1.0, 0, 0, 0, 0, 0], "gripper": -1}))
            m.sim.call(m.sim.cmd_step, steps=1)     # deterministic stepping instead of waiting on wall-clock
    st = client.get("/simulation/state").json()
    assert st["mode"] == "teleop" and st["ee_position"][0] > ee0[0] + 0.03
    assert st["recording"]["recording"] and st["recording"]["episode_type"] == "teleop_demo"
    client.post("/dataset/record/stop")
    demos = client.get("/demos").json()
    tele = [d for d in demos["demos"] if d["episode_type"] == "teleop_demo"]
    assert tele and tele[-1]["frames"] >= 10 and tele[-1]["aborted"] is True
    name = tele[-1]["name"]
    assert client.delete(f"/demos/{name}").status_code == 200
    assert not any(d["name"] == name for d in client.get("/demos").json()["demos"])
    assert client.delete("/demos/../etc").status_code in (400, 404)
    client.post("/teleop/stop")
    assert client.get("/simulation/state").json()["mode"] == "idle"


def test_teleop_refused_while_inference_runs(client):
    from backend import main as m
    m.sim.mode = "inference"
    try:
        assert client.post("/teleop/start", json={}).status_code == 409
    finally:
        m.sim.mode = "idle"


def test_scripted_controller_sorts_and_records(tmp_path):
    from simulation.environments.sorting_env import make_env
    from lerobot_adapter.demo_recorder import DemoRecorder, list_demos, discard_demo
    from training.demos.scripted_controller import ScriptedSortingController
    env = make_env(sb3=False); u = env.unwrapped
    rec = DemoRecorder(root=tmp_path, episode_type="scripted_demo", state_layout=u.state_layout, fps=10,
                       image_shape=(u.img_h, u.img_w, 3), robot_type=u.sim.robot_name)
    ctrl = ScriptedSortingController(env)
    successes = 0
    for ep in range(2):
        obs, info = env.reset(seed=4000 + ep); ctrl.reset()
        while True:
            a = ctrl.act()
            assert a.shape == (7,) and np.all(np.abs(a) <= 1.0)
            rec.add_step(obs, a, None, False, info)
            obs, r, term, trunc, info = env.step(a)
            rec.set_last_reward(r, term or trunc, bool(info["is_success"]))
            if term or trunc:
                break
        successes += int(info["is_success"])
        side = rec.end_episode({"success": info["is_success"], "placed_count": info["placed_count"], "reward": info["episode_reward"], "length": info["step"]})
        assert side["episode_type"] == "scripted_demo" and side["frames"] == info["step"]
    rec.close(); env.close()
    assert successes >= 1, "scripted controller should sort both parts in at least one of two episodes"
    demos = list_demos(tmp_path)
    assert len(demos) == 2 and all((Path(d["path"]) / "meta" / "info.json").exists() for d in demos)
    assert discard_demo(tmp_path, demos[0]["name"]) and len(list_demos(tmp_path)) == 1


@pytest.mark.slow
def test_bc_bootstrap_smoke(tmp_path):
    """Scripted demos -> BC pretraining -> PPO with aux BC steps, end to end through the real trainer."""
    env = dict(os.environ, MUJOCO_GL="egl")
    demos = tmp_path / "demos"
    res = subprocess.run([sys.executable, "scripts/record_scripted_demos.py", "--episodes", "3", "--out", str(demos), "--keep-failures"],
                         cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-2000:]
    overrides = {"total_timesteps": 1500, "n_envs": 2, "n_steps": 64, "batch_size": 64, "n_epochs": 1, "checkpoint_every_steps": 800,
                 "eval_every_steps": 0, "run_name": "pytest_bc", "log_dir": str(tmp_path / "logs"), "checkpoint_dir": str(tmp_path / "ckpt"),
                 "metrics_file": str(tmp_path / "logs/metrics.jsonl"), "status_file": str(tmp_path / "logs/status.json"),
                 "control_file": str(tmp_path / "logs/control.json"), "dataset": {"record_eval_episodes": False},
                 "bootstrap": {"init_from_demos": True, "demos_dir": str(demos), "successful_only": False, "bc_epochs": 3,
                               "bc_aux_coef": 1.0, "bc_aux_timesteps": 1500, "bc_aux_steps_per_iter": 2}}
    of = tmp_path / "ov.json"; of.write_text(json.dumps(overrides))
    res = subprocess.run([sys.executable, "-m", "training.train", "--overrides", str(of)], cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    log = (tmp_path / "logs/training.log").read_text()
    assert "BC pretraining:" in log and "BC done" in log and "checkpoint saved (bc_init)" in log
    status = json.loads((tmp_path / "logs/status.json").read_text())
    assert status["state"] == "completed" and status["bootstrap"]["enabled"] and status["bootstrap"]["demo_frames"] > 0
    assert status["bootstrap"]["after"]["action_mse"] <= status["bootstrap"]["before"]["action_mse"]
    metrics = [json.loads(l) for l in (tmp_path / "logs/metrics.jsonl").read_text().splitlines()]
    assert any("bc/aux_action_mse" in m for m in metrics)
    from training.checkpointing import list_checkpoints
    names = [c["name"] for c in list_checkpoints(tmp_path / "ckpt")]
    assert "checkpoint_000000000" in names
