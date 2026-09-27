"""API health, endpoints and WebSocket connection using an in-process app instance."""
import json
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from backend import main as m
    with TestClient(m.app) as c:
        m.sim.ready.wait(60)
        yield c
    m.sim.stop()


def test_health(client):
    r = client.get("/health"); assert r.status_code == 200
    d = r.json(); assert d["status"] == "ok" and d["robot"] == "UR10e" and d["device"] in ("CPU",) or d["device"].startswith("CUDA")


def test_state_reset_randomize(client):
    s = client.get("/simulation/state").json()
    assert "joint_positions" in s and len(s["joint_positions"]) == 6
    r = client.post("/simulation/reset", json={"seed": 5}); assert r.status_code == 200
    a = client.post("/simulation/randomize", json={"seed": 1}).json()["data"]["episode_config"]["object_poses"]
    b = client.post("/simulation/randomize", json={"seed": 2}).json()["data"]["episode_config"]["object_poses"]
    assert a != b


def test_step_and_mode(client):
    r = client.post("/simulation/step", json={"steps": 2}); assert r.status_code == 200 and r.json()["data"]["step"] >= 2
    assert client.post("/simulation/mode", json={"mode": "random"}).status_code == 200
    assert client.post("/simulation/mode", json={"mode": "idle"}).status_code == 200
    assert client.post("/simulation/mode", json={"mode": "bogus"}).status_code == 400


def test_training_status_and_checkpoints(client):
    assert "state" in client.get("/training/status").json()
    assert "metrics" in client.get("/training/metrics").json()
    d = client.get("/checkpoints").json(); assert "checkpoints" in d


def test_websocket_simulation(client):
    with client.websocket_connect("/ws/simulation") as ws:
        for _ in range(10):
            msg = json.loads(ws.receive_text())
            if msg["type"] == "frame":
                break
        assert msg["type"] == "frame" and len(msg["frame"]) > 1000 and "joint_positions" in msg["state"]


def test_websocket_training(client):
    with client.websocket_connect("/ws/training") as ws:
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "history" and "status" in msg
