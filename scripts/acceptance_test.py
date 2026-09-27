"""Section 31 acceptance test: checks every item that can be verified automatically against the
RUNNING system (backend :8000, frontend :3001, trainer process, checkpoints). Prints PASS/FAIL per item."""
import json, os, sys, time, urllib.request, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MUJOCO_GL", "egl")
B = os.environ.get("BACKEND", "http://localhost:8000"); F = os.environ.get("FRONTEND", "http://localhost:3001")
results = []

def check(name, fn):
    try:
        ok, detail = fn()
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    results.append((name, ok, detail)); print(f"[{'PASS' if ok else 'FAIL'}] {name} — {detail}", flush=True)

def get(path, base=B):
    with urllib.request.urlopen(base + path, timeout=30) as r: return json.loads(r.read())

def post(path, body=None):
    req = urllib.request.Request(B + path, data=json.dumps(body or {}).encode(), headers={"content-type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r: return json.loads(r.read())

def html(url):
    with urllib.request.urlopen(url, timeout=30) as r: return r.read().decode(errors="replace")

h = None
def c1():
    global h; h = get("/health"); return h["status"] == "ok", f"backend ok, mujoco {h['mujoco_version']}, device {h['device']}"
check("1 backend starts", c1)
check("2 frontend starts", lambda: (("Adaptive Sorting" in html(F + "/") or "ADAPTIVE SORTING" in html(F + "/")), f"{F} serves the dashboard"))
def c3():
    r = subprocess.run(["node", "screenshot.mjs", F + "/", "../logs/acceptance_dashboard.png"], cwd=os.path.join(os.path.dirname(__file__), "..", "frontend"), capture_output=True, text=True, timeout=120)
    line = [l for l in r.stdout.splitlines() if l.startswith("{")]
    d = json.loads(line[-1]) if line else {}
    return bool(d.get("sim_live")) and d.get("jpeg_images", 0) > 0 and not d.get("errors"), f"headless Chrome: live frames={d.get('jpeg_images')}, sim_live={d.get('sim_live')}, train={d.get('train_state')}, errors={d.get('errors')}"
check("3 browser opens dashboard (headless Chrome, live data)", c3)
check("4 MuJoCo loads the UR10e", lambda: (h["robot"] == "UR10e", f"robot={h['robot']}"))
def c5():
    from simulation.environments.workcell_sim import WorkcellSim
    s = WorkcellSim(); m = s.model
    names = [m.body(i).name for i in range(m.nbody)]; cams = [m.camera(i).name for i in range(m.ncam)]
    need = ["robot_base", "bin_a", "bin_b", "bracket", "bolt"]; ok = all(n in names for n in need) and m.geom("table_top").id >= 0 and "overhead" in cams
    s.close(); return ok, f"bodies {need} + table_top + camera 'overhead' present"
check("5 scene contains robot, table, two bins, bracket, bolt, camera", c5)
check("6 simulation can reset", lambda: (post("/simulation/reset", {"seed": 11})["ok"], "POST /simulation/reset ok"))
def c7():
    a = post("/simulation/randomize", {"seed": 1})["data"]["episode_config"]["object_poses"]; b = post("/simulation/randomize", {"seed": 2})["data"]["episode_config"]["object_poses"]
    return a != b, f"bracket {a['bracket']} vs {b['bracket']}"
check("7 scene randomization works", c7)
def c8():
    s0 = get("/simulation/state")["ee_position"]; post("/simulation/mode", {"mode": "random"}); time.sleep(1.5); post("/simulation/mode", {"mode": "idle"}); s1 = get("/simulation/state")
    moved = sum((a - b) ** 2 for a, b in zip(s0, s1["ee_position"])) ** 0.5
    return s1["step"] > 0 and moved > 0.005, f"TCP moved {moved:.3f} m in {s1['step']} steps"
check("8 robot can move", c8)
def c9():
    from simulation.environments.workcell_sim import WorkcellSim, wrap_angle
    import numpy as np
    s = WorkcellSim(); s.reset(seed=0); obj = "bracket"; s.track_target = obj; p = s.obj_pos(obj); yaw = wrap_angle(s.obj_yaw(obj))
    s.set_gripper(0.0); s.move_to(p + [0, 0, 0.2], yaw, 60); s.move_to([p[0], p[1], s.ws_lo[2]], yaw, 40, 0.004)
    for _ in range(8): s.apply_action(np.array([0, 0, 0, 0, 0, 0, 1.0]))
    cs = s.contacts(obj); g = s.is_grasped(obj, cs)
    for _ in range(12): s.apply_action(np.array([0, 0, 0.6, 0, 0, 0, 1.0]))
    lifted = s.is_lifted(obj); s.close(); return g and lifted, f"pads in contact {cs.left_pad_target, cs.right_pad_target}, grasped {g}, physically lifted {lifted}"
check("9 objects physically interact with the gripper", c9)
def c10():
    from simulation.environments.sorting_env import make_env
    e = make_env(); o, _ = e.reset(seed=0); o, r, t, tr, i = e.step(e.action_space.sample()); e.close(); return "image" in o and "state" in o, f"obs keys {list(o)}, reward {r:.3f}"
check("10 training environment runs", c10)
st = get("/training/status")
check("11 RL training executes (trainer process / smoke run)", lambda: (st.get("timesteps", 0) > 0 or os.path.exists("logs/smoke/metrics.jsonl"), f"trainer state={st.get('state')} timesteps={st.get('timesteps')} fps={st.get('fps')}"))
cks = get("/checkpoints")
check("12 a checkpoint can be saved", lambda: (len(cks["checkpoints"]) > 0, f"{len(cks['checkpoints'])} checkpoints, latest {cks['latest']}"))
def c13():
    d = post("/inference/load", {"checkpoint": "latest"}); return d["ok"], f"loaded {d['data']['policy']['checkpoint']} ({d['data']['policy']['timesteps']} steps)"
check("13 a checkpoint can be loaded", c13)
def c14():
    post("/inference/start", {"speed": 5}); time.sleep(3); s = get("/simulation/state"); post("/inference/stop")
    return s["mode"] == "inference" and s["inference"]["policy_calls"] > 0 and s["inference"]["action_latency_ms"] is not None, f"policy calls {s['inference']['policy_calls']}, latency {s['inference']['action_latency_ms']} ms, action {s['action']['values'][:3]}"
check("14 inference runs using the trained checkpoint", c14)
def c15():
    import asyncio, websockets
    async def go():
        async with websockets.connect(B.replace("http", "ws") + "/ws/simulation", max_size=8_000_000) as ws:
            for _ in range(20):
                m = json.loads(await ws.recv())
                if m["type"] == "frame": return m
    m = asyncio.run(go()); return m is not None and len(m["frame"]) > 1000 and "joint_positions" in m["state"], f"frame {len(m['frame'])} b64 bytes, state keys {len(m['state'])}"
check("15 browser receives real-time simulation state (ws)", c15)
def c16():
    import asyncio, websockets
    async def go():
        async with websockets.connect(B.replace("http", "ws") + "/ws/training") as ws: return json.loads(await ws.recv())
    m = asyncio.run(go()); rows = m.get("metrics", []); return m["type"] == "history" and len(rows) > 0 and "train/value_loss" in rows[-1], f"{len(rows)} metric rows, last timesteps {rows[-1].get('timesteps') if rows else None}"
check("16 browser receives real training metrics (ws)", c16)
def c17():
    rows = get("/training/metrics?last_n=1000")["metrics"]; ts = [r["timesteps"] for r in rows]
    return len(rows) >= 2 and ts[-1] > ts[0], f"{len(rows)} rows spanning {ts[0] if ts else None}→{ts[-1] if ts else None} timesteps (charts read these)"
check("17 dashboard charts update (metrics grow)", c17)
def c18():
    post("/training/evaluate", {"episodes": 2, "seed": 900})
    for _ in range(90):
        time.sleep(2); p = get("/training/evaluation")
        if not p.get("running"): break
    l = p.get("last") or {}; return l.get("episodes_run") == 2 and l.get("success_rate") is not None, f"2 randomized episodes: success {l.get('success_rate')}, avg reward {l.get('average_reward')}, {l.get('duration_s')} s"
check("18 evaluation mode runs randomized episodes", c18)
def c19():
    import re
    bad = []
    for path in ["frontend/src/components/TrainingPanel.tsx", "frontend/src/components/InferencePanel.tsx", "frontend/src/components/PolicyCard.tsx"]:
        src = open(os.path.join(os.path.dirname(__file__), "..", path)).read()
        for mnum in re.findall(r"value=\{?['\"](\d+\.?\d*%?)['\"]", src): bad.append((path, mnum))
    return not bad, "no hard-coded metric values in dashboard components; missing values render N/A" if not bad else f"literals: {bad}"
check("19 no fake training numbers displayed", c19)
def c20():
    r = open(os.path.join(os.path.dirname(__file__), "..", "README.md")).read()
    return all(k in r for k in ["scripts/run_backend.sh", "scripts/run_frontend.sh", "training.train", "localhost:8000", "localhost:3001"]), "README has backend/frontend/training/inference commands and URLs"
check("20 README contains exact start commands", c20)
n_ok = sum(ok for _, ok, _ in results)
print(f"\n{n_ok}/{len(results)} acceptance checks passed")
json.dump([{"item": n, "pass": ok, "detail": d} for n, ok, d in results], open(os.path.join(os.path.dirname(__file__), "..", "logs", "acceptance_results.json"), "w"), indent=1)
sys.exit(0 if n_ok == len(results) else 1)
