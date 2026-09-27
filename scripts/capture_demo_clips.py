"""Capture real simulation episodes for the recorded browser demo (Vercel static deployment).

Runs the trained policy in an in-process SimulationService on several scenarios and writes, per control step, the
viewport JPEG, the overhead (policy-input) PNG and the full state payload the dashboard normally receives over the
WebSocket. Also snapshots the REST payloads the dashboard reads (checkpoints, results, telemetry, catalog).
Everything written is real data from this system; the demo UI labels it as a recording.
"""
import os, sys, json, time, base64, shutil, argparse
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, cv2, requests
from simulation.config import load_configs, resolve_path
from backend.sim_service import SimulationService

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="frontend/public/demo")
ap.add_argument("--width", type=int, default=840)
ap.add_argument("--quality", type=int, default=68)
ap.add_argument("--max-steps", type=int, default=200)
ap.add_argument("--backend", default="http://localhost:8000")
args = ap.parse_args()
OUT = resolve_path(args.out)

cfg = load_configs()
svc = SimulationService(cfg["simulation"], cfg["environment"], cfg["training"]); svc.start(); svc.ready.wait(120)
svc.call(svc.cmd_load_checkpoint, checkpoint="best")
pol = svc.policy.info()
scen = requests.get(args.backend + "/simulation/scenarios").json()
custom = json.loads(resolve_path("configs/custom_scenarios.json").read_text()) if resolve_path("configs/custom_scenarios.json").exists() else {}
CLIPS = [("nominal", "Nominal layout"), ("bracket_rot75", "Bracket rotated 75°"), ("both_moved", "Both parts moved"), ("bolt_moved", "Bolt moved"), ("clutter_showcase", "Clutter: extra bracket + red cap")]

def compact_frame(msg, path_prefix):
    view = cv2.imdecode(np.frombuffer(base64.b64decode(msg["frame"]), np.uint8), cv2.IMREAD_COLOR)
    h = int(round(view.shape[0] * args.width / view.shape[1]))
    view = cv2.resize(view, (args.width, h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path_prefix) + ".jpg", view, [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])
    if msg.get("overhead"):
        ov = cv2.imdecode(np.frombuffer(base64.b64decode(msg["overhead"]), np.uint8), cv2.IMREAD_COLOR)
        cv2.imwrite(str(path_prefix) + "_o.png", ov)
    st = dict(msg["state"]); st["state_vector"] = None   # 44 floats/step not needed for playback
    return st

manifest = {"captured_iso": time.strftime("%Y-%m-%dT%H:%M:%S"), "checkpoint": pol, "fps": int(round(1 / svc.env.sim.control_dt)), "clips": [],
            "note": "Recorded playback of real simulation episodes (MuJoCo + trained PPO policy). Live control requires the local backend."}
for key, label in CLIPS:
    d = OUT / "clips" / key
    if d.exists(): shutil.rmtree(d)
    d.mkdir(parents=True)
    def prep():
        sb = svc.scene_builder
        if key in custom:
            sb.apply(custom[key]); poses = custom[key]["poses"]
        else:
            sb.ensure_defaults(); poses = scen["scenarios"][key]
        svc.cmd_reset(seed=777, randomize=True, fixed_poses=poses)
        svc.mode = "inference"; svc.paused = False
    svc.call(prep)
    frames = []
    for t in range(args.max_steps):
        msg = svc.call(lambda: svc._make_frame_message())
        frames.append({"i": t, "state": compact_frame(msg, d / f"f{t:04d}"), "t": msg["t"]})
        info = svc.call(lambda: (svc._step_once(), svc.last_info)[1])
        if info.get("is_success") or svc.env.t == 0:      # episode ended -> _on_episode_end reset the env
            msg = svc.call(lambda: svc._make_frame_message()); frames.append({"i": t + 1, "state": compact_frame(msg, d / f"f{t+1:04d}"), "t": msg["t"]})
            break
    last = svc.episode_results[-1] if svc.episode_results else {}
    (d / "frames.json").write_text(json.dumps({"frames": frames}, separators=(",", ":")))
    size_mb = sum(f.stat().st_size for f in d.iterdir()) / 1e6
    manifest["clips"].append({"id": key, "label": label, "frames": len(frames), "dir": f"demo/clips/{key}", "outcome": {k: last.get(k) for k in ("success", "placed_count", "reward", "length", "collision_steps")}, "size_mb": round(size_mb, 1)})
    print(f"clip {key:18s} {len(frames):3d} frames  {size_mb:5.1f} MB  outcome={manifest['clips'][-1]['outcome']}")
    svc.call(svc.cmd_set_mode, mode="idle")
svc.call(svc.scene_builder.ensure_defaults)

# ---- REST snapshots the dashboard reads (from the live backend, so they are exactly what the UI shows now) ----
S = OUT / "static"
for name, path in {"config": "/config", "checkpoints": "/checkpoints", "training_status": "/training/status", "training_metrics": "/training/metrics?last_n=400",
                   "training_evaluation": "/training/evaluation", "scorecard": "/evaluation/scorecard", "sweep": "/evaluation/sweep", "compare": "/evaluation/compare",
                   "demos": "/demos", "catalog": "/scene/catalog", "scenarios": "/simulation/scenarios", "reels": "/evaluation/reels?limit=12", "health": "/health", "keymap": "/teleop/keymap"}.items():
    r = requests.get(args.backend + path, timeout=60); (S / f"{name}.json").write_text(r.text)
reels = json.loads((S / "reels.json").read_text())
for rr in reels["reels"]:
    src = resolve_path("logs/eval_reels") / rr["id"] / "filmstrip.jpg"
    dst = OUT / "reels" / rr["id"].replace("/", "__") / "filmstrip.jpg"
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy(src, dst)
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e6
print(f"demo bundle: {total:.1f} MB in {OUT}")
svc.stop()
