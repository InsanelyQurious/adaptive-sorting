"""FastAPI backend: REST + WebSocket API for the Adaptive Sorting sorting demo."""
from __future__ import annotations
import asyncio
import json
import os
import queue
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from simulation.config import load_configs, resolve_path, PROJECT_ROOT
from backend.models import *  # noqa
from backend.sim_service import SimulationService
from backend.training_manager import TrainingManager
from backend.device import device_label, gpu_utilization
from training.checkpointing import list_checkpoints, latest_checkpoint, best_checkpoint
from lerobot_adapter.demo_recorder import list_demos, demo_summary, discard_demo, DEMO_TYPES

VERSION = "0.1.0"
CONFIGS = load_configs()
app = FastAPI(title="Adaptive Sorting — Robotic Component Sorting", version=VERSION)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

sim = SimulationService(CONFIGS["simulation"], CONFIGS["environment"], CONFIGS["training"])
trainer = TrainingManager(CONFIGS["training"])
START = time.time()


@app.on_event("startup")
async def _startup():
    sim.start()
    sim.ready.wait(timeout=120)
    if sim.error:
        print("SIM ERROR:", sim.error)
    # auto-load the BEST-evaluated checkpoint if one exists (not merely the most recent), so inference is one click away
    try:
        if best_checkpoint(CONFIGS["training"].get("checkpoint_dir", "checkpoints")) is not None:
            await asyncio.to_thread(sim.call, sim.cmd_load_checkpoint, checkpoint="best")
    except Exception as e:
        print("checkpoint autoload failed:", e)


@app.on_event("shutdown")
async def _shutdown():
    sim.stop()


def _ok(message: str = "", **data) -> dict:
    return {"ok": True, "message": message, "data": data or None}


async def _sim_call(fn, **kwargs):
    if not sim.ready.is_set() or sim.env is None:
        raise HTTPException(503, "simulation not ready")
    try:
        return await asyncio.to_thread(sim.call, fn, **kwargs)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        raise HTTPException(400, str(e))


# ----------------------------------------------------------------------------- health
@app.get("/health", response_model=HealthResponse)
async def health():
    import mujoco, torch, stable_baselines3
    try:
        import lerobot
        lr = lerobot.__version__
    except Exception:
        lr = None
    return HealthResponse(status="ok" if sim.ready.is_set() and sim.env is not None else "starting", version=VERSION,
                          device=device_label(sim.device, sim.device_name), device_name=sim.device_name,
                          mujoco_version=mujoco.__version__, torch_version=torch.__version__,
                          sb3_version=stable_baselines3.__version__, lerobot_version=lr,
                          robot=CONFIGS["simulation"]["robot"]["name"], sim_mode=sim.mode,
                          uptime_s=round(time.time() - START, 1))


@app.get("/config")
async def get_config():
    return {"simulation": CONFIGS["simulation"], "environment": CONFIGS["environment"], "training": CONFIGS["training"],
            "header_title": os.environ.get("DEMO_TITLE", "ADAPTIVE SORTING"),
            "header_subtitle": os.environ.get("DEMO_SUBTITLE", "Autonomous Robotic Component Sorting"),
            "header_tag": os.environ.get("DEMO_TAG", "MuJoCo × LeRobot")}


# ----------------------------------------------------------------------------- simulation
@app.get("/simulation/state")
async def simulation_state():
    if not sim.ready.is_set() or sim.env is None:
        raise HTTPException(503, "simulation not ready")
    return await asyncio.to_thread(sim.call, lambda: sim.state_payload())


@app.post("/simulation/reset")
async def simulation_reset(req: ResetRequest):
    st = await _sim_call(sim.cmd_reset, seed=req.seed, randomize=req.randomize,
                         randomization_level=req.randomization_level, fixed_poses=req.fixed_poses)
    return _ok("reset", state=st)


@app.post("/simulation/randomize")
async def simulation_randomize(req: RandomizeRequest):
    st = await _sim_call(sim.cmd_reset, seed=req.seed, randomize=True, randomization_level=req.randomization_level)
    return _ok("randomized", episode_config=st["episode_config"])


@app.post("/simulation/pause")
async def simulation_pause():
    return _ok("paused", **await _sim_call(sim.cmd_set_paused, paused=True))


@app.post("/simulation/resume")
async def simulation_resume():
    return _ok("resumed", **await _sim_call(sim.cmd_set_paused, paused=False))


@app.post("/simulation/step")
async def simulation_step(req: StepRequest):
    st = await _sim_call(sim.cmd_step, steps=req.steps)
    return _ok("stepped", step=st["step"], reward=st["reward"])


@app.post("/simulation/speed")
async def simulation_speed(req: SpeedRequest):
    sim.speed = float(req.speed)
    return _ok("speed set", speed=sim.speed)


@app.post("/simulation/mode")
async def simulation_mode(req: ModeRequest):
    return _ok("mode set", **await _sim_call(sim.cmd_set_mode, mode=req.mode))


@app.get("/simulation/episodes")
async def simulation_episodes(limit: int = 50):
    return {"episodes": list(sim.episode_results)[-limit:], "stats": sim.inference_stats}


# ----------------------------------------------------------------------------- training
@app.post("/training/start")
async def training_start(req: TrainingStartRequest):
    overrides: dict[str, Any] = {}
    for k in ("total_timesteps", "learning_rate", "batch_size", "gamma", "ent_coef", "seed", "n_envs", "run_name",
              "training_episodes"):
        v = getattr(req, k)
        if v is not None:
            overrides[k] = v
    if req.randomization_level is not None:
        overrides["randomization_level"] = req.randomization_level
    boot: dict[str, Any] = {}
    if req.init_from_demos is not None:
        boot["init_from_demos"] = bool(req.init_from_demos)
    if req.demo_types:
        boot["demo_types"] = list(req.demo_types)
    if req.bc_epochs is not None:
        boot["bc_epochs"] = int(req.bc_epochs)
    if boot:
        overrides["bootstrap"] = boot
    if boot.get("init_from_demos"):
        demos = [d for d in list_demos(sim.demos_dir) if d.get("episode_type") in (boot.get("demo_types") or CONFIGS["training"].get("bootstrap", {}).get("demo_types", list(DEMO_TYPES)))]
        if not demos:
            raise HTTPException(400, "init_from_demos requested but no demonstration episodes are recorded (Teleop tab -> record, or generate scripted demos)")
    if sim.mode in ("teleop", "scripted_demo"):
        await _sim_call(sim.cmd_set_mode, mode="idle")
    try:
        info = await asyncio.to_thread(trainer.start, overrides, req.resume_from)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return _ok("training started", **info)


@app.post("/training/pause")
async def training_pause():
    try:
        trainer.pause()
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return _ok("pause requested")


@app.post("/training/resume")
async def training_resume():
    try:
        trainer.resume()
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return _ok("resume requested")


@app.post("/training/stop")
async def training_stop():
    res = await asyncio.to_thread(trainer.stop)
    return _ok("stop requested", **res)


@app.post("/training/save")
async def training_save():
    try:
        trainer.save()
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return _ok("checkpoint save requested")


@app.post("/training/evaluate-in-trainer")
async def training_evaluate_in_trainer(req: EvaluateRequest):
    """Asks the running trainer process to run its periodic evaluation now (results land in metrics/status)."""
    try:
        trainer.evaluate(req.episodes)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return _ok("trainer evaluation requested", episodes=req.episodes)


@app.post("/training/evaluate")
async def training_evaluate(req: EvaluateRequest):
    """Runs evaluation on the live simulation with the loaded (or given) checkpoint."""
    if req.checkpoint:
        await _sim_call(sim.cmd_load_checkpoint, checkpoint=req.checkpoint)
    prog = await _sim_call(sim.cmd_start_evaluation, episodes=req.episodes,
                           randomization_level=req.randomization_level, seed=req.seed)
    return _ok("evaluation started", **prog)


def _all_scenarios() -> dict:
    """Built-in presets + user-authored custom scenarios (Phase 3)."""
    out = {k: {"poses": v, "builtin": True} for k, v in SCENARIOS.items()}
    try:
        from backend.scene_builder import load_custom_scenarios
        for k, v in load_custom_scenarios().items():
            out[k] = {"poses": v.get("poses", {}), "builtin": False, "layout": v.get("layout"), "variants": v.get("variants")}
    except Exception:
        pass
    return out


def _latest_json(name: str) -> dict | None:
    p = resolve_path(f"logs/{name}_latest.json")
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


@app.post("/evaluation/scorecard")
async def evaluation_scorecard(req: ScorecardRequest):
    """Generalization scorecard: Evaluation across every preset (and custom) scenario, back to back, on the live sim."""
    scen = _all_scenarios()
    names = req.scenarios or list(scen.keys())
    items = []
    for n in names:
        if n not in scen:
            raise HTTPException(404, f"unknown scenario {n}")
        items.append({"label": n, "fixed_poses": scen[n]["poses"], "layout": scen[n].get("layout"), "variants": scen[n].get("variants")})
    st = await _sim_call(sim.cmd_start_batch, kind="scorecard", items=items, episodes=req.episodes, seed=req.seed, randomization_level=req.randomization_level)
    return _ok("scorecard started", batch=st)


@app.get("/evaluation/scorecard")
async def evaluation_scorecard_get():
    return {"running": sim.batch_status() if (sim.batch and sim.batch["kind"] == "scorecard") else None, "last": _latest_json("scorecard")}


@app.post("/evaluation/sweep")
async def evaluation_sweep(req: SweepRequest):
    """Difficulty sweep: Evaluation at each randomization level (objects fully randomized at that level)."""
    items = [{"label": f"level {lv:.2f}", "randomization_level": float(lv), "level": float(lv)} for lv in req.levels]
    st = await _sim_call(sim.cmd_start_batch, kind="sweep", items=items, episodes=req.episodes, seed=req.seed)
    return _ok("difficulty sweep started", batch=st)


@app.get("/evaluation/sweep")
async def evaluation_sweep_get():
    return {"running": sim.batch_status() if (sim.batch and sim.batch["kind"] == "sweep") else None, "last": _latest_json("sweep")}


@app.post("/evaluation/compare")
async def evaluation_compare(req: CompareRequest):
    """Before/after: evaluate two checkpoints under identical conditions (same seeds, level, episode count), freshly."""
    for c in (req.checkpoint_a, req.checkpoint_b):
        from backend.policy_loader import resolve_checkpoint
        if resolve_checkpoint(c, CONFIGS["training"].get("checkpoint_dir", "checkpoints")) is None:
            raise HTTPException(404, f"checkpoint not found: {c}")
    items = [{"label": req.checkpoint_a, "checkpoint": req.checkpoint_a}, {"label": req.checkpoint_b, "checkpoint": req.checkpoint_b}]
    st = await _sim_call(sim.cmd_start_batch, kind="compare", items=items, episodes=req.episodes, seed=req.seed, randomization_level=req.randomization_level)
    return _ok("comparison started", batch=st)


@app.get("/evaluation/compare")
async def evaluation_compare_get():
    return {"running": sim.batch_status() if (sim.batch and sim.batch["kind"] == "compare") else None, "last": _latest_json("compare")}


@app.post("/evaluation/batch/abort")
async def evaluation_batch_abort():
    return _ok("batch aborted", **await _sim_call(sim.cmd_abort_batch))


@app.get("/evaluation/reels")
async def evaluation_reels(limit: int = 60):
    """Saved evaluation episodes (every failure + one success reference per evaluation) with frames and traces."""
    root = sim.reels_dir
    out = []
    if root.exists():
        for meta in sorted(root.rglob("meta.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
            try:
                m = json.loads(meta.read_text())
                d = meta.parent
                m["id"] = f"{d.parent.name}/{d.name}"
                m["files"] = sorted(f.name for f in d.iterdir() if f.suffix in (".jpg", ".png", ".json"))
                out.append(m)
            except Exception:
                continue
    return {"reels": out, "dir": str(root)}


@app.get("/evaluation/reels/{eval_id}/{ep}/{fname}")
async def evaluation_reel_file(eval_id: str, ep: str, fname: str):
    p = (sim.reels_dir / eval_id / ep / fname).resolve()
    if not str(p).startswith(str(sim.reels_dir.resolve())) or not p.exists():
        raise HTTPException(404, "not found")
    return FileResponse(p)


@app.get("/training/evaluation")
async def training_evaluation():
    prog = sim.eval_progress()
    latest = resolve_path("logs/evaluation_latest.json")
    if latest.exists() and not prog.get("running"):
        try:
            summary = json.loads(latest.read_text())
            prog["last"] = {k: v for k, v in summary.items() if k != "episodes"}
            prog["last_episodes"] = summary.get("episodes", [])[-50:]
        except Exception:
            pass
    return prog


@app.get("/training/status")
async def training_status():
    st = trainer.read_status()
    st["gpu_utilization"] = gpu_utilization()
    return st


@app.get("/training/metrics")
async def training_metrics(last_n: int = 500):
    return {"metrics": trainer.read_metrics(last_n)}


@app.get("/training/log")
async def training_log(lines: int = 200):
    p = resolve_path("logs/training.log")
    if not p.exists():
        return {"lines": []}
    with open(p, "rb") as f:
        tail = f.readlines()[-lines:]
    return {"lines": [ln.decode(errors="replace").rstrip() for ln in tail]}


# ----------------------------------------------------------------------------- checkpoints
@app.get("/checkpoints", response_model=CheckpointList)
async def checkpoints():
    cdir = CONFIGS["training"].get("checkpoint_dir", "checkpoints")
    cks = list_checkpoints(cdir)
    lt = latest_checkpoint(cdir)
    bt = best_checkpoint(cdir)
    return CheckpointList(checkpoints=cks, latest=lt.name if lt else None, best=bt.name if bt else None,
                          loaded=sim.policy.name if sim.policy else None)


# ----------------------------------------------------------------------------- inference
@app.post("/inference/load")
async def inference_load(req: LoadCheckpointRequest):
    info = await _sim_call(sim.cmd_load_checkpoint, checkpoint=req.checkpoint)
    return _ok("checkpoint loaded", policy=info)


@app.post("/inference/start")
async def inference_start(req: InferenceStartRequest):
    if req.checkpoint or sim.policy is None:
        await _sim_call(sim.cmd_load_checkpoint, checkpoint=req.checkpoint or "best")
    sim.deterministic = req.deterministic
    if req.speed:
        sim.speed = float(req.speed)
    res = await _sim_call(sim.cmd_set_mode, mode="inference")
    await _sim_call(sim.cmd_set_paused, paused=False)
    return _ok("inference running", **res, policy=sim.policy.info())


@app.post("/inference/stop")
async def inference_stop():
    res = await _sim_call(sim.cmd_set_mode, mode="idle")
    return _ok("inference stopped", **res)


@app.get("/inference/status")
async def inference_status():
    return {"mode": sim.mode, "paused": sim.paused, "policy": sim.policy.info() if sim.policy else None,
            "stats": sim.inference_stats, "device": device_label(sim.device, sim.device_name)}


# ----------------------------------------------------------------------------- dataset recording
@app.post("/dataset/record/start")
async def dataset_record_start(req: RecordStartRequest | None = None):
    """Record (RGB image, timestamp, robot state, action, reward, done) tuples from the live sim as
    LeRobotDataset episodes (one directory per episode; demos are tagged by episode_type)."""
    req = req or RecordStartRequest()
    return _ok("recording started", **await _sim_call(sim.cmd_record_start, name=req.name, episode_type=req.episode_type, reset=req.reset))


@app.post("/dataset/record/stop")
async def dataset_record_stop():
    return _ok("recording stopped", **await _sim_call(sim.cmd_record_stop))


@app.get("/dataset/recordings")
async def dataset_recordings():
    root = resolve_path("datasets")
    out = []
    if root.exists():
        for p in sorted(root.rglob("meta/info.json")):
            try:
                info = json.loads(p.read_text())
                out.append({"path": str(p.parent.parent), "episodes": info.get("total_episodes"), "frames": info.get("total_frames"),
                            "fps": info.get("fps"), "codebase_version": info.get("codebase_version")})
            except Exception:
                continue
    return {"recordings": out, "live": sim.recording_status()}


# ----------------------------------------------------------------------------- teleoperation + demonstrations
def _controllers_idle():
    """Teleop may only take the arm when neither training nor inference/evaluation is driving it."""
    if sim.estop:
        raise HTTPException(409, "E-STOP is engaged - reset it first")
    if trainer.is_running():
        raise HTTPException(409, "training is running - stop it before teleoperating")
    if sim.mode in ("inference", "evaluate", "random", "replay"):
        raise HTTPException(409, f"simulation is in {sim.mode} mode - stop it first")


@app.post("/teleop/start")
async def teleop_start(req: TeleopStartRequest | None = None):
    req = req or TeleopStartRequest()
    _controllers_idle()
    res = await _sim_call(sim.cmd_teleop_start, reset=req.reset)
    return _ok("teleop active", **res, keymap=TELEOP_KEYMAP)


@app.post("/teleop/stop")
async def teleop_stop():
    return _ok("teleop stopped", **await _sim_call(sim.cmd_teleop_stop))


@app.post("/teleop/action")
async def teleop_action(req: TeleopActionRequest):
    """REST fallback for /ws/teleop: set the current end-effector velocity command (same action space as the policy)."""
    if sim.mode != "teleop":
        raise HTTPException(409, "teleop is not active")
    sim.teleop_update(req.axes, req.gripper)
    return _ok("ok")


TELEOP_KEYMAP = {
    "W/S": "+X / -X (dX)", "A/D": "+Y / -Y (dY)", "Q/E": "+Z up / -Z down (dZ)",
    "J/L or ←/→": "yaw -/+ (dRz)", "I/K or ↑/↓": "dRx +/- (locked to 0 by control.max_delta_rot)",
    "U/O": "dRy +/- (locked to 0 by control.max_delta_rot)", "Space": "toggle gripper open/close",
    "Shift": "fine motion (30 %)", "R": "start / stop recording", "N": "new episode (reset + randomize)",
    "mouse drag on viewport": "X / Y motion while dragging",
}


@app.post("/teleop/pick_at")
async def teleop_pick_at(req: PickAtRequest):
    """Click-to-pick: resolve the clicked viewport pixel with a segmentation render of the viewport camera; an
    object starts an assisted pick-and-place by the scripted controller (default: its mapped bin), a bin re-routes it."""
    if sim.mode != "teleop":
        raise HTTPException(409, "click-to-pick is only available in teleop mode")
    return _ok("pick", **await _sim_call(sim.cmd_pick_at, u=req.u, v=req.v, bin_override=req.bin))


@app.post("/teleop/pick")
async def teleop_pick(req: PickByNameRequest):
    """Assisted pick-and-place by object name (used by the chat panel)."""
    if sim.mode != "teleop":
        _controllers_idle()
        await _sim_call(sim.cmd_teleop_start, reset=False)
    if req.object not in sim.env.objects:
        raise HTTPException(400, f"unknown object {req.object}; options: {sim.env.objects}")
    return _ok("assisted pick started", assist=await _sim_call(sim._start_assist, obj=req.object, bin_name=req.bin))


@app.post("/teleop/assist/cancel")
async def teleop_assist_cancel():
    return _ok("assist cancelled", **await _sim_call(sim.cmd_assist_cancel))


@app.post("/demos/{name}/replay")
async def demos_replay(name: str, autoplay: bool = True):
    """Deterministic open-loop replay of a recorded episode in the live viewport (mode: replay)."""
    if trainer.is_running():
        raise HTTPException(409, "training is running - stop it before replaying")
    try:
        return _ok("replay started", replay=await _sim_call(sim.cmd_replay_start, name=name, autoplay=autoplay))
    except HTTPException:
        raise


@app.post("/replay/seek")
async def replay_seek(req: ReplaySeekRequest):
    return _ok("seek", replay=await _sim_call(sim.cmd_replay_seek, index=req.index))


@app.post("/replay/play")
async def replay_play():
    return _ok("play", replay=await _sim_call(sim.cmd_replay_set_playing, playing=True))


@app.post("/replay/pause")
async def replay_pause():
    return _ok("pause", replay=await _sim_call(sim.cmd_replay_set_playing, playing=False))


@app.post("/replay/stop")
async def replay_stop():
    return _ok("replay stopped", **await _sim_call(sim.cmd_replay_stop))


@app.get("/demos/{name}/detail")
async def demos_detail(name: str):
    from lerobot_adapter.demo_recorder import load_demo_episode
    for root in (sim.demos_dir, resolve_path("datasets/live_recordings")):
        if (root / name / "demo.json").exists():
            ep = await asyncio.to_thread(load_demo_episode, root / name)
            return {"name": name, "frames": int(len(ep["actions"])), "actions": ep["actions"].round(3).tolist(),
                    "rewards": ep["rewards"].round(3).tolist(), "sidecar": {k: v for k, v in ep["sidecar"].items() if k != "initial_state"},
                    "has_initial_state": bool(ep["sidecar"].get("initial_state"))}
    raise HTTPException(404, f"no such recorded episode: {name}")


@app.get("/teleop/keymap")
async def teleop_keymap():
    return {"keymap": TELEOP_KEYMAP, "action_names": ["dX", "dY", "dZ", "dRx", "dRy", "dRz", "gripper"],
            "max_delta_pos_m": CONFIGS["simulation"]["control"]["max_delta_pos"],
            "max_delta_rot_rad": CONFIGS["simulation"]["control"]["max_delta_rot"], "control_hz": round(1.0 / CONFIGS["simulation"]["control"]["control_dt"], 1)}


@app.websocket("/ws/teleop")
async def ws_teleop(ws: WebSocket):
    """Browser -> backend teleop stream. Messages: {"type":"action","axes":[6 floats],"gripper":1|-1|null}.
    Commands stop being applied 0.5 s after the last message (dead-man), the gripper state persists."""
    await ws.accept()
    sim.teleop_clients += 1
    try:
        while True:
            raw = await ws.receive_text()
            try:
                m = json.loads(raw)
            except Exception:
                continue
            if m.get("type") == "action" and sim.mode == "teleop":
                sim.teleop_update(m.get("axes"), m.get("gripper"))
            elif m.get("type") == "ping":
                await ws.send_text(json.dumps({"type": "pong", "t": time.time()}))
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        sim.teleop_clients = max(0, sim.teleop_clients - 1)


@app.get("/demos")
async def demos_list():
    demos = list_demos(sim.demos_dir)
    return {"demos": demos, "summary": demo_summary(demos), "recording": sim.recording_status(), "dir": str(sim.demos_dir)}


@app.delete("/demos/{name}")
async def demos_discard(name: str):
    try:
        discard_demo(sim.demos_dir, name)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _ok(f"discarded {name}")


@app.post("/demos/scripted")
async def demos_scripted(req: ScriptedDemoRequest):
    """Generate `scripted_demo` episodes with the hand-coded ground-truth controller on the live sim.
    These are pipeline validators - never human demonstrations and never learned-policy performance."""
    _controllers_idle()
    if sim.mode == "teleop":
        await _sim_call(sim.cmd_teleop_stop)
    res = await _sim_call(sim.cmd_scripted_demos, episodes=req.episodes, noise_std=req.noise_std, record=req.record)
    return _ok("scripted demo generation started", **res)


@app.post("/simulation/frustum")
async def simulation_frustum(req: FrustumRequest):
    """Toggle the overhead camera's field-of-view overlay in the 3rd-person viewport (visual only)."""
    return _ok("frustum overlay", **await _sim_call(sim.cmd_set_frustum, enabled=req.enabled))


@app.post("/simulation/wrist_camera")
async def simulation_wrist_camera(req: FrustumRequest):
    """Toggle the wrist-mounted camera thumbnail in the dashboard (display only - not a policy input)."""
    return _ok("wrist camera", **await _sim_call(sim.cmd_set_wrist, enabled=req.enabled))


@app.get("/simulation/camera")
async def simulation_camera():
    """Pose / intrinsics of the learning camera, derived from the compiled MuJoCo model."""
    if not sim.ready.is_set() or sim.env is None:
        raise HTTPException(503, "simulation not ready")
    return sim.camera_info or {}


# ----------------------------------------------------------------------------- scene builder (parts catalog)
def _sb():
    if not sim.ready.is_set() or sim.scene_builder is None:
        raise HTTPException(503, "scene builder not ready")
    return sim.scene_builder


@app.get("/scene/catalog")
async def scene_catalog():
    """Parts catalog with thumbnails (rendered from the real MJCF bodies at start-up) + current table layout."""
    sb = _sb()
    return await asyncio.to_thread(sim.call, sb.catalog_payload)


@app.post("/scene/spawn")
async def scene_spawn(req: SpawnRequest):
    """Drop a catalog part on the table: (u, v) viewport pixel -> camera ray -> table plane, or explicit x/y."""
    sb = _sb()
    if req.u is None and req.x is None:
        raise HTTPException(400, "give either u/v (viewport pixel) or x/y (table coordinates)")
    if sim.mode in ("evaluate", "replay", "scripted_demo"):
        raise HTTPException(409, f"cannot author the scene while in {sim.mode} mode")
    try:
        res = await _sim_call(sb.spawn, part=req.part, u=req.u, v=req.v, x=req.x, y=req.y, yaw=req.yaw)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _ok(f"spawned {req.part}", **res)


@app.post("/scene/toggle")
async def scene_toggle(req: ToggleRequest):
    """Select (place) or deselect (park) any catalog part independently. A table with no bracket, no bolt, or
    several parts of one type is valid; the policy targets whichever bracket/bolt is the tracked one."""
    sb = _sb()
    if sim.mode in ("evaluate", "replay", "scripted_demo"):
        raise HTTPException(409, f"cannot author the scene while in {sim.mode} mode")
    try:
        res = await _sim_call(sb.toggle, part=req.part, on=req.on, u=req.u, v=req.v, x=req.x, y=req.y, yaw=req.yaw)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _ok(f"{req.part} {'selected' if res.get('on') else 'deselected'}", **res)


@app.post("/scene/defaults")
async def scene_defaults():
    sb = _sb()
    return _ok("default bracket and bolt placed", **await _sim_call(sb.ensure_defaults))


@app.delete("/scene/parts/{part}")
async def scene_remove(part: str):
    sb = _sb()
    try:
        return _ok(f"removed {part}", **await _sim_call(sb.remove, part=part))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/scene/clear")
async def scene_clear():
    sb = _sb()
    return _ok("table cleared", **await _sim_call(sb.clear))


@app.post("/scene/variants/reset")
async def scene_variants_reset():
    sb = _sb()
    return _ok("default variants restored", **await _sim_call(sb.reset_variants))


@app.post("/scene/scenarios")
async def scene_save_scenario(req: SaveScenarioRequest):
    """Store the current authored layout (task-object poses, variants, clutter) as a named custom scenario."""
    sb = _sb()
    try:
        res = await _sim_call(sb.save_scenario, name=req.name)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _ok(f"scenario saved: {res['name']}", **res)


@app.delete("/scene/scenarios/{name}")
async def scene_delete_scenario(name: str):
    sb = _sb()
    try:
        return _ok(f"scenario deleted: {name}", **await asyncio.to_thread(sb.delete_scenario, name))
    except FileNotFoundError:
        raise HTTPException(404, f"no custom scenario {name}")


# ----------------------------------------------------------------------------- safety (simulated e-stop) + export
@app.get("/safety")
async def safety_get():
    if not sim.ready.is_set() or sim.env is None:
        raise HTTPException(503, "simulation not ready")
    return await asyncio.to_thread(sim.call, sim.safety_status)


@app.post("/safety/estop")
async def safety_estop(reason: str = "operator"):
    """Engage the simulated e-stop (see README: halts between control steps, holds joint targets; simulation-only)."""
    return _ok("E-STOP engaged", **await _sim_call(sim.cmd_estop, reason=reason))


@app.post("/safety/reset")
async def safety_reset():
    return _ok("E-STOP reset", **await _sim_call(sim.cmd_estop_reset))


@app.get("/export/summary.html", response_class=HTMLResponse)
async def export_summary():
    """Static HTML executive summary built from persisted results + the live sim (real numbers only)."""
    from backend.export_summary import build_summary_html
    if not sim.ready.is_set() or sim.env is None:
        raise HTTPException(503, "simulation not ready")
    hdr = {"title": os.environ.get("DEMO_TITLE", "ADAPTIVE SORTING"), "subtitle": os.environ.get("DEMO_SUBTITLE", "Autonomous Robotic Component Sorting")}
    html_doc = await asyncio.to_thread(sim.call, build_summary_html, svc=sim, header=hdr)
    out = resolve_path("logs") / f"summary_{time.strftime('%Y%m%d_%H%M%S')}.html"
    out.write_text(html_doc)
    return HTMLResponse(html_doc, headers={"Content-Disposition": f'attachment; filename="{out.name}"'})


@app.get("/export/summary/preview", response_class=HTMLResponse)
async def export_summary_preview():
    from backend.export_summary import build_summary_html
    hdr = {"title": os.environ.get("DEMO_TITLE", "ADAPTIVE SORTING"), "subtitle": os.environ.get("DEMO_SUBTITLE", "Autonomous Robotic Component Sorting")}
    return HTMLResponse(await asyncio.to_thread(sim.call, build_summary_html, svc=sim, header=hdr))


# ----------------------------------------------------------------------------- demo scenarios
SCENARIOS = {
    "nominal": {"bracket": {"x": 0.12, "y": 0.10, "yaw": 0.0}, "bolt": {"x": 0.26, "y": -0.10, "yaw": 0.0}},
    "bracket_rot20": {"bracket": {"x": 0.12, "y": 0.10, "yaw": 0.349}, "bolt": {"x": 0.26, "y": -0.10, "yaw": 0.0}},
    "bracket_rot75": {"bracket": {"x": 0.12, "y": 0.10, "yaw": 1.309}, "bolt": {"x": 0.26, "y": -0.10, "yaw": 0.0}},
    "bolt_moved": {"bracket": {"x": 0.12, "y": 0.10, "yaw": 0.0}, "bolt": {"x": 0.06, "y": -0.18, "yaw": 1.2}},
    "both_moved": {"bracket": {"x": 0.30, "y": -0.05, "yaw": -0.9}, "bolt": {"x": 0.05, "y": 0.17, "yaw": 2.4}},
}


@app.get("/simulation/scenarios")
async def simulation_scenarios():
    scen = _all_scenarios()
    return {"scenarios": {k: v["poses"] for k, v in scen.items()}, "builtin": [k for k, v in scen.items() if v["builtin"]],
            "custom": [k for k, v in scen.items() if not v["builtin"]]}


@app.post("/simulation/scenario/{name}")
async def simulation_scenario(name: str, seed: int | None = None):
    """Reset to a named fixed object layout (other randomization stays at the configured level). Custom scenarios
    (Phase 3) also restore their authored part variants and clutter layout."""
    scen = _all_scenarios()
    if name not in scen:
        raise HTTPException(404, f"unknown scenario {name}; options: {list(scen)}")
    sc = scen[name]
    if not sc["builtin"]:
        from backend.scene_builder import apply_scenario_layout
        await _sim_call(apply_scenario_layout, sim=sim, scenario=sc)
    elif sim.scene_builder is not None:
        await _sim_call(sim.scene_builder.ensure_defaults)      # presets = clean table: bracket + bolt + bins, no extras
    st = await _sim_call(sim.cmd_reset, seed=seed, randomize=True, randomization_level=None, fixed_poses=sc["poses"])
    return _ok(f"scenario {name}", episode_config=st["episode_config"])


# ----------------------------------------------------------------------------- websockets
@app.websocket("/ws/simulation")
async def ws_simulation(ws: WebSocket):
    await ws.accept()
    q = sim.subscribe()
    try:
        while True:
            try:
                msg = await asyncio.to_thread(q.get, True, 1.0)
            except queue.Empty:
                await ws.send_text(json.dumps({"type": "heartbeat", "t": time.time()}))
                continue
            await ws.send_text(json.dumps(msg))
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        sim.unsubscribe(q)


@app.websocket("/ws/training")
async def ws_training(ws: WebSocket):
    """Pushes trainer status every second and every new metrics line as it appears."""
    await ws.accept()
    try:
        history = trainer.read_metrics(400)
        offset = trainer.metrics_size()
        await ws.send_text(json.dumps({"type": "history", "metrics": history, "status": trainer.read_status()}))
        while True:
            new, offset = trainer.read_new_metrics(offset)
            st = trainer.read_status()
            st["gpu_utilization"] = gpu_utilization()
            await ws.send_text(json.dumps({"type": "update", "metrics": new, "status": st, "t": time.time()}))
            await asyncio.sleep(1.0)
    except (WebSocketDisconnect, RuntimeError):
        pass


@app.websocket("/ws/inference")
async def ws_inference(ws: WebSocket):
    """Compact inference/evaluation feed (no frames): state payload + episode events."""
    await ws.accept()
    q = sim.subscribe()
    try:
        while True:
            try:
                msg = await asyncio.to_thread(q.get, True, 1.0)
            except queue.Empty:
                continue
            if msg.get("type") == "frame":
                await ws.send_text(json.dumps({"type": "state", "t": msg["t"], "state": msg["state"]}))
            else:
                await ws.send_text(json.dumps(msg))
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        sim.unsubscribe(q)


# ----------------------------------------------------------------------------- static fallback UI
STATIC_DIR = Path(__file__).parent / "static"
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"


@app.get("/", response_class=HTMLResponse)
async def index():
    if (FRONTEND_DIST / "index.html").exists():
        return FileResponse(FRONTEND_DIST / "index.html")
    return FileResponse(STATIC_DIR / "index.html")


if (FRONTEND_DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
