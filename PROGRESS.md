# PROGRESS

Living checkpoint of the autonomous build. Newest entries at the bottom of each phase.

## Phase 1 — System inspection  ✅
- Python 3.12.3, Node 22.23.2, npm 10.9.8, uv 0.9.26. 16 cores, 30 GB RAM.
- **No CUDA** (AMD Radeon 680M iGPU, Mesa). Training and inference run on CPU; dashboard shows `DEVICE: CPU`.
- Headless rendering works via `MUJOCO_GL=egl` (Mesa EGL on the AMD iGPU).
- Installed in `.venv`: mujoco 3.14.0, torch 2.11.0+cpu, gymnasium 1.3.0, stable-baselines3 2.9.0, fastapi 0.141, lerobot 0.6.1 (+ `[dataset]` extra).

## Phase 2 — Repository structure  ✅
Created per Section 21 (frontend/, backend/, simulation/{mujoco,assets,environments,robots,objects}, training/{policies,rewards,callbacks}, checkpoints/, lerobot_adapter/, datasets/, configs/, scripts/, tests/, logs/).

## Phase 3 — MuJoCo robot model validated in isolation  ✅
- Robot: **official MuJoCo Menagerie UR10e** (user confirmed mid-run: "its UR10e"). Kinematics, masses, inertias, actuator gains unmodified. Added gravity compensation (`gravcomp=1`) on the links so the position controller has zero steady-state sag.
- Gripper: custom parallel-jaw (two prismatic fingers coupled by an equality constraint, 140 mm stroke, high-friction pads, 60 N clamp). See ASSUMPTIONS.md.
- `scripts/validate_grasp.py`: loads cleanly, 0 warnings after 1 s of gravity, joint targets tracked, gripper opens/closes, **scripted contact-only pick-and-place lands 22/24 parts in the correct bin** across 12 randomized layouts.
- `scripts/stress_physics.py`: 24 episodes × 150 biased random actions → **0 unstable episodes** (after fixing: bolt head cylinder→box/capsule primitives, explicit object inertias, softer pad contacts).

## Phase 4 — Scene: bins, parts, camera, dressing  ✅
- Worktable, two blue bins (Bin A = bracket, Bin B = bolt), L-bracket 120×60×12 mm, M20×100 bolt, overhead camera on a boom (the learning camera), 3 viewport cameras, warehouse racks/pallets/columns/safety markings/lights.

## Phase 5 — Gymnasium env  ✅
- `simulation/environments/sorting_env.py` → `VolvoSorting-v0`; passes `gymnasium.utils.env_checker`.
- Obs: `observation.images.overhead` (64×64×3 uint8, real MuJoCo camera) + `observation.state` (44-d). Action: 7-d Cartesian delta + gripper.
- Throughput: ~370 env steps/s single process with image rendering (600 without).

## Phase 6 — Reward  ✅  `training/rewards/sorting_reward.py`, weights in `configs/environment.yaml`.
## Phase 7 — Domain randomization  ✅  `simulation/environments/randomization.py` (single configurable system, `level` scalar).

## Phase 8 — Walking skeleton  ✅
- FastAPI backend on :8000 (`scripts/run_backend.sh`), sim thread, `/ws/simulation` streams JPEG frames + compact state at ~10-15 fps; fallback HTML page at `/`.
- Verified: `/health`, `/simulation/randomize`, random-policy mode, WebSocket frames received by a client.

## Phase 9 — RL training loop  ✅
- `training/train.py` (SB3 PPO, MultiInputPolicy CNN+MLP, SubprocVecEnv×12, VecNormalize on state), callbacks in
  `training/callbacks/trainer_callbacks.py` (file-based pause/resume/stop/save/evaluate, status + JSONL metrics, atomic
  checkpoints, periodic eval, SIGTERM-safe). Smoke test (`--smoke`) passes; also covered by `tests/test_training_and_checkpoints.py`.
- Throughput: ~600–680 env steps/s end-to-end with 12 envs incl. 64×64 rendering and PPO updates on CPU.

## Phase 10 — LeRobot interface + dataset adapter  ✅
- Env emits `observation.images.overhead` / `observation.state`; `lerobot_adapter/dataset_writer.py` writes real LeRobotDataset v3.0
  (verified: written, reloaded, items have the expected keys/shapes). `lerobot_adapter/policy.py` exposes `select_action(batch)`.

## Phase 11 — Checkpointing  ✅  `training/checkpointing.py` (atomic dir rename, metadata, latest.json, list/load).
## Phase 12 — Inference mode  ✅  backend `SimulationService` inference/evaluate modes; `POST /inference/load|start|stop`; verified with the smoke checkpoint (1.4 ms action latency).
## Phase 13 — React/TS dashboard  ✅  Vite 8 + React 19 + Tailwind 4 + Recharts 3 on http://localhost:3001 (port 3000 was occupied by another app on this host). Tabs: Training, Inference, Policy, Checkpoints, Evaluation; viewport overlays; policy-loop panel; training config form. Palette validated with the dataviz validator (dark surface).
## Phase 14 — Live WebSocket wiring  ✅  `/ws/simulation` frames+state, `/ws/training` status+metrics tail (verified in headless Chrome: live frames, live metrics, charts updating).
## Phase 15 — Tests  ✅  26 pytest tests pass (env init/reset/randomization/action validation/joint+velocity limits/workspace/gripper/bin/grasp/placement/reward/timeout, checkpoint save+load, policy inference, LeRobot dataset, API health/endpoints/WebSockets, real training smoke run).

## Phase 16 — Long training run  ⏳
- **run1** (original reward): 384k steps, reward ↑ to ~60 but the policy learned to hover ~10 cm from the part at table height (grasp contacts 30 %, lift 0 %). Diagnosed with `scripts/diagnose_policy.py`. Stopped gracefully (checkpoint saved in 1.3 s); archived in `checkpoints/archive_run1`.
- **run2** (pre-grasp waypoint approach reward + push penalty): 100k steps, reward ↑ to ~120 (hovering above the part), grasp contacts fell to 12 % → local optimum. Archived in `checkpoints/archive_run2`.
- **run3** (run2 reward + initial-state curriculum: 25 % pre-grasp / 25 % grasped training resets; eval/inference reset from home): 200k steps, lift rate 20 % (curriculum starts) but placement 0 %, reward ↑ 168 — **root cause found**: dense per-step near/hold/lift/transport rewards accumulate over the 200-step horizon (≈7/step × 150 steps ≫ 80 placement+completion bonus), so holding the part in the air beat placing it. Archived in `checkpoints/archive_run3`.
- **run4** (potential-based shaping — only progress pays; sparse event bonuses; release penalty; curriculum kept): started 2026-09-26 00:12, 3M steps, 12 envs. Scripted reference pick earns ≈ +69 per part, hovering ≈ 0.

- run4 @ 325k: lift 23 %, placement 9 % (recent training episodes), reward trending up; from a hover the policy still drifts instead of descending. Curriculum tightened (pre-grasp starts 3–12 cm above the part, 35 %) and **resumed from checkpoint_000350040** through the API (`resume_from: latest`) — resume path verified.
- run4 @ 650k: plateau (grasp ≈ curriculum share, eval-from-home grasp 0 %). Probes: forcing descend+close from a hover grasps 12/12, the policy alone 0/12 → the *sequence* is never explored. Continuous gripper command was a big factor (holding the 20 mm bolt needs a[6] > 0.7 for consecutive steps; pad contact flickered, no grasp streak). **Switched to a binary gripper command** (`action.gripper_mode: binary`), push penalty −0.5 → −0.2, resumed weights as **run5** at 708k steps (2026-09-26 00:31).
- run5 @ 1.0M: placement from grasped starts ↑ to ~14 %, but grasp rate still ≈ curriculum share. Random-exploration probe: grasp within 30 steps from home 0/30, from hover 1/30 (xy drift), from a *straddle* (fingers around the part, open) 11/30. **Added straddle starts (25 %) to the curriculum** (reverse curriculum), resumed as **run6** at 1.03M steps (2026-09-26 00:42).
- run6 @ 1.3M: straddle starts did not convert either — the resumed policy commands "close" but simultaneously drifts up/sideways (a "rise away" prior baked in by runs 1–5); gripper timing ruled out (closes in <150 ms, grasps 8/8 even while rising slowly). Runs 4–6 archived in `checkpoints/archive_run4_6` (`logs/run4_6_metrics.jsonl`).
- **run7** = fresh policy from step 0 with the full stack: potential-based reward, binary gripper, curriculum (home 35 % / pre-grasp 20 % / straddle 25 % / grasped 20 %), grasp bonus 10, hold term 0.2/step. Started 2026-09-26 01:02.
- run7 @ 270k (01:12): **first real approach+grasp learning** — grasp 53–70 %, lift 42–55 % of recent training episodes; clean eval from home: grasp 40 % (10 ep). Placement 1–7 %. Reward crossed 0.
- run7 @ 547k (01:22): grasp 64–75 %, lift 54–60 %, placement 8–10 % (training eps); clean eval grasp 40 %, placement 0 %. Hold term (0.2/step) was making "hold" ≈ "place"; releasing over the bin never explored. Hold term → 0.05, added **over-bin starts (15 %)**; resumed as **run8** at 584k (01:24).
- run8 @ 843k (01:34): placement 24–28 % of training episodes (↑ from 13 %), first full two-part successes (1 %); clean eval: grasp 20–30 %, placement 10–20 % at 630–700k (10-episode evals, noisy). Training continues in the background.
- Added live dataset recording (`/dataset/record/*`, ● Rec button) and demo scenario presets (`/simulation/scenario/*`).

- run8 @ 1.1M (01:45): training episodes grasp ~70 %, lift ~63 %, placement ~25 %, full two-part success ~1 %. Clean 20-episode live evaluation of checkpoint_001115616 (seed 12345, full randomization): grasp 45 %, placement 15 %, full success 0 %, 0 collisions / drops / wrong-bin. Training left running toward 3M steps.

## Phase 17 — Acceptance test  ✅
- `scripts/acceptance_test.py`: 20/20 against the live system (2026-09-26 01:50). Full pytest suite: 26 passed.
- Section 31 items 1–20 verified; item 14/18 use the real run8 checkpoint. The policy's task success is still low (see run8 numbers) — reported honestly on the dashboard; the RL pipeline, checkpoints, inference, evaluation and telemetry are all real.

## Phase 18 — Application left running  ✅
- Backend :8000 (uvicorn), dashboard :3001 (vite preview), trainer PID in `logs/training_status.json` (run8, resumable with `--resume latest`). Live sim is in inference mode on the latest checkpoint.

---

# Incremental update — 2026-09-26 (session 2)

## Phase 19 — Naming audit (UR20 → UR10e)  ✅
- Grep of configs, MJCF, backend, frontend (incl. built `dist/`), datasets, tests and docs: the only "UR20" left was the
  historical note in `ASSUMPTIONS.md` #1, now reworded. `simulation/robots/ur10e.xml` is confirmed to be the Menagerie
  UR10e (link `pos`/`mass`/`diaginertia` diff against `third_party/mujoco_menagerie/universal_robots_ur10e/ur10e.xml`
  is empty apart from the base placement and the appended gripper). Datasets record `robot_type: UR10e`.

## Phase 20 — Placement-failure diagnosis  ✅  (full write-up: `DEBUGGING.md`)
- New tools: `scripts/diagnose_placement.py` (per-step after-grasp traces + frame strips in `logs/diagnosis/`),
  `scripts/probe_curriculum_policy.py` (detector check, curriculum-start probes, reward-pump measurement).
- Finding: detection was correct; the policy grasped, lifted for 1–3 steps and **opened the gripper** (12/12 grasped
  episodes), because (a) the lift/approach potentials were reset — not charged back — on grasp loss, so a
  grasp→lift→drop→re-grasp loop earned ≈ +4 per cycle, and (b) 15 % over-bin curriculum starts rewarded "lifted ⇒ open".
  From "grasped" curriculum starts the 3 M-step policy opened within 2–4 steps in 11/11 probes; from "over-bin" starts it
  placed 11/11.
- Fix: state-only potentials active during one phase per target and charged back on regress; alignment folded into the
  approach potential; release penalty only on commanded opens away from the bin; curriculum 10 % over-bin / 25 % grasped.
  Pump measurement after the fix: each extra cycle now nets ≈ −2.3. `tests/test_reward_shaping.py` guards this.
- `diag_baseline_fixedreward` (fresh PPO, fixed reward, 300 k steps, 12 envs, seed 42): training episodes grasp 54 % /
  lift 45 % / placement 10 % / success 0 %; 30-episode clean eval: grasp 7 %, placement 0 %. The bug is fixed; learning
  from scratch at this budget is still too slow → bootstrap (Phase 22).

## Phase 21 — Camera visibility  ✅
- `scene.xml`: visual-only `overhead_camera_unit` body (housing, lens barrel/ring/glass, LED, label, bracket) at the
  camera pose, all faces above the optical centre; FOV overlay geoms in group 5 (edges to the table-footprint corners +
  footprint outline). Overhead policy image verified pixel-identical before/after (and group 5 is forced off for it).
- The `workcell` viewport camera was re-framed (pos 1.70 −1.60 1.80, fovy 55°) — the unit was above the old frame's
  top edge, which is why "the camera was not visible".
- Backend: `GET /simulation/camera` (pose/tilt/fov/footprint from the compiled model), `POST /simulation/frustum`.
  Dashboard: readout next to the OVERHEAD CAMERA thumbnail ("looking straight down · tilt 0.0°"), FOV toggle button.
  Screenshot: `logs/dashboard_camera_fix.png`.

## Phase 22 — Teleop panel + imitation bootstrap  ✅ (pipeline), results below
- Backend: `teleop` and `scripted_demo` sim modes, `/teleop/*`, `/ws/teleop` (20 Hz, 0.5 s dead-man), per-episode
  `DemoRecorder` (`lerobot_adapter/demo_recorder.py`: one LeRobotDataset dir + `demo.json` sidecar per episode,
  `episode_type` tag), `/demos` list / `DELETE /demos/{name}` discard / `POST /demos/scripted`.
  `training/demos/scripted_controller.py` = ground-truth state machine through `env.step()`.
- Frontend: **Teleop** tab (keys, mouse drag, live axes, ● Recording, New episode, scripted-demo generator,
  Demonstrations table with Discard); "Initialize from demonstrations" checkbox + BC epochs in the Train dialog;
  bootstrap line + `bc/aux_action_mse` chart in the Training tab.
- BC: `training/bc_pretrain.py` — BC on the PPO network itself (direct weight transfer) + value-head regression +
  VecNormalize seeding, then interleaved BC aux steps during PPO (`TrainerControlCallback._bc_aux_update`).
- Validation: 30 scripted demos generated (29 successful, 2,238 frames) via `scripts/record_scripted_demos.py`;
  live API generation of 2 more (both successful); live teleop over `/ws/teleop` moved the TCP 0.40 m in 21 recorded
  steps (synthetic test episode discarded afterwards — it is not a human demo). BC on 29 demos: holdout action MSE
  0.233 → 0.042, gripper-sign accuracy 28 % → 96 % (8.9 s on CPU).
- Tests: `tests/test_teleop_demos_bc.py` (camera readout/frustum, teleop moves arm + records + discard, refusal while
  inference runs, scripted controller, slow BC end-to-end smoke). Full suite: 34 tests pass. Acceptance: 20/20.

### Bootstrap vs. baseline — measured (scripts/compare_checkpoints.py, 30 episodes, seeds 777–806, deterministic, clean env, full randomization; `logs/checkpoint_comparison.json`)

| checkpoint | grasp | lift | ≥1 placed | both placed | mean reward |
|---|---|---|---|---|---|
| old 3 M run (pre-fix reward), `archive_run9_fresh3M/checkpoint_003001344` | 50 % | 40 % | 0 % | 0 % | −8.5 |
| baseline: fixed reward, PPO from scratch, 300 k | 7 % | 0 % | 0 % | 0 % | −10.2 |
| **BC init only** (29 scripted demos, 0 PPO steps) | **90 %** | 23 % | **13 %** | **7 %** | **+7.8** |
| bootstrap: BC init → PPO 300 k (aux BC decaying to 0 over 150 k, lr 3e-4) | 23 % | 7 % | 3 % | 0 % | −6.8 |
| bootstrap @ 250 k | 30 % | 17 % | 3 % | 0 % | −5.2 |

Reading, honestly: the bootstrap **does** beat the un-bootstrapped run at equal budget (grasp 23 % vs 7 %; placement
3 % vs 0 % is within noise at n=30), and its training-episode metrics were far higher (grasp 77 % / lift 68 % /
placement 33 % / both 2 % vs 54 / 45 / 10 / 0). But the plain BC initialisation evaluates **better than after PPO
fine-tuning** — PPO's early updates (Gaussian exploration std 0.6, decaying BC term, curriculum start states not present
in the demos) erode the demonstrated behaviour. A second variant (`ppo_bc_bootstrap_v2_300k`: constant BC aux term,
PPO lr 1e-4, log_std_init −1.0) was started to test that; its result is appended below.

**Variant 2 result** (`ppo_bc_bootstrap_v2_300k`: constant BC aux term, PPO lr 1e-4, log_std_init −1.0; same 29 demos, seed 42,
300 k steps; `logs/checkpoint_comparison_v2.json`, same 30 seeds): BC init alone grasp 83 % / placement 10 % / both 0 %;
after PPO 250 k grasp 7 % / placement 0 %; after 300 k grasp 3 % / placement 0 % — while its *training-episode* placement
rate reached 42 % (curriculum starts). So the constant BC term did not prevent the clean-eval collapse either; the
policy specialises on the 70 % curriculum start states and loses the from-home approach the demos taught. Not pursued
further in this session.

**Hand-over state.** `checkpoints/` holds the v1 bootstrapped run (`ppo_bc_bootstrap_300k`, latest
`checkpoint_000301056`, loaded in the live sim, inference running); v2 and the older runs are under `checkpoints/archive_*`.
Bottom line for the user: (1) the placement bug is fixed and guarded by tests; (2) the demo → BC → PPO pipeline works end
to end and is ready for real teleop demonstrations; (3) at a 300 k-step budget the bootstrap improves PPO over training
from scratch (grasp 23 % vs 7 %) but PPO fine-tuning still degrades the BC policy (90 % → 23 % grasp), so the next lever is
either many more PPO steps with the demos in the loop, or human demos that cover the curriculum's start states, or
evaluating/fine-tuning with curriculum resets disabled.

## Phase 23 — Incremental Update 2: fix the BC→PPO regression  ✅  (diagnosis: `DEBUGGING.md` Part 2)
- Episode-level diagnosis (`scripts/diagnose_bc_regression.py`, `logs/diagnosis_bc_regression.json`): the regressing run
  never opened the gripper early any more; it progressively lost the **approach from home** (no-approach in 4/10 episodes at
  100 k → 6–7/10 at 250–301 k) while its action drift from the demos grew 0.03 → 0.06 → 0.19 → 0.22 → 0.27. Its rising
  training reward (+12 mean) came from curriculum start states (72 % of episodes, +11…+37 each); from-home training
  episodes averaged −8.7. No shaping term was being farmed → catastrophic forgetting + start-state mismatch, not reward.
- Fixes: in-loss BC anchor in every PPO minibatch (`PPOWithBC`, coef 1.0 → floor 0.25 over 1 M, never 0); `target_kl 0.02`;
  LR warm-up 0.1× with evaluation-gated ×2/×0.5 ramp; VecNormalize frozen after demo seeding; curriculum ×0.3 when
  bootstrapping; `log_std_init −1.0`; evaluate-then-save so each checkpoint carries its own 20-episode eval (BC init
  included); **best-by-eval** selection (★ in the Checkpoints tab, `Load best`, default for Run Inference and start-up).
  Reward shaping audited and left unchanged. Tests: `tests/test_update2_anchor_and_selection.py`; full suite 39/39.
- Run `ppo_bc_anchored_400k` (same 31 scripted demos, seed 42, 12 envs, 402 k steps, 34 min on CPU). Trainer evals (20 clean
  episodes each): BC init 0 % / grasp 70 % / place 5 % → 50 k: 10 % → 100 k: 15 % → 150 k: 10 % (LR halved) → 200 k: 20 % →
  300 k: 45 % → 350 k: 60 % → **400 k: success 70 %, grasp 100 %, placement 90 %, reward +144**. Drift vs demos stayed
  0.03–0.04 throughout (was 0.27). `best` = `checkpoint_000400032`.

### Before / after — Evaluation tab protocol (live sim, 20 episodes, randomization level 1.0, seed 4242)

| checkpoint | success (both parts) | grasp | ≥1 placed | avg reward | drops / wrong bin |
|---|---|---|---|---|---|
| before: `ppo_bc_bootstrap_300k` @ 275 k (what the UI had loaded) | **0 / 20 = 0 %** | 25 % | 0 % | −8.3 | 0 / 0 |
| before: `ppo_bc_bootstrap_300k` @ 301 k (latest) | **0 / 20 = 0 %** | 20 % | 5 % | −5.9 | 0 / 0 |
| after: `ppo_bc_anchored_400k` @ 400 k (★ best) | **12 / 20 = 60 %** | 100 % | 75 % | +127.0 | 1 / 0 |

Offline cross-check, 30 fixed seeds (777–806), deterministic (`logs/checkpoint_comparison_update2.json`): old bootstrap
best (250 k) success 0 % / placement 3 %; anchored BC init 0 % / 10 %; anchored 200 k 43 % / 80 %; anchored best 400 k
**60 % / 87 %**; anchored final 402 k 70 % / 83 %. Files: `logs/evaluation_before_update2*.json`, `logs/evaluation_after_update2_best.json`.

Hand-over: live sim in inference on `checkpoint_000400032` (★ best, also what start-up loads); older runs archived under
`checkpoints/archive_*`; Teleop tab ready for human demos (they join the BC set on the next bootstrapped run).

---

# Consolidated next phase — 2026-09-26 (session 3)

## Phase 1 — Teleop UX: replay + click-to-pick  ✅
- **Replay**: `POST /demos/{name}/replay`, `/replay/play|pause|seek|stop`; sim mode `replay` with a distinct
  "REPLAY · OPEN-LOOP" viewport indicator, scrub bar, ±1 step, speed buttons apply. Every recorded episode now stores an
  exact initial snapshot (`initial_state`: qpos/qvel/ctrl + controller targets + seed + episode config) in `demo.json`;
  older episodes fall back to seed + fixed poses. Fidelity is measured live (max deviation of the replayed continuous
  state vs the recorded one; exact-snapshot replay of an assisted episode: reward deviation 0.006, outcome reproduced).
- **Click-to-pick**: `POST /teleop/pick_at {u,v}` runs a MuJoCo segmentation render of the *viewport camera at the
  viewport resolution* (960×720), maps the clicked pixel to a geom → body (7-px neighbourhood search for thin parts),
  highlights the selection in the viewport, and hands the object to the **same** `ScriptedSortingController` used for
  scripted demos (`fixed_target` / `bin_override`), executing approach→grasp→lift→transport→place within the policy's
  limits; clicking a bin re-routes; Cancel/Esc aborts; manual jogging resumes afterwards. `POST /teleop/pick {object}` is
  the by-name path used by the chat panel. Recording an assisted pick retags the episode `assisted_demo` (distinct from
  `teleop_demo` and `scripted_demo`); the Demonstrations table and the training bootstrap line break counts down by type.
  Verified live: bracket click → placed in bin_a in 38 steps; bolt by name → bin_b in 44 steps.

## Phase 2 — Generalization proof  ✅
- Batch evaluation runner on the live sim (`cmd_start_batch`): scorecard / sweep / compare chain Evaluation runs and
  persist `logs/<kind>_latest.json` (+ timestamped). Evaluations now accept fixed scenario poses and a label.
- **Scorecard** (`POST/GET /evaluation/scorecard`), first real run (checkpoint_000400032, 5 episodes/scenario, 33 s):
  nominal 100 %, bracket 20° 80 %, bracket 75° 80 %, bolt moved 60 %, both moved 80 % success; grasp 100 % everywhere;
  0 collisions except 4 collision steps in "both moved". Table persists in the Evaluation tab.
- **Difficulty sweep** (`/evaluation/sweep`), 6 episodes/level: success 0 % @0.0 (fixed centre layout: one part placed
  every time, the second never within 200 steps), 33 % @0.25, 67 % @0.5, 33 % @0.75, 50 % @1.0 — non-monotonic, shown as is.
- **Failure reel**: every failed evaluation episode (+ one success reference per evaluation) is saved to
  `logs/eval_reels/<eval>/<ep>/` with a 12-frame filmstrip, first/middle/last frames (+ overhead), `trace.json`
  (per-step action/reward/TCP/objects/grasp) and `meta.json` with a derived failure reason; viewer in the Evaluation tab.
- Also landed here (used by Phases 5/6): per-episode `sim_seconds`, `control_effort`, `collision_steps`, `false_pick`;
  `parts_per_minute` and collision/effort aggregates in evaluation summaries and inference stats.

## Phase 3 — Parts catalog & scene builder  ✅
- `configs/parts_catalog.yaml` + a **parts pool** in `scene.xml`: bracket variants (L default, flat 120×60×8, U 100×50×30),
  fastener variants (hex M20×100 default, socket-head M16×80, hex M12×60), clutter copies (extra L-bracket, extra bolt)
  and the **rejection case** (red plastic cap). All are real free bodies with contact physics, parked at (−6, −6…) when
  not on the table — no model recompilation at demo time. Drop-tested: every part rests stably (measured rest heights
  written back to the catalog); parked bodies stay still (|qvel| < 1e-8).
- `WorkcellSim`: logical task objects (bracket, bolt) are bound to an *active body*; `set_variant` swaps geometry and parks
  the old body; `place_body` / `park_body`; `pads_grasping(body)` for the false-pick metric; the randomizer rebinds its
  nominal-geometry tables per active body. Env / reward / controller / curriculum are unchanged (they use logical names).
- `backend/scene_builder.py` + `/scene/*` API: catalog with thumbnails rendered at start-up from the MJCF bodies in a
  photo booth (`catalog_cam`), `spawn` by viewport pixel (camera ray ∩ table plane — same projection as click-to-pick) or
  by x/y, remove, clear, default variants, save/delete custom scenarios (`configs/custom_scenarios.json`). Clutter is
  re-applied after every reset; task objects can only be swapped, not removed (documented in the tab).
- **Scene** tab (catalog drag source, table contents, save-as-scenario) + viewport drop target; custom scenarios appear as
  ★ buttons in the scenario bar and are included in the scorecard.
- Verified live: variant swap → the same scripted controller picks the U-bracket (placed in 4 s); custom scenario
  `cluttered_u_bracket` saved and reloaded in 0.06 s; **preset scenarios load in 0.03–0.07 s** (unchanged path); an
  evaluation with the red cap on the table reports `false_picks: 0` (metric wired; the cap sat in a far corner).

## Phase 4 — Chat command panel  ✅
- `backend/chat.py`: fixed command registry (32 commands: robot / scene / training / inference / evaluation / camera /
  replay / scene-builder) with a deterministic Tier 1 regex parser; every handler calls the same backend action as
  the button. Unrecognised input answers "I didn't recognize that command" (nothing runs). Destructive commands
  (stop training, load checkpoint during inference, discard demo, clear table) return a confirmation prompt and execute
  only after a typed "yes"; a new command drops a stale pending confirmation. Buttons for the same actions now ask too.
- Tier 2: credential detection at start-up (`ANTHROPIC_API_KEY`, or Foundry key + resource); Claude (Anthropic SDK 1.8,
  function calling over tools generated from the registry, `strict` schemas, low effort) is used only for phrases Tier 1
  rejects; only a returned tool call executes; any failure → Tier 1 answer. A background probe checks that the endpoint
  actually serves a model and the panel reports the real state. On this machine a Foundry credential is present but
  **no Claude deployment is reachable** (`DeploymentNotFound` for every standard model id), so the panel truthfully shows
  Tier 2 off with that reason; setting `CHAT_LLM_MODEL` to the deployment name enables it.
- Floating collapsible panel on every tab (`ChatPanel.tsx`): typed text → parsed command + args + tier → real result;
  "what can I say?" lists every phrase. Verified phrases: randomize the scene · set randomization to 0.5 · switch to
  bracket 20° scenario · add the red cap · clear the table (+yes) · pick the bracket · run evaluation for 5 episodes ·
  run the generalization scorecard · toggle FOV overlay · replay the last recorded episode · load checkpoint latest ·
  discard demo 99 (validated) · stop training · what can I say?

## Phase 5 — Executive packaging  ✅
- Cycle time (parts sorted per simulated minute, from real episode lengths) in Inference and Evaluation; per-scenario
  parts/min in the scorecard.
- **Export Summary** (header button → `GET /export/summary.html`, also saved as `logs/summary_<ts>.html`): static
  self-contained HTML with the active checkpoint, last live evaluation KPIs, scorecard table, sweep chart (inline SVG) and
  table, before/after table, demo counts by type, live frames (workcell, overhead policy input, success/failure
  filmstrips) and safety figures. N/A wherever nothing has been measured.
- **Before/after**: `POST /evaluation/compare` re-evaluates two checkpoints under identical seeds/level/episodes on the
  live sim and restores the previously loaded checkpoint; side-by-side table in the Evaluation tab.

## Phase 6 — Safety / credibility  ✅
- Collision steps, episodes-with-collision, collision-penalty total and control effort (Σ|a|² per episode) are shown
  directly in Inference and Evaluation ("0 collisions across N evaluation episodes" line).
- Header `SAFETY` badge (envelope OK / violation / E-STOP) from live simulator figures (TCP-to-workspace margin,
  joint-limit margin, velocity utilisation vs UR10e limits, current collision) + **■ E-STOP** button: simulated e-stop
  that halts stepping after the current control step, freezes joint targets, and blocks every mode start until reset.
  README states exactly what it does and does not do (no sub-step interruption, no brake model, not certified).

## Optional — wrist camera  ✅ (display only)
- `gripper_cam` from the UR10e MJCF streams as a WRIST CAM thumbnail when toggled (`POST /simulation/wrist_camera`).
  It is **not** fed into the policy observation (the policy still sees overhead RGB + state).

### Before/after result (fresh, identical seeds 4242+i, level 1.0, 10 episodes each, live sim)
| checkpoint | success | grasp | ≥1 placed | avg reward | parts/min | collision steps |
|---|---|---|---|---|---|---|
| pre-fix bootstrap `archive_ppo_bc_bootstrap_300k_…/checkpoint_000250080` | 0 % | 10 % | 10 % | −4.3 | 0.30 | 0 |
| current ★ best `checkpoint_000400032` | 50 % | 90 % | 80 % | +117.8 | 6.26 | 0 |

### Executive-demo readiness checklist (measured 2026-09-26 16:30)
1. Presets load instantly with zero authoring: nominal 0.03 s, bracket 20° 0.06 s, bracket 75° 0.04 s, bolt moved 0.04 s, both moved 0.07 s; custom `cluttered_u_bracket` 0.06 s. ✅
2. Scorecard (5 ep/scenario, checkpoint_000400032): nominal 100 %, bracket 20° 80 %, bracket 75° 80 %, bolt moved 60 %, both moved 80 %. ✅
3. Difficulty sweep (6 ep/level): 0 % / 33 % / 67 % / 33 % / 50 % success at levels 0 / .25 / .5 / .75 / 1 — a real, non-monotonic run. ✅
4. Click-to-pick (bracket by pixel → placed in 38 steps; bolt by name → 44 steps; U-bracket variant → 4 s) and Replay (exact snapshot, reward deviation 0.006; scrub/pause/stop) verified live. ✅
5. Catalog places all variants (flat/U bracket, socket/M12 bolt), clutter and the red rejection cap; `cluttered_u_bracket` saved and reloaded. ✅
6. Chat: "randomize the scene", "pick the bracket", "run evaluation for 5 episodes", "run the generalization scorecard" all parsed and executed (Tier 1); panel states Tier 2 is **off** and why (Foundry credential present, no reachable Claude deployment). ✅
7. Export Summary: `logs/summary_20260926_162950.html`, 446 KB, scorecard + sweep SVG + compare table + 4 embedded frames, one "N/A" (live-session field). ✅
8. No fake values: every panel renders N/A when a figure is missing; scripted demos are labelled and excluded from policy metrics. ✅
Tests: 39/39 pytest. App left running: backend :8000, dashboard :3001, inference on ★ checkpoint_000400032.

## Incremental Update 6 — layout, rebrand, Tier 2 fix, physics fidelity  ✅
- **Three-column layout**: left = persistent, collapsible Scene / Parts Catalog (drag source next to its drop target);
  middle = viewport (now rendered 1120×720, 14:9, so it fills the column without side gaps) + policy loop + controls;
  right = tabs reordered **Teleop → Training → Policy → Checkpoints → Inference → Evaluation → Chat**. The chat panel is a
  full tab (same engine, tiers and confirmations; no floating popup).
- **Rebrand**: header title "ADAPTIVE SORTING" (env `DEMO_TITLE` still overrides); "Volvo" removed from the README title
  and intro, page title, backend app title/docstring, fallback page, export summary and acceptance test. Code identifiers
  (`VolvoSorting-v0`, `volvo_sorting_workcell`, dataset repo ids, the project directory) are unchanged.
- **Tier 2 LLM**: the earlier probe only tried first-party model ids. The chat engine now queries the Foundry resource's
  deployment list with the same credential; resource `<foundry-resource>` serves `claude-opus-5-5` and `claude-fable-5-1`
  (plus `model-router`, `gpt-6-astra`). Tier 2 verifies against `claude-opus-5-5` and answers free-form commands
  end-to-end (tool calling over the fixed command set; `strict` dropped because the API caps strict tools per request).
  The panel states that Tier 2 is active and that free-form phrasings make a live network call to that deployment.
- **Bins**: randomization moves bins ONLY at episode reset (`randomizer.sample()` is called from `sim.reset()` alone;
  verified by test), by up to ±4 cm × level. Scenario presets (fixed object poses) now pin the bins at nominal
  (`fixed_bins`, `episode_config.bins_fixed`), `randomization.randomize_bins: false` disables it globally, the viewport
  shows "BINS · FIXED / RANDOMIZED AT RESET ±x cm" and flashes "● NEW EPISODE · SCENE RESET #n · bins …" for ~2 s at every
  episode boundary, so a bin change reads as a reset, not a glitch.
- **Collision audit**: every new part has box/capsule collision geoms matching its visual (visual-only geoms are the
  decorative holes / hex facets / recess / cap top). `tests/test_physics_fidelity.py` (3 tests, **pass**): each variant,
  clutter part and the rejection cap rests on the table, is grasped by both pads, lifts by friction and lands inside a
  bin without clipping; 6 random-action episodes with clutter: no NaN, |qvel| < 50, no table penetration, bins immobile
  within episodes; scenario resets keep bins at nominal. Full suite: 42/42.


## UI cleanup pass  ✅
- Viewport overlays reduced to mode, target, grasp state, episode/step/placed, recording, reset flash, policy-camera
  thumbnail and two camera toggles (FOV overlay, wrist cam). Removed: camera pose/tilt/fov readout, "CAMERA · WORKCELL",
  robot/device/speed tag, reward in the overlay, persistent BINS label (the reset flash still says "bins moved").
- Control bar: Reset, Randomize, Variation slider, ▶ Run Policy, ■ Stop, Pause, Step, Speed, Train/Stop Train, Scenarios.
  Removed: Random-actions, Checkpoints shortcut, ● Rec (recording lives in Teleop), FOV (moved onto the viewport).
- Policy loop panel: camera → policy → five meaningful action channels (X, Y, Z, turn, grip open/close); no EE numbers.
- Header: title, subtitle, SIM/TRAIN/device badges, Export Summary. Removed: SAFETY badge, E-STOP, duplicate tagline.
- Right column tabs: Teleop → Training → Policy → Checkpoints → Inference → Evaluation. **Chat tab and backend chat engine
  removed** (Tier 1/Tier 2, `backend/chat.py`, `/chat*`). Inference tab: safety-envelope card removed.
- **Parts catalog**: every part is an independent toggle (click = place/remove, drag = place at a spot). A slot (bracket /
  bolt) is bound to its first selected variant or is *absent*; extra variants of a type are untracked parts. The env
  skips absent objects (no target, no drop check, zero pose in the state vector, success = all *present* placed).
  **Clear table** now parks everything incl. the two task objects (the previous version only removed extras - the bug);
  **Defaults** and every preset scenario restore bracket + bolt. Save-as-Scenario button removed (API kept).
- Verified live: bracket off → no bracket; both off → empty table; policy on an empty table idles (no target); flat
  bracket on → tracked; default bracket added → extra; flat off → default takes over; clear → empty; Nominal → defaults.

## Incremental Update 7 — cluttered-scene instability & Defaults scoping  ✅  (diagnosis: `DEBUGGING.md` part 3)
- **Both suspected problems were real.** (1) Scope: "Defaults" and the preset scenarios kept extra/rejection parts on
  the table; the checkpoint was trained on clean scenes only. (2) Physics: the bracket diverged when pinched between a
  finger pad and an 8 mm bin wall at release - 4/60 clean policy episodes at the 2 ms step, 6 % of live clean episodes,
  and 74 % of the reported live run once a clutter part kept being re-spawned inside a bin wall. Random actions and the
  scripted controller never destabilised (0/48), so the trigger is the policy's release behaviour, not the parts' meshes
  (all collision geoms were re-audited and match their visuals).
- Fixes: physics timestep 2 → 1 ms; light viscous damping on every free part (`physics.object_damping`); soft priority-1
  contacts on the three extra/rejection parts; scene builder keeps re-applied parts ≥ 11 cm from task objects and 7 cm
  outside bin footprints; **Defaults / presets = bracket + bolt + bins only**; clutter moved to the saved
  `clutter_showcase` scenario (nominal poses + Extra L-bracket + Red cap); `unstable_episodes` reported in every
  evaluation summary and as a scorecard column; optional training-time clutter (`randomization.clutter.prob`, off).
- Offline validation (policy, deterministic): 0 unstable in 60 + 60 clean episodes (seeds 20000/40000) and 0 in 30
  adversarial-clutter episodes; success 60-70 %. Full test suite 46/46 (4 new physics tests).
- **Live validation (checkpoint_000400032, level 1.0):**
  scorecard 5 ep/scenario - nominal 100 %, bracket 20° 100 %, bracket 75° 80 %, bolt moved 80 %, both moved 100 %,
  **clutter_showcase 100 %**, cluttered_u_bracket 0 % (U-bracket variant + clutter: both parts reached the bins in every
  episode but never both; 433 collision steps) - **0 unstable episodes in all 35**;
  **clutter_showcase, 24 episodes, randomized poses, extras present: success 67 %, grasp 100 %, ≥1 placed 88 %,
  0 unstable, 0 drops, 0 false picks, 33 collision steps**; clean table, 40 episodes (seed 4242): success 80 %,
  grasp 100 %, ≥1 placed 93 %, 0 unstable, 108 collision steps (arm-vs-bin contact steps, counted per control step).
- Recommendation: the checkpoint tolerates clutter only incidentally (clutter is image-only to it). For a real clutter
  capability, fine-tune with `randomization.clutter.prob ≈ 0.3` and track collision steps on `clutter_showcase`.
