# DEBUGGING.md — why 3 M PPO steps grasped but never placed

**Run diagnosed:** `ppo_sorting_20260926_073818` (fresh PPO, 3,001,344 steps, 2026-09-26 07:38–09:34, archived in
`checkpoints/archive_run9_fresh3M/`). Final trainer eval: success 0 %, grasp 50 %, placement 0 %, mean reward −1.5;
recent training episodes: grasp 79 %, lift 71 %, placement 9 %, length always 200.

## 1. What actually happens after a grasp (evidence)

Tool: `scripts/diagnose_placement.py` (rolls the checkpoint out from the home pose exactly like evaluation and logs
every step after the grasp; frame strips are written to `logs/diagnosis/ep*_workcell.png` / `ep*_overhead.png`,
per-step traces to `logs/diagnosis/ep*_trace.json`). 16 episodes, seeds 1000–1015, deterministic actions.

* 12/16 episodes grasped; **0/16 placed**. In *every* grasped episode the same thing happens:

  | ep | grasp step | what the policy does next |
  |---|---|---|
  | 2 | t=91 (grasp+lift in one step, +14.99 reward) | t=92: gripper command −1.00, dZ −1.00 → part dropped, −2.27 |
  | 3 | t=28 | t=31: opens (−2.12); re-grasps t=108, opens again t=109; 3 release penalties |
  | 4 | t=24 | opens t=28; 12 grasp/release cycles in the episode, −24 release penalties |
  | 5, 6, 7, 9 | t=68 / 40 / 23 / 26 | opens 1–3 steps after the grasp bonus; some re-grasp and repeat |

  The object never moves toward its bin (`dxy_bin` stays 0.38–0.58 m for the whole episode). The policy collects the
  one-time grasp (+10) and lift (+5) bonuses and then opens the gripper.

* Probe `scripts/probe_curriculum_policy.py`, section (2), same checkpoint from curriculum start states:
  * from **"grasped"** starts (holding the part in the air, far from the bin): 11/11 valid starts → the policy opens the
    gripper within 2–4 steps, **0 placements**;
  * from **"over-bin"** starts (holding the part above its correct bin): 11/11 → opens at step 0, **11/11 placements**;
  * from "straddle" starts: closes and lifts, then opens.

  So the network has learned the rule **"holding a lifted part ⇒ open the gripper"** — which is exactly right in the
  over-bin curriculum state (+40) and wrong everywhere else — and it does not condition the release on the
  `target_to_bin` features that are present in `observation.state`.

## 2. Ruling things out

| Hypothesis from the task | Result |
|---|---|
| Bin containment / placement detection is wrong (bounds, bin id, wrong object) | **No.** Probe section (1): a scripted carry + release fires `placed_count=1` in the correct bin in 5/6 tries (the 6th was a failed scripted grasp) and pays +40 placement, ≈ +56 total. Bin sites are 0.14 × 0.10 × 0.06 m boxes centred in each bin; `which_bin()` is checked for the *target* object; `bin_mapping` bracket→bin_a, bolt→bin_b is used consistently in env, reward and controller. |
| Transport reward gives no gradient once grasped | Partially. It was gated on `flags.lifted` (grasped **and** > 6 cm), so any dip re-armed it without penalty; but the real problem was that the policy never carried at all (see §3). |
| Gripper-release logic / a penalty term discourages placement | **Yes, indirectly** — see §3: the shaping could be *farmed* by releasing, and the release penalty (−2) was smaller than the reward a lift/drop/re-grasp cycle earned. `wrong_bin_penalty` never fired (0 wrong-bin events in 15 k episodes). `collision_penalty` averaged −0.4/episode — irrelevant. |
| Episode timeout cuts the episode before placement registers | **No.** A scripted pick-and-place takes 11–13 control steps after the grasp; episodes are 200 steps and the policy grasps at t≈25–90. |

## 3. Root cause: the shaping was not conservative

The potential-based terms were implemented as `weight · (φ_t − φ_{t−1})` **but the potential was reset to `None`
whenever the gating condition dropped**:

* `lift_shaping`: active only while `flags.grasped`. Lift 6–15 cm → up to +5; **open the gripper → the potential is
  discarded, the drop is never charged**; re-grasp at table height → φ restarts at 0 → lift again → +5 again.
* `approach_reward`: active only while *not* grasped. While holding, φ≈1 is discarded; after the drop the TCP is above
  the part → φ re-enters at a low value without a jump → descending onto the part is paid *again*.
* `align_reward` (+0.05/step) was paid only while not grasped and within 8 cm of the part — i.e. it paid for hovering
  next to a *released* part (up to +7/episode) and stopped as soon as the part was grasped.

Probe section (3) measured a scripted grasp→lift→release→re-grasp loop on the old reward: cycle 1 → +7.65,
cycle 2 → +11.70, cycle 3 → +14.27 (lift_shaping +4.0…+5.0 per cycle vs −2.0 release penalty). A reward pump.

Combined with the over-bin curriculum (15 % of training episodes paid +40 for opening while lifted) this makes
"grasp → lift → open" a stable local optimum: the drop costs −2 but is repaid by re-earned shaping, and the network's
simplest fit to the data is "open when lifted".

## 4. Fix (training/rewards/sorting_reward.py, configs/environment.yaml)

1. **Potentials are pure functions of state and are charged back on regress.** `lift_shaping` = f(object height) and
   `transport_reward` = f(object xy-distance to its bin) are active from the **first grasp until placement**,
   whatever the gripper does: dropping the part returns the lift progress, moving it away from the bin returns the
   transport progress. Each potential is switched on exactly once per target (approach: episode start → first grasp;
   lift/transport: first grasp → placement), so no gate can be cycled.
2. `approach_reward` stops for good at the first grasp of a target (no re-earning after a drop). Yaw alignment is
   folded into the approach potential (`approach_align_fraction`), the separate per-step `align_reward` is removed.
3. `release_penalty` fires only when the gripper is *commanded* open (contact flicker is not penalised) and not over the
   correct bin. On the placement step and while releasing over the correct bin the potentials are updated without
   payment, so dropping the part *into* the bin is not charged as "lost lift progress".
4. Curriculum share moved from over-bin starts (15 % → 10 %) to grasped starts (20 % → 25 %) so the policy sees more
   "carry it to the bin" than "just open".

Re-running probe (3) on the new reward: cycle 1 → +1.63 (includes the one-time lift bonus), then −0.67, −3.05,
−3.50 — **each extra cycle now loses ≈ 2.3**. Scripted carry + release still pays ≈ +56 (5/6). Regression tests:
`tests/test_reward_shaping.py`.

## 5. Does the fix alone learn placement? Honest answer: not within 300 k steps

`diag_baseline_fixedreward` (fresh PPO, fixed reward, 300 k steps, 12 envs, seed 42; logs in `logs/diag_baseline/`):
training episodes grasp 54 % / lift 45 % / placement 10 % / success 0 %; clean evaluations (from home) grasp 0–40 %,
placement 0 %. The reward pump is closed (placement in training episodes no longer comes with re-grasp loops), but
300 k steps is far below the ≈ 550 k the previous session needed before approach+grasp emerged, so the remaining gap
is sample efficiency — which is what the demonstration bootstrap (ASSUMPTIONS.md #16, README §"Teleop & imitation
bootstrap") addresses. The bootstrap-vs-baseline comparison is in `logs/checkpoint_comparison.json` and PROGRESS.md.

## 6. Reproduce

```bash
source .venv/bin/activate
python scripts/diagnose_placement.py --checkpoint checkpoints/archive_run9_fresh3M/checkpoint_003001344 --episodes 16
python scripts/probe_curriculum_policy.py checkpoints/archive_run9_fresh3M/checkpoint_003001344
python -m pytest tests/test_reward_shaping.py -q
```

---

# Part 2 — why the BC-bootstrapped policy got *worse* with PPO fine-tuning (Incremental Update 2)

**Run diagnosed:** `ppo_bc_bootstrap_300k` (BC init from 29 scripted demos → PPO 300 k, aux BC steps decaying to 0 over
150 k). Trainer evals: 0 % → 10 % success (250 k) → 0 % (300 k); training-episode reward rose smoothly to ≈ +6.5 while
clean evals scored −8 … −11.

## 7. Episode-level evidence (`scripts/diagnose_bc_regression.py`, seeds 777–786, deterministic, from home)

| checkpoint | drift vs demos (action MSE on demo frames) | phases reached in 10 episodes | grasp | ≥1 placed | mean R |
|---|---|---|---|---|---|
| BC init (0 PPO steps) | **0.032** | grasped 6 · lifted 1 · PLACED 3 | 100 % | 30 % | +23.3 |
| PPO 100 k | 0.064 | no-approach 4 · reached 1 · grasped 3 · carried-to-bin 2 | 50 % | 0 % | +0.4 |
| PPO 250 k | 0.185 | **no-approach 6** · grasped 2 · lifted 2 | 40 % | 0 % | −5.3 |
| PPO 275 k | 0.225 | **no-approach 7** · reached 1 · carried-to-bin 2 | 20 % | 0 % | −5.8 |
| PPO 301 k | 0.274 | **no-approach 7** · reached 1 · carried 1 · PLACED 1 | 20 % | 10 % | −3.3 |

What the trajectories show:
* The old failure mode is gone: **0 early gripper opens** while holding, in every checkpoint. When the policy does
  grasp it carries the part to the bin (min object–bin distance 0.04–0.10 m) and sometimes places it.
* What degrades is the **approach from the home pose**: in 6–7 of 10 episodes the TCP never gets within 5 cm of the
  part (min distance 0.06–0.15 m) — the skill BC had (100 % grasp) is being unlearned. Drift from the demonstrations
  grows monotonically with PPO steps (0.03 → 0.06 → 0.19 → 0.22 → 0.27) — catastrophic forgetting, not a bug.

## 8. Why the training curve went up anyway (train/eval gap)

Same 301 k checkpoint rolled out **the way training sees it** (stochastic actions, initial-state curriculum), 60 episodes:

| start type | share | mean R | ≥1 placed | dominant components |
|---|---|---|---|---|
| straddle (fingers around the part) | 22 % | +37.4 | 62 % | placement 27.7, grasp bonus 10.8, lift 5.4 |
| pregrasp (hover above part) | 17 % | +19.9 | 30 % | placement 16.0, grasp bonus 8.0 |
| grasped (already holding) | 27 % | +11.0 | 44 % | placement 17.5, lift 4.7, transport 3.2 |
| overbin | 7 % | +0.9 | 50 % | placement 20, drop −2.5 |
| **home (= evaluation protocol)** | **28 %** | **−8.7** | **6 %** | time/effort/timeout penalties; grasp bonus 2.4 |
| all | 100 % | **+12.0** | | |

The rising training reward is real task reward (placement bonuses) earned from **curriculum start states**, which make
up ~72 % of training episodes. The from-home episodes — the only ones the evaluation protocol measures — average −8.7
and contribute little gradient. No shaping term is being farmed: `approach_reward` / `lift_shaping` / `transport_reward`
do not appear among the dominant components of any start type, and the potentials are conservative since Part 1
(`tests/test_reward_shaping.py`). The gap is a **start-state distribution mismatch**, not reward exploitation.

## 9. Three concrete mechanisms of forgetting, and the fixes

| mechanism | evidence | fix (configs/training.yaml) |
|---|---|---|
| BC anchor vanished: interleaved aux steps (4 minibatches / iteration) decayed to 0 by 150 k, PPO ran 16 minibatches / iteration on ~O(1) normalised advantages | drift grows fastest after 100 k | **in-loss BC term** in every PPO minibatch (`training/policies/ppo_bc.py`), coefficient 1.0 → floor 0.25 over 1 M steps, never 0 (`bootstrap.bc_loss_*`) |
| unconstrained updates: approx-KL 0.05 / clip fraction 0.32 per update in the old run, lr 3e-4 from step 0 | drift 0.03 → 0.06 in the first 100 k | `target_kl: 0.02` early stopping; **LR warm-up** at 0.1× with an **evaluation-gated ramp** (×2 per non-regressing eval, ×0.5 on regression; `lr_gate`) |
| moving input normaliser: VecNormalize kept updating state statistics with curriculum/exploration data, re-mapping every input the BC network had learned | — | `bootstrap.freeze_obs_norm: true` (statistics seeded from the demos, then frozen) |
| start-state mismatch (§8) | home episodes 28 % of training, −8.7 mean | `bootstrap.curriculum_scale: 0.3` (≈ 79 % home starts when bootstrapping) |
| noisy exploration around a precise BC policy | std 0.6 | `log_std_init: −1.0` (std ≈ 0.37) |
| selection by recency | 250 k checkpoint (10 % success) shipped as "latest" 301 k (0 %) | **best-by-eval selection**: checkpoints saved at eval steps carry their own eval; `best` = highest success → placement → reward; UI ★, `Load best`, default for Run Inference and start-up; the BC init is evaluated too and competes |

Before/after numbers (Evaluation tab protocol, 20 episodes, rand 1.0, seed 4242) are in PROGRESS.md (Phase 23).

---

# Part 3 — "sim unstable" during cluttered inference (Incremental Update 7)

**Report:** live inference at 0.2 % success (1/507), most episodes flagged `sim unstable`, very high collision count,
table = task bracket + bolt + Extra L-bracket + Red plastic cap. Backend log: 371 of the last 500 episodes unstable;
MuJoCo warnings on DOF 14-16 (the **bolt**) and DOF 8-13 (the **bracket**) - i.e. the task objects diverged, not the
clutter bodies.

## 10. Two problems, separated by experiment (`scripts/diagnose_clutter.py`, `diagnose_clutter_contacts.py`, `diagnose_instability_rate.py`)

| driver | scene | unstable | success |
|---|---|---|---|
| random actions | clean / clutter | 0/12 / 0/12 | - |
| scripted controller | clean / clutter | 0/12 / 0/12 | 100 % / 100 % |
| trained policy | clean | **4/60 (7 %)** | 57 % |
| trained policy | clutter in the transport paths | 1/30 | 60 % |
| live inference, clean table, after the report | **3/52 (6 %)** | 56 % |

1. **Scope mismatch - real.** "Defaults" and every preset kept the extra/rejection parts on the table
   (`ensure_defaults` only re-placed the two task objects). The policy was trained on clean scenes and receives clutter
   only through the 64×64 image, so with clutter in the transport path it drags the grasped part through it. Fixed:
   Defaults / presets = bracket + bolt + bins only; clutter lives in the saved `clutter_showcase` (and
   `cluttered_u_bracket`) scenarios and in its own scorecard rows.
2. **Real physics instability - also real, and not clutter-specific.** Every divergence is the **bracket** (12 mm plate,
   56 mm flange) at or just after release into bin A: the flange top edge is pinched between a still-closing finger pad
   and the 8 mm bin wall/floor (contact list at divergence: `bin_a_w1~bracket_*`, `left_pad~bracket_*`), or a clutter
   part is pushed into a bin wall by the carried part. With the 2 ms step the constraint solver produces |qvel| of
   1e3-1e8 in one sub-step and the part tunnels through the table (one trace ends at (0.00, 0.00, -0.03) m: through the
   table top, into the floor). The live 74 % rate came from a clutter part re-spawned at its stored spot every reset
   after being knocked into a bin, so the pinch recurred each episode.

## 11. Fixes and their measured effect (same seeds, 60 clean policy episodes each unless noted)

| change | unstable | success | kept? |
|---|---|---|---|
| none (2 ms, stiff) | 4/60 | 57 % | - |
| soft/priority contacts on the 3 extra parts only | (clutter batch) 1/30 → 0/30 | 60 % | **yes** - they never touch the pads on purpose |
| free-joint damping 0.5 N·s/m, 0.005 N·m·s on all parts | 0/60, then 1/60 on a 2nd seed set | 67 % | **yes** (`physics.object_damping`) |
| lighter damping 0.3 / 0.002 | 1/60 | 68 % | no |
| soft bin contacts | 3/60 | 62 % | no |
| thicker bin walls | 3/60 | 55 % | no |
| softer pad contacts | 3/60 | **7 %** (grasps fail) | no |
| gripper force 60 → 30 N | 0/60 | 58 % | no (changes the gripper model) |
| impratio 10 → 1 | 0/60 | 57 % | no (softens friction) |
| **timestep 2 ms → 1 ms** | **0/60** | **70 %** | **yes** - pure integration accuracy |
| **final: 1 ms + damping + soft extras** | **0/60 + 0/60 (seeds 20000, 40000) + 0/30 clutter** | 60-70 % | - |

Also: the scene builder keeps re-applied parts ≥ 11 cm from the task objects and 7 cm outside every bin footprint
(`_free_spot`), so a knocked-away part can no longer be re-spawned inside a bin wall. Cost: physics per control step
doubles (single-env throughput ≈ 94 steps/s with rendering); the 10 Hz live loop is unaffected, training ~2× slower.
Tests: `tests/test_physics_fidelity.py` (clutter next to bins with the scripted controller, part spawned through a bin
wall, scene-builder clearance, damping configured on every part).

## 12. Can this checkpoint handle clutter?

Only incidentally. Its state input holds the two task-object poses; clutter appears solely in the image, and training
never contained extra parts, so it neither avoids them nor gets confused by them - it treats them as things it can
push through. The honest scorecard rows show what that yields (see PROGRESS.md). To make clutter a real capability,
fine-tune with `randomization.clutter.prob > 0` (training resets then drop 1-2 pool parts at free spots; wired, off by
default) and evaluate on `clutter_showcase`; expect the collision count, not just success, to be the metric to watch.
