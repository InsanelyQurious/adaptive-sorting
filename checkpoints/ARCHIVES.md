# Archived checkpoints
Each `archive_*` directory holds the checkpoints of one training run (see PROGRESS.md for the run history).
They are not listed in the dashboard's Checkpoints tab but remain loadable by absolute path
(`POST /inference/load {"checkpoint": "/abs/path/to/archive_run9_fresh3M/checkpoint_003001344"}`).
- archive_run1 .. archive_run4_6, archive_run8_late : session-1 runs (old reward, see ASSUMPTIONS.md #11)
- archive_run9_fresh3M : the 3.0M-step fresh PPO run (2026-09-26 07:38-09:34) diagnosed in DEBUGGING.md
  (grasp ~50 %, placement 0 % - the run that motivated the reward fix)
- (active in `checkpoints/`) `ppo_bc_bootstrap_300k` — BC init + PPO 300 k with a decaying BC aux term; the run compared against
  the baseline in PROGRESS.md / logs/checkpoint_comparison.json (its 0/25k/50k/75k checkpoints were overwritten by v2 before archiving was automated)
- archive_bc_bootstrap_v2_300k : `ppo_bc_bootstrap_v2_300k` — constant BC aux term, PPO lr 1e-4, log_std_init −1.0; clean eval worse (logs/checkpoint_comparison_v2.json).
  Its checkpoint_000000000 is the BC-only initialisation (29 scripted demos, 0 PPO steps): grasp 83 %, placement 10 % on the 30-seed eval.
