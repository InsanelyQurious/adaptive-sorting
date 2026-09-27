# lerobot_adapter

Where LeRobot (0.6.1) is and isn't in the loop:

| Concern | Implementation |
|---|---|
| Observation / action naming | Environment emits `observation.images.overhead`, `observation.state`, `action` (LeRobot conventions). |
| Dataset format | `dataset_writer.LeRobotEpisodeRecorder` writes a real `LeRobotDataset` v3.0 (parquet + images + meta) for evaluation rollouts. |
| Policy interface | `policy.SortingRLPolicy.select_action(batch)` wraps the PPO checkpoint with LeRobot's policy surface. |
| RL training loop | **stable-baselines3 PPO** (`training/train.py`). LeRobot's trainer is imitation-learning oriented and does not implement PPO/SAC. |

SB3's `torch.nn.ModuleDict` cannot hold keys containing dots, so `simulation/environments/sorting_env.py::SB3KeyAlias`
renames `observation.images.overhead → image` and `observation.state → state` **only for the RL library**.
