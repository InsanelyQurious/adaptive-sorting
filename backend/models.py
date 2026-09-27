"""Typed request/response models for the API."""
from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    version: str
    device: str
    device_name: Optional[str] = None
    mujoco_version: str
    torch_version: str
    sb3_version: str
    lerobot_version: Optional[str] = None
    robot: str
    sim_mode: str
    uptime_s: float


class ResetRequest(BaseModel):
    seed: Optional[int] = None
    randomize: bool = True
    randomization_level: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    fixed_poses: Optional[dict[str, dict[str, float]]] = None  # {"bracket": {"x":..,"y":..,"yaw":..}}


class RandomizeRequest(BaseModel):
    seed: Optional[int] = None
    randomization_level: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class SpeedRequest(BaseModel):
    speed: float = Field(ge=0.1, le=10.0)


class ModeRequest(BaseModel):
    mode: str  # idle | random | inference


class StepRequest(BaseModel):
    steps: int = Field(default=1, ge=1, le=100)


class TrainingStartRequest(BaseModel):
    total_timesteps: Optional[int] = Field(default=None, ge=1000)
    init_from_demos: Optional[bool] = Field(default=None, description="Behaviour-cloning bootstrap from datasets/demos before PPO")
    demo_types: Optional[list[str]] = Field(default=None, description="episode types to use for BC, e.g. ['teleop_demo','scripted_demo']")
    bc_epochs: Optional[int] = Field(default=None, ge=1, le=1000)
    training_episodes: Optional[int] = Field(default=None, ge=1, description="Optional cap on episodes")
    learning_rate: Optional[float] = Field(default=None, gt=0)
    batch_size: Optional[int] = Field(default=None, ge=8)
    gamma: Optional[float] = Field(default=None, ge=0, le=1)
    ent_coef: Optional[float] = Field(default=None, ge=0)
    seed: Optional[int] = None
    n_envs: Optional[int] = Field(default=None, ge=1, le=32)
    randomization_level: Optional[float] = Field(default=None, ge=0, le=1)
    resume_from: Optional[str] = Field(default=None, description="checkpoint name, 'latest', or null")
    run_name: Optional[str] = None


class EvaluateRequest(BaseModel):
    episodes: int = Field(default=10, ge=1, le=200)
    checkpoint: Optional[str] = None
    randomization_level: Optional[float] = Field(default=None, ge=0, le=1)
    seed: Optional[int] = None


class LoadCheckpointRequest(BaseModel):
    checkpoint: str  # name, 'best' or 'latest'


class InferenceStartRequest(BaseModel):
    checkpoint: Optional[str] = None
    speed: Optional[float] = None
    deterministic: bool = True


class ActionResponse(BaseModel):
    ok: bool
    message: str = ""
    data: Optional[dict[str, Any]] = None


class CheckpointInfo(BaseModel):
    name: str
    path: str
    timesteps: Optional[int] = None
    created_iso: Optional[str] = None
    algorithm: Optional[str] = None
    success_rate: Optional[float] = None
    mean_reward: Optional[float] = None
    placement_rate: Optional[float] = None
    grasp_rate: Optional[float] = None
    eval_episodes: Optional[int] = None
    eval_at_timesteps: Optional[int] = None
    eval_matches: Optional[bool] = None
    run_name: Optional[str] = None
    reason: Optional[str] = None
    episodes: Optional[int] = None
    device: Optional[str] = None
    size_mb: Optional[float] = None


class CheckpointList(BaseModel):
    checkpoints: list[CheckpointInfo]
    latest: Optional[str] = None
    best: Optional[str] = None   # highest eval success (ties: placement rate, eval reward); default for loading
    loaded: Optional[str] = None


class TeleopStartRequest(BaseModel):
    reset: bool = True


class TeleopActionRequest(BaseModel):
    axes: list[float] = Field(default_factory=lambda: [0.0] * 6, min_length=6, max_length=6)
    gripper: Optional[float] = None  # >0 closed, <=0 open, None = unchanged


class RecordStartRequest(BaseModel):
    name: Optional[str] = None
    episode_type: Optional[str] = None   # teleop_demo | scripted_demo | policy_rollout | random_policy | manual
    reset: bool = True


class ScriptedDemoRequest(BaseModel):
    episodes: int = Field(default=5, ge=1, le=200)
    noise_std: float = Field(default=0.0, ge=0.0, le=0.5)
    record: bool = True


class FrustumRequest(BaseModel):
    enabled: bool = True


class PickAtRequest(BaseModel):
    u: float = Field(ge=0.0, le=1.0, description="normalised x in the streamed viewport image")
    v: float = Field(ge=0.0, le=1.0, description="normalised y in the streamed viewport image")
    bin: Optional[str] = None


class PickByNameRequest(BaseModel):
    object: str
    bin: Optional[str] = None


class ReplaySeekRequest(BaseModel):
    index: int = Field(ge=0)


class ScorecardRequest(BaseModel):
    episodes: int = Field(default=5, ge=1, le=50)
    seed: Optional[int] = None
    randomization_level: Optional[float] = Field(default=None, ge=0, le=1)
    scenarios: Optional[list[str]] = None


class SweepRequest(BaseModel):
    levels: list[float] = Field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])
    episodes: int = Field(default=6, ge=1, le=50)
    seed: Optional[int] = None


class CompareRequest(BaseModel):
    checkpoint_a: str
    checkpoint_b: str
    episodes: int = Field(default=20, ge=1, le=100)
    seed: Optional[int] = 4242
    randomization_level: float = Field(default=1.0, ge=0, le=1)


class SpawnRequest(BaseModel):
    part: str
    u: Optional[float] = Field(default=None, ge=0, le=1)
    v: Optional[float] = Field(default=None, ge=0, le=1)
    x: Optional[float] = None
    y: Optional[float] = None
    yaw: Optional[float] = None


class SaveScenarioRequest(BaseModel):
    name: str = Field(min_length=1, max_length=48)


class ChatRequest(BaseModel):
    text: str = Field(min_length=1, max_length=400)


class ToggleRequest(BaseModel):
    part: str
    on: Optional[bool] = None       # None = flip
    u: Optional[float] = Field(default=None, ge=0, le=1)
    v: Optional[float] = Field(default=None, ge=0, le=1)
    x: Optional[float] = None
    y: Optional[float] = None
    yaw: Optional[float] = None
