// REST + WebSocket helpers. The backend port is 8000; when the built app is served by the
// backend itself, same-origin is used.
const explicit = (import.meta as any).env?.VITE_BACKEND_URL as string | undefined
export const BASE: string = explicit ?? ((location.port === '3001' || location.port === '3000' || location.port === '5173') ? `http://${location.hostname}:8000` : location.origin)
export const WS_BASE: string = BASE.replace(/^http/, 'ws')

export type Num = number | null | undefined

export interface ObjectState {
  position: number[]; yaw: number; in_bin: string | null; placed: boolean; grasped: boolean; lifted: boolean; target_bin: string
}
export interface PolicyInfo {
  name: string; checkpoint: string; algorithm: string; timesteps?: number; episodes?: number; device: string;
  trained_on_device?: string; eval_success_rate?: number | null; eval_mean_reward?: number | null;
  observation_type: string; action_type: string; action_dim: number; state_dim: number; image_shape?: number[] | null;
  normalized_state: boolean; created_iso?: string
}
export interface EvalProgress {
  running: boolean; episodes?: number; completed?: number; successes?: number; success_rate?: number | null; label?: string | null; id?: string | null;
  average_reward?: number | null; average_length?: number | null; checkpoint?: string; last?: any; last_episodes?: any[]
}
export interface CameraInfo {
  name: string; position: number[]; quaternion_wxyz: number[]; view_direction: number[]; image_up_world: number[]; image_up_axis: string;
  tilt_from_vertical_deg: number; description: string; fovy_deg: number; image_size: number[]; height_above_table_m: number;
  footprint_on_table_m: number[]; viewport?: { name: string; right: number[]; forward: number[]; position: number[] }
}
export interface TeleopState { active: boolean; clients: number; axes: number[]; gripper_closed: boolean; input_fresh: boolean }
export interface RecordingState { recording: boolean; episode_type: string | null; path: string | null; episodes: number | null; frames_in_episode: number | null }
export interface DemoInfo {
  name: string; path: string; episode_type: string; timestamp: string; frames: number; duration_s: number; aborted: boolean; seed?: number | null;
  outcome: { success: boolean; placed_count: number; reward: number | null; length: number | null; grasp_success: boolean; dropped: boolean; wrong_bin: boolean; truncated: boolean };
  controller?: string; size_mb?: number
}
export interface DemoCatalogue { demos: DemoInfo[]; summary: { total_episodes: number; total_frames: number; by_type: Record<string, { episodes: number; frames: number; successes: number; placed: number }> }; recording: RecordingState; dir: string }
export interface AssistState { active: boolean; object: string; bin: string; phase: string; retries: number; started_step: number; result: string | null; events: { t: number; event: string }[] }
export interface ReplayState { name: string; episode_type: string; index: number; n: number; playing: boolean; exact_initial_state: boolean; max_state_deviation: number; max_reward_deviation: number; recorded_outcome: any }
export interface SafetyState { estop: boolean; workspace_margin_m: number; joint_limit_margin_rad: number; velocity_utilization: number; collision_now: boolean; envelope_ok: boolean; limits: any; events: any[] }
export interface SimState {
  present_count?: number;
  last_reset: { t: number; episode: number; reset_count: number; bins: string; bins_fixed: boolean; level: number; mode: string } | null; bins_fixed: boolean;
  safety: SafetyState; show_wrist: boolean;
  scene: { variants: Record<string, string>; layout: { part: string; x: number; y: number; yaw: number }[]; reject_on_table: boolean; false_pick_this_episode: boolean } | null;
  assist: AssistState | null; replay: ReplayState | null; batch: { kind: string; id: string; i: number; n: number; status: string; current: string | null; results: any[]; episodes: number } | null;
  camera: CameraInfo | null; show_frustum: boolean; teleop: TeleopState; recording: RecordingState;
  scripted: { remaining: number; controller?: string; phase?: string; target?: string | null; retries?: number } | null;
  mode: string; paused: boolean; speed: number; sim_time: number; episode: number; step: number; max_steps: number;
  device: string; robot: string; joint_positions: number[]; joint_velocities: number[]; ee_position: number[]; ee_yaw: number;
  ee_target: number[]; gripper_opening: number; gripper_cmd_closed: number; objects: Record<string, ObjectState>;
  bins: Record<string, number[]>; target: string | null; target_bin: string | null; grasped: boolean; lifted: boolean;
  placed_count: number; collisions: number; action: { names: string[]; values: number[] }; reward: number; episode_reward: number;
  reward_components: Record<string, number>; state_vector: number[] | null; state_layout: string[]; episode_config: any;
  policy: PolicyInfo | null; inference: {
    episodes: number; successes: number; success_rate: Num; placement_rate: Num; grasp_rate: Num; action_latency_ms: Num; parts_per_minute: Num; parts_placed: number; sim_seconds: number; collision_steps: number; episodes_with_collision: number; control_effort_mean: Num; episode_control_effort: number;
    policy_calls: number; control_fps: Num; stream_fps: Num; deterministic: boolean
  }; evaluation: EvalProgress; last_episode: any; total_steps: number; error: string | null
}
export interface FrameMessage { type: 'frame'; t: number; frame: string; overhead: string | null; wrist?: string | null; overhead_size: number[]; state: SimState }

export interface TrainingStatus {
  state: string; pid?: number; process_alive?: boolean; run_name?: string; algorithm?: string; policy?: string; env_id?: string;
  device?: string; timesteps?: number; total_timesteps?: number; progress?: number; iteration?: number; elapsed_s?: number;
  fps?: Num; latest_checkpoint?: string | null; last_eval?: any; n_envs?: number; learning_rate?: Num; error?: string | null; bootstrap?: any; lr_gate?: any;
  episodes?: number; recent_success_rate?: Num; recent_grasp_rate?: Num; recent_lift_rate?: Num; recent_placement_rate?: Num;
  recent_mean_reward?: Num; recent_mean_length?: Num; recent_mean_collisions?: Num; gpu_utilization?: Num; training_config?: any
}
export type MetricRow = Record<string, number | string>

export async function post<T = any>(path: string, body: any = {}): Promise<T> {
  const r = await fetch(`${BASE}${path}`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) })
  const data = await r.json().catch(() => ({}))
  if (!r.ok) throw new Error(data.detail ?? `${r.status} ${r.statusText}`)
  return data
}
export async function get<T = any>(path: string): Promise<T> {
  const r = await fetch(`${BASE}${path}`)
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`)
  return r.json()
}

export const fmt = (v: Num, digits = 2, suffix = ''): string =>
  v === null || v === undefined || Number.isNaN(v) || !Number.isFinite(v) ? 'N/A' : `${Number(v).toFixed(digits)}${suffix}`
export const fmtInt = (v: Num): string => (v === null || v === undefined || Number.isNaN(v)) ? 'N/A' : Math.round(v).toLocaleString()
export const fmtPct = (v: Num, digits = 1): string => (v === null || v === undefined || Number.isNaN(v)) ? 'N/A' : `${(v * 100).toFixed(digits)}%`
export const fmtSci = (v: Num): string => (v === null || v === undefined || Number.isNaN(v)) ? 'N/A' : Number(v).toExponential(2)
export const fmtDur = (s: Num): string => {
  if (s === null || s === undefined || Number.isNaN(s)) return 'N/A'
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = Math.floor(s % 60)
  return h > 0 ? `${h}h ${m.toString().padStart(2, '0')}m` : `${m}m ${sec.toString().padStart(2, '0')}s`
}

export async function del<T = any>(path: string): Promise<T> {
  const r = await fetch(`${BASE}${path}`, { method: 'DELETE' })
  const data = await r.json().catch(() => ({}))
  if (!r.ok) throw new Error(data.detail ?? `${r.status} ${r.statusText}`)
  return data
}
