import { useState } from 'react'
import { fmt, fmtDur, fmtInt, fmtPct, fmtSci, post, type MetricRow, type TrainingStatus } from '../api'
import { MetricChart } from './Charts'
import { Badge, Btn, Card, Stat, StatGrid, Bar, type Tone } from './ui'

function last<T = number>(rows: MetricRow[], key: string): T | null {
  for (let i = rows.length - 1; i >= 0; i--) { const v = rows[i][key]; if (typeof v === 'number') return v as unknown as T }
  return null
}

const stateTone = (s?: string): Tone => s === 'running' ? 'good' : s === 'paused' || s === 'evaluating' || s === 'saving' || s === 'bc_pretraining' ? 'warn' : s === 'crashed' ? 'bad' : s === 'completed' ? 'info' : 'muted'

export function TrainingPanel({ status, metrics, flash }: { status: TrainingStatus | null; metrics: MetricRow[]; flash: (m: string, bad?: boolean) => void }) {
  const [showConfig, setShowConfig] = useState(false)
  const st = status?.state ?? 'not_started'
  const running = ['running', 'paused', 'evaluating', 'saving', 'starting', 'bc_pretraining'].includes(st) && status?.process_alive !== false
  const boot = status?.bootstrap
  const act = async (label: string, p: Promise<any>) => { try { await p; flash(label) } catch (e: any) { flash(`${label} failed: ${e.message}`, true) } }
  // moving average of episode reward over the last 10 logged rows
  const withMA = metrics.map((r, i) => {
    const w = metrics.slice(Math.max(0, i - 9), i + 1).map((x) => x['rollout/ep_rew_mean']).filter((v) => typeof v === 'number') as number[]
    return { ...r, ma_reward: w.length ? w.reduce((a, b) => a + b, 0) / w.length : undefined } as MetricRow
  })
  return (
    <div className="flex flex-col gap-2">
      <Card title="Training status" right={<Badge tone={stateTone(st)} pulse={st === 'running'}>{st.replace('_', ' ')}</Badge>}>
        <StatGrid cols={3}>
          <Stat label="Policy" value={status?.algorithm ? `${status.algorithm} · ${status.run_name ?? ''}` : 'N/A'} mono={false} title={status?.policy} />
          <Stat label="Environment" value={status?.env_id ?? 'N/A'} />
          <Stat label="Device" value={status?.device ?? 'N/A'} />
          <Stat label="Episodes" value={fmtInt(status?.episodes)} />
          <Stat label="Steps" value={status?.timesteps != null ? `${fmtInt(status.timesteps)} / ${fmtInt(status.total_timesteps)}` : 'N/A'} />
          <Stat label="Elapsed" value={fmtDur(status?.elapsed_s)} />
          <Stat label="Current reward" value={fmt(last(metrics, 'rollout/ep_rew_mean'))} title="mean episode reward, last rollout (SB3 rollout/ep_rew_mean)" />
          <Stat label="Avg reward (100 ep)" value={fmt(status?.recent_mean_reward)} />
          <Stat label="Success rate (100 ep)" value={fmtPct(status?.recent_success_rate)} />
          <Stat label="Grasp / lift / place" value={`${fmtPct(status?.recent_grasp_rate, 0)} / ${fmtPct(status?.recent_lift_rate, 0)} / ${fmtPct(status?.recent_placement_rate, 0)}`} />
          <Stat label="Episode length" value={fmt(status?.recent_mean_length, 1)} />
          <Stat label="Learning rate" value={fmtSci(last(metrics, 'train/learning_rate'))} />
          <Stat label="Entropy" value={fmt(last(metrics, 'train/entropy_loss') != null ? -(last(metrics, 'train/entropy_loss') as number) : null, 3)} title="policy entropy (= -entropy_loss)" />
          <Stat label="Value loss" value={fmt(last(metrics, 'train/value_loss'), 4)} />
          <Stat label="Policy loss" value={fmt(last(metrics, 'train/policy_gradient_loss'), 4)} />
          <Stat label="Clip fraction" value={fmt(last(metrics, 'train/clip_fraction'), 3)} />
          <Stat label="FPS" value={fmt(status?.fps, 0)} />
          <Stat label="GPU util" value={status?.gpu_utilization != null ? fmt(status.gpu_utilization, 0, '%') : 'N/A'} title="N/A without a CUDA device" />
        </StatGrid>
        <div className="mt-2"><Bar value={status?.progress ?? 0} /></div>
        {status?.last_eval && (
          <div className="mt-2 text-[10px] mono" style={{ color: 'var(--text-secondary)' }}>
            last eval @ {fmtInt(status.last_eval.timesteps)}: success {fmtPct(status.last_eval.success_rate)} · grasp {fmtPct(status.last_eval.grasp_rate)} · place {fmtPct(status.last_eval.placement_rate)} · reward {fmt(status.last_eval.mean_reward, 1)} · len {fmt(status.last_eval.mean_length, 0)} ({status.last_eval.episodes} ep)
          </div>
        )}
        {status?.latest_checkpoint && <div className="mt-1 text-[10px] mono" style={{ color: 'var(--text-secondary)' }}>latest checkpoint: {status.latest_checkpoint}</div>}
        {boot?.enabled && (
          <div className="mt-1 text-[10px] mono" style={{ color: 'var(--series-4)' }}>
            bootstrap: {boot.demo_episodes ?? 'N/A'} demos ({boot.demo_type_counts ? Object.entries(boot.demo_type_counts).map(([t, n]) => `${n} ${t}`).join(', ') : (boot.demo_types ?? []).join(', ') || 'N/A'}) · {fmtInt(boot.demo_frames)} frames
            {boot.stage === 'bc_pretraining' && ` · BC epoch ${boot.bc_epoch}/${boot.bc_epochs} · action MSE ${fmt(boot.train_action_mse, 4)}`}
            {boot.after && ` · BC holdout action MSE ${fmt(boot.before?.action_mse, 4)} → ${fmt(boot.after?.action_mse, 4)} · gripper acc ${fmtPct(boot.after?.gripper_sign_acc, 0)}`}
            {boot.bc_coef_now != null && boot.bc_coef_now > 0 && ` · BC anchor coef ${fmt(boot.bc_coef_now, 2)}`}
            {boot.bc_anchor?.demo_action_mse != null && ` · drift vs demos (action MSE) ${fmt(boot.bc_anchor.demo_action_mse, 4)}`}
            {boot.freeze_obs_norm && ' · obs-norm frozen'}
            {boot.curriculum_scale != null && boot.curriculum_scale !== 1 && ` · curriculum ×${fmt(boot.curriculum_scale, 2)}`}
          </div>
        )}
        {status?.lr_gate?.enabled && (
          <div className="mt-1 text-[10px] mono" style={{ color: 'var(--text-secondary)' }}>
            lr gate: scale {fmt(status.lr_gate.scale, 3)} → lr {fmtSci(status.lr_gate.lr)}{status.lr_gate.events?.length ? ` · last: ${status.lr_gate.events[status.lr_gate.events.length - 1].why} @ ${fmtInt(status.lr_gate.events[status.lr_gate.events.length - 1].timesteps)}` : ' · warm-up'}
            {status.training_config?.target_kl != null && ` · target KL ${status.training_config.target_kl}`}
          </div>
        )}
        {boot && boot.enabled === false && <div className="mt-1 text-[10px] mono" style={{ color: 'var(--text-muted)' }}>bootstrap: off (PPO from random init)</div>}
        {status?.error && <div className="mt-1 text-[10px] mono" style={{ color: 'var(--status-bad)' }}>error: {status.error}</div>}
        <div className="flex flex-wrap gap-1.5 mt-2">
          <Btn onClick={() => act('Pause requested', post('/training/pause'))} disabled={!running || st === 'paused'}>Pause</Btn>
          <Btn onClick={() => act('Resume requested', post('/training/resume'))} disabled={!running || st !== 'paused'}>Resume</Btn>
          <Btn tone="danger" onClick={() => { if (window.confirm('Stop the in-progress training run? A final checkpoint will be saved.')) act('Stop requested', post('/training/stop')) }} disabled={!running}>Stop</Btn>
          <Btn onClick={() => act('Checkpoint save requested', post('/training/save'))} disabled={!running}>Save Checkpoint</Btn>
          <Btn onClick={() => act('Trainer evaluation requested', post('/training/evaluate-in-trainer'))} disabled={!running} title="Run the periodic evaluation inside the trainer now">Evaluate (trainer)</Btn>
          <Btn onClick={() => setShowConfig((v) => !v)}>{showConfig ? 'Hide config' : 'Config'}</Btn>
        </div>
        {showConfig && status?.training_config && (
          <pre className="mono text-[10px] mt-2 p-2 scroll" style={{ background: 'var(--surface-2)', maxHeight: 160, overflow: 'auto', color: 'var(--text-secondary)' }}>{JSON.stringify(status.training_config, null, 1)}</pre>
        )}
      </Card>
      <MetricChart rows={withMA} title="Episode reward" series={[{ key: 'rollout/ep_rew_mean', label: 'reward' }, { key: 'ma_reward', label: 'moving avg' }]} yFormat={(v) => fmt(v, 1)} />
      <MetricChart rows={metrics} title="Success rate (last 100 ep)" series={[{ key: 'rollout/recent_success_rate', label: 'success' }]} yFormat={(v) => fmtPct(v, 0)} domain={[0, 1]} />
      <MetricChart rows={metrics} title="Grasp rate (last 100 ep)" series={[{ key: 'rollout/recent_grasp_rate', label: 'grasp' }]} yFormat={(v) => fmtPct(v, 0)} domain={[0, 1]} />
      <MetricChart rows={metrics} title="Lift / placement rate (last 100 ep)" series={[{ key: 'rollout/recent_lift_rate', label: 'lift' }, { key: 'rollout/recent_placement_rate', label: 'placement' }]} yFormat={(v) => fmtPct(v, 0)} domain={[0, 1]} />
      <MetricChart rows={metrics} title="Episode length" series={[{ key: 'rollout/ep_len_mean', label: 'length' }]} yFormat={(v) => fmt(v, 0)} />
      <MetricChart rows={metrics} title="Policy loss" series={[{ key: 'train/policy_gradient_loss', label: 'policy loss' }]} yFormat={(v) => fmt(v, 4)} />
      <MetricChart rows={metrics} title="Value loss" series={[{ key: 'train/value_loss', label: 'value loss' }]} yFormat={(v) => fmt(v, 3)} />
      <MetricChart rows={metrics} title="Entropy loss" series={[{ key: 'train/entropy_loss', label: 'entropy loss' }]} yFormat={(v) => fmt(v, 2)} />
      <MetricChart rows={metrics} title="Learning rate" series={[{ key: 'train/learning_rate', label: 'lr' }]} yFormat={(v) => fmtSci(v)} />
      {metrics.some((r) => typeof r['eval/success_rate'] === 'number') && <MetricChart rows={metrics} title="Eval success / placement (clean env, per periodic eval)" series={[{ key: 'eval/success_rate', label: 'success' }, { key: 'eval/placement_rate', label: 'placement' }]} yFormat={(v) => fmtPct(v, 0)} domain={[0, 1]} />}
      {metrics.some((r) => typeof r['bc/demo_action_mse'] === 'number') && <MetricChart rows={metrics} title="Drift from demonstrations (BC anchor action MSE)" series={[{ key: 'bc/demo_action_mse', label: 'action mse' }, { key: 'bc/coef', label: 'coef' }]} yFormat={(v) => fmt(v, 3)} />}
      {metrics.some((r) => typeof r['train/approx_kl'] === 'number') && <MetricChart rows={metrics} title="Approx KL per update (trust region)" series={[{ key: 'train/approx_kl', label: 'approx kl' }]} yFormat={(v) => fmt(v, 4)} />}
      {metrics.some((r) => typeof r['bc/aux_action_mse'] === 'number') && <MetricChart rows={metrics} title="BC aux action MSE (legacy interleaved steps)" series={[{ key: 'bc/aux_action_mse', label: 'action mse' }]} yFormat={(v) => fmt(v, 4)} />}
    </div>
  )
}
