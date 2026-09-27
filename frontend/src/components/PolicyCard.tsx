import type { SimState } from '../api'
import { fmtInt, fmtPct } from '../api'
import { Card, Stat, StatGrid } from './ui'

export function PolicyCard({ state }: { state: SimState | null }) {
  const p = state?.policy
  return (
    <Card title="Active policy">
      {!p ? <div className="text-[11px]" style={{ color: 'var(--text-muted)' }}>No checkpoint loaded. Checkpoints appear here as training saves them.</div> : (
        <StatGrid cols={2}>
          <Stat label="Policy" value={p.name} mono={false} />
          <Stat label="Algorithm" value={p.algorithm} />
          <Stat label="Observation" value={p.observation_type} mono={false} title={p.observation_type} />
          <Stat label="Action" value={`${p.action_type} · dim ${p.action_dim}`} mono={false} title={p.action_type} />
          <Stat label="Camera" value="overhead (MuJoCo)" mono={false} />
          <Stat label="Device" value={state?.device ?? 'N/A'} />
          <Stat label="Checkpoint" value={p.checkpoint} />
          <Stat label="Training steps" value={fmtInt(p.timesteps)} />
          <Stat label="Eval success (train-time)" value={fmtPct(p.eval_success_rate)} />
          <Stat label="Live success" value={fmtPct(state?.inference.success_rate)} />
        </StatGrid>
      )}
    </Card>
  )
}
