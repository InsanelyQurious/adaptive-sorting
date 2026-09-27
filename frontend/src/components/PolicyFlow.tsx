import type { FrameMessage } from '../api'
import { fmt } from '../api'

// What the policy sees -> what it decides. Kept minimal: camera image, the policy, the seven action channels.
export function PolicyFlow({ frame }: { frame: FrameMessage | null }) {
  const s = frame?.state
  const a = s?.action
  const active = s?.mode === 'inference' || s?.mode === 'evaluate'
  return (
    <div className="card p-2">
      <div className="flex items-stretch gap-2 text-[10px]">
        <div className="flex flex-col items-center gap-1" style={{ minWidth: 70 }}>
          <div className="stat-label">Sees</div>
          {frame?.overhead ? <img src={`data:image/jpeg;base64,${frame.overhead}`} width={56} height={56} alt="obs" style={{ imageRendering: 'pixelated', border: '1px solid var(--line-strong)' }} /> : <div style={{ width: 56, height: 56, background: 'var(--surface-3)' }} />}
        </div>
        <div className="flex items-center" style={{ color: 'var(--text-muted)' }}>→</div>
        <div className="flex flex-col items-center justify-center gap-0.5 px-2" style={{ minWidth: 72, border: `1px solid ${active ? 'var(--accent)' : 'var(--line-strong)'}`, borderRadius: 2 }}>
          <div className="stat-label">Policy</div>
          <div className="mono text-[11px]">{s?.policy ? s.policy.algorithm : 'none'}</div>
          <div className="mono" style={{ color: active ? 'var(--status-good)' : 'var(--text-muted)' }}>{active ? 'driving' : 'idle'}</div>
        </div>
        <div className="flex items-center" style={{ color: 'var(--text-muted)' }}>→</div>
        <div className="flex-1 min-w-0">
          <div className="stat-label mb-1">Decides (move X · Y · Z · turn · gripper)</div>
          {a ? a.names.filter((_, i) => i !== 3 && i !== 4).map((n) => {
            const i = a.names.indexOf(n); const v = a.values[i]
            return (
              <div key={n} className="flex items-center gap-1 leading-4">
                <span className="mono" style={{ width: 44, color: 'var(--text-secondary)' }}>{n === 'dRz' ? 'turn' : n === 'gripper' ? 'grip' : n.replace('d', '')}</span>
                <div className="flex-1 relative h-2" style={{ background: 'var(--surface-3)' }}>
                  <div className="absolute top-0 bottom-0" style={{ left: '50%', width: 1, background: 'var(--line-strong)' }} />
                  <div className="absolute top-0 bottom-0" style={{ left: v < 0 ? `${50 + v * 50}%` : '50%', width: `${Math.abs(v) * 50}%`, background: i === 6 ? 'var(--series-2)' : 'var(--series-1)' }} />
                </div>
                <span className="mono" style={{ width: 40, textAlign: 'right' }}>{i === 6 ? (v > 0 ? 'close' : 'open') : fmt(v, 2)}</span>
              </div>
            )
          }) : <div className="mono">N/A</div>}
        </div>
      </div>
    </div>
  )
}
