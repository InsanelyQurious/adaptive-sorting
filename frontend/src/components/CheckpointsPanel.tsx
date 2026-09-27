import { fmt, fmtInt, fmtPct, post, type SimState } from '../api'
import { usePolling } from '../hooks'
import { Btn, Card } from './ui'

interface Ck { name: string; timesteps: number | null; created_iso: string | null; success_rate: number | null; mean_reward: number | null; placement_rate: number | null; grasp_rate: number | null; eval_episodes: number | null; eval_at_timesteps: number | null; eval_matches: boolean | null; reason: string | null; size_mb: number | null }

// Checkpoints are ranked by their OWN evaluation (highest eval success, ties -> placement rate -> eval reward);
// "best" is what Load best / Run Inference / backend start-up pick. "latest" is only the most recent save.
export function CheckpointsPanel({ state, flash }: { state: SimState | null; flash: (m: string, bad?: boolean) => void }) {
  const { data, refresh } = usePolling<{ checkpoints: Ck[]; latest: string | null; best: string | null; loaded: string | null }>('/checkpoints', 5000)
  const load = async (name: string) => {
    if (state?.mode === 'inference' && !window.confirm(`Load ${name} while inference is running? The live policy will switch immediately.`)) return
    try { await post('/inference/load', { checkpoint: name }); flash(`Loaded ${name}`); refresh() } catch (e: any) { flash(`Load failed: ${e.message}`, true) }
  }
  const cks = (data?.checkpoints ?? []).slice().reverse()
  return (
    <Card title={`Checkpoints (${cks.length})`} right={<span className="flex gap-1"><Btn tone="accent" onClick={() => load('best')} disabled={!data?.best} title="Highest eval success rate (ties: placement rate, then eval reward)">★ Load best</Btn><Btn onClick={() => load('latest')} disabled={!data?.latest} title="Most recent save, regardless of its evaluation">Load latest</Btn></span>}>
      {data?.best && <div className="text-[10px] mono mb-1" style={{ color: 'var(--series-4)' }}>★ best by eval: {data.best}{data.best !== data.latest ? ` (latest is ${data.latest})` : ' (= latest)'}</div>}
      {cks.length === 0 ? <div className="text-[11px]" style={{ color: 'var(--text-muted)' }}>No checkpoints yet — start training.</div> : (
        <div className="scroll" style={{ maxHeight: 420, overflow: 'auto' }}>
          <table className="w-full text-[10px] mono">
            <thead style={{ color: 'var(--text-muted)' }}><tr className="text-left"><th className="pb-1">name</th><th>steps</th><th title="evaluation success rate (both parts sorted) at this checkpoint">eval succ</th><th title="≥1 part placed">eval place</th><th>eval R</th><th title="evaluation episodes · step the eval was run at">eval @</th><th>MB</th><th></th></tr></thead>
            <tbody>
              {cks.map((c) => {
                const loaded = state?.policy?.checkpoint === c.name
                const best = data?.best === c.name
                const stale = c.success_rate != null && c.eval_matches === false
                return (
                  <tr key={c.name} style={{ borderTop: '1px solid var(--line)', color: loaded ? 'var(--text-primary)' : 'var(--text-secondary)', background: best ? 'rgba(201,133,0,0.08)' : undefined }}>
                    <td className="py-1">{best && <span style={{ color: 'var(--series-4)' }} title="best by evaluation">★ </span>}{c.name}{loaded && <span style={{ color: 'var(--status-good)' }}> ●</span>}{c.reason === 'bc_init' && <span style={{ color: 'var(--text-muted)' }}> (BC init)</span>}</td>
                    <td>{fmtInt(c.timesteps)}</td>
                    <td style={{ color: stale ? 'var(--text-muted)' : undefined }} title={stale ? 'evaluation was run at a different step than this checkpoint (older run); not eligible for best' : ''}>{fmtPct(c.success_rate)}{stale ? '*' : ''}</td>
                    <td>{fmtPct(c.placement_rate)}</td>
                    <td>{fmt(c.mean_reward, 1)}</td>
                    <td style={{ color: 'var(--text-muted)' }}>{c.eval_episodes != null ? `${c.eval_episodes} ep · ${fmtInt(c.eval_at_timesteps)}` : 'N/A'}</td>
                    <td>{fmt(c.size_mb, 1)}</td>
                    <td className="text-right"><Btn onClick={() => load(c.name)} disabled={loaded}>{loaded ? 'loaded' : 'Load'}</Btn></td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
      <div className="text-[10px] mt-1" style={{ color: 'var(--text-muted)' }}>Eval columns come from the trainer's periodic evaluation (clean env, home start, deterministic) run at that checkpoint's step; * = eval from a different step (not eligible for best).</div>
    </Card>
  )
}
