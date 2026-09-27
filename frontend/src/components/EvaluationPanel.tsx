import { useState } from 'react'
import { BASE, fmt, fmtInt, fmtPct, post, type SimState } from '../api'
import { usePolling } from '../hooks'
import { Btn, Card, Stat, StatGrid, Bar, Badge } from './ui'
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis, CartesianGrid } from 'recharts'

// Every number here comes from real evaluation rollouts on the live sim (batch runs persist to logs/*_latest.json).
export function EvaluationPanel({ state, flash }: { state: SimState | null; flash: (m: string, bad?: boolean) => void }) {
  const [episodes, setEpisodes] = useState(10)
  const [level, setLevel] = useState(1.0)
  const [scEpisodes, setScEpisodes] = useState(5)
  const [swEpisodes, setSwEpisodes] = useState(6)
  const [cmpA, setCmpA] = useState('')
  const [cmpB, setCmpB] = useState('')
  const [cmpEpisodes, setCmpEpisodes] = useState(20)
  const live = state?.evaluation
  const batch = state?.batch
  const { data, refresh } = usePolling<any>('/training/evaluation', 4000, [live?.running])
  const scorecard = usePolling<any>('/evaluation/scorecard', 5000, [batch?.kind])
  const sweep = usePolling<any>('/evaluation/sweep', 5000, [batch?.kind])
  const compare = usePolling<any>('/evaluation/compare', 5000, [batch?.kind])
  const reels = usePolling<any>('/evaluation/reels?limit=24', 8000, [live?.running, batch?.kind])
  const cks = usePolling<any>('/checkpoints', 10000)
  const last = live?.running ? null : (data?.last ?? live?.last ?? null)
  const run = async (label: string, p: Promise<any>) => { try { await p; flash(label); refresh() } catch (e: any) { flash(`${label} failed: ${e.message}`, true) } }
  const busy = !!live?.running || !!batch
  const sc = scorecard.data?.last; const sw = sweep.data?.last; const cmp = compare.data?.last
  const swRows = (sw?.results ?? []).map((r: any) => ({ level: r.item?.level ?? r.randomization_level, success: r.success_rate, placement: r.placement_rate, grasp: r.grasp_rate, reward: r.average_reward }))
  return (
    <div className="flex flex-col gap-2">
      <Card title="Run evaluation (randomized episodes, live sim)" right={batch ? <Badge tone="warn" pulse>{batch.kind} {batch.i + 1}/{batch.n} · {batch.current}</Badge> : undefined}>
        <div className="flex items-end gap-2 flex-wrap">
          <label className="text-[10px]" style={{ width: 70 }}><div className="stat-label">Episodes</div><input type="number" min={1} max={200} value={episodes} onChange={(e) => setEpisodes(Number(e.target.value))} /></label>
          <label className="text-[10px]" style={{ width: 90 }}><div className="stat-label">Rand level</div><input type="number" min={0} max={1} step={0.1} value={level} onChange={(e) => setLevel(Number(e.target.value))} /></label>
          <Btn tone="accent" onClick={() => run(`Evaluation started (${episodes} episodes)`, post('/training/evaluate', { episodes, randomization_level: level }))} disabled={!state?.policy || busy}>Run Evaluation</Btn>
          {busy && <Btn tone="danger" onClick={() => run('Aborted', post('/evaluation/batch/abort'))}>Abort</Btn>}
        </div>
        {live?.running && (
          <div className="mt-2">
            <div className="text-[10px] mono mb-1" style={{ color: 'var(--text-secondary)' }}>{live.label ? `${live.label} · ` : ''}running {live.completed}/{live.episodes} · success so far {fmtPct(live.success_rate)} · avg reward {fmt(live.average_reward, 1)}</div>
            <Bar value={(live.completed ?? 0) / Math.max(1, live.episodes ?? 1)} />
          </div>
        )}
      </Card>

      <Card title="Aggregate results (last completed evaluation)">
        {!last ? <div className="text-[11px]" style={{ color: 'var(--text-muted)' }}>No evaluation run yet.</div> : (
          <>
            <StatGrid cols={3}>
              <Stat label="Checkpoint" value={last.checkpoint ?? 'N/A'} />
              <Stat label="Episodes run" value={fmtInt(last.episodes_run)} />
              <Stat label="Successes" value={fmtInt(last.successes)} />
              <Stat label="Success rate" value={fmtPct(last.success_rate)} />
              <Stat label="Grasp rate" value={fmtPct(last.grasp_rate)} />
              <Stat label="Placement rate" value={fmtPct(last.placement_rate)} />
              <Stat label="Avg reward" value={fmt(last.average_reward, 2)} />
              <Stat label="Avg length" value={fmt(last.average_length, 1)} />
              <Stat label="Cycle time" value={last.parts_per_minute != null ? `${fmt(last.parts_per_minute, 2)} parts/min` : 'N/A'} title={`${last.parts_placed_total} parts placed in ${last.sim_seconds_total} s of simulated time`} />
              <Stat label="Collisions" value={last.collisions_total != null ? `${fmtInt(last.collisions_total)} steps · ${fmtInt(last.episodes_with_collision)}/${last.episodes_run} ep` : 'N/A'} title="control steps with arm/gripper body contact against table or bins; episodes with ≥1 such step" />
              <Stat label="Control effort" value={last.control_effort_mean != null ? `${fmt(last.control_effort_mean, 1)} Σ|a|²/ep` : 'N/A'} title="mean per-episode sum of squared normalised actions (the control_effort_penalty component divided by its weight)" />
              <Stat label="Drops / wrong bin / false picks / unstable" value={`${fmtInt(last.drops)} / ${fmtInt(last.wrong_bin)} / ${fmtInt(last.false_picks ?? 0)} / ${fmtInt(last.unstable_episodes ?? 0)}`} title="false pick = the gripper grasped the rejection-case part; unstable = episode ended because the physics diverged (counts as failure)" />
              <Stat label="Rand level" value={fmt(last.randomization_level, 1)} />
              <Stat label="Duration" value={fmt(last.duration_s, 0, ' s')} />
              <Stat label="Finished" value={last.finished_iso ?? 'N/A'} />
            </StatGrid>
            <div className="mono text-[10px] mt-2" style={{ color: last.collisions_total === 0 ? 'var(--status-good)' : 'var(--status-warn)' }}>
              safety: {last.collisions_total === 0 ? `0 collisions across ${last.episodes_run} evaluation episodes` : `${last.collisions_total} collision steps in ${last.episodes_with_collision} of ${last.episodes_run} episodes`} · collision penalty total {fmt(last.collision_penalty_total, 1)} · control-effort penalty total {fmt(last.control_effort_penalty_total, 1)}
            </div>
            {data?.last_episodes?.length > 0 && (
              <div className="scroll mt-2" style={{ maxHeight: 180, overflow: 'auto' }}>
                <table className="w-full text-[10px] mono">
                  <thead style={{ color: 'var(--text-muted)' }}><tr className="text-left"><th>ep</th><th>seed</th><th>succ</th><th>placed</th><th>grasp</th><th>R</th><th>len</th><th>coll</th><th>bracket (x,y,yaw°)</th><th>bolt (x,y,yaw°)</th></tr></thead>
                  <tbody style={{ color: 'var(--text-secondary)' }}>
                    {data.last_episodes.map((e: any) => {
                      const op = e.object_configuration?.object_poses ?? {}
                      const f = (o: any) => o ? `${fmt(o.x, 2)},${fmt(o.y, 2)},${fmt(o.yaw * 57.3, 0)}` : 'N/A'
                      return (
                        <tr key={e.episode} style={{ borderTop: '1px solid var(--line)' }}>
                          <td>{e.episode}</td><td>{e.seed}</td><td style={{ color: e.success ? 'var(--status-good)' : 'var(--status-bad)' }}>{e.success ? 'yes' : 'no'}</td>
                          <td>{e.placed_count}</td><td>{e.grasp_success ? 'yes' : 'no'}</td><td>{fmt(e.reward, 1)}</td><td>{e.length}</td><td>{e.collisions}</td>
                          <td>{f(op.bracket)}</td><td>{f(op.bolt)}</td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </>
        )}
      </Card>

      <Card title="Generalization scorecard (all scenarios)" right={<span className="flex items-center gap-1"><label className="text-[10px] flex items-center gap-1"><span className="stat-label">ep/scenario</span><input type="number" min={1} max={50} value={scEpisodes} onChange={(e) => setScEpisodes(Number(e.target.value))} style={{ width: 50 }} /></label><Btn tone="accent" onClick={() => run('Scorecard started', post('/evaluation/scorecard', { episodes: scEpisodes }))} disabled={!state?.policy || busy}>Run scorecard</Btn></span>}>
        {!sc && !(batch?.kind === 'scorecard') ? <div className="text-[11px]" style={{ color: 'var(--text-muted)' }}>Not run yet. Runs Evaluation on every preset (and saved custom) scenario back to back; the table persists.</div> : (
          <>
            {sc && <div className="mono text-[10px] mb-1" style={{ color: 'var(--text-secondary)' }}>checkpoint {sc.results?.[0]?.checkpoint} · {sc.episodes} episodes/scenario · finished {sc.finished_iso} · {fmt(sc.duration_s, 0)} s</div>}
            <table className="w-full text-[10px] mono">
              <thead style={{ color: 'var(--text-muted)' }}><tr className="text-left"><th>scenario</th><th>success</th><th>grasp</th><th>placement</th><th>avg R</th><th>coll</th><th>unstable</th><th>parts/min</th></tr></thead>
              <tbody style={{ color: 'var(--text-secondary)' }}>
                {((batch?.kind === 'scorecard' ? batch.results : sc?.results) ?? []).map((r: any) => (
                  <tr key={r.label} style={{ borderTop: '1px solid var(--line)' }}>
                    <td style={{ color: 'var(--text-primary)' }}>{r.label}</td>
                    <td style={{ color: (r.success_rate ?? 0) >= 0.5 ? 'var(--status-good)' : (r.success_rate ?? 0) > 0 ? 'var(--status-warn)' : 'var(--status-bad)' }}>{fmtPct(r.success_rate, 0)} ({r.successes}/{r.episodes_run})</td>
                    <td>{fmtPct(r.grasp_rate, 0)}</td><td>{fmtPct(r.placement_rate, 0)}</td><td>{fmt(r.average_reward, 1)}</td><td>{r.collisions_total}</td><td style={{ color: (r.unstable_episodes ?? 0) > 0 ? 'var(--status-bad)' : undefined }}>{r.unstable_episodes ?? 0}</td><td>{fmt(r.parts_per_minute, 2)}</td>
                  </tr>
                ))}
                {batch?.kind === 'scorecard' && <tr><td colSpan={8} className="pt-1" style={{ color: 'var(--status-warn)' }}>running {batch.current} ({batch.i + 1}/{batch.n})…</td></tr>}
              </tbody>
            </table>
          </>
        )}
      </Card>

      <Card title="Randomization difficulty sweep" right={<span className="flex items-center gap-1"><label className="text-[10px] flex items-center gap-1"><span className="stat-label">ep/level</span><input type="number" min={1} max={50} value={swEpisodes} onChange={(e) => setSwEpisodes(Number(e.target.value))} style={{ width: 50 }} /></label><Btn tone="accent" onClick={() => run('Difficulty sweep started', post('/evaluation/sweep', { episodes: swEpisodes }))} disabled={!state?.policy || busy}>Run sweep</Btn></span>}>
        {!sw ? <div className="text-[11px]" style={{ color: 'var(--text-muted)' }}>Not run yet. Evaluates at randomization levels 0, 0.25, 0.5, 0.75, 1.0 (positions, yaw, scale, friction, bins, noise, lighting all scale with the level).</div> : (
          <>
            <div className="mono text-[10px] mb-1" style={{ color: 'var(--text-secondary)' }}>checkpoint {sw.results?.[0]?.checkpoint} · {sw.episodes} episodes/level · finished {sw.finished_iso}</div>
            <div style={{ height: 140 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={swRows} margin={{ top: 4, right: 8, left: -18, bottom: 0 }}>
                  <CartesianGrid stroke="var(--line)" strokeDasharray="2 4" vertical={false} />
                  <XAxis dataKey="level" tick={{ fontSize: 9, fill: 'var(--text-muted)' }} stroke="var(--line-strong)" />
                  <YAxis domain={[0, 1]} tick={{ fontSize: 9, fill: 'var(--text-muted)' }} tickFormatter={(v) => fmtPct(v, 0)} stroke="var(--line-strong)" width={60} />
                  <Tooltip contentStyle={{ background: 'var(--surface-2)', border: '1px solid var(--line-strong)', fontSize: 10 }} formatter={(v: any, n: any) => [fmtPct(Number(v), 0), n]} labelFormatter={(v) => `randomization level ${v}`} />
                  <Line type="monotone" dataKey="success" name="success" stroke="var(--series-1)" strokeWidth={2} dot isAnimationActive={false} />
                  <Line type="monotone" dataKey="placement" name="placement" stroke="var(--series-3)" strokeWidth={1.5} dot isAnimationActive={false} />
                  <Line type="monotone" dataKey="grasp" name="grasp" stroke="var(--series-4)" strokeWidth={1.5} dot isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="mono text-[10px]" style={{ color: 'var(--text-secondary)' }}>{swRows.map((r: any) => `L${r.level}: ${fmtPct(r.success, 0)} succ / ${fmtPct(r.placement, 0)} place`).join(' · ')}</div>
          </>
        )}
      </Card>

      <Card title="Before / after — two checkpoints, identical conditions, fresh evaluation">
        <div className="flex items-end gap-2 flex-wrap">
          <label className="text-[10px]" style={{ minWidth: 150 }}><div className="stat-label">Checkpoint A (before)</div><select value={cmpA} onChange={(e) => setCmpA(e.target.value)}><option value="">— select —</option>{(cks.data?.checkpoints ?? []).map((c: any) => <option key={c.name} value={c.name}>{c.name}</option>)}<option value="best">best</option><option value="latest">latest</option></select></label>
          <label className="text-[10px]" style={{ minWidth: 150 }}><div className="stat-label">Checkpoint B (after)</div><select value={cmpB} onChange={(e) => setCmpB(e.target.value)}><option value="">— select —</option>{(cks.data?.checkpoints ?? []).map((c: any) => <option key={c.name} value={c.name}>{c.name}</option>)}<option value="best">best</option><option value="latest">latest</option></select></label>
          <label className="text-[10px]" style={{ width: 60 }}><div className="stat-label">Episodes</div><input type="number" min={1} max={100} value={cmpEpisodes} onChange={(e) => setCmpEpisodes(Number(e.target.value))} /></label>
          <Btn tone="accent" onClick={() => run('Comparison started', post('/evaluation/compare', { checkpoint_a: cmpA, checkpoint_b: cmpB, episodes: cmpEpisodes }))} disabled={!cmpA || !cmpB || busy}>Compare</Btn>
        </div>
        <div className="text-[10px] mt-1" style={{ color: 'var(--text-muted)' }}>Archived checkpoints can be typed as an absolute path in the chat panel ("compare &lt;A&gt; with &lt;B&gt;"). Same seeds (4242+i), level 1.0, deterministic policy; the previously loaded checkpoint is restored afterwards.</div>
        {(cmp || batch?.kind === 'compare') && (
          <table className="w-full text-[10px] mono mt-2">
            <thead style={{ color: 'var(--text-muted)' }}><tr className="text-left"><th>checkpoint</th><th>success</th><th>grasp</th><th>placement</th><th>avg R</th><th>parts/min</th><th>collisions</th></tr></thead>
            <tbody style={{ color: 'var(--text-secondary)' }}>
              {((batch?.kind === 'compare' ? batch.results : cmp?.results) ?? []).map((r: any, i: number) => (
                <tr key={i} style={{ borderTop: '1px solid var(--line)' }}><td style={{ color: 'var(--text-primary)' }}>{i === 0 ? 'A · ' : 'B · '}{r.checkpoint}</td><td>{fmtPct(r.success_rate, 0)} ({r.successes}/{r.episodes_run})</td><td>{fmtPct(r.grasp_rate, 0)}</td><td>{fmtPct(r.placement_rate, 0)}</td><td>{fmt(r.average_reward, 1)}</td><td>{fmt(r.parts_per_minute, 2)}</td><td>{r.collisions_total}</td></tr>
              ))}
            </tbody>
          </table>
        )}
        {cmp && <div className="mono text-[10px] mt-1" style={{ color: 'var(--text-muted)' }}>{cmp.episodes} episodes each · seed {cmp.seed} · level {cmp.randomization_level} · finished {cmp.finished_iso}</div>}
      </Card>

      <Card title={`Failure reel (${(reels.data?.reels ?? []).filter((r: any) => !r.success).length} failed episodes saved)`} right={<Btn onClick={() => reels.refresh()}>Refresh</Btn>}>
        {(reels.data?.reels ?? []).length === 0 ? <div className="text-[11px]" style={{ color: 'var(--text-muted)' }}>Every failed evaluation episode (and one success reference per evaluation) is saved with a 12-frame filmstrip and a per-step trace under logs/eval_reels/.</div> : (
          <div className="scroll flex flex-col gap-2" style={{ maxHeight: 420, overflow: 'auto' }}>
            {(reels.data.reels as any[]).map((r) => (
              <div key={r.id} style={{ borderTop: '1px solid var(--line)', paddingTop: 4 }}>
                <div className="mono text-[10px] flex justify-between" style={{ color: r.success ? 'var(--status-good)' : 'var(--status-bad)' }}>
                  <span>{r.success ? 'SUCCESS (reference)' : `FAIL · ${r.failure_reason}`} · {r.label ? `${r.label} · ` : ''}seed {r.seed} · placed {r.placed_count}/2 · R {fmt(r.reward, 1)} · len {r.length}{r.collisions ? ` · ${r.collisions} coll` : ''}</span>
                  <a href={`${BASE}/evaluation/reels/${r.id}/trace.json`} target="_blank" rel="noreferrer" style={{ color: 'var(--accent)' }}>trace</a>
                </div>
                <a href={`${BASE}/evaluation/reels/${r.id}/filmstrip.jpg`} target="_blank" rel="noreferrer"><img src={`${BASE}/evaluation/reels/${r.id}/filmstrip.jpg`} alt="filmstrip" style={{ width: '100%', display: 'block', border: '1px solid var(--line)' }} /></a>
                <div className="mono text-[9px]" style={{ color: 'var(--text-muted)' }}>{r.id} · frames at steps {(r.frame_steps ?? []).join(', ')} · {r.saved_iso}</div>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  )
}
