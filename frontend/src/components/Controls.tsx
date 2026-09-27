import { useEffect, useState } from 'react'
import { get, post, type SimState } from '../api'
import { Btn } from './ui'

// Only controls that change what happens in the scene: reset/randomize, train, run/stop the policy, pause/step, speed, scenarios.
export function Controls({ state, trainingRunning, onStartTraining, onStopTraining, flash }: {
  state: SimState | null; trainingRunning: boolean; onStartTraining: () => void; onStopTraining: () => void; flash: (m: string, bad?: boolean) => void
}) {
  const [level, setLevel] = useState(1.0)
  const run = async (label: string, p: Promise<any>) => { try { await p; flash(label) } catch (e: any) { flash(`${label} failed: ${e.message}`, true) } }
  const mode = state?.mode
  const hasPolicy = !!state?.policy
  const builtin: [string, string][] = [['nominal', 'Nominal'], ['bracket_rot20', 'Bracket 20°'], ['bracket_rot75', 'Bracket 75°'], ['bolt_moved', 'Bolt moved'], ['both_moved', 'Both moved']]
  const [custom, setCustom] = useState<string[]>([])
  useEffect(() => { const load = () => get('/simulation/scenarios').then((d) => setCustom(d.custom ?? [])).catch(() => {}); load(); const t = setInterval(load, 8000); return () => clearInterval(t) }, [])
  const scenarios: [string, string][] = [...builtin, ...custom.map((c): [string, string] => [c, `★ ${c}`])]
  const stopPath = mode === 'teleop' ? '/teleop/stop' : mode === 'replay' ? '/replay/stop' : '/inference/stop'
  return (
    <div className="card p-2 flex flex-col gap-1.5">
      <div className="flex flex-wrap items-center gap-1.5">
        <Btn onClick={() => run('Scene reset', post('/simulation/reset', { randomize: true, randomization_level: level }))}>Reset</Btn>
        <Btn onClick={() => run('Scene randomized', post('/simulation/randomize', { randomization_level: level }))}>Randomize</Btn>
        <span className="flex items-center gap-1 ml-1" title="How much object positions, sizes, friction, bins, lighting vary at each reset (0 = fixed layout, 1 = full)">
          <span className="stat-label">Variation</span>
          <input type="range" min={0} max={1} step={0.1} value={level} onChange={(e) => setLevel(Number(e.target.value))} style={{ width: 70 }} />
          <span className="mono text-[10px]" style={{ width: 24 }}>{level.toFixed(1)}</span>
        </span>
        <span className="w-px h-5 mx-1" style={{ background: 'var(--line-strong)' }} />
        <Btn tone="accent" active={mode === 'inference'} disabled={!hasPolicy} title={hasPolicy ? 'Let the trained policy sort the parts' : 'Load a checkpoint first'}
          onClick={() => run('Policy running', post('/inference/start', {}))}>▶ Run Policy</Btn>
        <Btn onClick={() => run('Stopped', post(stopPath))} disabled={mode === 'idle'}>■ Stop</Btn>
        <Btn active={!!state?.paused} onClick={() => run(state?.paused ? 'Resumed' : 'Paused', post(state?.paused ? '/simulation/resume' : '/simulation/pause'))}>{state?.paused ? 'Resume' : 'Pause'}</Btn>
        <Btn onClick={() => run('Step', post('/simulation/step', { steps: 1 }))} title="Advance one control step">Step</Btn>
        <span className="w-px h-5 mx-1" style={{ background: 'var(--line-strong)' }} />
        <span className="stat-label mr-1">Speed</span>
        {[0.5, 1, 2, 5].map((sp) => <Btn key={sp} active={state?.speed === sp} onClick={() => run(`Speed ${sp}x`, post('/simulation/speed', { speed: sp }))}>{sp}x</Btn>)}
        <span className="w-px h-5 mx-1" style={{ background: 'var(--line-strong)' }} />
        <Btn tone="accent" onClick={onStartTraining} disabled={trainingRunning}>Train</Btn>
        <Btn tone="danger" onClick={onStopTraining} disabled={!trainingRunning}>Stop Train</Btn>
      </div>
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="stat-label mr-1" title="Preset layouts (bracket and bolt at fixed positions)">Scenario</span>
        {scenarios.map(([k, label]) => <Btn key={k} onClick={() => run(`Scenario: ${label}`, post(`/simulation/scenario/${k}`))}>{label}</Btn>)}
      </div>
    </div>
  )
}
