import { useEffect, useRef, useState } from 'react'
import { WS_BASE, del, fmt, fmtInt, post, type DemoCatalogue, type SimState } from '../api'
import { usePolling } from '../hooks'
import { Badge, Btn, Card, Stat, StatGrid } from './ui'
import { teleopBus } from '../teleopBus'

// Human-in-the-loop teleoperation (Section 43). Keys are composed into the SAME 7-d action the policy emits
// (dX dY dZ dRx dRy dRz gripper in [-1, 1]) and sent over /ws/teleop at 20 Hz; the backend applies them through
// the same IK / joint / velocity / workspace limits. Recording writes LeRobot episodes tagged episode_type=teleop_demo.
const KEY_AXES: Record<string, [number, number]> = {
  w: [0, 1], s: [0, -1], a: [1, 1], d: [1, -1], q: [2, 1], e: [2, -1],
  i: [3, 1], k: [3, -1], arrowup: [3, 1], arrowdown: [3, -1],
  u: [4, 1], o: [4, -1],
  j: [5, -1], l: [5, 1], arrowleft: [5, -1], arrowright: [5, 1],
}
const AXIS_NAMES = ['dX', 'dY', 'dZ', 'dRx', 'dRy', 'dRz']
const TYPE_COLORS: Record<string, string> = { teleop_demo: 'var(--series-3)', assisted_demo: '#b98cff', scripted_demo: 'var(--series-4)', policy_rollout: 'var(--series-1)' }
const TYPE_LABEL: Record<string, string> = { teleop_demo: 'teleop (human)', assisted_demo: 'assisted (human click)', scripted_demo: 'scripted (batch)', policy_rollout: 'policy', random_policy: 'random', manual: 'manual' }

export function TeleopPanel({ state, trainingRunning, flash }: { state: SimState | null; trainingRunning: boolean; flash: (m: string, bad?: boolean) => void }) {
  const active = state?.mode === 'teleop'
  const recording = !!state?.recording?.recording
  const pressed = useRef<Set<string>>(new Set())
  const gripper = useRef<number>(-1)
  const ws = useRef<WebSocket | null>(null)
  const [localAxes, setLocalAxes] = useState<number[]>([0, 0, 0, 0, 0, 0])
  const [scriptedN, setScriptedN] = useState(10)
  const [noise, setNoise] = useState(0.03)
  const [wsOk, setWsOk] = useState(false)
  const demos = usePolling<DemoCatalogue>('/demos', 3000, [recording, state?.mode])
  const act = async (label: string, p: Promise<any>) => { try { await p; flash(label) } catch (e: any) { flash(`${label} failed: ${e.message}`, true) } }
  const replay = state?.replay
  const replaying = state?.mode === 'replay'
  const assist = state?.assist
  const startReplay = (name: string) => act(`Replaying ${name}`, post(`/demos/${encodeURIComponent(name)}/replay`))
  const pickByName = (o: string) => act(`Assisted pick: ${o}`, post('/teleop/pick', { object: o }))
  const cancelAssist = () => act('Assist cancelled', post('/teleop/assist/cancel'))

  const start = () => act('Teleop active — keyboard focus on the page', post('/teleop/start', { reset: true }))
  const stop = () => act('Teleop stopped', post('/teleop/stop'))
  const toggleRec = () => act(recording ? 'Recording stopped' : 'Recording teleop demo', post(recording ? '/dataset/record/stop' : '/dataset/record/start', { episode_type: 'teleop_demo', reset: true }))
  const newEpisode = () => act('New episode', post('/simulation/reset', { randomize: true }))
  const scripted = () => act(`Generating ${scriptedN} scripted demos`, post('/demos/scripted', { episodes: scriptedN, noise_std: noise, record: true }))
  const discard = (name: string) => { if (!window.confirm(`Discard demonstration ${name}? This deletes its recorded episode.`)) return; act(`Discarded ${name}`, del(`/demos/${encodeURIComponent(name)}`)).then(() => demos.refresh()) }

  // keyboard + websocket only while teleop is the active controller
  useEffect(() => {
    if (!active) { pressed.current.clear(); setLocalAxes([0, 0, 0, 0, 0, 0]); ws.current?.close(); ws.current = null; return }
    gripper.current = state?.teleop?.gripper_closed ? 1 : -1
    const isTyping = (e: KeyboardEvent) => ['INPUT', 'TEXTAREA', 'SELECT'].includes((e.target as HTMLElement)?.tagName)
    const down = (e: KeyboardEvent) => {
      if (isTyping(e)) return
      const k = e.key.toLowerCase()
      if (k === 'escape') { post('/teleop/assist/cancel').catch(() => {}); return }
      if (k === ' ') { e.preventDefault(); gripper.current = gripper.current > 0 ? -1 : 1; send(); return }
      if (k === 'r') { toggleRec(); return }
      if (k === 'n') { newEpisode(); return }
      if (k in KEY_AXES || k === 'shift') { e.preventDefault(); pressed.current.add(k); send() }
    }
    const up = (e: KeyboardEvent) => { const k = e.key.toLowerCase(); if (pressed.current.delete(k)) send() }
    const compose = (): number[] => {
      const axes = [0, 0, 0, 0, 0, 0]
      const scale = pressed.current.has('shift') ? 0.3 : 1.0
      for (const k of pressed.current) { const m = KEY_AXES[k]; if (m) axes[m[0]] += m[1] }
      // mouse drag on the viewport -> table-plane motion in the viewport camera's screen axes
      const cam = state?.camera?.viewport
      if (teleopBus.drag.active && cam) {
        const r = cam.right, f = cam.forward
        axes[0] += teleopBus.drag.dx * r[0] + teleopBus.drag.dy * f[0]
        axes[1] += teleopBus.drag.dx * r[1] + teleopBus.drag.dy * f[1]
      }
      return axes.map((v) => Math.max(-1, Math.min(1, v * scale)))
    }
    const send = () => {
      const axes = compose()
      setLocalAxes(axes)
      if (ws.current?.readyState === WebSocket.OPEN) ws.current.send(JSON.stringify({ type: 'action', axes, gripper: gripper.current }))
    }
    const connect = () => {
      const sock = new WebSocket(`${WS_BASE}/ws/teleop`)
      sock.onopen = () => setWsOk(true)
      sock.onclose = () => { setWsOk(false); if (ws.current === sock) setTimeout(() => { if (ws.current === sock) connect() }, 1000) }
      ws.current = sock
    }
    connect()
    window.addEventListener('keydown', down); window.addEventListener('keyup', up)
    const timer = window.setInterval(send, 50) // 20 Hz keep-alive; the backend stops motion 0.5 s after the last message
    return () => { window.removeEventListener('keydown', down); window.removeEventListener('keyup', up); clearInterval(timer); const s = ws.current; ws.current = null; s?.close() }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active])

  const canStart = !trainingRunning && !!state && ['idle', 'teleop'].includes(state.mode)
  const list = demos.data?.demos ?? []
  const byType = demos.data?.summary.by_type ?? {}
  return (
    <div className="flex flex-col gap-2">
      <Card title="Teleoperation" right={<Badge tone={active ? 'good' : 'muted'} pulse={active}>{active ? (wsOk ? 'ACTIVE · WS OK' : 'ACTIVE · WS…') : 'OFF'}</Badge>}>
        <div className="flex flex-wrap gap-1.5 mb-2">
          <Btn tone="accent" onClick={start} disabled={!canStart || active} title={trainingRunning ? 'Stop training first' : (!canStart ? 'Stop inference/evaluation first' : '')}>Start Teleop</Btn>
          <Btn onClick={stop} disabled={!active}>Stop Teleop</Btn>
          <Btn tone={recording ? 'danger' : ''} active={recording} onClick={toggleRec} disabled={!active && !recording}>{recording ? '■ Stop Recording' : '● Start Recording'}</Btn>
          <Btn onClick={newEpisode} disabled={!active}>New episode (N)</Btn>
        </div>
        {recording && <div className="mono text-[11px] mb-2" style={{ color: 'var(--status-bad)' }}><span className="dot dot-pulse" style={{ background: 'var(--status-bad)', display: 'inline-block', marginRight: 6 }} />● Recording {state?.recording?.episode_type} · episode {(state?.recording?.episodes ?? 0) + 1} · {state?.recording?.frames_in_episode ?? 0} frames</div>}
        {trainingRunning && <div className="text-[10px] mb-2" style={{ color: 'var(--status-warn)' }}>Training is running — the arm has one controller at a time. Stop training to teleoperate.</div>}
        <StatGrid cols={3}>
          <Stat label="Gripper" value={state?.teleop?.gripper_closed ? 'CLOSED' : 'OPEN'} />
          <Stat label="Input" value={active ? (state?.teleop?.input_fresh ? 'live' : 'idle (dead-man)') : 'N/A'} mono={false} />
          <Stat label="Episode" value={state ? `${state.episode} · step ${state.step}/${state.max_steps}` : 'N/A'} />
          <Stat label="Target" value={state?.target ? `${state.target} → ${state.target_bin}` : 'none'} />
          <Stat label="Grasp" value={state ? (state.grasped ? (state.lifted ? 'holding · lifted' : 'holding') : 'open') : 'N/A'} mono={false} />
          <Stat label="Placed" value={state ? `${state.placed_count}/2` : 'N/A'} />
        </StatGrid>
        <div className="mt-2">
          {AXIS_NAMES.map((n, i) => {
            const v = active ? (state?.teleop?.axes?.[i] ?? localAxes[i]) : 0
            return (
              <div key={n} className="flex items-center gap-1 leading-4 text-[10px]">
                <span className="mono" style={{ width: 30, color: 'var(--text-secondary)' }}>{n}</span>
                <div className="flex-1 relative h-2" style={{ background: 'var(--surface-3)' }}>
                  <div className="absolute top-0 bottom-0" style={{ left: '50%', width: 1, background: 'var(--line-strong)' }} />
                  <div className="absolute top-0 bottom-0" style={{ left: v < 0 ? `${50 + v * 50}%` : '50%', width: `${Math.abs(v) * 50}%`, background: 'var(--series-3)' }} />
                </div>
                <span className="mono" style={{ width: 36, textAlign: 'right' }}>{fmt(v, 2)}</span>
              </div>
            )
          })}
        </div>
        <div className="mono text-[10px] mt-2 grid grid-cols-2 gap-x-3" style={{ color: 'var(--text-secondary)' }}>
          <div>W / S · ±X</div><div>A / D · ±Y</div><div>Q / E · Z up / down</div><div>J / L (← →) · yaw</div>
          <div>I / K, U / O · dRx, dRy (locked = 0)</div><div>Space · gripper</div><div>Shift · fine (30 %)</div><div>R · record · N · new episode</div>
          <div className="col-span-2">Mouse drag on the viewport · X/Y motion (screen axes)</div>
        </div>
        <div className="text-[10px] mt-2" style={{ color: 'var(--text-muted)' }}>Same action space, IK controller and joint/velocity/workspace limits as the PPO policy; each recorded step stores (observation.images.overhead, observation.state, action, reward, done).</div>
      </Card>

      <Card title="Click-to-pick (assisted teleop)" right={<Badge tone={assist?.active ? 'good' : 'muted'} pulse={!!assist?.active}>{assist?.active ? 'EXECUTING' : 'IDLE'}</Badge>}>
        <div className="text-[10px] mb-2" style={{ color: 'var(--text-secondary)' }}>In Teleop, click a part in the viewport (segmentation pass on the backend resolves the pixel to a body); the <span className="mono">scripted_demo</span> controller then executes approach → grasp → lift → transport → place into the part's bin within the policy's limits. Click a bin during execution to re-route. Recorded episodes become <span className="mono" style={{ color: '#b98cff' }}>assisted_demo</span>.</div>
        <div className="flex flex-wrap gap-1.5 items-center">
          {(state ? Object.keys(state.objects) : []).map((o) => <Btn key={o} onClick={() => pickByName(o)} disabled={!active || !!assist?.active || state?.objects[o]?.placed}>Pick {o}</Btn>)}
          <Btn tone="danger" onClick={cancelAssist} disabled={!assist?.active}>Cancel (Esc)</Btn>
          {assist && <span className="mono text-[10px]" style={{ color: assist.active ? '#d9c8ff' : 'var(--text-muted)' }}>{assist.object} → {assist.bin} · {assist.active ? `phase ${assist.phase}` : `done: ${assist.result}`}</span>}
        </div>
      </Card>

      <Card title="Episode replay (open-loop, deterministic)" right={<Badge tone={replaying ? 'warn' : 'muted'} pulse={!!replay?.playing}>{replaying ? (replay?.playing ? 'PLAYING' : 'PAUSED') : 'OFF'}</Badge>}>
        {!replay ? <div className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Press ▶ on a demonstration below: the sim resets to that episode's stored initial state and re-applies its recorded actions step by step in the main viewport (REPLAY indicator). Speed buttons apply.</div> : (
          <>
            <div className="mono text-[10px] mb-1" style={{ color: 'var(--text-secondary)' }}>{replay.name} · <span style={{ color: TYPE_COLORS[replay.episode_type] }}>{replay.episode_type}</span> · step {replay.index}/{replay.n} · {replay.exact_initial_state ? 'exact snapshot' : 'seed-reconstructed'} · state deviation {fmt(replay.max_state_deviation, 4)} · reward deviation {fmt(replay.max_reward_deviation, 4)}</div>
            <input type="range" min={0} max={replay.n} value={replay.index} style={{ width: '100%' }} onChange={(e) => post('/replay/seek', { index: Number(e.target.value) }).catch(() => {})} />
            <div className="flex gap-1.5 mt-1 items-center">
              <Btn onClick={() => post(replay.playing ? '/replay/pause' : '/replay/play').catch(() => {})}>{replay.playing ? '❚❚ Pause' : '▶ Play'}</Btn>
              <Btn onClick={() => post('/replay/seek', { index: 0 }).catch(() => {})}>⟲ Start</Btn>
              <Btn onClick={() => post('/replay/seek', { index: Math.max(0, replay.index - 1) }).catch(() => {})}>−1</Btn>
              <Btn onClick={() => post('/replay/seek', { index: Math.min(replay.n, replay.index + 1) }).catch(() => {})}>+1</Btn>
              <Btn tone="danger" onClick={() => act('Replay stopped', post('/replay/stop'))}>Stop Replay</Btn>
              {replay.recorded_outcome && <span className="mono text-[10px]" style={{ color: 'var(--text-muted)' }}>recorded outcome: {replay.recorded_outcome.success ? 'success' : `${replay.recorded_outcome.placed_count}/2 placed`} · R {fmt(replay.recorded_outcome.reward, 1)}</span>}
            </div>
          </>
        )}
      </Card>

      <Card title="Scripted demos (pipeline check — not human, not the policy)">
        <div className="flex items-end gap-2 flex-wrap">
          <label className="text-[10px]" style={{ width: 70 }}><div className="stat-label">Episodes</div><input type="number" min={1} max={200} value={scriptedN} onChange={(e) => setScriptedN(Number(e.target.value))} /></label>
          <label className="text-[10px]" style={{ width: 80 }}><div className="stat-label">Motion noise</div><input type="number" min={0} max={0.5} step={0.01} value={noise} onChange={(e) => setNoise(Number(e.target.value))} /></label>
          <Btn onClick={scripted} disabled={trainingRunning || (state?.mode !== 'idle' && state?.mode !== 'teleop')}>Generate scripted demos</Btn>
          {state?.mode === 'scripted_demo' && <span className="mono text-[10px]" style={{ color: 'var(--status-warn)' }}>running · {state.scripted?.remaining} left · phase {state.scripted?.phase}</span>}
        </div>
        <div className="text-[10px] mt-1" style={{ color: 'var(--text-muted)' }}>Hand-coded controller with ground-truth object/bin poses; episodes are tagged <span className="mono">scripted_demo</span> and never enter any policy success metric.</div>
      </Card>

      <Card title={`Demonstrations (${demos.data?.summary.total_episodes ?? 0} episodes · ${fmtInt(demos.data?.summary.total_frames)} frames)`} right={<Btn onClick={() => demos.refresh()}>Refresh</Btn>}>
        <div className="mono text-[10px] mb-1" style={{ color: 'var(--text-secondary)' }}>
          {Object.entries(byType).map(([t, b]) => <span key={t} className="mr-3" style={{ color: TYPE_COLORS[t] }}>{TYPE_LABEL[t] ?? t}: {b.episodes} ep · {b.successes} success · {b.frames} frames</span>)}
          {Object.keys(byType).length === 0 && 'none recorded yet'}
        </div>
        {list.length > 0 && (
          <div className="scroll" style={{ maxHeight: 260, overflow: 'auto' }}>
            <table className="w-full text-[10px] mono">
              <thead style={{ color: 'var(--text-muted)' }}><tr className="text-left"><th>type</th><th>time</th><th>outcome</th><th>placed</th><th>R</th><th>dur</th><th></th><th></th></tr></thead>
              <tbody style={{ color: 'var(--text-secondary)' }}>
                {list.slice().reverse().map((d) => (
                  <tr key={d.name} style={{ borderTop: '1px solid var(--line)' }} title={d.name}>
                    <td style={{ color: TYPE_COLORS[d.episode_type] }} title={TYPE_LABEL[d.episode_type] ?? d.episode_type}>{d.episode_type.replace('_demo', '')}</td>
                    <td>{d.timestamp?.slice(5, 16).replace('T', ' ')}</td>
                    <td style={{ color: d.outcome.success ? 'var(--status-good)' : 'var(--status-bad)' }}>{d.aborted ? 'aborted' : d.outcome.success ? 'success' : d.outcome.dropped ? 'dropped' : d.outcome.wrong_bin ? 'wrong bin' : 'incomplete'}</td>
                    <td>{d.outcome.placed_count}/2</td><td>{fmt(d.outcome.reward, 0)}</td><td>{fmt(d.duration_s, 1)}s</td>
                    <td><Btn onClick={() => startReplay(d.name)} disabled={trainingRunning || (state?.mode !== 'idle' && state?.mode !== 'teleop' && state?.mode !== 'replay')} title="Replay this episode open-loop in the viewport">▶</Btn></td>
                    <td className="text-right"><Btn tone="danger" onClick={() => discard(d.name)}>Discard</Btn></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  )
}
