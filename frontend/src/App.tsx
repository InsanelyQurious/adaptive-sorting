import { useEffect, useState } from 'react'
import { BASE, get, post } from './api'
import { useSimulationSocket, useTrainingSocket } from './hooks'
import { Viewport } from './components/Viewport'
import { Controls } from './components/Controls'
import { PolicyFlow } from './components/PolicyFlow'
import { TrainingPanel } from './components/TrainingPanel'
import { InferencePanel } from './components/InferencePanel'
import { PolicyCard } from './components/PolicyCard'
import { CheckpointsPanel } from './components/CheckpointsPanel'
import { EvaluationPanel } from './components/EvaluationPanel'
import { TrainingConfigForm } from './components/TrainingConfigForm'
import { TeleopPanel } from './components/TeleopPanel'
import { ScenePanel } from './components/ScenePanel'
import { Badge, Tabs } from './components/ui'

const TABS = ['Teleop', 'Training', 'Policy', 'Checkpoints', 'Inference', 'Evaluation']

export default function App() {
  const { frame, state, connected } = useSimulationSocket()
  const { status, metrics, connected: trainConnected } = useTrainingSocket()
  const [tab, setTab] = useState('Teleop')
  const [sceneOpen, setSceneOpen] = useState(true)
  const [showTrainForm, setShowTrainForm] = useState(false)
  const [toast, setToast] = useState<{ m: string; bad: boolean } | null>(null)
  const [header, setHeader] = useState({ title: 'ADAPTIVE SORTING', subtitle: 'Autonomous Robotic Component Sorting', tag: 'MuJoCo × LeRobot' })
  const flash = (m: string, bad = false) => { setToast({ m, bad }); setTimeout(() => setToast(null), 3500) }
  useEffect(() => { get('/config').then((c) => setHeader({ title: c.header_title, subtitle: c.header_subtitle, tag: c.header_tag })).catch(() => {}) }, [])
  const trainingRunning = !!status && ['running', 'paused', 'evaluating', 'saving', 'starting', 'bc_pretraining'].includes(status.state) && status.process_alive !== false
  const stopTraining = async () => { if (!window.confirm('Stop the in-progress training run? A final checkpoint will be saved.')) return; try { await post('/training/stop'); flash('Training stop requested (checkpoint saved on exit)') } catch (e: any) { flash(e.message, true) } }
  const simLabel = state ? (state.mode === 'inference' ? 'POLICY RUNNING' : state.mode === 'evaluate' ? 'EVALUATING' : state.mode === 'teleop' ? 'TELEOP' : state.mode === 'replay' ? 'REPLAY' : state.mode === 'scripted_demo' ? 'SCRIPTED DEMO' : 'IDLE') : 'OFFLINE'
  return (
    <div className="h-full flex flex-col" style={{ padding: 8, gap: 8 }}>
      <header className="card flex items-center justify-between px-3 py-2">
        <div className="flex items-baseline gap-3">
          <div className="text-[15px] font-semibold tracking-[0.22em]">{header.title}</div>
          <div className="text-[11px] tracking-[0.12em] uppercase" style={{ color: 'var(--text-secondary)' }}>{header.subtitle}</div>
          <div className="text-[11px] tracking-[0.12em] uppercase" style={{ color: 'var(--text-muted)' }}>{header.tag}</div>
        </div>
        <div className="flex items-center gap-2">
          <Badge tone={connected ? 'good' : 'bad'} pulse={connected}>SIM {connected ? simLabel : 'OFFLINE'}</Badge>
          <Badge tone={trainConnected ? (trainingRunning ? 'good' : 'muted') : 'bad'} pulse={trainingRunning}>TRAIN {status?.state?.replace('_', ' ') ?? 'N/A'}</Badge>
          <Badge tone="info">{state?.device ?? 'N/A'}</Badge>
          <a className="btn" href={`${BASE}/export/summary.html`} target="_blank" rel="noreferrer" title="Download a static HTML summary of the current results">Export Summary</a>
        </div>
      </header>
      <div className="flex-1 min-h-0 flex" style={{ gap: 8 }}>
        <aside className="flex flex-col min-h-0" style={{ flex: sceneOpen ? '0 0 290px' : '0 0 28px', transition: 'flex-basis 150ms' }}>
          <div className="card flex items-center justify-between px-2 py-1 mb-1" style={{ cursor: 'pointer' }} onClick={() => setSceneOpen((v) => !v)} title={sceneOpen ? 'Collapse the parts catalog' : 'Expand the parts catalog'}>
            {sceneOpen ? <span className="card-title">Parts on the table</span> : <span className="card-title" style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)', padding: '6px 0' }}>Parts</span>}
            <span className="mono text-[10px]" style={{ color: 'var(--text-muted)' }}>{sceneOpen ? '◂' : '▸'}</span>
          </div>
          {sceneOpen && <div className="flex-1 min-h-0 overflow-auto scroll"><ScenePanel state={state} flash={flash} /></div>}
        </aside>
        <section className="flex flex-col min-w-0 min-h-0" style={{ flex: '1 1 auto', gap: 8 }}>
          <Viewport frame={frame} connected={connected} flash={flash} />
          <Controls state={state} trainingRunning={trainingRunning} onStartTraining={() => { setShowTrainForm(true); setTab('Training') }} onStopTraining={stopTraining} flash={flash} />
          <PolicyFlow frame={frame} />
        </section>
        <aside className="flex flex-col min-w-0 min-h-0" style={{ flex: '0 0 470px' }}>
          <Tabs tabs={TABS} value={tab} onChange={setTab} />
          <div className="flex-1 min-h-0 overflow-auto scroll pt-2 flex flex-col" style={{ gap: 8 }}>
            {showTrainForm && <TrainingConfigForm onClose={() => setShowTrainForm(false)} flash={flash} />}
            {tab === 'Training' && <TrainingPanel status={status} metrics={metrics} flash={flash} />}
            {tab === 'Inference' && <InferencePanel frame={frame} />}
            {tab === 'Teleop' && <TeleopPanel state={state} trainingRunning={trainingRunning} flash={flash} />}
            {tab === 'Policy' && <PolicyCard state={state} />}
            {tab === 'Checkpoints' && <CheckpointsPanel state={state} flash={flash} />}
            {tab === 'Evaluation' && <EvaluationPanel state={state} flash={flash} />}
          </div>
        </aside>
      </div>
      {toast && <div className="fixed bottom-3 left-3 card px-3 py-2 text-[11px] mono" style={{ borderColor: toast.bad ? 'var(--status-bad)' : 'var(--accent)', zIndex: 60 }}>{toast.m}</div>}
    </div>
  )
}
