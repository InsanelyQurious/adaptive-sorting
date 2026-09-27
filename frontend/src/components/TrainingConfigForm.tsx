import { useEffect, useState } from 'react'
import { get, post } from '../api'
import { Btn, Card } from './ui'

// Configurable from the UI (Section 16); defaults come from configs/training.yaml via /config.
export function TrainingConfigForm({ onClose, flash }: { onClose: () => void; flash: (m: string, bad?: boolean) => void }) {
  const [cfg, setCfg] = useState<any>({ total_timesteps: 3000000, learning_rate: 0.0003, batch_size: 768, gamma: 0.99, ent_coef: 0.003, seed: 42, n_envs: 12, randomization_level: 1.0, training_episodes: '' })
  const [resume, setResume] = useState<string>('')
  const [cks, setCks] = useState<string[]>([])
  const [initDemos, setInitDemos] = useState(false)
  const [bcEpochs, setBcEpochs] = useState(40)
  const [demoSummary, setDemoSummary] = useState<any>(null)
  useEffect(() => {
    get('/config').then((c) => setCfg((prev: any) => ({ ...prev, total_timesteps: c.training.total_timesteps, learning_rate: c.training.learning_rate, batch_size: c.training.batch_size, gamma: c.training.gamma, ent_coef: c.training.ent_coef, seed: c.training.seed, n_envs: c.training.n_envs, randomization_level: c.environment.randomization.level }))).catch(() => {})
    get('/checkpoints').then((d) => setCks(d.checkpoints.map((c: any) => c.name))).catch(() => {})
    get('/demos').then((d) => setDemoSummary(d.summary)).catch(() => {})
  }, [])
  const demoEpisodes: number = demoSummary?.total_episodes ?? 0
  const demoTypes: string[] = Object.keys(demoSummary?.by_type ?? {})
  const set = (k: string) => (e: any) => setCfg({ ...cfg, [k]: e.target.value })
  const start = async () => {
    const body: any = { total_timesteps: Number(cfg.total_timesteps), learning_rate: Number(cfg.learning_rate), batch_size: Number(cfg.batch_size), gamma: Number(cfg.gamma), ent_coef: Number(cfg.ent_coef), seed: Number(cfg.seed), n_envs: Number(cfg.n_envs), randomization_level: Number(cfg.randomization_level) }
    if (cfg.training_episodes) body.training_episodes = Number(cfg.training_episodes)
    if (resume) body.resume_from = resume
    if (initDemos) { body.init_from_demos = true; body.bc_epochs = Number(bcEpochs) }
    try { await post('/training/start', body); flash('Training started'); onClose() } catch (e: any) { flash(`Start failed: ${e.message}`, true) }
  }
  const fields: [string, string, number | undefined][] = [['total_timesteps', 'Training steps', 1000], ['training_episodes', 'Episode cap (optional)', 1], ['learning_rate', 'Learning rate', 0.00001], ['batch_size', 'Batch size', 8], ['gamma', 'Discount γ', 0.001], ['ent_coef', 'Entropy coef', 0.0001], ['seed', 'Seed', 1], ['n_envs', 'Parallel envs', 1], ['randomization_level', 'Randomization level', 0.1]]
  return (
    <Card title="Start training run" right={<Btn onClick={onClose}>✕</Btn>}>
      <div className="grid grid-cols-3 gap-2">
        {fields.map(([k, label, step]) => (
          <label key={k} className="text-[10px]"><div className="stat-label">{label}</div><input type="number" step={step} value={cfg[k]} onChange={set(k)} /></label>
        ))}
        <label className="text-[10px] col-span-3 flex items-center gap-2" title="Behaviour-cloning pretraining of the PPO network on recorded demonstrations (Teleop tab), then PPO fine-tuning with decaying BC auxiliary steps">
          <input type="checkbox" checked={initDemos} disabled={demoEpisodes === 0} onChange={(e) => setInitDemos(e.target.checked)} style={{ width: 'auto' }} />
          <span>Initialize from demonstrations {demoEpisodes > 0 ? `(${demoEpisodes} episodes: ${demoTypes.join(', ')})` : '(none recorded — see Teleop tab)'}</span>
          {initDemos && <span className="flex items-center gap-1"><span className="stat-label">BC epochs</span><input type="number" min={1} max={500} value={bcEpochs} onChange={(e) => setBcEpochs(Number(e.target.value))} style={{ width: 60 }} /></span>}
        </label>
        <label className="text-[10px] col-span-3"><div className="stat-label">Resume from checkpoint</div>
          <select value={resume} onChange={(e) => setResume(e.target.value)}><option value="">— fresh run —</option><option value="latest">latest</option>{cks.map((c) => <option key={c} value={c}>{c}</option>)}</select></label>
      </div>
      <div className="flex gap-2 mt-2"><Btn tone="accent" onClick={start}>Start</Btn><Btn onClick={onClose}>Cancel</Btn></div>
      <div className="text-[10px] mt-2" style={{ color: 'var(--text-muted)' }}>Training runs as a separate persistent process; closing this browser does not affect it. Other hyperparameters come from configs/training.yaml.</div>
    </Card>
  )
}
