import type { FrameMessage } from '../api'
import { fmt, fmtInt, fmtPct } from '../api'
import { Badge, Card, Stat, StatGrid } from './ui'

export function InferencePanel({ frame }: { frame: FrameMessage | null }) {
  const s = frame?.state
  const inf = s?.inference
  const running = s?.mode === 'inference' || s?.mode === 'evaluate'
  const tgt = s?.target
  const tgtObj = tgt && s ? s.objects[tgt] : null
  return (
    <div className="flex flex-col gap-2">
      <Card title="Inference" right={<Badge tone={running ? 'good' : 'muted'} pulse={running && !s?.paused}>{s?.mode ?? 'N/A'}{s?.paused ? ' · paused' : ''}</Badge>}>
        <StatGrid cols={3}>
          <Stat label="Policy" value={s?.policy?.name ?? 'N/A'} mono={false} />
          <Stat label="Checkpoint" value={s?.policy?.checkpoint ?? 'N/A'} />
          <Stat label="Device" value={s?.device ?? 'N/A'} />
          <Stat label="Inference FPS" value={running ? fmt(inf?.control_fps, 1) : 'N/A'} title="policy+physics control steps per second" />
          <Stat label="Action latency" value={inf?.action_latency_ms != null ? fmt(inf.action_latency_ms, 2, ' ms') : 'N/A'} />
          <Stat label="Episode" value={s ? `${s.episode} · step ${s.step}/${s.max_steps}` : 'N/A'} />
          <Stat label="Task" value="sort bracket → Bin A, bolt → Bin B" mono={false} />
          <Stat label="Target object" value={tgt ? `${tgt} → ${s?.target_bin}` : (s ? 'none (done)' : 'N/A')} />
          <Stat label="Current object pose" value={tgtObj ? `(${fmt(tgtObj.position[0], 2)}, ${fmt(tgtObj.position[1], 2)}) yaw ${fmt(tgtObj.yaw * 57.2958, 0)}°` : 'N/A'} />
          <Stat label="Grasp status" value={s ? (s.grasped ? (s.lifted ? 'holding · lifted' : 'holding') : 'open / none') : 'N/A'} mono={false} />
          <Stat label="Placement" value={s ? `${s.placed_count}/2 placed${s.objects && Object.values(s.objects).some((o) => o.in_bin && o.in_bin !== o.target_bin) ? ' · WRONG BIN' : ''}` : 'N/A'} />
          <Stat label="Success rate" value={inf ? `${fmtPct(inf.success_rate)} (${inf.successes}/${inf.episodes})` : 'N/A'} />
          <Stat label="Grasp / place rate" value={inf ? `${fmtPct(inf.grasp_rate, 0)} / ${fmtPct(inf.placement_rate, 0)}` : 'N/A'} />
          <Stat label="Current reward" value={fmt(s?.reward, 3)} />
          <Stat label="Episode reward" value={fmt(s?.episode_reward, 2)} />
          <Stat label="Cycle time" value={inf?.parts_per_minute != null ? `${fmt(inf.parts_per_minute, 2)} parts/min` : 'N/A'} title={inf ? `${inf.parts_placed} parts placed in ${inf.sim_seconds} s simulated (this session)` : ''} />
          <Stat label="Collisions" value={inf ? `${fmtInt(inf.collision_steps)} steps · ${fmtInt(inf.episodes_with_collision)}/${inf.episodes} ep` : 'N/A'} title="control steps with arm/gripper contact against table or bins, and episodes with at least one" />
          <Stat label="Control effort" value={inf?.control_effort_mean != null ? `${fmt(inf.control_effort_mean, 1)} Σ|a|²/ep` : 'N/A'} title={`this episode so far: ${fmt(inf?.episode_control_effort, 1)}`} />
        </StatGrid>
        {s?.last_episode && (
          <div className="mt-2 text-[10px] mono" style={{ color: 'var(--text-secondary)' }}>
            last episode #{s.last_episode.episode}: {s.last_episode.success ? 'SUCCESS' : 'failure'} · placed {s.last_episode.placed_count} · reward {fmt(s.last_episode.reward, 1)} · len {s.last_episode.length}{s.last_episode.dropped ? ' · dropped' : ''}{s.last_episode.wrong_bin ? ' · wrong bin' : ''}{s.last_episode.unstable ? ' · sim unstable' : ''}
          </div>
        )}
      </Card>
      <Card title="Scene configuration (this episode)">
        {s ? (
          <div className="mono text-[10px] grid grid-cols-2 gap-x-3 gap-y-0.5" style={{ color: 'var(--text-secondary)' }}>
            {Object.entries(s.objects).map(([n, o]) => (
              <div key={n}>{n}: ({fmt(o.position[0], 3)}, {fmt(o.position[1], 3)}, {fmt(o.position[2], 3)}) yaw {fmt(o.yaw * 57.2958, 0)}° {o.in_bin ? `· in ${o.in_bin}` : ''}</div>
            ))}
            {Object.entries(s.bins).map(([n, p]) => <div key={n}>{n}: ({fmt(p[0], 3)}, {fmt(p[1], 3)})</div>)}
            <div>scale: {Object.entries(s.episode_config?.object_scales ?? {}).map(([k, v]) => `${k} ${fmt(v as number, 2)}`).join(', ')}</div>
            <div>friction: {Object.entries(s.episode_config?.frictions ?? {}).map(([k, v]) => `${k} ${fmt(v as number, 2)}`).join(', ')}</div>
            <div>rand level: {fmt(s.episode_config?.level, 1)} · noise σ {fmt(s.episode_config?.state_noise_std, 3)}</div>
            <div>collisions: {s.collisions}</div>
          </div>
        ) : 'N/A'}
      </Card>
      <details className="card p-2">
        <summary className="card-title">Observation · state vector ({s?.state_vector?.length ?? 0})</summary>
        <div className="mono text-[10px] grid grid-cols-2 gap-x-3 mt-1 scroll" style={{ maxHeight: 220, overflow: 'auto', color: 'var(--text-secondary)' }}>
          {s?.state_vector?.map((v, i) => <div key={i} className="flex justify-between"><span>{s.state_layout[i]}</span><span style={{ color: 'var(--text-primary)' }}>{fmt(v, 3)}</span></div>)}
        </div>
      </details>
      <details className="card p-2">
        <summary className="card-title">Action ({s?.action.values.length ?? 0}) · reward components</summary>
        <div className="mono text-[10px] mt-1" style={{ color: 'var(--text-secondary)' }}>
          {s?.action.names.map((n, i) => <span key={n} className="inline-block mr-3">{n} <span style={{ color: 'var(--text-primary)' }}>{fmt(s.action.values[i], 3)}</span></span>)}
        </div>
        <div className="mono text-[10px] grid grid-cols-2 gap-x-3 mt-1" style={{ color: 'var(--text-secondary)' }}>
          {s && Object.entries(s.reward_components).filter(([, v]) => v !== 0).map(([k, v]) => <div key={k} className="flex justify-between"><span>{k}</span><span style={{ color: v >= 0 ? 'var(--text-primary)' : 'var(--status-bad)' }}>{fmt(v, 3)}</span></div>)}
        </div>
      </details>
      <details className="card p-2" open>
        <summary className="card-title">Joint state (rad, rad/s)</summary>
        <div className="mono text-[10px] mt-1 grid grid-cols-3 gap-x-2" style={{ color: 'var(--text-secondary)' }}>
          {s?.joint_positions.map((q, i) => <div key={i}>J{i + 1} <span style={{ color: 'var(--text-primary)' }}>{fmt(q, 3)}</span> <span style={{ color: 'var(--text-muted)' }}>{fmt(s.joint_velocities[i], 2)}</span></div>)}
        </div>
        <div className="mono text-[10px] mt-1" style={{ color: 'var(--text-secondary)' }}>TCP ({fmt(s?.ee_position[0], 3)}, {fmt(s?.ee_position[1], 3)}, {fmt(s?.ee_position[2], 3)}) yaw {fmt((s?.ee_yaw ?? NaN) * 57.2958, 0)}° · gripper {s ? `${(s.gripper_opening * 100).toFixed(0)}% open, cmd ${(s.gripper_cmd_closed * 100).toFixed(0)}% closed` : 'N/A'} · sim t {fmt(s?.sim_time, 1)} s · total steps {fmtInt(s?.total_steps)}</div>
      </details>
    </div>
  )
}
