import { useEffect, useRef, useState } from 'react'
import type { FrameMessage } from '../api'
import { post } from '../api'
import { teleopBus } from '../teleopBus'

const TYPE_COLORS: Record<string, string> = { teleop_demo: 'var(--series-3)', assisted_demo: '#b98cff', scripted_demo: 'var(--series-4)', policy_rollout: 'var(--series-1)' }

// Main 3D view. Overlays show only what is happening (mode, target, grasp, progress) and the policy's camera.
// Teleop: drag = jog the arm in the table plane, click a part = assisted pick. Parts from the catalog can be dropped here.
export function Viewport({ frame, connected, flash }: { frame: FrameMessage | null; connected: boolean; flash?: (m: string, bad?: boolean) => void }) {
  const s = frame?.state
  const imgRef = useRef<HTMLImageElement | null>(null)
  const down = useRef<{ x: number; y: number; moved: boolean } | null>(null)
  const [pick, setPick] = useState<null | { x: number; y: number; label: string; ok: boolean }>(null)
  const [dropHover, setDropHover] = useState(false)
  const [resetFlash, setResetFlash] = useState<null | { count: number; bins: boolean }>(null)
  const lastResetCount = useRef<number | null>(null)
  const teleop = s?.mode === 'teleop'
  const replay = s?.mode === 'replay'
  useEffect(() => {
    const lr = s?.last_reset
    if (!lr) return
    if (lastResetCount.current !== null && lr.reset_count !== lastResetCount.current) {
      setResetFlash({ count: lr.reset_count, bins: !lr.bins_fixed })
      const t = setTimeout(() => setResetFlash(null), 1800)
      lastResetCount.current = lr.reset_count
      return () => clearTimeout(t)
    }
    lastResetCount.current = lr.reset_count
  }, [s?.last_reset?.reset_count])
  const modeLabel = s ? (s.mode === 'inference' ? 'POLICY' : s.mode === 'evaluate' ? 'EVALUATION' : s.mode === 'random' ? 'RANDOM ACTIONS' : s.mode === 'teleop' ? (s.assist?.active ? 'TELEOP · ASSISTED PICK' : 'TELEOP') : s.mode === 'scripted_demo' ? 'SCRIPTED DEMO' : s.mode === 'replay' ? 'REPLAY' : 'IDLE') : '—'
  const imageUV = (e: { clientX: number; clientY: number }): { u: number; v: number; x: number; y: number } | null => {
    const img = imgRef.current; if (!img) return null
    const r = img.getBoundingClientRect(); const nw = img.naturalWidth || 1120, nh = img.naturalHeight || 720
    const scale = Math.min(r.width / nw, r.height / nh); const dw = nw * scale, dh = nh * scale
    const ox = r.left + (r.width - dw) / 2, oy = r.top + (r.height - dh) / 2
    const u = (e.clientX - ox) / dw, v = (e.clientY - oy) / dh
    if (u < 0 || u > 1 || v < 0 || v > 1) return null
    return { u, v, x: e.clientX - r.left, y: e.clientY - r.top }
  }
  const onDown = (e: React.MouseEvent) => { if (!teleop) return; down.current = { x: e.clientX, y: e.clientY, moved: false } }
  const onMove = (e: React.MouseEvent) => {
    if (!teleop || !down.current) return
    const dx = e.clientX - down.current.x, dy = e.clientY - down.current.y
    if (Math.abs(dx) > 5 || Math.abs(dy) > 5) { down.current.moved = true; teleopBus.drag.active = true }
    if (teleopBus.drag.active) { teleopBus.drag.dx = Math.max(-1, Math.min(1, dx / 120)); teleopBus.drag.dy = Math.max(-1, Math.min(1, -dy / 120)) }
  }
  const endDrag = () => { teleopBus.drag.active = false; teleopBus.drag.dx = 0; teleopBus.drag.dy = 0 }
  const onUp = async (e: React.MouseEvent) => {
    const d = down.current; down.current = null; endDrag()
    if (!teleop || !d || d.moved) return
    const uv = imageUV(e); if (!uv) return
    try {
      const res = await post('/teleop/pick_at', { u: uv.u, v: uv.v })
      const hit = res.data?.hit
      const label = hit?.kind === 'object' ? `PICKING ${String(hit.name).toUpperCase()} → ${String(res.data?.assist?.bin ?? '').toUpperCase().replace('_', ' ')}` : hit?.kind === 'bin' ? `BIN ${String(hit.name).toUpperCase().replace('_', ' ')}` : 'NO PART HERE'
      setPick({ x: uv.x, y: uv.y, label, ok: hit?.kind === 'object' || hit?.kind === 'bin' })
      setTimeout(() => setPick(null), 1500)
    } catch (err: any) { flash?.(`Pick failed: ${err.message}`, true) }
  }
  const onDragOver = (e: React.DragEvent) => { if (e.dataTransfer.types.includes('text/x-part')) { e.preventDefault(); e.dataTransfer.dropEffect = 'copy'; setDropHover(true) } }
  const onDrop = async (e: React.DragEvent) => {
    e.preventDefault(); setDropHover(false)
    const part = e.dataTransfer.getData('text/x-part'); if (!part) return
    const uv = imageUV(e); if (!uv) { flash?.('Drop inside the 3D view', true); return }
    try { await post('/scene/toggle', { part, on: true, u: uv.u, v: uv.v }); setPick({ x: uv.x, y: uv.y, label: `PLACED ${part.toUpperCase()}`, ok: true }); setTimeout(() => setPick(null), 1500) }
    catch (err: any) { flash?.(`Placing failed: ${err.message}`, true) }
  }
  const stop = (e: React.SyntheticEvent) => e.stopPropagation()
  return (
    <div className="card relative overflow-hidden flex-1 flex items-center justify-center" style={{ background: '#06080b', minHeight: 320, cursor: teleop ? 'crosshair' : 'default', userSelect: 'none' }}
      onMouseDown={onDown} onMouseMove={onMove} onMouseUp={onUp} onMouseLeave={() => { down.current = null; endDrag() }}
      onDragOver={onDragOver} onDragLeave={() => setDropHover(false)} onDrop={onDrop}>
      {frame ? <img ref={imgRef} src={`data:image/jpeg;base64,${frame.frame}`} alt="workcell" className="w-full h-full object-contain" draggable={false} />
        : <div className="text-[11px] tracking-widest" style={{ color: 'var(--text-muted)' }}>{connected ? 'WAITING FOR FRAMES' : 'CONNECTING TO SIMULATION'}</div>}
      {dropHover && <div className="absolute inset-0 pointer-events-none" style={{ border: '2px dashed var(--series-3)', boxShadow: 'inset 0 0 40px rgba(25,158,112,0.25)', zIndex: 2 }} />}
      {pick && (
        <div className="absolute pointer-events-none" style={{ left: pick.x - 18, top: pick.y - 18 }}>
          <div style={{ width: 36, height: 36, borderRadius: '50%', border: `2px solid ${pick.ok ? '#b98cff' : 'var(--status-bad)'}`, boxShadow: `0 0 12px ${pick.ok ? '#b98cff' : 'var(--status-bad)'}` }} />
          <div className="overlay-tag" style={{ position: 'static', display: 'inline-block', marginTop: 4, whiteSpace: 'nowrap', color: pick.ok ? '#d9c8ff' : undefined }}>{pick.label}</div>
        </div>
      )}
      {s && (
        <>
          <div className="overlay-tag" style={{ top: 8, left: 8, color: replay ? '#ffb347' : teleop ? (s.assist?.active ? '#b98cff' : 'var(--series-3)') : s.mode === 'inference' ? 'var(--accent)' : undefined }}>{modeLabel}{s.paused ? ' · PAUSED' : ''}</div>
          {replay && s.replay && <div className="overlay-tag" style={{ top: 34, left: 8, color: '#ffb347', borderColor: '#ffb347' }}>{s.replay.playing ? '▶' : '❚❚'} step {s.replay.index}/{s.replay.n}</div>}
          {teleop && s.assist?.active && <div className="overlay-tag" style={{ top: 34, left: 8, color: '#d9c8ff', borderColor: '#b98cff' }}>{s.assist.object.toUpperCase()} → {s.assist.bin.toUpperCase().replace('_', ' ')} · {s.assist.phase}</div>}
          {teleop && !s.assist?.active && <div className="overlay-tag" style={{ top: 34, left: 8, color: 'var(--text-muted)' }}>click a part to pick it · drag to move the arm</div>}
          {s.recording?.recording && <div className="overlay-tag" style={{ top: 60, left: 8, color: 'var(--status-bad)', borderColor: 'var(--status-bad)' }}>● REC <span style={{ color: TYPE_COLORS[s.recording.episode_type ?? ''] }}>{s.recording.episode_type?.replace('_demo', '').toUpperCase()}</span></div>}
          <div className="absolute flex gap-1" style={{ top: 8, right: 8 }}>
            <button className="overlay-tag" style={{ position: 'static', cursor: 'pointer', color: s.show_frustum ? '#40d9ff' : undefined, borderColor: s.show_frustum ? '#40d9ff' : undefined }}
              onClick={(e) => { stop(e); post('/simulation/frustum', { enabled: !s.show_frustum }).catch(() => {}) }} onMouseDown={stop} onMouseUp={stop} title="Show the policy camera's field of view in the scene">CAMERA FOV {s.show_frustum ? 'ON' : 'OFF'}</button>
            <button className="overlay-tag" style={{ position: 'static', cursor: 'pointer', color: s.show_wrist ? '#40d9ff' : undefined, borderColor: s.show_wrist ? '#40d9ff' : undefined }}
              onClick={(e) => { stop(e); post('/simulation/wrist_camera', { enabled: !s.show_wrist }).catch(() => {}) }} onMouseDown={stop} onMouseUp={stop} title="Show the wrist camera view">WRIST CAM {s.show_wrist ? 'ON' : 'OFF'}</button>
          </div>
          {s.show_wrist && frame?.wrist && (
            <div className="absolute" style={{ top: 34, right: 8 }}>
              <img src={`data:image/jpeg;base64,${frame.wrist}`} alt="wrist" width={192} height={144} style={{ display: 'block', border: '1px solid var(--line-strong)' }} />
              <div className="overlay-tag" style={{ position: 'static', display: 'inline-block', marginTop: 2 }}>WRIST CAMERA</div>
            </div>
          )}
          {resetFlash && (
            <div className="overlay-tag" style={{ top: '42%', left: '50%', transform: 'translate(-50%, -50%)', fontSize: 13, padding: '8px 16px', color: '#ffd166', borderColor: '#ffd166', background: 'rgba(10,12,16,0.85)' }}>
              ● NEW EPISODE{resetFlash.bins ? ' · bins moved' : ''}
            </div>
          )}
          <div className="overlay-tag" style={{ bottom: 8, left: 8 }}>TARGET · {s.target ? `${s.target.toUpperCase()} → ${s.target_bin?.toUpperCase().replace('_', ' ')}` : 'NONE'}</div>
          <div className="overlay-tag" style={{ bottom: 8, left: 210, color: s.grasped ? 'var(--status-good)' : undefined }}>{s.grasped ? (s.lifted ? 'HOLDING · LIFTED' : 'HOLDING') : 'GRIPPER OPEN'}</div>
          <div className="overlay-tag" style={{ bottom: 8, right: 8 }}>EPISODE {s.episode} · STEP {s.step}/{s.max_steps} · PLACED {s.placed_count}/{(s as any).present_count ?? Object.keys(s.objects).length}</div>
          {s.placed_count > 0 && s.target === null && <div className="overlay-tag" style={{ top: 86, left: 8, color: 'var(--status-good)', borderColor: 'var(--status-good)' }}>✓ ALL PARTS SORTED</div>}
          {frame?.overhead && (
            <div className="absolute" style={{ bottom: 36, right: 8 }}>
              <div className="overlay-tag" style={{ position: 'static', display: 'inline-block', marginBottom: 2 }}>POLICY CAMERA</div>
              <img src={`data:image/jpeg;base64,${frame.overhead}`} alt="overhead" width={128} height={128} style={{ display: 'block', border: '1px solid var(--line-strong)', imageRendering: 'pixelated' }} />
            </div>
          )}
        </>
      )}
    </div>
  )
}
