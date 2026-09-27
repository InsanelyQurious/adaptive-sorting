import { post } from '../api'
import type { SimState } from '../api'
import { usePolling } from '../hooks'
import { Btn } from './ui'

interface Part { name: string; label: string; category: string; role: string; slot?: string; color: string; thumbnail_png_b64: string | null; on_table: boolean; slot_bound: string | null }
interface Catalog { parts: Part[]; variants: Record<string, string | null>; absent_slots: string[]; layout: { part: string; x: number; y: number; yaw: number }[]; task_objects: string[]; on_table: string[] }

// Parts catalog: every part is an independent on/off selection. Click to select (placed at a free spot) or drag onto
// the 3D view to choose where. Click a selected part to remove it. Any combination is valid, including nothing.
export function ScenePanel({ state, flash }: { state: SimState | null; flash: (m: string, bad?: boolean) => void }) {
  const cat = usePolling<Catalog>('/scene/catalog', 3000, [state?.scene?.layout?.length, JSON.stringify(state?.scene?.variants ?? {}), state?.episode])
  const act = async (label: string, p: Promise<any>) => { try { await p; flash(label); cat.refresh() } catch (e: any) { flash(`${label} failed: ${e.message}`, true) } }
  const parts = cat.data?.parts ?? []
  const groups: [string, string][] = [['bracket', 'Brackets → Bin A'], ['fastener', 'Fasteners → Bin B'], ['clutter', 'Extra parts'], ['rejection', 'Does not belong']]
  const locked = !!state && ['evaluate', 'replay', 'scripted_demo'].includes(state.mode)
  const onTable = parts.filter((p) => p.on_table)
  return (
    <div className="flex flex-col gap-2">
      <div className="card p-2">
        <div className="text-[10px] mb-2" style={{ color: 'var(--text-secondary)' }}>Click a part to put it on the table, click again to take it off. Drag onto the 3D view to choose the spot.</div>
        {groups.map(([cat_, title]) => (
          <div key={cat_} className="mb-2">
            <div className="stat-label mb-1">{title}</div>
            <div className="flex flex-wrap gap-1.5">
              {parts.filter((p) => p.category === cat_).map((p) => (
                <div key={p.name} draggable={!locked} onDragStart={(e) => { e.dataTransfer.setData('text/x-part', p.name); e.dataTransfer.effectAllowed = 'copy' }}
                  title={`${p.label}${p.on_table ? ' — on the table (click to remove)' : ' — click to place'}`}
                  onClick={() => !locked && act(p.on_table ? `Removed ${p.name}` : `Placed ${p.name}`, post('/scene/toggle', { part: p.name }))}
                  className="card p-1 flex flex-col items-center relative" style={{ width: 82, cursor: locked ? 'not-allowed' : 'pointer', borderColor: p.on_table ? (p.slot_bound ? 'var(--accent)' : 'var(--series-3)') : 'var(--line)', opacity: locked ? 0.5 : 1, background: p.on_table ? 'rgba(57,135,229,0.08)' : undefined }}>
                  {p.on_table && <span className="absolute" style={{ top: 2, right: 4, color: p.slot_bound ? 'var(--accent)' : 'var(--series-3)', fontSize: 12 }}>✓</span>}
                  {p.thumbnail_png_b64 ? <img src={`data:image/png;base64,${p.thumbnail_png_b64}`} width={64} height={64} alt={p.name} draggable={false} /> : <div style={{ width: 64, height: 64, background: p.color }} />}
                  <div className="text-[9px] text-center leading-3 mt-1" style={{ color: 'var(--text-primary)' }}>{p.label.replace(/ \(.*\)$/, '')}</div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
      <div className="card p-2">
        <div className="flex items-center justify-between mb-1">
          <div className="card-title">On the table ({onTable.length})</div>
          <div className="flex gap-1">
            <Btn onClick={() => act('Table cleared', post('/scene/clear'))} disabled={locked || onTable.length === 0}>Clear table</Btn>
            <Btn onClick={() => act('Default parts placed', post('/scene/defaults'))} disabled={locked}>Defaults</Btn>
          </div>
        </div>
        {onTable.length === 0 ? <div className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Nothing on the table. Pick parts above or choose a scenario.</div> : (
          <div className="text-[10px] flex flex-col" style={{ color: 'var(--text-secondary)' }}>
            {onTable.map((p) => (
              <div key={p.name} className="flex justify-between items-center" style={{ borderTop: '1px solid var(--line)', padding: '2px 0' }}>
                <span><span style={{ color: 'var(--text-primary)' }}>{p.label.replace(/ \(.*\)$/, '')}</span>{p.slot_bound ? <span style={{ color: 'var(--accent)' }}> · the {p.slot_bound} the policy sorts</span> : p.role === 'reject' ? <span style={{ color: 'var(--status-bad)' }}> · should be left alone</span> : <span style={{ color: 'var(--text-muted)' }}> · extra</span>}</span>
                <Btn tone="danger" onClick={() => act(`Removed ${p.name}`, post('/scene/toggle', { part: p.name, on: false }))} disabled={locked}>✕</Btn>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
