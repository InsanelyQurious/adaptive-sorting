import type { ReactNode } from 'react'

export function Card({ title, right, children, className = '' }: { title?: string; right?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <div className={`card p-2.5 ${className}`}>
      {(title || right) && (
        <div className="flex items-center justify-between mb-2">
          <div className="card-title">{title}</div>
          <div>{right}</div>
        </div>
      )}
      {children}
    </div>
  )
}

export function Stat({ label, value, mono = true, title }: { label: string; value: ReactNode; mono?: boolean; title?: string }) {
  return (
    <div className="min-w-0" title={title}>
      <div className="stat-label truncate">{label}</div>
      <div className={`stat-value truncate ${mono ? 'mono' : ''}`}>{value}</div>
    </div>
  )
}

export function StatGrid({ children, cols = 3 }: { children: ReactNode; cols?: number }) {
  return <div className="grid gap-x-3 gap-y-2" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}>{children}</div>
}

export type Tone = 'good' | 'warn' | 'bad' | 'info' | 'muted'
const toneColor: Record<Tone, string> = { good: 'var(--status-good)', warn: 'var(--status-warn)', bad: 'var(--status-bad)', info: 'var(--status-info)', muted: 'var(--text-muted)' }

export function Badge({ tone = 'muted', children, pulse = false }: { tone?: Tone; children: ReactNode; pulse?: boolean }) {
  return (
    <span className="badge">
      <span className={`dot ${pulse ? 'dot-pulse' : ''}`} style={{ background: toneColor[tone] }} />
      {children}
    </span>
  )
}

export function Btn({ children, onClick, tone = '', disabled = false, active = false, title }: {
  children: ReactNode; onClick?: () => void; tone?: '' | 'accent' | 'danger'; disabled?: boolean; active?: boolean; title?: string
}) {
  return (
    <button className={`btn ${tone ? `btn-${tone}` : ''} ${active ? 'btn-active' : ''}`} onClick={onClick} disabled={disabled} title={title}>
      {children}
    </button>
  )
}

export function Tabs({ tabs, value, onChange }: { tabs: string[]; value: string; onChange: (t: string) => void }) {
  return (
    <div className="flex flex-wrap border-b" style={{ borderColor: 'var(--line)' }}>
      {tabs.map((t) => (
        <div key={t} className={`tab ${t === value ? 'tab-active' : ''}`} onClick={() => onChange(t)}>{t}</div>
      ))}
    </div>
  )
}

export function Bar({ value, max = 1, color = 'var(--series-1)' }: { value: number | null | undefined; max?: number; color?: string }) {
  const pct = value === null || value === undefined ? 0 : Math.max(0, Math.min(100, (value / max) * 100))
  return (
    <div className="h-1.5 w-full" style={{ background: 'var(--surface-3)' }}>
      <div className="h-1.5" style={{ width: `${pct}%`, background: color, transition: 'width 200ms' }} />
    </div>
  )
}
