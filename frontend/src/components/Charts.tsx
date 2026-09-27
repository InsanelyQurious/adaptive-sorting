// Small real-time line charts (Recharts). Single series per chart; the reward chart carries two
// (episode reward + moving average) with a legend. Colors from the validated dark palette.
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis, CartesianGrid, Legend } from 'recharts'
import type { MetricRow } from '../api'
import { fmt, fmtInt } from '../api'

export interface Series { key: string; label: string; color?: string }

export function MetricChart({ rows, series, title, height = 110, yFormat = (v: number) => fmt(v, 2), xKey = 'timesteps', domain }: {
  rows: MetricRow[]; series: Series[]; title: string; height?: number; yFormat?: (v: number) => string; xKey?: string; domain?: [number, number]
}) {
  const data = rows.filter((r) => series.some((s) => typeof r[s.key] === 'number'))
  const has = data.length > 0
  return (
    <div className="card p-2">
      <div className="flex justify-between items-baseline mb-1">
        <div className="card-title">{title}</div>
        <div className="mono text-[10px]" style={{ color: 'var(--text-secondary)' }}>
          {has ? series.map((s) => `${series.length > 1 ? s.label + ' ' : ''}${yFormat(Number(data[data.length - 1][s.key]))}`).join(' · ') : 'N/A'}
        </div>
      </div>
      <div style={{ height }}>
        {has ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 4, right: 6, left: -18, bottom: 0 }}>
              <CartesianGrid stroke="var(--line)" strokeDasharray="2 4" vertical={false} />
              <XAxis dataKey={xKey} tick={{ fontSize: 9, fill: 'var(--text-muted)' }} tickFormatter={(v) => fmtInt(v)} stroke="var(--line-strong)" minTickGap={30} />
              <YAxis tick={{ fontSize: 9, fill: 'var(--text-muted)' }} tickFormatter={(v) => yFormat(v)} stroke="var(--line-strong)" width={60} domain={domain ?? ['auto', 'auto']} allowDataOverflow={!!domain} />
              <Tooltip contentStyle={{ background: 'var(--surface-2)', border: '1px solid var(--line-strong)', fontSize: 10, padding: '4px 8px' }}
                labelStyle={{ color: 'var(--text-secondary)' }} itemStyle={{ color: 'var(--text-primary)' }}
                labelFormatter={(v) => `${xKey}: ${fmtInt(Number(v))}`} formatter={(v: any, name: any) => [yFormat(Number(v)), name]} />
              {series.length > 1 && <Legend wrapperStyle={{ fontSize: 10, color: 'var(--text-secondary)' }} iconSize={8} />}
              {series.map((s, i) => (
                <Line key={s.key} type="monotone" dataKey={s.key} name={s.label} stroke={s.color ?? `var(--series-${i + 1})`} strokeWidth={2} dot={false} isAnimationActive={false} connectNulls />
              ))}
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="h-full flex items-center justify-center text-[10px]" style={{ color: 'var(--text-muted)' }}>no data yet</div>
        )}
      </div>
    </div>
  )
}
