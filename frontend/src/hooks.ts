import { useEffect, useRef, useState } from 'react'
import { WS_BASE, get, type FrameMessage, type MetricRow, type SimState, type TrainingStatus } from './api'

export function useSimulationSocket() {
  const [frame, setFrame] = useState<FrameMessage | null>(null)
  const [events, setEvents] = useState<any[]>([])
  const [connected, setConnected] = useState(false)
  useEffect(() => {
    let ws: WebSocket | null = null
    let closed = false
    let latest: FrameMessage | null = null
    let raf = 0
    const flush = () => { raf = 0; if (latest) setFrame(latest) }
    const connect = () => {
      ws = new WebSocket(`${WS_BASE}/ws/simulation`)
      ws.onopen = () => setConnected(true)
      ws.onclose = () => { setConnected(false); if (!closed) setTimeout(connect, 1500) }
      ws.onerror = () => ws?.close()
      ws.onmessage = (e) => {
        const m = JSON.parse(e.data)
        if (m.type === 'frame') { latest = m; if (!raf) raf = requestAnimationFrame(flush) }
        else if (m.type !== 'heartbeat') setEvents((ev) => [...ev.slice(-49), m])
      }
    }
    connect()
    return () => { closed = true; ws?.close(); if (raf) cancelAnimationFrame(raf) }
  }, [])
  return { frame, state: frame?.state ?? null as SimState | null, events, connected }
}

export function useTrainingSocket() {
  const [status, setStatus] = useState<TrainingStatus | null>(null)
  const [metrics, setMetrics] = useState<MetricRow[]>([])
  const [connected, setConnected] = useState(false)
  useEffect(() => {
    let ws: WebSocket | null = null
    let closed = false
    const connect = () => {
      ws = new WebSocket(`${WS_BASE}/ws/training`)
      ws.onopen = () => setConnected(true)
      ws.onclose = () => { setConnected(false); if (!closed) setTimeout(connect, 2000) }
      ws.onerror = () => ws?.close()
      ws.onmessage = (e) => {
        const m = JSON.parse(e.data)
        if (m.type === 'history') { setMetrics(m.metrics ?? []); setStatus(m.status) }
        else if (m.type === 'update') {
          if (m.metrics?.length) setMetrics((prev) => [...prev, ...m.metrics].slice(-2000))
          setStatus(m.status)
        }
      }
    }
    connect()
    return () => { closed = true; ws?.close() }
  }, [])
  return { status, metrics, connected }
}

export function usePolling<T>(path: string, intervalMs: number, deps: any[] = []) {
  const [data, setData] = useState<T | null>(null)
  const timer = useRef<number | null>(null)
  const refresh = async () => { try { setData(await get<T>(path)) } catch { /* backend down */ } }
  useEffect(() => {
    refresh()
    timer.current = window.setInterval(refresh, intervalMs)
    return () => { if (timer.current) clearInterval(timer.current) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)
  return { data, refresh }
}
