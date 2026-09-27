"""One-click executive summary (Phase 5): a static, self-contained HTML snapshot built only from persisted
results and the live simulation - every number is read from a real file or the running system; missing items say N/A."""
from __future__ import annotations
import base64
import html
import json
import time
from pathlib import Path
import cv2

from simulation.config import resolve_path


def _load(name):
    p = resolve_path(f"logs/{name}")
    try:
        return json.loads(p.read_text()) if p.exists() else None
    except Exception:
        return None


def _pct(v):
    return "N/A" if v is None else f"{100 * float(v):.0f} %"


def _num(v, d=2, suffix=""):
    return "N/A" if v is None else f"{float(v):.{d}f}{suffix}"


def _img_b64(arr, q=85):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""


def _file_b64(p: Path):
    try:
        return base64.b64encode(p.read_bytes()).decode("ascii")
    except Exception:
        return None


def sweep_svg(results: list[dict]) -> str:
    if not results:
        return "<p>N/A</p>"
    pts = [(float(r["item"].get("level", r.get("randomization_level") or 0)), float(r["success_rate"] or 0), float(r["placement_rate"] or 0)) for r in results]
    W, H, m = 420, 180, 30
    def X(x): return m + x * (W - 2 * m)
    def Y(y): return H - m - y * (H - 2 * m)
    poly = lambda k: " ".join(f"{X(p[0]):.1f},{Y(p[k]):.1f}" for p in pts)
    grid = "".join(f'<line x1="{m}" y1="{Y(g):.1f}" x2="{W-m}" y2="{Y(g):.1f}" stroke="#333" stroke-dasharray="2,4"/><text x="2" y="{Y(g)+4:.1f}" font-size="9" fill="#888">{int(g*100)}%</text>' for g in (0, .25, .5, .75, 1))
    dots = "".join(f'<circle cx="{X(p[0]):.1f}" cy="{Y(p[1]):.1f}" r="3" fill="#3987e5"/><text x="{X(p[0])-8:.1f}" y="{H-8}" font-size="9" fill="#888">{p[0]:.2f}</text>' for p in pts)
    return (f'<svg width="{W}" height="{H}" style="background:#0f1216;border:1px solid #333">{grid}'
            f'<polyline points="{poly(1)}" fill="none" stroke="#3987e5" stroke-width="2"/><polyline points="{poly(2)}" fill="none" stroke="#199e70" stroke-width="1.5"/>{dots}'
            f'<text x="{m}" y="12" font-size="10" fill="#3987e5">success</text><text x="{m+60}" y="12" font-size="10" fill="#199e70">placement</text>'
            f'<text x="{W/2-40}" y="{H-1}" font-size="9" fill="#888"></text></svg>')


def build_summary_html(svc, header: dict) -> str:
    sim = svc.env.sim
    st = svc.state_payload()
    pol = st.get("policy") or {}
    ev = _load("evaluation_latest.json") or {}
    sc = _load("scorecard_latest.json")
    sw = _load("sweep_latest.json")
    cmp_ = _load("compare_latest.json")
    demos = []
    try:
        from lerobot_adapter.demo_recorder import list_demos, demo_summary
        demos = demo_summary(list_demos(svc.demos_dir)).get("by_type", {})
    except Exception:
        pass
    frames = [("Live workcell view (now)", _img_b64(sim.render_camera(svc.view_cam, 640, 480, show_frustum=False)))]
    if svc.obs is not None and "observation.images.overhead" in svc.obs:
        ov = cv2.resize(svc.obs["observation.images.overhead"], (192, 192), interpolation=cv2.INTER_NEAREST)
        frames.append(("Overhead camera - policy input (64×64, upscaled)", _img_b64(ov)))
    reels = sorted(svc.reels_dir.rglob("meta.json"), key=lambda p: p.stat().st_mtime, reverse=True) if svc.reels_dir.exists() else []
    succ = next((p for p in reels if p.parent.name.endswith("success")), None)
    fail = next((p for p in reels if p.parent.name.endswith("fail")), None)
    for label, p in (("Representative successful evaluation episode (filmstrip)", succ), ("Representative failed evaluation episode (filmstrip)", fail)):
        if p and (p.parent / "filmstrip.jpg").exists():
            frames.append((label, _file_b64(p.parent / "filmstrip.jpg")))
    rows = lambda rs: "".join(f"<tr><td>{html.escape(str(r.get('label')))}</td><td>{_pct(r.get('success_rate'))} ({r.get('successes')}/{r.get('episodes_run')})</td><td>{_pct(r.get('grasp_rate'))}</td><td>{_pct(r.get('placement_rate'))}</td><td>{_num(r.get('average_reward'),1)}</td><td>{r.get('collisions_total')}</td><td>{_num(r.get('parts_per_minute'))}</td></tr>" for r in rs)
    inf = st.get("inference") or {}
    h = f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(header.get('title','ADAPTIVE SORTING'))} - summary</title>
<style>body{{font-family:Inter,Segoe UI,system-ui,sans-serif;background:#0a0c10;color:#e6ebf2;margin:24px;font-size:13px}}h1{{letter-spacing:.2em;font-size:18px}}h2{{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:#62708a;margin-top:26px}}
table{{border-collapse:collapse;font-family:ui-monospace,monospace;font-size:12px}}td,th{{border-bottom:1px solid #262e39;padding:4px 10px;text-align:left}}th{{color:#62708a}}.kpi{{display:inline-block;min-width:150px;margin:6px 18px 6px 0}}.kpi b{{display:block;font-size:22px;color:#e6ebf2}}.kpi span{{color:#62708a;font-size:10px;letter-spacing:.08em;text-transform:uppercase}}.note{{color:#9aa6b5;font-size:11px}}img{{border:1px solid #364051;display:block;max-width:100%}}.frames{{display:flex;flex-wrap:wrap;gap:14px}}.card{{background:#0f1216;border:1px solid #262e39;padding:10px 14px}}</style></head><body>
<h1>{html.escape(header.get('title','ADAPTIVE SORTING'))} — {html.escape(header.get('subtitle','Autonomous Robotic Component Sorting'))}</h1>
<div class="note">Executive summary exported {time.strftime('%Y-%m-%d %H:%M:%S')} · UR10e · MuJoCo {__import__('mujoco').__version__} · PPO (stable-baselines3) with BC bootstrap · device {html.escape(st.get('device','N/A'))}. All figures below are measured on real simulated rollouts; N/A means not yet measured.</div>
<h2>Active policy</h2><div class="card">checkpoint <b>{html.escape(str(pol.get('checkpoint','N/A')))}</b> · run {html.escape(str(pol.get('name','N/A')))} · {pol.get('timesteps','N/A')} PPO steps · trainer eval at save: success {_pct(pol.get('eval_success_rate'))}, placement {_pct(pol.get('eval_placement_rate'))}</div>
<h2>Last live evaluation ({ev.get('episodes_run','N/A')} episodes, randomization {ev.get('randomization_level','N/A')}, {html.escape(str(ev.get('finished_iso','N/A')))})</h2>
<div class="card"><div class="kpi"><b>{_pct(ev.get('success_rate'))}</b><span>both parts sorted</span></div><div class="kpi"><b>{_pct(ev.get('grasp_rate'))}</b><span>grasp rate</span></div><div class="kpi"><b>{_pct(ev.get('placement_rate'))}</b><span>≥1 part placed</span></div>
<div class="kpi"><b>{_num(ev.get('parts_per_minute'))}</b><span>parts per minute (cycle time)</span></div><div class="kpi"><b>{ev.get('collisions_total','N/A')}</b><span>collision steps in {ev.get('episodes_run','N/A')} episodes</span></div><div class="kpi"><b>{_num(ev.get('control_effort_mean'),1)}</b><span>control effort Σ|a|² / episode</span></div><div class="kpi"><b>{ev.get('false_picks','N/A')}</b><span>false picks (rejection part)</span></div></div>
<h2>Generalization scorecard {('(' + str(sc.get('episodes')) + ' episodes per scenario, ' + html.escape(str(sc.get('finished_iso'))) + ')') if sc else ''}</h2>
{('<table><tr><th>scenario</th><th>success</th><th>grasp</th><th>placement</th><th>avg reward</th><th>collision steps</th><th>parts/min</th></tr>' + rows(sc.get('results', [])) + '</table>') if sc else '<p class="note">N/A — not run yet</p>'}
<h2>Randomization difficulty sweep {('(' + str(sw.get('episodes')) + ' episodes per level)') if sw else ''}</h2>
{(sweep_svg(sw.get('results', [])) + '<table><tr><th>level</th><th>success</th><th>grasp</th><th>placement</th><th>avg reward</th><th>collision steps</th><th>parts/min</th></tr>' + rows(sw.get('results', [])) + '</table>') if sw else '<p class="note">N/A — not run yet</p>'}
<h2>Before / after (fresh evaluation, identical seeds)</h2>
{('<table><tr><th>checkpoint</th><th>success</th><th>grasp</th><th>placement</th><th>avg reward</th><th>collision steps</th><th>parts/min</th></tr>' + rows([{**r, 'label': r.get('checkpoint')} for r in cmp_.get('results', [])]) + f"</table><p class='note'>{cmp_.get('episodes')} episodes each, seed {cmp_.get('seed')}, level {cmp_.get('randomization_level')}</p>") if cmp_ else '<p class="note">N/A — not run yet</p>'}
<h2>Live inference session</h2><div class="card">{inf.get('episodes',0)} episodes · success {_pct(inf.get('success_rate'))} · {_num(inf.get('parts_per_minute'))} parts/min · {inf.get('collision_steps',0)} collision steps · mode {html.escape(st.get('mode',''))}</div>
<h2>Demonstrations available for imitation bootstrap</h2><div class="card">{' · '.join(f"{html.escape(k)}: {v['episodes']} episodes ({v['frames']} frames)" for k, v in demos.items()) or 'none'} <span class="note">— scripted_demo episodes are pipeline validators (hand-coded controller with ground truth), not human data and never a policy metric.</span></div>
<h2>Frames</h2><div class="frames">{''.join(f'<div><div class="note">{html.escape(l)}</div><img src="data:image/jpeg;base64,{b}"/></div>' for l, b in frames if b)}</div>
<h2>Safety</h2><div class="card note">Simulated e-stop {'ENGAGED' if st.get('safety',{}).get('estop') else 'not engaged'} · envelope {'OK' if st.get('safety',{}).get('envelope_ok') else 'VIOLATION'} · workspace margin {_num(st.get('safety',{}).get('workspace_margin_m'),3,' m')} · joint-limit margin {_num(st.get('safety',{}).get('joint_limit_margin_rad'),3,' rad')} · velocity utilisation {_pct(st.get('safety',{}).get('velocity_utilization'))}. The e-stop halts the control loop between 100 ms control steps and holds joint targets; it is a simulation-only status/interlock, not a certified safety function.</div>
</body></html>"""
    return h
