"""The dashboard on :8400 over data/.

The RoboARCade dashboard: what is under data/, for everyone. Zero-shot evaluation (data/episodes, twenty
launches a page, every episode with its video, transcript and score) and Analysis (the episode folders
data/analysis.json lists, aggregated, and a page per task that plays each one). It launches nothing: `roboarcade eval` does. agent_log.py reads the agent logs.

Run: source set_env.sh && python -m dashboard      # http://localhost:8400
"""
import html
import json
import mimetypes
import os
import subprocess
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from core.agent_log import shot_bytes, transcript, transcript_file


PORT = int(os.environ.get("ROBOARCADE_EVAL_PORT", "8400"))
REPO = Path(__file__).resolve().parents[1]
PAGE = 20
RANGE_CHUNK = 4 << 20
STATE = {"episodes": [], "runs": [], "scanned": 0.0}
EPISODES = REPO / "data" / "episodes"
ANALYSIS = REPO / "data" / "analysis.json"


def scan():
    out = []
    for mp in EPISODES.glob("*/*/*/meta.json"):
        meta = _read(mp)
        if not meta:
            continue
        meta.update(path=str(mp.parent), summary=_read(mp.parent / "summary.json"))
        out.append(meta)
    return sorted(out, key=lambda m: (m["stamp"], m["run"], m["label"]), reverse=True)


def runs(rolls):
    groups = {}
    for m in rolls:
        g = groups.setdefault(m["run"], dict(run=m["run"], stamp=m["stamp"], task=m["task"], model=m["model"], interface=m["interface"],
                                              final=m.get("final"), status=m.get("status"), case_study=m.get("case_study"), episodes=[]))
        g["episodes"].append(m)
    for g in groups.values():
        g["episodes"].sort(key=lambda m: m["label"])
        g["n"], g["solved"] = len(g["episodes"]), sum(1 for m in g["episodes"] if m.get("solved"))
        g["graded"] = sum(1 for m in g["episodes"] if m.get("solved") is not None)
    return sorted(groups.values(), key=lambda g: g["stamp"], reverse=True)


def _read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _scan_loop():
    while True:
        try:
            STATE["episodes"] = scan()
            STATE["runs"] = runs(STATE["episodes"])
            STATE["scanned"] = time.time()
        except Exception as exc:
            print("scan:", exc, flush=True)
        time.sleep(30)




CSS = """.quick{font-size:13px;font-weight:400;margin-left:10px}
:root{--bg:#F5F6F4;--panel:#FFF;--ink:#171B19;--ink2:#555E58;--rule:#D6DAD5;--accent:#17614B;--zebra:#EEF1EE;--ok:#17614B;--bad:#B23A2E}
@media(prefers-color-scheme:dark){:root{--bg:#12150F;--panel:#1A1E19;--ink:#E8EBE5;--ink2:#A3ADA5;--rule:#2C332C;--accent:#74C7A6;--zebra:#20251F;--ok:#74C7A6;--bad:#E07A6F}}
*{box-sizing:border-box}html,body{overflow-x:hidden}body{margin:0;background:var(--bg);color:var(--ink);font:14.5px/1.45 "Iowan Old Style",Charter,Georgia,serif}
.wrap{max-width:70rem;margin:0 auto;padding:.9rem 1.1rem 3rem}.wrap.wide{max-width:none;padding:.9rem 1.6rem 3rem}
nav{display:flex;gap:1rem;align-items:baseline;border-bottom:1px solid var(--rule);padding-bottom:.5rem;margin-bottom:.9rem;font-size:13.5px}
nav .brand{font-weight:700;font-size:15px;margin-right:.4rem}nav a{color:var(--accent);text-decoration:none;padding:2px 6px;border-radius:4px}
nav a.on{background:var(--zebra);color:var(--ink);font-weight:600}nav .right{margin-left:auto;font-size:12px;color:var(--ink2)}
h1{font-size:22px;line-height:1.15;margin:0 0 .5rem}h2{font-size:16px;margin:1.2rem 0 .4rem}p{margin:.3rem 0 .6rem}
table{border-collapse:collapse;width:100%;table-layout:fixed;font:12.5px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--panel);border:1px solid var(--rule)}
th,td{text-align:left;padding:4px 8px;border-bottom:1px solid var(--rule);vertical-align:top}td{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}thead th{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--ink2);font-weight:600}
tbody tr:nth-child(even){background:var(--zebra)}tr.excluded td{color:var(--ink2)}tr.all td{background:var(--zebra);font-weight:700}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px}.mut{color:var(--ink2)}.num{text-align:right;font-variant-numeric:tabular-nums}
a{color:var(--accent)}.ok{color:var(--ok);font-weight:700}.bad{color:var(--bad);font-weight:700}
.badge{display:inline-block;font:11px ui-monospace,SFMono-Regular,Menlo,monospace;padding:0 .4em;border-radius:3px;border:1px solid var(--rule);color:var(--ink2);margin-left:.3rem}
.pager{display:flex;gap:.3rem;align-items:center;margin:.5rem 0;font:12.5px ui-monospace,SFMono-Regular,Menlo,monospace}.pager a,.pager span{padding:.1rem .45rem;border:1px solid var(--rule);border-radius:4px;text-decoration:none;background:var(--panel)}
.pager span.on{background:var(--accent);color:#fff;border-color:var(--accent)}.pager span.gap{border:0;background:none;color:var(--ink2)}
.cards{display:flex;gap:.5rem;flex-wrap:wrap;margin:.4rem 0 .8rem}.card{border:1px solid var(--rule);border-radius:6px;padding:.4rem .7rem;min-width:7.5rem;background:var(--panel)}
.card .k{font-size:10.5px;text-transform:uppercase;letter-spacing:.06em;color:var(--ink2)}.card .v{font-size:15px;font-weight:700}
.stages{list-style:none;padding:0;margin:.2rem 0 0;font-size:12.5px}.stages li::before{content:"✓ ";color:var(--ok)}.stages li.no::before{content:"✗ ";color:var(--bad)}
img,video{max-width:100%}video{width:100%;max-width:60rem;border-radius:6px;background:#000;display:block;margin:.3rem 0 .8rem}
.ev{display:flex;gap:.5rem;padding:.25rem .45rem;border-left:3px solid var(--rule);margin:.1rem 0;font-size:13px;background:var(--panel)}
.ev .tag{flex:none;width:1.2rem;text-align:center}.ev .c{white-space:pre-wrap;overflow-wrap:anywhere;word-break:break-word;min-width:0}
.ev.does{border-color:var(--accent)}.ev.does .tool{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-weight:700;color:var(--accent)}.ev.does .args{font:12px ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--ink2)}
.ev.result{color:var(--ink2);font:12px ui-monospace,SFMono-Regular,Menlo,monospace}.ev.result.err{color:var(--bad)}.ev.thinks{color:var(--ink2);font-style:italic}
.ev.shot img{height:100px;border-radius:4px;display:block}.ev.compact{border-color:#C98A2B}
.sync{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(0,1fr);gap:1rem;align-items:start}
.sync .player{position:sticky;top:.5rem}.sync .player video{max-width:none}
.sync .trans{position:relative;max-height:calc(100vh - 1rem);overflow-y:auto;padding-right:.3rem}
.blk{border:1px solid var(--rule);border-radius:6px;margin:.35rem 0;padding:.15rem .3rem .3rem;cursor:pointer}
.blk.on{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent)}
.blk .bh{font-size:12px;color:var(--ink2);margin:.1rem .15rem .2rem;font-variant-numeric:tabular-nums}
@media(max-width:900px){.sync{grid-template-columns:1fr}.sync .player{position:static}.sync .trans{max-height:none}}
"""

_EV_ICONS = {"thinks": "💭", "says": "🗣", "does": "🔧", "result": "↩", "shot": "📷", "wpmark": "⚙", "compact": "✂"}


def page(title, body, active="", wide=False):
    nav = "".join(f'<a href="{href}" class="{"on" if active == key else ""}">{label}</a>'
                  for key, href, label in (("eval", "/", "Zero-shot evaluation"),
                                           ("analysis", "/analysis", "Analysis")))
    when = time.strftime("%H:%M:%S", time.localtime(STATE["scanned"])) if STATE["scanned"] else "scanning"
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>RoboARCade · {html.escape(title)}</title><style>{CSS}</style></head><body><div class="wrap{" wide" if wide or active == "analysis" else ""}">'
            f'<nav><span class="brand">RoboARCade</span>{nav}<span class="right">data/ scanned {when}</span></nav>{body}</div></body></html>')


def _q(s, safe=""):
    return urllib.parse.quote(s, safe=safe)


def _when(stamp):
    s = stamp.replace("_", "")
    return f"{s[:4]}-{s[4:6]}-{s[6:8]} {s[8:10]}:{s[10:12]}" if len(s) >= 12 and s[:12].isdigit() else html.escape(stamp)


NAMES = {"gpt-5.6-sol": "GPT-5.6-Sol", "claude-opus-5": "Claude Opus 5", "claude-fable-5-1": "Claude Fable 5.1", "gpt-6-astra": "GPT-6-Astra", "gpt-6.1-sol": "GPT-6.1-Sol"}


def cohort(m):
    return f'{NAMES.get(m["model"], m["model"])} · {"RGBD" if m["interface"].startswith("rgbd") else "VIA"}'


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return sum(xs) / len(xs) if xs else None


def _per_eef(s, k, scale=1):
    return s[k] / scale / s["eef_commands"] if s.get(k) is not None and s.get("eef_commands") else None


METRICS = [("cost ($)", lambda s: s.get("cost_usd"), 2), ("EEF cmds", lambda s: s.get("eef_commands"), 0),
           ("turns / EEF", lambda s: _per_eef(s, "model_turns"), 1), ("calls / EEF", lambda s: _per_eef(s, "tool_calls"), 1),
           ("wall / EEF (min)", lambda s: _per_eef(s, "agent_wall_seconds", 60), 2)]


def _metric_cells(groups, solved_only):
    out = []
    for _, f, d in METRICS:
        v = _mean([_mean([f(s) for s in g if s["solved"] or not solved_only]) for g in groups])
        out.append("<td class=num>--</td>" if v is None else f"<td class=num>{v:.{d}f}</td>")
    return "".join(out)


def _metric_table(rows):
    head = "".join(f"<th class=num>{n}</th>" for n, _, _ in METRICS)
    body = ""
    for name, groups in rows:
        success = _mean([_mean([s["score"] for s in g]) for g in groups])
        body += (f'<tr><td>{html.escape(name)}</td><td class=num>{"--" if success is None else f"{100 * success:.1f}"}</td>'
                 f'{_metric_cells(groups, False)}{_metric_cells(groups, True)}<td class=num>{sum(map(len, groups))}</td></tr>')
    return ('<table><thead><tr><th></th><th class=num>success (%)</th>' + head.replace("<th class=num>", "<th class=num>all: ", 1)
            + head.replace("<th class=num>", "<th class=num>success only: ", 1) + f'<th class=num>episodes</th></tr></thead><tbody>{body}</tbody></table>')


def _auto_improve(sections):
    by_path = {_rel(m): m for m in STATE["episodes"] if m.get("summary")}
    out = ""
    for sec in sections:
        arms = {a: [by_path[p] for p in ps if p in by_path] for a, ps in sec["arms"].items()}
        tasks = sorted({m["task"] for ms in arms.values() for m in ms})
        groups = {a: [[m["summary"] for m in ms if m["task"] == t] for t in tasks if any(m["task"] == t for m in ms)] for a, ms in arms.items()}
        def count(ss):
            return (f'{sum(bool(s["solved"]) for s in ss)} / {sum(bool(s["timed_out"]) for s in ss)}' if ss else "--")
        def claims(ss):
            failed = [s for s in ss if not s["solved"]]
            n = sum(s["ended_by"] == "agent" and not s["timed_out"] for s in failed)
            return f"{n}/{len(failed)} ({int(100 * n / len(failed) + 0.5)}%)" if failed else "--"
        per_task = "".join(f'<tr><td>{html.escape(t.replace("_", " "))}</td>' + "".join(f'<td class=num>{count([m["summary"] for m in arms[a] if m["task"] == t])}</td>' for a in arms) + "</tr>" for t in tasks)
        allss = {a: [m["summary"] for m in arms[a]] for a in arms}
        per_task += ('<tr class=all><td>All</td>' + "".join(f"<td class=num>{count(allss[a])}</td>" for a in arms) + "</tr>"
                     + '<tr><td>&nbsp;&nbsp;failures ended by a success claim</td>' + "".join(f"<td class=num>{claims(allss[a])}</td>" for a in arms) + "</tr>")
        out += (f'<h3>{html.escape(sec["title"])}</h3>' + _metric_table([(a, groups[a]) for a in arms])
                + f'<p class="mut">Solved / timed out, per task and arm.</p><table><thead><tr><th>task</th>{"".join(f"<th class=num>{html.escape(a)}</th>" for a in arms)}</tr></thead><tbody>{per_task}</tbody></table>')
    return out


def render_analysis():
    sel, listed = _listed()
    OMITTED_TASKS, dropped = set(sel.get("excluded_tasks", [])), set(sel.get("dropped", []))
    final = [m for m in listed if m.get("summary") and _rel(m) not in dropped]
    cohorts = sorted({cohort(m) for m in final}, key=lambda c: (c.split(" · ")[1] == "RGBD", c))
    tasks = sorted({m["task"] for m in final})
    cell = {}
    for m in final:
        cell.setdefault((m["task"], cohort(m)), []).append(m["summary"])
    def score(ss):
        return _mean([s["score"] for s in ss])
    included = [t for t in tasks if t not in OMITTED_TASKS]
    order = sorted(included, key=lambda t: _mean([score(cell.get((t, c), [])) for c in cohorts]) or 0) + sorted(t for t in tasks if t in OMITTED_TASKS)
    pct = lambda v, d=0: "" if v is None else f"{100 * v:.{d}f}%"
    def entry(ss, d=0):
        if not ss:
            return '<span class="mut">—</span>'
        return f'{sum(s["solved"] for s in ss)} / {len(ss)} · {pct(score(ss), d) or "unscored"}'
    rows = "".join(f'<tr class="{"excluded" if t in OMITTED_TASKS else ""}"><td><a href="/analysis/{_q(t)}">{html.escape(t.replace("_", " "))}</a>{" · excluded" if t in OMITTED_TASKS else ""}</td>'
                   + "".join(f'<td class="num">{entry(cell.get((t, c), []))}</td>' for c in cohorts) + "</tr>" for t in order)
    allrow = "<tr class=all><td><b>All tasks</b></td>" + "".join(
        f'<td class="num"><b>{entry([s for t in included for s in cell.get((t, c), [])], 1)}</b></td>' for c in cohorts) + "</tr>"
    head = "".join(f"<th class=num>{html.escape(c)}</th>" for c in cohorts)
    tasks_table = (f'<h2>Task successes</h2><p class="mut">Each cell is solved / episodes and the mean subtask score (the share of the '
                   f'task\'s subtasks done, from each episode\'s subtasks.json). Excluded tasks are shown but left out of every total.</p>'
                   f'<table><thead><tr><th>task</th>{head}</tr></thead><tbody>{rows}{allrow}</tbody></table>')
    def row(c):
        ss = [s for t in included for s in cell.get((t, c), [])]
        ratio = lambda k: _mean([s[k] / s["eef_commands"] for s in ss if s.get(k) is not None and s.get("eef_commands")])
        f = lambda v, d: "--" if v is None else f"{v:.{d}f}"
        return (f'<tr><td>{html.escape(c.split(" · ")[0])}</td><td>{c.split(" · ")[1]}</td><td class=num>{f(None if score(ss) is None else 100 * score(ss), 1)}</td>'
                f'<td class=num>{f(_mean([s["cost_usd"] for s in ss]), 2)}</td><td class=num>{f(ratio("model_turns"), 1)}</td>'
                f'<td class=num>{f(ratio("tool_calls"), 1)}</td><td class=num>{len(ss)}</td></tr>')
    table1 = ('<h2>Table 1. Zero-shot evaluation</h2><p class="mut">Success is the mean subtask score over the episodes of the included tasks; '
              'cost and model turns and tool calls per EEF command are per episode means.</p>'
              '<table><thead><tr><th>model</th><th>interface</th><th class=num>success (%)</th><th class=num>cost ($)</th>'
              '<th class=num>turns / EEF</th><th class=num>calls / EEF</th><th class=num>episodes</th></tr></thead><tbody>'
              + "".join(row(c) for c in cohorts) + "</tbody></table>")
    body = (f'<h1>Analysis</h1><p class="mut">Computed from the {len(final)} episode folders listed in data/analysis.json ("episodes") and not '
            f'dropped ("dropped", the unticked boxes under Episodes); the tasks under "excluded_tasks" are shown but left out of every total. '
            f'A task name opens its episodes\' videos.</p>'
            f'<div class="two">{tasks_table}</div><div class="two">{table1}</div>'
            )
    body += ('<h2>Table 1, all metrics</h2><p class="mut">Per episode means over the included tasks; success only keeps the solved episodes. '
             'Wall / EEF is the agent\'s wall time without offline Blender rendering (summary.json agent_wall_seconds) per EEF command.</p>'
             + _metric_table([(c, [[s for t in included for s in cell.get((t, c), [])]]) for c in cohorts]))
    if sel.get("auto_improve"):
        body += ('<h2>Autonomous improvement</h2><p class="mut">The episodes data/analysis.json lists under "auto_improve" for each arm; '
                 'every metric is the mean over the task means.</p>' + _auto_improve(sel["auto_improve"]))
    by_task = {}
    for m in listed:
        by_task.setdefault(m["task"], []).append(m)
    body += ('<h2>Episodes <a class="quick" href="/quickcheck">&#9654; Quick check</a></h2><p class="mut">Every episode folder data/analysis.json lists. Untick one to drop it from the analysis; '
             'reload to see the totals change.</p>'
             + "".join(f'<details><summary>{html.escape(t)} <span class="mut">{sum(_rel(m) not in dropped for m in ms)} of {len(ms)} counted</span></summary>'
                       + "".join(f'<div class="mono">{_drop_box(m, dropped)} <a href="{_episode_href(m)}">{html.escape(_rel(m))}</a> '
                                 f'<span class="mut">{html.escape(cohort(m))}</span></div>' for m in sorted(ms, key=_rel)) + '</details>'
                       for t, ms in sorted(by_task.items()))
             + f'<script>{DROP_JS}</script>')
    return page("Analysis", body, "analysis")


QUICK_JS = """const seen = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { e.target.preload = "auto"; seen.unobserve(e.target); } }), {rootMargin: "400px"});
document.querySelectorAll("video").forEach(v => { seen.observe(v); v.addEventListener("play", () => { v.playbackRate = +document.getElementById("rate").value; }); });
document.getElementById("rate").addEventListener("change", e => document.querySelectorAll("video").forEach(v => { v.playbackRate = +e.target.value; }));"""


def render_quickcheck(task):
    sel, listed = _listed()
    by_task = {}
    for m in listed:
        by_task.setdefault(m["task"], []).append(m)
    links = " · ".join(f'<a href="/quickcheck?task={_q(t)}"{" class=on" if t == task else ""}>{html.escape(t.replace("_", " "))}</a>'
                       f' <span class="mut">{len(ms)}</span>' for t, ms in sorted(by_task.items()))
    body = (f'<h1>Quick check</h1><p class="mut"><a href="/analysis">Analysis</a> · only the Blender replay of every listed episode, by task and '
            f'model, loaded as you scroll. Speed <select id="rate">{"".join(f"<option{sel_} value={r}>{r}x</option>" for r, sel_ in ((1, ""), (2, " selected"), (4, ""), (8, "")))}</select></p>'
            f'<p class="tasks">{links}</p>')
    if task in by_task:
        cohorts = {}
        for m in sorted(by_task[task], key=lambda m: ("RGBD" in cohort(m), cohort(m), m["seed"])):
            cohorts.setdefault(cohort(m), []).append(m)
        def tile(m):
            d = Path(m["path"])
            video = next((rel for rel in ("blender_feeds.mp4", "full_trajectory/blender_feeds.mp4") if (d / rel).is_file()), None)
            head = (f'<div class="qh">seed {m["seed"]} <span class="{"ok" if m.get("solved") else "bad"}">'
                    f'{"solved" if m.get("solved") else "failed"}</span> <a href="{_episode_href(m)}">log</a></div>')
            return head + (f'<video controls muted playsinline preload="none" src="/media/{_folder(m)}/{video}#t=0.1"></video>'
                           if video else '<div class="mut">no Blender video</div>')
        body += "".join(f'<h2>{html.escape(c)}</h2><div class="quickgrid">{"".join(f"<div>{tile(m)}</div>" for m in rs)}</div>'
                        for c, rs in cohorts.items())
    body += ('<style>.quickgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(30rem,1fr));gap:12px}'
             '.quickgrid video{width:100%;background:#000;border-radius:4px}.qh{font-size:13px;margin:2px 0}'
             '.tasks a.on{font-weight:600}</style>'
             f'<script>{QUICK_JS}</script>')
    return page("Quick check", body, "analysis")


def _rel(m):
    return os.path.relpath(m["path"], REPO)


def _listed():
    sel = _read(ANALYSIS) or {}
    paths = set(sel.get("episodes", []))
    return sel, [m for m in STATE["episodes"] if _rel(m) in paths]


def _drop_box(m, dropped):
    return f'<input type="checkbox" class="drop" data-path="{html.escape(_rel(m))}" title="counted in the analysis"{"" if _rel(m) in dropped else " checked"}>'


DROP_JS = """document.addEventListener('change', e => { if (e.target.classList.contains('drop'))
  fetch('/analysis/drop', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path: e.target.dataset.path, drop: !e.target.checked }) })
    .then(r => { if (!r.ok) { e.target.checked = !e.target.checked; alert('not saved'); } }); });"""

TASK_CSS = """.seeds{display:grid;grid-template-columns:repeat(auto-fill,minmax(26rem,1fr));gap:12px;align-items:start}.seed{display:grid;gap:3px}
.seed h4{margin:0;display:flex;gap:8px;align-items:baseline;font-size:13px}.seed h4 .r{margin-left:auto;font-weight:400;font-size:12px}.seed.dropped{opacity:.45}
.seed video{width:100%;background:#000;border-radius:4px}"""


def render_analysis_task(task):
    sel, listed = _listed()
    dropped, ms = set(sel.get("dropped", [])), [m for m in listed if m["task"] == task]
    if not ms:
        return None
    cohorts = {}
    for m in sorted(ms, key=lambda m: ("RGBD" in cohort(m), cohort(m), m["seed"])):
        cohorts.setdefault(cohort(m), []).append(m)
    def seed(m):
        d, score = Path(m["path"]), (m.get("summary") or {}).get("score")
        video = next((rel for rel in ("blender_feeds.mp4", "full_trajectory/blender_feeds.mp4") if (d / rel).is_file()), "")
        out = '<span class="mut">ungraded</span>' if m.get("solved") is None else f'<span class="{"ok" if m["solved"] else "bad"}">{"solved" if m["solved"] else "failed"}</span>'
        return (f'<div class="seed{" dropped" if _rel(m) in dropped else ""}">'
                f'<h4><span class="who">{html.escape(cohort(m))}, seed {m["seed"]}</span>{out}'
                f'<span class="mut">{"" if score is None else f"score {score:.2f}"}</span>'
                f'<span class="r"><label>{_drop_box(m, dropped)} counted</label> · <a href="{_episode_href(m)}">log</a></span></h4>'
                + (f'<video controls muted playsinline preload="none" src="/media/{_folder(m)}/{video}#t=0.1"></video>' if video
                   else '<div class="mut">no Blender video</div>') + '</div>')
    body = (f'<h1>{html.escape(task.replace("_", " "))}</h1><p class="mut"><a href="/analysis">Analysis</a> · {len(ms)} episodes listed in data/analysis.json'
            f'{" · this task is excluded from every total" if task in sel.get("excluded_tasks", []) else ""}.</p>'
            + "".join(f'<h2>{html.escape(c)} <span class="mut">{sum(bool(m.get("solved")) for m in rs)} of {len(rs)} solved</span></h2>'
                      f'<div class="seeds">{"".join(seed(m) for m in rs)}</div>' for c, rs in cohorts.items())
            + f'<style>{TASK_CSS}</style><script>{DROP_JS}</script>')
    return page(task, body, "analysis")


def _episode_href(m):
    return f'/episode/{_folder(m)}'


def _folder(m):
    return "/".join(_q(x) for x in Path(m["path"]).parts[-3:])


def _score_cell(m):
    s = (m.get("summary") or {}).get("score")
    return '<span class="mut">—</span>' if s is None else f'{s:.2f}'


def render_eval(query):
    runs = STATE["runs"]
    task, model = query.get("task", ""), query.get("model", "")
    if task:
        runs = [r for r in runs if r["task"] == task]
    if model:
        runs = [r for r in runs if r["model"] == model]
    n = len(runs)
    try:
        current = max(1, min(int(query.get("page", 1)), max(1, -(-n // PAGE))))
    except ValueError:
        current = 1
    shown = runs[(current - 1) * PAGE:current * PAGE]
    rows = "".join(
        f'<tr><td class="mono">{_when(r["stamp"])}</td><td><a href="/?task={_q(r["task"])}">{html.escape(r["task"])}</a></td>'
        f'<td>{html.escape(r["model"])}</td><td class="mono">{html.escape(r["interface"])}</td>'
        f'<td class="num"><span class="{"ok" if r["solved"] else ""}">{r["solved"]}</span> / {r["graded"]}</td>'
        f'<td>{"<span class=badge>final</span>" if r["final"] else ""}'
        f'{f"<span class=badge>{html.escape(str(r["case_study"]))}</span>" if r.get("case_study") else ""}</td>'
        f'<td class="mono" title="{html.escape(r["run"])}"><a href="/run/{_q(r["run"], "/")}">{html.escape(r["run"])}</a></td></tr>'
        for r in shown)
    pages = -(-n // PAGE)
    keep = "".join(f"&{k}={_q(v)}" for k, v in (("task", task), ("model", model)) if v)
    pager = _pager(current, pages, keep)
    first, last = (current - 1) * PAGE + 1, min(current * PAGE, n)
    filters = (f' · task <b>{html.escape(task)}</b>' if task else "") + (f' · model <b>{html.escape(model)}</b>' if model else "")
    clear = ' · <a href="/">all</a>' if task or model else ""
    body = (f'<h1>Zero-shot evaluation</h1><p class="mut">{n} launches under data/episodes{filters}{clear}; '
            f'showing {first if n else 0} to {last}. A launch is one task, one model, some seeds; open it for its episodes.</p>'
            f'<div class="pager">{pager}</div>'
            f'<table><colgroup><col style="width:9.5rem"><col style="width:15rem"><col style="width:10rem"><col style="width:9rem"><col style="width:7rem"><col style="width:5rem"><col></colgroup><thead><tr><th>launched</th><th>task</th><th>model</th><th>interface</th><th class="num">solved / graded</th><th></th><th>launch</th></tr></thead>'
            f'<tbody>{rows or "<tr><td colspan=7 class=mut>nothing published yet</td></tr>"}</tbody></table><div class="pager">{pager}</div>')
    return page("Zero-shot evaluation", body, "eval")


def _pager(current, pages, keep):
    if pages <= 1:
        return ""
    lo = max(1, min(current - 2, pages - 4))
    hi = min(pages, lo + 4)
    out = []
    if current > 1:
        out.append(f'<a href="/?page={current - 1}{keep}">‹</a>')
    if lo > 1:
        out.append(f'<a href="/?page=1{keep}">1</a>' + ('<span class="gap">…</span>' if lo > 2 else ""))
    out += [f'<span class="on">{p}</span>' if p == current else f'<a href="/?page={p}{keep}">{p}</a>' for p in range(lo, hi + 1)]
    if hi < pages:
        out.append(('<span class="gap">…</span>' if hi < pages - 1 else "") + f'<a href="/?page={pages}{keep}">{pages}</a>')
    if current < pages:
        out.append(f'<a href="/?page={current + 1}{keep}">›</a>')
    return "".join(out)


def render_run(run):
    r = next((x for x in STATE["runs"] if x["run"] == run), None)
    if not r:
        return None
    rows = "".join(
        f'<tr><td class="mono"><a href="{_episode_href(m)}">{html.escape(m["label"])}</a></td>'
        f'<td>{"<span class=ok>solved</span>" if m["solved"] else ("<span class=bad>failed</span>" if m["solved"] is False else "<span class=mut>ungraded</span>")}</td>'
        f'<td class="num">{_score_cell(m)}</td><td class="mono mut">{html.escape(Path(m["path"]).name)}</td></tr>' for m in r["episodes"])
    meta = r["episodes"][0].get("run_meta") or {}
    tot = meta.get("totals") or {}
    cards = "".join(f'<div class="card"><div class="k">{k}</div><div class="v">{html.escape(str(v))}</div></div>' for k, v in (
        ("task", r["task"]), ("model", r["model"]), ("interface", r["interface"]), ("status", r["status"] or "—"),
        ("solved", f'{r["solved"]} / {r["graded"]}'), ("cost", f'${tot["cost_usd"]:.2f}' if tot.get("cost_usd") is not None else "—")))
    body = (f'<h1><span class="mono">{html.escape(run)}</span></h1><div class="cards">{cards}</div>'
            f'<table><thead><tr><th>seed</th><th>outcome</th><th class="num">subtasks</th><th>folder</th></tr></thead><tbody>{rows}</tbody></table>')
    return page(run, body, "eval")


def _episode(task, model, name):
    d = EPISODES / task / model / name
    return d if d.is_dir() and d.resolve().is_relative_to(EPISODES.resolve()) else None


def _video(d):
    for rel in ("blender_feeds.mp4", "full_trajectory/blender_feeds.mp4"):
        if (d / rel).is_file():
            return rel
    mp4v = d / "cameras.mp4"
    if mp4v.is_file():
        h264 = d / "cameras_h264.mp4"
        if not h264.is_file():
            import imageio_ffmpeg
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(mp4v), "-c:v", "libx264",
                            "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(h264)], capture_output=True, timeout=600)
        if h264.is_file():
            return "cameras_h264.mp4"
    return None


def _ev(e, base):
    if e["kind"] == "shot":
        c = f'<img loading="lazy" src="{base}/{e["img"]}">'
    elif e["kind"] == "does":
        args = ", ".join(f"{k}={html.escape(str(v))[:160]}" for k, v in list((e.get("args") or {}).items())[:6])
        c = f'<span class="tool">{html.escape(e["tool"])}</span> <span class="args">{args}</span>'
    elif e["kind"] == "compact":
        c = "context compacted: everything above was replaced by a summary"
    else:
        c = html.escape(e.get("text", ""))[:3000]
    cls = e["kind"] + (" err" if e.get("err") else "")
    return f'<div class="ev {cls}"><span class="tag">{_EV_ICONS.get(e["kind"], "•")}</span><div class="c">{c}</div></div>'


def render_episode(task, model, name):
    d = _episode(task, model, name)
    if not d:
        return None
    meta = _read(d / "meta.json") or {}
    verdict = _read(d / "verdict.json") or {}
    subs = _read(d / "subtasks.json")
    solved = meta.get("solved")
    cards = [("task", task), ("model", model), ("seed", meta.get("seed", "—")), ("interface", meta.get("interface", "—")),
             ("outcome", "solved" if solved else "failed" if solved is False else "ungraded"),
             ("sim steps", f'{verdict.get("sim_steps_used", "—")} / {verdict.get("sim_steps_budget", "—")}'),
             ("ended by", verdict.get("ended_by", "—"))]
    cards_html = "".join(f'<div class="card"><div class="k">{k}</div><div class="v {"ok" if v == "solved" else "bad" if v == "failed" else ""}">{html.escape(str(v))}</div></div>' for k, v in cards)
    if subs:
        stages = "".join(f'<li class="{"" if t["done"] else "no"}">{html.escape(t["label"])}</li>' for t in subs["subtasks"])
        cards_html += (f'<div class="card"><div class="k">subtasks</div><div class="v">{sum(t["done"] for t in subs["subtasks"])} / '
                       f'{len(subs["subtasks"])}</div><ul class="stages">{stages}</ul></div>')
    video = _video(d)
    base = f'/media/{_q(task)}/{_q(model)}/{_q(name)}'
    vid = f'<video controls preload="metadata" src="{base}/{video}"></video>' if video else '<p class="mut">no video for this episode</p>'
    tf = transcript_file(d)
    if tf:
        events = transcript(tf)["events"]
        trans = _synced(events, _read(d / "cameras_segments.json") or {}, f'/shot/{_q(task)}/{_q(model)}/{_q(name)}')
        trans = (f'<div class="mut" style="font-size:12px;margin-bottom:.3rem">{len(events)} events, grouped by the waypoint they lead '
                 f'up to; the block of the waypoint playing is outlined. Click a block to play its waypoint. '
                 f'<label><input type="checkbox" id="follow" checked> follow the video</label></div>{trans}')
    else:
        trans = '<p class="mut">no transcript file in this episode</p>'
    body = (f'<h1>{html.escape(task)} · {html.escape(model)} · <span class="mono">{html.escape(name)}</span></h1>'
            f'<p class="mut">launch <a href="/run/{_q(meta.get("run", ""), "/")}" class="mono">{html.escape(meta.get("run", ""))}</a> · '
            f'folder <span class="mono">{html.escape(str(d))}</span></p><div class="cards">{cards_html}</div>'
            f'<div class="sync"><div class="player">{vid}</div><div class="trans">{trans}</div></div><script>{SYNC_JS}</script>')
    return page(f"{task} {name}", body, "eval", wide=True)


def _synced(events, segs, shots):
    fps = segs.get("fps") or 30
    span = {s["waypoint"]: (s["start"] / fps, s["end"] / fps) for s in segs.get("segments", [])}
    blocks = {}
    for e in events:
        blocks.setdefault(e["wp"], []).append(e)
    out, prev = [], 0.0
    for k in sorted(blocks):
        t0, t1 = span.get(k, (prev, prev))
        if k not in span and k > max(span, default=-1):
            t1 = 1e9
        prev = t1 if t1 < 1e9 else prev
        head = (f"waypoint {k} · {int(t0 // 60)}:{t0 % 60:04.1f}" if k in span else
                "after the last waypoint" if t1 >= 1e9 else f"waypoint {k} · not recorded")
        out.append(f'<div class="blk" data-t0="{t0:.3f}" data-t1="{t1:.3f}"><div class="bh">{head}</div>'
                   + "".join(_ev(e, shots) for e in blocks[k]) + "</div>")
    return "".join(out)


SYNC_JS = """(() => {
  const v = document.querySelector('.sync video'), box = document.querySelector('.sync .trans');
  const bs = [...document.querySelectorAll('.blk')], follow = document.getElementById('follow');
  if (!v || !bs.length) return;
  let cur = null;
  v.addEventListener('timeupdate', () => {
    const t = v.currentTime, b = bs.find(b => t >= +b.dataset.t0 && t < +b.dataset.t1) || null;
    if (b === cur) return;
    cur?.classList.remove('on'); cur = b;
    if (b) { b.classList.add('on'); if (follow?.checked) box.scrollTo({ top: b.offsetTop - 8, behavior: 'smooth' }); }
  });
  bs.forEach(b => b.addEventListener('click', e => {
    if (e.target.closest('a, input, label') || +b.dataset.t0 === +b.dataset.t1) return;
    v.currentTime = +b.dataset.t0 + 0.02; v.play();
  }));
})();"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="text/html; charset=utf-8"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _file(self, path):
        if not path.is_file():
            return self._send(404, "not found", "text/plain")
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        size = path.stat().st_size
        rng = self.headers.get("Range")
        with path.open("rb") as f:
            if rng and rng.startswith("bytes="):
                start, _, end = rng[6:].partition("-")
                start = int(start or 0)
                end = min(int(end) if end else start + RANGE_CHUNK - 1, size - 1)
                f.seek(start)
                data = f.read(end - start + 1)
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                data = f.read()
                self.send_response(200)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urllib.parse.urlsplit(self.path)
        p, query = u.path, dict(urllib.parse.parse_qsl(u.query))
        parts = [urllib.parse.unquote(x) for x in p.strip("/").split("/")] if p.strip("/") else []
        if p == "/":
            return self._send(200, render_eval(query))
        if p == "/analysis":
            return self._send(200, render_analysis())
        if p == "/quickcheck":
            return self._send(200, render_quickcheck(query.get("task")))
        if parts[:1] == ["analysis"] and len(parts) == 2:
            html_ = render_analysis_task(parts[1])
            return self._send(200, html_) if html_ else self._send(404, "no listed episodes of this task", "text/plain")
        if parts[:1] == ["run"] and len(parts) > 1:
            html_ = render_run("/".join(parts[1:]))
            return self._send(200, html_) if html_ else self._send(404, "unknown launch", "text/plain")
        if parts[:1] == ["episode"] and len(parts) == 4:
            html_ = render_episode(*parts[1:])
            return self._send(200, html_) if html_ else self._send(404, "unknown episode", "text/plain")
        if parts[:1] == ["media"] and len(parts) > 4 and (d := _episode(*parts[1:4])):
            f = (d / "/".join(parts[4:])).resolve()
            return self._file(f) if f.is_relative_to(d.resolve()) else self._send(404, "not found", "text/plain")
        if parts[:1] == ["shot"] and len(parts) == 5 and (d := _episode(*parts[1:4])) and parts[4].isdigit():
            tf = transcript_file(d)
            got = shot_bytes(tf, int(parts[4])) if tf else None
            return self._send(200, got[1], got[0]) if got else self._send(404, "no shot", "text/plain")
        self._send(404, "not found", "text/plain")

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or "{}")
        with _JSON_FILE_LOCK:
            if self.path == "/analysis/drop":
                sel = _read(ANALYSIS)
                if body.get("path") not in sel["episodes"]:
                    return self._send(400, "not a listed episode", "text/plain")
                dropped = set(sel.get("dropped", [])) - {body["path"]}
                sel["dropped"] = sorted(dropped | {body["path"]} if body.get("drop") else dropped)
                _write(ANALYSIS, sel)
                return self._send(200, "{}", "application/json")
        self._send(404, "not found", "text/plain")

    def log_message(self, *args):
        pass


_JSON_FILE_LOCK = threading.Lock()


def _write(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    tmp.replace(path)


class _Server(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    threading.Thread(target=_scan_loop, daemon=True).start()
    print(f"dashboard on http://localhost:{PORT}", flush=True)
    _Server(("", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
