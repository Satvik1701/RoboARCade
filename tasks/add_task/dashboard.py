"""Teleop dashboard: pick a task in the browser, teleop it, come back for another."""
import atexit
import functools
import html
import json
import os
import socket
import subprocess
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from core.tasks import (TASKS, CATEGORY_GAME, CATEGORY_IN_PROGRESS, CATEGORY_REAL_WORLD, TASK_CATEGORIES,
                           list_tasks)

BASE = int(os.environ.get("ROBOARCADE_BASE_PORT", "8200"))
VIEWER_PORT = BASE + 100
REPO = Path(__file__).resolve().parents[2]
THUMBS = REPO / "data" / "task_thumbs"
DEMOS = REPO / "data" / "task_demos"
DOCS = REPO / "data" / "task_docs.json"
STATIC = {"/thumb/": (THUMBS, ".png", "image/png"),
          "/demo/": (DEMOS, ".mp4", "video/mp4")}

FAMILY_LABEL = {
    "robomimic": "robomimic", "libero": "LIBERO",
    "fabrica": "Fabrica", "robodojo": "RoboDojo", "ogbench": "OGBench",
    "roboarcade": "RoboARCade", "pix2dex": "SPD", "autobio": "AutoBio",
}
CATEGORY_LABEL = {
    CATEGORY_REAL_WORLD: "Real-world tasks",
    CATEGORY_GAME: "Games",
    CATEGORY_IN_PROGRESS: "In progress",
    None: "Unclassified",
}


def family_key(task):
    fam = task_spec(task).get("family", "")
    return "libero" if fam.startswith("libero") else fam


def slug(task):
    return task.replace("/", "_")


@functools.lru_cache(maxsize=None)
def task_spec(task):
    p = TASKS / task / "task.json"
    return json.loads(p.read_text()) if p.exists() else {}


def task_language(task):
    return html.escape(task_spec(task).get("language", ""))


def grouped(tasks):
    groups = {}
    for t in tasks:
        groups.setdefault(family_key(t), []).append(t)
    order = list(FAMILY_LABEL)
    return {k: groups[k] for k in sorted(groups, key=lambda k: (order.index(k) if k in order else 99, k))}


def categorized(tasks):
    groups = {}
    for task in tasks:
        declared = task_spec(task).get("category")
        category = declared if declared in TASK_CATEGORIES else None
        groups.setdefault(category, []).append(task)
    return {category: groups[category]
            for category in (CATEGORY_REAL_WORLD, CATEGORY_GAME, CATEGORY_IN_PROGRESS, None)
            if category in groups}


def task_card(t):
    lang = task_language(t)
    s = slug(t)
    vid = (f'<video class="demovid" src="/demo/{s}.mp4" muted loop playsinline '
           f'preload="none"></video>') if (DEMOS / f"{s}.mp4").exists() else ""
    return (
        f'<button class="task" data-task="{t}" title="{lang}">'
        f'<span class="thumbwrap">'
        f'<img class="thumb" src="/thumb/{s}.png" loading="lazy" alt="" '
        f'onerror="this.classList.add(\'noimg\')">'
        f'{vid}</span>'
        f'<span class="tid">{t}</span>'
        f'<span class="lang">{lang}</span></button>'
    )


def render_docs():
    if not DOCS.exists():
        return docs_page("Task documentation", DOCS_INTRO,
            '<p style="color:var(--ink3)">No docs yet — run <code>python tasks/add_task/task_docs.py</code>.</p>')
    docs = json.loads(DOCS.read_text())
    secs = []
    for fam, tasks in grouped(sorted(docs)).items():
        rows = []
        for task in tasks:
            d = docs[task]
            rows.append(
                f'<tr><td class="tid">{task}</td>'
                f'<td>{html.escape(d["language"])}</td>'
                f'<td>{_rand_groups(d["randomization"])}</td>'
                f'<td>{_success(task, d["success"])}</td>'
                f'<td class="mono">{d["max_len"]}</td></tr>')
        secs.append(f'<h2>{FAMILY_LABEL.get(fam, fam)} <span class="count">{len(rows)}</span></h2>'
                    f'<div class="tablewrap"><table><thead><tr><th>task</th><th>instruction</th>'
                    f'<th>randomization groups &mdash; objects in a group move together</th>'
                    f'<th>success</th><th>max&nbsp;len</th>'
                    f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')
    return docs_page("Task documentation", DOCS_INTRO, "".join(secs))


def docs_page(title, intro, rows):
    return PAGE_DOCS.replace("{{TITLE}}", title).replace("{{INTRO}}", intro).replace("{{ROWS}}", rows)


def _success(task, spec):
    from core.grade_episode import describe, predicate_names
    out = "".join(f'<div class="crit" style="padding-left:{d * 0.7:g}rem">'
                  f'{"&bull; " if d else ""}{html.escape(t)}</div>' for d, t in describe(spec, task))
    chips = "".join(f'<span class="pchip">{html.escape(p)}</span>'
                    for p in predicate_names(spec))
    return out + f'<div class="preds">{chips}</div>'


def _rand_groups(rand):
    def names(objs):
        return " ".join(f'#{o["num"]}&nbsp;{o["name"]}{"*" if o["static"] else ""}'
                        for o in objs)

    groups = {}
    for o in rand["objects"]:
        groups.setdefault(o["group"], []).append(o)
    lines = [f'<div class="obj"><span class="grp">G{g + 1}</span><b>{names(objs)}</b></div>'
             for g, objs in sorted(groups.items()) if g is not None]
    r = rand["region"]
    lines.append(f'<div class="obj"><span class="grp">range</span>'
                 f'<span class="tier">&plusmn;{r["x"][1] * 100:.0f}/{r["y"][1] * 100:.0f}cm '
                 f'&plusmn;{r["yaw"][1] * 57.3:.0f}&deg;</span>'
                 f'<span class="note">applies to every group</span></div>')
    return "".join(lines) or '<div class="obj">&mdash;</div>'


def render_dashboard():
    cards = []
    for category, category_tasks in categorized(list_tasks()).items():
        items = "".join(task_card(t) for fam_tasks in grouped(category_tasks).values()
                        for t in fam_tasks)
        cards.append(
            f'<section class="category"><h2>{CATEGORY_LABEL[category]} '
            f'<span class="count">{len(category_tasks)}</span></h2>'
            f'<div class="grid">{items}</div></section>'
        )
    return PAGE.replace("{{CARDS}}", "".join(cards))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path in ("/", "") or self.path.startswith("/?"):
            self._send(200, render_dashboard(), "text/html; charset=utf-8")
        elif self.path == "/docs":
            self._send(200, render_docs(), "text/html; charset=utf-8")
        elif self.path == "/viewer":
            host = self.headers.get("Host", "localhost").rsplit(":", 1)[0]
            self.send_response(302)
            self.send_header("Location", f"http://{host}:{VIEWER_PORT}/")
            self.end_headers()
        elif (st := STATIC.get(self.path[:self.path.find("/", 1) + 1])) and self.path.endswith(st[1]):
            f = st[0] / os.path.basename(self.path)
            self._send(200, f.read_bytes(), st[2]) if f.exists() else self._send(404, b"", st[2])
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path.startswith("/api/launch"):
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            task = qs.get("task", [""])[0]
            if task not in list_tasks():
                self._send(400, json.dumps({"error": f"unknown task {task}"}))
                return
            self.server.launch_task = task
            self._send(200, json.dumps({"ok": True, "task": task}))
            threading.Thread(target=self.server.shutdown, daemon=True).start()
        else:
            self._send(404, json.dumps({"error": "not found"}))


class _Server(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def serve_dashboard() -> str | None:
    httpd = _Server(("", BASE), Handler)
    httpd.launch_task = None
    print(f"[dashboard] http://localhost:{BASE}/  (ready)", flush=True)
    httpd.serve_forever()
    httpd.server_close()
    return httpd.launch_task


def run_teleop(task: str):
    env = dict(os.environ, ROBOARCADE_BASE_PORT=str(BASE), PYTHONPATH=str(REPO), ROBOARCADE_RENDER="mujoco")
    print(f"[dashboard] launching teleop: {task}", flush=True)
    proc = subprocess.Popen(
        [sys.executable, "-m", "core.sim_env_server", "--task", task, "--teleop", "--dashboard"],
        cwd=str(REPO), env=env,
    )
    proc.wait()
    print(f"[dashboard] teleop ended: {task}", flush=True)


def launch_viewer():
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", VIEWER_PORT)) == 0:
            print(f"[dashboard] 3D viewer DISABLED: :{VIEWER_PORT} is already in use "
                  f"(a leaked viewer from an earlier run?). Free it and restart.",
                  flush=True)
            return
    proc = subprocess.Popen(
        [sys.executable, "-m", "tasks.add_task.geom_viewer", "--port", str(VIEWER_PORT)],
        cwd=str(REPO), env=dict(os.environ, PYTHONPATH=str(REPO)),
    )
    atexit.register(lambda: proc.poll() is None and proc.terminate())
    print(f"[dashboard] 3D viewer on :{VIEWER_PORT}", flush=True)


def main():
    print("[dashboard] teleop dashboard — Ctrl-C to quit", flush=True)
    launch_viewer()
    while True:
        task = serve_dashboard()
        if not task:
            break
        run_teleop(task)


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>roboarcade · teleop dashboard</title>
<style>
  :root{color-scheme:light;--ink:#101418;--ink2:#4a545f;--ink3:#77828e;
    --surface:#fbfcfd;--surface2:#f1f4f7;--rule:#dde3ea;--accent:#2a78d6;
    --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
    --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;}
  @media (prefers-color-scheme:dark){:root{--ink:#f2f5f8;--ink2:#b3bdc8;--ink3:#7f8a96;
    --surface:#14171a;--surface2:#1d2126;--rule:#2e343b;--accent:#3987e5;}}
  *{box-sizing:border-box}
  body{margin:0;background:var(--surface);color:var(--ink);font-family:var(--sans);line-height:1.5}
  .wrap{max-width:60rem;margin:0 auto;padding:2.5rem 1.25rem 4rem}
  header{border-bottom:2px solid var(--ink);padding-bottom:1rem;margin-bottom:1.5rem;
    display:flex;justify-content:space-between;align-items:flex-start;gap:1rem 1.5rem;
    flex-wrap:wrap}
  .head-left{flex:1 1 22rem}
  .eyebrow{font-family:var(--mono);font-size:.7rem;letter-spacing:.14em;text-transform:uppercase;
    color:var(--ink3);margin:0 0 .4rem}
  h1{font-size:1.9rem;margin:0 0 .3rem;letter-spacing:-.02em}
  .sub{color:var(--ink2);margin:0;font-size:.95rem}
  .dbg-btn{flex-shrink:0;border:1px solid var(--accent);border-radius:6px;
    background:color-mix(in srgb,var(--accent) 10%,transparent);color:var(--accent);
    font-family:var(--mono);font-size:.82rem;font-weight:600;padding:.6rem .9rem;white-space:nowrap;
    cursor:pointer;transition:background .12s}
  .dbg-btn:hover{background:color-mix(in srgb,var(--accent) 20%,transparent)}
  section.category{margin:0 0 2.5rem}
  h2{font-size:1.25rem;margin:0 0 1rem;display:flex;align-items:baseline;gap:.5rem;
    border-bottom:1px solid var(--rule);padding-bottom:.4rem}
  h3{font-size:.82rem;letter-spacing:.08em;text-transform:uppercase;color:var(--ink2);
    margin:0 0 .55rem;display:flex;align-items:baseline;gap:.5rem}
  .count{font-family:var(--mono);font-size:.72rem;color:var(--ink3);font-weight:400}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(11rem,1fr));gap:.6rem}
  button{font:inherit;cursor:pointer}
  .task{border:1px solid var(--rule);background:var(--surface2);border-radius:6px;
    text-align:left;padding:.5rem;display:flex;flex-direction:column;gap:.25rem;
    color:var(--ink);transition:border-color .12s,transform .05s;min-width:0;overflow:hidden}
  .task:hover{border-color:var(--accent)}
  .task:active{transform:translateY(1px)}
  .task>*{min-width:0;max-width:100%}
  .thumb{width:100%;aspect-ratio:1/1;object-fit:cover;border-radius:4px;
    background:var(--surface);display:block}
  .thumbwrap{position:relative;display:block;width:100%}
  .thumbwrap .thumb{width:100%}
  .demovid{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;
    border-radius:4px;opacity:0;transition:opacity .15s;pointer-events:none}
  .task:hover .demovid.playing{opacity:1}
  .thumb.noimg{background:repeating-linear-gradient(45deg,var(--surface),var(--surface) 6px,
    var(--surface2) 6px,var(--surface2) 12px)}
  .tid{font-family:var(--mono);font-size:.78rem;font-weight:600;overflow:hidden;
    text-overflow:ellipsis;white-space:nowrap}
  .lang{font-size:.72rem;color:var(--ink3);line-height:1.35;
    display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
  #overlay{position:fixed;inset:0;background:color-mix(in srgb,var(--surface) 88%,transparent);
    display:none;align-items:center;justify-content:center;flex-direction:column;gap:1rem;z-index:10}
  #overlay.on{display:flex}
  .spin{width:34px;height:34px;border:3px solid var(--rule);border-top-color:var(--accent);
    border-radius:50%;animation:spin .8s linear infinite}
  @keyframes spin{to{transform:rotate(360deg)}}
  @media (prefers-reduced-motion:reduce){.spin{animation:none}}
  #overlay .msg{font-family:var(--mono);font-size:.85rem;color:var(--ink2)}
</style></head><body>
<div class="wrap">
  <header>
    <div class="head-left">
      <p class="eyebrow">roboarcade · teleop</p>
      <h1>Pick a task to drive</h1>
      <p class="sub">Click a task to open its teleop stack. End Episode (or Dashboard) brings you back here.</p>
    </div>
    <div style="display:flex;gap:.5rem;flex-wrap:wrap">
      <a class="dbg-btn" href="/docs" style="text-decoration:none">&#128196; Docs</a>
      <a class="dbg-btn" href="/viewer" style="text-decoration:none">&#129482; 3D viewer</a>
    </div>
  </header>
  {{CARDS}}
</div>
<div id="overlay"><div class="spin"></div><div class="msg" id="omsg">Launching…</div></div>
<script>
  const overlay=document.getElementById('overlay'), omsg=document.getElementById('omsg');
  async function launch(task){
    overlay.classList.add('on'); omsg.textContent='Launching '+task+' …';
    const r=await fetch('/api/launch?task='+encodeURIComponent(task),{method:'POST'});
    if(!r.ok){omsg.textContent='Failed to launch '+task; return;}
    omsg.textContent='Booting teleop for '+task+' … (~15s)';
    // poll until sim_env_server holds this port: only its server answers /instance.json
    for(let i=0;i<120;i++){
      await new Promise(s=>setTimeout(s,1000));
      try{const r=await fetch('/instance.json',{cache:'no-store'});
          if(r.ok){location.reload(); return;}}catch(_){}
    }
    omsg.textContent='Teleop did not come up — reload to retry.';
  }
  document.querySelectorAll('[data-task]').forEach(b=>b.addEventListener('click',
    ()=>launch(b.dataset.task)));
  document.querySelectorAll('.task').forEach(b=>{
    const v=b.querySelector('.demovid'); if(!v) return;
    b.addEventListener('mouseenter',()=>{v.classList.add('playing');v.play().catch(()=>{});});
    b.addEventListener('mouseleave',()=>{v.pause();v.currentTime=0;v.classList.remove('playing');});
  });
</script>
</body></html>"""


PAGE_DOCS = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>roboarcade · {{TITLE}}</title>
<style>
  :root{color-scheme:light;--ink:#101418;--ink2:#4a545f;--ink3:#77828e;
    --surface:#fbfcfd;--surface2:#f1f4f7;--rule:#dde3ea;--accent:#2a78d6;
    --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
    --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;}
  @media (prefers-color-scheme:dark){:root{--ink:#f2f5f8;--ink2:#b3bdc8;--ink3:#7f8a96;
    --surface:#14171a;--surface2:#1d2126;--rule:#2e343b;--accent:#3987e5;}}
  *{box-sizing:border-box}
  body{margin:0;background:var(--surface);color:var(--ink);font-family:var(--sans);line-height:1.5}
  .wrap{max-width:72rem;margin:0 auto;padding:2.5rem 1.25rem 4rem}
  header{border-bottom:2px solid var(--ink);padding-bottom:1rem;margin-bottom:1.5rem;
    display:flex;justify-content:space-between;align-items:baseline;gap:1rem}
  h1{font-size:1.9rem;margin:0;letter-spacing:-.02em}
  a.back{font-family:var(--mono);font-size:.85rem;color:var(--accent);text-decoration:none}
  h2{font-size:1rem;margin:2rem 0 .6rem;display:flex;align-items:baseline;gap:.5rem;
    border-bottom:1px solid var(--rule);padding-bottom:.4rem}
  .count{font-family:var(--mono);font-size:.72rem;color:var(--ink3);font-weight:400}
  .tablewrap{overflow-x:auto}
  table{border-collapse:collapse;width:100%;font-size:.82rem;min-width:52rem}
  th,td{text-align:left;padding:.5rem .6rem;border-bottom:1px solid var(--rule);vertical-align:top}
  th{font-size:.68rem;letter-spacing:.06em;text-transform:uppercase;color:var(--ink3);font-weight:500}
  .tid{font-family:var(--mono);font-weight:600;white-space:nowrap}
  .mono{font-family:var(--mono)}
  .obj{font-family:var(--mono);font-size:.74rem;color:var(--ink2);padding:.15rem 0;
    display:flex;flex-wrap:wrap;align-items:center;gap:.3rem}
  .obj b{color:var(--ink);min-width:14rem;flex:1}
  .crit{font-size:.74rem;color:var(--ink2);line-height:1.45;min-width:15rem}
  .preds{display:flex;flex-wrap:wrap;gap:.25rem;margin-top:.35rem}
  .pchip{font-family:var(--mono);font-size:.63rem;letter-spacing:.02em;
    padding:.1em .4em;border-radius:3px;background:var(--surface2);color:var(--ink3);
    border:1px solid var(--rule)}
  .obj .note{font-size:.68rem;color:var(--ink3)}
  .grp{font-size:.66rem;padding:.02em .4em;border-radius:3px;background:var(--surface2);
    color:var(--ink3);min-width:3.4rem;text-align:center}
  .tier{font-size:.68rem;padding:.05em .45em;border-radius:3px;color:#fff;background:#3f9153}
  code{font-family:var(--mono);background:var(--surface2);padding:.1em .35em;border-radius:3px}
</style></head><body>
<div class="wrap">
  <header><h1>{{TITLE}}</h1><a class="back" href="/">&larr; dashboard</a></header>
  <p style="color:var(--ink2);margin:0 0 1rem;font-size:.9rem">{{INTRO}}</p>
  {{ROWS}}
</div></body></html>"""

DOCS_INTRO = """With randomization OFF every object sits at its base pose. With it ON, each <b>group</b>
    below is drawn ONE shared jitter (&plusmn;x, &plusmn;y, &plusmn;yaw about the group
    centroid), so grouped objects keep their relative pose &mdash; an assembled fixture moves
    as a unit. Ranges are half-widths; a <code>*</code> marks a welded (jointless) scene body.
    Generated by <code>tasks/add_task/task_docs.py</code>."""


if __name__ == "__main__":
    main()
