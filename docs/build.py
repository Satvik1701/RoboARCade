"""Build the standalone docs site: docs/site/**.html from the repository's markdown (README, docs/**, package READMEs,
the interface guides) and the code map, one page each with a shared sidebar. Run from the repo root:

    .venv/bin/python scripts/code_map.py && .venv/bin/python docs/interfaces/build.py
    .venv/bin/python docs/architecture/task-rubrics/build.py && .venv/bin/python docs/build.py [--web https://github.com/<owner>/<repo>]

Open docs/site/index.html directly or serve the folder (cd docs/site && python3 -m http.server 8099). Without --web, links to
repository files point at the checkout two levels up. With --web the site is self-contained for a web server: the
architecture page and its assets are copied in, and every other repository file links to the GitHub blob."""
import html, re, shutil, sys
from pathlib import Path
import markdown

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "docs/site"
WEB = sys.argv[sys.argv.index("--web") + 1].rstrip("/") if "--web" in sys.argv else None
# the sidebar; other markdown files under docs/ are rendered (so links to them work) but not listed
GROUPS = [
    ("Start here", [("Overview", "README.md")]),
    ("Code", [("Code map", "docs/code/CODE_MAP.md"), ("Controller", "docs/CONTROLLER.md"), ("Sandbox", "docs/SANDBOX.md"),
              ("Grading", "docs/design/grading.md")]),
    ("Interfaces", [("VIA", "docs/interfaces/via_cam.md"), ("VIA bimanual", "docs/interfaces/via_bimanual_cam.md"),
                    ("Multi-cam RGBD", "docs/interfaces/rgbd_waypoint.md")]),
    ("Evaluation", [("Accounts", "docs/ACCOUNTS.md")]),
    ("Tasks", [("Tasks and subtasks", "docs/architecture/task-rubrics/rubrics.md"), ("Adding a task", "tasks/add_task/README.md")]),
    ("Dashboard", [("The dashboard", "docs/DASHBOARDS.md"), ("Cost accounting", "docs/COST_ACCOUNTING.md")]),
]
SKIP = ("docs/site/", "docs/code/IMPLEMENTATION_NOTES.md", "docs/architecture/task-rubrics/beam-", "docs/architecture/task-rubrics/pcr-")
CSS = """*{box-sizing:border-box}
:root{--bg:#F5F6F4;--panel:#FFF;--ink:#171B19;--ink2:#555E58;--rule:#D6DAD5;--accent:#17614B;--zebra:#EEF1EE}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#12150F;--panel:#1A1E19;--ink:#E8EBE5;--ink2:#A3ADA5;--rule:#2C332C;--accent:#74C7A6;--zebra:#20251F}}
body{margin:0;background:var(--bg);color:var(--ink);font:15.5px/1.55 "Iowan Old Style",Charter,Georgia,serif;display:grid;grid-template-columns:260px minmax(0,1fr);min-height:100vh}
nav{border-right:1px solid var(--rule);padding:18px 14px;font-size:13.5px;position:sticky;top:0;height:100vh;overflow:auto}
nav h1{font-size:15px;margin:0 0 10px}nav h1 a{color:var(--ink);text-decoration:none}
nav h2{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--ink2);margin:14px 0 4px}
nav a{display:block;color:var(--accent);text-decoration:none;padding:2px 6px;border-radius:4px}nav a.now{background:var(--zebra);color:var(--ink);font-weight:600}
main{padding:28px clamp(16px,4vw,48px) 60px;max-width:1000px}main img{max-width:100%}
h1{font-size:30px;line-height:1.15;margin:0 0 14px}h2{font-size:22px;border-bottom:2px solid var(--accent);padding-bottom:4px;margin:30px 0 10px}h3{font-size:16px;margin:20px 0 6px}
code{font:13px ui-monospace,SFMono-Regular,Menlo,monospace}pre{background:var(--panel);border:1px solid var(--rule);padding:10px 12px;overflow:auto;font-size:12.5px}
table{border-collapse:collapse;font:13px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--panel);border:1px solid var(--rule);margin:10px 0}
th,td{text-align:left;padding:5px 9px;border-bottom:1px solid var(--rule);vertical-align:top}thead th{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--ink2)}
tbody tr:nth-child(even){background:var(--zebra)}td.n{text-align:right;font-variant-numeric:tabular-nums}.src{font-size:12.5px;color:var(--ink2);margin:0 0 18px}.src a{color:var(--accent)}
@media(max-width:800px){body{grid-template-columns:minmax(0,1fr)}nav{position:static;height:auto;border-right:0;border-bottom:1px solid var(--rule)}}"""


def page_rel(src):
    """The site path of a rendered source; the README is the landing page."""
    return "index.html" if src == "README.md" else str(Path(src).with_suffix(".html"))


def page_of(src):
    return SITE / page_rel(src)


def repo_link(rel, up):
    """A repository file that is not a rendered page: an architecture page is copied into the site (so a served site
    is self-contained); anything else is the checkout two levels above docs/site, or on the web its GitHub blob."""
    if rel.startswith("docs/architecture/"):
        return f"{up}{rel}"
    return f"{WEB}/blob/main/{rel}" if WEB else f"{up}../../{rel}"


def links_to(pages, page, src=""):
    """href rewriting for one page: rendered markdown goes to its page, anything else in the repo through repo_link,
    external links stay."""
    up = "../" * (len(page.relative_to(SITE).parts) - 1)
    here = Path(src).parent
    def fix(m):
        attr, target = m.group(1), m.group(2)
        if re.match(r"^(https?:|mailto:|#|data:)", target):
            return m.group(0)
        path, _, frag = target.partition("#")
        resolved = (ROOT / here / path).resolve() if path else None
        if resolved is None or not resolved.is_relative_to(ROOT):
            return m.group(0)
        rel = str(resolved.relative_to(ROOT))
        anchor = "#" + frag if frag else ""
        return f'{attr}="{up}{page_rel(rel)}{anchor}"' if rel in pages else f'{attr}="{repo_link(rel, up)}{anchor}"'
    return fix, up


def shell(title, body, src, pages, page):
    fix, up = links_to(pages, page, src)
    nav = [f'<h1><a href="{up}index.html">RoboARCade docs</a></h1>']
    for group, items in GROUPS:
        nav.append(f"<h2>{html.escape(group)}</h2>")
        for t, s in items:
            href = s if s.startswith("http") else f"{up}{page_rel(s)}" if s in pages else repo_link(s, up)
            nav.append(f'<a href="{href}" class="{"now" if s == src else ""}">{html.escape(t)}</a>')
    source = f'<p class="src">Source: <a href="{repo_link(src, up)}">{html.escape(src)}</a></p>' if src else ""
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{html.escape(title)} · RoboARCade docs</title><style>{CSS}</style></head><body><nav>{"".join(nav)}</nav>'
            f'<main>{source}{body}</main></body></html>')


def render(src, title, pages):
    page = page_of(src); page.parent.mkdir(parents=True, exist_ok=True)
    body = markdown.markdown((ROOT / src).read_text(), extensions=["tables", "fenced_code", "toc", "sane_lists"])
    fix, _up = links_to(pages, page, src)
    page.write_text(shell(title, re.sub(r'(href|src)="([^"]+)"', fix, body), src, pages, page))


REF = re.compile(r'(?:href|src)=["\']([^"\'#?]+)|["\']((?:[\w.-]+/)*[\w.-]+\.(?:png|jpe?g|svg|mp4|json|js|css|html|md|txt|patch))["\']')


def copy_architecture():
    """Copy into the site only the architecture files the pages reach: every local link, script, style and quoted
    media path, followed transitively through the copied html, js and css (the recorded rollout's screenshots come
    along because the trace names them). The records and check screenshots beside them stay out."""
    src, dst = ROOT / "docs/architecture", SITE / "docs/architecture"
    seen = set()

    def follow(q, doc):
        if q.is_file() and q.is_relative_to(src) and q not in seen:
            seen.add(q)
            if q.suffix == ".html":
                scan(q.read_text(errors="ignore"), q.parent)
            elif q.suffix in (".js", ".css"):          # a script's paths resolve against the page that loads it
                scan(q.read_text(errors="ignore"), doc)

    def scan(text, doc):
        for m in REF.finditer(text):
            t = m.group(1) or m.group(2)
            if not re.match(r"^(https?:|mailto:|data:|/)", t):
                q = (doc / t).resolve()
                follow(src / q.relative_to(dst) if q.is_relative_to(dst) else q, doc)

    for page in SITE.rglob("*.html"):
        scan(page.read_text(), page.parent)
    for q in seen:
        (dst / q.relative_to(src)).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(q, dst / q.relative_to(src))
    print(f"copied {len(seen)} referenced architecture files, {sum(q.stat().st_size for q in seen) / 1e6:.0f} MB")


def title_of(src):
    for _g, items in GROUPS:
        for t, s in items:
            if s == src:
                return t
    text = (ROOT / src).read_text()
    if src.endswith(".html"):
        return html.unescape(re.search(r"<title>(.*?)</title>", text).group(1))
    first = next((l for l in text.splitlines() if l.startswith("#")), src)
    return first.lstrip("# ").strip()


def main():
    if SITE.exists():
        shutil.rmtree(SITE)
    sources = ["README.md", "tasks/add_task/README.md"]
    sources += [str(p.relative_to(ROOT)) for p in sorted((ROOT / "docs").rglob("*.md"))]
    sources = [s for s in dict.fromkeys(sources) if not s.startswith(SKIP)]
    pages = {s: title_of(s) for s in sources}
    for s in sources:
        render(s, pages[s], pages)
    copy_architecture()
    print(f"wrote {len(sources)} pages to {SITE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
