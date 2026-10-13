"""Write docs/architecture/task-rubrics/rubrics.md: per task its instruction, the success predicate the simulator
defines, its subtasks (grade.py, what every rollout is scored with) and every file of its folder, expandable, with its
path from the repository root. Read from tasks/ only. Run from the repo root:
.venv/bin/python docs/architecture/task-rubrics/build.py"""
import html
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from core.grade_episode import describe, task_grade     # noqa: E402
from core.tasks import TASKS, list_tasks     # noqa: E402

INLINE = 300_000      # bytes: larger files (three scene.xml) are linked, not inlined
FILES = ("task.json", "grade.py", "dynamics.py", "render.py", "test.py", "scene.xml", "manifest.json")


def rows(tree, task):
    return "".join(f'<div style="padding-left:{d * 1.1:g}em">{"&bull; " if d else ""}{html.escape(t)}</div>'
                   for d, t in describe(tree, task))


def when(t, last):
    if last:
        out = "the task's success (task.json's tree above) holds at the end, for half a second"
    else:
        out = "once its check held" + (" after " + ", ".join(f"#{j + 1}" for j in t.needs) if t.needs else "") if t.ever else "its check holds at the end"
    return out + ("; and " + ", ".join(f"#{j + 1}" for j in t.end_needs) + " done at the end" if t.end_needs else "")


def files(task):
    out = []
    for name in FILES:
        f = TASKS / task / name
        if not f.is_file():
            continue
        rel, size = f.relative_to(ROOT).as_posix(), f.stat().st_size
        body = (f"<pre>{html.escape(f.read_text(errors='replace'))}</pre>" if size <= INLINE
                else f'<p>{size // 1000:,} KB, too large to show here: <code>{rel}</code></p>')
        out.append(f"<details><summary><code>{rel}</code> ({max(size // 1000, 1):,} KB)</summary>{body}</details>")
    return "\n".join(out)


def section(task):
    spec = json.loads((TASKS / task / "task.json").read_text())
    module = task_grade(task)
    title = task.replace("_", " ")
    out = [f"## {title}\n\n{html.escape(spec['language'])}\n\n"
           f"<details><summary><b>Success</b> (task.json; the last subtask)</summary>"
           f"<div style=\"max-height:22em;overflow-y:auto;border:1px solid var(--rule);padding:6px 10px\">{rows(spec['success'], task)}</div></details>\n"]
    subs = module.subtasks()
    body = "".join(f"<tr><td>{i}</td><td>{html.escape(t.label)}</td><td>{when(t, i == len(subs))}</td></tr>"
                   for i, t in enumerate(subs, 1))
    inv = " An invariant (`INVARIANT` in grade.py) freezes the grade at its first break." if hasattr(module, "INVARIANT") else ""
    out.append(f"\n**Subtasks** (`tasks/{task}/grade.py`): the score is the share done; solved when all are.{inv}\n\n"
               f"<table><thead><tr><th>#</th><th>subtask</th><th>done when</th></tr></thead><tbody>{body}</tbody></table>\n")
    out.append(f"\n**Files**\n\n{files(task)}\n")
    return "".join(out)


tasks = sorted(list_tasks())
page = ("# Tasks and subtasks\n\n"
        f"Every task of `tasks/`: its instruction, the success predicate, the subtasks "
        f"every rollout is scored with (grade.py; the rule is on the Grading page) and every file of the "
        f"task's folder, expandable, with its path from the repository root.\n\n" + "\n".join(section(t) for t in tasks))
(HERE / "rubrics.md").write_text(page)
print(f"wrote {HERE / 'rubrics.md'} ({len(page) // 1024} KB, {len(tasks)} tasks)")
