"""One folder per episode: data/episodes/<task>/<model>/<stamp>_seed<k>_<run hash>/.

The evaluator publishes a seed here the moment it is graded (hard links of the seed folder: episode.json,
verdict.json, cameras_states.npz, the transcript, screenshots and videos; no extra disk), plus meta.json (task,
model, interface, the launch it came from), subtasks.json (core.grade_episode, run once here) and summary.json.
The dashboard reads nothing else. `python -m core.publish_episode` publishes every seed under data/run_eval once.
"""
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

from core.agent_sandbox import MODEL_IDS

REPO = Path(__file__).resolve().parents[1]
EPISODES = REPO / "data" / "episodes"


def _read(folder, name):
    return json.loads((folder / name).read_text()) if (folder / name).is_file() else None


def _link(src, dst):
    if dst.exists():
        if os.path.samefile(src, dst):
            return
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _grade(roll):
    from core.grade_episode import graded
    try:
        graded(roll)
    except (OSError, ValueError, ImportError) as exc:
        (roll / "grading_error.txt").write_text(f"{type(exc).__name__}: {exc}\n")


def score(roll):
    subs, verdict = _read(roll, "subtasks.json"), _read(roll, "verdict.json")
    if subs:
        value, source = subs["score"], subs["source"]
    elif verdict and verdict.get("wall_clock_timeout"):
        value, source = 0.0, "wall-clock timeout"
    else:
        return None, None, None
    return value, source, value == 1


def attach_logs(out, run_dir, label):
    num = label.removeprefix("seed")
    wanted = [(f"logs/*seed{num}_*sim.log", "sim.log")]
    if not (out / "codex_session.jsonl").exists():
        wanted.append((f"logs/*seed{num}_port*claude*.log", "claude.log"))
    for pattern, name in wanted:
        src = next(iter(sorted(run_dir.glob(pattern))), None)
        if src is not None:
            _link(src, out / name)


def publish(seed_dir, run_dir=None):
    seed_dir = Path(seed_dir).resolve()
    run_dir = Path(run_dir).resolve() if run_dir else seed_dir.parent
    run_meta = _read(run_dir, "meta.json") or {}
    episode = _read(seed_dir, "episode.json") or {}
    task = episode.get("task") or run_meta.get("task") or ""
    model = episode.get("model") or run_meta.get("model") or "unknown"
    model = MODEL_IDS.get(model, model)
    label = seed_dir.name
    stamp = run_dir.name if run_dir.name[:8].isdigit() else next((p for p in run_dir.parts[::-1] if p[:8].isdigit()), run_dir.name)
    eval_root = REPO / "data" / "run_eval"
    run = str(run_dir.relative_to(eval_root)) if run_dir.is_relative_to(eval_root) else str(run_dir)
    out = EPISODES / task / model / f"{stamp}_{label}_{hashlib.sha1(run.encode()).hexdigest()[:6]}"
    for src in (q for q in seed_dir.rglob("*") if q.is_file()):
        dst = out / src.relative_to(seed_dir)
        dst.parent.mkdir(parents=True, exist_ok=True)
        _link(src, dst)
    out.mkdir(parents=True, exist_ok=True)
    if (seed_dir / "verdict.json").is_file() and not (out / "subtasks.json").exists():
        _grade(out)
    meta = dict(task=task, model=model, seed=episode.get("seed"), label=label, run=run, stamp=stamp,
                interface=Path(run_meta.get("interface") or episode.get("interface") or "via_cam").name,
                final=bool(run_meta.get("final")), case_study=run_meta.get("case_study"),
                status=run_meta.get("status"), source=str(seed_dir),
                solved=score(out)[2], run_meta=run_meta)
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    attach_logs(out, run_dir, label)
    (out / "summary.json").write_text(json.dumps(summarize(out, model), indent=1) + "\n")
    return out


def _render_seconds(roll):
    log = Path(roll) / "sim.log"
    ms = [int(x) for x in re.findall(r"render\] \d+ cameras in (\d+) ms", log.read_text(errors="replace"))] if log.is_file() else []
    return sum(ms) / 1000 if ms else None


def summarize(roll, model):
    from core.agent_log import transcript_file, transcript_replies, transcript_tools
    roll = Path(roll)
    verdict, episode = _read(roll, "verdict.json") or {}, _read(roll, "episode.json") or {}
    tf = transcript_file(roll)
    calls = transcript_tools(tf) if tf else None
    codex = episode.get("agent_runtime") == "codex"
    metrics = episode.get("metrics") or {}
    turns = metrics.get("num_turns") if codex else (transcript_replies(tf) if tf else None)
    if (roll / "waypoints.jsonl").is_file():
        eef = sum(1 for line in (roll / "waypoints.jsonl").open() if line.strip())
    else:
        segs = _read(roll, "cameras_segments.json")
        eef = len(segs["segments"]) if segs and "segments" in segs else None
    value, source, solved = score(roll)
    wall, render = verdict.get("elapsed_seconds"), _render_seconds(roll)
    out = dict(solved=solved, timed_out=bool(verdict.get("timed_out")), ended_by=verdict.get("ended_by"),
               score=value, score_source=source, sim_steps=verdict.get("sim_steps_used"),
               wall_seconds=wall, agent_wall_seconds=wall - render if wall is not None and render is not None else None,
               eef_commands=eef, model_turns=turns, tool_calls=len(calls) if calls is not None else None,
               tokens=metrics.get("tokens_budget"),
               cost_usd=metrics.get("cost_usd"))
    if (roll / "recovered_from").is_dir():
        before = summarize(roll / "recovered_from", model)
        add = lambda a, b: a + b if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None
        for k in ("cost_usd", "eef_commands", "tool_calls", "wall_seconds", "agent_wall_seconds", "tokens") + (() if codex else ("model_turns",)):
            out[k] = add(out[k], before[k])
    return out


if __name__ == "__main__":
    eval_root = REPO / "data" / "run_eval"
    n = 0
    for mp in sorted(eval_root.glob("*/meta.json")) + sorted(eval_root.glob("*/*/meta.json")):
        run_dir = mp.parent
        for seed_dir in sorted(run_dir.glob("seed*")):
            if seed_dir.is_dir() and (seed_dir / "episode.json").exists():
                publish(seed_dir, run_dir)
                n += 1
                if n % 200 == 0:
                    print(n, flush=True)
    print(f"published {n} episodes under {EPISODES}", file=sys.stderr)
