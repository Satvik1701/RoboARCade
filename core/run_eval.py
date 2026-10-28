"""Eval launcher: one task, N seeds.

Each seed is an isolated stack on its own ROBOARCADE_BASE_PORT: launch sim_env_server, launch a
sandboxed agent whose MCP server attaches to that seed's browser, wait for the verdict.

    python -m core.run_eval --task bowl_on_plate --seeds 5 -j 4          # 4 seeds at a time
    python -m core.run_eval --task bowl_on_plate --run-dir DIR --print-results   # verdicts on disk only

--interface names the package under evaluation (via_cam, via_bimanual_cam, or an interface
development artifact path); its guide.md is appended to the system prompt.
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from datetime import datetime
from pathlib import Path

import numpy as np

from core.agent_sandbox import AUTO_COMPACT_WINDOW, MODEL_IDS, REPO_ROOT, model_runtime, require_codex_home
from core.accounts import PLAN_5H_TOKENS, PLAN_7D_TOKENS, pick_plan
from core.tasks import resolve_task
from core.publish_episode import publish as publish_episode
from core import run_seed
from core.run_seed import (EVAL_SEED_BASE, MCP_CONFIG_PATH, Experiment, allowed_tools,
                              build_prompt, finished_result, interface_guide, kill_ports, stack_ports, terminate_group)


from core.utils import get_json_file, interface_arm_count, interface_dir, episode_identity, setup_logging

logger = logging.getLogger(__name__)


def task_slug(task: str) -> str:
    return task.replace("/", "_")


PORT_STRIDE = 100


RUN_STAMP = os.environ.get("ROBOARCADE_RUN_STAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")


def git_commit() -> str:
    return subprocess.run(["git", "-C", REPO_ROOT, "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def run_dir_for(exp: Experiment) -> str:
    return os.path.join(REPO_ROOT, exp.data_root, RUN_STAMP)


def latest_run_dir(exp: Experiment) -> str:
    root = os.path.join(REPO_ROOT, exp.data_root)
    stamped = sorted(glob.glob(os.path.join(root, "[0-9]" * 8 + "_" + "[0-9]" * 6)))
    return stamped[-1] if stamped else root


def staged_package(interface: str) -> str:
    if os.path.sep in interface or os.path.isdir(interface):
        return interface
    name = interface.replace("-", "_")
    src = interface_dir(interface)
    if not src.is_dir():
        return interface
    h = hashlib.sha256()
    for f in sorted(src.rglob("*")):
        if f.is_file() and "__pycache__" not in f.parts:
            h.update(f.relative_to(src).as_posix().encode() + b"\0" + f.read_bytes())
    out = Path(REPO_ROOT, "data", "interfaces", f"{name}-{h.hexdigest()[:8]}")
    if not out.is_dir():
        tmp = out.with_name(f"{out.name}.{os.getpid()}")
        shutil.copytree(src, tmp, ignore=shutil.ignore_patterns("__pycache__"))
        try:
            os.rename(tmp, out)
        except OSError:
            shutil.rmtree(tmp)
    return str(out)


def render_replay(episode: str, seed: int | None = None) -> bool:
    states = os.path.join(episode, "cameras_states.npz")
    out = os.path.join(episode, "blender_feeds.mp4")
    if os.path.isfile(out):
        return True
    if not os.path.isfile(states) or not os.path.isfile(os.path.join(episode, "verdict.json")):
        return False
    t0 = time.monotonic()
    if subprocess.run([sys.executable, "-m", "core.render.replay", episode], cwd=REPO_ROOT).returncode:
        logger.warning(f"[video] seed {seed}: render failed")
        return False
    elapsed = time.monotonic() - t0
    with np.load(states) as saved:
        n_frames = len(saved["states"])
    if seed is None:
        _, seed = episode_identity(episode)
    logger.info(f"[video] seed {seed}: {out} "
                f"({elapsed / n_frames * 1000:.0f} ms/frame, {elapsed / 60:.1f} min)")
    return True


def render_replays(run_dir: str) -> None:
    rolls = [d for d in sorted(glob.glob(os.path.join(run_dir, "*")))
             if os.path.isfile(os.path.join(d, "cameras_states.npz"))
             and not os.path.isfile(os.path.join(d, "blender_feeds.mp4"))]
    if not rolls:
        return
    logger.info(f"[video] run is graded and every stack is down; rendering "
                f"{len(rolls)} replay video(s)")
    for d in rolls:
        render_replay(d)


def print_one(res: dict, tag: str = "") -> None:
    s = res.get("success")
    mark = "✓" if s else ("·" if s is None else "✗")
    detail = ""
    if res.get("sim_steps_used") is not None:
        detail = f"  steps={res['sim_steps_used']}  {res.get('elapsed_seconds')}s"
    flags = ""
    if res.get("timed_out"):
        flags += " [timed_out]"
    if res.get("reused"):
        flags += " [reused]"
    if res.get("status") not in ("ok", None):
        flags += f" [{res.get('status')}]"
    label = f"{tag}  " if tag else ""
    logger.info(f"  [{mark}] {label}seed {res.get('seed')}{detail}{flags}")


def summarize_and_write(
    results, exp: Experiment, name: str, out_dir: str, interrupted: bool,
    interface: str, write: bool = True
) -> None:
    n = len(results)
    n_success = sum(1 for r in results if r.get("success"))
    n_graded = sum(1 for r in results if r.get("success") is not None)
    rate = (n_success / n_graded) if n_graded else 0.0

    m = [r.get("metrics") or {} for r in results]
    totals = {k: sum(x.get(k) or 0 for x in m)
              for k in ("tokens_budget", "tokens_in", "tokens_out", "cache_read")}
    totals["cost_usd"] = round(sum(x.get("cost_usd") or 0 for x in m), 4)
    if model_runtime(exp.model) == "antigravity" and any(x.get("cost_usd") is None for x in m):
        totals["cost_usd"] = None
    totals["cost_estimated_seeds"] = sum(1 for x in m if x.get("cost_estimated"))
    totals["compactions"] = sum(len(x.get("compactions") or ()) for x in m)
    totals["wall_seconds"] = round(sum(x.get("wall_seconds") or 0 for x in m), 1)
    totals["api_seconds"] = round(sum(x.get("api_seconds") or 0 for x in m), 1)
    billed = [((r.get("metrics") or {}).get("billed_to") or (r.get("auth") or {}).get("auth"))
              for r in results]
    totals["on_plan"] = billed.count("plan")
    totals["on_api_key"] = billed.count("api_key")
    tok = totals["tokens_budget"]
    plan_est = ({"five_hour_pct": round(100 * tok / PLAN_5H_TOKENS, 1),
                 "seven_day_pct": round(100 * tok / PLAN_7D_TOKENS, 1)}
                if tok and model_runtime(exp.model) == "claude" else None)

    if write:
        meta_path = os.path.join(out_dir, "meta.json")
        meta = get_json_file(meta_path) or {}
        meta.update({"task": exp.task, "model": exp.model,
                     "effort": None if model_runtime(exp.model) == "antigravity" else exp.effort,
                     "interface": interface,
                     "render": exp.render, "plan": exp.plan,
                     "auto_compact_window": None if model_runtime(exp.model) == "antigravity" else AUTO_COMPACT_WINDOW,
                     "status": "interrupted" if interrupted else "done",
                     "totals": totals, "plan_est": plan_est})
        if model_runtime(exp.model) == "antigravity":
            from core.antigravity import runtime_metadata
            meta.update(runtime_metadata(exp.effort))
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=1)

    logger.info("\n=== summary ===")
    if totals["cost_usd"] or totals["tokens_budget"]:
        cost = f"${totals['cost_usd']:.2f}" if totals["cost_usd"] is not None else "unknown"
        logger.info(f"  cost: {cost} · {totals['tokens_budget']:,} tokens"
                    + (f" · plan ≈{plan_est['five_hour_pct']}% 5h, {plan_est['seven_day_pct']}% wk"
                       if plan_est else "")
                    + f" · seeds on plan/key: {totals['on_plan']}/{totals['on_api_key']}"
                    + (f"  [>= : {totals['cost_estimated_seeds']} seed(s) priced from tokens,"
                       f" lower bound]" if totals.get("cost_estimated_seeds") else ""))
    logger.info(f"  run: {name}  (task: {exp.task}, interface: {interface}, "
                f"render: {exp.render})")
    ungraded = n - n_graded
    ungraded_note = f"  ({ungraded} ungraded)" if ungraded else ""
    logger.info(f"  success: {n_success}/{n_graded} graded = {rate * 100:.1f}%{ungraded_note}")
    if interrupted:
        logger.info("  (interrupted before all seeds finished)")


def grouped_runs(exp, seeds, ports, max_parallel, logs_dir, tools, interface, extras,
                 group_factory=None, seed_runner=None):
    from core.render.group import RenderGroup, lane_capacities
    group_factory = group_factory or RenderGroup
    seed_runner = seed_runner or run_seed.run_seed
    pending = iter(zip(seeds, ports))
    startup_lock = threading.Lock()
    pool = ThreadPoolExecutor(max_workers=max_parallel)
    slots, futures = {}, {}

    def admit(slot, capacity):
        batch = []
        for _ in range(capacity):
            pair = next(pending, None)
            if pair is None:
                break
            batch.append(pair)
        if not batch or run_seed.SHUTDOWN.is_set():
            return
        group = group_factory([seed for seed, _ in batch],
                              Path(logs_dir).parent / "render_groups", startup_lock)
        slots[slot] = {"group": group, "remaining": len(batch), "capacity": capacity}
        for seed, port in batch:
            future = pool.submit(seed_runner, exp, seed, port, logs_dir, tools, interface,
                                 extras, render_group=group)
            futures[future] = (slot, seed)

    completed = False
    try:
        for slot, capacity in enumerate(lane_capacities(max_parallel, exp.seeds_per_blender)):
            admit(slot, capacity)
        while futures:
            done, _ = wait(futures, timeout=.5, return_when=FIRST_COMPLETED)
            if run_seed.SHUTDOWN.is_set():
                break
            for future in done:
                slot, seed = futures.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    logger.exception(f"seed{seed} crashed")
                    result = {"seed": seed, "status": "error", "success": None, "error": repr(exc)}
                yield result
                item = slots[slot]
                item["remaining"] -= 1
                if item["remaining"] == 0:
                    item["group"].close()
                    del slots[slot]
                    admit(slot, item["capacity"])
        completed = not futures
    finally:
        if not completed:
            run_seed.SHUTDOWN.set()
        pool.shutdown(wait=False, cancel_futures=True)
        cleanup_errors = []
        for item in slots.values():
            try:
                item["group"].close()
            except Exception as exc:
                cleanup_errors.append(exc)
                logger.exception("Could not close render group")
        if cleanup_errors:
            raise RuntimeError("Render group cleanup failed; inspect render_groups logs") from cleanup_errors[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one task across N seeds, -j of them at a time.",
    )
    parser.add_argument("-j", "--max-parallel", type=int, default=1)
    parser.add_argument(
        "--interface",
        default="via_cam",
        help="interface package under evaluation: a shipped name (via_cam, via_bimanual_cam) or a "
        "path to a package dir with tools.py (+ optional guide.md).",
    )
    parser.add_argument("--print-results", action="store_true", help="no run; print the summary of --run-dir.")
    parser.add_argument("--isolated", dest="isolated", action="store_true", default=True,
                        help="run each agent in a bwrap sandbox (default).")
    parser.add_argument("--no-isolated", dest="isolated", action="store_false",
                        help="UNSAFE: run the agent unsandboxed (answer key reachable).")
    parser.add_argument("--task", required=True, help="task folder name, e.g. bowl_on_plate.")
    parser.add_argument("--seeds", type=int, default=2, help="number of seeds (default 2).")
    parser.add_argument("--seed-start", type=int, default=EVAL_SEED_BASE,
                        help="first seed (default 1000000).")
    parser.add_argument("--model", default="opus", help="agent model (default opus).")
    parser.add_argument("--ready-timeout", type=float, default=None, metavar="SECONDS",
                        help="how long to wait for a seed's sim + browser to come up (default 600).")
    parser.add_argument("--wall-clock", type=float, default=None, metavar="SECONDS",
                        help="wall-clock cap on one episode, resumes included "
                             "(default 21600; the sim budget is separate: "
                             "termination.max_len in the pack).")
    parser.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"],
                        help="reasoning effort (default xhigh; Gemini SDK maps xhigh/max to high).")
    parser.add_argument("--render", default=None, choices=["blender", "mujoco"],
                        help="renderer for the agent's camera feeds "
                             "(default blender; mujoco is the fast rasteriser).")
    parser.add_argument("--seeds-per-blender", type=int, choices=(1, 2, 3, 4, 5), default=None,
                        help="seeds sharing each Blender (default 1: dedicated; ceiling 5, experimental). "
                             "With -j9 and K3, three fixed groups run concurrently. Linux/GPU only.")
    parser.add_argument("--language", default="", help="natural-language goal (default: the task's).")
    parser.add_argument("--base-port", type=int, default=8100,
                        help="first port stack (default 8100; set it clear of the dashboards).")
    parser.add_argument("--run-dir", help="run dir (default data/run_eval/<task>_<model>_<stamp>).")
    parser.add_argument("--instruct", action="append", default=[], metavar="FILE",
                        help="a system-prompt file every seed reads after the guide; repeatable.")
    parser.add_argument("--plan", default=None, metavar="NAME",
                        help="bill a subscription login from secrets/<runtime>_plans.txt: "
                             "a name, or auto for the plan with the most room in its rate "
                             "windows; without it the API key file decides")
    parser.add_argument("--no-video", action="store_true",
                        help="skip the default post-seed Blender replay. Replays never run "
                             "while another seed stack is live.")
    args = parser.parse_args()
    if args.model:
        args.model = MODEL_IDS.get(args.model, args.model)
    setup_logging()
    args.interface = staged_package(args.interface)
    if not os.path.isdir(args.interface):
        parser.error(f"--interface {args.interface!r}: no such package dir, and no shipped "
                     "interface by that name under interfaces/")

    if args.max_parallel < 1:
        parser.error("--max-parallel must be >= 1")
    name = (os.path.basename(args.run_dir.rstrip("/")) if args.run_dir
            else f"{task_slug(args.task)}_{args.model}_{time.strftime('%m%d_%H%M%S')}")
    exp = Experiment(task=args.task, language=args.language, n_seeds=args.seeds, seed_start=args.seed_start,
                     data_root=args.run_dir or os.path.join("data/run_eval", name), base_port=args.base_port,
                     model=args.model, isolated=args.isolated,
                     **{k: v for k, v in (("effort", args.effort), ("render", args.render),
                                          ("seeds_per_blender", args.seeds_per_blender),
                                          ("episode_timeout", args.wall_clock),
                                          ("ready_timeout", args.ready_timeout)) if v is not None})
    spec = json.loads((resolve_task(exp.task) / "task.json").read_text())
    if args.wall_clock is None and spec.get("termination", {}).get("episode_timeout"):
        exp = dataclasses.replace(exp, episode_timeout=spec["termination"]["episode_timeout"])
    if exp.seeds_per_blender > 1:
        if exp.render != "blender":
            parser.error("--seeds-per-blender >1 requires Blender")
        if sys.platform != "linux":
            parser.error("Shared Blender currently requires Linux")
    if model_runtime(exp.model) == "antigravity":
        if not exp.isolated:
            parser.error("Antigravity requires sandboxing")
        from core.antigravity import api_key, validate_runtime, backend
        if args.plan:
            parser.error("Antigravity API mode requires API-key billing; omit --plan")
        if args.effort and backend() != "sdk":
            parser.error("Explicit Gemini thinking levels require ROBOARCADE_ANTIGRAVITY_BACKEND=sdk")
        api_key()
        validate_runtime()
    if args.plan:
        plan, plan_dir = pick_plan(model_runtime(exp.model), args.plan)
        exp = dataclasses.replace(exp, plan=plan, plan_dir=plan_dir, plan_auto=args.plan == "auto")
        if model_runtime(exp.model) == "claude":
            if os.path.basename(plan_dir) != "store":
                parser.error(f"plan {plan!r} must point at an auth-only root's store/, not {plan_dir}")
            os.environ["ROBOARCADE_CLAUDE_AUTH_DIR"] = os.path.dirname(plan_dir)
            os.environ["CLAUDE_CONFIG_DIR"] = plan_dir
    if interface_arm_count(args.interface) > len(spec.get("robots", ())):
        parser.error(f"interface {args.interface!r} is bimanual but {exp.task!r} is a one-arm pack")
    out_dir = latest_run_dir(exp) if args.print_results else run_dir_for(exp)
    if not args.print_results and model_runtime(exp.model) == "codex":
        require_codex_home(exp.plan_dir)
    seeds = [exp.seed_start + i for i in range(exp.n_seeds)]
    ports = [exp.base_port + i * PORT_STRIDE for i in range(exp.n_seeds)]
    if not args.print_results:
        mp = os.path.join(run_dir_for(exp), "meta.json")
        os.makedirs(os.path.dirname(mp), exist_ok=True)
        meta = get_json_file(mp) or {}
        if meta and model_runtime(exp.model) == "antigravity":
            from core.antigravity import runtime_metadata
            expected = runtime_metadata(exp.effort)
            if any(meta.get(k) != expected[k] for k in ('antigravity_backend', 'thinking_level')):
                parser.error('Existing run uses a different Gemini thinking configuration; use a new --run-dir')
        meta.update({"task": exp.task, "model": exp.model,
                     "effort": None if model_runtime(exp.model) == "antigravity" else exp.effort,
                     "interface": args.interface, "render": exp.render,
                     "seeds_per_blender": exp.seeds_per_blender,
                     "episode_timeout": exp.episode_timeout,
                     "language": exp.language,
                     "plan": exp.plan,
                     "status": "running", "isolated": exp.isolated,
                     "seed_list": seeds, "commit": git_commit(),
                     "created": meta.get("created")
                                or datetime.now().isoformat(timespec="seconds")})
        if model_runtime(exp.model) == "antigravity":
            from core.antigravity import runtime_metadata
            meta.update(runtime_metadata(exp.effort))
        json.dump(meta, open(mp, "w"), indent=1)

    if args.print_results:
        logger.info(f"=== results: run={name}  task={exp.task} ===")
        results = []
        for seed, port in zip(seeds, ports):
            demo_folder = os.path.join(out_dir, f"seed{seed}")
            res = finished_result(seed, port, demo_folder, exp.task) or {
                "seed": seed,
                "base_port": port,
                "demo_folder": demo_folder,
                "status": "no_verdict",
                "success": None,
            }
            results.append(res)
            print_one(res)
        summarize_and_write(results, exp, name, out_dir, interrupted=False,
                            interface=args.interface, write=False)
        return

    tools = allowed_tools(MCP_CONFIG_PATH)
    logs_dir = os.path.join(out_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    max_parallel = min(args.max_parallel, exp.n_seeds)
    if exp.render == "blender":
        from core import render as _render
        ok, why = _render.available()
        if not ok:
            raise SystemExit(f"--render blender but {why}")

    guide = interface_guide(args.interface)
    logger.info("\n".join([
        f"=== eval: {name}, {exp.n_seeds} seeds, {max_parallel} at a time ===",
        f"  interface: {args.interface}",
        f"  mcp tools allowed: {' '.join(tools)}",
        f"  task={exp.task!r}  model={exp.model}  "
        f"seeds->ports={dict(zip(seeds, ports))}",
        f"=== agent prompt ===\n{build_prompt(exp.language)}",
        f"=== interface guide ===\n{open(guide).read() if guide else 'None'}",
    ]))

    results: list[dict] = []
    all_ports = list(ports)
    interrupted = errored = False
    pool = None
    try:
        extras: list[str] = list(args.instruct)
        if max_parallel == 1 and exp.seeds_per_blender == 1:
            for seed, port in zip(seeds, ports):
                try:
                    res = run_seed.run_seed(exp, seed, port, logs_dir, tools, args.interface, extras)
                except Exception as e:
                    logger.exception(f"seed{seed} crashed")
                    res = {"seed": seed, "status": "error", "success": None, "error": repr(e)}
                results.append(res)
                print_one(res)
                if not args.no_video:
                    render_replay(res.get("demo_folder", ""), seed)
        elif exp.seeds_per_blender > 1:
            for res in grouped_runs(exp, seeds, ports, max_parallel, logs_dir, tools,
                                    args.interface, extras):
                results.append(res)
                print_one(res)
        else:
            pool = ThreadPoolExecutor(max_workers=max_parallel)
            futs = {pool.submit(run_seed.run_seed, exp, seed, port, logs_dir, tools, args.interface,
                                extras): seed for seed, port in zip(seeds, ports)}
            for fut in as_completed(futs):
                try:
                    res = fut.result()
                except Exception as e:
                    logger.exception(f"seed{futs[fut]} crashed")
                    res = {"seed": futs[fut], "status": "error", "success": None, "error": repr(e)}
                results.append(res)
                print_one(res)
                if not args.no_video and all(other.done() for other in futs):
                    render_replay(res.get("demo_folder", ""), res.get("seed"))
    except KeyboardInterrupt:
        interrupted = True
        logger.info("\n[interrupted] killing all stacks ...")
    except Exception:
        errored = True
        logger.exception("[run] crashed")
    finally:
        run_seed.SHUTDOWN.set()
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        with run_seed.CHILDREN_LOCK:
            children = list(run_seed.CHILDREN)
        for proc in children:
            terminate_group(proc)
        for port in all_ports:
            kill_ports(stack_ports(port))
        results.sort(key=lambda r: r.get("seed", 0))
        summarize_and_write(results, exp, name, out_dir, interrupted, args.interface)

    mp = os.path.join(out_dir, "meta.json")
    meta = get_json_file(mp) or {}
    if errored:
        meta["status"] = "error"
    meta["isolated"] = exp.isolated
    json.dump(meta, open(mp, "w"), indent=1)
    if not args.no_video and not interrupted and not errored:
        render_replays(out_dir)
    for seed in seeds:
        seed_dir = os.path.join(out_dir, f"seed{seed}")
        if os.path.isfile(os.path.join(seed_dir, "verdict.json")):
            publish_episode(seed_dir, out_dir)


if __name__ == "__main__":
    main()
