"""One seed end to end: simulator, sandboxed agent, grade.

On its own port stack: start the simulator server and the sandboxed agent, resume the agent until the episode
ends or the budget runs out, then grade the episode. run_eval.py calls run_seed() once per seed and publishes
the graded folder.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime

from core import antigravity
from core.agent_log import (agent_metrics, api_stalled, auth_failed, codex_metrics, codex_session_file, codex_session_id, condense,
                               condense_codex, mcp_toolless, mcp_ui_unattached, plan_limited, session_started)
from core.accounts import PLAN_FULL, apply_auth, choose_auth, codex_plan_with_room, repoint_codex_login, seed_auth
from core.agent_sandbox import (AUTO_COMPACT_WINDOW, CODEX_AUTO_COMPACT_WINDOW, DISALLOWED_TOOLS, PRECOMPACT_PROMPT,
                                   REPO_ROOT, build_agent_cmd, build_codex_cmd, codex_agents_md, codex_seed_home, model_runtime,
                                   require_codex_home)
from core.grade_episode import graded
from core.render import RENDER_FAILED_EXIT
from core.tasks import TASKS, resolve_task
from core.ui_for_agent import cdp_ready, end_episode
from core.utils import get_json, get_json_file, interface_manifest

logger = logging.getLogger(__name__)

EVAL_SEED_BASE = 1_000_000


@dataclass(frozen=True)
class Experiment:
    task: str
    language: str
    n_seeds: int = 2
    seed_start: int = EVAL_SEED_BASE
    data_root: str = "data/run_eval/dev"
    base_port: int = 8100
    model: str = "opus"
    effort: str = "xhigh"
    render: str = "blender"
    seeds_per_blender: int = 1
    episode_timeout: float = 21600.0
    ready_timeout: float = 600.0
    grace: float = 300.0
    plan: str | None = None
    plan_dir: str | None = None
    plan_auto: bool = False
    isolated: bool = True


SET_ENV = os.path.join(REPO_ROOT, "set_env.sh")

NUM_POINT = 400_000

STREAM_FREQ = 1

SETTLE = float(os.environ.get("ROBOARCADE_SETTLE", "30"))


def stack_ports(base_port: int) -> tuple[int, ...]:
    return tuple(base_port + i for i in range(4))


def _proc_ppid(pid: int) -> int | None:
    try:
        for line in pathlib.Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("PPid:"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


def _pid_belongs_to_run(pid: int) -> bool:
    launcher = os.getpid()
    current, seen = pid, set()
    while current > 1 and current not in seen:
        if current == launcher:
            return True
        seen.add(current)
        parent = _proc_ppid(current)
        if parent is None:
            return False
        current = parent
    return current == launcher


def kill_ports(ports) -> None:
    try:
        out = subprocess.run(["lsof", f"-tiTCP:{ports[0]}-{ports[-1]}", "-sTCP:LISTEN"],
                             capture_output=True, text=True).stdout
    except FileNotFoundError:
        return
    pids = {int(value) for value in out.split()}
    if not all(_pid_belongs_to_run(pid) for pid in pids):
        return
    for pid in sorted(pids):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def wait_for_sim_ready(proc: subprocess.Popen, base_port: int, ready_timeout_s: float, settle_s: float,
                       demo_folder: str | None = None, ui: bool = True):
    env_url = f"http://localhost:{base_port + 1}/env.json"
    deadline = time.time() + ready_timeout_s
    foreign = None
    while time.time() < deadline:
        if proc.poll() is not None:
            return None
        info = get_json(env_url)
        if info and info.get("demo_folder") and demo_folder and (
                os.path.realpath(info["demo_folder"]) != os.path.realpath(demo_folder)):
            if info["demo_folder"] != foreign:
                foreign = info["demo_folder"]
                logger.info(f":{base_port} is served by a stale sim for {foreign}; waiting for ours")
        elif (info and info.get("demo_folder") and info.get("frame_ready")
                and (not ui or cdp_ready(base_port))):
            time.sleep(settle_s)
            return info
        time.sleep(0.5)
    return None


def terminate_group(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    for sig, wait_s in ((signal.SIGTERM, 10.0), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=wait_s)
            return
        except subprocess.TimeoutExpired:
            continue


def launch_sim(task: str, demo_folder: str, seed: int, env: dict, log_f,
               record_state: bool = False, ui: bool = True, rgbd: bool = False) -> subprocess.Popen:
    cmd = [
        "bash", "-c", f'source "{SET_ENV}" && exec "$@"', "_",
        "python", "-m", "core.sim_env_server",
        "--task", task, "--demo_folder", demo_folder, "--seed", str(seed),
        "--num_point", str(NUM_POINT), "--stream_freq", str(STREAM_FREQ),
        *(["--record-state"] if record_state else []),
        *([] if ui else ["--no-ui"]),
        *(["--rgbd"] if rgbd else []),
    ]
    child_env = dict(env)
    child_env.setdefault("ROBOARCADE_CHROMIUM_LOG", log_f.name.removesuffix("_sim.log") + "_chromium.log")
    return subprocess.Popen(cmd, cwd=REPO_ROOT, env=child_env, stdout=log_f,
                            stderr=subprocess.STDOUT, start_new_session=True)


MCP_CONFIG_PATH = os.path.join(REPO_ROOT, ".mcp.json")


def allowed_tools(config_path: str) -> list[str]:
    cfg = get_json_file(config_path) or {}
    names = list((cfg.get("mcpServers") or {}).keys()) or ["roboarcade"]
    return [f"mcp__{name}" for name in names]


def interface_guide(interface: str) -> str | None:
    guide = os.path.join(interface, "guide.md")
    return guide if os.path.isfile(guide) else None


MAX_API_STALL = 1800.0

SIM_STALL_S = 180


def _cpu_seconds(pid: int) -> float:
    try:
        with open(f"/proc/{pid}/stat") as f:
            parts = f.read().rsplit(") ", 1)[1].split()
        return (int(parts[11]) + int(parts[12])) / os.sysconf("SC_CLK_TCK")
    except (OSError, IndexError, ValueError):
        return 0.0


def sim_faulted(sim, demo_folder: str, since: float) -> bool:
    if sim.poll() is None or SHUTDOWN.is_set():
        return False
    v = os.path.join(demo_folder, "verdict.json")
    return not (os.path.exists(v) and os.path.getmtime(v) >= since)


def build_prompt(language: str) -> str:
    task_line = f"Task: {language}\n\n" if language else ""
    return (
        "Control the simulated robot arm to complete the following task.\n\n"
        f"{task_line}"
        "The interface guide in your system prompt describes the UI and the tools.\n"
        "Nothing is graded until you say so. When you believe the task is complete, call "
        "end_episode. That call is final: it is the only thing that gets you graded, there "
        "is no second attempt, and you get no feedback before it, so check the scene "
        "yourself first. Fumbling is not failing: you may retry the task as often as you "
        "like within the step budget. What you cannot retry is the declaration.\n"
        "Before each tool call, say in one or two sentences what you are about to do and "
        "why: what you believe the scene looks like, what the last result told you, and "
        "what this action is meant to achieve. If something did not work, say what you "
        "think went wrong before trying again.\n"
        "Act fully autonomously; do not ask questions."
    )


CHILDREN: set[subprocess.Popen] = set()

CHILDREN_LOCK = threading.Lock()

SHUTDOWN = threading.Event()


def _tree_cpu(sim) -> float:
    kids = subprocess.run(["pgrep", "-P", str(sim.pid)], capture_output=True,
                          text=True).stdout.split()
    return sum(_cpu_seconds(int(p)) for p in [sim.pid, *kids])


def _run_agent_process(cmd, log_path, timeout_s, env, tag, sim, demo_folder, since, stall,
                       stdin=None) -> str:
    t0 = time.time()
    poll_s = 30.0
    with open(log_path, "a") as log_f, open(log_path + ".err", "a") as err_f:
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=log_f, stderr=err_f,
                                stdin=stdin, start_new_session=True)
        with CHILDREN_LOCK:
            CHILDREN.add(proc)
        try:
            while True:
                try:
                    proc.wait(timeout=poll_s)
                    return "agent_error" if env.get("ROBOARCADE_AGY_STATE") and proc.returncode else "ok"
                except subprocess.TimeoutExpired:
                    pass
                if env.get("ROBOARCADE_AGY_STATE") and antigravity.startup_error(log_path):
                    terminate_group(proc)
                    return "agent_error"
                if sim.poll() is not None:
                    if sim_faulted(sim, demo_folder, since):
                        logger.warning(f"[{tag}] the sim exited ({sim.returncode}) with no verdict; "
                                       "the agent is driving a dead UI; ending it now.")
                        terminate_group(proc)
                        return "ok"
                elif not stall["stalled"]:
                    now = _tree_cpu(sim)
                    stall["quiet"] = (0.0 if now > stall["last"] + 0.02 * poll_s
                                      else stall["quiet"] + poll_s)
                    stall["last"] = now
                    if stall["quiet"] >= SIM_STALL_S:
                        stall["stalled"] = True
                        logger.warning(f"[{tag}] the sim has burned no CPU for {stall['quiet']:.0f}s "
                                       f"while still running: it is stuck, not busy; writing "
                                       f"what we can to {stall['out']}")
                        spy = shutil.which("py-spy")
                        pids = [sim.pid] + [int(x) for x in subprocess.run(
                            ["pgrep", "-P", str(sim.pid)], capture_output=True,
                            text=True).stdout.split()]
                        with open(stall["out"], "w") as f:
                            f.write(f"sim burned no CPU for {stall['quiet']:.0f}s at "
                                    f"{time.strftime('%H:%M:%S')}; pids {pids}\n")
                            for pid in pids:
                                f.write(f"\n===== pid {pid}: {_cpu_seconds(pid):.1f}s cpu =====\n")
                                if spy:
                                    traced = subprocess.run([spy, "dump", "--pid", str(pid)],
                                                            capture_output=True, text=True,
                                                            timeout=60)
                                    f.write(traced.stdout or traced.stderr)
                if time.time() - t0 > timeout_s:
                    terminate_group(proc)
                    return "timeout"
        except KeyboardInterrupt:
            terminate_group(proc)
            raise
        finally:
            with CHILDREN_LOCK:
                CHILDREN.discard(proc)


def run_agent(runtime, prompt, model, log_path, timeout_s, tools, env, session_id, instruct_files,
              effort, resume, tag, sim, demo_folder, since, stall, isolated) -> str:
    iface = env["ROBOARCADE_INTERFACE"]
    base_port = int(env["ROBOARCADE_BASE_PORT"])
    stdin = None
    if runtime == "antigravity":
        if not isolated:
            raise ValueError("Antigravity evaluation requires the agent sandbox")
        if "ROBOARCADE_AGY_STATE" not in env:
            raise ValueError("Antigravity requires local per-seed session state")
        cmd = antigravity.command(prompt=prompt, model=model, session_id=session_id,
                                  base_port=base_port, pkg=iface, resume=resume)
        stdin = subprocess.DEVNULL
    elif runtime == "codex":
        cwd = codex_agents_md(demo_folder, instruct_files)
        env = {**env, "ROBOARCADE_AGENT_CWD": cwd}
        env.pop("ROBOARCADE_BASE_PORT")
        cmd = build_codex_cmd(prompt=prompt, model=model, session_id=session_id,
                              mcp_config=MCP_CONFIG_PATH, base_port=base_port,
                              isolated=isolated, cwd=cwd, effort=effort, resume=resume,
                              pkg=iface, runtime_env=env)
        stdin = subprocess.DEVNULL
    else:
        cmd = build_agent_cmd(prompt=prompt, model=model, session_id=session_id,
                              mcp_config=MCP_CONFIG_PATH, allowed=tools,
                              disallowed=DISALLOWED_TOOLS,
                              base_port=base_port, isolated=isolated,
                              instruct_files=instruct_files,
                              pkg=iface if os.path.isdir(iface) else None,
                              effort=effort, resume=resume)
    return _run_agent_process(cmd, log_path, timeout_s, env, tag, sim, demo_folder, since, stall, stdin)


def finished_result(seed: int, base_port: int, demo_folder: str,
                    task: str | None = None,
                    interface: str | None = None) -> dict | None:
    verdict = get_json_file(os.path.join(demo_folder, "verdict.json"))
    if verdict is None:
        return None
    if task is not None and verdict.get("task") and verdict["task"] != task:
        logger.warning(f"seed{seed}: verdict at {demo_folder} is for task "
                       f"{verdict['task']!r}, not {task!r}; ignoring it")
        return None
    ep = get_json_file(os.path.join(demo_folder, "episode.json")) or {}
    if (interface is not None and ep.get("interface")
            and os.path.basename(os.path.normpath(ep["interface"]))
            != os.path.basename(os.path.normpath(interface))):
        logger.warning(f"seed{seed}: episode at {demo_folder} used interface "
                       f"{ep['interface']!r}, not {interface!r}; ignoring it")
        return None
    timed_out = bool(verdict.get("timed_out"))
    score = solved = None
    if verdict.get("wall_clock_timeout"):
        score, solved = 0.0, False
    else:
        try:
            grade = graded(demo_folder)
            score, solved = grade["score"], grade.get("solved", grade["score"] == 1)
        except (OSError, ValueError, KeyError) as exc:
            logger.error(f"seed{seed}: grading {demo_folder} failed: {type(exc).__name__}: {exc}")
    return {
        "seed": seed,
        "base_port": base_port,
        "demo_folder": demo_folder,
        "status": "ok" if solved is not None else "grading_failed",
        "success": solved,
        "score": score,
        "timed_out": timed_out,
        "sim_steps_used": verdict.get("sim_steps_used"),
        "elapsed_seconds": verdict.get("elapsed_seconds"),
    }


def write_timeout_verdict(demo_folder: str, exp: Experiment, seed: int) -> None:
    os.makedirs(demo_folder, exist_ok=True)
    path = os.path.join(demo_folder, "verdict.json")
    verdict = get_json_file(path) or {"task": exp.task, "seed": seed,
                                      "elapsed_seconds": round(exp.episode_timeout, 1),
                                      "sim_steps_used": None}
    verdict.update(timed_out=True, wall_clock_timeout=True)
    with open(path, "w") as f:
        json.dump(verdict, f, indent=2)


RESUME_PROMPT = (
    "The task is NOT solved yet and simulation budget remains. Continue "
    "working on it now. Re-check the scene with a screenshot first: the "
    "arm may have been reset. Do not stop before the interface reports "
    "EPISODE OVER or you run out of waypoint budget."
)


def episode_over(base_port: int) -> bool:
    return bool((get_json(f"http://localhost:{base_port + 1}/env.json") or {})
                .get("episode_over"))


def drive_antigravity(exp, tag, prompt, log, env, instruct_files, tools, sim, folder, since, result, stall):
    deadline = time.time() + exp.episode_timeout
    session = None
    resumes = 0
    provider_retries = 0
    # agy's state-file locking stalls on NFS. Keep live state on local disk and
    # archive it after the process exits; all resumes share this same directory.
    with tempfile.TemporaryDirectory(prefix="roboarcade-agy-", dir="/tmp") as work:
        state = antigravity.prepare(work, exp.model, MCP_CONFIG_PATH,
                                    int(env["ROBOARCADE_BASE_PORT"]), env, instruct_files)
        env = {**env, "ROBOARCADE_AGY_STATE": state}
        try:
            while not SHUTDOWN.is_set():
                remaining = deadline - time.time()
                if remaining <= 0:
                    return "timeout", None
                status = run_agent("antigravity", prompt if not session else RESUME_PROMPT, exp.model,
                                   log, remaining, tools, env, session, instruct_files, exp.effort,
                                   bool(session), tag, sim, folder, since, stall, exp.isolated)
                session, error = antigravity.outcome(log)
                result.update(session_id=session, resumes=resumes)
                if status == "timeout":
                    return status, None
                if error or status != "ok" or not session:
                    if (session and antigravity.retryable_error(error) and provider_retries < 3
                            and sim is not None and sim.poll() is None
                            and not episode_over(int(env["ROBOARCADE_BASE_PORT"]))):
                        remaining = deadline - time.time()
                        delay = min(15 * 2 ** provider_retries, max(0, remaining))
                        if remaining > delay:
                            provider_retries += 1
                            result.setdefault("provider_retries", []).append({
                                "error": error, "delay_seconds": delay,
                                "session_id": session})
                            logger.warning(f"[{tag}] transient Antigravity error; resuming the same "
                                           f"conversation after {delay}s (retry {provider_retries}/3)")
                            SHUTDOWN.wait(delay)
                            resumes += 1
                            continue
                    result["agent_error"] = error or "Antigravity exited without a successful session"
                    return "agent_error", result["agent_error"]
                if sim.poll() is not None or episode_over(int(env["ROBOARCADE_BASE_PORT"])):
                    return status, None
                if mcp_ui_unattached(os.path.join(folder, "screenshots")):
                    result["mcp_failed"] = True
                    return "agent_error", "MCP did not attach to the simulator"
                resumes += 1
            result["agent_error"] = "Antigravity run interrupted"
            return "shutdown", result["agent_error"]
        finally:
            shutil.copytree(state, os.path.join(folder, "antigravity"), dirs_exist_ok=True)


def drive_agent(exp, tag, prompt, claude_log, env, instruct_files, tools, sim, demo_folder,
                episode_t0, result, stall) -> tuple[str, str | None]:
    if model_runtime(exp.model) == "antigravity":
        return drive_antigravity(exp, tag, prompt, claude_log, env, instruct_files, tools,
                                 sim, demo_folder, episode_t0, result, stall)
    t0 = time.time()
    base_port = int(env["ROBOARCADE_BASE_PORT"])
    runtime = model_runtime(exp.model)
    session = why = None
    resumes = api_stalls = 0
    futile_attempts = 0
    stalled_s = 0.0
    held_s = 0.0

    def left():
        return exp.episode_timeout - (time.time() - t0)

    text, resume, timeout_s = prompt, False, left()
    if timeout_s <= 0:
        return "timeout", None
    while True:
        if not resume:
            session = None if runtime == "codex" else str(uuid.uuid4())
            result["session_id"] = session
        r0 = time.time()
        status = run_agent(runtime, text, exp.model, claude_log, timeout_s, tools, env, session,
                           instruct_files, exp.effort, resume, tag, sim, demo_folder,
                           episode_t0, stall, exp.isolated)
        if runtime == "codex":
            session = codex_session_id(claude_log)
            result["session_id"] = session

        why = auth_failed(claude_log, session)
        if why:
            logger.warning(f"[{tag}] agent never authenticated ({why[:80]})")
            result["auth_error"] = why
            break
        if runtime == "codex" and exp.plan and plan_limited(runtime, claude_log, session):
            h0 = time.time()
            logger.warning(f"[{tag}] the Codex plan's window is full; pausing until one has room")
            pick = wait_for_codex_room(exp, tag)
            if pick is None:
                break
            repoint_codex_login(env["CODEX_HOME"], pick[1])
            result["auth"] = {"auth": "plan", "reason": f"ChatGPT auth at {pick[1]}/auth.json"}
            held_s += time.time() - h0
            futile_attempts = 0
        if (runtime == "claude" and exp.plan and "ANTHROPIC_API_KEY" not in env
                and plan_limited(runtime, claude_log, session, exp.plan_dir)):
            key = choose_auth()
            if key["auth"] == "api_key":
                apply_auth(env, key)
                result["auth"] = {"auth": "api_key",
                                  "reason": f"plan {exp.plan} window filled mid-episode"}
                logger.warning(f"[{tag}] plan window full; the rest of this seed bills the API key")
        if mcp_ui_unattached(os.path.join(demo_folder, "screenshots")):
            logger.warning(f"[{tag}] RoboARCade tools never attached to the UI; "
                           "leaving seed ungraded for rerun")
            result["mcp_failed"] = True
            break
        toolless = mcp_toolless(claude_log)
        if toolless and not result.get("mcp_retry") and not sim_faulted(sim, demo_folder, episode_t0):
            logger.warning(f"[{tag}] agent had no roboarcade MCP tools; retrying on a fresh session")
            result["mcp_retry"] = True
            text, resume, timeout_s = prompt, False, left()
            continue
        if toolless:
            logger.warning(f"[{tag}] still no roboarcade MCP tools; leaving seed ungraded for rerun")
            result["mcp_failed"] = True
            break

        if api_stalled(claude_log) and stalled_s < MAX_API_STALL:
            wait_s = min(60.0 * 2 ** min(api_stalls, 4), 480.0)
            api_stalls += 1
            logger.info(f"[{tag}] server-side API fault (stall #{api_stalls}); holding "
                        f"{wait_s:.0f}s, the episode clock is paused meanwhile")
            SHUTDOWN.wait(wait_s)
            stalled_s = min(stalled_s + (time.time() - r0), MAX_API_STALL)
        else:
            api_stalls = 0
            futile_attempts = futile_attempts + 1 if (time.time() - r0 < 20 and status != "timeout") else 0

        for _ in range(6):
            if sim.poll() is not None or episode_over(base_port):
                break
            time.sleep(5)
        remaining_s = left() + stalled_s + held_s
        if (sim.poll() is not None or episode_over(base_port) or SHUTDOWN.is_set()
                or remaining_s <= 60):
            break
        if futile_attempts >= 3:
            logger.warning(f"[{tag}] 3 consecutive attempts died instantly; giving up on the agent")
            break
        resumes += 1
        if session_started(claude_log, session):
            logger.info(f"[{tag}] agent stopped with the sim still live; resuming "
                        f"(#{resumes}, {remaining_s:.0f}s of the episode budget left)")
            text, resume, timeout_s = RESUME_PROMPT, True, remaining_s
        else:
            logger.warning(f"[{tag}] session {(session or 'unknown')[:8]} never materialized; "
                           f"relaunching fresh (#{resumes}, {remaining_s:.0f}s left)")
            text, resume, timeout_s = prompt, False, remaining_s
    result["resumes"] = resumes
    result["api_stalled_seconds"] = round(stalled_s, 1)
    return status, why


GT_GLOBS = [
    "tasks/*/grade.py",
    "core/grade_episode.py",
    "core/sim_env.py",
    "core/initial_layout.py",
    "core/tasks.py",
    "tasks/*/dynamics.py",
    "core/sim_env_server.py",
    "interfaces/*/tools.py",
    "interfaces/*/guide.md",
    "core/UI/index.html",
]


def gt_manifest(extra=()) -> dict[str, str]:
    repo = pathlib.Path(__file__).resolve().parent.parent
    files = [p for g in GT_GLOBS for p in sorted(repo.glob(g))]
    files += sorted(TASKS.glob("*/task.json"))
    files += [p for d in extra for p in sorted(pathlib.Path(d).rglob("*"))
              if p.is_file() and "__pycache__" not in p.parts]
    return {os.path.relpath(p, repo): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def diff(before: dict[str, str], after: dict[str, str]) -> list[str]:
    return sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))


def integrity_capture(task, interface):
    files = gt_manifest(extra=[interface])
    pack = resolve_task(task)
    for name in ("scene.xml", "manifest.json"):
        path = pack / name
        files[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files


def _write_receipt(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("x") as stream:
            os.chmod(temp, 0o600)
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def integrity_begin(folder, task, seed, interface, isolated):
    record = dict(schema=1, attempt_id=uuid.uuid4().hex, task=task, seed=seed,
                  interface=str(interface), isolated=bool(isolated),
                  started_epoch=time.time(), status="running",
                  before=integrity_capture(task, interface))
    _write_receipt(pathlib.Path(folder) / "integrity.json", record)
    return record


def integrity_finish(folder, start):
    record = dict(start, checked_epoch=time.time())
    try:
        record["after"] = integrity_capture(start["task"], start["interface"])
        record["changed"] = diff(start["before"], record["after"])
        record["intact"] = not record["changed"]
        record["status"] = "verified" if record["intact"] else "changed"
    except (OSError, ValueError, KeyError) as exc:
        record.update(status="error", intact=None,
                      error=f"{type(exc).__name__}: {exc}")
    _write_receipt(pathlib.Path(folder) / "integrity.json", record)
    return {k: record[k] for k in ("attempt_id", "status", "intact", "checked_epoch")}


def run_seed(
    exp: Experiment, seed: int, base_port: int, logs_dir: str, tools: list[str], interface: str,
    extra=(), demo_folder: str | None = None, render_group=None,
) -> dict:
    tag = f"seed{seed}"
    manifest = interface_manifest(interface)
    demo_folder = demo_folder or os.path.join(os.path.dirname(logs_dir), f"seed{seed}")
    result: dict = {"seed": seed, "base_port": base_port, "demo_folder": demo_folder}

    if SHUTDOWN.is_set():
        return {**result, "status": "skipped", "success": None}

    existing = finished_result(seed, base_port, demo_folder, exp.task,
                               interface=interface)
    if existing is not None:
        existing["reused"] = True
        logger.info(f"[{tag}] already finished (solved={existing['success']}); skipping rerun")
        return existing

    sim_log = os.path.join(logs_dir, f"{tag}_port{base_port}_sim.log")
    claude_log = os.path.join(logs_dir, f"{tag}_port{base_port}_claude.log")
    runtime = model_runtime(exp.model)

    env = dict(os.environ)
    for key in ("ROBOARCADE_RENDER_GROUP_SOCKET", "ROBOARCADE_RENDER_GROUP_SEED",
                "ROBOARCADE_RENDER_GROUP_TOKEN", "ROBOARCADE_RENDER_OWNER_PID"):
        env.pop(key, None)
    env["ROBOARCADE_BASE_PORT"] = str(base_port)
    env["ROBOARCADE_INTERFACE"] = interface
    env["ROBOARCADE_RENDER"] = exp.render
    if extra:
        env["ROBOARCADE_INSTRUCT_FILES"] = ":".join(extra)
    env.setdefault("MCP_TIMEOUT", "300000")
    if runtime == "antigravity":
        if not exp.isolated:
            raise ValueError("Antigravity evaluation requires sandboxing")
        env["GEMINI_API_KEY"] = antigravity.api_key()
        auth = {"auth": "api_key", "reason": "Gemini API via Antigravity " + antigravity.backend()}
    elif runtime == "codex":
        env.pop("OPENAI_API_KEY", None)
        plan_dir = exp.plan_dir
        if exp.plan:
            pick = wait_for_codex_room(exp, tag)
            if pick is None:
                result["status"] = "shutdown"
                return result
            plan_dir = pick[1]
        env["CODEX_HOME"] = codex_seed_home(tag, plan_dir)
        auth = {"auth": "plan",
                "reason": f"ChatGPT auth at {require_codex_home(plan_dir)}/auth.json"}
    else:
        auth = seed_auth(exp)
        apply_auth(env, auth)
    logger.info(f"[{tag}] auth: {auth['auth']} ({auth['reason']})")
    result["auth"] = {k: v for k, v in auth.items() if k != "key"}

    integrity_start = integrity_begin(demo_folder, exp.task, seed, interface, exp.isolated)
    with open(sim_log, "w") as slog:
        sim = None
        try:
            with render_group.startup_lock if render_group else nullcontext():
                if SHUTDOWN.is_set():
                    return {**result, "status": "skipped", "success": None}
                sim_env = dict(env)
                sim_env.pop("GEMINI_API_KEY", None)
                if render_group:
                    sim_env.update(render_group.environment(seed))
                    result["render_group"] = {"id": render_group.id, "seeds": render_group.seeds}
                sim = launch_sim(exp.task, demo_folder, seed, sim_env, slog, record_state=True,
                                 ui=manifest.get("ui", True), rgbd=manifest.get("rgbd", False))
                with CHILDREN_LOCK:
                    CHILDREN.add(sim)
                info = wait_for_sim_ready(sim, base_port, exp.ready_timeout,
                                          0 if render_group or not manifest.get("ui", True) else SETTLE,
                                          demo_folder, ui=manifest.get("ui", True))
            if render_group and info is not None:
                SHUTDOWN.wait(SETTLE)
            if info is None:
                terminate_group(sim)
                result["status"] = ("render_failed" if sim.returncode == RENDER_FAILED_EXIT
                                    else "sim_failed_to_start")
                result["success"] = None
                return result

            live_language = str(info.get("language") or exp.language)
            env["ROBOARCADE_TASK_LANGUAGE"] = live_language
            episode_t0 = time.time()
            stall = {"last": 0.0, "quiet": 0.0, "stalled": False,
                     "out": os.path.join(logs_dir, f"{tag}_port{base_port}_stall.txt")}
            if runtime == "claude":
                env["ROBOARCADE_PROJECTS_DIR"] = os.path.join(demo_folder, "claude_projects")
            env["ROBOARCADE_SHOTS_DIR"] = os.path.join(demo_folder, "screenshots")
            logger.info(f"[{tag}] sim ready on :{base_port}  task={info.get('task')!r}  demo_folder={demo_folder}")

            if SHUTDOWN.is_set():
                result["status"] = "skipped"
                result["success"] = None
                return result

            guide = interface_guide(interface)
            instruct_files = [*([guide] if guide else []), *extra]
            prompt = build_prompt(live_language)
            logger.info(f"[{tag}] launching {runtime} ({exp.model}) ...")
            t0 = time.time()
            claude_status, why = drive_agent(exp, tag, prompt, claude_log, env, instruct_files,
                                             tools, sim, demo_folder, episode_t0, result, stall)
            result["sim_stalled"] = stall["stalled"]
            if sim_faulted(sim, demo_folder, episode_t0):
                result["sim_crashed"] = True
                result["sim_returncode"] = sim.returncode
                result["render_failed"] = sim.returncode == RENDER_FAILED_EXIT
                with open(sim_log, errors="ignore") as f:
                    result["sim_log_tail"] = "".join(f.readlines()[-25:])

            result["claude_status"] = claude_status
            result["metrics"] = (antigravity.metrics(claude_log, time.time() - t0,
                                                    os.path.join(demo_folder, "antigravity", "state")) if runtime == "antigravity" else
                                 agent_metrics(claude_log) if runtime == "claude" else
                                 codex_metrics(claude_log, env["CODEX_HOME"],
                                               result.get("session_id"), exp.model,
                                               time.time() - t0))
            if runtime == "codex":
                session_file = codex_session_file(env["CODEX_HOME"], result.get("session_id"))
                if session_file:
                    shutil.copy(session_file, os.path.join(demo_folder, "codex_session.jsonl"))
            cost = result["metrics"].get("cost_usd")
            logger.info(f"[{tag}] {runtime} finished ({claude_status}, {time.time() - t0:.0f}s"
                        f"{', $%.2f' % cost if cost else ''}); waiting for sim_env_server to save ...")

            if result.get("sim_crashed"):
                logger.warning(f"[{tag}] the sim died mid-episode; leaving seed ungraded for rerun")
            elif (why or result.get("mcp_failed")) and sim.poll() is None:
                logger.warning(f"[{tag}] infrastructure failure; leaving seed ungraded for rerun")
            else:
                if sim.poll() is None and not episode_over(base_port):
                    if not manifest.get("ui", True):
                        os.kill(sim.pid, signal.SIGUSR1)
                    else:
                        try:
                            end_episode(base_port)
                        except Exception as e:
                            logger.info(f"[{tag}] could not click End Episode ({e}); signalling sim_env_server")
                            os.kill(sim.pid, signal.SIGUSR1)
                try:
                    sim.wait(timeout=exp.grace)
                except subprocess.TimeoutExpired:
                    logger.info(f"[{tag}] sim_env_server still running after grace; terminating (no verdict).")
        finally:
            try:
                if sim is not None:
                    terminate_group(sim)
                    with CHILDREN_LOCK:
                        CHILDREN.discard(sim)
                if runtime == "codex":
                    shutil.rmtree(env["CODEX_HOME"], ignore_errors=True)
            finally:
                result["integrity"] = integrity_finish(demo_folder, integrity_start)

    if runtime == "antigravity":
        shutil.copy(claude_log, os.path.join(demo_folder, "antigravity.jsonl"))
    session = os.path.join(demo_folder, "codex_session.jsonl")
    with open(os.path.join(demo_folder, "conversation.txt"), "w") as f:
        f.write(antigravity.condense(claude_log) if runtime == "antigravity" else
                condense_codex(session) if runtime == "codex" and os.path.isfile(session)
                else condense(claude_log))
    with open(os.path.join(demo_folder, "episode.json"), "w") as f:
        json.dump({"seed": seed, "task": exp.task, "language": exp.language,
                   "model": exp.model, "effort": None if runtime == "antigravity" else exp.effort, "interface": interface,
                   "reasoning_policy": "model_default" if runtime == "antigravity" else exp.effort,
                   "render": "failed" if result.get("render_failed") else exp.render,
                   "seeds_per_blender": exp.seeds_per_blender,
                   "render_group": result.get("render_group"),
                   "integrity": result.get("integrity"),
                   "base_port": base_port, "session_id": result.get("session_id"),
                   "claude_status": result.get("claude_status"),
                   "auth": result.get("auth"), "metrics": result.get("metrics"),
                   "prompt": prompt, "instruct_files": instruct_files,
                   "isolated": exp.isolated,
                   "agent_runtime": runtime,
                   **(antigravity.runtime_metadata() if runtime == "antigravity" else {}),
                   "auto_compact_window": AUTO_COMPACT_WINDOW if runtime == "claude" else None,
                   "codex_auto_compact_window": (CODEX_AUTO_COMPACT_WINDOW
                                                  if runtime == "codex" else None),
                   "compact_prompt_source": (None if runtime == "antigravity"
                                             else PRECOMPACT_PROMPT),
                   "resumes": result.get("resumes"),
                   "api_stalled_seconds": result.get("api_stalled_seconds"),
                   "recorded_at": datetime.now().isoformat(timespec="seconds")},
                  f, indent=2)

    if result.get("agent_error"):
        result.update(status="agent_error", success=None)
        return result
    if result.get("claude_status") == "timeout":
        write_timeout_verdict(demo_folder, exp, seed)
    finished = finished_result(seed, base_port, demo_folder, exp.task)
    if finished is not None:
        result.update(finished)
        return result
    for key in ("auth_error", "render_failed", "sim_crashed", "mcp_failed"):
        if result.get(key):
            result["status"] = {"auth_error": "auth_failed"}.get(key, key)
            result["success"] = None
            return result
    if result.get("claude_status") == "timeout":
        result["status"] = "timeout"
        result["success"] = False
        result["timed_out"] = True
    else:
        result["status"] = "no_verdict"
        result["success"] = None
    return result


def wait_for_codex_room(exp, tag) -> tuple[str, str] | None:
    while not SHUTDOWN.is_set():
        pick = codex_plan_with_room(exp)
        if pick:
            return pick
        logger.warning(f"[{tag}] every Codex plan is at {PLAN_FULL}% or more; holding the "
                       "seed 10 min before looking again")
        SHUTDOWN.wait(600)
    return None
