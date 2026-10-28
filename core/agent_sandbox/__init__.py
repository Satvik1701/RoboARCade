"""The agent sandbox: the agent command lines that agent_sandbox.sh runs inside it.

agent_sandbox.sh (this folder) builds the bubblewrap box and runs the agent in it, with config_mounts.sh,
connect_sandbox_to_sim_and_network.py (its only way out) and probe.sh (its self-test). This file builds the `claude -p` and
`codex exec` command lines (MCP, compaction and model settings, the Codex per-seed home) and lists the roboarcade files
the box exposes (only what the MCP server imports). `--no-isolated` runs the same command line without the box.
"""
import json
import os
import pathlib
import shlex
import tempfile

from core.utils import get_json_file

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ISOLATED_SCRIPT = os.path.join(REPO_ROOT, "core", "agent_sandbox", "agent_sandbox.sh")

CODEX_HOME = os.path.join(REPO_ROOT, "data", "sc", "codex")

DISALLOWED_TOOLS = [
    "Bash", "BashOutput", "KillShell", "KillBash",
    "Read", "Edit", "Write", "NotebookEdit",
    "Glob", "Grep", "Task", "WebFetch", "WebSearch",
    "CronCreate", "CronDelete", "CronList", "DesignSync", "EnterWorktree", "ExitWorktree",
    "ListAgents", "Monitor", "PushNotification", "ReportFindings", "ScheduleWakeup",
    "SendMessage", "Skill", "Workflow",
]


COMPACT_AT = int(os.environ.get("ROBOARCADE_AUTO_COMPACT_WINDOW", "245000"))

AUTO_COMPACT_WINDOW = COMPACT_AT + 33_000 if COMPACT_AT else 0

CODEX_AUTO_COMPACT_WINDOW = COMPACT_AT or None

PRECOMPACT_PROMPT = os.path.join(REPO_ROOT, "core", "compaction", "prompt.txt")

MODEL_IDS = {"opus": "claude-opus-5", "sonnet": "claude-sonnet-5", "haiku": "claude-haiku-4-5-20251001",
             "fable": "claude-fable-5-1"}


def build_agent_cmd(*, prompt, model, session_id, mcp_config, allowed, disallowed,
                    base_port, isolated, instruct_files=(), pkg=None, effort=None, resume=False):
    cmd = ["claude", "-p", prompt,
           *(["--resume", session_id] if resume else ["--session-id", session_id]),
           "--output-format", "stream-json", "--verbose",
           "--mcp-config", mcp_config, "--strict-mcp-config", "--model", model]
    if effort:
        cmd += ["--effort", effort]
    if isolated:
        cmd += ["--setting-sources", ""]
    if AUTO_COMPACT_WINDOW:
        settings = {"autoCompactWindow": AUTO_COMPACT_WINDOW}
        settings["hooks"] = {"PreCompact": [{"matcher": "", "hooks": [
            {"type": "command", "command": f"cat {shlex.quote(PRECOMPACT_PROMPT)}"}]}]}
        cmd += ["--settings", json.dumps(settings)]
    for path in instruct_files:
        cmd += ["--append-system-prompt-file", path]
    cmd += ["--disallowedTools", *disallowed, "--allowedTools", *allowed]
    if isolated:
        iso = [ISOLATED_SCRIPT, str(base_port)]
        if pkg:
            iso += ["--pkg", pkg]
        cmd = [*iso, "--", *cmd]
    return cmd


def model_runtime(model: str) -> str:
    if model.lower().startswith("gemini-"):
        return "antigravity"
    return "codex" if model.lower().startswith("gpt-") else "claude"


def _toml_str(value: str) -> str:
    return json.dumps(value)


PROXY_ENV = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "NO_PROXY", "no_proxy")


def codex_mcp_overrides(mcp_config: str, base_port: int, runtime_env=None) -> list[str]:
    flags = []
    for name, spec in ((get_json_file(mcp_config) or {}).get("mcpServers") or {}).items():
        command = spec.get("command", "")
        if not os.path.isabs(command):
            command = os.path.join(REPO_ROOT, command)
        args = []
        for arg in spec.get("args") or []:
            if not os.path.isabs(arg) and ("/" in arg or arg.endswith(".py")):
                arg = os.path.join(REPO_ROOT, arg)
            args.append(arg)
        inherited = {key: value for key, value in (runtime_env or {}).items()
                     if key in {"ROBOARCADE_INTERFACE", "ROBOARCADE_SHOTS_DIR",
                                "ROBOARCADE_TASK_LANGUAGE", "CODEX_HOME"}}
        server_env = {**inherited, **(spec.get("env") or {}),
                      "ROBOARCADE_BASE_PORT": str(base_port)}
        args_toml = "[" + ",".join(_toml_str(arg) for arg in args) + "]"
        env_toml = "{" + ",".join(
            f"{key}={_toml_str(str(value))}" for key, value in server_env.items()) + "}"
        env_vars = "[" + ",".join(_toml_str(k) for k in PROXY_ENV) + "]"
        flags += ["-c", (f"mcp_servers.{name}={{command={_toml_str(command)},"
                         f"args={args_toml},env={env_toml},env_vars={env_vars},"
                         "startup_timeout_sec=300,tool_timeout_sec=300}")]
    if flags:
        flags += ["-c", "mcp_optional_startup_grace_ms=120000"]
    return flags


def codex_agents_md(demo_folder: str, instruct_files) -> str:
    cwd = pathlib.Path(demo_folder, "codex_cwd")
    cwd.mkdir(parents=True, exist_ok=True)
    parts = [pathlib.Path(path).read_text() for path in instruct_files]
    (cwd / "AGENTS.md").write_text("\n\n".join(parts))
    return str(cwd)


CODEX_FEATURES_OFF = ("view_image", "shell_tool", "multi_agent", "apps", "plugins",
                      "remote_plugin", "browser_use", "computer_use", "image_generation")


def build_codex_cmd(*, prompt, model, session_id, mcp_config, base_port, isolated,
                    cwd, effort, resume=False, pkg=None, runtime_env=None):
    cmd = ["codex", "exec", *(("resume",) if resume else ())]
    cmd += codex_mcp_overrides(mcp_config, base_port, runtime_env)
    cmd += ["-c", f"compact_prompt={_toml_str(pathlib.Path(PRECOMPACT_PROMPT).read_text())}"]
    if CODEX_AUTO_COMPACT_WINDOW:
        cmd += ["-c", f"model_auto_compact_token_limit={CODEX_AUTO_COMPACT_WINDOW}"]
    for key in CODEX_FEATURES_OFF:
        cmd += ["-c", f"features.{key}=false"]
    cmd += ["-c", 'web_search="disabled"',
            "-c", f"model_reasoning_effort={_toml_str(effort)}"]
    cmd += ["-m", model, "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check", "--ignore-user-config"]
    if not resume:
        cmd += ["-C", cwd]
    cmd += ["--json", *([session_id] if resume else []), prompt]
    if isolated:
        iso = [ISOLATED_SCRIPT, str(base_port)]
        if pkg:
            iso += ["--pkg", pkg]
        cmd = [*iso, "--", *cmd]
    return cmd


def require_codex_home(destination: str | None = None) -> str:
    home = pathlib.Path(destination or CODEX_HOME)
    if not (home / "auth.json").is_file():
        raise RuntimeError(f"Codex eval login missing. Run: CODEX_HOME={home} codex login")
    return str(home.resolve())


def codex_seed_home(seed: str, shared_home: str | None = None,
                    tmp_root: str | None = None) -> str:
    shared = pathlib.Path(require_codex_home(shared_home))
    root = tmp_root or os.environ.get("TMPDIR") or tempfile.gettempdir()
    home = pathlib.Path(tempfile.mkdtemp(prefix=f"codex_home_{seed}_", dir=root))
    (home / "auth.json").symlink_to(shared / "auth.json")
    return str(home.resolve())


def codex_text_cmd(model, effort, out_file, cwd) -> list[str]:
    cmd = ["codex", "exec", "-C", cwd]
    for key in CODEX_FEATURES_OFF:
        cmd += ["-c", f"features.{key}=false"]
    cmd += ["-c", 'web_search="disabled"',
            "-c", f"model_reasoning_effort={_toml_str(effort)}",
            "-m", model, "--skip-git-repo-check", "--ignore-user-config",
            "--json", "-o", out_file]
    return cmd

SANDBOX_IMPORT_FILES = (
    "__init__.py", "mcp_server.py", "kernel.py", "utils.py")
