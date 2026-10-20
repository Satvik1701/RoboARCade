"""Antigravity CLI with Gemini API-key billing and a robot-only tool surface."""
import json
import os
from pathlib import Path

from core.agent_sandbox import ISOLATED_SCRIPT, REPO_ROOT


def api_key():
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        path = Path(os.environ.get("ROBOARCADE_GEMINI_KEY_FILE", "secrets/gemini_key.txt"))
        if path.is_file():
            key = path.read_text().strip()
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY or provision secrets/gemini_key.txt")
    return key


def prepare(folder, model, mcp_config, base_port, env, instruct_files):
    """Fresh per-seed CLI state; no personal login, settings, plugins or history."""
    root = Path(folder) / "antigravity"
    config = root / "config"
    state = root / "state"
    config.mkdir(parents=True, exist_ok=True)
    state.mkdir(exist_ok=True)
    settings = {
        "modelProvider": "gemini",
        "customModelsConfig": {"customModels": {model: {
            "modelName": model, "modelFeatures": {"supportsImages": True}}}},
        "permissions": {"allow": ["mcp(roboarcade/*)",
                                  "read_file(/tmp/home/.gemini/antigravity-cli/brain)",
                                  "read_file(/tmp/home/.gemini/antigravity-cli/mcp/roboarcade)"], "ask": [],
                        "deny": [f"{action}(*)" for action in (
                            "write_file", "command", "unsandboxed", "read_url", "execute_url")]
                        + [f"read_file({p})" for p in (REPO_ROOT, "/proc", "/sys", "/dev")]},
    }
    (config / "settings.json").write_text(json.dumps(settings, indent=2))
    mcp = json.loads(Path(mcp_config).read_text())
    for spec in mcp["mcpServers"].values():
        spec["command"] = os.path.join(REPO_ROOT, spec["command"])
        spec["args"] = [os.path.join(REPO_ROOT, a) if not os.path.isabs(a) and
                        ("/" in a or a.endswith(".py")) else a for a in spec.get("args", [])]
        spec["env"] = {**spec.get("env", {}), **{key: env[key] for key in (
            "ROBOARCADE_INTERFACE", "ROBOARCADE_SHOTS_DIR", "ROBOARCADE_TASK_LANGUAGE") if key in env},
            "ROBOARCADE_BASE_PORT": str(base_port), "PYTHONPATH": REPO_ROOT}
    (config / "mcp_config.json").write_text(json.dumps(mcp, indent=2))
    guide = "\n\n".join(Path(p).read_text() for p in instruct_files)
    (config / "robot.md").write_text(
        "---\nname: robot\ndescription: Control the simulated robot using its MCP interface.\n"
        "mainAgent: true\nsubagent: false\ntools: [view_file]\nexcludeDefaultComponents: true\ninheritMcp: true\ninheritCustomizations: false\n---\n"
        "The robot MCP server is named roboarcade. When its image response is offloaded "
        "to a file, use view_file to see that image before deciding your next action. "
        "Before first using a tool, read its schema with view_file at "
        "/tmp/home/.gemini/antigravity-cli/mcp/roboarcade/<tool_name>.json.\n\n" + guide + "\n")
    return str(root.resolve())


def command(*, prompt, model, session_id, base_port, pkg, resume=False):
    # The custom model selects the API model exactly; agy's --effort catalog is not
    # defined for custom API models. Record model-default reasoning in episode.json.
    cmd = ["agy", "--agent", "robot", "--model", model, "--output-format", "stream-json",
           "--disable-slash-commands", "--project", "/tmp/home/workspace"]
    if resume:
        cmd += ["--conversation", session_id]
    return [ISOLATED_SCRIPT, str(base_port), "--pkg", pkg, "--", *cmd, "-p", prompt]


def records(path):
    if not Path(path).exists():
        return
    with open(path) as stream:
        for line in stream:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue  # A writer may still be completing the last line.


def outcome(path):
    session, result = None, None
    for event in records(path):
        if event.get("event") == "init":
            session = event.get("conversation_id")
            result = None
        elif event.get("event") == "result":
            result = event["result"]
            session = result.get("conversation_id") or session
    error = startup_error(path)
    if not result:
        error = error or "Antigravity exited without a result"
    elif result.get("status") != "SUCCESS":
        error = result.get("error") or result.get("status")
    return session, error


def startup_error(path):
    """Stop a missing-tool loop before spending the episode budget."""
    missing = 0
    for event in records(path):
        step = event.get("step_update") or {}
        info = step.get("tool_info") or {}
        if info.get("name") != "call_mcp_tool":
            continue
        if step.get("state") == "DONE":
            return None
        if step.get("state") == "ERROR":
            message = (info.get("error") or {}).get("message", "")
            missing += "not enabled for server" in message
    if missing >= 3:
        return "Antigravity could not discover robot MCP tools (three rejected calls)"
    return None


def metrics(path, wall_seconds, state=None):
    usage = {}
    completed_responses = {}
    turns = calls = 0
    for event in records(path):
        if event.get("event") == "result":
            result = event["result"]
            for key, value in (result.get("usage") or {}).items():
                usage[key] = usage.get(key, 0) + value
            turns += result.get("num_turns", 0)
        step = event.get("step_update") or {}
        if step.get("step_type") == "agent_response" and step.get("state") == "DONE":
            completed_responses[(step.get("conversation_id"), step.get("step_index"))] = step
        if step.get("step_type") == "tool" and step.get("state") in ("DONE", "ERROR"):
            calls += 1
    # Result events are absent on a host timeout. Native transcripts retain the
    # completed model requests, including those before the interrupted request.
    transcripts = list(Path(state).glob("brain/*/.system_generated/logs/transcript_full.jsonl")) if state else []
    if transcripts:
        usage = {}
        turns = 0
        for transcript in transcripts:
            for row in records(transcript):
                if row.get("type") != "PLANNER_RESPONSE":
                    continue
                turns += 1
                for key in ("input_tokens", "output_tokens", "cache_read_tokens"):
                    usage[key] = usage.get(key, 0) + row.get(key, 0)
    # Native transcript rows omit thinking usage. Recover this subset of output
    # tokens from terminal stream updates, deduplicated by conversation and step.
    # Do not label a partial count as the episode total.
    thinking = [(step.get("usage") or {}).get("thinking_tokens")
                for step in completed_responses.values()]
    if thinking and len(thinking) == turns and all(value is not None for value in thinking):
        usage["thinking_tokens"] = sum(thinking)
    return dict(wall_seconds=wall_seconds, agent_wall_seconds=wall_seconds,
                cost_usd=None, billed_to="api_key", num_turns=turns, tool_calls=calls,
                tokens_in=usage.get("input_tokens", 0), tokens_out=usage.get("output_tokens", 0),
                cache_read=usage.get("cache_read_tokens", 0), cache_creation=0,
                tokens_budget=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                tokens={"input": usage.get("input_tokens", 0), "output": usage.get("output_tokens", 0),
                        "thinking": usage.get("thinking_tokens"), "cache_read": usage.get("cache_read_tokens", 0)},
                usage=usage)


def condense(path):
    lines = []
    message = ""
    for event in records(path):
        step = event.get("step_update") or {}
        if step.get("text_delta"):
            message += step["text_delta"]
        if step.get("step_type") == "tool" and step.get("state") in ("DONE", "ERROR"):
            if message:
                lines.append(message)
                message = ""
            info = step.get("tool_info") or {}
            name, args = info.get("name", step.get("tool_name", "tool")), info.get("parameters", {})
            if name == "call_mcp_tool":
                name, args = args.get("ToolName", name), args.get("Arguments", {})
            lines.append(f"{name}: {json.dumps(args)}")
            if info.get("error"):
                lines.append(str(info["error"]))
    if message:
        lines.append(message)
    return "\n".join(lines)
