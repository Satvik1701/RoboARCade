"""Write docs/interfaces/<package>.md for each shipped interface: what the model is given on an evaluation
episode, verbatim. The system prompt (the interface guide), the first user message (the task prompt), the
pre-compaction prompt, and every MCP tool with the description and input schema the model sees, loaded from
the package's tools.py the same way the MCP server loads it. Run: .venv/bin/python docs/interfaces/build.py"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from core.mcp_server import load_interface   # noqa: E402
from core.run_eval import build_prompt        # noqa: E402

# (package, page title, the arm names a dual-arm pack reports; the tools add an `arm` parameter for them)
INTERFACES = [("via_cam", "VIA", ()), ("via_bimanual_cam", "VIA bimanual", ("move", "hold")),
              ("rgbd_waypoint", "Multi-cam RGBD", ())]
PRECOMPACT = (ROOT / "core/compaction/prompt.txt").read_text()


def fence(text, lang="text"):
    return f"```{lang}\n{text.rstrip()}\n```\n"


def tool_section(tool):
    props = tool.inputSchema.get("properties", {})
    required = set(tool.inputSchema.get("required", ()))
    rows = []
    for name, p in props.items():
        kind = p.get("type", "")
        if "enum" in p:
            kind = " | ".join(f"`{v}`" for v in p["enum"])
        elif kind == "array":
            kind = f"array of {p.get('items', {}).get('type', '')}"
        extra = ", ".join(f"{k} {v}" for k, v in p.items() if k in ("minimum", "maximum", "default", "minItems", "maxItems"))
        desc = " ".join(str(p.get("description", "")).split())
        rows.append(f"| `{name}` | {kind} | {'yes' if name in required else ''} | {desc}{(' (' + extra + ')') if extra else ''} |")
    table = ("\n| Parameter | Type | Required | Description |\n|---|---|---|---|\n" + "\n".join(rows) + "\n") if rows else "\nNo parameters.\n"
    return f"### `{tool.name}`\n\n{fence(tool.description)}{table}"


for package, title, arms in INTERFACES:
    tools = [t.as_mcp_tool(arms) for t in load_interface(package).values()]
    guide = (ROOT / "interfaces" / package / "guide.md").read_text()
    arm_note = (f" This page renders them for a two-arm pack whose arms are named `{arms[0]}` and `{arms[1]}`, "
                f"which is where the `arm` parameter comes from." if arms else "")
    page = (
        f"# {title} (`{package}`)\n\n"
        f"What the model receives on an episode with this interface, verbatim, in the order it arrives:\n\n"
        f"- **System prompt**: standing instructions, kept for the whole episode. The interface guide "
        f"`interfaces/{package}/guide.md`, appended to the runtime's own system prompt "
        f"(`--append-system-prompt-file` for Claude Code, `AGENTS.md` for Codex).\n"
        f"- **First user message**: the request the model works on. The task prompt, identical for every episode "
        f"except the `Task:` line, which carries the pack's instruction. No further user message follows: the rest "
        f"of the conversation is the model's tool calls and their results (text plus images).\n"
        f"- **Tools**: the {len(tools)} MCP tools of [`tools.py`](../../interfaces/{package}/tools.py), "
        f"with the description and input schema the model sees.{arm_note}\n"
        f"- **Pre-compaction prompt**: appended to the runtime's summarizer when the context is compacted.\n\n"
        f"## System prompt\n\n{fence(guide, 'markdown')}\n"
        f"## First user message\n\n{fence(build_prompt('<the instruction from the pack, task.json language>'))}\n"
        f"## Pre-compaction prompt\n\n{fence(PRECOMPACT)}\n"
        f"## Tools\n\n" + "\n".join(tool_section(t) for t in tools))
    (HERE / f"{package}.md").write_text(page)
    print(f"wrote docs/interfaces/{package}.md ({len(tools)} tools, {len(page) // 1024} KB)")
