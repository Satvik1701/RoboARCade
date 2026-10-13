"""MCP server: serves an interface package's tools to the agent.

It loads an interface package's tools.py (default: via_cam) and dispatches MCP calls to it. An interface with a browser
page gets a UiKernel attached over CDP to sim_env_server's Chromium; one whose manifest.json sets "ui": false gets a
SimKernel and uses the simulator's API alone.

Start sim_env_server.py first. Select the interface with --interface <name|path> or
ROBOARCADE_INTERFACE (a bare name resolves to interfaces/<name>/; a path
points at any directory with a tools.py).
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib.util
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jsonschema
import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server
from playwright.async_api import async_playwright

from core.kernel import ATTACH_TIMEOUT_S, CDP_URL, EPISODE_OVER, UI_GONE, UI_UNATTACHED, SimKernel, UiKernel, attach_ui, \
    fetch_sim_json
from core.utils import interface_dir, interface_manifest, setup_logging

logger = logging.getLogger(__name__)

app = Server("roboarcade")

_ctx: UiKernel | None = None
_tools: dict = {}
_arms: tuple = ()
_env_ready = asyncio.Event()
_ready = asyncio.Event()
_attach_error: str | None = None


def load_interface(spec: str):
    tools_py = interface_dir(spec) / "tools.py"
    if not tools_py.is_file():
        raise ModuleNotFoundError(f"no interface package {spec!r} at {tools_py}")
    mod_spec = importlib.util.spec_from_file_location("roboarcade_interface_tools", tools_py)
    module = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(module)
    return module.TOOLS


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    await _env_ready.wait()
    return [tool.as_mcp_tool(_arms) for tool in _tools.values()]


@app.call_tool()
async def call_tool(name: str, arguments: dict | None) -> list[types.ContentBlock]:
    await _ready.wait()
    if _ctx is None:
        return [types.TextContent(type="text", text=f"UI not attached: {_attach_error}")]
    t0, before = time.time(), (await fetch_sim_json("env.json")) or {}
    blocks = await _run(name, arguments, before)
    await _log_tool(name, arguments, blocks, t0, before)
    return blocks


async def _log_tool(name, arguments, blocks, t0, before):
    after = (await fetch_sim_json("env.json")) or {}
    shots = before.get("screenshots_dir") or after.get("screenshots_dir")
    if not shots or not os.path.isdir(shots):
        return
    row = {"t": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t0)), "tool": name,
           "args": arguments or {}, "steps_before": before.get("sim_steps_used"),
           "steps_after": after.get("sim_steps_used"), "wall_s": round(time.time() - t0, 2),
           "images": sum(getattr(b, "type", "") == "image" for b in blocks),
           "result": " ".join(" ".join(b.text for b in blocks if b.type == "text").split())[:300]}
    with open(os.path.join(shots, "tools.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")


async def _run(name: str, arguments: dict | None, env_before: dict):
    page = getattr(_ctx, "page", None)
    if page is not None and (not page.context.browser.is_connected() or page.is_closed()):
        return [types.TextContent(type="text", text=UI_GONE)]
    if env_before.get("episode_over"):
        return [types.TextContent(type="text", text=EPISODE_OVER)]
    tool = _tools.get(name)
    if tool is None:
        return [types.TextContent(type="text", text=f"Unknown tool: {name}")]
    try:
        jsonschema.validate(arguments or {}, tool.as_mcp_tool(_ctx.arms).inputSchema)
    except jsonschema.ValidationError as e:
        return [types.TextContent(type="text", text=f"Input validation error: {e.message}")]
    return await tool(_ctx, arguments or {})


async def main() -> None:
    global _ctx, _tools
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interface",
        default=os.environ.get("ROBOARCADE_INTERFACE", "via-cam"),
        help="interface package: a shipped name (via_cam, via_bimanual_cam) or a path to a package dir",
    )
    args = parser.parse_args()
    _tools = load_interface(args.interface)
    ui = interface_manifest(args.interface).get("ui", True)
    logger.info(f"loaded interface {args.interface!r} ({len(_tools)} tools, {'with' if ui else 'without'} a browser page)")

    async with contextlib.AsyncExitStack() as stack:
        pw = await stack.enter_async_context(async_playwright()) if ui else None
        attach = asyncio.create_task(_attach(pw))
        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, app.create_initialization_options())
        attach.cancel()


async def _attach(pw) -> None:
    global _ctx, _arms, _attach_error
    try:
        deadline = time.time() + ATTACH_TIMEOUT_S
        while (info := await fetch_sim_json("env.json")) is None and time.time() < deadline:
            await asyncio.sleep(0.5)
        info = info or {}
        _arms = tuple(info.get("arms") or ())
        _env_ready.set()
        if pw is None:
            _ctx = SimKernel(arms=_arms)
            _ready.set()
            return
        logger.info(f"Connecting to sim_env_server browser at {CDP_URL} ...")
        page = await attach_ui(pw)
        logger.info(f"Attached to sim_env_server browser (page url={page.url}).")
        _ctx = UiKernel(page, ui_z_offset=float(info["ui_z_offset"]), arms=_arms)
    except Exception as e:
        _attach_error = f"{type(e).__name__}: {e}"
        logger.error(f"[attach] {_attach_error}")
        shots = os.environ.get("ROBOARCADE_SHOTS_DIR")
        if shots and os.path.isdir(shots):
            Path(shots, UI_UNATTACHED).write_text(_attach_error)
        _env_ready.set()
    _ready.set()


def _log_to_shots_dir() -> None:
    shots = os.environ.get("ROBOARCADE_SHOTS_DIR")
    if not shots or not os.path.isdir(shots):
        return
    h = logging.FileHandler(os.path.join(shots, "mcp_server.log"))
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(h)


if __name__ == "__main__":
    setup_logging()
    _log_to_shots_dir()
    try:
        asyncio.run(main())
    except Exception:
        logging.getLogger(__name__).exception("mcp server died")
        raise
