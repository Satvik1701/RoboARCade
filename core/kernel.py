"""What interface tools run on: the browser page or the simulator's API.

UiKernel drives the 3D browser page over CDP; sim_env_server owns the Chromium, and the kernel edits the page through
`window.roboarcade` (core/UI/index.html), the same functions the page's key and mouse handlers call. SimKernel has no
page and uses the simulator's API alone (fetch_sim_json, send_command). Interface packages
(`interfaces/<name>/tools.py`) define their MCP tools on one of the two.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time

import httpx
import mcp.types as types
import websockets

logger = logging.getLogger(__name__)

BASE_PORT = int(os.environ.get("ROBOARCADE_BASE_PORT", "8100"))
CDP_URL = f"http://localhost:{BASE_PORT + 3}"
SIM_HTTP_BASE = f"http://localhost:{BASE_PORT + 1}"
UI_URL = f"http://localhost:{BASE_PORT}"

SCREENSHOT_QUALITY = 60

GRIPPER_STEP_M = 0.06 / 10.0

EPISODE_OVER = "EPISODE OVER: the episode has ended and no further actions are possible. Stop now."
UI_GONE = "The simulation UI is no longer reachable; the episode is over. Stop."
UI_UNATTACHED = "ui_unattached.txt"


def fmt4(value) -> str:
    return f"{float(value):.4f}"


async def fetch_sim_json(path: str) -> dict | None:
    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            resp = await client.get(f"{SIM_HTTP_BASE}/{path}")
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return None


async def send_command(command: dict) -> None:
    async with websockets.connect(f"ws://localhost:{BASE_PORT + 2}") as ws:
        await ws.send(json.dumps([command]))


class SimKernel:
    def __init__(self, arms: tuple = ()):
        self.arms = arms


class UiKernel:
    def __init__(self, page, ui_z_offset: float = 0.8, arms: tuple = ()):
        self.page = page
        self.ui_z_offset = ui_z_offset
        self.arms = arms

    def _image_content(self, img: bytes) -> types.ImageContent:
        return types.ImageContent(
            type="image", data=base64.b64encode(img).decode(), mimeType="image/jpeg"
        )

    async def snap(self, text: str | None = None) -> list[types.ContentBlock]:
        img = await self.page.screenshot(type="jpeg", quality=SCREENSHOT_QUALITY, scale="css")
        text_blocks = [types.TextContent(type="text", text=text)] if text else []
        return [*text_blocks, self._image_content(img)]

    async def show_cursor(self, x: int, y: int, x1: int | None = None, y1: int | None = None) -> None:
        await self.call("showCursor", x, y, x1, y1)

    async def perform_drag(self, x1: int, y1: int, x2: int, y2: int,
                           button: str = "left", steps: int = 25,
                           held_key: str | None = None) -> None:
        if held_key:
            await self.page.keyboard.down(held_key)
        try:
            await self.page.mouse.move(x1, y1)
            await self.page.mouse.down(button=button)
            await self.page.mouse.move(x2, y2, steps=steps)
            await self.page.mouse.up(button=button)
        finally:
            if held_key:
                await self.page.keyboard.up(held_key)
        await self.show_cursor(x2, y2, x1, y1)

    async def uv_to_xy(self, u: float, v: float) -> tuple[int, int, int, int]:
        size = await self.page.evaluate(
            "() => ({ width: window.innerWidth, height: window.innerHeight })"
        )
        width = int(size["width"])
        height = int(size["height"])
        x = min(max(round(float(u) * width), 0), max(width - 1, 0))
        y = min(max(round(float(v) * height), 0), max(height - 1, 0))
        return x, y, width, height

    async def contact_point_text(self, x: int, y: int, gripper: str) -> str:
        preview = await self.page.evaluate("([x, y]) => window.previewContactPointUI(x, y)", [x, y])
        surface = preview.get("surface")
        point = preview.get("point")
        if point:
            robot = self.ui_to_robot(point["x"], point["y"], point["z"])
            where = "" if surface == "pointcloud" else f" via {surface} feed"
            return (
                f"  contact point if clicked here{where} (robot, m): "
                f"x={robot['x']:.4f}  y={robot['y']:.4f}  z={robot['z']:.4f}"
            )
        if surface == "gripper":
            return f"  contact point: none - over the {gripper} (a click here drags, not places)"
        if surface == "other":
            return "  contact point: none - not a placement surface (point-cloud canvas or a camera feed)"
        return "  contact point: none - (u, v) is not over a point in the cloud"

    def ui_to_robot(self, ui_x: float, ui_y: float, ui_z: float) -> dict[str, float]:
        return {
            "x": round(ui_x / 10.0, 4),
            "y": round(-ui_z / 10.0, 4),
            "z": round(ui_y / 10.0 + self.ui_z_offset, 4),
        }

    def robot_to_ui(self, x: float, y: float, z: float) -> dict[str, float]:
        return {"x": x * 10.0, "y": (z - self.ui_z_offset) * 10.0, "z": -y * 10.0}

    async def call(self, action: str, *args) -> dict:
        result = await self.page.evaluate(
            "([action, args]) => window.roboarcade[action](...args)", [action, list(args)])
        if "gripper" in result:
            ui = result["gripper"]["ui_position"]
            result["gripper"]["robot_position"] = self.ui_to_robot(ui["x"], ui["y"], ui["z"])
        return result

    async def camera_pose(self) -> dict:
        return (await self.call("state"))["camera"]

    def format_camera_pose(self, pose: dict) -> str:
        t = pose["target"]
        rt = self.ui_to_robot(t["x"], t["y"], t["z"])
        return (
            "Camera pose\n"
            f"  azimuth:   {fmt4(pose['azimuth'])} deg (horizontal orbit angle)\n"
            f"  elevation: {fmt4(pose['elevation'])} deg (0=horizontal, 90=top-down)\n"
            f"  distance:  {fmt4(pose['distance'] / 10.0)} m (camera to target)\n"
            f"  target (robot, m): x={fmt4(rt['x'])}  y={fmt4(rt['y'])}  z={fmt4(rt['z'])}"
        )

    async def camera_pose_text(self) -> str:
        return self.format_camera_pose(await self.camera_pose())

    async def gripper_pose(self) -> dict:
        return (await self.call("state"))["gripper"]

    async def set_arm(self, arm: str) -> None:
        await self.call("selectArm", arm)

    async def home(self, arm: str | None = None) -> dict:
        return await self.call("home", arm)


ATTACH_TIMEOUT_S = 60.0
CDP_TIMEOUT_MS = 40_000


async def attach_ui(pw):
    deadline, t0, tries = time.time() + ATTACH_TIMEOUT_S, time.time(), 0
    browser = page = None
    while page is None and (left := deadline - time.time()) > 0:
        if browser is None:
            tries += 1
            try:
                browser = await pw.chromium.connect_over_cdp(
                    CDP_URL, timeout=min(CDP_TIMEOUT_MS, left * 1000))
            except Exception as e:
                logger.warning(f"[attach_ui] connect attempt {tries} failed after "
                               f"{time.time() - t0:.1f}s: {type(e).__name__}: {str(e)[:200]}")
                await asyncio.sleep(0.5)
                continue
        page = next((p for c in browser.contexts for p in c.pages if p.url.startswith(UI_URL)),
                    None)
        if page is None:
            await asyncio.sleep(0.25)
    logger.info(f"[attach_ui] attached={page is not None} after {time.time() - t0:.1f}s "
                f"in {tries} connect attempt(s)")
    if page is None:
        seen = [p.url for c in browser.contexts for p in c.pages] if browser else "not connected"
        raise RuntimeError(f"no page at {UI_URL} within {ATTACH_TIMEOUT_S:.0f}s "
                           f"(is sim_env_server running?); browser pages={seen}")

    await page.set_viewport_size({"width": 1400, "height": 900})
    return page
