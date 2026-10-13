"""Immediate waypoint motion over the simulator's API.

The real gripper pose comes from robot.json; waypoints go out as robot-frame commands on the command socket.
A waypoint is done once robot.json's update_seq passes the value read before sending; the simulator bumps it
after the waypoint has run and its RGB-D snapshot is published.

Ported from hengyuan-hu/via-dev@2b749084cd6dc09d3b4399901f2d1cdb21931917, baseline/mcp_server.py; nothing
upstream is imported.
"""
from __future__ import annotations

import asyncio
import json
import time

import mcp.types as types
import numpy as np
from scipy.spatial.transform import Rotation

from core.kernel import EPISODE_OVER, fetch_sim_json, send_command


waypoint_lock = asyncio.Lock()

def text(value):
    return types.TextContent(type="text", text=value)


async def robot():
    return (await fetch_sim_json("robot.json")) or {}


async def prepare(ctx, *, action=False):
    if ((await fetch_sim_json("env.json")) or {}).get("episode_over"):
        return [text(EPISODE_OVER)]
    pending = getattr(ctx, "_via_pending", None)
    if pending is not None:
        if (await robot()).get("update_seq", -1) > pending:
            ctx._via_pending = None
        elif action:
            raise RuntimeError("Previous waypoint is still unacknowledged. Use observe; do not resend it.")
    return None


def selected_arm(ctx, arguments):
    side = arguments.get("arm")
    if len(ctx.arms) > 1:
        if side not in ("left", "right"):
            raise ValueError('Bimanual control requires arm="left" or arm="right" on every pose/gripper call')
    elif side is not None:
        raise ValueError('The single-arm task does not take an arm argument')
    return side


async def real_pose(ctx, side=None):
    arms = (await robot()).get("arms") or {}
    arm = next((a for a, v in arms.items() if v["side"] == side), None) if side else next(iter(arms), None)
    if arm is None:
        return None
    v = arms[arm]
    return {"pos": np.array(v["position"]), "euler": np.array(v["euler"]), "open": v["gripper_open"],
            "arm": arm, "side": v["side"]}


def proprio_dict(pose):
    rot = Rotation.from_euler("xyz", pose["euler"])
    rounded = lambda v: [round(float(x), 4) for x in v]
    return {**({"arm": pose["side"], "robot_arm": pose["arm"]} if pose.get("side") else {}),
            "position": rounded(pose["pos"]),
            "approach": rounded(rot.apply([0, 0, 1])),
            "opening": rounded(rot.apply([0, 1, 0])),
            "rpy_deg": rounded(np.degrees(pose["euler"])),
            "gripper": "open" if pose["open"] else "closed"}


def proprio_text(pose):
    if pose is None:
        return "Gripper pose unavailable: no sim frame received yet."
    return "Current real gripper pose:\n" + json.dumps(proprio_dict(pose))


async def execute(ctx, pose, pos, euler, opened, header):
    prev = (await robot()).get("update_seq", -1)
    ctx._via_pending = prev
    await send_command({"robot_waypoint": {"arm": pose["arm"], "position": [float(v) for v in pos],
                                           "euler": [float(v) for v in euler], "gripper_open": float(opened)}})
    stalled, last, checked = 0, -1, time.monotonic()
    while (await robot()).get("update_seq", -1) <= prev:
        await asyncio.sleep(0.2)
        if time.monotonic() - checked < 15:
            continue
        checked = time.monotonic()
        env = (await fetch_sim_json("env.json")) or {}
        if env.get("episode_over"):
            return [text(EPISODE_OVER)]
        used = env.get("sim_steps_used", -1)
        stalled = stalled + 1 if used == last else 0
        last = used
        if stalled >= 4:
            raise RuntimeError("Waypoint completion unacknowledged: simulator stopped progressing. "
                               "Use observe to check; do not resend the move.")
    ctx._via_pending = None
    env = (await fetch_sim_json("env.json")) or {}
    after = await real_pose(ctx, pose["side"])
    note = "\n" + EPISODE_OVER if env.get("episode_over") else ""
    delta = ""
    if after is not None:
        delta = "\nActual position delta (m): " + json.dumps(
            [round(float(v), 4) for v in after["pos"] - pose["pos"]])
    return await ctx.snap(header + "\n" + proprio_text(after) + delta + note)


class Tool:
    input_schema = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}

    def as_mcp_tool(self, arms=()):
        schema = self.input_schema
        description = self.description
        if len(arms) > 1 and self.name in {"get_proprio", "set_pose", "set_gripper"}:
            schema = {**schema, "properties": {**schema["properties"], "arm": {
                "type": "string", "enum": ["left", "right"],
                "description": "Arm side in the default scene view; right blue, left orange."}},
                "required": [*schema.get("required", []), "arm"]}
            description += " Requires explicit arm on every call; the other arm keeps its command."
        return types.Tool(name=self.name, description=description, inputSchema=schema)

    async def __call__(self, ctx, arguments):
        async with waypoint_lock:
            stopped = await prepare(ctx, action=self.name in {"set_pose", "set_gripper", "end_episode"})
            if stopped is not None:
                return stopped
            return await self.run(ctx, arguments)


class GetProprio(Tool):
    name = "get_proprio"
    description = ("Read the real gripper: position is fingertip midpoint in robot-frame metres; "
                   "approach and opening are unit vectors; rpy_deg is scipy xyz Euler orientation "
                   "in degrees, Rz(yaw) Ry(pitch) Rx(roll). Top-down is roll=+/-180, pitch=yaw=0. "
                   "gripper is the real robot's last open/closed command (closed also while holding "
                   "an object), not measured jaw width. No action or screenshot.")

    async def run(self, ctx, arguments):
        return [text(proprio_text(await real_pose(ctx, selected_arm(ctx, arguments))))]


class SetPose(Tool):
    name = "set_pose"
    description = ("Move immediately from the CURRENT real pose by one waypoint. d_x/d_y/d_z "
                   "are metres along fixed robot axes (+z lifts). d_roll/d_pitch/d_yaw are degrees "
                   "about fixed robot X/Y/Z axes, composed roll then pitch then yaw as dR * Rcurrent, "
                   "not additions to Euler readouts. Rotation pivots about the fingertip midpoint; "
                   "position stays fixed unless translation is requested. Prefer one rotation axis "
                   "per call. Omitted deltas are zero; all-zero is a no-op. Returns screenshot, "
                   "actual pose and actual position delta; blocked moves can stop short. No jaw "
                   "change; use set_gripper separately. Robot z includes the task's table height.")
    input_schema = {"type": "object", "properties": {
        k: {"type": "number", "default": 0, "description": unit}
        for k, unit in [("d_x", "Robot X metres"), ("d_y", "Robot Y metres"),
                        ("d_z", "Robot Z metres"), ("d_roll", "Fixed robot X degrees"),
                        ("d_pitch", "Fixed robot Y degrees"), ("d_yaw", "Fixed robot Z degrees")]},
                    "required": [], "additionalProperties": False}

    async def run(self, ctx, arguments):
        d = np.array([float(arguments.get(k, 0)) for k in self.input_schema["properties"]])
        if not np.isfinite(d).all():
            raise ValueError("Pose deltas must be finite numbers")
        pose = await real_pose(ctx, selected_arm(ctx, arguments))
        if pose is None or not np.any(d):
            return await ctx.snap("set_pose did not run: no frame or all-zero deltas.\n" + proprio_text(pose))
        target = (Rotation.from_euler("xyz", d[3:], degrees=True)
                  * Rotation.from_euler("xyz", pose["euler"])).as_euler("xyz")
        return await execute(ctx, pose, pose["pos"] + d[:3], target, pose["open"], "Executed set_pose.")


class SetGripper(Tool):
    name = "set_gripper"
    description = ("Open or close the real gripper immediately, with no pose change (one waypoint). "
                   "Already in that command state is a no-op. Returns screenshot and real pose. "
                   "Close only once the fingers surround the object; move and change jaws separately.")
    input_schema = {"type": "object", "properties": {"state": {"type": "string", "enum": ["open", "closed"]}},
                    "required": ["state"], "additionalProperties": False}

    async def run(self, ctx, arguments):
        if arguments["state"] not in ("open", "closed"):
            raise ValueError("state must be open or closed")
        pose = await real_pose(ctx, selected_arm(ctx, arguments))
        opened = arguments["state"] == "open"
        if pose is None or pose["open"] == opened:
            return await ctx.snap("set_gripper did not run: no frame or already in that state.\n" + proprio_text(pose))
        return await execute(ctx, pose, pose["pos"], pose["euler"], opened, "Executed set_gripper.")
