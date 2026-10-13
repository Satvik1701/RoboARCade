"""The via-cam interface tools.

Beyond the VIA tools: `camera_views` overlays live renderings of every selectable
camera on the point-cloud panel, `camera_feed_select` switches the 3rd-person feed
(clicks on the feed then project through that camera), and `camera_raw` fetches one
camera's native frame from sim_env_server's /cam/<name>.jpg without the browser.
via_bimanual_cam/tools.py subclasses these tools with dual-arm wording.
"""
from __future__ import annotations

import re

import httpx
import mcp.types as types
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from core.kernel import (
    EPISODE_OVER,
    GRIPPER_STEP_M,
    UiKernel,
    SIM_HTTP_BASE,
    fetch_sim_json,
    fmt4,
)

FRAMES_PREAMBLE = (
    "Frame (shared by all spatial tools):\n"
    "  robot - metres; the canonical frame every pose, delta, and camera target "
    "reports in and that every tool input expects.\n"
    "  orientation - the gripper's pose is reported as two robot-frame unit "
    "vectors: 'approach' (the direction the fingers reach) and 'opening' (the "
    "fingertip-to-fingertip axis the jaws open/close along).\n"
    "Robot Z position carries a fixed height offset, so a canonical top-down "
    "gripper reads z~1.0, not 0; reason about position deltas.\n\n"
)

_UV_PROPS = {
    "u": {
        "type": "number",
        "minimum": 0.0,
        "maximum": 1.0,
        "description": "Fraction across the screenshot, 0.0 left to 1.0 right.",
    },
    "v": {
        "type": "number",
        "minimum": 0.0,
        "maximum": 1.0,
        "description": "Fraction down the screenshot, 0.0 top to 1.0 bottom.",
    },
}


def format_gripper_pose(pose: dict, gripper: str, heading: str = "Current Target Gripper Pose") -> str:
    def fmt_dir(value: float) -> str:
        return f"{float(value):+.3f}"

    robot_pos = pose["robot_position"]
    approach = pose["robot_approach"]
    opening = pose["robot_opening"]
    return (
        f"{heading}:\n"
        f"Position is the midpoint between the fingertips of the {gripper}. "
        "Orientation is given as two robot-frame unit vectors, approach and opening.\n"
        f"  position (robot, m):    x={fmt4(robot_pos['x'])}  "
        f"y={fmt4(robot_pos['y'])}  z={fmt4(robot_pos['z'])}\n"
        f"  approach (robot, unit): x={fmt_dir(approach['x'])}  "
        f"y={fmt_dir(approach['y'])}  z={fmt_dir(approach['z'])}  "
        "(direction the fingers reach; the grasp-approach axis)\n"
        f"  opening (robot, unit):  x={fmt_dir(opening['x'])}  "
        f"y={fmt_dir(opening['y'])}  z={fmt_dir(opening['z'])}  "
        "(fingertip-to-fingertip axis the jaws open/close along)\n"
        f"  gripper: {'open' if pose['gripper_open'] else 'closed'}"
    )

class ToolBase:
    name: str
    description: str
    input_schema: dict
    frames_preamble: bool = False
    preamble = FRAMES_PREAMBLE
    arm_param: bool = False
    gripper = "blue target gripper"

    def as_mcp_tool(self, arms: tuple = ()) -> types.Tool:
        description = self.description
        if self.frames_preamble:
            description = self.preamble + description
        schema = self.input_schema
        if self.arm_param and len(arms) > 1:
            schema = {**schema, "properties": {**schema["properties"], "arm": self.arm_prop(arms)}}
        return types.Tool(name=self.name, description=description, inputSchema=schema)

    def arm_prop(self, arms: tuple) -> dict:
        return {
            "type": "string",
            "enum": list(arms),
            "description": (
                f"Arm this tool commands. Default: {arms[0]!r} (the move arm). "
                "The selection is sticky - it persists for later tools until changed."
            ),
        }

    async def _select_arm(self, ctx: UiKernel, arguments: dict) -> None:
        if arguments.get("arm"):
            await ctx.set_arm(str(arguments["arm"]))

    async def format_pose(self, ctx: UiKernel, pose: dict | None) -> str:
        return format_gripper_pose(pose, self.gripper)

    async def pose_text(self, ctx: UiKernel) -> str:
        return await self.format_pose(ctx, await ctx.gripper_pose())

    async def contact_text(self, ctx: UiKernel, x: int, y: int) -> str:
        return await ctx.contact_point_text(x, y, self.gripper)

    async def __call__(self, ctx: UiKernel, arguments: dict) -> list[types.ContentBlock]:
        raise NotImplementedError

    async def _start_hits_gripper(self, ctx: UiKernel, x: int, y: int) -> bool:
        hit = await ctx.page.evaluate(
            """([x, y]) => {
            const rect = renderer.domElement.getBoundingClientRect();
            if (
                x < rect.left || x > rect.right ||
                y < rect.top || y > rect.bottom ||
                rect.width <= 0 || rect.height <= 0
            ) {
                return false;
            }

            const raycaster = new THREE.Raycaster();
            raycaster.setFromCamera(new THREE.Vector2(
                ((x - rect.left) / rect.width) * 2 - 1,
                -((y - rect.top) / rect.height) * 2 + 1
            ), camera);
            selectedObject.updateMatrixWorld(true);
            return raycaster.intersectObject(selectedObject, false).length > 0;
        }""",
            [x, y],
        )
        return bool(hit)


class ScreenshotTool(ToolBase):
    name = "screenshot"
    description = (
        "Capture a screenshot of the robot interface.\n\n"
        "The window shows:\n"
        "  - Left sidebar: two camera feeds. Both are rendered from the real "
        "scene and they refresh only after execute_waypoint moves the real "
        "gripper.\n"
        "      - 3rd-person view: a fixed camera looking at the table and gripper. "
        "Best for analyzing the full scene layout (objects relative to each other "
        "and the gripper).\n"
        "      - Wrist view: a camera on the gripper looking down between the fingers. "
        "Best for the close-up gripper-object relationship (whether the gripper is "
        "well positioned for a grasp or place).\n"
        "  - Right main view: a 3-D point cloud reconstruction of the scene, with "
        "robot-frame axes (X red, Y green, Z blue) and the blue target gripper mesh you "
        "command. The blue gripper renders RED while the controller cannot reach its current pose - a reachability warning only, never a collision or contact indicator. "
        "Best for 3D / depth understanding that the flat camera images "
        "can't give.\n\n"
        "Call this first, or whenever you need a fresh view without taking an action."
    )
    input_schema = {"type": "object", "properties": {}, "required": []}

    async def __call__(self, ctx, arguments):
        return await ctx.snap()


class HoverTool(ToolBase):
    name = "hover"
    description = (
        "Move the mouse to normalized screenshot coordinates (u, v) and show "
        "a visible crosshair overlay in the returned screenshot.\n\n"
        "Coordinate convention:\n"
        "  u = fraction across the screenshot, 0.0 left -> 1.0 right\n"
        "  v = fraction down the screenshot, 0.0 top -> 1.0 bottom\n\n"
        "Reading the overlay:\n"
        "  The exact (u, v) point is the crosshair's center dot.\n\n"
        "Use hover to inspect a spot before acting on it. It takes no action and "
        "gives two readouts:\n"
        "  - Visual: the crosshair (a persistent DOM overlay, not the OS cursor) "
        "marks exactly where (u, v) lands, confirming it's on the intended UI "
        "element or point-cloud region.\n"
        "  - Contact point (robot-frame x/y/z, m), computed read-only. On the "
        "point-cloud canvas or a camera feed it reports the cloud point under the "
        "cursor (the ray needn't be pixel-exact - it hits within a small radius). "
        "Two uses:\n"
        "      1. Probe the robot-frame coordinate of any point of interest.\n"
        "      2. Preview a teleport - when it lands on a point it is exactly the "
        "salient point a gripper_teleport_via_click at this same (u, v) would select. "
        "Note this is the clicked point itself, NOT the resulting gripper pose (which "
        "lands ~0.07 m back along the approach axis).\n"
        "    It reads 'none' when (u, v) is not over a cloud point (empty space), "
        "over the blue target gripper, or over UI chrome rather than the point-cloud "
        "canvas or a camera feed."
    )
    input_schema = {"type": "object", "properties": dict(_UV_PROPS), "required": ["u", "v"]}

    async def __call__(self, ctx, arguments):
        u = float(arguments["u"])
        v = float(arguments["v"])
        x, y, width, height = await ctx.uv_to_xy(u, v)
        await ctx.page.mouse.move(x, y)
        await ctx.show_cursor(x, y)
        return await ctx.snap(
            f"Hovered at u={u:.4f}, v={v:.4f} "
            f"(browser pixel x={x}, y={y}, viewport={width}x{height})\n"
            f"{await self.contact_text(ctx, x, y)}"
        )


class GripperTeleportViaClickTool(ToolBase):
    name = "gripper_teleport_via_click"
    frames_preamble = True
    arm_param = True
    description = (
        "Fast-move the blue target gripper near a clicked salient point. "
        "Returns a screenshot. Best for quickly setting a subtask's starting "
        "position; refine afterward with gripper_drag/move/rotate/toggle.\n\n"
        "Coordinate convention:\n"
        "  u = fraction across the screenshot, 0.0 left -> 1.0 right\n"
        "  v = fraction down the screenshot, 0.0 top -> 1.0 bottom\n\n"
        "Where to click:\n"
        "  - Left-panel 2-D camera views: best when the target is clearly "
        "visible in either camera feed.\n"
        "  - Right-panel 3-D point-cloud view: best when locating the target "
        "takes some 3D reasoning about depth.\n"
        "  - Click the object's center, not its edge, so the gripper lands "
        "centered over it for a reliable grasp or place.\n\n"
        "What happens:\n"
        "  The gripper lands ~0.07 m back from the clicked point along its "
        "current approach axis (opposite the reach direction), not on the point "
        "itself - so a short forward move/descent then reaches it. E.g. if the "
        "gripper currently points straight down, this puts it ~0.07 m directly "
        "above the click. Translation only - the gripper's orientation is "
        "preserved.\n"
        "  The readout reports the clicked point as the 'selected salient point' "
        "and the resulting gripper pose separately; check both, plus the "
        "screenshot. Clicking the blue target gripper itself does nothing here - that "
        "is reserved for gripper_drag."
    )
    input_schema = {"type": "object", "properties": dict(_UV_PROPS), "required": ["u", "v"]}

    async def _surface_at(self, ctx: UiKernel, x: int, y: int) -> dict:
        surface = await ctx.page.evaluate(
            """([x, y]) => {
            const el = document.elementFromPoint(x, y);
            const closest = (selector) => el ? el.closest(selector) : null;

            if (closest('#btn-record')) {
                return { kind: 'action', label: 'Execute Waypoint button', tool: 'execute_waypoint' };
            }
            if (closest('#btn-end')) {
                return { kind: 'action', label: 'End Episode button', tool: 'end_episode' };
            }

            const cam = closest('#cam-agentview, #cam-wrist, #cam-wrist2');
            if (cam) {
                const lbl = cam.closest('.cam-feed').querySelector('.cam-label');
                return {
                    kind: 'gripper_surface',
                    label: cam.id === 'cam-agentview'
                        ? 'agentview camera feed' : lbl.textContent.trim() + ' feed',
                };
            }

            if (el === renderer.domElement) {
                return { kind: 'gripper_surface', label: '3-D point-cloud canvas', canvas: true };
            }

            let label = 'unknown UI surface';
            if (el) {
                label = el.id ? `#${el.id}` : el.tagName.toLowerCase();
            }
            return { kind: 'unsupported', label };
        }""",
            [x, y],
        )
        if surface.get("canvas") and await self._start_hits_gripper(ctx, x, y):
            return {"kind": "unsupported", "label": "target gripper mesh; use gripper_drag"}
        return surface

    async def __call__(self, ctx, arguments):
        await self._select_arm(ctx, arguments)
        u = float(arguments["u"])
        v = float(arguments["v"])
        x, y, width, height = await ctx.uv_to_xy(u, v)
        surface = await self._surface_at(ctx, x, y)

        await ctx.show_cursor(x, y)
        if surface["kind"] == "action":
            return await ctx.snap(
                f"gripper_teleport_via_click did not run: u={u:.4f}, v={v:.4f} lands on the "
                f"{surface['label']}. Use the {surface['tool']} tool instead.\n"
                f"  browser pixel: x={x}, y={y}, viewport={width}x{height}\n"
                f"{await self.pose_text(ctx)}"
            )

        if surface["kind"] != "gripper_surface":
            return await ctx.snap(
                f"gripper_teleport_via_click did not run: u={u:.4f}, v={v:.4f} lands on "
                f"{surface['label']}, not a gripper placement surface.\n"
                "  Click the 3-D point-cloud canvas or a camera feed, or use "
                "execute_waypoint/end_episode for episode actions.\n"
                f"  browser pixel: x={x}, y={y}, viewport={width}x{height}\n"
                f"{await self.pose_text(ctx)}"
            )

        await ctx.page.mouse.move(x, y)
        result = await ctx.call("teleport", x, y)
        clicked_point = result["click"]
        if clicked_point:
            robot = ctx.ui_to_robot(clicked_point["x"], clicked_point["y"], clicked_point["z"])
            return await ctx.snap(
                f"Teleported {self.gripper} from {surface['label']} click "
                f"at u={u:.4f}, v={v:.4f}\n"
                f"  browser pixel: x={x}, y={y}, viewport={width}x{height}\n"
                f"  selected salient point (robot, m): "
                f"x={robot['x']:.4f}  y={robot['y']:.4f}  z={robot['z']:.4f}\n"
                f"{await self.format_pose(ctx, result['gripper'])}"
            )
        return await ctx.snap(
            f"Clicked {surface['label']} at u={u:.4f}, v={v:.4f}, "
            "but no 3-D gripper target is available.\n"
            f"  browser pixel: x={x}, y={y}, viewport={width}x{height}\n"
            f"{await self.format_pose(ctx, result['gripper'])}"
        )


class GripperDragTool(ToolBase):
    name = "gripper_drag"
    frames_preamble = True
    arm_param = True
    description = (
        "Drag the blue target gripper between normalized screenshot coords "
        "(u across 0->1 left->right, v down 0->1 top->bottom). Returns a screenshot.\n\n"
        "The blue target gripper is the target pose you command; this tool moves it. "
        "The gray point-cloud gripper is actual sensor data and is not a draggable mesh.\n\n"
        "Start point: u1/v1 must ray-hit the solid blue target gripper mesh - a finger, wrist, "
        "or body, not the empty gap between the fingers or the gray gripper. A miss is "
        "refused with a no-op.\n\n"
        "Motion model: free drag moves the gripper with the cursor in the view "
        "plane, ~0.0006 m per CSS pixel of mouse movement, before constraints.\n\n"
        "Movement modes - free is the default (no lock); x/y/z lock motion to one "
        "robot-frame axis (the fixed robot X/Y/Z, not the gripper-attached gizmo "
        "axes that gripper_rotate spins about) and are exact at any camera angle, "
        "so trust the pose readout, not the picture:\n"
        "  free - unconstrained: follows the cursor in the camera view plane (camera-dependent)\n"
        "  x    - lock to robot X only\n"
        "  y    - lock to robot Y only\n"
        "  z    - lock to robot Z / vertical\n\n"
        "Tip: hover first to confirm u1/v1 lands on the gripper, then use small single-axis "
        "drags. The screenshot shows start/end markers, end crosshair, the arrow path, and the "
        "updated pose."
    )
    miss_hint = ("  aim at a solid blue jaw block, wrist, or body; avoid the hollow "
                 "slot between jaws and the inactive gray gripper.")
    input_schema = {
        "type": "object",
        "properties": {
            "u1": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Start fraction across the screenshot, 0.0 left to 1.0 right.",
            },
            "v1": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Start fraction down the screenshot, 0.0 top to 1.0 bottom.",
            },
            "u2": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "End fraction across the screenshot, 0.0 left to 1.0 right.",
            },
            "v2": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "End fraction down the screenshot, 0.0 top to 1.0 bottom.",
            },
            "constraint": {
                "type": "string",
                "enum": ["free", "x", "y", "z"],
                "description": "Movement constraint. Default: free.",
            },
            "steps": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100,
                "description": "Mouse move interpolation steps. Default: 25.",
            },
        },
        "required": ["u1", "v1", "u2", "v2"],
    }

    async def __call__(self, ctx, arguments):
        await self._select_arm(ctx, arguments)
        u1 = float(arguments["u1"])
        v1 = float(arguments["v1"])
        u2 = float(arguments["u2"])
        v2 = float(arguments["v2"])
        x1, y1, width, height = await ctx.uv_to_xy(u1, v1)
        x2, y2, _, _ = await ctx.uv_to_xy(u2, v2)
        steps = int(arguments.get("steps", 25))

        constraint = arguments.get("constraint", "free")

        if not await self._start_hits_gripper(ctx, x1, y1):
            await ctx.show_cursor(x1, y1)
            return await ctx.snap(
                f"gripper_drag did not run: u1={u1:.4f}, v1={v1:.4f} "
                f"does not ray-hit solid geometry on the {self.gripper} mesh.\n"
                f"  browser pixel: x={x1}, y={y1}, viewport={width}x{height}\n"
                f"{self.miss_hint}\n"
                f"{await self.pose_text(ctx)}"
            )

        held_key = None if constraint == "free" else constraint
        await ctx.perform_drag(x1, y1, x2, y2, steps=steps, held_key=held_key)

        return await ctx.snap(
            f"Gripper dragged u={u1:.4f}, v={v1:.4f} -> u={u2:.4f}, v={v2:.4f} "
            f"[constraint={constraint}]\n"
            f"  browser pixels: ({x1}, {y1}) -> ({x2}, {y2}), "
            f"viewport={width}x{height}, steps={steps}\n"
            f"{await self.pose_text(ctx)}"
        )


MAX_TRANSLATE_STEP_M = 0.1


class GripperTranslateTool(ToolBase):
    name = "gripper_translate"
    frames_preamble = True
    arm_param = True
    description = (
        "Translate the blue target gripper by a metric offset along the robot "
        "frame axes, in metres. Returns a screenshot. You specify the offset "
        "directly - no start point, mesh hit, or mouse drag needed - so it is the "
        "exact, camera-independent way to shift the gripper along the world axes. "
        "This is the keyed equivalent of gripper_drag's x/y/z axis locks without "
        "the cursor imprecision; the orientation is unchanged.\n\n"
        "Axes are the fixed robot X/Y/Z (the colored arrows), not the "
        "gripper-attached gizmo axes that gripper_rotate spins about, so the "
        "offsets are exact at any camera angle - trust the pose readout:\n"
        "  d_x - along robot X\n"
        "  d_y - along robot Y\n"
        "  d_z - along robot Z (vertical: +z lifts, -z lowers)\n\n"
        "Set one offset for a pure single-axis move, or several to move "
        "diagonally; omitted offsets default to 0 and at least one must be "
        f"nonzero. Each offset must be within +/-{MAX_TRANSLATE_STEP_M} m to keep "
        "moves gradual; larger requests are rejected, so call repeatedly to "
        "translate further. The returned text reports "
        "the actual pose delta after the move."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "d_x": {
                "type": "number",
                "minimum": -MAX_TRANSLATE_STEP_M,
                "maximum": MAX_TRANSLATE_STEP_M,
                "description": (
                    "Offset along robot X, metres. Must be within "
                    f"+/-{MAX_TRANSLATE_STEP_M}. Default: 0."
                ),
            },
            "d_y": {
                "type": "number",
                "minimum": -MAX_TRANSLATE_STEP_M,
                "maximum": MAX_TRANSLATE_STEP_M,
                "description": (
                    "Offset along robot Y, metres. Must be within "
                    f"+/-{MAX_TRANSLATE_STEP_M}. Default: 0."
                ),
            },
            "d_z": {
                "type": "number",
                "minimum": -MAX_TRANSLATE_STEP_M,
                "maximum": MAX_TRANSLATE_STEP_M,
                "description": (
                    "Offset along robot Z (vertical); +z lifts, -z lowers, metres. "
                    f"Must be within +/-{MAX_TRANSLATE_STEP_M}. Default: 0."
                ),
            },
        },
        "required": [],
    }

    async def __call__(self, ctx, arguments):
        await self._select_arm(ctx, arguments)
        d_x, d_y, d_z = (float(arguments.get(k) or 0.0) for k in ("d_x", "d_y", "d_z"))

        if d_x == 0.0 and d_y == 0.0 and d_z == 0.0:
            return await ctx.snap(
                "gripper_translate did not run: all offsets are zero; pass a "
                "nonzero d_x, d_y, or d_z.\n"
                f"{await self.pose_text(ctx)}"
            )

        pose = await ctx.gripper_pose()
        new_pose = (await ctx.call("translate", d_x, d_y, d_z))["gripper"]
        before_robot = pose["robot_position"]
        after_robot = new_pose["robot_position"]
        delta_robot = {axis: after_robot[axis] - before_robot[axis] for axis in ("x", "y", "z")}

        return await ctx.snap(
            f"Gripper translated by (robot, m) d_x={d_x:.4f}, d_y={d_y:.4f}, d_z={d_z:.4f}.\n"
            f"  delta (robot, m): x={fmt4(delta_robot['x'])}  "
            f"y={fmt4(delta_robot['y'])}  z={fmt4(delta_robot['z'])}\n"
            f"{await self.format_pose(ctx, new_pose)}"
        )


class GripperToggleTool(ToolBase):
    name = "gripper_toggle"
    arm_param = True
    description = (
        "Open or close the blue target gripper. Returns a screenshot.\n\n"
        "Wraps the UI 'g' key, which swaps the gripper mesh between open and "
        "closed. Pass state to drive a known result without checking the current "
        "state first:\n"
        "  state='toggle' - flip the current state (default)\n"
        "  state='open'   - ensure open  (no-op if already open)\n"
        "  state='closed' - ensure closed (no-op if already closed)\n\n"
        "The returned message and screenshot report the resulting open/closed "
        "state, read back after the swap."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "state": {
                "type": "string",
                "enum": ["toggle", "open", "closed"],
                "description": "Target state. Default: toggle.",
            },
        },
    }

    async def __call__(self, ctx, arguments):
        await self._select_arm(ctx, arguments)
        state = arguments.get("state", "toggle")

        is_open = (await ctx.gripper_pose())["gripper_open"]
        want_open = (not is_open) if state == "toggle" else (state == "open")
        if want_open == is_open:
            return await ctx.snap(
                f"Gripper already {'open' if is_open else 'closed'}; no change.\n"
                f"{await self.pose_text(ctx)}"
            )

        new_pose = (await ctx.call("toggleGripper"))["gripper"]
        verb = "toggled" if state == "toggle" else "set"
        return await ctx.snap(
            f"Gripper {verb} -> {'open' if new_pose['gripper_open'] else 'closed'}.\n"
            f"{await self.format_pose(ctx, new_pose)}")


class ResetTargetGripperTool(ToolBase):
    name = "gripper_reset"
    frames_preamble = True
    arm_param = True
    description = (
        "Reset the blue target gripper back onto the real gripper's current pose - "
        "position, orientation, and open/closed state. Returns a screenshot.\n\n"
        "Use this to undo experimental edits: freely try gripper_drag, "
        "gripper_advance_or_retreat, gripper_rotate, or gripper_toggle to plan a "
        "move, see how it looks, then reset to the real pose before committing the "
        "move you actually want. This only moves the blue target gripper - it does "
        "not move the real robot and records no waypoint."
    )
    input_schema = {"type": "object", "properties": {}, "required": []}

    _STATUS_REASONS = {"no_real_pose": "the real gripper pose has not been received yet"}

    async def __call__(self, ctx, arguments):
        await self._select_arm(ctx, arguments)
        result = await ctx.call("reset")
        status = result["status"]
        if status != "ok":
            reason = self._STATUS_REASONS.get(status, f"unexpected status {status!r}")
            return await ctx.snap(
                f"gripper_reset did not run: {reason}.\n"
                f"{await self.format_pose(ctx, result['gripper'])}"
            )

        return await ctx.snap(
            f"Reset the {self.gripper} to the real gripper's current pose "
            "(position, orientation, open/closed).\n"
            f"{await self.format_pose(ctx, result['gripper'])}"
        )


class GripperAdvanceOrRetreatTool(ToolBase):
    name = "gripper_advance_or_retreat"
    frames_preamble = True
    arm_param = True
    description = (
        "Move the blue target gripper along its approach axis with f/b. "
        "Returns a screenshot.\n\n"
        "  direction='f' - step along the approach axis (toward where "
        "the fingers point)\n"
        "  direction='b' - step opposite the approach axis (retreat / "
        "back out)\n"
        "  In robot frame the direction follows the current orientation (see "
        "Frame above): top-down, 'f' is -Z (down) and 'b' is +Z (lift / back "
        "out); on a side-grasp the same steps are horizontal.\n\n"
        "If the current orientation makes the f/b direction ambiguous, call "
        "gripper_get_pose first to see the approach axis drawn in the "
        "screenshot.\n\n"
        "Scale:\n"
        f"  One step is about {GRIPPER_STEP_M:.3f} m along the approach axis. "
        "Use 1-3 steps for small corrections and larger counts for coarse "
        "approach/retreat.\n\n"
        "The returned text reports the actual pose delta after f/b keypresses."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "direction": {
                "type": "string",
                "enum": ["f", "b"],
                "description": "f moves forward; b moves backward.",
            },
            "steps": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "description": "Number of f/b keypresses. Default: 1.",
            },
        },
        "required": ["direction"],
    }

    KEY_LABELS = {"f": "forward", "b": "backward"}

    async def __call__(self, ctx, arguments):
        await self._select_arm(ctx, arguments)
        key = arguments["direction"]

        pose = await ctx.gripper_pose()
        steps = int(arguments.get("steps", 1))
        new_pose = (await ctx.call("advance", steps if key == "f" else -steps))["gripper"]
        before_robot = pose["robot_position"]
        after_robot = new_pose["robot_position"]
        delta_robot = {axis: after_robot[axis] - before_robot[axis] for axis in ("x", "y", "z")}
        return await ctx.snap(
            f"Gripper moved {self.KEY_LABELS[key]} with '{key}' x{steps} "
            f"(nominal approach-axis distance {steps * GRIPPER_STEP_M:.4f} m).\n"
            f"  delta (robot, m): x={fmt4(delta_robot['x'])}  "
            f"y={fmt4(delta_robot['y'])}  z={fmt4(delta_robot['z'])}\n"
            f"{await self.format_pose(ctx, new_pose)}"
        )


class GripperRotateTool(ToolBase):
    name = "gripper_rotate"
    frames_preamble = True
    arm_param = True
    description = (
        "Rotate the blue target gripper about one gizmo axis (x, y, or z) by a "
        "signed angle in degrees. Returns a screenshot.\n\n"
        "IMPORTANT: call `gripper_show_rotation_gizmo` before EVERY rotation and "
        "pick the axis and direction from the gizmo you see - do not guess. The "
        "x/y/z are the gripper's own gizmo axes, not the world/robot X/Y/Z (the "
        "colored arrows); they line up only at the top-down pose and move with the "
        "gripper after any tilt, so the same letter points a different way once it "
        "is tilted.\n\n"
        "The gripper has three mutually perpendicular gizmo axes: the approach axis "
        "(where the fingers reach), the opening axis (the finger line), and a "
        "third axis perpendicular to both. Each rotation spins about one, leaving "
        "it fixed and re-aiming the other two (the arrow on each ring shows the + "
        "direction):\n"
        "* x - about the third axis; both the approach and opening axes re-aim.\n"
        "* y - about the opening axis; opening stays fixed, approach re-aims.\n"
        "* z - about the approach axis; approach stays fixed, opening re-aims "
        "(twist in place).\n\n"
        "The returned text reports the resulting approach and opening axes; the "
        "rotation is in place (position unchanged). "
        "If you are unsure about the axis or direction, rotate in small amount and observe "
        "its outcome."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "axis": {
                "type": "string",
                "enum": ["x", "y", "z"],
                "description": (
                    "Gizmo rotation axis to spin about: x, y, or z. Choose it from "
                    "gripper_show_rotation_gizmo."
                ),
            },
            "angle": {
                "type": "number",
                "minimum": -90,
                "maximum": 90,
                "description": (
                    "Signed rotation angle in degrees, within "
                    "[-90, 90]. Prefer small steps (e.g. 10-45 degrees) so you can "
                    "observe the intermediate result and adjust before overshooting."
                ),
            },
        },
        "required": ["axis", "angle"],
    }

    async def __call__(self, ctx, arguments):
        await self._select_arm(ctx, arguments)
        axis = arguments["axis"]
        angle = float(arguments["angle"])
        new_pose = (await ctx.call("rotate", axis, angle))["gripper"]
        return await ctx.snap(
            f"Rotated {fmt4(angle)} deg about gizmo '{axis}' axis (right-handed). "
            "Rotation is in place (position unchanged).\n"
            f"{await self.format_pose(ctx, new_pose)}"
        )


class GetGripperPoseTool(ToolBase):
    name = "gripper_get_pose"
    frames_preamble = True
    arm_param = True
    description = (
        "Return the current blue target gripper pose and visually show its "
        "approach axis in the screenshot. Besides giving a readout of the "
        "exact current gripper pose, the drawn approach axis makes this the "
        "best orientation check before using gripper_advance_or_retreat: it "
        "shows which way f/b will step. Returns a screenshot.\n\n"
        "Reports the center of the fingertips of the blue target gripper in "
        "robot metres, its orientation as approach and opening robot-frame unit "
        "vectors, and open/closed state. "
        "The visual axis guide is shown without moving the gripper."
    )
    input_schema = {"type": "object", "properties": {}, "required": []}

    async def __call__(self, ctx, arguments):
        await self._select_arm(ctx, arguments)
        pose = (await ctx.call("guides", "axis"))["gripper"]
        try:
            return await ctx.snap(
                "Showing current gripper approach axis visually; use this "
                "to choose gripper_advance_or_retreat f/b direction and interpret gripper_drag. "
                "No movement.\n"
                f"{await self.format_pose(ctx, pose)}"
            )
        finally:
            await ctx.call("guides", "none")


class ShowGripperRotationGizmoTool(ToolBase):
    name = "gripper_show_rotation_gizmo"
    frames_preamble = True
    description = (
        "Show the gripper's rotation gizmo (the three rotation rings) together "
        "with its approach axis in the screenshot. Always call this first to "
        "choose the gripper_rotate axis: it gives an intuitive visual "
        "understanding of the rotation axes (the x/y/z gizmo rings). The arrow on "
        "each rotation ring points to the + direction of that rotation. Returns a "
        "screenshot.\n\n"
        "Reports the center of the fingertips of the blue target gripper in "
        "robot metres, its orientation as approach and opening robot-frame unit "
        "vectors, and open/closed state."
    )
    input_schema = {"type": "object", "properties": {}, "required": []}

    async def __call__(self, ctx, arguments):
        await self._select_arm(ctx, arguments)
        pose = (await ctx.call("guides", "gizmo"))["gripper"]
        try:
            return await ctx.snap(
                "Showing current gripper rotation gizmo and approach axis "
                "visually; use this to choose gripper_rotate axis and direction. "
                "No movement.\n"
                f"{await self.format_pose(ctx, pose)}"
            )
        finally:
            await ctx.call("guides", "none")


MAX_ORBIT_STEP_DEG = 45.0
MIN_ELEVATION_DEG = 0.0
MAX_ELEVATION_DEG = 85.0


class CameraOrbitViaKeyTool(ToolBase):
    name = "camera_orbit_via_key"
    description = (
        "Orbit the camera around the scene target by an angular delta, in degrees. "
        "Returns a screenshot. You specify the two orbit angles directly - no start "
        "point or open-space click needed - so it is the simplest way to reframe.\n\n"
        "Both deltas are RELATIVE to the current pose (read camera_get_pose for the "
        "absolute angles). The distance and look-at target are unchanged - this only "
        "rotates the viewpoint.\n\n"
        "Directions:\n"
        "  +d_azimuth swings the camera counterclockwise seen from above while "
        "-d_azimuth swings it clockwise. At the default view, azimuth=90 with "
        "Y=0 looking towards -X. +d_azimuth will swing the camera to +Y and "
        "gradually move towards looking at -Y at azimuth=180.\n"
        "  +d_elevation raises the camera toward a top-down view (elevation -> 90); "
        "-d_elevation lowers it toward a front or side horizontal view "
        "(elevation -> 0).\n\n"
        f"Each delta must be within +/-{MAX_ORBIT_STEP_DEG:.0f} deg to keep camera "
        "moves gradual; larger requests are rejected, so call repeatedly to orbit "
        f"further. Elevation is clamped to [{MIN_ELEVATION_DEG:.0f}, "
        f"{MAX_ELEVATION_DEG:.0f}] (the horizon is allowed; kept just off exact "
        "top-down, which is degenerate); azimuth wraps at 360. Returns the updated "
        "camera pose."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "d_azimuth": {
                "type": "number",
                "minimum": -MAX_ORBIT_STEP_DEG,
                "maximum": MAX_ORBIT_STEP_DEG,
                "description": (
                    "Change in horizontal orbit angle, degrees. + orbits the camera "
                    f"left, - orbits right. Must be within +/-{MAX_ORBIT_STEP_DEG:.0f}."
                ),
            },
            "d_elevation": {
                "type": "number",
                "minimum": -MAX_ORBIT_STEP_DEG,
                "maximum": MAX_ORBIT_STEP_DEG,
                "description": (
                    "Change in vertical angle, degrees. + raises the camera toward "
                    "top-down (90), - lowers it toward the horizon (0). Must be "
                    f"within +/-{MAX_ORBIT_STEP_DEG:.0f}."
                ),
            },
        },
        "required": ["d_azimuth", "d_elevation"],
    }

    async def __call__(self, ctx, arguments):
        d_az = float(arguments["d_azimuth"])
        d_el = float(arguments["d_elevation"])
        result = await ctx.call("cameraOrbit", d_az, d_el)
        note = "" if result["elevation_set"] == result["elevation_wanted"] else (
            f"\n  note: elevation clamped to {result['elevation_set']:.4f} "
            f"(valid range [{MIN_ELEVATION_DEG:.0f}, {MAX_ELEVATION_DEG:.0f}])")
        return await ctx.snap(
            f"Orbited camera by d_azimuth={d_az:.4f}, d_elevation={d_el:.4f} deg{note}\n"
            f"{ctx.format_camera_pose(result['camera'])}")


MAX_PAN_STEP_M = 0.2


class CameraPanViaKeyTool(ToolBase):
    name = "camera_pan_via_key"
    description = (
        "Pan the camera (shift its look-at point) by a metric offset, in robot-frame "
        "metres. Returns a screenshot. You specify the shift directly - no start "
        "point or drag needed - so it is the simplest way to re-center the view "
        "without rotating.\n\n"
        "The viewing angle (azimuth/elevation) and distance are unchanged - the "
        "camera and its target slide together by the same amount, so only the "
        "framing moves.\n\n"
        "Directions (relative to the current view):\n"
        "  +d_right pans right: the look-at point slides toward screen-right, so you "
        "see more of what was off the right edge (scene content appears to slide "
        "left). -d_right pans left.\n"
        "  +d_up pans up along world-Z (robot +z): you see more above (scene content "
        "appears to slide down). -d_up pans down.\n\n"
        f"Each offset must be within +/-{MAX_PAN_STEP_M} m to keep moves gradual; "
        "larger requests are rejected, so call repeatedly to pan further. Returns "
        "the updated camera pose, whose target shows the new framing center."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "d_right": {
                "type": "number",
                "minimum": -MAX_PAN_STEP_M,
                "maximum": MAX_PAN_STEP_M,
                "description": (
                    "Horizontal pan along camera-right, robot metres. + pans right "
                    "(reveals scene to the right), - pans left. Must be within "
                    f"+/-{MAX_PAN_STEP_M}."
                ),
            },
            "d_up": {
                "type": "number",
                "minimum": -MAX_PAN_STEP_M,
                "maximum": MAX_PAN_STEP_M,
                "description": (
                    "Vertical pan along world-Z (robot +z), metres. + pans up "
                    "(reveals scene above), - pans down. Must be within "
                    f"+/-{MAX_PAN_STEP_M}."
                ),
            },
        },
        "required": ["d_right", "d_up"],
    }

    async def __call__(self, ctx, arguments):
        d_right = float(arguments["d_right"])
        d_up = float(arguments["d_up"])

        ui_units_per_m = 10.0
        result = await ctx.call("cameraPan", d_right * ui_units_per_m, d_up * ui_units_per_m)
        return await ctx.snap(f"Panned camera by d_right={d_right:.4f}, d_up={d_up:.4f} m\n"
                              f"{ctx.format_camera_pose(result['camera'])}")


class CameraZoomTool(ToolBase):
    name = "camera_zoom"
    description = (
        "Zoom the camera toward normalized screenshot coordinates. Returns a screenshot.\n\n"
        "The browser zooms toward the point under the mouse. u/v should be "
        "inside the point-cloud view.\n\n"
        "Coordinate convention:\n"
        "  u = fraction across the screenshot, 0.0 left -> 1.0 right\n"
        "  v = fraction down the screenshot, 0.0 top -> 1.0 bottom\n\n"
        "Positive steps zoom in; negative steps zoom out. Internally this sends "
        "scroll-wheel events to the browser. Each step changes the "
        "camera-to-target distance by a constant ~0.03 m, regardless of current "
        "zoom level or direction (so steps ~= |delta_distance| / 0.03). "
        "Off-center zoom centers also shift composition toward that "
        "point; use this deliberately to reframe while zooming. Returns the "
        "updated camera pose."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "u": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Zoom center fraction across the screenshot.",
            },
            "v": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Zoom center fraction down the screenshot.",
            },
            "steps": {
                "type": "integer",
                "minimum": -20,
                "maximum": 20,
                "description": "Scroll steps. Positive zooms in; negative zooms out. Default: 1.",
            },
        },
        "required": ["u", "v"],
    }

    async def __call__(self, ctx, arguments):
        u = float(arguments["u"])
        v = float(arguments["v"])
        steps = int(arguments.get("steps", 1))
        x, y, width, height = await ctx.uv_to_xy(u, v)

        await ctx.page.mouse.move(x, y)
        await ctx.show_cursor(x, y)
        result = await ctx.call("cameraZoom", x, y, steps)

        if steps == 0:
            action = "no zoom"
        else:
            action = "zoom in" if steps > 0 else "zoom out"
        return await ctx.snap(
            f"Camera zoom {steps} steps ({action}); zoom center u={u:.4f}, v={v:.4f}\n"
            f"  browser pixel: x={x}, y={y}, viewport={width}x{height}\n"
            f"{ctx.format_camera_pose(result['camera'])}"
        )


class CameraResetTool(ToolBase):
    name = "camera_reset"
    description = (
        "Restore the initial camera view. Returns a screenshot and the updated "
        "camera pose.\n\n"
        "Strategy: the default view is framed so most operations are viable from "
        "it, so prefer working there and return to it with this tool after "
        "exploring. Reach for camera_orbit_via_key / camera_pan_via_key only "
        "when the default view is insufficient - e.g. to resolve an occlusion, or "
        "to judge the gripper-to-object relationship (depth, alignment, clearance) "
        "from another angle - then reset here to re-establish a known frame."
    )
    input_schema = {"type": "object", "properties": {}, "required": []}

    async def __call__(self, ctx, arguments):
        result = await ctx.call("cameraReset")
        return await ctx.snap("Camera reset to initial view\n"
                              f"{ctx.format_camera_pose(result['camera'])}")


class CameraOrbitViaDragTool(ToolBase):
    name = "camera_orbit_via_drag"
    description = (
        "Orbit the camera by left-dragging the point cloud, from (u1, v1) to "
        "(u2, v2). Returns a screenshot and the updated camera pose.\n\n"
        "The drag must START ON EMPTY SPACE. A left-drag that begins on the blue "
        "target gripper moves the GRIPPER instead of the camera, so this tool "
        "refuses that start point rather than silently displacing the waypoint.\n\n"
        "Motion model: the view swings ~0.29 deg of azimuth per CSS pixel of "
        "horizontal movement and the same in elevation vertically; dragging right "
        "swings the scene left. camera_orbit_via_key is the tool to reach for when "
        "you know the angle you want - use this one to nudge the view by eye."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "u1": {
                "type": "number", "minimum": 0.0, "maximum": 1.0,
                "description": "Start fraction across the screenshot, 0.0 left to 1.0 right.",
            },
            "v1": {
                "type": "number", "minimum": 0.0, "maximum": 1.0,
                "description": "Start fraction down the screenshot, 0.0 top to 1.0 bottom.",
            },
            "u2": {
                "type": "number", "minimum": 0.0, "maximum": 1.0,
                "description": "End fraction across the screenshot, 0.0 left to 1.0 right.",
            },
            "v2": {
                "type": "number", "minimum": 0.0, "maximum": 1.0,
                "description": "End fraction down the screenshot, 0.0 top to 1.0 bottom.",
            },
            "steps": {
                "type": "integer", "minimum": 1, "maximum": 100,
                "description": "Mouse move interpolation steps. Default: 25.",
            },
        },
        "required": ["u1", "v1", "u2", "v2"],
    }

    async def __call__(self, ctx, arguments):
        u1, v1 = float(arguments["u1"]), float(arguments["v1"])
        u2, v2 = float(arguments["u2"]), float(arguments["v2"])
        steps = int(arguments.get("steps", 25))
        x1, y1, _, _ = await ctx.uv_to_xy(u1, v1)
        x2, y2, _, _ = await ctx.uv_to_xy(u2, v2)

        if await self._start_hits_gripper(ctx, x1, y1):
            await ctx.show_cursor(x1, y1)
            return await ctx.snap(
                f"camera_orbit_via_drag did not run: u1={u1:.4f}, v1={v1:.4f} is on the "
                "target gripper, where a left-drag would move the gripper instead of the "
                "camera. Start the drag on empty space."
            )

        await ctx.perform_drag(x1, y1, x2, y2, steps=steps)
        return await ctx.snap(
            f"Orbited camera by dragging ({u1:.4f}, {v1:.4f}) -> ({u2:.4f}, {v2:.4f})\n"
            f"{await ctx.camera_pose_text()}"
        )


class CameraPanViaDragTool(ToolBase):
    name = "camera_pan_via_drag"
    description = (
        "Pan the camera by right-dragging from (u1, v1) to (u2, v2): the camera and "
        "its target slide together, so the view shifts without rotating. Returns a "
        "screenshot and the updated camera pose.\n\n"
        "Right-drag pans wherever it starts - the gripper does not intercept it - so "
        "unlike camera_orbit_via_drag any start point works. Dragging right moves the "
        "scene right (the camera moves left). Use it to re-centre a workspace that "
        "orbiting or zooming has pushed off-screen."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "u1": {
                "type": "number", "minimum": 0.0, "maximum": 1.0,
                "description": "Start fraction across the screenshot, 0.0 left to 1.0 right.",
            },
            "v1": {
                "type": "number", "minimum": 0.0, "maximum": 1.0,
                "description": "Start fraction down the screenshot, 0.0 top to 1.0 bottom.",
            },
            "u2": {
                "type": "number", "minimum": 0.0, "maximum": 1.0,
                "description": "End fraction across the screenshot, 0.0 left to 1.0 right.",
            },
            "v2": {
                "type": "number", "minimum": 0.0, "maximum": 1.0,
                "description": "End fraction down the screenshot, 0.0 top to 1.0 bottom.",
            },
            "steps": {
                "type": "integer", "minimum": 1, "maximum": 100,
                "description": "Mouse move interpolation steps. Default: 25.",
            },
        },
        "required": ["u1", "v1", "u2", "v2"],
    }

    async def __call__(self, ctx, arguments):
        u1, v1 = float(arguments["u1"]), float(arguments["v1"])
        u2, v2 = float(arguments["u2"]), float(arguments["v2"])
        steps = int(arguments.get("steps", 25))
        x1, y1, _, _ = await ctx.uv_to_xy(u1, v1)
        x2, y2, _, _ = await ctx.uv_to_xy(u2, v2)
        await ctx.perform_drag(x1, y1, x2, y2, button="right", steps=steps)
        return await ctx.snap(
            f"Panned camera by right-dragging ({u1:.4f}, {v1:.4f}) -> ({u2:.4f}, {v2:.4f})\n"
            f"{await ctx.camera_pose_text()}"
        )


class CameraSetPoseTool(ToolBase):
    name = "camera_set_pose"
    description = (
        "Place the camera at an ABSOLUTE pose: orbit angles, distance, and the "
        "target it looks at. Returns a screenshot and the resulting pose.\n\n"
        "Every field is in the same convention camera_get_pose reports, so a pose "
        "read from there can be handed straight back to return to a known view: "
        "azimuth is the horizontal orbit angle in degrees, elevation is 0=horizontal "
        f"and 90=top-down (clamped to [{MIN_ELEVATION_DEG:.0f}, {MAX_ELEVATION_DEG:.0f}] "
        "so the view never reaches the degenerate pole), distance is camera-to-target "
        "in metres, and the target is a robot-frame point in metres.\n\n"
        "This is the tool for restoring a framing you noted earlier, or for jumping "
        "to a named viewpoint in one call instead of accumulating orbit/pan/zoom "
        "deltas. camera_reset returns to the pack's default view."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "azimuth": {
                "type": "number",
                "description": "Horizontal orbit angle, degrees (wraps at 360).",
            },
            "elevation": {
                "type": "number",
                "description": (
                    f"Vertical angle, degrees. 0=horizontal, 90=top-down; clamped to "
                    f"[{MIN_ELEVATION_DEG:.0f}, {MAX_ELEVATION_DEG:.0f}]."
                ),
            },
            "distance": {
                "type": "number",
                "exclusiveMinimum": 0.0,
                "description": "Camera-to-target separation, metres.",
            },
            "target_x": {"type": "number", "description": "Target x, robot frame, metres."},
            "target_y": {"type": "number", "description": "Target y, robot frame, metres."},
            "target_z": {"type": "number", "description": "Target z, robot frame, metres."},
        },
        "required": ["azimuth", "elevation", "distance",
                     "target_x", "target_y", "target_z"],
    }

    async def __call__(self, ctx, arguments):
        az = float(arguments["azimuth"]) % 360.0
        wanted = float(arguments["elevation"])
        dist = float(arguments["distance"])
        el = min(max(wanted, MIN_ELEVATION_DEG), MAX_ELEVATION_DEG)
        note = "" if el == wanted else (
            f"\n  note: elevation clamped to {el:.4f} "
            f"(valid range [{MIN_ELEVATION_DEG:.0f}, {MAX_ELEVATION_DEG:.0f}])")
        t = ctx.robot_to_ui(float(arguments["target_x"]), float(arguments["target_y"]),
                            float(arguments["target_z"]))
        result = await ctx.call("cameraSet", az, el, dist * 10.0, t)
        return await ctx.snap(
            f"Camera set to azimuth={az:.4f}, elevation={el:.4f}, distance={dist:.4f} m{note}\n"
            f"{ctx.format_camera_pose(result['camera'])}")


class GetCameraPoseTool(ToolBase):
    name = "camera_get_pose"
    description = (
        "Return the current camera pose without taking an action.\n\n"
        "The camera orbits a target point, reported in the robot frame (metres) "
        "like every other pose: target (robot, m) is the world point the camera "
        "looks at, and distance is the camera-to-target separation in metres. "
        "azimuth/elevation are the viewing direction onto that target: azimuth is "
        "the horizontal orbit angle in degrees; elevation is 0=horizontal (level "
        f"with the target) and 90=top-down, but orbiting keeps it within "
        f"[{MIN_ELEVATION_DEG:.0f}, {MAX_ELEVATION_DEG:.0f}] so it never reaches the "
        "degenerate top-down pole (where azimuth would collapse to 0). The target is "
        "reported because pan and "
        "off-center zoom shift the scene center. Use these values to reason about "
        "the current framing before adjusting it with other camera tools."
    )
    input_schema = {"type": "object", "properties": {}, "required": []}

    async def __call__(self, ctx, arguments):
        return [types.TextContent(type="text", text=await ctx.camera_pose_text())]


async def _wait_for_waypoint(ctx: UiKernel, result: dict, hover_selector: str) -> str:
    prev = result["waypointDone"]
    await ctx.page.locator(hover_selector).hover()
    note = ""
    stalled_polls, last_steps_used = 0, -1
    while True:
        try:
            await ctx.page.wait_for_function(
                "prev => (window.__waypointDoneCount || 0) > prev",
                arg=prev,
                timeout=15_000,
            )
            break
        except PlaywrightTimeoutError:
            used = (await fetch_sim_json("env.json") or {}).get("sim_steps_used", -1)
            stalled_polls = stalled_polls + 1 if used == last_steps_used else 0
            last_steps_used = used
            if stalled_polls >= 4:
                raise RuntimeError(f"execute_waypoint did not complete: the simulator has not "
                                   f"stepped for 60 s at step {used}. The target is unchanged; "
                                   "call execute_waypoint again.")

    env = await fetch_sim_json("env.json") or {}
    if env.get("episode_over"):
        note += "\n\n" + EPISODE_OVER
    return note


class ExecuteTool(ToolBase):
    name = "execute_waypoint"
    arm_param = True
    description = (
        "Execute the current gripper target and record one waypoint. Returns a screenshot.\n\n"
        "This asks the robot controller to move the real gripper to the current "
        "target pose: position, orientation, and final open/closed gripper state. "
        "Use this after meaningful target-setting operations such as "
        "gripper_teleport_via_click, gripper_drag, gripper_advance_or_retreat, gripper_rotate, "
        "or gripper_toggle. Execute a gripper_toggle on its own. Do not combine it "
        "with a pose change in the same waypoint, since moving and opening/closing "
        "the gripper at once is unreliable.\n\n"
        "Execution uses the final target as the next base state for later operations. "
        "The controller interpolates between the current pose and target pose; it "
        "does not replay the exact UI trajectory used to place the target. For "
        "example, if you toggled the gripper twice before executing, it only tries "
        "to match the final open/closed state, without intermediate toggles.\n\n"
        "If the episode ends on this waypoint (goal reached, or the step budget is "
        "spent) the result says EPISODE OVER. Stop then: no further tools."
    )
    input_schema = {"type": "object", "properties": {}, "required": []}

    async def __call__(self, ctx, arguments):
        await self._select_arm(ctx, arguments)
        result = await ctx.call("recordWaypoint")
        note = await _wait_for_waypoint(ctx, result, "#btn-record")

        return await ctx.snap(
            "Executed current target and recorded waypoint.\n"
            f"{await self.pose_text(ctx)}{note}"
        )


class GripperHomeTool(ToolBase):
    name = "gripper_home"
    frames_preamble = True
    arm_param = True
    description = (
        "gripper_home sets the target gripper to the arm's start pose; execute it like "
        "any other waypoint. When the target is green the controller drives a straight "
        "line there. When it is red no straight path exists from the current posture, "
        "but executing it still works: the controller moves the joints directly to the "
        "start posture, which can sweep the arm through whatever is in the way, so move "
        "clear of objects first. Use it when every target you try is red; it always "
        "succeeds. The target jaw command stays unchanged. Returns a screenshot and "
        "target gripper pose readout."
    )
    input_schema = {"type": "object", "properties": {}, "required": []}

    async def __call__(self, ctx, arguments):
        await ctx.home(arguments.get("arm"))
        return await ctx.snap(
            "Set target gripper to the arm's start pose. Nothing has executed.\n"
            f"{await self.pose_text(ctx)}"
        )


class EndEpisodeTool(ToolBase):
    name = "end_episode"
    description = (
        "Save/end the current episode and reset or advance the task. Returns a screenshot.\n\n"
        "Call this only after the final waypoint for the episode has been executed."
    )
    input_schema = {"type": "object", "properties": {}, "required": []}

    async def __call__(self, ctx, arguments):
        snap_before_reset = await ctx.snap("Episode ended and saved.")
        await ctx.page.evaluate("() => window.roboarcade.endEpisode('agent')")
        return snap_before_reset


class CameraViewsTool(ToolBase):
    name = "camera_views"
    description = (
        "Show live renderings from EVERY selectable camera at once, overlaid "
        "large over the 3-D point-cloud panel (returned in the screenshot; the "
        "overlay is removed afterwards). Cameras shown:\n"
        "  - agentview: the pack's default third_person camera, exposed by role\n"
        "  - ring0..ring2: the depth cameras the point cloud is FUSED from, "
        "spaced 120 deg apart around the workspace looking inward. View k looks "
        "FROM azimuth 90+k*120 deg in the robot frame: ring0 from +y (screen "
        "right in the default 3-D view), ring1 from behind-left, ring2 from "
        "front-right. They sit off the world axes on purpose, so a flat face "
        "is never seen edge-on from all three.\n"
        "  - birdseye: a straight-down overhead view of the whole workspace "
        "(top of its image = the robot's side, image right = +y) - the best "
        "vantage for xy layout and alignment, with no height cues.\n"
        "  - zoomed_birdseye: the same straight-down view from half the "
        "height - twice the detail on the middle of the workspace (the edges "
        "fall outside its frame). Pair with camera_raw for insertion-grade "
        "close reads.\n\n"
        "  - eyelevel: a near-horizontal view across the table from the "
        "agentview's side, 10 deg above the surface. The only view that "
        "shows HEIGHT directly - every other camera looks down, where a "
        "height difference reads only as a size change. Use it to judge "
        "how far above a surface the gripper or a held object really is.\n"
        "  - farside: the second ground-level station, a quarter turn round "
        "from the eyelevel and much closer in: it sits just past the far (+y) "
        "edge of the workspace, a hand's width above the surface, looking back "
        "across it. Height reads here too, and whatever stands at the far edge "
        "is in the near field at a useful size.\n\n"
        "Use this to survey the scene from every angle when something is "
        "occluded or ambiguous, then keep the most useful vantage on screen "
        "with camera_feed_select. Frames refresh after every execute_waypoint."
    )
    input_schema = {"type": "object", "properties": {}, "required": []}

    async def __call__(self, ctx, arguments):
        await ctx.page.evaluate("() => window.__camGallery(true)")
        try:
            return await ctx.snap(
                "Live views from every selectable camera, tiled over the "
                "point-cloud panel and labeled (ring views by the azimuth they "
                "look FROM; CURRENT marks the active 3rd-person feed). Keep one "
                "on screen with camera_feed_select(view=<name>)."
            )
        finally:
            await ctx.page.evaluate("() => window.__camGallery(false)")


class CameraRawTool(ToolBase):
    name = "camera_raw"
    description = (
        "Return the RAW rendered RGB image from ONE camera, exactly as the sim "
        "renders it: native camera resolution, no UI chrome, tiles, overlays, or "
        "browser rescaling. Selectable cameras:\n"
        "  - 'agentview': the pack's third_person camera, exposed by role\n"
        "  - 'wrist': the gripper's eye-in-hand camera (dual-arm tasks also serve "
        "'wrist_<arm>' for the extra arms)\n"
        "  - 'ring0'..'ring2': the point-cloud ring, view k looking inward from "
        "azimuth 90+k*120 deg\n"
        "  - 'birdseye': straight-down overhead (top of image = the robot's side, "
        "image right = +y)\n"
        "  - 'zoomed_birdseye': the same overhead from half the height - 2x the "
        "pixels per cm on the workspace center, the sharpest view for fine "
        "alignment and insertions\n"
        "  - 'eyelevel': near-horizontal across the table from the agentview's "
        "side - one of the two views that read HEIGHT directly (the others look "
        "down, where height reads only as a size change)\n"
        "  - 'farside': a hand's width above the surface just past the far (+y) "
        "edge of the workspace, looking back across it - the other height read, "
        "and the closest view of whatever stands at that edge\n\n"
        "Resolution: every camera renders at 448x448.\n\n"
        "Complements the composited views: use camera_views to survey every angle "
        "at once, then this for a full-fidelity close read of the one that matters "
        "(fine alignment, small/partially occluded objects, text or texture the "
        "screenshot tiles blur). The frame is from the last completed observation - "
        "it refreshes after every execute_waypoint, not live. An unknown name "
        "returns the available list."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "view": {
                "type": "string",
                "description": ("Camera to fetch: 'agentview', 'wrist', "
                                "'ring0'..'ring2', 'birdseye', 'zoomed_birdseye', "
                                "'eyelevel', or 'farside'."),
            },
        },
        "required": ["view"],
    }

    async def __call__(self, ctx, arguments):
        view = arguments.get("view", "")
        if not isinstance(view, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,128}", view) is None:
            return [types.TextContent(type="text", text="Invalid camera name: use only letters, digits, underscores or hyphens.")]
        env = await fetch_sim_json("env.json") or {}
        known = [name for name in (env.get("camera_views") or [])
                 if isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", name)]
        if view not in known:
            return [types.TextContent(type="text", text=f"Unknown camera. Selectable cameras: {', '.join(known)}.")]
        try:
            async with httpx.AsyncClient(timeout=2.0, follow_redirects=False) as client:
                async with client.stream("GET", f"{SIM_HTTP_BASE}/cam/{view}.jpg") as response:
                    response.raise_for_status()
                    if response.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "image/jpeg":
                        raise ValueError("Camera response is not image/jpeg")
                    chunks, size = [], 0
                    async for chunk in response.aiter_bytes(chunk_size=65536):
                        size += len(chunk)
                        if size > 8 * 1024 * 1024:
                            raise ValueError("Camera image exceeds 8 MiB")
                        chunks.append(chunk)
                    body = b"".join(chunks)
        except httpx.HTTPError:
            return [types.TextContent(
                type="text",
                text=f"No frame for {view!r}: camera request failed or no observation is available yet.")]
        except ValueError:
            return [types.TextContent(type="text", text="Camera server returned an invalid or oversized JPEG response.")]
        if len(body) > 8 * 1024 * 1024 or not body.startswith(b"\xff\xd8\xff"):
            return [types.TextContent(type="text", text="Camera server returned an invalid JPEG.")]
        import io
        from PIL import Image
        try:
            with Image.open(io.BytesIO(body)) as frame:
                if frame.format != "JPEG" or not (0 < frame.width <= 4096 and 0 < frame.height <= 4096):
                    raise ValueError("Unexpected camera image dimensions or format")
                frame.load()
        except (OSError, ValueError, Image.DecompressionBombError):
            return [types.TextContent(type="text", text="Camera server returned an invalid JPEG.")]
        return [
            types.TextContent(
                type="text",
                text=(f"Raw {view} camera image (native sim render, from the last "
                      f"completed observation).")),
            ctx._image_content(body),
        ]


class CameraFeedSelectTool(ToolBase):
    name = "camera_feed_select"
    description = (
        "Switch the 3rd-person camera feed (top-left panel) to any selectable "
        "camera: 'agentview' (the pack's third_person role), 'ring0'..'ring2' (the point-cloud "
        "ring, view k looking inward from azimuth 90+k*120 deg), 'birdseye' "
        "(straight-down overhead), 'zoomed_birdseye' (the same view from "
        "half the height - 2x detail on the workspace center), or 'eyelevel' "
        "(near-horizontal across the table, the one view that shows height "
        "directly). The choice is "
        "sticky - the panel keeps "
        "streaming that camera, refreshed after every execute_waypoint, until "
        "changed - and clicks on the feed (gripper_teleport_via_click, hover) "
        "project through the SELECTED camera, so click-to-place keeps working. "
        "Survey the options first with camera_views; pass 'agentview' to "
        "restore the pack's third_person camera by role. Returns a screenshot "
        "showing the new feed."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "view": {
                "type": "string",
                "description": ("Camera to show in the 3rd-person panel: "
                                "'agentview', 'ring0'..'ring2', 'birdseye', or 'eyelevel'."),
            },
        },
        "required": ["view"],
    }

    async def __call__(self, ctx, arguments):
        view = str(arguments.get("view", ""))
        r = await ctx.call("selectFeed", view)
        if not r.get("ok"):
            return [types.TextContent(
                type="text",
                text=(f"Unknown view {view!r}. Selectable cameras: "
                      f"{', '.join(r.get('known', []))}."))]
        if view == "agentview":
            note = "the pack's third_person camera"
        elif view.startswith("ring"):
            note = f"the point-cloud ring camera looking inward from azimuth {90 + int(view[4:]) * 120} deg"
        else:
            note = ("the straight-down overhead camera (top of image = the robot's side, "
                    "image right = +y)"
                    + ("; half-height, 2x detail on the workspace center"
                       if view.startswith("zoomed") else ""))
        return await ctx.snap(
            f"3rd-person feed now shows {view} - {note}. The selection is "
            "sticky and clicks on the feed project through this camera. "
            "Pass view='agentview' to restore the default."
        )


TOOLS: dict[str, ToolBase] = {
    t.name: t
    for t in (
        ScreenshotTool(),
        HoverTool(),
        GripperTeleportViaClickTool(),
        GripperDragTool(),
        GripperTranslateTool(),
        GripperToggleTool(),
        ResetTargetGripperTool(),
        GripperAdvanceOrRetreatTool(),
        GripperRotateTool(),
        GetGripperPoseTool(),
        ShowGripperRotationGizmoTool(),
        CameraOrbitViaKeyTool(),
        CameraPanViaKeyTool(),
        CameraZoomTool(),
        CameraResetTool(),
        CameraOrbitViaDragTool(),
        CameraPanViaDragTool(),
        CameraSetPoseTool(),
        GetCameraPoseTool(),
        CameraViewsTool(),
        CameraFeedSelectTool(),
        CameraRawTool(),
        ExecuteTool(),
        GripperHomeTool(),
        EndEpisodeTool(),
    )
}
