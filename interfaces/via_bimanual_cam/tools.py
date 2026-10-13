"""Via-cam's 25 tools for dual-arm tasks.

Two target grippers coloured by arm (BLUE = right, ORANGE = left), waypoint
tools taking arm='right'|'left' (the page maps sides to env arm names), and
arm-aware readouts. Everything else is via-cam's implementation; only the
wording changes here.
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("via_cam_tools", Path(__file__).with_name("via_cam.py"))
via = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(via)

ARM_COLOR = {"right": "blue", "left": "orange"}
ARM_PROP = {
    "type": "string",
    "enum": ["right", "left"],
    "description": (
        "Arm this tool commands: 'right' is the BLUE target gripper, "
        "'left' is the ORANGE one (sides as seen in the default view). "
        "The selection is sticky - it persists for later tools until "
        "changed. Default: the currently selected arm (the fully "
        "opaque gripper)."
    ),
}


class Arm:
    preamble = via.FRAMES_PREAMBLE.replace(
        "Robot Z position",
        "  arms - two arms, each with its own target gripper: 'right' = the BLUE "
        "gripper, 'left' = the ORANGE one (sides as seen in the default view). "
        "Tools with an arm parameter command the named arm; the selection is "
        "sticky and the selected gripper renders fully opaque.\n"
        "Robot Z position")
    gripper = "target gripper"

    def arm_prop(self, arms):
        return ARM_PROP

    async def format_pose(self, ctx, pose):
        arm_info = await ctx.page.evaluate("() => window.__armInfo ? window.__armInfo() : null")
        if not arm_info:
            return via.format_gripper_pose(pose, self.gripper)
        side = arm_info["selected"]
        return via.format_gripper_pose(pose, self.gripper,
                                       f"Current {side.upper()} arm ({ARM_COLOR[side]}) target gripper pose")

    async def contact_text(self, ctx, x, y):
        return await ctx.contact_point_text(x, y, "selected target gripper")


DESCRIPTION_EDITS = {
    "screenshot": (
        "Capture a screenshot of the robot interface.\n\n"
        "The window shows:\n"
        "  - Left sidebar: camera feeds, rendered from the real scene; they "
        "refresh only after execute_waypoint moves a real gripper.\n"
        "      - 3rd-person view: a fixed camera looking at the table and both "
        "arms. Best for analyzing the full scene layout (objects relative to "
        "each other and the grippers).\n"
        "      - One wrist view PER ARM, each a camera on that arm's gripper "
        "looking down between its fingers, titled by side and color (RIGHT "
        "wrist = blue arm, LEFT wrist = orange arm). Best for the close-up "
        "gripper-object relationship on that arm - during a handoff, check "
        "both.\n"
        "  - Right main view: a 3-D point cloud reconstruction of the scene, "
        "with robot-frame axes (X red, Y green, Z blue) and TWO target gripper "
        "meshes you command - BLUE for the RIGHT arm and ORANGE for the LEFT "
        "arm (sides as seen in this default view). The selected arm's gripper "
        "is fully opaque; the other arm's is dimmed but stays visible. "
        "Translucent blue/orange patches on the table mark each arm's "
        "reachable workspace; where they overlap, both arms can operate (the "
        "handoff zone). A sidebar chip also names the selected arm. Either "
        "gripper renders RED while the controller cannot reach its pose - a "
        "reachability warning only, never a collision or contact indicator. "
        "Best for 3D / depth understanding that the flat camera images can't "
        "give.\n\n"
        "Call this first, or whenever you need a fresh view without taking an action."
    ),
    "hover": [("over the blue target gripper,", "over the selected target gripper,")],
    "gripper_teleport_via_click": (
        "Fast-move the selected arm's target gripper (blue = right arm, orange "
        "= left) near a clicked salient point. Returns a screenshot. Best for "
        "quickly setting a subtask's starting position; refine afterward with "
        "gripper_drag/move/rotate/toggle.\n\n"
        "Coordinate convention:\n"
        "  u = fraction across the screenshot, 0.0 left -> 1.0 right\n"
        "  v = fraction down the screenshot, 0.0 top -> 1.0 bottom\n\n"
        "Where to click:\n"
        "  - Left-panel 2-D camera views: best when the target is clearly "
        "visible in either camera feed.\n"
        "  - Right-panel 3-D point-cloud view: best when locating the target "
        "takes some 3D reasoning about depth.\n"
        "  - Click the object's center, not its edge, so the gripper lands "
        "centered over it for a reliable grasp or place.\n"
        "  - Avoid clicking on the OTHER arm's target gripper mesh: the meshes "
        "are not scene geometry, so the click selects whatever cloud point lies "
        "behind it.\n\n"
        "What happens:\n"
        "  The gripper lands ~0.07 m back from the clicked point along its "
        "current approach axis (opposite the reach direction), not on the point "
        "itself - so a short forward move/descent then reaches it. E.g. if the "
        "gripper currently points straight down, this puts it ~0.07 m directly "
        "above the click. Translation only - the gripper's orientation is "
        "preserved.\n"
        "  The readout reports the clicked point as the 'selected salient point' "
        "and the resulting gripper pose separately; check both, plus the "
        "screenshot. Clicking the selected target gripper itself does nothing "
        "here - that is reserved for gripper_drag."
    ),
    "gripper_drag": (
        "Drag the selected arm's target gripper between normalized screenshot "
        "coords (u across 0->1 left->right, v down 0->1 top->bottom). Returns a "
        "screenshot.\n\n"
        "The target grippers are the poses you command (blue = right arm, "
        "orange = left); this tool moves the SELECTED one. The gray point-cloud "
        "grippers are actual sensor data and not draggable meshes.\n\n"
        "Start point: u1/v1 must ray-hit the solid mesh of the SELECTED (fully "
        "opaque) target gripper - a finger, wrist, or body, not the empty gap "
        "between the fingers, the dimmed other-arm gripper, or a gray gripper. "
        "A miss is refused with a no-op; to drag the other arm, pass the arm "
        "parameter first.\n\n"
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
    ),
    "gripper_translate": [("Translate the blue target gripper",
                           "Translate the selected arm's target gripper")],
    "gripper_toggle": [("Open or close the blue target gripper.",
                        "Open or close the selected arm's target gripper (blue = right arm, "
                        "orange = left).")],
    "gripper_reset": [
        ("Reset the blue target gripper back onto the real gripper's current pose",
         "Reset the selected arm's target gripper back onto that arm's real gripper pose"),
        ("This only moves the blue target gripper", "This only moves the target gripper"),
    ],
    "gripper_advance_or_retreat": [("Move the blue target gripper",
                                    "Move the selected arm's target gripper")],
    "gripper_rotate": [("Rotate the blue target gripper",
                        "Rotate the selected arm's target gripper")],
    "gripper_get_pose": [
        ("Return the current blue target gripper pose",
         "Return the selected arm's current target gripper pose"),
        ("shows which way f/b will step. Returns",
         "shows which way f/b will step. Pass arm to read the other arm's target. Returns"),
        ("fingertips of the blue target gripper", "fingertips of the selected target gripper"),
    ],
    "gripper_show_rotation_gizmo": [
        ("Show the gripper's rotation gizmo", "Show the selected arm's rotation gizmo"),
        ("fingertips of the blue target gripper", "fingertips of the selected target gripper"),
    ],
    "camera_orbit_via_key": [
        ("(elevation -> 0).\n\n",
         "(elevation -> 0).\n\n"
         "Note the right/left arm naming and gripper colors are FIXED per episode "
         "(from the default view); they do not swap when the camera orbits to the "
         "other side.\n\n"),
    ],
    "camera_reset": [("viable from it, so prefer",
                      "viable from it (it is also the view the right/left arm naming refers "
                      "to), so prefer")],
    "camera_orbit_via_drag": [
        ("A left-drag that begins on the blue target gripper moves the GRIPPER instead of "
         "the camera, so this tool refuses that start point rather than silently displacing "
         "the waypoint.",
         "A left-drag that begins on the SELECTED arm's target gripper moves the GRIPPER "
         "instead of the camera, so this tool refuses that start point rather than silently "
         "displacing the waypoint (the other arm's dimmed gripper does not intercept drags)."),
    ],
    "execute_waypoint": [
        ("Execute the current gripper target and record one waypoint. Returns a screenshot.\n\n"
         "This asks the robot controller to move the real gripper to the current "
         "target pose: position, orientation, and final open/closed gripper state. ",
         "Execute the SELECTED arm's current gripper target and record one "
         "waypoint. Returns a screenshot.\n\n"
         "This asks the robot controller to move that arm's real gripper to its "
         "current target pose: position, orientation, and final open/closed "
         "gripper state. The other arm is not commanded - it holds its pose (and "
         "its grasp) while this arm moves; its staged target stays where you "
         "left it. Arms execute one at a time: plan each arm's waypoints "
         "separately and alternate as the task requires.\n\n"),
    ],
}
ATTRS = {
    "gripper_drag": {"miss_hint": (
        "  aim at a solid jaw block, wrist, or body of the fully opaque "
        "gripper; avoid the hollow slot between jaws, the gray real "
        "grippers, and the dimmed gripper (that one belongs to the other "
        "arm - pass the arm parameter to command it).")},
    "gripper_show_rotation_gizmo": {"arm_param": True},
}


def _tool(cls):
    desc = DESCRIPTION_EDITS.get(cls.name, cls.description)
    if isinstance(desc, list):
        edits, desc = desc, cls.description
        for old, new in edits:
            assert old in desc, (cls.name, old)
            desc = desc.replace(old, new)
    return type(cls.__name__, (Arm, cls), {"description": desc, **ATTRS.get(cls.name, {})})()


TOOLS = {t.name: t for t in map(_tool, (
    via.ScreenshotTool,
    via.HoverTool,
    via.GripperTeleportViaClickTool,
    via.GripperDragTool,
    via.GripperTranslateTool,
    via.GripperToggleTool,
    via.ResetTargetGripperTool,
    via.GripperAdvanceOrRetreatTool,
    via.GripperRotateTool,
    via.GetGripperPoseTool,
    via.ShowGripperRotationGizmoTool,
    via.CameraOrbitViaKeyTool,
    via.CameraPanViaKeyTool,
    via.CameraZoomTool,
    via.CameraResetTool,
    via.CameraOrbitViaDragTool,
    via.CameraPanViaDragTool,
    via.CameraSetPoseTool,
    via.GetCameraPoseTool,
    via.ExecuteTool,
    via.GripperHomeTool,
    via.EndEpisodeTool,
    via.CameraViewsTool,
    via.CameraFeedSelectTool,
    via.CameraRawTool,
))}
