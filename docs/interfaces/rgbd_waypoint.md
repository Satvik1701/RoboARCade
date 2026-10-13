# Multi-cam RGBD (`rgbd_waypoint`)

What the model receives on an episode with this interface, verbatim, in the order it arrives:

- **System prompt**: standing instructions, kept for the whole episode. The interface guide `interfaces/rgbd_waypoint/guide.md`, appended to the runtime's own system prompt (`--append-system-prompt-file` for Claude Code, `AGENTS.md` for Codex).
- **First user message**: the request the model works on. The task prompt, identical for every episode except the `Task:` line, which carries the pack's instruction. No further user message follows: the rest of the conversation is the model's tool calls and their results (text plus images).
- **Tools**: the 7 MCP tools of [`tools.py`](../../interfaces/rgbd_waypoint/tools.py), with the description and input schema the model sees.
- **Pre-compaction prompt**: appended to the runtime's summarizer when the context is compacted.

## System prompt

```markdown
# RGB-D waypoint control
Operate the single Panda arm or the two YAM arms using only the seven registered tools. Call observe first.

observe() defaults to agentview and all wrist cameras. It returns each camera's224x224 RGB image and aligned grayscale depth PNG, with intrinsics K and T_robot_from_camera. Separate mode places labels immediately before images. Tiled mode sends one PNG containing the same224x224 panels, with explicit panel xywh coordinates. Intrinsics always refer to individual panels; subtract a panel offset before interpreting its pixels. The selected views persist for action feedback. camera_views surveys all cameras at224x224 RGB; camera_raw returns one native RGB image (normally448x448) for details. Neither changes the selection.

Camera coordinates are X right,Y down,Z forward. Depth is optical-axis Z in metres, not ray distance. Depth PNG code0 means invalid or outside0.05..3.0m; valid codes1..255 map to z=0.05+(code-1)*2.95/254. Darker valid pixels are nearer. This is quantized visual depth, not an exact depth oracle. Black invalid pixels are not zero-distance surfaces. No per-frame normalization. K matches the returned image, with top-left pixel center(0,0). T_robot_from_camera maps camera points to the same coordinate frame as get_proprio/set_pose (RoboARCade simulator world metres). Wrist extrinsics change with the arm. Calibration and images share frame_id.

On two-arm tasks, every get_proprio, set_pose and set_gripper call requires arm="left" or arm="right". These sides match the default scene view (right blue, left orange); observation metadata maps sides to robot arm names and wrist cameras. Both arms use the same simulator world coordinate frame, not separate base frames. For example, set_pose(arm="right", d_z=0.01) lifts the right arm by 1 cm from its actual pose. The other arm keeps its existing controller command; physical contact may still move it. Actions execute one at a time and wait for completion; alternate arms for handoffs. Omit arm on single-arm tasks.

get_proprio reports actual fingertip-midpoint position, orientation, and the commanded open/closed state, not measured jaw width or grasp success. set_pose immediately executes translations in metres and rotations in degrees around fixed robot axes, pivoting at the fingertip midpoint. Omitted deltas are zero. Rotations compose onto the current rotation; they are not additions to Euler readouts. set_gripper separately opens/closes. Your existing controller handles interpolation and contact; blocked moves may stop short. Each action returns actual pose/displacement plus fresh selected RGB-D views. Inspect feedback and correct incrementally.

There is no target preview, hover, point-cloud screenshot, camera-motion tool or separate execute action. Use native camera_raw only when224px images lack needed detail. Reuse useful observation views rather than repeatedly requesting every camera. End with end_episode when finished. Stop on EPISODE OVER; no grading feedback is available.
```

## First user message

```text
Control the simulated robot arm to complete the following task.

Task: <the instruction from the pack, task.json language>

The interface guide in your system prompt describes the UI and the tools.
Nothing is graded until you say so. When you believe the task is complete, call end_episode. That call is final: it is the only thing that gets you graded, there is no second attempt, and you get no feedback before it, so check the scene yourself first. Fumbling is not failing: you may retry the task as often as you like within the step budget. What you cannot retry is the declaration.
Before each tool call, say in one or two sentences what you are about to do and why: what you believe the scene looks like, what the last result told you, and what this action is meant to achieve. If something did not work, say what you think went wrong before trying again.
Act fully autonomously; do not ask questions.
```

## Pre-compaction prompt

```text
This conversation drives a simulated robot arm through a 3D UI. The summary replaces
every screenshot, so it is the only record of physical state that survives. Alongside
the sections above, carry these forward concretely, with numbers, not prose:

1. Scene state now: for every object, where it currently is (world coordinates when you
   have them from hover/get_pose, otherwise its position relative to a named landmark or
   another object). Note any object that has been moved from where it started, and any
   that is in a pose the task did not intend (tipped, dropped, out of reach).
2. Gripper state now: open or closed, whether it is holding something and what, the
   current end-effector pose, and any waypoint that was set but not yet executed.
3. Progress against the task: which sub-goals are done and verified, which are in
   progress, and what the immediate next physical action is.
4. What has already failed, and the diagnosis. Grasps that slipped or closed on nothing,
   placements that missed or knocked something over, waypoints the controller could not
   reach. For each, say what you concluded caused it, so the approach is not repeated
   verbatim. This is the most valuable thing to preserve and the easiest to lose.
5. Calibration learned by doing: click-to-world offsets, approach heights and descent
   depths that worked, yaw values that aligned the jaws, camera poses that made a region
   legible. State them as usable numbers.

Be specific enough that the next turn can act without another look around: "red cube at
(0.12, -0.31), yellow staged left of it, gripper open and empty 8 cm above the red cube;
the first grasp at z=0.83 closed on air, 0.79 held", not "made progress on the blocks".

After compaction, before taking another physical action, get a fresh screenshot and call
gripper_get_pose. Reconcile both observations with the summary; when they disagree, trust
the fresh observations and revise the plan before moving the robot.
```

## Tools

### `observe`

```text
Read synchronized calibrated RGB-D for selected cameras at224x224; remembers this ordered selection for action feedback. No movement. Defaults agentview and all wrist cameras; includes wrist-to-arm mapping for bimanual tasks.
```

| Parameter | Type | Required | Description |
|---|---|---|---|
| `views` | array of string |  |  (minItems 1, maxItems 9) |

### `camera_raw`

```text
Read one native RGB camera image, normally448x448, no resize/depth/UI. Does not change selected observation cameras.
```

| Parameter | Type | Required | Description |
|---|---|---|---|
| `view` | string | yes |  |

### `camera_views`

```text
Survey all available cameras as individually labeled224x224 RGB images in stable order. No depth; does not change action-feedback selection.
```

No parameters.

### `get_proprio`

```text
Read the real gripper: position is fingertip midpoint in robot-frame metres; approach and opening are unit vectors; rpy_deg is scipy xyz Euler orientation in degrees, Rz(yaw) Ry(pitch) Rx(roll). Top-down is roll=+/-180, pitch=yaw=0. gripper is the real robot's last open/closed command (closed also while holding an object), not measured jaw width. No action or screenshot.
```

No parameters.

### `set_pose`

```text
Move immediately from the CURRENT real pose by one waypoint. d_x/d_y/d_z are metres along fixed robot axes (+z lifts). d_roll/d_pitch/d_yaw are degrees about fixed robot X/Y/Z axes, composed roll then pitch then yaw as dR * Rcurrent, not additions to Euler readouts. Rotation pivots about the fingertip midpoint; position stays fixed unless translation is requested. Prefer one rotation axis per call. Omitted deltas are zero; all-zero is a no-op. Returns selected calibrated RGB-D images, actual pose and actual position delta; blocked moves can stop short. No jaw change; use set_gripper separately. Robot z includes the task's table height.
```

| Parameter | Type | Required | Description |
|---|---|---|---|
| `d_x` | number |  | Robot X metres (default 0) |
| `d_y` | number |  | Robot Y metres (default 0) |
| `d_z` | number |  | Robot Z metres (default 0) |
| `d_roll` | number |  | Fixed robot X degrees (default 0) |
| `d_pitch` | number |  | Fixed robot Y degrees (default 0) |
| `d_yaw` | number |  | Fixed robot Z degrees (default 0) |

### `set_gripper`

```text
Open or close the real gripper immediately, with no pose change (one waypoint). Already in that command state is a no-op. Returns selected calibrated RGB-D images and real pose. Close only once the fingers surround the object; move and change jaws separately.
```

| Parameter | Type | Required | Description |
|---|---|---|---|
| `state` | `open` | `closed` | yes |  |

### `end_episode`

```text
Declare completion and end the episode. Returns confirmation only; no grading or success signal.
```

No parameters.
