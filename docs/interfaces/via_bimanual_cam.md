# VIA bimanual (`via_bimanual_cam`)

What the model receives on an episode with this interface, verbatim, in the order it arrives:

- **System prompt**: standing instructions, kept for the whole episode. The interface guide `interfaces/via_bimanual_cam/guide.md`, appended to the runtime's own system prompt (`--append-system-prompt-file` for Claude Code, `AGENTS.md` for Codex).
- **First user message**: the request the model works on. The task prompt, identical for every episode except the `Task:` line, which carries the pack's instruction. No further user message follows: the rest of the conversation is the model's tool calls and their results (text plus images).
- **Tools**: the 25 MCP tools of [`tools.py`](../../interfaces/via_bimanual_cam/tools.py), with the description and input schema the model sees. This page renders them for a two-arm pack whose arms are named `move` and `hold`, which is where the `arm` parameter comes from.
- **Pre-compaction prompt**: appended to the runtime's summarizer when the context is compacted.

## System prompt

```markdown
# Robot Control Guide (via-bimanual-cam interface)

You operate TWO simulated robot arms through MCP tools to complete bimanual
manipulation tasks (e.g. pick with one arm, hand off to the other). You
control the robots via a 3D browser GUI, similar to 3D design software or a
3D game.

## General Information

The scene contains two robot arms, each with a gripper, and an environment
where you perform manipulations as instructed.

Each arm has its own virtual "target gripper", color-coded by side:

* **RIGHT arm = BLUE target gripper**
* **LEFT arm = ORANGE target gripper**

Sides are as seen in the DEFAULT view (`camera_reset`); they are fixed for
the whole episode and do NOT swap when you orbit the camera. A sidebar chip
names the currently commanded arm, and the selected arm's target gripper
renders fully opaque while the other arm's is dimmed.

You control the robots by setting and executing a sequence of "waypoints",
one arm at a time:
* Pick an arm with the `arm` argument (`'right'` or `'left'`) on any gripper
  tool. The selection is sticky -- later tools keep acting on that arm until
  you name the other one.
* Set: move that arm's target gripper to a desired pose (waypoint) via the
  tools.
* Execute: a controller brings that arm's real gripper to its target via a
  linear interpolation while obeying physical constraints. The OTHER arm is
  not commanded -- it holds its pose and its grasp, and its staged target
  stays where you left it.
* Check: after the waypoint is reached, the tool returns an image showing
  the updated UI (outcome). Decide the next pose based on the outcome. If
  the outcome isn't what you intended, diagnose why it failed and retry that
  step or any earlier one that caused it.

For example, to hand a cube from one arm to the other: grasp and lift it
with the first arm, bring it into the shared workspace, move the second
arm's open gripper around the held cube from a clear direction, close the
second gripper, then open the first and retreat it -- each step is one
waypoint on one arm.

* gripper_home drives the arm straight back to its start pose along a joint-space path; it does not check for collisions any more than a waypoint does, so lift clear first

## Workspaces

Translucent patches on the table mark each arm's reachable workspace:
**blue = right arm, orange = left arm**. Where the patches overlap, both
arms can operate -- plan handoffs and shared manipulation there. Outside its
patch an arm cannot reach; don't drag an object beyond the holding arm's
patch and expect the other arm to take it from there.

## Tips

### Decomposition and Planning
* Decompose the task into **small subtasks** by breaking the goal into a
  short sequence of simple stages, and decide which arm does each stage.
* Command one arm at a time; plan each arm's waypoints separately. Alternate
  arms as the task requires.
* Prefer closed-loop control over open-loop. Adjust the plan as you observe
  the outcome of previous waypoints.
* Give `gripper_toggle` its own `execute_waypoint` cycle. Do NOT combine it
  with another pose change (translate/rotate/advance...) in one waypoint.
* Be careful of collisions when making plans for how the end effector
  should move, especially when holding objects -- the held object extends
  beyond the fingers.
* Keep consecutive waypoints close together. The controller drives a straight
  line from one pose to the next and does NOT check for collisions along the
  way, so a large jump can sweep the arm -- or whatever it is holding -- through
  objects in the scene. Prefer several short waypoints that steer around
  obstacles over one long one, and avoid combining a big translation with a
  big rotation in a single waypoint.
* For a handoff, approach the held object with the receiving gripper
  PERPENDICULAR to the holding gripper's fingers so the two grippers don't
  collide, and only open the giving gripper after the receiving one has
  closed on the object.

### Precision
* Be precise with `gripper_teleport_via_click` -- it's the best way to set a
  good initial position.
  - The clicked point lands on the gripper's approach axis, so a precise click
    will make future approaching easy.
  - Use `hover` to double-check whether the predicted (u,v) would land where
    you intend to click.
  - Avoid clicking on the other arm's target gripper mesh: the meshes are not
    scene geometry, so the click selects whatever cloud point lies behind it.
* Placing an object onto another object is a precision-sensitive task. First
  reason about where the held object should land, and then reason about where
  to set the target gripper, accounting for the relation between the gripper
  and the held object.
* Always check the visual from the point cloud or the camera feeds. Do NOT
  solely rely on the coordinates from `hover`. When necessary, make fine
  adjustments with translate, rotate, advance_or_retreat, and similar
  operations.

* The target gripper renders RED when the controller cannot REACH its current
  pose -- the sidebar's "controller error here: X mm / Y°" readout shows the
  expected miss. This is a reachability warning ONLY: it never indicates
  collision or contact. Fingers touching, pressing, or visually intersecting
  an object is normal during a grasp and needs no fixing -- respond to red by
  moving the target somewhere reachable, not by opening or lifting out of a
  good grasp.

### Viewing and Camera Movement
* Camera feeds on the left
  - A 3rd-person view plus one wrist view PER ARM, titled by side and color
    (RIGHT wrist = blue arm, LEFT wrist = orange arm).
  - The 3rd-person view is good for understanding the scene, disambiguating
    objects, and seeing both arms at once.
  - Each wrist view is a close-up from that arm's gripper -- use the matching
    side's view for fine adjustments, and check both during a handoff.
  - `gripper_teleport_via_click` also accepts clicking on a camera feed.
    Sometimes this could be more convenient than clicking on the point cloud.
* The 3D point cloud on the right
  - This is the main workspace. It constructs the 3D scene from multiple depth
    cameras.
  - The initial view (resettable via `camera_reset`) is framed to show both
    arms and their workspaces, and it is the view the right/left naming
    refers to. It is a good default so prefer working in it.
  - A pure top-down view (elevation >= 80) may not be as useful as you think,
    because the grippers and robot arms may block most of the workspace.
  - Sideview (orbit by changing azimuth +/- 30~45) or frontview (elevation
    ~ 10) could be very helpful if gripper-gripper or object-gripper relations
    are hard to interpret in the default view -- especially the alignment of
    the two grippers during a handoff.
  - Move the camera gently, and reset to the initial view via `camera_reset`
    to recover from a failed camera operation.

### Extra camera views (via-bimanual-cam)
* The other arm's wrist camera is a viewpoint you control: park the idle arm where its
  wrist feed looks at the working jaw when a step needs precision or the fixed cameras
  are occluded, then check that feed with `camera_feed_select` or `camera_raw`.
* `camera_views` overlays live renderings of EVERY selectable camera over the
  big point-cloud panel: the default agentview, the 3-camera ring the point
  cloud is fused from (view k looks inward from azimuth 90+k*120°: ring0 from
  +y right, ring1 behind-left, ring2 front-right), a straight-down birdseye, a
  `zoomed_birdseye` from half the height (2× detail on the workspace center),
  and an `eyelevel` view looking near-horizontally across the table.
  Survey it whenever the current views leave something occluded or ambiguous.
* `camera_feed_select(view=...)` switches the 3rd-person panel to any of those
  cameras, sticky until changed; clicks on the feed then project through the
  selected camera, so `gripper_teleport_via_click` keeps working. The birdseye
  (top of image = robot's side, right = +y) is the best vantage for xy
  alignment; a ring view facing an occlusion resolves it. `view='agentview'`
  restores the default.
* `camera_raw(view=...)` returns ONE camera's frame at native render fidelity --
  no UI tiles, overlays, or browser rescaling. Views: `agentview`, `wrist`,
  `ring0`..`ring2`, `birdseye`, `zoomed_birdseye`, `eyelevel`. Reach for it when a
  screenshot's small tiles aren't enough: fine alignment checks, small or
  partially occluded objects, texture/text a composited view blurs.
  `camera_raw(view='zoomed_birdseye')` is the sharpest xy-alignment read
  available -- use it before and after fine placement/insertion waypoints.
  Every camera renders at 448×448.
* `eyelevel` is the only view that shows HEIGHT directly. Every other camera
  looks down, where "5 cm above the table" and "resting on it" differ by a few
  pixels of size; edge-on they are unmistakable. Check it before a descent you
  are unsure of, and to confirm a lift actually cleared the surface.
* Feeds refresh after every `execute_waypoint`, like the other panels.

### Common failures
* Grasping too high is a common failure. Descend deep so that the fingers and
  the object to hold have enough contact surface.
* Commanding the wrong arm: check the sidebar chip and which gripper is fully
  opaque before executing; pass `arm` explicitly when in doubt.
* Gripper-gripper collision during handoffs: keep the two approach axes
  roughly perpendicular and leave clearance until the receiving gripper is
  aligned.

## Episode protocol

* `execute_waypoint` reports EPISODE OVER when the episode ends (goal reached or
  step budget spent) -- stop; every further tool call returns the same line.
* Nothing is graded until you declare. `end_episode` is how you submit: it is
  the only thing that gets you graded, it is FINAL, and there is no feedback
  before it. Check the scene yourself first.
* The episode does NOT end on its own when the task is achieved -- it runs until
  you declare or the step budget is spent. If an attempt fails (dropped object,
  toppled stack), diagnose and retry; fumbling is not failing. What you cannot
  retry is the declaration.
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

### `screenshot`

```text
Capture a screenshot of the robot interface.

The window shows:
  - Left sidebar: camera feeds, rendered from the real scene; they refresh only after execute_waypoint moves a real gripper.
      - 3rd-person view: a fixed camera looking at the table and both arms. Best for analyzing the full scene layout (objects relative to each other and the grippers).
      - One wrist view PER ARM, each a camera on that arm's gripper looking down between its fingers, titled by side and color (RIGHT wrist = blue arm, LEFT wrist = orange arm). Best for the close-up gripper-object relationship on that arm - during a handoff, check both.
  - Right main view: a 3-D point cloud reconstruction of the scene, with robot-frame axes (X red, Y green, Z blue) and TWO target gripper meshes you command - BLUE for the RIGHT arm and ORANGE for the LEFT arm (sides as seen in this default view). The selected arm's gripper is fully opaque; the other arm's is dimmed but stays visible. Translucent blue/orange patches on the table mark each arm's reachable workspace; where they overlap, both arms can operate (the handoff zone). A sidebar chip also names the selected arm. Either gripper renders RED while the controller cannot reach its pose - a reachability warning only, never a collision or contact indicator. Best for 3D / depth understanding that the flat camera images can't give.

Call this first, or whenever you need a fresh view without taking an action.
```

No parameters.

### `hover`

```text
Move the mouse to normalized screenshot coordinates (u, v) and show a visible crosshair overlay in the returned screenshot.

Coordinate convention:
  u = fraction across the screenshot, 0.0 left -> 1.0 right
  v = fraction down the screenshot, 0.0 top -> 1.0 bottom

Reading the overlay:
  The exact (u, v) point is the crosshair's center dot.

Use hover to inspect a spot before acting on it. It takes no action and gives two readouts:
  - Visual: the crosshair (a persistent DOM overlay, not the OS cursor) marks exactly where (u, v) lands, confirming it's on the intended UI element or point-cloud region.
  - Contact point (robot-frame x/y/z, m), computed read-only. On the point-cloud canvas or a camera feed it reports the cloud point under the cursor (the ray needn't be pixel-exact - it hits within a small radius). Two uses:
      1. Probe the robot-frame coordinate of any point of interest.
      2. Preview a teleport - when it lands on a point it is exactly the salient point a gripper_teleport_via_click at this same (u, v) would select. Note this is the clicked point itself, NOT the resulting gripper pose (which lands ~0.07 m back along the approach axis).
    It reads 'none' when (u, v) is not over a cloud point (empty space), over the selected target gripper, or over UI chrome rather than the point-cloud canvas or a camera feed.
```

| Parameter | Type | Required | Description |
|---|---|---|---|
| `u` | number | yes | Fraction across the screenshot, 0.0 left to 1.0 right. (minimum 0.0, maximum 1.0) |
| `v` | number | yes | Fraction down the screenshot, 0.0 top to 1.0 bottom. (minimum 0.0, maximum 1.0) |

### `gripper_teleport_via_click`

```text
Frame (shared by all spatial tools):
  robot - metres; the canonical frame every pose, delta, and camera target reports in and that every tool input expects.
  orientation - the gripper's pose is reported as two robot-frame unit vectors: 'approach' (the direction the fingers reach) and 'opening' (the fingertip-to-fingertip axis the jaws open/close along).
  arms - two arms, each with its own target gripper: 'right' = the BLUE gripper, 'left' = the ORANGE one (sides as seen in the default view). Tools with an arm parameter command the named arm; the selection is sticky and the selected gripper renders fully opaque.
Robot Z position carries a fixed height offset, so a canonical top-down gripper reads z~1.0, not 0; reason about position deltas.

Fast-move the selected arm's target gripper (blue = right arm, orange = left) near a clicked salient point. Returns a screenshot. Best for quickly setting a subtask's starting position; refine afterward with gripper_drag/move/rotate/toggle.

Coordinate convention:
  u = fraction across the screenshot, 0.0 left -> 1.0 right
  v = fraction down the screenshot, 0.0 top -> 1.0 bottom

Where to click:
  - Left-panel 2-D camera views: best when the target is clearly visible in either camera feed.
  - Right-panel 3-D point-cloud view: best when locating the target takes some 3D reasoning about depth.
  - Click the object's center, not its edge, so the gripper lands centered over it for a reliable grasp or place.
  - Avoid clicking on the OTHER arm's target gripper mesh: the meshes are not scene geometry, so the click selects whatever cloud point lies behind it.

What happens:
  The gripper lands ~0.07 m back from the clicked point along its current approach axis (opposite the reach direction), not on the point itself - so a short forward move/descent then reaches it. E.g. if the gripper currently points straight down, this puts it ~0.07 m directly above the click. Translation only - the gripper's orientation is preserved.
  The readout reports the clicked point as the 'selected salient point' and the resulting gripper pose separately; check both, plus the screenshot. Clicking the selected target gripper itself does nothing here - that is reserved for gripper_drag.
```

| Parameter | Type | Required | Description |
|---|---|---|---|
| `u` | number | yes | Fraction across the screenshot, 0.0 left to 1.0 right. (minimum 0.0, maximum 1.0) |
| `v` | number | yes | Fraction down the screenshot, 0.0 top to 1.0 bottom. (minimum 0.0, maximum 1.0) |
| `arm` | `right` | `left` |  | Arm this tool commands: 'right' is the BLUE target gripper, 'left' is the ORANGE one (sides as seen in the default view). The selection is sticky - it persists for later tools until changed. Default: the currently selected arm (the fully opaque gripper). |

### `gripper_drag`

```text
Frame (shared by all spatial tools):
  robot - metres; the canonical frame every pose, delta, and camera target reports in and that every tool input expects.
  orientation - the gripper's pose is reported as two robot-frame unit vectors: 'approach' (the direction the fingers reach) and 'opening' (the fingertip-to-fingertip axis the jaws open/close along).
  arms - two arms, each with its own target gripper: 'right' = the BLUE gripper, 'left' = the ORANGE one (sides as seen in the default view). Tools with an arm parameter command the named arm; the selection is sticky and the selected gripper renders fully opaque.
Robot Z position carries a fixed height offset, so a canonical top-down gripper reads z~1.0, not 0; reason about position deltas.

Drag the selected arm's target gripper between normalized screenshot coords (u across 0->1 left->right, v down 0->1 top->bottom). Returns a screenshot.

The target grippers are the poses you command (blue = right arm, orange = left); this tool moves the SELECTED one. The gray point-cloud grippers are actual sensor data and not draggable meshes.

Start point: u1/v1 must ray-hit the solid mesh of the SELECTED (fully opaque) target gripper - a finger, wrist, or body, not the empty gap between the fingers, the dimmed other-arm gripper, or a gray gripper. A miss is refused with a no-op; to drag the other arm, pass the arm parameter first.

Motion model: free drag moves the gripper with the cursor in the view plane, ~0.0006 m per CSS pixel of mouse movement, before constraints.

Movement modes - free is the default (no lock); x/y/z lock motion to one robot-frame axis (the fixed robot X/Y/Z, not the gripper-attached gizmo axes that gripper_rotate spins about) and are exact at any camera angle, so trust the pose readout, not the picture:
  free - unconstrained: follows the cursor in the camera view plane (camera-dependent)
  x    - lock to robot X only
  y    - lock to robot Y only
  z    - lock to robot Z / vertical

Tip: hover first to confirm u1/v1 lands on the gripper, then use small single-axis drags. The screenshot shows start/end markers, end crosshair, the arrow path, and the updated pose.
```

| Parameter | Type | Required | Description |
|---|---|---|---|
| `u1` | number | yes | Start fraction across the screenshot, 0.0 left to 1.0 right. (minimum 0.0, maximum 1.0) |
| `v1` | number | yes | Start fraction down the screenshot, 0.0 top to 1.0 bottom. (minimum 0.0, maximum 1.0) |
| `u2` | number | yes | End fraction across the screenshot, 0.0 left to 1.0 right. (minimum 0.0, maximum 1.0) |
| `v2` | number | yes | End fraction down the screenshot, 0.0 top to 1.0 bottom. (minimum 0.0, maximum 1.0) |
| `constraint` | `free` | `x` | `y` | `z` |  | Movement constraint. Default: free. |
| `steps` | integer |  | Mouse move interpolation steps. Default: 25. (minimum 1, maximum 100) |
| `arm` | `right` | `left` |  | Arm this tool commands: 'right' is the BLUE target gripper, 'left' is the ORANGE one (sides as seen in the default view). The selection is sticky - it persists for later tools until changed. Default: the currently selected arm (the fully opaque gripper). |

### `gripper_translate`

```text
Frame (shared by all spatial tools):
  robot - metres; the canonical frame every pose, delta, and camera target reports in and that every tool input expects.
  orientation - the gripper's pose is reported as two robot-frame unit vectors: 'approach' (the direction the fingers reach) and 'opening' (the fingertip-to-fingertip axis the jaws open/close along).
  arms - two arms, each with its own target gripper: 'right' = the BLUE gripper, 'left' = the ORANGE one (sides as seen in the default view). Tools with an arm parameter command the named arm; the selection is sticky and the selected gripper renders fully opaque.
Robot Z position carries a fixed height offset, so a canonical top-down gripper reads z~1.0, not 0; reason about position deltas.

Translate the selected arm's target gripper by a metric offset along the robot frame axes, in metres. Returns a screenshot. You specify the offset directly - no start point, mesh hit, or mouse drag needed - so it is the exact, camera-independent way to shift the gripper along the world axes. This is the keyed equivalent of gripper_drag's x/y/z axis locks without the cursor imprecision; the orientation is unchanged.

Axes are the fixed robot X/Y/Z (the colored arrows), not the gripper-attached gizmo axes that gripper_rotate spins about, so the offsets are exact at any camera angle - trust the pose readout:
  d_x - along robot X
  d_y - along robot Y
  d_z - along robot Z (vertical: +z lifts, -z lowers)

Set one offset for a pure single-axis move, or several to move diagonally; omitted offsets default to 0 and at least one must be nonzero. Each offset must be within +/-0.1 m to keep moves gradual; larger requests are rejected, so call repeatedly to translate further. The returned text reports the actual pose delta after the move.
```

| Parameter | Type | Required | Description |
|---|---|---|---|
| `d_x` | number |  | Offset along robot X, metres. Must be within +/-0.1. Default: 0. (minimum -0.1, maximum 0.1) |
| `d_y` | number |  | Offset along robot Y, metres. Must be within +/-0.1. Default: 0. (minimum -0.1, maximum 0.1) |
| `d_z` | number |  | Offset along robot Z (vertical); +z lifts, -z lowers, metres. Must be within +/-0.1. Default: 0. (minimum -0.1, maximum 0.1) |
| `arm` | `right` | `left` |  | Arm this tool commands: 'right' is the BLUE target gripper, 'left' is the ORANGE one (sides as seen in the default view). The selection is sticky - it persists for later tools until changed. Default: the currently selected arm (the fully opaque gripper). |

### `gripper_toggle`

```text
Open or close the selected arm's target gripper (blue = right arm, orange = left). Returns a screenshot.

Wraps the UI 'g' key, which swaps the gripper mesh between open and closed. Pass state to drive a known result without checking the current state first:
  state='toggle' - flip the current state (default)
  state='open'   - ensure open  (no-op if already open)
  state='closed' - ensure closed (no-op if already closed)

The returned message and screenshot report the resulting open/closed state, read back after the swap.
```

| Parameter | Type | Required | Description |
|---|---|---|---|
| `state` | `toggle` | `open` | `closed` |  | Target state. Default: toggle. |
| `arm` | `right` | `left` |  | Arm this tool commands: 'right' is the BLUE target gripper, 'left' is the ORANGE one (sides as seen in the default view). The selection is sticky - it persists for later tools until changed. Default: the currently selected arm (the fully opaque gripper). |

### `gripper_reset`

```text
Frame (shared by all spatial tools):
  robot - metres; the canonical frame every pose, delta, and camera target reports in and that every tool input expects.
  orientation - the gripper's pose is reported as two robot-frame unit vectors: 'approach' (the direction the fingers reach) and 'opening' (the fingertip-to-fingertip axis the jaws open/close along).
  arms - two arms, each with its own target gripper: 'right' = the BLUE gripper, 'left' = the ORANGE one (sides as seen in the default view). Tools with an arm parameter command the named arm; the selection is sticky and the selected gripper renders fully opaque.
Robot Z position carries a fixed height offset, so a canonical top-down gripper reads z~1.0, not 0; reason about position deltas.

Reset the selected arm's target gripper back onto that arm's real gripper pose - position, orientation, and open/closed state. Returns a screenshot.

Use this to undo experimental edits: freely try gripper_drag, gripper_advance_or_retreat, gripper_rotate, or gripper_toggle to plan a move, see how it looks, then reset to the real pose before committing the move you actually want. This only moves the target gripper - it does not move the real robot and records no waypoint.
```

| Parameter | Type | Required | Description |
|---|---|---|---|
| `arm` | `right` | `left` |  | Arm this tool commands: 'right' is the BLUE target gripper, 'left' is the ORANGE one (sides as seen in the default view). The selection is sticky - it persists for later tools until changed. Default: the currently selected arm (the fully opaque gripper). |

### `gripper_advance_or_retreat`

```text
Frame (shared by all spatial tools):
  robot - metres; the canonical frame every pose, delta, and camera target reports in and that every tool input expects.
  orientation - the gripper's pose is reported as two robot-frame unit vectors: 'approach' (the direction the fingers reach) and 'opening' (the fingertip-to-fingertip axis the jaws open/close along).
  arms - two arms, each with its own target gripper: 'right' = the BLUE gripper, 'left' = the ORANGE one (sides as seen in the default view). Tools with an arm parameter command the named arm; the selection is sticky and the selected gripper renders fully opaque.
Robot Z position carries a fixed height offset, so a canonical top-down gripper reads z~1.0, not 0; reason about position deltas.

Move the selected arm's target gripper along its approach axis with f/b. Returns a screenshot.

  direction='f' - step along the approach axis (toward where the fingers point)
  direction='b' - step opposite the approach axis (retreat / back out)
  In robot frame the direction follows the current orientation (see Frame above): top-down, 'f' is -Z (down) and 'b' is +Z (lift / back out); on a side-grasp the same steps are horizontal.

If the current orientation makes the f/b direction ambiguous, call gripper_get_pose first to see the approach axis drawn in the screenshot.

Scale:
  One step is about 0.006 m along the approach axis. Use 1-3 steps for small corrections and larger counts for coarse approach/retreat.

The returned text reports the actual pose delta after f/b keypresses.
```

| Parameter | Type | Required | Description |
|---|---|---|---|
| `direction` | `f` | `b` | yes | f moves forward; b moves backward. |
| `steps` | integer |  | Number of f/b keypresses. Default: 1. (minimum 1, maximum 10) |
| `arm` | `right` | `left` |  | Arm this tool commands: 'right' is the BLUE target gripper, 'left' is the ORANGE one (sides as seen in the default view). The selection is sticky - it persists for later tools until changed. Default: the currently selected arm (the fully opaque gripper). |

### `gripper_rotate`

```text
Frame (shared by all spatial tools):
  robot - metres; the canonical frame every pose, delta, and camera target reports in and that every tool input expects.
  orientation - the gripper's pose is reported as two robot-frame unit vectors: 'approach' (the direction the fingers reach) and 'opening' (the fingertip-to-fingertip axis the jaws open/close along).
  arms - two arms, each with its own target gripper: 'right' = the BLUE gripper, 'left' = the ORANGE one (sides as seen in the default view). Tools with an arm parameter command the named arm; the selection is sticky and the selected gripper renders fully opaque.
Robot Z position carries a fixed height offset, so a canonical top-down gripper reads z~1.0, not 0; reason about position deltas.

Rotate the selected arm's target gripper about one gizmo axis (x, y, or z) by a signed angle in degrees. Returns a screenshot.

IMPORTANT: call `gripper_show_rotation_gizmo` before EVERY rotation and pick the axis and direction from the gizmo you see - do not guess. The x/y/z are the gripper's own gizmo axes, not the world/robot X/Y/Z (the colored arrows); they line up only at the top-down pose and move with the gripper after any tilt, so the same letter points a different way once it is tilted.

The gripper has three mutually perpendicular gizmo axes: the approach axis (where the fingers reach), the opening axis (the finger line), and a third axis perpendicular to both. Each rotation spins about one, leaving it fixed and re-aiming the other two (the arrow on each ring shows the + direction):
* x - about the third axis; both the approach and opening axes re-aim.
* y - about the opening axis; opening stays fixed, approach re-aims.
* z - about the approach axis; approach stays fixed, opening re-aims (twist in place).

The returned text reports the resulting approach and opening axes; the rotation is in place (position unchanged). If you are unsure about the axis or direction, rotate in small amount and observe its outcome.
```

| Parameter | Type | Required | Description |
|---|---|---|---|
| `axis` | `x` | `y` | `z` | yes | Gizmo rotation axis to spin about: x, y, or z. Choose it from gripper_show_rotation_gizmo. |
| `angle` | number | yes | Signed rotation angle in degrees, within [-90, 90]. Prefer small steps (e.g. 10-45 degrees) so you can observe the intermediate result and adjust before overshooting. (minimum -90, maximum 90) |
| `arm` | `right` | `left` |  | Arm this tool commands: 'right' is the BLUE target gripper, 'left' is the ORANGE one (sides as seen in the default view). The selection is sticky - it persists for later tools until changed. Default: the currently selected arm (the fully opaque gripper). |

### `gripper_get_pose`

```text
Frame (shared by all spatial tools):
  robot - metres; the canonical frame every pose, delta, and camera target reports in and that every tool input expects.
  orientation - the gripper's pose is reported as two robot-frame unit vectors: 'approach' (the direction the fingers reach) and 'opening' (the fingertip-to-fingertip axis the jaws open/close along).
  arms - two arms, each with its own target gripper: 'right' = the BLUE gripper, 'left' = the ORANGE one (sides as seen in the default view). Tools with an arm parameter command the named arm; the selection is sticky and the selected gripper renders fully opaque.
Robot Z position carries a fixed height offset, so a canonical top-down gripper reads z~1.0, not 0; reason about position deltas.

Return the selected arm's current target gripper pose and visually show its approach axis in the screenshot. Besides giving a readout of the exact current gripper pose, the drawn approach axis makes this the best orientation check before using gripper_advance_or_retreat: it shows which way f/b will step. Pass arm to read the other arm's target. Returns a screenshot.

Reports the center of the fingertips of the selected target gripper in robot metres, its orientation as approach and opening robot-frame unit vectors, and open/closed state. The visual axis guide is shown without moving the gripper.
```

| Parameter | Type | Required | Description |
|---|---|---|---|
| `arm` | `right` | `left` |  | Arm this tool commands: 'right' is the BLUE target gripper, 'left' is the ORANGE one (sides as seen in the default view). The selection is sticky - it persists for later tools until changed. Default: the currently selected arm (the fully opaque gripper). |

### `gripper_show_rotation_gizmo`

```text
Frame (shared by all spatial tools):
  robot - metres; the canonical frame every pose, delta, and camera target reports in and that every tool input expects.
  orientation - the gripper's pose is reported as two robot-frame unit vectors: 'approach' (the direction the fingers reach) and 'opening' (the fingertip-to-fingertip axis the jaws open/close along).
  arms - two arms, each with its own target gripper: 'right' = the BLUE gripper, 'left' = the ORANGE one (sides as seen in the default view). Tools with an arm parameter command the named arm; the selection is sticky and the selected gripper renders fully opaque.
Robot Z position carries a fixed height offset, so a canonical top-down gripper reads z~1.0, not 0; reason about position deltas.

Show the selected arm's rotation gizmo (the three rotation rings) together with its approach axis in the screenshot. Always call this first to choose the gripper_rotate axis: it gives an intuitive visual understanding of the rotation axes (the x/y/z gizmo rings). The arrow on each rotation ring points to the + direction of that rotation. Returns a screenshot.

Reports the center of the fingertips of the selected target gripper in robot metres, its orientation as approach and opening robot-frame unit vectors, and open/closed state.
```

| Parameter | Type | Required | Description |
|---|---|---|---|
| `arm` | `right` | `left` |  | Arm this tool commands: 'right' is the BLUE target gripper, 'left' is the ORANGE one (sides as seen in the default view). The selection is sticky - it persists for later tools until changed. Default: the currently selected arm (the fully opaque gripper). |

### `camera_orbit_via_key`

```text
Orbit the camera around the scene target by an angular delta, in degrees. Returns a screenshot. You specify the two orbit angles directly - no start point or open-space click needed - so it is the simplest way to reframe.

Both deltas are RELATIVE to the current pose (read camera_get_pose for the absolute angles). The distance and look-at target are unchanged - this only rotates the viewpoint.

Directions:
  +d_azimuth swings the camera counterclockwise seen from above while -d_azimuth swings it clockwise. At the default view, azimuth=90 with Y=0 looking towards -X. +d_azimuth will swing the camera to +Y and gradually move towards looking at -Y at azimuth=180.
  +d_elevation raises the camera toward a top-down view (elevation -> 90); -d_elevation lowers it toward a front or side horizontal view (elevation -> 0).

Note the right/left arm naming and gripper colors are FIXED per episode (from the default view); they do not swap when the camera orbits to the other side.

Each delta must be within +/-45 deg to keep camera moves gradual; larger requests are rejected, so call repeatedly to orbit further. Elevation is clamped to [0, 85] (the horizon is allowed; kept just off exact top-down, which is degenerate); azimuth wraps at 360. Returns the updated camera pose.
```

| Parameter | Type | Required | Description |
|---|---|---|---|
| `d_azimuth` | number | yes | Change in horizontal orbit angle, degrees. + orbits the camera left, - orbits right. Must be within +/-45. (minimum -45.0, maximum 45.0) |
| `d_elevation` | number | yes | Change in vertical angle, degrees. + raises the camera toward top-down (90), - lowers it toward the horizon (0). Must be within +/-45. (minimum -45.0, maximum 45.0) |

### `camera_pan_via_key`

```text
Pan the camera (shift its look-at point) by a metric offset, in robot-frame metres. Returns a screenshot. You specify the shift directly - no start point or drag needed - so it is the simplest way to re-center the view without rotating.

The viewing angle (azimuth/elevation) and distance are unchanged - the camera and its target slide together by the same amount, so only the framing moves.

Directions (relative to the current view):
  +d_right pans right: the look-at point slides toward screen-right, so you see more of what was off the right edge (scene content appears to slide left). -d_right pans left.
  +d_up pans up along world-Z (robot +z): you see more above (scene content appears to slide down). -d_up pans down.

Each offset must be within +/-0.2 m to keep moves gradual; larger requests are rejected, so call repeatedly to pan further. Returns the updated camera pose, whose target shows the new framing center.
```

| Parameter | Type | Required | Description |
|---|---|---|---|
| `d_right` | number | yes | Horizontal pan along camera-right, robot metres. + pans right (reveals scene to the right), - pans left. Must be within +/-0.2. (minimum -0.2, maximum 0.2) |
| `d_up` | number | yes | Vertical pan along world-Z (robot +z), metres. + pans up (reveals scene above), - pans down. Must be within +/-0.2. (minimum -0.2, maximum 0.2) |

### `camera_zoom`

```text
Zoom the camera toward normalized screenshot coordinates. Returns a screenshot.

The browser zooms toward the point under the mouse. u/v should be inside the point-cloud view.

Coordinate convention:
  u = fraction across the screenshot, 0.0 left -> 1.0 right
  v = fraction down the screenshot, 0.0 top -> 1.0 bottom

Positive steps zoom in; negative steps zoom out. Internally this sends scroll-wheel events to the browser. Each step changes the camera-to-target distance by a constant ~0.03 m, regardless of current zoom level or direction (so steps ~= |delta_distance| / 0.03). Off-center zoom centers also shift composition toward that point; use this deliberately to reframe while zooming. Returns the updated camera pose.
```

| Parameter | Type | Required | Description |
|---|---|---|---|
| `u` | number | yes | Zoom center fraction across the screenshot. (minimum 0.0, maximum 1.0) |
| `v` | number | yes | Zoom center fraction down the screenshot. (minimum 0.0, maximum 1.0) |
| `steps` | integer |  | Scroll steps. Positive zooms in; negative zooms out. Default: 1. (minimum -20, maximum 20) |

### `camera_reset`

```text
Restore the initial camera view. Returns a screenshot and the updated camera pose.

Strategy: the default view is framed so most operations are viable from it (it is also the view the right/left arm naming refers to), so prefer working there and return to it with this tool after exploring. Reach for camera_orbit_via_key / camera_pan_via_key only when the default view is insufficient - e.g. to resolve an occlusion, or to judge the gripper-to-object relationship (depth, alignment, clearance) from another angle - then reset here to re-establish a known frame.
```

No parameters.

### `camera_orbit_via_drag`

```text
Orbit the camera by left-dragging the point cloud, from (u1, v1) to (u2, v2). Returns a screenshot and the updated camera pose.

The drag must START ON EMPTY SPACE. A left-drag that begins on the SELECTED arm's target gripper moves the GRIPPER instead of the camera, so this tool refuses that start point rather than silently displacing the waypoint (the other arm's dimmed gripper does not intercept drags).

Motion model: the view swings ~0.29 deg of azimuth per CSS pixel of horizontal movement and the same in elevation vertically; dragging right swings the scene left. camera_orbit_via_key is the tool to reach for when you know the angle you want - use this one to nudge the view by eye.
```

| Parameter | Type | Required | Description |
|---|---|---|---|
| `u1` | number | yes | Start fraction across the screenshot, 0.0 left to 1.0 right. (minimum 0.0, maximum 1.0) |
| `v1` | number | yes | Start fraction down the screenshot, 0.0 top to 1.0 bottom. (minimum 0.0, maximum 1.0) |
| `u2` | number | yes | End fraction across the screenshot, 0.0 left to 1.0 right. (minimum 0.0, maximum 1.0) |
| `v2` | number | yes | End fraction down the screenshot, 0.0 top to 1.0 bottom. (minimum 0.0, maximum 1.0) |
| `steps` | integer |  | Mouse move interpolation steps. Default: 25. (minimum 1, maximum 100) |

### `camera_pan_via_drag`

```text
Pan the camera by right-dragging from (u1, v1) to (u2, v2): the camera and its target slide together, so the view shifts without rotating. Returns a screenshot and the updated camera pose.

Right-drag pans wherever it starts - the gripper does not intercept it - so unlike camera_orbit_via_drag any start point works. Dragging right moves the scene right (the camera moves left). Use it to re-centre a workspace that orbiting or zooming has pushed off-screen.
```

| Parameter | Type | Required | Description |
|---|---|---|---|
| `u1` | number | yes | Start fraction across the screenshot, 0.0 left to 1.0 right. (minimum 0.0, maximum 1.0) |
| `v1` | number | yes | Start fraction down the screenshot, 0.0 top to 1.0 bottom. (minimum 0.0, maximum 1.0) |
| `u2` | number | yes | End fraction across the screenshot, 0.0 left to 1.0 right. (minimum 0.0, maximum 1.0) |
| `v2` | number | yes | End fraction down the screenshot, 0.0 top to 1.0 bottom. (minimum 0.0, maximum 1.0) |
| `steps` | integer |  | Mouse move interpolation steps. Default: 25. (minimum 1, maximum 100) |

### `camera_set_pose`

```text
Place the camera at an ABSOLUTE pose: orbit angles, distance, and the target it looks at. Returns a screenshot and the resulting pose.

Every field is in the same convention camera_get_pose reports, so a pose read from there can be handed straight back to return to a known view: azimuth is the horizontal orbit angle in degrees, elevation is 0=horizontal and 90=top-down (clamped to [0, 85] so the view never reaches the degenerate pole), distance is camera-to-target in metres, and the target is a robot-frame point in metres.

This is the tool for restoring a framing you noted earlier, or for jumping to a named viewpoint in one call instead of accumulating orbit/pan/zoom deltas. camera_reset returns to the pack's default view.
```

| Parameter | Type | Required | Description |
|---|---|---|---|
| `azimuth` | number | yes | Horizontal orbit angle, degrees (wraps at 360). |
| `elevation` | number | yes | Vertical angle, degrees. 0=horizontal, 90=top-down; clamped to [0, 85]. |
| `distance` | number | yes | Camera-to-target separation, metres. |
| `target_x` | number | yes | Target x, robot frame, metres. |
| `target_y` | number | yes | Target y, robot frame, metres. |
| `target_z` | number | yes | Target z, robot frame, metres. |

### `camera_get_pose`

```text
Return the current camera pose without taking an action.

The camera orbits a target point, reported in the robot frame (metres) like every other pose: target (robot, m) is the world point the camera looks at, and distance is the camera-to-target separation in metres. azimuth/elevation are the viewing direction onto that target: azimuth is the horizontal orbit angle in degrees; elevation is 0=horizontal (level with the target) and 90=top-down, but orbiting keeps it within [0, 85] so it never reaches the degenerate top-down pole (where azimuth would collapse to 0). The target is reported because pan and off-center zoom shift the scene center. Use these values to reason about the current framing before adjusting it with other camera tools.
```

No parameters.

### `execute_waypoint`

```text
Execute the SELECTED arm's current gripper target and record one waypoint. Returns a screenshot.

This asks the robot controller to move that arm's real gripper to its current target pose: position, orientation, and final open/closed gripper state. The other arm is not commanded - it holds its pose (and its grasp) while this arm moves; its staged target stays where you left it. Arms execute one at a time: plan each arm's waypoints separately and alternate as the task requires.

Use this after meaningful target-setting operations such as gripper_teleport_via_click, gripper_drag, gripper_advance_or_retreat, gripper_rotate, or gripper_toggle. Execute a gripper_toggle on its own. Do not combine it with a pose change in the same waypoint, since moving and opening/closing the gripper at once is unreliable.

Execution uses the final target as the next base state for later operations. The controller interpolates between the current pose and target pose; it does not replay the exact UI trajectory used to place the target. For example, if you toggled the gripper twice before executing, it only tries to match the final open/closed state, without intermediate toggles.

If the episode ends on this waypoint (goal reached, or the step budget is spent) the result says EPISODE OVER. Stop then: no further tools.
```

| Parameter | Type | Required | Description |
|---|---|---|---|
| `arm` | `right` | `left` |  | Arm this tool commands: 'right' is the BLUE target gripper, 'left' is the ORANGE one (sides as seen in the default view). The selection is sticky - it persists for later tools until changed. Default: the currently selected arm (the fully opaque gripper). |

### `gripper_home`

```text
Frame (shared by all spatial tools):
  robot - metres; the canonical frame every pose, delta, and camera target reports in and that every tool input expects.
  orientation - the gripper's pose is reported as two robot-frame unit vectors: 'approach' (the direction the fingers reach) and 'opening' (the fingertip-to-fingertip axis the jaws open/close along).
  arms - two arms, each with its own target gripper: 'right' = the BLUE gripper, 'left' = the ORANGE one (sides as seen in the default view). Tools with an arm parameter command the named arm; the selection is sticky and the selected gripper renders fully opaque.
Robot Z position carries a fixed height offset, so a canonical top-down gripper reads z~1.0, not 0; reason about position deltas.

gripper_home sets the target gripper to the arm's start pose; execute it like any other waypoint. When the target is green the controller drives a straight line there. When it is red no straight path exists from the current posture, but executing it still works: the controller moves the joints directly to the start posture, which can sweep the arm through whatever is in the way, so move clear of objects first. Use it when every target you try is red; it always succeeds. The target jaw command stays unchanged. Returns a screenshot and target gripper pose readout.
```

| Parameter | Type | Required | Description |
|---|---|---|---|
| `arm` | `right` | `left` |  | Arm this tool commands: 'right' is the BLUE target gripper, 'left' is the ORANGE one (sides as seen in the default view). The selection is sticky - it persists for later tools until changed. Default: the currently selected arm (the fully opaque gripper). |

### `end_episode`

```text
Save/end the current episode and reset or advance the task. Returns a screenshot.

Call this only after the final waypoint for the episode has been executed.
```

No parameters.

### `camera_views`

```text
Show live renderings from EVERY selectable camera at once, overlaid large over the 3-D point-cloud panel (returned in the screenshot; the overlay is removed afterwards). Cameras shown:
  - agentview: the pack's default third_person camera, exposed by role
  - ring0..ring2: the depth cameras the point cloud is FUSED from, spaced 120 deg apart around the workspace looking inward. View k looks FROM azimuth 90+k*120 deg in the robot frame: ring0 from +y (screen right in the default 3-D view), ring1 from behind-left, ring2 from front-right. They sit off the world axes on purpose, so a flat face is never seen edge-on from all three.
  - birdseye: a straight-down overhead view of the whole workspace (top of its image = the robot's side, image right = +y) - the best vantage for xy layout and alignment, with no height cues.
  - zoomed_birdseye: the same straight-down view from half the height - twice the detail on the middle of the workspace (the edges fall outside its frame). Pair with camera_raw for insertion-grade close reads.

  - eyelevel: a near-horizontal view across the table from the agentview's side, 10 deg above the surface. The only view that shows HEIGHT directly - every other camera looks down, where a height difference reads only as a size change. Use it to judge how far above a surface the gripper or a held object really is.
  - farside: the second ground-level station, a quarter turn round from the eyelevel and much closer in: it sits just past the far (+y) edge of the workspace, a hand's width above the surface, looking back across it. Height reads here too, and whatever stands at the far edge is in the near field at a useful size.

Use this to survey the scene from every angle when something is occluded or ambiguous, then keep the most useful vantage on screen with camera_feed_select. Frames refresh after every execute_waypoint.
```

No parameters.

### `camera_feed_select`

```text
Switch the 3rd-person camera feed (top-left panel) to any selectable camera: 'agentview' (the pack's third_person role), 'ring0'..'ring2' (the point-cloud ring, view k looking inward from azimuth 90+k*120 deg), 'birdseye' (straight-down overhead), 'zoomed_birdseye' (the same view from half the height - 2x detail on the workspace center), or 'eyelevel' (near-horizontal across the table, the one view that shows height directly). The choice is sticky - the panel keeps streaming that camera, refreshed after every execute_waypoint, until changed - and clicks on the feed (gripper_teleport_via_click, hover) project through the SELECTED camera, so click-to-place keeps working. Survey the options first with camera_views; pass 'agentview' to restore the pack's third_person camera by role. Returns a screenshot showing the new feed.
```

| Parameter | Type | Required | Description |
|---|---|---|---|
| `view` | string | yes | Camera to show in the 3rd-person panel: 'agentview', 'ring0'..'ring2', 'birdseye', or 'eyelevel'. |

### `camera_raw`

```text
Return the RAW rendered RGB image from ONE camera, exactly as the sim renders it: native camera resolution, no UI chrome, tiles, overlays, or browser rescaling. Selectable cameras:
  - 'agentview': the pack's third_person camera, exposed by role
  - 'wrist': the gripper's eye-in-hand camera (dual-arm tasks also serve 'wrist_<arm>' for the extra arms)
  - 'ring0'..'ring2': the point-cloud ring, view k looking inward from azimuth 90+k*120 deg
  - 'birdseye': straight-down overhead (top of image = the robot's side, image right = +y)
  - 'zoomed_birdseye': the same overhead from half the height - 2x the pixels per cm on the workspace center, the sharpest view for fine alignment and insertions
  - 'eyelevel': near-horizontal across the table from the agentview's side - one of the two views that read HEIGHT directly (the others look down, where height reads only as a size change)
  - 'farside': a hand's width above the surface just past the far (+y) edge of the workspace, looking back across it - the other height read, and the closest view of whatever stands at that edge

Resolution: every camera renders at 448x448.

Complements the composited views: use camera_views to survey every angle at once, then this for a full-fidelity close read of the one that matters (fine alignment, small/partially occluded objects, text or texture the screenshot tiles blur). The frame is from the last completed observation - it refreshes after every execute_waypoint, not live. An unknown name returns the available list.
```

| Parameter | Type | Required | Description |
|---|---|---|---|
| `view` | string | yes | Camera to fetch: 'agentview', 'wrist', 'ring0'..'ring2', 'birdseye', 'zoomed_birdseye', 'eyelevel', or 'farside'. |
