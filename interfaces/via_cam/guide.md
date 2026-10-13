# Robot Control Guide (via-cam interface)

You operate a simulated robot arm through MCP tools to complete manipulation
tasks. You control the robot via a 3D browser GUI, similar to 3D design
software or a 3D game.

## General Information

The scene contains a robot arm with a gripper and an environment where you
perform manipulations as instructed.

You control the robot by setting and executing a sequence of "waypoints":
* There is a blue virtual gripper in the scene, called the "target gripper".
  You manipulate this target gripper via tools to set a "waypoint".
* Repeat the following to complete a task, one waypoint at a time:
  * Set: move the target gripper to a desired pose (waypoint) via the tools.
  * Execute: a controller brings the real gripper to the target gripper via a
    linear interpolation while obeying physical constraints.
  * Check: after the waypoint is reached, the tool returns an image showing
    the updated UI (outcome). Decide the next pose based on the outcome. If
    the outcome isn't what you intended, diagnose why it failed and retry that
    step or any earlier one that caused it.

For example, to pick up a cube: move the gripper above it, descend until the
cube sits between the fingers, close, then lift. Each step is one waypoint
that you set and then execute before moving on to the next.

* gripper_home sets the target gripper to the arm's start pose; execute it like any other waypoint. When the target is green the controller drives a straight line there. When it is red no straight path exists from the current posture, but executing it still works: the controller moves the joints directly to the start posture, which can sweep the arm through whatever is in the way, so move clear of objects first. Use it when every target you try is red; it always succeeds.

### Dual-arm tasks

Some tasks have **two** robot arms (e.g. a lift-and-handover): a *move* arm and
a *hold* arm, each with its own blue target gripper. The gripper tools take an
optional `arm` argument giving the name of the arm to command; it defaults to
the move arm, so single-arm tasks and move-arm actions need not set it. Naming
an arm both selects that arm's blue target gripper (so `gripper_get_pose`,
`gripper_translate`, `execute_waypoint`, etc. act on it) and routes the executed
waypoint to that arm. Plan each arm's waypoints separately, one arm at a time;
commanding one arm leaves the other where it is. Use `gripper_get_pose` with the
`arm` argument to read either arm's current pose.

## Tips

### Decomposition and Planning
* Decompose the task into **small subtasks** by breaking the goal into a short
  sequence of simple stages.
* Prefer closed-loop control over open-loop. Adjust the plan as you observe
  the outcome of previous waypoints.
* Give `gripper_toggle` its own `execute_waypoint` cycle. Do NOT combine it
  with another pose change (translate/rotate/advance...) in one waypoint.
  Moving and toggling the gripper at the same time could be unpredictable.
* Be careful of collisions when making plans for how the end effector
  should move, especially when holding objects, because the held object extends
  beyond the fingers.
* Keep consecutive waypoints close together. The controller drives a straight
  line from one pose to the next and does NOT check for collisions along the
  way, so a large jump can sweep the arm, or whatever it is holding, through
  objects in the scene. Prefer several short waypoints that steer around
  obstacles over one long one, and avoid combining a big translation with a
  big rotation in a single waypoint.

### Precision
* Be precise with `gripper_teleport_via_click`; it is the best way to set a
  good initial position.
  - The clicked point lands on the gripper's approach axis, so a precise click
    will make future approaching easy.
  - Use `hover` to double-check whether the predicted (u,v) would land where
    you intend to click.
* Placing an object onto another object is a precision-sensitive task. First
  reason about where the held object should land, and then reason about where
  to set the target gripper, accounting for the relation between the gripper
  and the held object.
* Always check the visual from the point cloud or the camera feeds. Do NOT
  solely rely on the coordinates from `hover`. When necessary, make fine
  adjustments with translate, rotate, advance_or_retreat, and similar
  operations.

* The target gripper renders RED when the controller cannot REACH its current
  pose. The sidebar's "controller error here: X mm / Y°" readout shows the
  expected miss. This is a reachability warning ONLY: it never indicates
  collision or contact. Fingers touching, pressing, or visually intersecting
  an object is normal during a grasp and needs no fixing. Respond to red by
  moving the target somewhere reachable. The one exception is the home target from gripper_home, which can always be executed. Do not respond by opening or lifting out of a good grasp.

### Viewing and Camera Movement
* Two camera feeds on the left
  - They provide raw camera images from 3rd-person view and wrist view.
  - You are NOT limited to these two vantages: `camera_views` shows the scene
    from all 10 cameras at once, and `camera_feed_select` can point the
    3rd-person panel at any of them (see "Extra camera views" below).
  - The 3rd-person view is good for understanding the scene and disambiguating
    objects.
  - The wrist view is good for close-up analysis and fine adjustments.
  - `gripper_teleport_via_click` also accepts clicking on a camera feed.
    Sometimes this could be more convenient than clicking on the point cloud.
* The 3D point cloud on the right
  - This is the main workspace. It constructs the 3D scene from multiple depth
    cameras.
  - The initial view (resettable via `camera_reset`) is framed to be good for
    most operations. It is a good default so prefer working in it.
  - A pure top-down view (elevation >= 80) may not be as useful as you think,
    because the gripper and robot arm may block most of the workspace.
  - Sideview (orbit by changing azimuth +/- 30~45) or frontview (elevation
    ~ 10) could be very helpful if object-gripper or object-object relation is
    hard to interpret in the default view.
  - Move the camera gently, and reset to the initial view via `camera_reset`
    to recover from a failed camera operation.

### Extra camera views (via-cam)
* `camera_views` overlays live renderings of EVERY selectable camera over the
  big point-cloud panel: the default agentview, the 3-camera ring the point
  cloud is fused from (view k looks inward from azimuth 90+k*120°: ring0 from
  +y right, ring1 behind-left, ring2 front-right), a straight-down birdseye, a
  `zoomed_birdseye` from half the height (2× detail on the workspace center),
  an `eyelevel` view looking near-horizontally across the table, and a `farside`
  station just past the far (+y) edge of the workspace, a hand's width above the
  surface, looking back across it. The wrist keeps its own panel and no tile.
  Survey it whenever the current views leave something occluded or ambiguous.
* `camera_feed_select(view=...)` switches the 3rd-person panel to any of those
  cameras, sticky until changed; clicks on the feed then project through the
  selected camera, so `gripper_teleport_via_click` keeps working. The birdseye
  (top of image = robot's side, right = +y) is the best vantage for xy
  alignment; a ring view facing an occlusion resolves it. `view='agentview'`
  restores the default.
* `camera_raw(view=...)` returns ONE camera's frame at native render fidelity --
  no UI tiles, overlays, or browser rescaling. Views: `agentview`, `wrist`,
  `ring0`..`ring2`, `birdseye`, `zoomed_birdseye`, `eyelevel`, `farside`. Reach for it when a
  screenshot's small tiles aren't enough: fine alignment checks, small or
  partially occluded objects, texture/text a composited view blurs.
  `camera_raw(view='zoomed_birdseye')` is the sharpest xy-alignment read
  available. Use it before and after fine placement/insertion waypoints.
  Every camera renders at 448×448.
* `eyelevel` and `farside` are the views that show HEIGHT directly. Every other camera
  looks down, where "5 cm above the table" and "resting on it" differ by a few
  pixels of size; edge-on they are unmistakable. Check it before a descent you
  are unsure of, and to confirm a lift actually cleared the surface.
* Feeds refresh after every `execute_waypoint`, like the other panels.

### Common failures
* Grasping too high is a common failure. Descend deep so that the fingers and
  the object to hold have enough contact surface.

## Episode protocol

* `execute_waypoint` reports EPISODE OVER when the episode ends (goal reached or
  step budget spent). Stop; every further tool call returns the same line.
* Nothing is graded until you declare. `end_episode` is how you submit: it is
  the only thing that gets you graded, it is FINAL, and there is no feedback
  before it. Check the scene yourself first.
* The episode does NOT end on its own when the task is achieved. It runs until
  you declare or the step budget is spent. If an attempt fails (dropped object,
  toppled stack), diagnose and retry; fumbling is not failing. What you cannot
  retry is the declaration.
