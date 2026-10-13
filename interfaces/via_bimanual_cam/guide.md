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
