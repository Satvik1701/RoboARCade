# Robot Control Guide

You operate a simulated robot arm through the `roboarcade` MCP server (`core/mcp_server.py`) to
complete manipulation tasks across the robomimic, LIBERO, and Fabrica task families.


## General Information

You control the robot via a 3D browser GUI, similar to 3D design software or a 3D game.
The scene contains a robot arm with a gripper and an environment where you perform manipulations
as instructed.

You control the robot by setting and executing a sequence of "waypoints":
* There is a blue virtual gripper in the scene, called the "target gripper". You manipulate this
  target gripper via tools to set a "waypoint".
* Repeat the following to complete a task, one waypoint at a time:
  * Set: move the target gripper to a desired pose (waypoint) via the tools defined in the MCP server.
  * Execute: A controller will bring the real gripper to the target gripper via a linear
    interpolation while obeying physical constraints.
  * Check: After the waypoint is reached, the tool will return an image showing the updated UI
    (outcome). Decide the next pose based on the outcome. If the outcome isn't what you intended,
    diagnose why it failed and retry that step or any earlier one that caused it.

For example, to pick up a cube: move the gripper above it, descend until the cube sits between
the fingers, close, then lift. Each step is one waypoint that you set and then execute before
moving on to the next.

**Dual-arm tasks (only if a task's spec enables a second arm):** the scene has two target
grippers (a *move* arm and a *hold* arm). Waypoint tools take an `arm` argument; it defaults
to the move arm, so single-arm tasks behave exactly as above.


## Tips

### Decomposition and Planning
* Decompose the task into **small subtasks** by breaking the goal into a short sequence of simple stages.
* Prefer closed-loop control over open-loop. Adjust the plan as you observe the outcome of previous waypoints.
* Give `gripper_toggle` its own `execute_waypoint` cycle. Do NOT combine it with another pose
  change (translate/rotate/advance...) in one waypoint. Moving and toggling the gripper at the
  same time could be unpredictable.
* Be careful of collisions when making plans for how the end effector
  should move, especially when holding objects, since the held object extends
  beyond the fingers.
* Keep consecutive waypoints close together. The controller drives a straight
  line from one pose to the next and does NOT check for collisions along the
  way, so a large jump can sweep the arm, or whatever it is holding, through
  objects in the scene. Prefer several short waypoints that steer around
  obstacles over one long one, and avoid combining a big translation with a
  big rotation in a single waypoint.

### Precision
* Be precise with `gripper_teleport_via_click`, the best way to set a good initial position.
  - The clicked point lands on the gripper's approach axis, so a precise click will make future approaching easy.
  - Use `hover` to double-check whether the predicted (u,v) would land where you intend to click.
* Placing an object onto another object is a precision-sensitive task. First reason about where
  the held object should land, and then reason about where to set the target gripper, accounting
  for the relation between the gripper and the held object.
* Always check the visual from the point cloud or the camera feeds. Do NOT solely rely on the
  coordinates from `hover`. When necessary, make fine adjustments with translate, rotate,
  advance_or_retreat, and similar operations.
* The target gripper renders RED when the controller cannot REACH its current
  pose (the sidebar shows the expected miss in mm). Reachability warning ONLY --
  never a collision or contact indicator: fingers touching or intersecting an
  object is normal during a grasp; respond by moving the target somewhere
  reachable, not by abandoning a good grasp.

### Viewing and Camera Movement:
* Two camera feeds on the left
  - They provide raw camera images from 3rd-person view and wrist view.
  - The 3rd-person view is good for understanding the scene and disambiguating objects.
  - The wrist view is good for close-up analysis and fine adjustments.
  - `gripper_teleport_via_click` also accepts clicking on a camera feed. Sometimes this could be
    more convenient than clicking on the point cloud.
* The 3D point cloud on the right
  - This is the main workspace. It constructs the 3D scene from multiple depth cameras.
  - The initial view (resettable via `camera_reset`) is framed to be good for most operations. It
    is a good default so prefer working in it.
  - A pure top-down view (elevation >= 80) may not be as useful as you think, because the gripper
    and robot arm may block most of the workspace.
  - Sideview (orbit by changing azimuth +/- 30~45) or frontview (elevation ~ 10) could be very
    helpful if object-gripper or object-object relation is hard to interpret in the default view.
  - Move the camera gently, and reset to the initial view via `camera_reset` to recover from a
    failed camera operation.

### Common failures
* Grasping too high is a common failure. Descend deep so that the fingers and the object to hold
  have enough contact surface.


## Code style (binding, L11)

Simple, minimal, no AI bloat. Concretely:
* Write the smallest code that solves the actual problem. No speculative abstractions,
  config options, or "future-proofing": an abstraction must have ≥2 real users today.
* One obvious way: if a helper is used once, inline it. If a module doesn't earn its
  existence, delete it.
* No defensive bloat: no broad try/except wrappers, no redundant validation of internal
  invariants, no logging noise.
* Comments only where the code can't say it (units, frames, physics rationale, gotchas).
  No narration of what the code obviously does, no boilerplate docstrings.
* Match the terse idiom of the modules ported from sphinx-via-claude; keep diffs reviewable.

## Repo orientation

This guide documents the **default (via-cam) interface**; the docs site is built by `docs/build.py`. Tasks are the
folders under `tasks/` (one per task: task.json, scene.xml, manifest.json, grade.py, test.py, and dynamics.py when
the scene moves by itself) loaded by
`core/sim_env.py`, with the meshes under `assets/`. The predicate language is `core/grade_episode.py`;
`core/grade_episode.py` scores each episode once, with the task's `grade.py` (see `docs/design/grading.md`).
