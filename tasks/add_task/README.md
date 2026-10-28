# Adding a task

The human side of a new task: the dashboard to drive it, the viser viewer to check its meshes, the thumbnail
and doc extractors the dashboard reads, and a randomization stress test. A task is a folder under `tasks/`
(task.json, scene.xml, manifest.json, grade.py, test.py; the folder contract is
`tasks/README.md`) with its
meshes and textures under `assets/tasks/<family>/<name>/`, synced to Hugging Face rather than committed.

`core/sim_env.py` loads the folder (`packs.load_scene` resolves the meshes), `core/initial_layout.py` places the
randomized objects (regions, separation, the reach gate), `core/grade_episode.py` compiles `success`,
`core/grade_episode.py` replays a recorded episode through the task's `grade.py`.

## The loop

1. **Author the pack**: copy the closest folder under `tasks/`, edit its scene and `task.json`, put the meshes
   under `assets/tasks/<family>/<name>/`.
2. **Check the meshes.** Start the dashboard and open `/viewer`: the viser page draws every task's visual meshes
   beside its collision geoms in orange, so an offset collider or a missing hull is visible at once.

       ROBOARCADE_BASE_PORT=8500 .venv/bin/python -m tasks.add_task.dashboard

3. **Drive it.** Click the task on the dashboard: the same UI and controller the agents get, rendered with
   MuJoCo. Randomize re-rolls the layout (watch for objects out of reach or intersecting), Record demo saves
   a teleop video under `data/task_demos/`, End Episode returns to the dashboard. Teleop episodes are never
   scored, but the live predicate readout shows whether the success tree is satisfiable.
4. **Stress the randomization**: `python -m tasks.add_task.rerandomize <task>` re-rolls the layout many times
   and reports collisions and reach failures.
5. **Run the tests**: `.venv/bin/pytest tasks/<task>` runs the task's own `test.py`, which covers the
   predicate leaves its tree uses.
6. **Refresh the dashboard extras**: `python -m tasks.add_task.thumbs` renders the card thumbnail,
   `python -m tasks.add_task.task_docs` extracts the per-task documentation the cards show.
7. **Evaluate**: `python -m core.run_eval --task <task> --seeds 5`.
   Each episode is graded with `tasks/<task>/grade.py` as it is published.

## Rules the packs follow

- One robot model per pack: single Panda, two Pandas or two YAMs, authored in `task.json`; there is no
  embodiment override.
- Every graded movable body needs a `robot_clear` (or a leaf that carries it) so success is not a held pose.
- Sequence stages latch in order and an invariant that stays broken for 5 control steps kills the episode.
- Randomization tiers name what moves; objects authored out of reach skip the reach gate.
- No dashes in the task language.

## Asset inspector

Inspect a body before evaluating agents. This uses the actual task model and its
`dynamics.py`; changes stay in memory and never rewrite the task or asset files.

```bash
uv sync --group dev
uv run python -m tasks.add_task.inspect_asset \
  --task start_incubation --body mixer --port 8410
```

Open `http://localhost:8410` and click the asset to interact. Drag to orbit and
scroll to zoom. Use **Task** to switch between all available task environments;
`--task` only selects the initial task and defaults to `start_incubation`.
Visual meshes and orange collision geometry are both visible initially. Turn off
**Visual meshes** for solid collision geometry, or turn off **Show collisions**
for appearance alone. Physics starts paused for inspection; mouse pokes still
advance physics, and **Run physics** enables continuous simulation. Switching
tasks clears the prior scene and saved state. A command-line body selection applies
only to the initial task; subsequent selections show the whole environment.
Reset, collision visibility, and Blender rendering are on the main
panel; diagnostics and exports are tucked under **More tools**. Omit `--body` to inspect the whole task, or use another
body name to include its descendants. Add `--host 0.0.0.0` for access from another
machine. `--seed` selects the initial state; authored layout is used without
additional randomization.

- **Appearance:** textured visual meshes, orbit/zoom, named sites and axes.
- **Collisions:** orange collision geometry (convex hulls for meshes), live contact
  pairs, penetration distance, and normal forces. Infinite planes appear as finite
  squares in the preview. The Blender image shows appearance; use the overlay for collisions.
- **Physics:** live startup, run/pause, single steps, simulation speed, reset,
  and save/restore including task userdata. The live task readout uses the optional
  `Dynamics.readout(env)` method, falling back to raw userdata.
- **Mouse interaction:** click a visible surface to apply a brief force at the
  clicked point. Small spring-loaded slide buttons are pressed along their travel
  axis and released automatically. No joint panel or numeric input is needed.
  This exercises dynamics with applied forces; use the task's contact tests or
  teleoperation dashboard to verify robot contact and reach.
- **Blender:** render the current state from the inspector camera or any task camera,
  optionally isolating the selected body. Requires the normal RoboARCade Blender
  installation (`ROBOARCADE_BLENDER` and `ROBOARCADE_BLENDER_PY` can override paths).
  Rendering pauses simulation. The image is shown in the page and saved as a PNG.
- **Export:** a timestamped folder under `data/asset_inspector` contains the compiled
  model (`scene.mjb`), full MuJoCo integration state (`state.npz`), and a JSON report
  with mass/inertia, joint state/limits, collision masks, friction, and dynamics.
  Blender exports include `blender.png`. Use `--output` and `--resolution` to change
  the destination and image size (default 1024).

This is an inspection tool, not a graded rollout. It advances physics and task
physics hooks; it does not execute robot waypoints or the evaluation protocol.
The interactive geometry preview covers rigid geoms; use Blender/the task dashboard
for deformable bodies and the task's tests for behavior assertions.
