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
