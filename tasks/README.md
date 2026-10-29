# Tasks

One folder per task. The folder name is the task id (`python -m core.run_eval --task bowl_on_plate`).

| File | Holds |
|---|---|
| `task.json` | the instruction (`language`), `family` and `name`, `robots`, `cameras` with roles, `objects`, `randomization`, the `success` predicate tree, `termination` |
| `scene.xml` | the MuJoCo model: table, objects, cameras (the robot is attached from `assets/robots/`) |
| `manifest.json` | provenance and file hashes |
| `grade.py` | the task's subtasks: `subtasks()` returns a list of `Subtask`, and `INVARIANT(s)` is optional (below) |
| `dynamics.py` | optional, for a task with moving parts: a conveyor, a scripted shuffle, a mixer, a dealt board (below) |
| `render.py` | optional: Blender looks for the names this scene coins, `LOOKS` and `WORDS` shaped like `core/render/materials.py` |
| `test.py` | the task's own tests: mechanics, predicates, scripted motions |

Meshes and textures are not in git. They live under `assets/tasks/<family>/<name>/` (`family` and `name` from
task.json; `name` may carry a subpath such as `jenga/tower`), the layout every `scene.xml` was authored against, and
`core.tasks.load_scene` resolves the scene's relative mesh and texture dirs there. Shared mesh libraries
(`assets/robodojo_mjcf`, `fabrica_mjcf`, `pix2dex_mjcf`, `autobio`) sit beside them. `python scripts/sync_assets.py
pull` fetches exactly the files the packs reference.

Published task assets are complete. Final collision definitions are committed in `scene.xml`;
external meshes, textures and robot reachability grids are distributed through Hugging Face.
Asset creation and collision fitting happen before publication. Pulling assets and running a task
load those published files directly, without generation scripts or source-asset edits.

## Adding a task

1. Copy the closest folder, edit `task.json` and `scene.xml`, put the meshes under `assets/tasks/<family>/<name>/`.
2. Check the meshes and drive the task from the task dashboard (`tasks/add_task/README.md`).
3. Write `grade.py`, starting from the closest task's, and `dynamics.py` if the scene moves by itself (below).
4. Write `test.py` and run it with `python -m pytest tasks/<task>`.
5. `python -m core.run_eval --task <task> --seeds 5`. Each episode is graded with `grade.py` as it is published
   (`python -m core.grade_episode <episode>` prints the grade of one).

Nothing else in the repository names a task.

## Predicates and subtasks

`core/grade_episode.py` is the predicate language: it compiles a JSON tree (`{"pred": "all", "of": [...]}`) into a
function of the simulator state, the language task.json's `success` is written in. The leaves (`touching`, `in_site_box`, `joint_in`, `robot_clear`, ...) and the combinators `all`, `any`,
`not`, `count`, `sequence` are shared by every task. A leaf only one task needs is a function in that task's `grade.py`,
named as the leaf (`docs/design/grading.md` shows one).

`grade.py` lists the task's subtasks as Python checks on the scene, `s.<leaf>(**args)` for one leaf or `s.tree(spec)`
for a tree. The rule that turns them into the subtask score is the same for every task and lives in
`core/grade_episode.py`; `docs/design/grading.md` explains it.

## Moving parts: `dynamics.py`

A task whose scene moves by itself defines `class Dynamics` in `dynamics.py`. The simulator builds it once,
`Dynamics(env)`, and calls whichever of these methods it has, each with the simulator as `env`:

| method | when |
|---|---|
| `deal(env)` | at every reset, after the objects are placed: deal this episode's random choices from `env._rng` |
| `physics_step(env)` | before every physics step, once the episode is dealt: move scripted bodies, drive a belt or a motor |
| `control_step(env)` | after every control step (20 times a second) |
| `restore(env)` | after a saved state is loaded (an undo or a replay): repaint anything the state does not carry |
| `over(env)` | return True to end the episode (a one way conveyor that carried every case away) |

`env.episode_t0` is the sim time the episode started. Shell game, pack tools, vortex, incubation and
the two Lights Out puzzles are working examples. Nothing outside the task folder changes to add one.
