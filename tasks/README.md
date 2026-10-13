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

`env.episode_t0` is the sim time the episode started. The five tasks that move (shell game, pack tools, vortex and
the two Lights Out puzzles) are working examples. Nothing outside the task folder changes to add one.

## Test coverage

"None" means the task has checks but no scripted robot motion; "missing" means no `test.py` yet. The shared pack
checks and the opt-in waypoint sweep exercise every pack.

| Task | Scripted robot motions | Other checks and the remaining gap |
|---|---|---|
| [set_up_pcr_run](set_up_pcr_run/test.py) | PCR lid contact with both fingers | No full scripted solve. |
| [start_incubation](start_incubation/test.py) | None | Recorded tube release and seating, toppling, friction caps and physical wells. |
| [vortex_sample](vortex_sample/test.py) | None | Mixer mechanism, pressure and hold gating, speed, liquid and state reset. |
| [beam](beam/test.py) | None | Constructed placement predicate; the shared controller sweeps also cover the beam. |
| [cooling_manifold](cooling_manifold/test.py) | Pick, lift, carry through random poses, spin about the jaw axis | Recorded seating and release, contact pair checks. |
| gamepad | missing | Shared pack and assembly checks only. |
| [stool_circular](stool_circular/test.py) | None | Position-only predicate checks. |
| [bowl_on_plate](bowl_on_plate/test.py) | Rim grasp, lift, carry, lower and release onto the plate | Full solve plus a lift-only probe; placement predicates. |
| [open_middle_drawer_halfway](open_middle_drawer_halfway/test.py) | Recorded home-return motions | Reachability and drawer predicates; no full scripted solve. |
| [serve_warm_drinks](serve_warm_drinks/test.py) | None | Press edge checks. |
| soup_and_sauce | missing | Shared pack, fixture and terminal grading checks only. |
| [start_making_breakfast](start_making_breakfast/test.py) | None | Constructed and settled placement predicate. |
| [tidy_counter](tidy_counter/test.py) | None | Bowl containment invariant and pack checks. |
| [puzzle_4x6](puzzle_4x6/test.py) | None | Seeded board, direct button state changes and success predicates. |
| [puzzle_6x7](puzzle_6x7/test.py) | None | Seeded board, camera convention and button toggles. |
| [bottles_in_bin](bottles_in_bin/test.py) | None | Bottle mass and contact declarations, randomization, appearance and clearance predicate. |
| [cup_pyramid](cup_pyramid/test.py) | None | Cup contacts and randomization, constructed support states. |
| [handover_dominos](handover_dominos/test.py) | None | Initial piles, constructed line and cascade checks; some contacts are mocked. |
| [jenga_tower](jenga_tower/test.py) | None | Initial piles and a constructed supported tower. |
| [hang_mugs_and_rack](hang_mugs_and_rack/test.py) | None | Authored contacts, constructed fixture goals and upright plate checks. |
| [lock_the_coins_in_the_strongbox](lock_the_coins_in_the_strongbox/test.py) | Grasp the lid tab with both pads; reject a one-pad brush | Key, rotor and lid mechanics, randomized starts and the coin sequence. |
| organize_table_and_plug_in_charger | missing | Shared pack and waypoint checks only. |
| [pack_tools_into_toolboxes](pack_tools_into_toolboxes/test.py) | Hold a tool while the conveyor advances | Conveyor removal, seated and rail riders, belt speeds; no complete packing script. |
| sort_lego_bricks | missing | Shared pack checks only. |
| sort_pills | missing | Shared pack checks only. |
| [dual_stack](dual_stack/test.py) | Pick, rotate, carry, place two blocks and retreat | Full solve and constructed goal predicates. |
| [square](square/test.py) | Sub-mm translations and small rotations | Waypoint and controller regressions, placement predicate; no insertion script. |
| [shell_game](shell_game/test.py) | None | Direct button presses and scripted scene motion. |

The bowl and dual-stack motion scripts drive the real controller through the test VIA kernel and write MuJoCo
preview videos to `data/test_videos/`; they are engineering checks with ground-truth targeting, not model
evaluations. A new motion regression should define the seed or starting pose, drive the production controller,
assert acquisition and retention during the motion, then release and the physical outcome; keep negative cases
(one-pad contacts, unintended sticking) beside the valid grasps, and never tune a parameter for one trajectory
or relax an assertion to make a change pass.
