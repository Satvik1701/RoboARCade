# How an episode is graded

A **episode** is one attempt by an agent at one task. Each task is split into **subtasks**, the steps it takes to
finish it (open the door, put the mug inside, serve the mug). An episode gets one number, its **score**: the share of
subtasks done. It is **solved** when every subtask is done. The last subtask is always the whole task.

```bash
.venv/bin/python -m core.grade_episode <episode folder>           # print the subtasks and the score
.venv/bin/python -m core.grade_episode <episode folder> --write   # and save them as subtasks.json in the folder
```

## How grading runs

While the agent works, the simulator saves a snapshot of the whole scene 20 times a second, in
`cameras_states.npz`. It also saves where it placed the fixed objects, such as pegs. When the agent says it is done,
the simulator keeps running until nothing in the scene moves, then half a second more, so the last snapshot shows
the scene at rest.

Nothing is graded while the agent works. `run_eval`, which runs the agents, grades each episode as soon as it ends.
It saves the result as `subtasks.json` in the episode's folder, under `data/episodes/<task>/<model>/`. When grading,
`grade_episode.py` rebuilds the scene from `tasks/<task>/` and the episode's seed (the number that sets the starting
layout). It loads the snapshots one by one, in order, and asks each subtask whether it is done at that moment.

A `subtasks.json` looks like this:

```json
{"source": "grade.py", "score": 0.67, "solved": false, "invariant_broken_at": null,
 "subtasks": [{"label": "open the microwave door", "done": true}, {"label": "put the mug inside", "done": true},
              {"label": "serve the mug on the plate", "done": false}]}
```

The dashboard's Analysis tab averages these scores. `verdict.json` in the same folder only says how the episode
ended: who ended it, whether time ran out, and how many steps it used. The Tasks and subtasks page lists every
task's subtasks and files.

## Files

| file | what it holds |
|---|---|
| `tasks/<task>/task.json` | the instruction, the scene setup, and `success`: what "the task is done" means |
| `tasks/<task>/grade.py` | the task's subtasks |
| `core/grade_episode.py` | the ready-made checks tasks use, such as `on_site` (an object rests on a spot) or `joint_in` (a door or knob is within a range) |
| `core/grade_episode.py` | the rules below, and the code that replays the snapshots |

## A subtask

Each subtask is a name and a check. The check is a small function that looks at one snapshot and answers true or
false: is this subtask done right now?

```python
from core.grade_episode import Subtask

def subtasks():
    return [
        Subtask('open the microwave door', lambda s: s.joint_in(joint='microwave_door', lo=-2.0, hi=-1.2), ever=True),
        Subtask('put the mug inside', lambda s: s.in_site_box(obj='mug_main', site='microwave_inside'), ever=True, needs=(0,)),
        Subtask('serve the mug on the plate', lambda s: s.success()),
    ]
```

Inside a check, `s` is the scene at that moment:

- `s.on_site(...)`, `s.joint_in(...)` and the like run one of the ready-made checks in `core/grade_episode.py`;
- `s.success()` asks whether the whole task is done, held for half a second; the last subtask always uses it;
- `s.env` is the simulator itself, for reading anything else, such as where an object is.

There are a few types of subtasks, and these are all of them:

- sequential: a subtask is done if its check is true in the last snapshot. This is the default.
- `ever=True`: some steps are undone on purpose later, like switching a mixer on and then off. They are done if
  the check was true at any moment.
- `needs=(0,)`: an `ever` subtask only counts once subtask 0 was done first.
- `end_needs=(5,)`: the subtask also needs subtask 5 done at the end. In the strongbox task, closing the lid only
  counts if all four coins (subtask 5) are inside at the end.
- the last subtask: always `s.success()`, the whole task done and held for half a second.
- `INVARIANT`: a task can name a rule that must hold the whole time, such as "the mug stays upright", in
  `INVARIANT`. The first moment it breaks, grading stops there: sequential subtasks are judged at the last snapshot
  before the break, `ever` subtasks by what they earned before it, and `end_needs` still apply. Nothing after the
  break counts.

If the agent runs out of time, the last subtask is not done.

## Adding a new task

Write `tasks/<task>/task.json`, `scene.xml`, `manifest.json`, `grade.py` and `test.py`, plus `dynamics.py` if
the scene moves by itself (see `tasks/README.md`). Copy `grade.py` from the
most similar task and keep its last subtask as `s.success()`. Run one episode, then grade its folder with the
command at the top and read the result of each subtask.

A new kind of check, such as how far an object is tilted, goes in one of two places.

**Only one task needs it.** Write it in that task's `grade.py` as a function named as the leaf, taking the leaf's
fields and returning a check of the simulator state; its docstring, with the fields filled in, is the sentence the
Tasks page shows. It is then a leaf like any other: subtasks call it as `s.<name>(...)` and task.json's `success`
names it as `{"pred": "<name>", ...}`.

```python
import numpy as np

def tilted(body, max_deg):
    """{body} tilted more than {max_deg} degrees"""
    def check(env):
        z = env.data.body(body).xmat.reshape(3, 3)[:, 2]          # the direction the object's top points
        return np.degrees(np.arccos(np.clip(z[2], -1, 1))) > max_deg
    return check
```

A check with state (one that latches, counts or keeps a baseline) is an object with `__call__` instead of a closure;
the teleop sidebar shows no instant truth for it.

**Multiple tasks need it.** Add it to `core/grade_episode.py`, next to the ready-made checks, so every task can use it.
That takes two small additions: one `if pred == "<name>":` branch in `_build` that returns the check, and one line in
the table of `_rows` that puts it into words for the Tasks page. A tilt check already exists there: `axis_up`.
