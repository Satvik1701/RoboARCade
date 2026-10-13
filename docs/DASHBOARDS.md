# The dashboard

## Running it: `python -m dashboard` (port 8400)

`dashboard/dashboard.py` is the server. It reads two things under `data/`: `episodes/` and
`analysis.json`. It launches nothing: `python -m core.run_eval` does.

- **Zero-shot evaluation**: the launches under `data/episodes`, twenty a page, newest first. A launch opens its
  episodes; an episode opens its video, the agent's transcript with every screenshot and the subtasks.
- **Analysis**: Task successes, Table 1 (also with every metric) and autonomous improvement, computed from the `summary.json` of every episode folder listed in
  `data/analysis.json` (`"episodes"`: folder paths; `"excluded_tasks"`: shown, left out of every total). To add runs
  to the analysis, add their episode folders there.

## An episode folder

`data/episodes/<task>/<model>/<stamp>_seed<k>_<run hash>/`, written by `core/publish_episode.py` when the evaluator
finishes a seed. Files are only ever added to it.

| File | What |
|---|---|
| `episode.json`, `verdict.json` (how the episode ended), `cameras_states.npz`, screenshots, videos | what the evaluator wrote for the seed |
| `claude.log` / `codex_session.jsonl`, `sim.log` | the agent's transcript and the simulator log |
| `meta.json` | task, model, interface, the launch it came from |
| `subtasks.json` | the grade, done once when the episode ends by `core.grade_episode` with the task's `grade.py`: each subtask done or not, the subtask score, solved |
| `rubric_score.json` | the older milestone grade, on episodes graded before `grade.py` |
| `summary.json` | score, solved, cost, turns, tool calls, EEF commands, wall time: what the Analysis tab averages |
| `step_score.json` | the paper's step scores of the 660 Table 1 episodes, kept as the reference their `subtasks.json` came from |
| `recovered_from/` | the attempt before a resume, which the summary adds in |

The score in `summary.json` is the subtask score from `subtasks.json`, and solved means a score of 1.
