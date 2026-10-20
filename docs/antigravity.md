# Gemini through Antigravity

Use the native `agy` CLI (tested with 1.4.2) and Gemini API-key billing. This is
separate from Google's `gemini` CLI. Install `agy` using Google's
[official instructions](https://www.antigravity.google/docs/install?tab=cli).

Put the key in the gitignored `secrets/gemini_key.txt`, set `GEMINI_API_KEY`, or
point `ROBOARCADE_GEMINI_KEY_FILE` at a dedicated key file. Do not put the key in
commands, MCP configuration, or committed files.

```sh
.venv/bin/python -m core.run_eval \
  --model gemini-4-argon --task bowl_on_plate --interface via_cam \
  --seeds 2 --max-parallel 1 --wall-clock 3600 --no-video

.venv/bin/python -m core.run_eval \
  --model gemini-4-argon --task bottles_in_bin --interface via_bimanual_cam \
  --seeds 2 --max-parallel 1 --wall-clock 3600 --no-video
```

These are smoke tests with a one-hour cap per episode. Benchmark runs use the
usual budget instead. The renderer defaults to Blender. State recording and
host-side grading follow the existing evaluator; an API or runtime failure is
left ungraded. A timeout remains a timeout, not a successful declaration.

See the [2026-10-20 smoke results](antigravity-smoke-test.md) for the completed
two-task, two-seed trial and its limitations.

Antigravity runs inside the same bubblewrap filesystem and network boundary as
Claude and Codex. It sees the selected read-only interface and MCP imports, its
own screenshots, and fresh per-seed CLI state. Task definitions, grading code,
assets, other runs, credential files, and personal CLI configuration are hidden.
Network access is limited to the simulation port stack and the Gemini API host.
The key is passed through the agent environment, never mounted as a credential
file or passed to the simulator.

Generated settings select `modelProvider: gemini` and register the exact model
ID as a custom model because agy 1.4.2's API-key catalog does not yet list Argon.
There is no fallback to another model. Reasoning uses the model default; the
custom-model route does not support agy's catalog-based `--effort` flag. The
evaluator rejects `--effort`, subscription `--plan`, and `--no-isolated` for this
runtime rather than mislabeling the run. Episode metadata records the runtime
and reasoning policy, and native JSONL events preserve the conversation.

The custom main agent excludes default components and receives the interface
guide. Robot MCP calls and `view_file` for its own saved observations and cached
robot tool schemas are allowed;
shell, file writes, and web actions are denied. The image viewer is necessary
because agy offloads MCP images to files. No task or grader files are exposed.
Resumes keep the same Antigravity conversation and per-seed state. Live CLI
state lives under `/tmp` because its state-file locking stalls on NFS; it is
archived into the episode's `antigravity/` directory when the agent stops.
Three rejected MCP calls before any successful MCP call stop the run as an
ungraded infrastructure error. Provider-reported usage is recorded, including
completed requests before a timeout; dollar cost is left unknown.

## Comparison with the existing runners

This integration is not yet a matched-configuration benchmark against Codex or
Claude. The shared evaluator supplies the same task prompt, selected interface
guide and robot tools, task seed, simulator settings, simulation-step budget,
and grader. The interface and task files are unchanged from public main.
All three runtimes use RoboARCade's outer filesystem and network sandbox, with
provider-specific configuration and API access.

The remaining differences matter when interpreting results:

| Setting | Codex / Claude runners | Antigravity runner |
| --- | --- | --- |
| Task prompt | Shared `build_prompt` | Same function |
| Interface guide | Codex `AGENTS.md`; Claude appended system instructions | Same guide inside custom agent instructions, plus image/schema access instructions |
| Reasoning | Explicit `xhigh` by default, overridable | Model default; custom-model effort override is unsupported by the tested CLI |
| Compaction instructions | Shared `core/compaction/prompt.txt` | Not installed into native compaction |
| Compaction target | Both target 245,000 tokens, unless overridden. Codex receives 245,000 directly; Claude receives `autoCompactWindow` 278,000 to compensate for its 33,000-token reserve | Native behavior; no matching target configured |
| Observation delivery | Robot MCP responses | Robot MCP responses plus native `view_file` for offloaded images and schemas |
| Episode wall clock | Default 21,600 seconds, unless overridden | Same evaluator default; current smoke command explicitly uses 3,600 seconds |
| Failure recovery | Existing provider/auth retry and API-stall handling | Separate Antigravity driver; runtime errors are left ungraded |

The full prompts also include each CLI's native instructions; sharing task and
interface text does not make those native prompts identical. Antigravity's
automatic context checkpoints are described in its
[context documentation](https://antigravity.google/docs/cli/commands/context).
No equivalent custom compaction prompt/threshold control has been verified for
the tested CLI integration. SDK compaction controls are not evidence that the
native CLI accepts the same settings. Do not label these smoke runs as having
matched compaction or reasoning settings.

Run integration tests with:

```sh
.venv/bin/python -m pytest tests/test_antigravity.py -q
```
