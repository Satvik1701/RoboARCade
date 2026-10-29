# Gemini through Antigravity

New SDK evaluations explicitly set Gemini's thinking level. The evaluator's
default `--effort xhigh` maps to Argon's highest supported level, `high`;
`--effort low`, `medium`, and `high` select those levels directly, and `max`
also maps to `high`. This selects each provider's highest supported setting;
it does not imply identical reasoning compute across providers. The requested
effort and actual thinking level are saved in run and episode metadata and the
SDK startup event. A missing SDK thinking setting fails instead of silently
using a provider default.

**Historical cohort:** the 110 website runs launched before this correction
omitted the thinking level. They used Argon's documented API default, **medium**,
and must remain labeled medium. Run high evaluations in a new output directory;
never reuse the old scores as high results. The SDK request-capture check confirmed
that the old configuration omitted `thinkingConfig`, while the corrected one sends
`thinkingConfig.thinkingLevel: "high"`. See Google's
[thinking documentation](https://ai.google.dev/gemini-api/docs/thinking/).

For an explicit, verified 245k compaction setting, use the
[SDK setup](#sdk-backend-with-a-verified-245k-threshold) below. The SDK is opt-in
through `ROBOARCADE_ANTIGRAVITY_BACKEND=sdk`; episode metadata records the backend.

## Native CLI backend

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

## SDK backend with a verified 245k threshold

The opt-in SDK backend pins `google-antigravity==0.1.22` and explicitly supplies
`CompactionConfig(token_threshold=245000)`. It uses the native Antigravity local
harness, API-key billing, the exact requested model, explicit thinking configuration,
and a 65,536-token maximum response. SDK reasoning is explicitly `high` by default;
the historical model-default behavior described in earlier smoke results was medium
for Argon. The CLI remains the default backend.
The SDK requires its own virtual environment so its dependencies do not change
the simulator environment. Use the same base Python as the evaluator:

```sh
uv venv --python "$(readlink -f .venv/bin/python)" data/antigravity-sdk
uv pip install --python data/antigravity-sdk/bin/python google-antigravity==0.1.22
export ROBOARCADE_AGY_PYTHON="$PWD/data/antigravity-sdk/bin/python"
export ROBOARCADE_ANTIGRAVITY_BACKEND=sdk
export ROBOARCADE_AUTO_COMPACT_WINDOW=245000

.venv/bin/python -m core.run_eval \
  --model gemini-4-argon --task bowl_on_plate --interface via_cam \
  --seeds 2 --max-parallel 2 --effort high --wall-clock 3600 --no-video
```

The SDK uses the same outer sandbox, robot MCP interface, interface guide,
task prompt, simulator, and grader. It exposes robot tools eagerly and permits
only the image viewer for saved observations, with a path guard that resolves
symlinks. Shell, web, subagents, and file-writing tools are disabled. Sessions
are isolated per seed; a restart resumes the saved conversation. Model usage,
thinking tokens, checkpoints, and raw SDK steps are archived with the episode.
The SDK does not expose an exact API-request count, so `num_turns` is unknown;
`completed_model_steps` is reported separately. Cached tokens are counted once,
and thinking tokens are a subset of total output tokens.

**245k is the native harness's trajectory-token estimate, not the Gemini API's
prompt-token count.** A live test on 2026-10-20 produced a completed checkpoint
at the 245k setting; the identical workload at 350k did not compact. The model
recalled a marker present only in an earlier user message both after compaction
and after a process restart. API prompt usage dropped from roughly 72k before
the checkpoint to 26k on the recall turn; the control reached roughly 96k.
This verifies the configured native threshold and retention, not a 245k API
prompt boundary. Reproduce the live test (billable API requests) with:

```sh
"$ROBOARCADE_AGY_PYTHON" scripts/verify_antigravity_compaction.py \
  --output /tmp/antigravity-compaction-verification
```

Each output directory must be new. Reports include the checkpoint IDs, token
usage, marker answers, and resumed conversation ID. State should live on a local
filesystem because native session locking can stall on network filesystems.

The tested native CLI 1.4.3 did not honor `AGY_CLI_AUTO_COMPACT_OVERRIDE` in a
live low-threshold probe. Do not use that variable to claim matched compaction.
The SDK is a separately recorded backend; it still uses native compaction
instructions instead of RoboARCade's shared compaction prompt. The earlier CLI
smoke scores remain CLI results and do not validate SDK task performance.
