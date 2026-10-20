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
  --seeds 2 --max-parallel 1 --wall-clock 1200 --no-video

.venv/bin/python -m core.run_eval \
  --model gemini-4-argon --task bottles_in_bin --interface via_bimanual_cam \
  --seeds 2 --max-parallel 1 --wall-clock 1200 --no-video
```

These are smoke tests with a 20-minute cap per episode. Benchmark runs use the
usual budget instead. The renderer defaults to Blender. State recording and
host-side grading follow the existing evaluator; an API or runtime failure is
left ungraded. A timeout remains a timeout, not a successful declaration.

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

Run integration tests with:

```sh
.venv/bin/python -m pytest tests/test_antigravity.py -q
```
