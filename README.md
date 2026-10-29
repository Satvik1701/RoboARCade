# RoboARCade

A benchmark of 22 long-horizon manipulation tasks in which general agents (Claude Code, Codex) control simulated
robot arms through tools. Videos and full traces of every episode: [roboarcade.org](https://roboarcade.org).

## Quickstart

You need Linux with an EGL-capable GPU driver, [uv](https://docs.astral.sh/uv/), Node.js (only to install the agent
CLIs below), and bubblewrap: every agent runs in a sandbox ([`docs/SANDBOX.md`](docs/SANDBOX.md)). Install it with
`sudo apt install bubblewrap`, or without sudo point `ROBOARCADE_BWRAP` at a `bwrap` binary; either way
`unshare -Urn true` must succeed.

**1. Install**

```bash
git clone https://github.com/Satvik1701/RoboARCade.git roboarcade && cd roboarcade
uv sync                                   # dependencies into .venv
source set_env.sh                         # in every new shell
python -m playwright install chromium     # the browser the agent's interface runs in
```

**2. Assets and Blender**

```bash
python scripts/sync_assets.py pull        # task meshes and textures from Hugging Face, about 1.3 GB
mkdir -p tools && curl -L https://download.blender.org/release/Blender4.5/blender-4.5.12-linux-x64.tar.xz | tar -xJ -C tools
tools/blender-4.5.12-linux-x64/4.5/python/bin/python3.11 -m pip install --target tools/blender-py --no-deps \
    mujoco==3.10.0 pillow typing_extensions etils absl-py glfw
```

The GitHub task scenes define the final collision geometry. The HF download supplies
the finished meshes, textures, robot models and reachability grids. Pulling these assets
is sufficient: installation and evaluation never rebuild or rewrite them.

Blender renders the agent's camera feeds and the episode video, as in the paper.

**3. Credentials: an Anthropic, OpenAI, or Gemini API key, or your plan**

Claude models run through Claude Code and GPT models through Codex. Install the one you will use, unless `claude`
or `codex` is already on your PATH:

```bash
npm install -g @anthropic-ai/claude-code   # Claude
npm install -g @openai/codex               # GPT
```

| You have | Do once | Then add to the run |
|---|---|---|
| Anthropic API key | `mkdir -p secrets && echo "sk-ant-..." > secrets/anthropic_key.txt` | nothing |
| OpenAI API key | the three lines below | `--model gpt-6-astra --plan openai` |
| Claude plan | `python -m core.accounts login claude NAME` | `--plan NAME` |
| ChatGPT plan | `python -m core.accounts login codex NAME` | `--model gpt-6-astra --plan NAME` |

```bash
mkdir -p secrets/codex_auth/openai/store
printenv OPENAI_API_KEY | CODEX_HOME=secrets/codex_auth/openai/store codex login --with-api-key
python -m core.accounts login codex openai
```

Gemini models run through the native Antigravity `agy` CLI. For `gemini-4-argon`,
put your API key in `secrets/gemini_key.txt` and add `--model gemini-4-argon`.
See [Antigravity setup and smoke tests](docs/antigravity.md) for installation and sandbox details.

Everything goes under `secrets/`, which is gitignored. More in [`docs/ACCOUNTS.md`](docs/ACCOUNTS.md).

**4. Run one seed**

```bash
python -m core.run_eval --task bowl_on_plate --seeds 1     # Claude Opus 5 on your Anthropic key
```

Add the options from the table for another credential. One seed takes about 20 to 30 minutes; on an API key it costs
a few dollars (our test runs cost $7 to $8). `--model` takes `opus`, `sonnet`, `haiku`, `fable` or a GPT
model name such as `gpt-6-astra`.

**5. Watch it**

```bash
python -m dashboard                       # http://localhost:8400
```

The dashboard lists the episode with its video, the agent's transcript and screenshots, and the grade.

Also: `python -m core.sim_env_server --task bowl_on_plate --teleop` drives a task yourself in the browser, and
`--render mujoco --no-video` runs a seed without Blender. Task packs are the folders under `tasks/`; their meshes and
textures are the Hugging Face dataset [`roboarcade/roboarcade-assets`](https://huggingface.co/datasets/roboarcade/roboarcade-assets).

### Asset credits

The meshes, textures and robot models in [`roboarcade/roboarcade-assets`](https://huggingface.co/datasets/roboarcade/roboarcade-assets)
come from these projects. We thank their authors; each project's own license and terms apply to its files.

| Source | Tasks | What we use | License |
|---|---|---|---|
| [AutoBio](https://github.com/autobio-bench/AutoBio/) | PCR setup, Start incubation, Vortex sample | Digitized lab instruments and consumables. You can also get them from the AutoBio repository. | see the AutoBio repository |
| [Fabrica](https://github.com/yunshengtian/Fabrica) | Beam, Cooling manifold, Gamepad, Stool | Assembly parts | MIT |
| [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | Serve warm drinks, Start making breakfast, Tidy counter | Kitchen scenes and objects | MIT |
| [OGBench](https://github.com/seohongpark/ogbench) | Lights Out (6 by 7) | The puzzle | MIT |
| [SPD](https://spd.bot) | Toss bottles in bin, Cup pyramid, Handover dominos, Jenga tower, Hang mugs and rack plates | Bimanual VR teleoperation scenes and objects | courtesy of the SPD authors |
| [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo) | Strongbox, Organize table and plug in charger, Pack tools into toolboxes, Sort Lego bricks | Task objects | MIT |
| [ABC](https://github.com/amazon-far/abc) | Sort Lego bricks | Lego brick models | Apache-2.0 |
| [robomimic](https://github.com/ARISE-Initiative/robomimic) | Square nut assembly | The robosuite nut assembly scene | MIT |
| [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) | all Panda tasks | Franka Emika Panda | Apache-2.0 |
| [I2RT](https://github.com/i2rt-robotics/i2rt) | all YAM tasks | YAM arm, via MuJoCo Menagerie | MIT |

### Docs

The docs site (code map, sandbox, accounts, tasks) is static HTML built from the repository:

```bash
.venv/bin/python scripts/code_map.py && .venv/bin/python docs/interfaces/build.py
.venv/bin/python docs/architecture/task-rubrics/build.py && .venv/bin/python docs/build.py
cd docs/site && python3 -m http.server 8099      # http://localhost:8099/
```
