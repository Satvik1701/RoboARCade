# RoboARCade

A benchmark of 22 long-horizon manipulation tasks in which general agents (Claude Code, Codex) control simulated
robot arms through tools. Videos and full traces of every episode: [roboarcade.org](https://roboarcade.org).

## Quickstart: Multi-view RGBD

Multi-view RGBD is the default interface for both single-arm and two-arm tasks.
This quick start uses a Claude API key and MuJoCo camera rendering; no browser or Blender installation is needed.

You need Linux with an EGL-capable GPU driver, [uv](https://docs.astral.sh/uv/), and
[bubblewrap](docs/SANDBOX.md) (`sudo apt install bubblewrap`). The sandbox requires
`unshare -Urn true` to succeed.

**Install once:**

```bash
git clone https://github.com/Satvik1701/RoboARCade.git roboarcade
cd roboarcade
uv sync
source set_env.sh
curl -fsSL https://claude.ai/install.sh | bash   # skip if claude is already installed
export PATH="$HOME/.local/bin:$PATH"
```

**Set your keys, download the assets, and run:**

```bash
export ANTHROPIC_API_KEY="your-model-api-key"
export HF_TOKEN="your-hugging-face-read-token"
python scripts/sync_assets.py pull
python -m core.run_eval --task bowl_on_plate --seeds 1 --render mujoco --no-video
```

This runs Claude Opus 5 with Multi-view RGBD in the sandbox, records the rollout, and grades it.
The asset download is needed once; in a new shell, run `source set_env.sh` and set your model key again.
`--no-video` skips the extra Blender replay. For the paper's renderer, see [Blender rendering](#blender-rendering).

Change `--task` to another folder under `tasks/`, or add `--model` to select a model.
For OpenAI API keys and subscription plans, see [Accounts](docs/ACCOUNTS.md);
for Gemini, see [Antigravity setup](docs/antigravity.md).

To view runs, transcripts, recordings, and grades:

```bash
python -m dashboard                       # http://localhost:8400
```

## Testing VIA

VIA additionally needs Chromium and its Linux system libraries. Install them with
[Playwright](https://playwright.dev/python/docs/browsers#install-system-dependencies)
(the system-library step may require sudo), then select the VIA interface:

```bash
python -m playwright install --with-deps chromium
python -m core.run_eval --task bowl_on_plate --interface via_cam --seeds 1 --render mujoco --no-video
```

For two-arm tasks, use `--interface via_bimanual_cam`, for example with `--task bottles_in_bin`.
VIA runs headlessly and uses the same keys, assets, and sandbox as Multi-view RGBD.

## Blender rendering

The paper's evaluations use Blender for camera feeds and replay videos. Install it once:

```bash
mkdir -p tools
curl -L https://download.blender.org/release/Blender4.5/blender-4.5.12-linux-x64.tar.xz | tar -xJ -C tools
tools/blender-4.5.12-linux-x64/4.5/python/bin/python3.11 -m pip install --target tools/blender-py --no-deps \
    mujoco==3.10.0 pillow typing_extensions etils absl-py glfw
python -m core.run_eval --task bowl_on_plate --seeds 1
```

Blender remains the default renderer when `--render` is omitted. It works with either interface.
The GitHub task scenes define the final collision geometry; the
[Hugging Face assets](https://huggingface.co/datasets/roboarcade/roboarcade-assets)
supply the finished meshes, textures, robot models, and reachability grids.
Installation and evaluation never rebuild or rewrite these assets.

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
