# RoboARCade

A benchmark of 22 long-horizon manipulation tasks in which general agents (Claude Code, Codex, Antigravity) control simulated
robot arms through tools. Videos and full traces of every episode: [roboarcade.org](https://roboarcade.org).

## Quickstart: Multi-view RGBD

Multi-view RGBD is the default interface for all supported model runtimes and both
single-arm and two-arm tasks. You need Linux x86_64 with an EGL-capable GPU driver,
Git, and curl. Setup installs missing system packages through apt/sudo when needed;
the sandbox requires unprivileged user namespaces.

```bash
git clone https://github.com/Satvik1701/RoboARCade.git roboarcade
cd roboarcade
mkdir -p secrets
```

Put your HF token and your model provider's API key in these files (plain text, one
key per file). Only one model key is needed; you can add others later.

| Key | File |
|---|---|
| Hugging Face | `secrets/hf_key.txt` |
| Anthropic / Claude | `secrets/anthropic_key.txt` |
| OpenAI / GPT | `secrets/openai_key.txt` |
| Google / Gemini | `secrets/gemini_key.txt` |

Then run one setup script:

```bash
bash scripts/setup.sh
```

It installs Python dependencies and the agent runtimes for the keys you supplied,
configures API-key authentication, and downloads the finished HF assets. `secrets/`
is gitignored. No subscription login is needed. Rerun setup after adding another
provider key.

Start testing with a model ID your API key can access:

```bash
source set_env.sh
python -m core.run_eval --task bowl_on_plate --model MODEL_ID --seeds 1 --render mujoco --no-video
```

Replace `MODEL_ID` with your Claude, GPT, or Gemini model ID. Change `--task` to
another folder under `tasks/`. This uses MuJoCo camera rendering, records and grades
the rollout, and skips the extra Blender video replay. No browser or Blender is
needed. API-key billing is the default; [subscription plans](docs/ACCOUNTS.md) are optional.

To view runs, transcripts, recordings, and grades:

```bash
python -m dashboard                       # http://localhost:8400
```

## Testing VIA

VIA additionally needs Chromium and its Linux system libraries. The same setup
script installs them with `--via` (system packages may require sudo):

```bash
bash scripts/setup.sh --via
source set_env.sh
python -m core.run_eval --task bowl_on_plate --model MODEL_ID --interface via_cam --seeds 1 --render mujoco --no-video
```

For two-arm tasks, use `--interface via_bimanual_cam`. The keys and sandbox are the same.

## Blender rendering

The paper's evaluations use Blender for camera feeds and replay videos. Install it once:

```bash
mkdir -p tools
curl -L https://download.blender.org/release/Blender4.5/blender-4.5.12-linux-x64.tar.xz | tar -xJ -C tools
tools/blender-4.5.12-linux-x64/4.5/python/bin/python3.11 -m pip install --target tools/blender-py --no-deps \
    mujoco==3.10.0 pillow typing_extensions etils absl-py glfw
python -m core.run_eval --task bowl_on_plate --model MODEL_ID --seeds 1
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
