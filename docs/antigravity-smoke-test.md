# Gemini Argon smoke test — 2026-10-20

The fresh two-task, two-seed smoke test completed with **0/4 task completions**.
All four episodes reached the 20-minute wall-clock cap. Robot tool access,
image viewing, and sandbox integrity worked; three episodes executed robot
waypoints, while one completed observations without executing a waypoint.

Configuration: runtime commit `948fe10`, native Antigravity CLI 1.4.2, exact
model `gemini-4-argon`, API-key authentication, default dynamic reasoning,
Blender rendering, state recording, and sandboxing enabled. Replay video
generation was disabled. Both tasks ran concurrently, with two parallel seeds
per task. `bowl_on_plate` used `via_cam`; `bottles_in_bin` used
`via_bimanual_cam`.

| Task | Seed | Result | Completed waypoint calls | Final simulation steps |
| --- | ---: | --- | ---: | ---: |
| bottles_in_bin | 1000000 | Timeout | 4 | 152 |
| bottles_in_bin | 1000001 | Timeout | 4 | 144 |
| bowl_on_plate | 1000000 | Timeout | 4 | 88 |
| bowl_on_plate | 1000001 | Timeout | 0 | 10 |

No episode called `end_episode`; the harness stopped each at its deadline.
Simulation steps include harness settling and do not alone establish agent
motion. Sandbox integrity was verified for all four episodes.

Each bowl episode had one completed model response reach 65,536 output tokens,
with long pauses between tool actions. This is a likely contributor to the
timeouts, not evidence of a disconnected simulator. These short trials do not
establish full-budget benchmark performance. API cost is unknown.

Earlier integration attempts are excluded. They exposed NFS locking of live CLI
state and missing access to offloaded images and cached tool schemas. The tested
runtime keeps live state on local `/tmp`, archives it after exit, and permits
the native viewer to read the agent's observations and robot schemas.

Validation: nine integration tests passed; all 24 sandbox probes passed.
A live simulator preflight read tool schemas, called the camera and pose tools,
and viewed the returned images. The first and last archived observation images
from every smoke episode decoded successfully.

Local artifacts, relative to the checkout:

- `data/run_eval/gemini_argon_smoke_bowl_v5_20261020/20261020_074951/`
- `data/run_eval/gemini_argon_smoke_bottles_v5_20261020/20261020_074951/`

Each root contains `seed1000000` and `seed1000001` episode folders with native
events and archived CLI state. Artifacts are local and are not included in Git.
The implementation remains on `gemini-argon-antigravity`; no merge to main or
full-task evaluation was performed.
