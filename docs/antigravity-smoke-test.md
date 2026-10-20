# Gemini Argon smoke test — 2026-10-20

## One-hour rerun

The one-hour rerun completed with **1/4 task completions**. Bowl-on-plate seed
1000000 declared success and received score 1.0; the other three seeds reached
the wall-clock limit. Sandbox integrity remained intact in all four episodes.

| Task | Seed | Result | Simulation steps | Elapsed seconds, including shutdown |
| --- | ---: | --- | ---: | ---: |
| bowl_on_plate | 1000000 | Success, score 1.0 | 181 | 1324.7 |
| bowl_on_plate | 1000001 | Timeout | 88 | 3655.4 |
| bottles_in_bin | 1000000 | Timeout | 384 | 3635.1 |
| bottles_in_bin | 1000001 | Timeout | 104 | 3634.7 |

These runs launched with runtime commit `43105b6`, sandboxing and Blender enabled,
the same interfaces and parallelism as below, and a 3600-second agent deadline.
They do not test a new 245k Antigravity compaction configuration.

The unsuccessful bowl seed did not declare completion before its deadline.
Its native event log records four model responses lasting 509.9, 535.6, 524.0,
and 528.3 seconds, totaling about 35 minutes. All completed model-response
durations sum to about 54 minutes. These are response durations, not
time-to-first-token measurements or a matched comparison with Astra.

The four long responses each reached 65,536 output tokens, including 65,533
reported thinking tokens, and each was followed by an output-limit error and
continuation. Together they generated 262,144 output tokens in 2,097.8 seconds
(122–129 tokens/second per response, including waiting time). The other 36
completed responses had a median duration of 21.3 seconds. Native stream usage
records 324,508 thinking tokens across all 40 completed responses; thinking is
already included in output tokens and must not be added again. Earlier episode
summaries omitted this thinking breakdown on timeout. These logs do not separate
first-token waiting from generation time.

Artifacts:

- `data/run_eval/gemini_argon_smoke_bowl_v6_20261020/20261020_082123/`
- `data/run_eval/gemini_argon_smoke_bottles_v6_20261020/20261020_082119/`

## Initial 20-minute smoke test

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
