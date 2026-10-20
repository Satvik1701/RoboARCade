# Gemini Argon smoke test — 2026-10-20

## SDK verification and fresh smoke test

The opt-in SDK backend uses `google-antigravity==0.1.22`, the exact
`gemini-4-argon` model, API-key billing, model-default reasoning, a 65,536-token
response limit, and explicit native compaction at 245,000 estimated trajectory
tokens. The native CLI remains a separately labeled backend.

Verification completed:

- A real checkpoint occurred at the 245k setting. An identical 350k control did
  not compact. A marker present only in the first user message survived both
  compaction and a process restart. This measures Antigravity's native token
  estimate, not an exact Gemini API prompt-token boundary.
- All 25 sandbox probes passed. A robot preflight viewed screenshots and resumed
  the same conversation in another process, with correct cumulative token usage.
- All 28 task environments constructed, and all 56 task/seed initializations
  succeeded for seeds 1000000 and 1000001. This checks setup, not agent performance.
- Fifteen regression tests passed, including path traversal rejection, timeout
  usage accounting, and delayed network-bridge startup.

The fresh smoke uses Blender, unchanged task/interface definitions, sandboxing,
and a one-hour deadline per episode. Both bowl seeds scored 1.0 and passed
integrity checks: seed 1000000 took 601.0 seconds and seed 1000001 took 744.5
seconds (including shutdown). Both bottle seeds reached the one-hour deadline:
seed 1000000 scored 0.50 (two of four bottles, 3635.6 seconds including shutdown)
and seed 1000001 scored 0.75 (three of four bottles, 3638.7 seconds).
All four scored episodes passed integrity checks. The first and last archived
observation images from each episode decoded successfully. This is 2/4 full
task completions; the SDK integration is verified, but the smoke does not show
perfect task performance. No full-task suite or main-branch merge has occurred.

The first bottles seed 1000000 attempt failed its API-connection preflight
before making an API request and was left ungraded. The sandbox now waits for
host bridge sockets and retries transient connection-check failures. The same
seed was restarted in a separate retry directory; the original failure remains
preserved. No manipulation failure was retried or discarded.

Artifacts:

- `data/run_eval/antigravity_sdk_verification_20261020/`
- `data/run_eval/gemini_argon_sdk_smoke_bowl_20261020/20261020_134322/`
- `data/run_eval/gemini_argon_sdk_smoke_bottles_20261020/20261020_134322/`
- `data/run_eval/gemini_argon_sdk_smoke_bottles_retry_20261020/20261020_134534/`

The initial SDK smoke launched at `f7b0b58`; the network-startup retry launched
at `6200566`. A later accounting-only fix (`90cb035`) includes restored SDK
history in completed-step counts. Exact API request count remains unknown and
is not inferred from tool-step counts. See [setup](antigravity.md) for the opt-in
SDK command and the remaining differences from Claude/Codex.

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
