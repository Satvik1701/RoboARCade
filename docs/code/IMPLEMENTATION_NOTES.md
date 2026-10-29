# Implementation notes

The quirks behind the code (units, rationale, gotchas, invariants a name cannot carry), by folder and file in the code map's order.

## `core/`

### `core/UI/index.html`

- UI units are robot metres times ten; robot y lies along UI minus z and robot z along UI y; the UI origin sits at robot z equal to ui_z_offset (the tabletop).
- After ui_config.json loads, the main script is injected as a classic script so its top level bindings stay visible to page.evaluate.
- The target's jaw mesh follows the commanded jaw state, never the measured width, since a held object can prop the jaws wide enough to read as open.
- The open and closed gripper OBJ files share topology, so the closed target lerps to the real jaw width.
- Each waypoint payload carries only the selected gripper because sim_env_server consumes the last entry of a message.
- home is true only when the target has not moved since homeArm put it there: homeTargetKeys holds the raw position and quaternion, renderFrame drops a key once the target differs (every tool call renders, so an edit and its exact undo cannot both slip between frames), and each send clears it. Only gripper_home followed by Execute gets the joint space return.
- A socket drop reconnects in place while /instance.json still names this sim, since a reload would reset the target and drop what it holds.
- Rendering runs only within half a second of input or when the scene changed, and MSAA is teleop only: on software WebGL an unconditional cadence or an antialiased frame starves every CDP call.

### `core/sim_env_server.py`

- Ports run up from ROBOARCADE_BASE_PORT (default 8100) so several stacks can run at once: +0 the browser page (CSP confined, not with --no-ui), +1 the broadcast WebSocket and the JSON endpoints (/env.json, /robot.json, /rgbd.json with --rgbd, /ui_config.json, /reachable, /cam/*), the simulator's API, the same for every interface, +2 the command WebSocket (the page's waypoints, robot-frame waypoints, {"done": true}), +3 Chromium CDP (not with --no-ui); kernel.py and run_seed.py read the same variable.
- FINAL_SNAP_HOLD_S: after the verdict, time for the MCP execute tool's final screenshot before Chromium goes.
- _empty sends the CORS header even on 403, else the browser hides the status and the page's "give up on 403" checks never fire.
- _ui_euler_to_robot: the UI frame is the robot frame rotated minus 90 degrees about x (UI y = robot z, UI z = minus robot y), and the UI gripper's identity pose is the robot eef frame rolled pi about x. transform_robotframe_to_uiframe and back: one UI unit is 0.1 m, and _ui_z_offset is the tabletop height in metres the UI maps its z=0 table to and from.
- _record_state banks the sim state per frame (about 2 kB per frame); teleop always banks state so a demo is replayable.
- _task_end_time is latched when the episode ends; the final render is not the agent's time.
- _ended_by is one of agent, ui, harness, sim or scene; only agent is the model's own claim that it is finished. _end_requested is SIGUSR1 from run_seed: the agent stopped without end_episode, the episode ends without the settle and is never graded as the agent's own claim (ended_by harness).
- _episode_over is served on /env.json and the tools stop on it; it is set at the terminal decision, before the last rerender, so a tool returning mid-render sees the episode over. frame_ready on /env.json: the ports answer before the first (Blender, about 2 min) frame.
- _hold_render keeps a forced Blender preview from being repainted by the 1 Hz idle MuJoCo stream; anything but another preview moves the scene, so the held render is then stale.
- _probe_pool has a single worker: concurrent probes on the shared scratch MjData segfault.
- Thread safety: server threads never call mj_forward or mj_step or write env.data; frames are built in the sim thread (_send) and routes read plain attributes and the per-tick _qpos snapshot (also where the reachability probe starts its IK). get_point_cloud's cam_xpos and cam_xmat read for static cameras is racy but benign (they never move). Only the sim thread puts on self.frames, so _send's full() check stays valid.
- Camera feed labels (agentview, wrist, wrist_<robot>) are stable interface roles; compiled camera names stay internal. Intrinsics are fixed per scene; the wrist extrinsics change per frame.
- verdict.json is dumped with default=str so a stray numpy scalar cannot lose the file the eval driver grades from; success_spec is copied in because packs are not versioned with the code; init_state is per-arm eef plus joint angles.
- _read_init_poses reads at reset because by episode end _live_xyz reports where things ended up.
- Teleop starts objects at their base pose; its Randomize init toggle re-rolls live.
- probe_reachable: the solver and scratch data depend only on the model and are built once per arm. A held object (the free root of an env.held_bodies body) moves with the hand; left at its stale pose it reads as a collision, so held payloads are placed at their hand-relative endpoint pose before re-colliding, and only new contact pairs count (the grasp itself is the current state). mj_forward also runs collision, filling d.contact.
- prepare_point_cloud subsamples with linspace because the points are in raster order: stride plus truncate would drop a spatial band. Bytes rather than float lists are about 100x faster.
- collect_workspaces and collect_init_regions: reach masks per object at its own height (the footprint changes several cm per cm of height) on one absolute 2 cm lattice, the reach voxel grid's resolution, shared with the workspace slice; one mask per group, keyed by its first member and expressed about the group's pivot (a group translates rigidly); under full randomization the sampler draws from the whole legal area, not the declared range; cells are tested in absolute coordinates because offsetting and re-adding loses the last bit and shifts boundary cells; with declared stations only their cells are shown, tagged by station index as a third element (-1 for none); one slice per object height, not one mean-height layer; the support rect and the point-cloud crop are both hard placement constraints. A saved-state base pose may float above the surface, so the resting z comes from rest_z.
- listen_ui: the last waypoint of a batch is taken; inputs and trace frames bypass the cmds gate so they are never dropped and are tagged with the waypoint being set up (last + 1). A second Execute while one is running is dropped. Wire contract: a url ending in <gripper>.obj (not _closed) means open. The [ui_cmd] log line's t= is epoch seconds.
- last_update_msg is replayed to a (re)connecting client; update_ui_seq dedupes it page-side. robot.json's update_seq counts observation updates; a waypoint ends with one, published after its RGB-D snapshot, and interfaces/rgbd_waypoint polls it.
- /objects serves each free object's live pose as [x, y, z, qw, qx, qy, qz], to bake into a scene. The ground truth routes (/objects, /success_spec, /success_state, /init_regions) exist for the human teleop scene only: in agent runs interface code could read exact poses and trivialize the task.
- /reachable with a malformed query answers reachable: None and the page tints nothing. A path with no handler falls through to the normal WebSocket handshake; the query part carries /reachable's arguments and the page's cache-buster.
- broadcast_loop: under Blender (or a held preview) only rendered frames reach the feeds and the /cam cache, else the 1 Hz idle MuJoCo observe repaints them. An empty cloud (skip_pcl) makes the page keep its last geometry (2.4 MB per frame otherwise). rendered_colors: Blender colours are tone-mapped and the page's midtone lift is for MuJoCo. gripper_cmd_open is commanded, not measured: a held object holds the jaws past 0.9. A None frame is the teardown wake-up: a thread parked in get() blocks interpreter exit. Non-feed cameras are JPEG encoded on demand for /cam.
- websockets.serve: max_size=None because a trace-frame JPEG can exceed the 1 MiB default; ping_timeout=120 because a loaded machine can starve pings past the 20 s default. teardown's put(None, timeout=5) wakes the broadcast thread's get() and times out if the loop is dead.
- apply_waypoint_mode: a waypoint the page flags home: true is the home waypoint, straight when green and joints when red; any other waypoint is a move_to even when it lands on the home pose. An absent or unknown arm means the default (move) arm.
- _bank_frame records ActMode.Dense, which extends the current waypoint's span.
- _save_task_demo transcodes to H.264 because browsers cannot play mp4v; the file is the dashboard's hover preview.
- UNDO_DEPTH is snapshots kept; one is pushed before every waypoint (the E key reverts). A snapshot carries the recorder's span count too: a waypoint run while recording was paused adds no span, so an undo must not discard the span before it. Undo restores q_des as well (else the actuators drag the arm toward the undone goal) and restore's dynamics hook repaints model side presentation before the render.
- _blender_preview: teleop streams MuJoCo because Blender takes seconds per frame (much longer without a GPU); the preview shows what the scene looks like to a scored agent, and the first call also pays the worker start and the USD export.
- _reroll_init is a scene edit, not a waypoint; a changed station band restarts the systematic walk; redeal_episode re-draws any per-episode randomness of the task.
- The SIGTERM handler finalizes the media first: a SIGTERMed stack keeps its video (else cameras.mp4 has no moov atom) but gets no verdict.
- record_demo: the first Blender render runs after the servers are up (it can take minutes on CPU); the untouched scene is banked so an episode with no waypoint still has a video. The hold at the end: 0.4 s after a teleop End Episode click so the last update frame reaches the page before the socket close reloads it onto the dashboard; 3 s when teleop ended by the sim or the horizon so the person reads the outcome first; FINAL_SNAP_HOLD_S for an agent run.
- one_episode: the phase is SETUP during reset so its idle or reorient steps cannot end the episode. Between waypoints only the feeds are streamed (no depth, no cloud); a state change sends its own update frame. Teleop-only commands (undo, reroll, restart, record controls, blender_render) are ignored in agent episodes: undo and reroll would be free retries and a forced render would buy Blender frames a mujoco eval withheld. The scene itself can end the episode without a declaration (ended_by scene or sim); a timed out episode does not get its last subtask. inputs.msgpack is written from a sorted copy because the listener thread may still append; frames are flushed after events.

### `core/initial_layout.py`

- Seed RNG contract: draws come off env._rng in a fixed order, so changing a gate, a constant or the order relays every seed's layout. REJECT_TRIES and REACH_MARGIN_FRACTION are part of it.
- DEFAULT_INIT_REGION is the init jitter about each object's base pose (xy in m, yaw in rad) unless the pack sets randomization.region, applied on top of the base orientation.
- REACH_MARGIN_FRACTION is the horizontal erosion of the reach grid as a fraction of its radius: room for the approach.
- CORNER_SIGNS are the 8 corners of a unit box as signs on its half extents.
- _obb_intersect is the separating axis test (Ericson, RTCD 4.4); each box is (centre, world axes as columns, half extents); R[i, j] is axisA_i dot axisB_j, t the separation in A's frame, and the 1e-9 epsilon on absR handles near parallel edges.
- _yaw_bands: yaw_offset_deg [lo, hi] is one band, [[..], [..]] a choice between bands (a quarter turn left or right without everything between).
- _rest_offset is the depth of the body's lowest geom corner below its origin, right for any geom type.
- _base_placements: init_states packs collapse to state 0 and also get their scene joint qpos written; an object with no randomization entry takes the scene pose; _settled_bases (learned by reset) override.
- REJECT_TRIES is the per group draws before the authored pose fallback. STROKE_SAMPLES is the samples along an articulated joint's stroke. LAYOUT_TRIES is the whole scene redraws when a group has no legal placement.
- _place_objects clears _legal_cache because lattices are fitted against this draw's fixtures, which the last draw moved. rest_z is recomputed from the base poses because it carries the previous episode's state. _reach_site_lift is the held object's height: a scored site must also be reachable that much higher. Gates measured at the authored pose only bind where that pose passes them: a gate must never reject the layout the pack ships, and an object shipped beyond the arms (never grasped) is not gated on reach. _path_at_base must be empty while measuring because the gate reads it. A group is membership only; objects in no group are singletons. Riding fixed scenery means not being drawn at all. A rider joins its support's group and only there, or a later singleton pass puts it back down elsewhere. pinned {name: (pos, quat)} bodies are written first and excluded from the draw. Groups are drawn stationed first, then biggest: each group only dodges what is already down, so a bulky fixture never threads itself between scattered small objects. A big group drawn early can strand a later one; then the whole layout is redrawn in place (objects stay where the partial draw left them, per the seed RNG contract).
- _place_groups: an authored overlap exemption is kept only with a fixed obstacle or within the group; free objects in different groups are drawn independently. The authored pose fallback is validated against drawn groups only: neighbours still at base are the pack's own (possibly touching) layout. Later draws must not block the stroke of articulated groups already down.
- _xy_points is the world xy of every OBB corner of the body's geometry at its current pose; footprint_radius is rotation invariant.
- static_poses is the jointless bodies the layout places, as the model holds them now: the part of a reset no sim state carries.
- _in_pc_crop: an object outside the crop is invisible.
- _rim_reachable tests reach out to the jaw span each way: a fixture is grasped by its handle, not straddled, so the span bounds it rather than the object's radius.
- _body_reach_ok: an articulated fixture is worked at its near face (the handle side), anything else at its centre plus the jaw span rim where the authored pose already cleared the rim.
- _reach_ok: bodies must be reachable again after any declared lift, and sites at rest and lifted, since the workspace narrows with height.
- clipped_station clips to group_center_box, not legal_cells, because that lattice is cut at the face_robot yaw, which a station overrides; x is None when the station declares none; a station declared entirely outside the legal area collapses to the nearest edge.
- _write_objects: a static body is welded and moved via the model. MuJoCo sleep skips fixed body kinematics even after model poses change, so the sleep flag is cleared around mj_kinematics.
- can_open drives each scored hinge or slide to its open end, one joint at a time (not cumulative), and tests its moving part against the immovable: fixed furniture (randomized statics included) and the robot base. Movable objects and the arm are not obstacles (the agent can move them). Only the joints the success predicate names count, when the fixture has any. The fixture is not its own obstacle.
- _front_heading is the world heading of the fixture's operable face: a slide's axis signed toward its open end, or centre to door for a hinge; None if neither.
- EDGE_MARGIN_M is metres inside the crop edge; the voxelised cloud clips the last cm. placement_region is the xy box an object's footprint must stay inside, crop and support intersected and pulled in by EDGE_MARGIN_M, used by both the lattice and the accept test. group_center_box is the box the group's pivot may occupy, placement_region pulled in per member by its own extent and offset from the pivot (the member nearest an edge binds).
- YAW_SAMPLES is 33 because the extent is a max over yaw samples and 33 keeps the corner cut under 0.4 mm. footprint_extent is tighter than the radius for an off centre body; face_robot and stationed bodies get the circle, since their heading is set by where they land.
- legal_cells is the pivot positions on a 2 cm lattice passing center_fits and, against fixed furniture only, overlap, can_open and the handle path; the min_base_distance standoff filters the lattice itself, so with no legal cell the authored pose survives.
- center_fits is the static half of the accept test on nominal coordinates (containment and reach of bodies and scored sites); overlap and support still run on the drawn pose.
- _sample_group draws one rigid 2D transform (dx, dy, yaw about the group centroid): a station or legal cell under full_rand, else a jitter within the region. A cell gets 1 cm of jitter because the lattice is a search aid, not a grid to sit on. Nothing legal means stay put.
- _others_ctx flattens everything a group must clear to (owner keys, geom ids, per geom AABB lo, hi) for one vectorised broad phase. placed limits the free object obstacles to groups already drawn (one still to draw will move anyway). A randomized static is also an obstacle, but not against itself. A free obstacle still above the surface drops, so its AABB floor is lowered to the support top.
- _overlapping: a dropped object lands where it likes within a few cm, so LANDING_MARGIN_M (xy clearance in m from everything it could land on) is kept clear; either side may still be above the surface, so its landing column stands in for it.
- _landing_column: a free object authored above its support drops onto whatever is under it, so its footprint must be clear all the way down: the world aligned box from the support top to the object's top, xy tight to its oriented geoms plus pad, or None when the object already sits on the surface (within 5 mm) or the pack has no support.
- _all_overlaps uses OBBs, not contacts, because insertion parts have collision hulls inset from the visual mesh; keys are the object name (free) or body id (obstacle).
- _geoms_overlap: set AABBs, then bounding spheres, then the exact OBB test on the pairs left.
- _handle_path_reachable: the handle is the moving part's point nearest the base, tested along the whole stroke of every scored joint of an articulated_reach fixture; a pack that ships it failing is not gated on it.
- _graspable: boxes inflated by grasp_clearance must clear the considered free objects; fixtures and the table are the pack's own geometry, not tested.
- _min_sep_ok: the other endpoint of a declared pair is either already placed or still at its base. _referents_ok: on is settled by grouping before any draw.
- reach_grid is the reachable volume as a voxel grid in base frame (assets/robots/<model>/reach.npz), eroded by REACH_MARGIN_FRACTION per height slice; None when the robot ships no grid. reachable asks whether the fingertip reaches this world point top down, True without a grid; packs mount the arm at a yaw (base_quat); outside the probed volume is out of reach.
- systematic_extremes: the extreme draws for bodies (default everything with stations), position major (position extremes outer, rotation extremes inner). A stationed body gives each station's y ends times yaw ends, any other the corners of its legal cell extent times its yaw ends (one yaw for face_robot).
- place_at_extreme: places bodies at extreme k (wrapping) while everything else draws normally; returns the chosen corner, or None if the selection has no extremes. It pins first, then draws the rest around it, so the others dodge the corner rather than the fixture's random pose. A corner bypasses the accept loop, so it says when it is not actually legal.
- init_regions: per object 1 based number, xy/yaw half ranges, group id; source of truth for the teleop region boxes and the docs table. Statics are numbered after the free objects so free object numbering matches /objects.

### `core/run_seed.py`

- Experiment's episode_timeout, ready_timeout and grace, SETTLE and MAX_API_STALL are seconds.
- Eval seeds start at 1,000,000, clear of the seeds used while developing tasks and interfaces.
- effort is pinned to xhigh because upstream VIA runs xhigh.
- episode_timeout is wall clock per episode, resumes included. It, not termination.max_len, bounds a long task: sim steps are cheap while an agent turn costs about 3 min of deliberation, so 21600 s buys about 120 waypoints. A pack that needs more sets termination.episode_timeout; --wall-clock beats both.
- ready_timeout covers the sim UI, the browser and the opening frame; Blender's first scene build takes about 127 s on CPU only. grace waits for sim_env_server to save after the agent exits, since End Episode queues behind a waypoint still executing and one can settle for minutes.
- SETTLE waits after readiness because the page spends about 20 s uploading the cloud, and a CDP attach during that window blocks until claude drops the MCP server.
- NUM_POINT (400k) is about 85 percent of the fused cloud (about 475k points), 9.6 MB per broadcast.
- MAX_API_STALL caps the time credited back for an API outage, or an outage pins the sim open forever.
- The frozen sim watch keys on the sim's CPU time, not its frames: frames advance only on a waypoint, so a thinking agent is quiet.
- Port cleanup kills only this launcher's own descendants; a listener that is not one means another run holds the stack.
- wait_for_sim_ready requires the sim to serve this seed's demo folder: a sim left behind by a killed run answers on the same port with another folder.
- Differs from upstream VIA: the guide is an appended system prompt (the sandbox blanks CLAUDE.md), and only the agent ends the episode.
- claude silently drops an MCP server that is slow to initialize; the in box bridges plus the CDP attach exceed the 30 s default under parallel seeds, hence the raised MCP timeout.
- Each Codex seed gets its own home on local disk: Codex keeps sqlite state there, which concurrent seeds corrupt when the home is shared over NFS; only the login file is shared.
- A toolless agent (MCP server stuck pending) gets one fresh retry, then it is a harness fault and the seed stays ungraded for a rerun.
- An agent that stopped without end_episode is ended by SIGUSR1 to sim_env_server (the same end path minus the page flush) and is graded like any other episode; publish_episode applies the same rule.
- A simulator exiting with RENDER_FAILED_EXIT (Blender failed under --render blender) is status render_failed, left ungraded, and its episode.json records render as failed.
- The integrity receipt hashes every file that defines the score or the scene before the agent starts and again once it is done. Matching hashes show the files match at those two observations, not that no transient edit happened between them.

### `core/grade_episode.py`

- PRESS_DEPTH_FRACTION (0.25) is the fraction of a spring button's travel from rest to the down stop that counts as a press. A button's joint range straddles zero with range[0] the down stop. PressEdge fires once when the joint passes that depth and re arms only once it is back above half of it. It is shared by press_count and the scene scripts.
- _touching compares root bodies because articulated props carry their geoms on children, not the named body.
- _grasped is SimEnv.held_bodies for the matching arms, after robosuite's _check_grasp.
- _released_at_rest exists because placement boxes are loose enough to pass mid carry, so the object must also be released and still: linear speed at most 10 mm/s and angular speed at most 0.1 rad/s.
- _bbox_world returns the world corners of the local AABBs of every geom in the body's subtree (RoboDojo bbox vertices), subtree because hierarchical props keep their geoms on children, vectorised over the cached subtree because callers may run it every physics step.
- _footprint_hull returns the CCW vertices of the convex hull of the xy footprint, None when fewer than 3 points or collinear, with Qhull for speed since callers may hull thousands of corners every physics step. _hull_contains_xy relies on that CCW order.
- _subtrees_touching is the signed distance fallback for resting bodies whose sleep omits data.contact.
- _in_site_geom tests containment in the site's own frame (a world AABB skews on yawed fixtures) with 1 cm floor slack on whichever site axis points most downward.
- above means obj's centre at or above base's (LIBERO check_ontop).
- joint_in clamps into the joint range first: bands end on mechanical stops and the solver lets a joint rest microns past its limit, which would read as out of band.
- over_site is a site frame box with no release gating, so it can constrain a held object; lo and hi allow an asymmetric band (a floor keeps an object below the opening from passing).
- on_site is LIBERO SiteObject.under taken in the site's own frame (upstream leaves the offset in world axes, right only for unrotated sites), plus contact with the site's parent when given, plus _released_at_rest.
- touching_part tests subtree contact because touching compares root bodies and cannot single out one part of an articulated body.
- axis_dot dots the body's local axis (default +z) in world with world_axis (default +z); abs makes it axis only, either end may point either way.
- pose_within: quat is wxyz like mujoco xquat. rel_to states the goal in the mating body's frame so it follows the assembly. axis grades a part that is round about that body axis on the axis direction alone, spin free. sym means the part maps onto itself every 360/sym degrees about its own z, so the goal is compared against each of those turns. pos_tol_xy bounds the lateral error taken in the reference frame, including when that assembly is tilted, while the overall position bound remains.
- eef_near_start is RoboDojo all_robot_back_to_origin: eef within per axis pos_tol (m) and rot_tol_deg of its episode start pose; cos_half_rot_tol is the quaternion half angle.
- in_bbox is RoboDojo is_A_in_B, quirks included: a's centre inside the hull of b's bbox footprint and above b's floor only (no ceiling).
- depth_in is RoboDojo is_A_depth_in_B: b's bbox top minus a's bbox bottom over z_threshold.
- axis_up is RoboDojo is_axis_up: local axis within tol_deg of world +z, direction sensitive; axis_dot's compare is inclusive, so exactly tol_deg counts as up.
- robot_clear: only contact with the named bodies counts; other scenery may be touched.
- gripper_open is RoboDojo is_all_gripper_open: every arm's jaw fraction at least min.
- _Combine runs every child every step (stateful leaves must see the whole episode) and is a class rather than a closure so the checkpoint's deepcopy reaches the children.
- _Baseline grades against what capture(env) returns on its first call (episode start, since the checker is rebuilt per reset), that first call included.
- _Sequence: earlier stages latch on the first step they hold (a press is an event, not a state) but the last stage must still hold at the graded instant, or the agent could reach the goal, wreck the scene and still succeed. At most one stage latches per step. Every stage runs every step so a stage's temporal leaves (press counter, captured start pose) see the whole episode. The invariant fails the sequence for good once broken INVARIANT_KILL_STEPS steps in a row (5, a quarter second at 20 Hz), so a one step bounce is forgiven but a real topple is not; nothing latches or passes while the invariant is broken.
- _PressCount: exactly count presses; an extra press fails, as upstream.
- hold_steps: the tree's own dwell and never under half a second, so one lucky frame is not a solve.
- _rows returns node None for stateful nodes (sequence stages, holds, leaves with state), which have no instant truth; a leaf with state is an object, a closure has none.
- describe and describe_state are the answer key: teleop only, sim_env_server gates them like /objects.
- Scene.success must be called at every state in order: the streak and any stateful clause advance once per state. Scene._success is [check, hold, streak, frame].
- Subtask.check is Scene to bool, called at every step in order, stateful checks keep their state. Subtask.objects names the bodies or joints a try at it touches, for a check that names none. Subtask.presses marks a subtask done press by press (a puzzle board).
- evaluate: the initial state never earns, but its checks still ran so stateful ones start there.
- replay: each Scene is the same object, moved to the next state. lenient is never used for grading: a recording made before the task's scene changed replays on the archived version (assets/replay_packs) its states fit, and one made before fixture poses were saved replays on its seed's layout, redrawn by a seeded reset. A duplicated control tick (same time within 1e-7 s) is skipped.

### `core/sim_env.py`

- PHYSICS_DT is the MuJoCo step in seconds unless the pack declares physics.timestep. Friction is solved at the velocity level with contacts regenerated every step, so a pinched object turned about the jaw axis walks across the pads in proportion to the step; 0.25 ms cuts the slip about 4x versus 2 ms, smaller gains nothing.
- _close_before_egl_exit closes a GL owner on GC and before MuJoCo terminates its process display.
- Phase: SETUP is reset and its homing steps, nothing may end the episode; RUNNING the agent drives; DONE is absorbing, the episode is over.
- The pack argument of SimEnv is an archived version of the task's folder.
- A task whose scene moves by itself has tasks/<task>/dynamics.py: Dynamics(env) with any of deal, physics_step, control_step, restore and over, called through _dynamics.
- The compiler sizes the arena for a quiet scene and a contact burst then runs mj_stackAlloc out of memory: the floor is 64 MB, keeping a pack's larger size memory.
- The pedestal under a raised base is cosmetic; a zero height cylinder will not compile, hence the 1 mm guard on base height. MuJoCo cylinder size is radius, half height.
- step_stats is per waypoint [ncon max, nefc max, seconds, steps], read and cleared by sim_env_server when it writes the waypoint row, so a contact explosion in a live episode reports itself.
- Per arm state is keyed by robot name and arms[0] is the arm=None default. robot_prefix is a tuple so predicates' name.startswith(env.robot_prefix) matches any arm.
- Jaw width is linear in finger qpos (width_coef); the sign convention is per gripper.
- cameras keep task declaration order; behaviour (wrist, point cloud) comes from the role, not the name.
- Success is graded after the episode (core/grade_episode.py); _check is the live checker a task's dynamics.py may still run.
- full_rand draws anywhere legal (the authored layout is one sample, not the task).
- randomization keys: face_robot derives yaw from position; min_sep is xy in m; min_base_distance is body to m from the base; grasp_clearance is m of jaw room, 0 off; articulated_reach fixtures are gated on the swept handle path of the scored joint, not centre or rim; reach_sites are scored sites far from their body's pivot, and the bodies above one are gated by the site alone; lift_reach names movable grasp targets that must remain reachable after their declared vertical lift; placement_regions is {body: [{"y": [lo, hi], "yaw_offset_deg": [lo, hi]}, ...]} discrete stations where the yaw is an offset from facing the robot.
- referents are the relations the task wording relies on: on (obj rides of), nearest (obj beats rival to of by margin), left_of (obj sits margin m to image left of of, that is -y, since the third person camera looks back along -x).
- episode_t0 is the sim time the episode started, the clock dynamics run on. _dealt: dynamics run only once the episode is dealt.
- _blender None means not yet chosen (see rerender); False means MuJoCo for this episode.
- _free_objects in declared order is the number shown next to each object in the UI. _obj_dofadr is the free joint velocity address.
- Jointless bodies in a randomization group (static objects) are moved via model.body pos and quat so they stay welded; their base pose is snapshotted in _static_base.
- The support is the body roots task.json lists under support; their geoms bound the layout but are not obstacles. Overlap sets: each non plane geom belongs to the free object subtree owning it or to its own fixed obstacle body; robot and support geoms are excluded. Jointless robot base bodies are obstacles too: an articulation opening into them can never open. _static_geoms is the whole subtree because some props carry no geoms on the named body. _support_rect is the xy bounds of the support surface (the slab may sit on a child body). _support_top_z is the top of the support surface, where a dropped object lands.
- rest_z is filled by reset after the objects settle. _rim_at_base, _centre_at_base and _path_at_base are per draw batch: did the base pose clear the rim, centre reach and swept handle tests. _heading_at_base is the authored heading of each face_robot fixture. _settled_bases are the base poses objects actually come to rest at. _home_euler is per robot model, pack overridable.
- SETTLE_EXTRA_S is seconds past idle_step to resolve a randomized overlap.
- held_bodies is the definition of held that grading and the reach probe share: a non robot body touched by every finger root of the arm, that body itself and not its subtree (the grading semantics). data is the live MjData unless a probe passes its scratch copy.
- sort_pills lists its tablet bottles and the pill boxes under support because the old name match on "table" picked them up; layouts depend on it.
- _unsupported: a cluster resting only on itself (stored poses can float stacks above the table) does not count as supported.
- _objects_at_rest: free objects nearly still (under 3e-3 m/s) and no pair penetrating over 1 mm. The speed bound is tight because the solver extrudes an object authored into the table at about 1 cm/s.
- object_color is the dominant display colour, [r, g, b] in 0 to 1, an rbound cubed weighted mean over visual geoms of the texture mean or the geom rgba.
- reset: init_states packs store poses above the support, so the resting pose is learned once and drawn from (not uniform, which would stack everything on one base xy). A pack already at rest needs no settle. A settle may only lower things; anything up (over 5 mm) or sideways (over 2 cm) is the solver blasting an authored overlap apart, not a resting pose, so the stored poses are kept. The first reset must settle from the same clean state as any later one, hence the second _clean_state. Nothing in reset may end the episode (SETUP). Each eef is reoriented to home_euler in place because home_qpos leaves the Panda about 7 deg off. The homing steps are not the agent's, so num_step is zeroed after them.
- STATE_SPEC is INTEGRATION since qpos and qvel alone omit ctrl and the solver warm start, and a rewind would drift instead of continuing step for step.
- _lift_out_of_penetration: an object authored into the table and handed to the solver as is gets launched.
- redeal_episode deals everything besides object poses from the reset RNG (the task's dynamics), then fresh checker latches and baselines. Teleop Randomize calls it after re placing objects, so a reroll is a whole new episode.
- checkpoint's grading_state is the live checker's latches, which rewind with the scene. Nothing in core advances _check during an episode; only tests and their helpers do.
- reset runs the dynamics restore hook on the clean state, before any settle: the model is shared across episodes and a task's dynamics may have edited it (pack_tools zeroes parked cases' contacts).
- forward is plain mj_forward, named after the removed SuperDex twin and kept for the task code and core/initial_layout.py that still call it.
- _step_physics is the one place physics advances. Task dynamics run per physics step (per control step a carried object would sag and snap back) and are off while reset settles the scene.
- _advance_episode_state: SETUP cannot end; DONE is absorbing.
- robot_clear: predicate leaves always pass the bodies whose achievement they protect, so contact with unrelated jointed scenery is irrelevant. A predicate may name a massless marker child; contact is carried by its nearest joint bearing ancestor, whose pose the marker follows.
- rerender swaps Blender renders into obs in place at waypoint boundaries only (observe runs every step). Depth and sensor RGB stay with the active physics engine. force renders even under ROBOARCADE_RENDER=mujoco, for the teleop preview button: teleop streams MuJoCo because Blender costs about 2 s a frame. The GroupRenderer is created outside the fallback on purpose: a failed shared render must not silently become an observation from another engine.
- get_camera_extrinsics: a MuJoCo camera looks along -z and the pinhole model wants x right, y down, z forward, hence the diag(1, -1, -1).
- get_point_cloud fuses every world camera; the wrist rides the arm and would smear the cloud.

### `core/agent_log.py`

- API_FAULT_MARKERS leave out 429 on purpose: a rate limit is the run's own consumption, not a server fault to credit back.
- Parsed logs are cached on disk: transcripts run 10 to 85 MB and a cold NFS read is about 5 MB/s. Each call returns a fresh object because render_episode mutates the events.
- Recorded logs name the MCP server robocua (mcp__robocua__ tools); the live run check mcp_toolless accepts only roboarcade.
- mcp_ui_unattached reads the UI_UNATTACHED marker mcp_server writes in the screenshots dir when the CDP attach fails, not the tool result text; only run_seed calls it, on a live seed, so recorded logs need no fallback.
- auth_failed reads only an error result (is_error) or a Codex error event: the agent's own final message can quote a marker such as "session has ended".
- One API message streams as several assistant events sharing message.id, each repeating its usage. Claude Code's num_turns counts tool round trips, not model replies.
- The log accumulates every session of a seed, so the final result event is scoped by session_id: a killed retry must not read as the first attempt's 401.
- A bare Claude rate_limit event is also how a transient 429 arrives, so it marks a full plan only with the plan's meter at 100.
- Prices are $/MTok for input, output, cache write (Claude Code's 1 h TTL, 2x) and cache read, applied only where the CLI reported no cost; an unpriced model gets None rather than a wrong number. Messages after the last result event (an attempt killed at the wall clock) are priced from their own usage, a lower bound.
- Codex quota notifications can repeat the last request's usage, so cumulative usage must advance before it counts as another charge.

### `core/control.py`

- Proprio.eef_quat is xyzw (scipy order); mink wants wxyz, hence the reorder before SO3.
- ArmIKMink is QP differential IK (mink) for one arm's position actuators. The controller calls solve_pose once per control step: a kinematic solve (FrameTask on the eef site, posture bias toward the start, joint limits as hard QP constraints) from the current joint reference to the step's Cartesian reference, on the nearest branch; the result is the actuator command. substep is the closed-loop variant on measured qpos (stalls softly on contact), kept as a test probe.
- GAIN is the fraction of error corrected per solve: at 1.0 the closed loop on measured qpos limit-cycles (servo noise round-trips through the QP) and shakes grasped objects out.
- POSTURE_COST is for redundancy resolution only: larger, the posture residual far from home competes with the small incremental frame residual and large turns fall short.
- DAMPING is the QP Tikhonov term: far below the task costs (a mm of hold error destabilizes pinch grasps), nonzero to pin free dofs.
- DQ_MAX is rad per substep. LEASH is the max |q_des - q_actual| in rad, servo anti-windup; smaller is smoother but slows reconfiguring away from joint limits. DT is one solve per physics step.
- solve_pose solves on the branch nearest q_init: the posture target is temporarily q_init (substep expects the home target back afterwards) and steps are clamped to DQ_MAX like substep, so the solve cannot jump branches. last_pos_err is the kinematic residual without servo lag: "no solution" against "the arm is still catching up".
- WaypointController call chain, outermost first: move_to (pose, then jaws), update_pose (converge, re-solving from where the arm ended up), _drive_to (one drive plus per-step bookkeeping), _pose_steps (the trajectory). apply_action holds a pose expressed as a delta from the current measured pose while the jaws move; update_gripper closes with a self-centring admittance.
- reaches keeps its own solver and scratch MjData per arm (_probes): the UI probe runs in a thread while the arm may be driving, so it must never touch the live IK.
- _payload is read once per control step: the feedforward needs no finer resolution than the jaw command. The feedforward is qfrc_bias/kp (nulls the PD servo's droop) minus qfrc_gravcomp (a gravcomp model already carries it) plus the held mass's wrench; the hold uses it too, so there is no standing droop.
- _hold_pose solves once off the reference and holds it, as _pose_steps does: re-solving against measured qpos every substep closes a loop faster than the servo can answer, and it limit-cycles as soon as the fingers touch anything.
- _JOINT_LIMS are joint ceilings in rad/s and rad/s^2. _LIN_LIMS are m/s, m/s^2 and m/s^3 and _ROT_LIMS the rad equivalents; the tighter wins. The linear limits are brisk on purpose: the joint ceilings are the real limiter and the path waits on them. Rotation is slower than the arm could manage: pad friction alone holds a carried part, and a fast flip shakes it out even on a perfect path. MIN_WAYPOINT_CYCLES is the shortest waypoint in control cycles: the cap keeps a mm-scale move from implying huge limits (v about 1/distance) that hit a numerical corner in ruckig.
- _pose_steps is the waypoint drive: straight line plus slerp on a jerk-limited (ruckig) path parameter, IK each step, position actuators commanded; it yields the OTG-finished flag once per control step and the caller observes and records. Interpolating in pose, not joints, keeps a carried part off the table. The path advances only while the joints keep up, else the reference runs ahead and the arm jumps to close the gap; a RuckigError (fragile profile) finishes at the joint ceilings. The joint velocity and acceleration ceilings (dq_max_rad and ddq_max_rad per control step) apply as one scale on the step so the direction is untouched: the arm slows, it never deviates. ik.q_des follows q_ref to keep the diff-IK state consistent for a hold. kp is the position-actuator stiffness. A non-finite qpos means the sim went unstable.
- _drive_to: OTG finished means the reference arrived; the servo then settles for at most 25 control steps, or until the eef is within 1e-4 m and ARRIVE_ROT_TOL, and the return reports whether the eef actually arrived, orientation included.
- HOME_TOL is rad per joint: the ramp's last target is q_home and it is held until the joints arrive. _reconfigure_home ramps in joint space over 100 control steps, limit-free so any pinned branch gets there, then holds q_home for at most 100 more. home() takes the straight line when the path probe is green, else the joint ramp (which keeps the current jaw command).
- reaches is the one obstacle-blind reach definition, shared by waypoint execution and the UI probe; q0 is an arm-joint vector, default the current measured joints. It also walks the straight path (REACH_PATH_SAMPLES) because the controller follows a straight Cartesian path on the live IK branch and an endpoint-only solve can jump branches and pass a target no path reaches.
- ARRIVE_TOL is in m, what "the arm got there" means; ARRIVE_ROT_TOL is in rad, without it a stalled in-place turn scores as arrived. REFINE_TOL_M is the floor on closed-loop refinement: update_pose silently skips a smaller correction, so it must stay below the smallest move worth commanding (the drive converges to about 0.1 mm anyway); it is separate from ARRIVE_TOL, which also gates reachability (the UI's red target) and the arrival verdict, where tightening only costs retries. REFINE_ROT_TOL_RAD is floored by the servo: settled drives hold orientation to about 0.2 deg and the reset pose sits about 0.45 deg off canonical, so a value under about 0.5 deg would re-drive every call; still 3x tighter than ARRIVE_ROT_TOL. RESOLVE_TRIES re-solves from where the arm ended up, while still improving. TRACK_TOL is the m the IK solution may miss the reference before the drive stops: past it the IK trades position for orientation and can swing a carried part into the table.
- SETTLE_VERDICT_STEPS is control steps (2 s at 20 Hz) a declaration may wait for the scene to rest; settle_for_verdict keeps the last commands, runs until the free objects stop, then for as long as success must hold, so the recording ends on a held scene.
- update_pose returns early when already there: a gripper-only move_to re-sends the current pose, and a zero-length drive would read as a failure and trigger retries. A held payload is the only contact allowed to persist through a retry, compared by body id so a contact moving between the payload's collision meshes still matches. An unreachable target (a bad click) gets one best-effort drive, no retries. Retries stop when one stops buying anything (under 1 mm): a wall does not move.
- _arm_contacts returns the other side of each contact pair touching the arm. _held_payload_geoms: geoms of free objects touching a closed gripper, plus the subtree of every jointed non robot root whose subtree both finger pads touch (one pad on a lid and the other on its lever still holds the lid); this is deliberately wider than held_bodies, which grading uses. _finger_bodies: declared finger roots and all of their descendant bodies.
- Self-centring close: a 1-DOF admittance B*qdot = F along the jaw axis so the coupled fingers close symmetrically on a fixed object. It engages after |F| > CENTRING_ENGAGE_FORCE_N (10 N) for CENTRING_ENGAGE_TIME_S (0.2 s): free objects drag under 6 N, fixed ones push 17 N or more. The reference is the close-start pose, never the measured one (that would ratchet disturbances in). CENTRING_DAMPING_NS_PER_M is in N*s/m; CENTRING_MAX_VEL_M_S is a safety bound only; CENTRING_MAX_DISP_M bounds the shift; CENTRING_SETTLE_VEL_M_S is the admittance's "stopped"; MIN_JAW_STABLE_STEPS is the control steps the jaws and the admittance must sit still. Only a close is fought. The settle test uses the realized step, not the demanded qdot: pinned at the max displacement the demand never goes quiet and the test would never pass. Both directions end when the jaws and the admittance stop moving; an absolute open width is never reached because the width asymptotes just under it.
- _grasp_alignment_force is the full 3-D contact force on the fingers (friction included) projected on the jaw axis; mj_contactForce is in the contact frame (rows of c.frame, world) and acts on geom2, so the sign flips when the finger is geom1. _jaw_axis is the world unit vector of the first finger joint's axis.

### `core/run_eval.py`

- ROBOARCADE_RUN_STAMP pins the run stamp, and relaunching into a stamped run dir reuses the seeds that already finished.
- PORT_STRIDE is 100 although a stack binds 4 ports, which leaves headroom.
- A shipped interface name is copied to data/interfaces/<name>-<sha8> (the box masks core/), content addressed and never rewritten, so a second driver cannot pull a live sandbox's bind out from under it.
- Multi feed replays render after every stack is down: the agent's own feeds use the same GPU. They run one at a time because two Blender workers on one GPU are slower than one, and in a subprocess so mujoco never loads in the launcher and a renderer crash cannot take the run's summary down.
- Group slots: each slot owns at most K concurrent seeds, a short last slot keeps total concurrency at most -j, and a slot gets a new group only after every seed of its old group finishes and that Blender is closed.
- --render blender with no Blender toolchain stops the run before any seed starts.
- A seed priced from tokens (killed before the CLI's cost event) is a lower bound.
- A bimanual interface on a single arm pack silently degrades to the one arm tools.

### `core/ui_for_agent.py`

- INSTANCE_ID is served at /instance.json: on a socket drop the page reconnects in place if the id it saw at boot still answers, else reloads whatever owns the port now.
- csp is 'self' plus this stack's four localhost ports and nothing external, so an agent-authored page cannot reach outside the eval; inline and eval stay allowed because msgpack-lite uses Function().
- UiRequestHandler falls back to UI_DIR for shared assets (meshes, vendor/); Cache-Control no-store because meshes and the page change between sessions and a stale cached mesh draws wrong.
- http_server returns a server that is bound (so the browser's one-shot navigation cannot race it) but not yet serving; the caller runs serve_forever. port is the stack's base port.
- CHROMIUM_READY_TIMEOUT_S: a cold Chromium on a loaded node can take well over 10 s to open its CDP port. start replaces a wedged launch up to CHROMIUM_START_ATTEMPTS times.
- end_episode clicks the page's End Episode over CDP: the page flushes the input log and trace frames before sending done, which a signal to sim_env_server cannot do. It connects to 127.0.0.1 because localhost may resolve to ::1 first and Chromium listens on IPv4 only. done_by is 'harness', not an agent declaration, so the episode is left ungraded.
- AgentBrowser launches Chromium raw, not through Playwright: its node driver instruments the point-cloud WebSocket and swaps GBs over a session. No start_new_session: Chromium stays in sim_env_server's process group so a group kill reaches it; stop() uses terminate(), not killpg, and is idempotent. _drain keeps at most CHROMIUM_LOG_MAX_BYTES of Chromium's stderr on disk.
- Chromium flags: the background throttling flags keep unfocused or occluded windows rendering, else screenshots go stale; SwiftShader is software WebGL so the page needs no GPU (a shared GPU can lose the context) and screenshots are never blank, with small idle cost since the page renders only on change; --no-sandbox where AppArmor denies unprivileged user namespaces, since Chromium's zygote sandbox cannot start there (the page is our own UI, and the agent's sandbox is bwrap anyway).

### `core/accounts.py`

- PLAN_5H_TOKENS and PLAN_7D_TOKENS are rough Max 20x window capacities, plus or minus 50 percent, in tokens_budget units (input + output + cache creation).
- PLAN_FULL is a percent: a Codex seed neither starts nor continues on a plan this full.
- The CLI resolves ANTHROPIC_API_KEY before any OAuth profile, so choosing a plan means removing an inherited key.
- Claude plan meters come from the undocumented OAuth usage endpoint: advisory, not billing truth.
- Codex reports its rate windows only in band, so reading them costs one cheap no tool turn on METER_MODEL; a refused probe means the plan is spent, not unreadable.
- A plan's store holds the credential file and nothing else, which is what the sandbox mounts; login runs the CLI with its config in a throwaway directory.

### `core/kernel.py`

- The port offsets from ROBOARCADE_BASE_PORT (UI HTTP +0, broadcast WS and JSON +1, UI command WS +2, Chromium CDP +3) must match sim_env_server.py.
- SCREENSHOT_QUALITY 60: Claude Code drops old images past about 24 MiB of base64 per transcript, which busts the prompt cache each turn; q60 is about 93k chars against about 141k at q85 and looks the same at 2x zoom.
- GRIPPER_STEP_M is metres per advance step: the UI's 0.06 UI-unit step at 1 UI unit = 0.1 m.
- EPISODE_OVER and UI_GONE are reason-free on purpose: the agent must stop either way, and the verdict is not the agent's to read.
- send_command's payloads: a robot-frame waypoint {"robot_waypoint": {"arm", "position", "euler", "gripper_open"}} or {"done": True, "done_by": "agent"}. fetch_sim_json is best effort and returns None on failure.
- show_cursor's crosshair is centered so it does not hide the target; with (x1, y1) a drag path is drawn under it and fades after 5 s.
- ui_to_robot converts UI-frame Three.js units (1 = 0.1 m, y up) to robot-frame metres. robot_to_ui is unrounded so camera_set_pose round-trips a camera_get_pose target.
- camera_pose reports azimuth and elevation in degrees and distance in UI units, about the camera target. gripper_pose is the blue target gripper at the fingertip center. call runs window.roboarcade.<action> (core/UI/index.html) in one page.evaluate and returns {gripper, camera} plus anything the action adds.
- CDP_TIMEOUT_MS is per connect attempt: a stalled CDP handshake restarts from zero on retry, so a slow browser needs one long attempt rather than several short ones, capped by the ATTACH_TIMEOUT_S deadline. In the eval sandbox the browser sits behind unix-socket bridges and may be uploading a large point cloud on software WebGL, so the handshake and the page listing can each take tens of seconds; connect, retries and the page listing share one deadline so their waits cannot compound. Page targets arrive asynchronously after connect.
- attach_ui sets the viewport to 1400x900 because a raw CDP-attached window otherwise reports its title-bar-shrunk content area.

### `core/episode_recorder.py`

- vis_dim squashes the square 448x448 feeds to 4:3 on purpose: the tiled video is 320x240 per tile.
- state_fn is the sim state saved per written frame; None saves no states. static_fn gives (names, poses) of layout-placed jointless bodies, saved once.
- One (state, waypoint_idx) per written frame: state i is frame i; discard_last_waypoint relies on it.
- The writer opens lazily on the first frame, when the dims are known. _spans holds one [mp4_frame_start, mp4_frame_end] per live waypoint.
- The RGB feeds are picked by "image" in the key; the "_depth" keys (HxWx1 float) are left out by name.
- States are stored as float64: float32 rounding makes a restored episode diverge.
- Undone waypoints: their states are dropped at once; their mp4 frames are already encoded, so they are marked dead and cut in end_episode. A span's end can sit past a dead region (frames recorded after the undo extend the surviving waypoint), so each edge moves by the dead frames before it.
- Frames read back by VideoCapture are already BGR as written; obs frames are RGB and converted on write. _tile_views is a near-square grid of labeled tiles, blank-padded to fill the last row.

### `core/publish_episode.py`

- Parallel launches share a stamp; the run id's hash in the folder name keeps their seeds apart.
- An episode is graded once, here, unless run_eval already did it.
- A resumed episode (recovered_from/) adds the attempt before the resume to its costs and counts.
- episode.json is the record: its task is the task folder name, metrics.cost_usd the audited cost, agent_runtime set. Episodes published before October 6 2026 were rewritten to that (the original beside it as episode.json.before_migration, a copy, so the run_eval seed folder keeps the old file); republishing such a seed from run_eval links the old episode.json back.

### `core/mcp_server.py`

- The sys.path insert is for a bare-script launch.
- The MCP handshake must not wait for the browser or the client may time out on a slow Chromium: the tool list waits only for env.json (_env_ready); the first tool call waits for the CDP attach (_ready). Dual-arm tasks (env.json "arms" has more than one name) get an arm selector on the waypoint tools.
- _log_tool appends one tools.jsonl row per call (the arguments and the sim steps they spanned) so a transcript can be joined to the recorded states.
- _run's env_before is env.json as fetched before the call; {} (a momentary fetch failure) is not an ended episode.
- _attach without a sim_env_server (dev launch) falls back to defaults; on failure the tools answer with the reason instead of hanging.
- A failed attach also writes UI_UNATTACHED (the reason) into ROBOARCADE_SHOTS_DIR, which run_seed reads to leave the seed ungraded for rerun.
- _log_to_shots_dir: the agent CLI discards this process's stderr; the screenshots dir is the one writable path that survives the eval sandbox, so a startup crash leaves a trace.

### `core/tasks.py`

- TASKS holds one folder per task: task.json, scene.xml, manifest.json, grade.py, test.py.
- Three ring cameras plus the aux views keep 98 percent of an eight ring cloud's occupied volume above the table: viewpoint diversity beats ring density, since ring cameras share one height and shadows. RING_N is the generated point cloud ring replacing the pc pair. RING_PHASE is in degrees; phase 0 would put a ring camera on +x, duplicating the agentview.
- EYELEVEL_ELEV_DEG is degrees above the table for the eyelevel camera, placed on the agentview side (+x), the one camera that reads height directly.
- Farside camera: FARSIDE_BACK_M beyond the crop's +y edge, FARSIDE_UP_M (a hand's width) above the table surface, looking back across it, so approach heights near the far edge read directly; it is the second ground level station, a quarter turn from the eyelevel.
- task.json category is how the task dashboard groups the cards.
- asset_dir: assets/tasks/<family>/<name> is the layout scene.xml was authored against (name may carry a subpath); an archived replay pack under assets/ holds its own meshes.
- _look_at_wxyz returns a wxyz quaternion for a roll free camera (MuJoCo looks along -z).
- _add_camera raises on a name collision itself because MuJoCo's duplicate name error does not say which side declared the camera.
- add_cameras: the ring is built from the pc cameras' pos, which must be world coordinates (cam.pos is body local; a pc camera inside a body would put the ring on the floor); the ring replaces the pair as cloud sources at the pair's mean radius, height and fovy. The birdseye quat Rz(+90) puts the robot side (-x) at image top; the zoomed twin sits at half the height. Third person cameras carry their own fovy in scene.xml: MuJoCo's default 45 frames tighter than the cloud covers, so those agentview cameras say 55.

### `core/utils.py`

- interface_manifest: besides arms the manifest says what the simulator serves: ui (default true) the browser page, rgbd (default false) /rgbd.json snapshots.
- interface_arm_count is the minimum arm count the manifest requires (default 1); adaptive packages declare supported_arm_counts and read the actual arms from env.json, so their minimum stays one and selecting an interface never adds a robot to a task.
- _QUIET_LOGGERS chatter at INFO, so setup_logging sets them to WARNING.
- setup_logging uses stderr so a child's logs stay captured.
- deproject maps a depth image in metres to Nx3 world points via intrinsics K and the cam to world transform.

### `core/agent_sandbox/agent_sandbox.sh`

- MOUNTS binds, in order: the binaries the wrapper, the probe and the PreCompact hook use (claude is native, no node) and the CA bundle for curl only; the venv python's base interpreter and stdlib with the .so files its extension modules link, and an empty site-packages that is its own mount so the final --remount-ro / skips it; the agent binary, the venv, the MCP server's import surface, the bridge, the hook prompt and .mcp.json; the probe, the package, the screenshot dir and the instruction files; the run dir and the login and config mounts.
- Bind order matters: --tmpfs /tmp comes before anything under it, $CLAUDE_CFG before its projects/ replacement, and --remount-ro / last (rw binds made before it stay rw).
- --unshare-pid makes the box's PID 1 exit kill whatever the agent orphaned; --die-with-parent covers only our own death.
- bwrap runs as a child, not exec'd, so the EXIT trap can reap the bridges.
- Never --setenv a secret: argv is world readable via /proc/<pid>/cmdline. HOME must be non empty (claude crashes in uv_os_homedir) but needs no real directory.
- platform.claude.com is on the Claude egress list because the CLI refreshes an expired OAuth token there; without it a plan login older than its access token fails every seed with a 401.
- ROBOARCADE_BWRAP points at a vendored bwrap on machines that ship no bubblewrap.

### `core/agent_sandbox/__init__.py`

- Built in tools are denied so the agent perceives and acts only through the MCP tools (no reading task.json poses off disk); Task too, or a subagent regains the filesystem.
- COMPACT_AT is one compaction point for both runtimes, 245k tokens, the threshold the Codex CLI itself applies to a 258k window. Claude Code compacts 33k tokens below its autoCompactWindow, hence the offset. It must stay well below the roughly 583k tokens at which the CLI's 24 MiB media strip (which counts bytes at q60 screenshots) starts rewriting the prompt cache every turn; re derive if SCREENSHOT_QUALITY rises. ROBOARCADE_AUTO_COMPACT_WINDOW overrides it, 0 disables.
- codex exec composes the first turn after a fixed startup grace whatever startup_timeout_sec says; the roboarcade server needs 6 to 9 s on a loaded node, so the grace is raised or the model starts with no tools.
- Codex starts an MCP server with only the env given in its config, so the box's egress proxy variables are forwarded by name.
- The eval's Codex login must be its own: copied refresh tokens invalidate each other.

### `core/agent_sandbox/connect_sandbox_to_sim_and_network.py`

- Egress is default deny: the proxy accepts only CONNECT to a name in ALLOW on port 443 (no ssh or smtp on an allowed name), no plain HTTP proxying.
- The tunnel clears create_connection's timeout so an idle CDP session does not tear it down.
- Every socket exists before the box starts, so a dial never races a bind.

### `core/render/worker.py`

- mujoco is installed into tools/blender-py with --no-deps because plain pip drags numpy 2.x under a bpy built against 1.26.4.
- MARK must match core/render/__init__.py.
- SAMPLES is per pixel and 32 is the measured knee (4, 32, 64 spp at 51, 86, 127 ms).
- Geoms are relabelled to group 1 (visual, kept) or 4 (dropped) because the exporter filters by group and would otherwise export the collision hulls too, z fighting the meshes.
- The exporter has no mjGEOM_FLEX, so flexes stay out of the USD and are built from the compiled topology, whose vertices alone move per frame. A 2D flex's faces are its flex_elem triples, since flex_shell holds only boundary edge pairs.
- Cycles rather than EEVEE because EEVEE needs a GL context. Devices are probed OPTIX then CUDA because Cycles silently drops to the CPU (about 50 s for eight cameras) on a node without OptiX.
- apply_unit_conversion_scale is off because the exporter authors no metersPerUnit and USD's default would shrink the scene 100x.
- Primitives are retiled from texrepeat because USD bakes one tile per face; texuniform repeats per metre (Object coordinates), otherwise over the object bounds (Generated coordinates).
- MuJoCo and Blender share the camera frame (view along negative Z, Y up), so set_cam copies the pose unchanged.
- Exposure is solved once per camera from a posed 64 px probe (p99 linear luminance at AUTOEXPOSE_TARGET) and kept for the episode, or the feed flickers; an unposed probe saw the underside of the floor and blew out every view.

### `core/render/group.py`

- Only the driver's own simulators receive the private Unix endpoint and capability. Physics, depth, browser, runtime and recorder stay per seed. There is no retry, scene switching, CPU fallback or cross run pool. A broken worker fails its group; a disconnected client does not. The driver closes each group before admitting its replacement.
- Blender's USD import stores solid RGB at half precision, so a seed's initial colour differences are sent float16 rounded while later changes stay raw.
- The task name is validated because the worker opens tasks/<task>/render.py by it.
- A lost reply may be fetched again, but a sequence number is never rendered twice.
- Every PNG is read into the reply before the next seed renders, since the worker overwrites the same files.
- A seed's .mjb slot is unlinked once register replies, by which time Blender has imported it.
- The group directory lives under /tmp rather than the log root because a Unix socket path is limited to 108 bytes.
- The broker gets an allowlisted environment, never provider credentials.

### `core/render/materials.py`

- The packs ship albedo only, so grain, pores, brush lay and glaze are shader graphs (+4 to 9 ms a frame); each keeps the pack's albedo as base colour.
- Procedural surfaces are built in object space at world scale, independent of the mesh UVs.
- A look is the Principled BSDF inputs and the surface on top. coat and transmission are omitted where the surface builder owns them; specular is reflection strength apart from roughness; alpha is tinted plastic that still shadows its contents; base restates a colour the USD exporter drops.
- Robot geoms are classified from the model's arm prefixes, never by name.
- WORDS match whole words (tray does not match stray_beam) and the first match wins, so the specific names come first (burner_plate before burner, knob_metal before knob).
- The unnamed fall back to metal at reflectance 0.6 or more (LIBERO authors 0.8 to 0.9 on the shiny things) and plastic otherwise.
- liquid stays opaque because glass transmission casts a full shadow.
- A task's render.py adds the looks and the names its own scene coins (LOOKS and WORDS of the same shape). Its WORDS are matched before core's and its LOOKS replace core's of the same name. Core keeps only nouns that are not one scene family's: kitchen fixtures and foods live with the LIBERO and kitchen tasks that name them.

### `core/render/__init__.py`

- Under ROBOARCADE_RENDER=blender a renderer that cannot start or dies (dedicated or shared) exits the simulator with RENDER_FAILED_EXIT and run_seed records the seed render_failed: a MuJoCo substitute would score an eval on frames another engine drew. Only the forced teleop preview falls back to MuJoCo.
- An unknown ROBOARCADE_RENDER value raises, so a typo cannot score an eval on the wrong engine.
- MARK prefixes every worker reply, since Cycles writes its own progress to the same stdout, and must match worker.py.
- The timeouts are seconds, sized for a cold node: a fabrica USD import, a first OptiX kernel compile (about 5 min for synchronized cold workers), and eight cameras on the CPU fallback (about 50 s, against about 2 s on GPU).
- Frames are rendered at the pack's image_size because rerender drops any frame whose shape differs from MuJoCo's.
- A failed render closes the worker and raises with the tail of its stderr, which is kept so a dying worker leaves its traceback.

### `core/render/replay.py`

- A recording with static poses and an RGBA bank is restored without env.reset, since resetting the authored layout first can create contacts the episode never had; older recordings still need the seeded reset.
- Output goes to <name>.part.mp4 first, a suffix FFmpeg accepts and the dashboard does not pick up, then is renamed; keyint=1 keeps every frame seekable.

### `core/render/state.py`

- geom_rgba carries the delta from the model's own import, or every geom's colour under complete_rgba, since only the broker knows a shared worker's import baseline.

### `core/render/lifecycle.py`

- follow_parent runs in the child itself, never as subprocess.preexec_fn in a multithreaded driver.

## `interfaces/`

### `interfaces/rgbd_waypoint/motion.py`

- A waypoint is marked pending before it is sent, so a disconnect after the send cannot resend it.
- Tool calls run one at a time, so each delta starts from the previous acknowledged pose. The lock is module level because one MCP server process serves one context.

### `interfaces/rgbd_waypoint/sensors.py`

- RGB and depth are downsampled on the same centre of pixel grid, depth by nearest neighbour so no surface is invented at a discontinuity.

### `interfaces/via_bimanual_cam/tools.py`

- via_cam.py is a symlink to ../via_cam/tools.py, loaded by path because the sandbox masks interfaces and the staged eval copy must carry the implementation.

### `interfaces/via_cam/tools.py`

- The mouse is moved for real before a teleport or camera zoom so the page's hover styles follow it.
- sim_env_server refreshes env.json before the update_ui frame that ticks the page's __waypointDoneCount, so env.json is current once a waypoint's wait ends.
- A move can render for a minute, so the waypoint wait gives up only when sim_steps_used stops advancing for 4 polls of 15 s.
- MAX_ELEVATION_DEG stops short of 90 because an exact top down view is degenerate (azimuth collapses to 0).
- camera_raw treats camera names as data, never URL paths: separators, dots and queries are rejected, and only a bounded, decodable JPEG is returned.
- camera_feed_select's ring azimuths mirror sim_env's RING_PHASE and 360 / RING_N.

## `dashboard/`

### `dashboard/dashboard.py`

- An episode's URL path is its folder's own names, since a few older folders are named by the model's short name while their meta.json carries the full one.
- cameras.mp4 is mp4v, which browsers cannot decode, so it is transcoded once to cameras_h264.mp4.
- The POST handlers read, change and rewrite one JSON file under one lock.

## `tasks/`


### `tasks/add_task/dashboard.py`

- The dashboard and each teleop stack take turns on the same ports (8200 to 8202 by default), so one thing is live at a time: the dashboard serves on BASE, a task click launches sim_env_server --task X --teleop on those ports, and when it exits (End Episode or Dashboard) the dashboard serves again.
- Run: source set_env.sh && python -m tasks.add_task.dashboard, then open http://localhost:8200/ with ssh forwards for 8200, 8201, 8202 and 8300. The 3D viewer (viser) serves itself on BASE+100 and /viewer redirects there; the operator forwards VIEWER_PORT as well.
- Demos are human teleop demos from the UI Record demo button. The docs json is written by tasks/add_task/task_docs.py and may list a renamed task.
- Family display order and labels: all libero_* suites collapse into one LIBERO section. grouped returns {family: [task]} in FAMILY_LABEL order with unknown families after; categorized returns {category: [task]} with absent and invalid declarations unclassified.
- The success column uses the same describe() the teleop sidebar shows, then the predicate types the tree exercises. A group has one shared dx/dy/dyaw; the range is the scene's single region.
- A sim_env_server's just freed port can be in TIME_WAIT. The dashboard serves until /api/launch calls shutdown(), then sim_env_server binds the same port next.
- Live teleop always renders with MuJoCo, whatever ROBOARCADE_RENDER the shell exported.
- viser silently falls back to another port when it cannot bind, which would leave /viewer redirecting to a dead VIEWER_PORT, so the dashboard refuses to start on a taken port.

### `tasks/add_task/geom_viewer.py`

- Collision geoms draw in orange; the teleop dashboard's /viewer redirects here. viser is a dev dependency; nothing on the eval path imports it.
- Run: source set_env.sh && python -m tasks.add_task.geom_viewer [--port 8300]. It is reached over the operator's ssh forward.
- The geom mesh helper returns local frame (verts, faces), or None for plane, ellipsoid and hfield. Assets are absent on a bare clone.

### `tasks/add_task/thumbs.py`

- Renders a small third person thumbnail at the pack's base pose so the card is stable between regenerations; a uniform pack stacks every object on one reference xy at base. The fixed seed keeps it deterministic.
- Cached to data/task_thumbs/<slug>.png, skip if exists, fast families first.
- Usage: source set_env.sh && python tasks/add_task/thumbs.py [--blender] [--force]. --blender renders through Cycles (what evals see) at a Blender launch plus a USD export per task; --force re-renders existing thumbnails, e.g. after switching engine.
- Rendered at the pack's image_size and resampled: fewer Cycles artifacts at 240 px.

### `tasks/add_task/task_docs.py`

- Extracts init base pose, randomization groups, success and robots into data/task_docs.json. Objects are at their base pose, ordered and numbered to match the teleop scene labels; fixed entries are welded scene bodies.
- Run: source set_env.sh && python tasks/add_task/task_docs.py

### `tasks/add_task/rerandomize.py`

- Re-rolls follow the teleop Randomize init path (place then settle_until_rest). A real contact penetration over 2 mm counts as a real intersection.
- Run: source set_env.sh && python tasks/add_task/rerandomize.py [task] [N]

### `tasks/beam/test.py`

- test_fabrica_pose_within: pose_within fires at the socket, not 1 cm away (over the 5 mm tolerance) or rotated 90 deg (over the 15 deg tolerance). The goal is in the mating body's frame, so the test composes it the same way. The loose part starts on the table.
- test_touching_part_survives_missing_sleep_contacts: MuJoCo omits contacts between sleeping trees; the signed geom distance still sees the support.
- pose_within symmetry case: a height offset is not lateral error, the lateral tolerance is radial rather than per axis, and the existing beam symmetry is preserved.
- test_fabrica_pose_within_axis_only_frees_the_spin: a round stick (gamepad) or a cap in a socket that keeps any twist (stool) is graded on its axis direction. A 90 degree spin about that axis still counts, a 30 degree tilt of the axis does not, and with no axis given the spin fails.

### `tasks/bottles_in_bin/scene.xml`
- The bottles' own collision classes carry no rolling friction (0.5 0.005 0); only the explicit bottle to table and bottle to bin pairs keep rolling 0.001, so a bottle cannot stand tilted forever. With rolling on the robot contact, the YAM pad's contact spheres locked on the internal face between two of a bottle's convex pieces (1 to 3 kN pinch) and the bottle stayed on the open gripper. Replaying 106 recorded grasps, none that held dropped without it.

### `tasks/bottles_in_bin/test.py`

- The bin, a toss target beyond reach, is boxed 20 cm around its authored pose; the bottles still draw over the whole lattice.
- Rolling friction 0.001: at 0.01 a bottle set down 10 degrees off upright stood tilted forever.
- Landing bounce: explicit underdamped pairs against the table and bin only, so the pad contacts keep the bottle's critically damped response (2026-09-14).
- test_in_site_box_rejects_robot_contact_with_child_body: a hierarchical object's collision child cannot bypass the release gate.

### `tasks/bowl_on_plate/test.py`

- test_bowl_on_plate: pick the black bowl by one rim and place it centered on the plate.
- test_bowl_on_plate_success_tree: bowl on plate requires physical support, not a hovering alignment; a centered but floating bowl has no contact.
- test_libero_on_success_tree: a settled bowl on plate state fires, the init state does not. No init jitter there: it tilts the plate about 13 deg and the bowl slides off. The bowl is dropped and settled onto the plate for genuine contact.
- _seq_lift_spec: lift the bowl high, then put it back down; two end states a single instantaneous tree cannot tell apart from never lifted.
- test_axis_up_is_direction_sensitive: ports RoboDojo is_axis_up, which takes arccos of the raw dot (no abs). Asking for [0,0,1] must reject an upside down object and asking for [0,0,-1] must accept exactly that; getting this wrong silently passes every flipped object. Yaw alone never tips the up axis.
- test_gripper_open_fraction: is_all_gripper_open, jaw opening as a fraction of the full span, every arm. Reset leaves the jaws wide.
- test_hull_contains_xy_is_oriented: the footprint is the convex hull of the box corners, not an axis aligned box. A 45 degree rotated square must reject a point in its AABB corner but outside the real diamond; an axis aligned shortcut passes this silently. The boundary counts as inside.
- test_in_bbox_and_depth_in: ports is_A_in_B (centre in b's footprint and above its floor) and is_A_depth_in_B (b's rim minus a's underside over a threshold).
- Sequence end state: the order latches, the end state does not. Nothing is graded until the agent declares, so a sequence that stayed true would let it reach the goal, undo it, and still submit. Restoring the end state without redoing the earlier stages does not count.
- test_sequence_rejects_the_end_state_alone: the final scene, reached without ever passing through stage 1, must not grade true. reset() clears progress.
- test_sequence_invariant_kills_progress_for_good: latching remembers that a stage happened; the invariant stops an episode from wrecking the scene in between and walking into the next stage. A cube on the floor for a quarter second kills it for the rest of the episode, while a bounce that breaks the invariant for fewer steps than the kill window is forgiven.
- test_sequence_advances_one_stage_per_step: two stages true at the same instant latch on consecutive steps rather than together, so an episode that never distinguishably passed through them in order does not skip ahead. An interruption clears the two step streak; stage 1 latches only on its third step.
- test_grading_state_rewinds_with_the_scene: save_state/load_state must carry the latch; banking a stage, rewinding and then doing it wrong is the cheat this pairing closes.
- test_describe_state_rows_align_and_turn_true_on_the_solved_scene: one truth per scored row; leaves flip as the scene is solved (teleop sidebar).

### `tasks/cooling_manifold/test.py`

- Provenance: from test_controller_random.py (scripted controller and contact checks) and test_manifold_contacts.py (socket spin friction must release the jam without changing the grasp).
- PIECE part5 uses coacd hulls: it is the part actually ejected in eval.
- GRIP_SPAWNS_X_Y_YAW yaw is in radians. JAW_SPINS_DEG are single turns from the carry orientation, not cumulative. JAW_SPIN_MAX_DEV_M is how far an in place rotation may drift.
- _grasp_piece: fixed pick, jaws across the 31 mm width (perpendicular to the piece's long axis): hover open, descend to mid height, close with no motion, lift clear of the table.
- _jaw_axis_yaw: world heading of the axis the fingers open along, from the two finger bodies.
- run_grip_trial: grasp at one spawn, fly n_poses random poses holding it (each pose found reachable from the current one, kept off the table, jaws closed throughout); returns the pose at which the piece was lost (None if it survived) plus the per pose rotation. It models the dangerous boundary exactly: contact still exists but the jaw command is open.
- _jaw_offset: the slip metric, how far the piece sits from the fingertip midpoint. A part that stays put reads a near constant offset; one walking out of the jaws grows it.
- _Excursion: recorder stand in that samples the eef every control step (the caller blanks env.observe).
- run_jaw_spin_trial: grasp, then spin about the opening axis with position held; reports per angle retention and slip. It lifts clear first because at grasp height a 30 deg turn puts the arm through the table, and vets each spun pose because at the workspace edge IK trades position for orientation.
- test_held_piece_survives_jaw_axis_rotation: the pads sit on the opening axis, so only torsional friction (friction[1], condim=4) resists.
- test_in_place_rotation_stays_in_place: an in place rotation must not bow through a Cartesian arc (a straight joint space line did, swinging the held part into the table).
- The socket fixture predates the 0.25 mm hull erosion, so the jaws and the cap close that gap. With zero hull slack the recorded cap jammed 8 mm high at spin friction 0.3; since the erosion (2026-09-20) it seats at either value.

### `tasks/cup_pyramid/test.py`

- Provenance: from test_tasks.py, SPD cup_stack/pyramid authored pack and physics checks.
- Bottle grade wall contact: at solref 0.01 a 20 N close sank the pads 6.7 mm through the 2.9 mm wall and the cup hung on the opened finger (2026-09-13).
- The retained fallback stays legal, while randomized resets move every cup as its own upright group through the two arm reach union.
- The pyramid must hold for one second at 20 Hz before the streak counts. A geometrically supported looking pyramid whose base starts 9 cm above the table is not a pyramid on the table and falls apart within that second.
- The physical 2 to 1 arrangement: both bottoms on the table, the top touching both. Inverted is the same axis, other end.

### `tasks/cup_pyramid/grade.py`

- BASES: per cup, inverted, released, on the table in the lower band. TOP: the complete pyramid. PAIR_MAX_XY_M: two bases this close in xy can carry a top.

### `tasks/dual_stack/test.py`

- Provenance: from test_default_interface.py, robomimic/dual_stack scripted VIA motion checks.
- _fold: folds a yaw in degrees to (-period/2, period/2]; cubes are 90 deg symmetric, the two finger jaw 180 deg symmetric. The housing direction is mod 180 symmetric and the wrist (joint7, about 166 deg range, home +45 deg) must stay clear of its limit.
- pick_and_place: grasp, then place as a diamond (45 deg to the red) corner to corner along the red's face diagonal; two 5 cm cubes side by side on the red have zero COM margin. udir is the red top face diagonal (unit xy); diamond_yaw is red yaw + 45 in (0, 90], not folded mod 90 (the +/-90 alias puts a finger over the block already on the red). pre_yaw=90 turns the long hand housing parallel to the built stack for the pick.
- Descent is two stage: one 12.5 cm waypoint's lateral IK transient drags the block. The second step goes to block center (0.10 + half size).
- The in grip offset is cancelled after the rotation since it turns with the jaw.
- Lowering is two stage: the block pendulums freely about the grip line (point pad contacts, no torsional grip) and a single long descent swings it. A closed loop XY correction follows because in grip slip plus reach tolerance eat the roughly 8.5 mm COM margin.
- Retreat is two stage: a large first waypoint jerks laterally (P step about 5 cm) and the finger bodies can clip the block's top half, so it pops up gently first, rising along the jaw axis since a vertical retreat's IK transient topples the block. It parks away from the blocks before any big reorientation because joint swings can sweep links near the workspace.
- test_dual_stack_predicate: the predicate fires (3 step dwell) for the diamond arrangement, not with a block on the table. Release is +6 mm, above the roughly 3.5 mm in grip sag and low enough not to bounce; the red half height is 0.030 (1.2x) and the green half 0.025. Diamond centers sit plus or minus D along the red face diagonal, in a frame that follows the red's spawn yaw (tiered randomization yaws every block). Green2 is aimed from green1's settled position with at least 10 mm air gap because the held block pendulums 2 to 4 mm and one face graze slingshots green1 off, and it releases 12 mm higher so a swinging bottom corner cannot catch red's top edge.
- test_dual_stack_clause_same_level: offsets keep each block inside on_red's dz_tol individually while their level difference exceeds same_level_tol, isolating the clause. Width 0.02 is below gripper_open_min 0.06.
- test_robot_touching_helper: contact graph clause tested on the helper directly; a block overlapping a finger registers as robot contact, a block on the table does not. It overlaps the hand body itself because the grip site midpoint sits between open fingers (8 cm span vs 5 cm block), which is contact free by design.

### `tasks/dual_stack/grade.py`

- The velocity loop checks that every block is at rest.

### `tasks/handover_dominos/grade.py`

- A block stands when its axis is within acos(0.9) of vertical, it moved under 2 mm over 0.1 s and the robot does not touch it. k blocks stand in a line when k of them lie within 2 cm of one line with neighbours 3.5 to 10.5 cm apart. Every other recorded frame (0.1 s) is read, as the paper's replay did.
- DOWN_DOT 0.7: at least 45 degrees from vertical is toppled. STABLE_LIN_VEL_M_S 0.01 (10 mm/s) rejects a line still sliding into place; STABLE_ANG_VEL_RAD_S 0.1 (5.7 deg/s) one still tipping into place. STANDING_Z_M is for 120.6 mm blocks on the table with 12 mm tolerance.
- line_count: largest set of points (xy rows) on one line within LINE_TOL_M whose neighbours are MIN_GAP_M to MAX_GAP_M apart.
- DominoChain latches a stable standing line, then requires one contact propagated knockdown. End poses alone cannot tell a cascade from knocking the blocks down one at a time, so the robot may touch at most the first falling member and the fallen set grows only through observed block to block contacts. Every endpoint pair is tried as the line direction: no privileged world axis, and cheap for a few dozen blocks.
- The standing check samples once per state.

### `tasks/handover_dominos/test.py`

- Provenance: from test_tasks.py, SPD jenga/handover_dominos authored pack and physics checks.
- Full yaw radial extents leave a central lane more than four block widths wide.
- A second directly manipulated member permanently invalidates the event, even if the blocks are subsequently arranged as one connected fallen chain; one robot knock is permitted.

### `tasks/hang_mugs_and_rack/scene.xml`

- The plate_pad pairs (each plate collision hull against each named collision geom of both yam fingers, 1248) give a grasp a rolling term, the stand in for a pad's contact patch, that the same plate must not have against the world: with rolling friction 0.05 everywhere a plate stood on a 5 mm rack rail for good (seed 1000000). An explicit pair replaces the dynamic contact for its geom pair, so nothing smaller (priority or solmix on a geom) keeps the rolling term to the pinch alone.
- The pairs name the attached geoms with their arm prefix (move_, hold_), so the scene compiles only with its robots attached.
- The plate_pad pairs carry a 1 mm margin. Without it a closed pad pushing a plate sideways slipped 28 to 35 mm into a plate hull within 50 ms (6 of 10 perturbed restarts of gpt-6-astra seed 1000000 frame 3413); with it, 0 of 10, and the one reproducible plate grasp still holds. Cutting the pair rolling term instead also stops it but loses that grasp.
- Every mug hull x stand hull contact is an explicit mug_stand pair (1298): condim 6, friction 4 4 0.1 0.005 0.005, the 4 ms response. The mug class friction 4 exists for pad grip and wins the max mix against the stand, so a handle root corner over a peg locked into a kN wedge; with condim 4 a mug hanging from one point rocked with no loss for a minute. The rolling term damps the rocking, mu 4 keeps hung mugs on their pegs (mu 0.5 to 1 dropped 5 to 8 of 19 replayed hangs).
- noslip_iterations is 0 (MuJoCo default). The no-slip pass fed the locked wedge into a free stand that spun for up to 100 s untouched by the robot; with 0 the same state stops dead. Mug grasps replayed from recorded episodes also held more often without it.
- Each mug collides through 59 convex hulls cut from its visual mesh: 16 angular sectors x 3 height bands of the wall (radius above 0.6 r_out, so the cup stays open), one base below the inner floor, 10 handle segments by angle around the opening. The old 16 hulls stood 1 to 4 mm proud of the white mug and filled a quarter of either handle opening. Mass and inertia sit in an explicit inertial with the values the visual mesh gave, so the hulls (density 0) cannot move them.

### `tasks/hang_mugs_and_rack/test.py`

- Provenance: from test_tasks.py, SPD kitchen_cleanup/hang_and_rack authored pack and physics checks.
- The mugs draw in the middle of the bench between the fixtures (user, 2026-09-10). The pack's meshes and textures live in the asset mirror.
- test_spd_kitchen_pad_pairs_carry_the_rolling_term: a plate's own contact is physical (no rolling resistance against the rack or the counter); the pinch keeps its rolling term through the pad to plate pairs scene.xml declares, one per plate hull and finger geom.
- The unsupported case: every proximity, height, and orientation clause holds, but none of the four bodies is supported by its fixture. The supported case uses edge poses released from physical drops into two existing rack slots.

### `tasks/jenga_tower/test.py`

- Provenance: from test_tasks.py, SPD jenga/tower authored pack and physics checks.
- Full yaw radial extents plus the side bands reserve more than the 120.6 mm square tower footprint around y=0, rather than relying on sampled layouts alone.
- The solved fixture is four layers of three from twelve blocks; the six spares lie flat on the table. A block with its long axis along y spans 41 mm across the 60 mm pitch. Two supporting blocks below are required; one is not enough.
- _put places a free body by name, not via _obj_qadr: scripted scenery (the shuttle, the tokens) is deliberately not declared a manipulable object.

### `tasks/jenga_tower/grade.py`

- layers reads the block centre height band per layer, in meters, from the `bands` of the success tree's tower_support leaf, so task.json is the one source.
- layers returns (complete layers, blocks laid in the next one) of the tallest consistent tower: a layer needs three released blocks in its band, flat, long axis along the layer's direction (alternating, either one first), the first on the table and each later one touching two blocks of the layer below.

### `tasks/lock_the_coins_in_the_strongbox/test.py`

- Provenance: from test_controller_random.py (scripted controller and contact checks) and test_lockbox.py (mechanical and grading regressions for the authored RoboDojo cash box).
- The centered grasp nudges the hinge open; fully closed, one pad only brushes the tab.
- _seat_key puts the free key at its case relative seated pose, including rotor angle. That fixture starts seated; the real insertion route depresses the pin. Existing lid and bolt fixtures isolate the reverse interlock, holding the depth pin clear; dedicated depth tests cover its engaged state.
- The key cavity sweep includes the visible aperture and the real rear bearing.
- A fixed 0.6 N.m opening torque distinguishes the bolt under hook states. A fixed 0.2 N.m turn reaches the <=0.60 lock band only with the lid closed.
- The visible seated key, not an equality slave, is the physical drive interface.
- The lid is opened to drop the coins but no stage asks for it.
- The 6 mm half thickness bow is the table foot after the Rx(90) lay flat. Full random yaw (user, 2026-09-11): twelve seeds spread well round the circle.

### `tasks/open_middle_drawer_halfway/test.py`

- Merged from test_controller_random.py (libero/open_middle_drawer_halfway: scripted controller and contact checks) and test_home_waypoint.py (archived drawer states pin both routes used to execute a home target).
- test_reaches_replays_recorded_drawer_paths: GO seed 8, waypoint 22 must reject its joint limit branch flip; waypoint 12 stays green. The arm vectors are the saved segment start states from cameras_states.npz. The first endpoint was free of contact but landed 366 mm short; the second stopped on the stove, which is intentionally outside obstacle blind reachability.
- CASES are (run, seed, waypoint) in the order encoded by _STATE_B64: all 16 gripper_home starts where the legacy joint sweep disturbed the scene.
- RED_CASES are every red home target in the 41 archived calls, in _RED_STATE_B64 order.
- The state fixture is float32, ample for these physical expectations. Each row is time, qpos, qvel, ctrl captured from cameras_states.npz; warm starts are cleared on restore so the fixture does not depend on solver state.
- num_step < 200 means the hold ended because the joints arrived.
- bot.recorder = SimpleNamespace(waypoint_idx=0): the waypoint row reads its number; _rec (None) is what home() gets.
- The orientation dict {"x": e0, "y": e2, "z": -e1} is the page's YZX Euler, as sent.
- test_only_the_home_flag_uses_controller_home: the page marks a gripper_home target with home: true and sim_env_server dispatches it to home(); the same pose (as the page received it in UI_CONFIG) without the flag is an ordinary move_to.
- ("GO", 1, 43) hooked the cabinet 10 mm at the 2 ms physics step and went under at 0.25 ms.
- test_libero_open_drawer_region_specific: ported LIBERO Open(middle_region); fires only for the named drawer's joint, opening the wrong drawer must not count. The first check is False because the drawer starts closed, the second because it is the wrong drawer.

### `tasks/open_middle_drawer_halfway/grade.py`

- The subtask list ends with the task's own success.

### `tasks/organize_table_and_plug_in_charger/grade.py`

- pens(s) counts pens standing in the holder and released.
- tips_in counts the charger as plugged only when both blade tip centres (charger_tip_0 and charger_tip_1) sit less than 13 mm above the strip's base in the socket frame. The strip face is at 21 mm and genuine plugs rest at 8.4 to 9.0 mm; the depth_in AABB clause it replaces also passed a charger standing on the face with its blades beside the slots (16.5 to 18.8 mm).

### `tasks/organize_table_and_plug_in_charger/scene.xml`

- The pen holder collides through 31 convex hulls cut from its visual mesh (holder_*.obj): the base slab, and per hex side the inner wall, a thin skirt beside the floor pocket and three frame bands that leave the two spiral grooves open. Every hull stays within 0.35 mm of the visual surface; the 26 CoACD hulls stood 0.7 mm proud everywhere, filled the grooves (2.2 mm) and reached up to 3 mm into the cup, so a finger pressed into the wall could end up trapped between opposite hulls. The explicit inertial keeps the mass at 0.190 kg and the inertia unchanged.
- The headphones collide through 366 convex hulls (headphones_hull_*.obj): CoACD at threshold 0.02 on a watertight solid of the visual mesh (fast winding number on a 0.4 mm grid), each hull then refit to the visual surface points it contains. They stand at most 2.6 mm proud of the visual (1.3 mm at the 99th percentile) and leave no visual surface uncovered by more than 0.14 mm; the 83 CoACD hulls stood up to 6 mm proud. The explicit inertial keeps the mass at 0.925 kg and the inertia unchanged.
- The charger's collision geoms have priority 1, so their own friction 1.0 governs every charger contact instead of the pads' 2.0, which held the charger balanced on a fingertip or pinched at the fingertip nubs. condim 4 with torsional 0.3 keeps the torsional friction the pad contacts had; with condim 3 a pinched charger turns in the hand.
- The strip's mesh hulls use the slot boxes' solref 0.004 and solimp 0.998, so a blade pushed off a slot meets plastic as stiff as the slot walls.

### `tasks/pack_tools_into_toolboxes/dynamics.py`

- deal draws the visible start and an immutable release schedule from the reset RNG.
- Seeded entry variety is drawn once per body at the deal so a seed's episode is fixed: a tool gets a lateral offset across the belt and a yaw about vertical, a case one of the listed yaws (its seats turn with it). box_yaw_deg values are degrees.
- The visible row is dealt from its authored stations. A pitched one way feed leaves only its seeded leader live and releases the parked followers later; four tools at 0.3 m pitch cannot all fit behind the hood on this belt.
- spawn releases the reset dealt queue at fixed conveyor travel distances.
- physics_step drives current belt contacts with bounded tangential friction. A rider past `end` wraps back to `start` with whatever is standing in it, so a case and its contents stay together; everything else is left to ordinary contact physics. With `exit: remove` the belts are one way: a loose good past `end` parks at its exit pose and a case is carried OFF_STAGE_SHIFT_M (10 m) along the belt axis with its contents and pinned there every step, graded as it left (user, 2026-09-13).
- Two belts need no code: put both slabs in one body and _subtree_geoms collects them, so one `conveyor` block drives both; `slab_speeds` (geom name to m/s) lets one slab run at its own speed, while `speed` stays the tool belt's and the release schedule's metre.
- The wrap loop skips off_stage bodies because contacts lag the park by a step.
- Parked bodies get contype and conaffinity 0 (restored from the copies taken at construction) and are still pinned every step: gravity acts on them, and body_gravcomp cannot be switched on at runtime because ngravcomp is fixed at compile time.
- restore derives the parked set from qpos (axis coordinate beyond end + OFF_STAGE_SHIFT_M / 2), so undo, replay, SimEnv.reset and the teleop reroll bring the contacts back with the state.
- carry_group returns `name` plus every loose good standing in its footprint, the same containment test grading uses, so a wrap can never separate a case from its contents. Contacts are too flaky for this: a tool resting on two splayed handles registers a normal 45 deg off vertical, and a bouncing one registers none at all.
- Only loose goods with an exit pose ride in a case, never a peer case: a touching case once had its centre swallowed at wrap, and the size proxy that guarded against it dropped a 287 mm hammer from a 317 mm tray.
- entry_occupied tells whether a translated body or group would overlap another free object.
- over is True once every case has left the scene on one way belts, so nothing more can be packed.

### `tasks/pack_tools_into_toolboxes/test.py`

- Merged from test_sim_env.py (robodojo/pack_tools_into_toolboxes: scene mechanism checks).
- _put_tool_past_conveyor_end places a tool on the belt just beyond its configured travel limit; pose z is lowered 0.5 mm into the belt because a tool resting at exactly zero gap has no contact yet.
- test_toolbox_conveyor_remove_is_one_way: a missed queued tool parks once; its now due spawn entry cannot release it again.
- test_toolbox_conveyor_seated_rider_leaves_with_box: a removable tool inside a box leaves the scene with the box, pinned as it left, and the episode is over once every case has left. It models a one step carrier contact gap while its contained tool still touches the belt; removal must use containment, not the carrier's instantaneous contact. It ends by undoing to before the park (contacts back, nothing off stage) and resetting to a second episode whose cases match a fresh SimEnv's, the cross episode case of the shared model. The seat cells have solid floors, so a seated tool never touches the belt itself; with the box lifted nothing rides, and the group wrap rests on containment. The case goes 10 m down the belt with its tool (one way, not back to the start); scene_over is still False because toolbox1 remains; the 40 step loop checks the case stays pinned against gravity. The second case is dealt later; its authored station is on the belt, lowered 0.5 mm so it rides.
- test_toolbox_conveyor_static_rail_rider_does_not_leave_with_box: a tool supported by the fixed centre rail is not loose cargo in a nearby box. rail_top + 0.01654 puts the tape underside on the rail; the rail rider stays while the case leaves without it.
- test_toolbox_conveyor_does_not_remove_held_rider: robot contact wins even when the grasped tool still touches the moving slab.
- test_toolbox_conveyor_case_belt_runs_at_its_own_speed: slab_speeds lets the case belt run slower than the tool belt (the user's 2026-09-13 request); `speed` stays the tool belt's and the release schedule's metre. 8000 steps are 2 s at the 0.25 ms timestep.

### `tasks/puzzle_4x6/test.py`

- Merged from test_lights_out.py (ogbench/puzzle_4x6: board and button checks).
- test_reset_paints_dealt_init_in_first_rendered_frame renders immediately after reset, projects each cap centre into agentview and classifies the actual pixels (red has R > B, blue has B > R), catching a model only paint that happened after the first image was captured. Row major columns run along +y in the scene and left to right in the front image.
- Tests drive the board through env.dynamics.control_step. Pressing buttonbox_joint_8 (row 1, col 2) toggles 2, 7, 8, 9, 14. Holding at the bottom stop for 10 steps toggles once, not ten times. Any extra press after solving breaks the goal, and restoring the solved saved state restores the solved board and its paint.

### `tasks/puzzle_4x6/grade.py`

- lights_out only reads the board dynamics.py keeps in data.userdata, so it replays from any saved state with no game state of its own.

### `tasks/puzzle_4x6/dynamics.py`

- The board is the button toggle board of OGBench PuzzleEnv: a press (buttonbox_joint_i first reaching DOWN_M) toggles the button and its orthogonal neighbours. scene.xml's nuserdata is 2n: the board, then the down flags of the last control step.
- Presses are detected once per control step, as the old success check did, so the deal, the toggles and the grades of earlier episodes are unchanged.
- RGBA is the ogbench palette: index 0 is red, 1 is blue.
- DOWN_M is on the 0.024 m button travel (from ogbench PuzzleEnv.post_step).
- toggle_matrix is the GF(2) board delta: column i is the plus shape toggled by press i.
- deal draws an init that a sampled press set turns all red; the goal is every button red. The shipped sizes are full rank (guarded by tests), so the sampled press set is the unique minimum solution without rejection.
- The deal paints the board before the episode's first rendered frame; restore repaints after a rewind or a replayed state.

### `tasks/puzzle_4x6/add_board_to_states.py`

- Episodes recorded before 2026-10-06 saved states 2n floats short (no userdata), and core's replay cannot load them until this rewrites them. userdata is the last block of mjSTATE_INTEGRATION, so the board and down flags are appended columns. Rebuilt boards match the recorded rgba banks except where the recorder caught the paint one frame late.

### `tasks/puzzle_6x7/test.py`

- Merged from test_lights_out.py (ogbench/puzzle_6x7: board and button checks).
- The four press checks are the near left corner, the near edge, an interior button, and the far right corner.

### `tasks/puzzle_6x7/grade.py`

- lights_out is imported from puzzle_4x6: the same board, 6x7.

### `tasks/serve_warm_drinks/test.py`

- test_press_edge_latches_only_with_live_conditions: a pass through has no edge, and a press while the door is open cannot bank credit. Passing through the live condition without pressing cannot latch the first stage; a full press while the door is open is seen but cannot be banked; a down edge with the door closed latches and then the live final stage succeeds.

### `tasks/serve_warm_drinks/grade.py`

- DOOR counts the microwave door as open below the joint value.
- Provenance: the DOOR threshold comes from LIBERO's libero/envs/objects/articulated_objects.py.

### `tasks/set_up_pcr_run/test.py`

- Merged from test_controller_random.py (autobio/set_up_pcr_run: scripted controller and contact checks).

### `tasks/shell_game/test.py`

- Merged from test_shell_game.py (seed dealt ROYGBIV shell game on multi press triggers) and test_sim_env.py (shell_game: scene mechanism checks).
- _cover puts shell_i onto ROYGBIV[i]: uncover_order needs a covered baseline before any reveal counts, so a row that was never covered can never be uncovered. _uncover lifts each cube's shell clear, one per call, in `order`.
- test_shell_game_needs_both_press_stages_then_the_rainbow: green once, blue five times, then the ROYGBIV row. The row alone, even dealt solved, must not grade before the presses; green alone fails; four blue presses are not five. The row is re asserted after the presses, and stage 3 latches a step after stage 2.
- test_shell_game_row_never_uncovered_fails: the row alone is not the task. This passed before: the scene starts with the shells parked 0.18 m behind the cubes, so with no covered baseline every cube read as uncovered on the first call, in rainbow order, and a row built under seven shells graded as a solve. The test covers and never lifts one.
- test_shell_game_uncover_out_of_turn_fails: violet before indigo; the reveal order is graded, not just the row.
- test_shell_game_reveal_survives_a_shell_parked_alongside: seven shells have to go somewhere while the row is built; setting one down next to an already uncovered cube must not un reveal it.
- _steps takes sim seconds, not step counts, because the scene script schedules on sim time; it calls e._check(e) per step as _advance_episode_state does per control step. _push ends with a 0.08 s release so the spring returns and the edge re arms.
- test_wrong_row_then_rainbow: negative first, positive second, because success latches.
- scripted: the shell game's deal comes from the seed, swaps hang off press edges.
- dwell holds still for this much sim time, which only advances while a waypoint runs; that is why the schedule hangs off button presses. press is one down edge: hold the button past the trigger depth, then let it spring back so the edge detector re arms.
- test_script_waits_for_its_trigger: nothing moves on a clock; an agent only advances sim time while a waypoint runs, so a phase it has to witness must be one it releases itself.
- In test_cover_runs_on_the_press_and_release_is_inert the 4 s dwell is long after the button release: nothing else runs.
- After every leg has run, each shell's last leg is over, so physics owns it again: the agent must be able to lift the shells to build the rainbow.
- test_pieces_stay_pinned_until_their_last_leg: a covered piece with swap legs still to run is held by the script.
- test_script_does_not_sag_between_control_steps: the schedule is written every sim step; applied once per control step instead, a carried shell free falls 12 mm and snaps back 20 times a second. The dwell lands inside the 0.3 to 0.7 s carry.
- test_a_reset_rearms_the_schedule: reset settles the scene and homes the arm, and both step physics, so the clock, the press edges and the dealt hand have to start over with them.
- test_trigger_leg_outranks_clock_legs: among opened legs a trigger gated leg outranks every clock scheduled one, even those opening later (else a stop button could never halt anything for good). The legs are a clock shuttle right then later left, plus the stop button holding wherever the press finds the body. After the first shuttle leg the body is at y = 0.10; the stop is pressed before the second leg opens, and the later clock leg must lose.
- _put places a free body by name, not via _obj_qadr: scripted scenery (the shuttle, the tokens) is deliberately not declared a manipulable object.
- test_press_count_needs_a_release_between_presses: holding the button down for 20 steps is one press, not many; the clause counts edges. Release and press again makes two.
- test_lateral_order_needs_the_gap_and_the_row: spelling is an order; no test on one block at a time separates C A F E from F A C E, and a letter shoved out of the row should not still count as next in it. The final orange placement is the right order but off the row.

### `tasks/shell_game/dynamics.py`

- deal draws the cube row and the swap schedule. Legs are absolute positions, so the whole evolution is fixed there. cube_at maps slot to cube body; shells park one per slot; the cover is up, over, down on press 1 of cover_btn; each shuffle press does one swap.
- Legs are ordered by trigger count, press index, time; a leg supersedes earlier ones the moment it opens (a stop button interrupts a running schedule).
- self.pressed maps joint to the sim time of each down edge.
- physics_step: a leg's clock is episode start or the Nth press of its `after` triggers; a body is kinematic only inside its legs and goes back to physics after its last one. Both clocks are sim time, so a rewind (mj_setState) replays the schedule for free. A rewind restores the clock but not the pressed list, so presses now in the future are dropped.
- Among opened legs more triggers outrank fewer, then the last to open wins (by time, not list position); ties go to the later entry. A leg not yet started is skipped; after the last leg is done physics owns the body again. A leg starts wherever the body is when it first runs.
- epoch is the sim time the leg measures from: episode start, or the last of its triggers reaching press N (None while any is short).

### `tasks/shell_game/grade.py`

- SETUP is the cover press, then the shuffles, in order.
- UNCOVER_TOL is metres in xy: a cube this close to a shell's centre is under it. UNCOVER_DZ is metres in z too: in xy alone, a shell carried overhead would count as covering.
- UncoverOrder: objects uncovered in the given order. An object counts only once it has been covered and then revealed, so a scene that starts uncovered does not count as every reveal at once. A reveal while an earlier object is still hidden breaks the run for good; reveals latch, so re covering an object cannot undo one.
- In lateral_order, `other` is the other horizontal axis.
- The first k cubes uncovered in rainbow order are each their own latching check (uncover_order keeps state).

### `tasks/sort_pills/grade.py`

- DOSE is the exact number of pills of each kind per compartment.

### `tasks/sort_pills/render.py`

- amber is translucent amber plastic, not clear glass: the pills show through a tinted, low gloss wall.
- In WORDS amber comes before pill because capsule_bottle is the bottle while bottle_capsule_1 is a pill.

### `tasks/square/test.py`

- Merged from test_controller_random.py (robomimic/square: scripted controller and contact checks).
- The move_to call in test_small_waypoints_do_not_break_the_pacer must not raise.
- test_sub_arrival_translation_actually_moves: a correction below ARRIVE_TOL must drive, not pass while leaving the arm still.
- test_resending_reset_pose_is_zero_step: the tighter refinement threshold must retain the intentional exact pose fast path.
- test_square_success_tree ports NutAssemblySquare._check_success on constructed states: start with the nut off the peg (False), on the peg below the height band with the gripper away (True), off the peg laterally (False).

### `tasks/start_incubation/scene.xml`

- Torsional friction is 0.002 (a 2 mm patch) on every scene geom, through the top level geom default and the explicit hull frictions, and the task's robot is franka_panda_low_torsion: 0.002 outside the finger bodies, 0.01 on the fingertip pads. With the 0.3 pads and hulls a loose tube was launched or carried by one pad and a released tube wedged in a machine well (e8a11b5).
- The cap lives in a robot variant because a scene cannot override an attached robot's geoms; the variant reads franka_panda's meshes and copies its reach map.
- Every tube_1 and tube_2 collision geom has priority 1, so the tube's own friction 1 (not the pads' 2) governs pad contacts and a pad cannot carry a tube by friction alone past ft/fn 1. Priority hands the tube's solref 0.004 and solimp 0.998 to all its contacts as well: pad contacts are harder than the old mix (0.949 0.974) and the tube rests 0.075 mm higher on the table.

### `tasks/start_incubation/test.py`

- Merged from test_tasks.py (autobio/start_incubation: authored pack and physics checks) and test_task_torsional_friction.py (a task local spin friction cap must not retune other contact physics).
- test_autobio_start_incubation_has_physical_wells_and_ordered_return: the visible ThermoMixer wells are physical and the press is temporally bound.
- test_in_start_site_requires_the_hole_the_layout_dealt: a tube scores as returned only in the hole it started in; seated where dealt is True, a different empty hole is False.
- test_low_torsion_robot_changes_only_torsional_coefficients: the variant differs from franka_panda only in torsional friction, lowered to 0.002 outside the finger bodies and 0.01 on them, never raised.
- test_start_incubation_caps_every_torsional_coefficient: the compiled task carries both caps on every geom and has no explicit pair, which would override the geom coefficients.
- test_recorded_incubation_contact_failures: the final geometry and friction must not launch or carry the recorded loose tube.
- test_round_tip_topples_between_holes: the taper must never introduce an invisible flat standing base.

### `tasks/start_incubation/grade.py`

- press: program 1 pressed (a down edge) for the first and only time, with both tubes seated in different wells, after a moment both were seated unpressed.
- The first return subtask evaluates the tree first because its in_start_site leaves read the first state.

### `tasks/start_incubation/render.py`

- The microtube look has no opaque override: the sample shows through its alpha.

### `tasks/start_making_breakfast/scene.xml`
- bread_1_flat_lower_crust, the invisible flat box along the slice's lower edge that keeps it standing in the toaster, collides only with toaster_1_bread_carriage (contype and conaffinity 2 on the box, 3 on the carriage; everything else is 1). Colliding with everything, its 100 by 7 mm flat foot let the slice balance on its edge on the plate or the table.
- Each moka pot collides only through 18 convex hulls cut from its visual meshes (moka_pot_*_col.obj, shared by both pots); the 15 LIBERO boxes g2 to g16 stood 3 to 12 mm proud of the visual pot, so fingers stopped on phantom corners. Body: lower chamber, waist band and upper chamber with the lid, split at the narrowest cross section (9 mm below the body origin) and 7 mm above it, where the upper chamber steps out. Handle: an arm and a top block cut by planes, then five wedges cut by planes through a point inside the handle's loop, so no hull fills the gap between handle and body. Spout, lid lip, lid tab, safety valve, hinge plate, two hinge knuckles and the knob are one hull each. Every hull stays within 1.5 mm of the visual surface except the spout and the lid lip, which close over the pouring opening by up to 5 mm. The explicit inertial keeps the mass at 0.4 kg and the inertia unchanged. The hulls were generated outside the repo with scipy ConvexHull from the visual vertices.

### `tasks/start_making_breakfast/test.py`

- test_on_site_fires_on_a_settled_placement drops the moka pot onto the burner from 15 cm and lets it settle; on_site must fire. The first check is False because it is not already solved at reset.

### `tasks/tidy_counter/grade.py`
- The wine bottle must stand within 20 degrees of upright. The two placements count in either order (no needs on the bowl, and the success tree is a plain all, not a sequence). The bowl subtask and the success also need the middle and bottom drawers closed, so a scene left with other drawers open does not score them. The cabinet volume in INVARIANT stops at 0.1 m above the body_region site: a bowl resting on the cabinet top sits at about 0.107 m and must not count as pushed into the cabinet.

### `tasks/tidy_counter/scene.xml`
- The wine bottle weighs 0.75 kg (its geoms' density scaled from LIBERO's 100, which gave 51 g). At 51 g the Panda finger_0 collision hull, which fills the finger's recess, could pinch the neck and lift it with the jaws closed beside the pads; at 0.75 kg those phantom pinches slip while pad grasps still carry it.

### `tasks/tidy_counter/test.py`

- Merged from test_tasks.py (libero/tidy_counter: authored pack and physics checks).
- test_tidy_bowl_invariant_is_the_cabinet_volume_not_a_touch: a grasped bowl brushing the cabinet is manipulation; its centre inside the cabinet's body volume, outside the drawer, is a pass through. The rule is geometric so a 4 mm plate that MuJoCo lets the bowl slip through still counts. The three placements are centred inside the cabinet (a push through), standing on the top plate, and in front of the cabinet where the open drawer carries it.

### `tasks/vortex_sample/scene.xml`

- Every tube collision geom has priority 1, so the tube's own friction 1 governs its pad contacts: the Fable seed 1000000 tube hung at 1:42 from two distal nub edges at the pad friction cap of 2 and now falls. Priority also gives the tube's solref 0.004 and solimp 0.998 to all its contacts (pads, table, rack, cup).

### `tasks/vortex_sample/test.py`

- Merged from test_vortex_driver.py (pack driven motor behavior, independent of the waypoint acceptance route).
- test_vortex_tilt_invariant_latches_failure checks 5 steps: a quarter second past 15 degrees, not a one step bounce.

### `tasks/vortex_sample/dynamics.py`

- Dynamics: four pack owned userdata slots hold count, sampled control tick, liquid response and episode origin. IDs are cached on the instance; all evolving state lives in mjData, so a rewind restores it.
- forces returns the positive normal load from external bodies, and from the named sample.
- physics_step samples the mixer once per control step, holding its motor command between samples.
- paint is visual only (liquid tilt and lamp); physics shapes and mass stay unchanged.

### `tasks/vortex_sample/render.py`

- The microtube look has no opaque override: the sample shows through its alpha.

## `scripts/`

### `scripts/build_reach_voxels.py`

- solve_pose is a local diff IK, so every voxel is tried from each manifest reach.seeds posture and the union stored.
- A reach.mirror_xz robot is mirrored because a seed set that is not mirror closed finds one side and not the other.
- Adjacent links touch at home already, so self collision means a pair that does not touch at rest.

### `scripts/sync_assets.py`

- Push stages its archives under assets/.cache, gitignored and on the same filesystem as the files.

## Task test support

### `tasks/helpers.py`

- One session scoped scene is shared by the tests that read it without changing what later tests rely on.
- POS_TOL is in meters; the name stays because tasks/square/test.py imports it.
- goal_state is the stable diamond goal arrangement from the kernel level probe.
- _press does one full press per repetition (released, down to the stop, back up, checking each step) so the counter sees a release and then a rising edge.
- nodes walks a success tree pre order: the node, its `of` children, then `invariant`.
- _robot_in_contact ignores the held part, which always touches.
- _probe_ik is kinematics only. Reachability belongs to the production controller: its final q is recovered by chaining the same pose samples, so the next candidate is tested from the redundant arm branch a real Cartesian drive will occupy.
- _table_top_z counts the box half height of the tallest table geom.
- ViaInterface is an in process stand in for the via-cam waypoint tools, with no browser. gripper_teleport_via_click puts the clicked point on the approach axis, backed off `retreat` meters; gripper_rotate turns about a world axis through the target position; a positive dist advances along the approach axis, a negative one retreats.
- project emulates a precise click: world point to (u, v) on a camera feed.
- Episode carries sim_env_server's episode loop duties: run to the end and write the video. wait_success holds still until the success tree holds or max_steps runs out.

### `tasks/sim_kernel.py`

- The dev only ground truth calls (object_poses, state) raise in eval mode.
- Pending waypoint targets are keyed by arm name; arm None means the first robot.
- Setting phase to RUNNING is sim_env_server's job in a real episode.
- raycast returns the 3D world point under pixel (u, v) of a camera feed.
