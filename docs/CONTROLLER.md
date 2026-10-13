# Controller

How one blue target pose becomes joint torques (`core/control.py`; the servos are in
`assets/robots/<model>/*.xml`).

```
blue pose ──► straight line + slerp, paced by a 1-DOF ruckig OTG (per control step)
          ──► solve_pose IK (mink QP, nearest branch), joint v/a caps
          ──► position actuators + gravity feedforward (per physics substep) ──► torque
```

Rates: physics `dt = 0.002 s`; control at the pack's `control_freq` (20 Hz → 25 substeps
per control step). The OTG and IK run once per control step, the servos every substep.

## Path — `WaypointController._pose_steps`

The waypoint is interpolated **in pose space**: `p(s) = p0 + s·d`, `R(s) = slerp(r0, rt, s)`
for a scalar `s ∈ [0, 1]` driven by a one-dimensional jerk-limited ruckig profile. Its
limits are `_LIN_LIMS` / `_ROT_LIMS` divided by the move's distance / angle (the tighter
wins), capped so a mm-scale move still spans `MIN_WAYPOINT_CYCLES` control steps (v ≈
1/distance otherwise hits a numerical corner in ruckig). Pose interpolation, not joint
interpolation, is what keeps a carried part off the table on an in-place rotation.

## IK — `ArmIKMink.solve_pose`

Each control step the interpolated pose is IK-solved by a mink QP (FrameTask on
`grip_site` + posture regularizer toward the seed + hard joint limits), iterated to
convergence from the current reference `q_ref`. Steps are clamped to `DQ_MAX`, so the
solve returns the branch **nearest** the seed and cannot flip the arm.

The joint step `q_want − q_ref` is then scaled — as one factor, direction untouched — to
`_JOINT_LIMS` (v = 2 rad/s, a = 6 rad/s²). While the joints cannot keep up the path
parameter **waits**, so the reference never runs ahead and makes the arm jump.

## Servo — position actuators + gravity feedforward

`q_ref` goes straight to the `position` actuators with a feedforward that nulls the PD
droop (~8 mm open-loop, ~0.1 mm with it):

```
ctrl = q_ref + (qfrc_bias − qfrc_gravcomp) / kp
```

`qfrc_gravcomp` comes off because a gravcomp model (YAM) already carries it. Panda gains
(`panda.xml`, `biasprm="0 -kp -kv"`): joints 1–4 kp 4500–3500 / kv 450–350, joints 5–7
kp 2000 / kv 200; the gripper is one more position servo.

## Arrival and retries

- `_drive_to`: once the OTG reports finished the servo settles (capped at 25 steps) and the
  drive returns arrived iff position error < `ARRIVE_TOL` (2 mm) **and** rotation error <
  `ARRIVE_ROT_TOL` (0.03 rad — without it a stalled in-place turn scores as arrived). It
  bails early when the kinematic residual exceeds `TRACK_TOL` (1 cm) or the sim goes
  non-finite.
- `update_pose`: an unreachable target (`_ik_reaches`, obstacle-blind, from home) gets one
  best-effort drive. Otherwise up to `RESOLVE_TRIES` re-drives re-solved from where the arm
  ended up, stopping when a retry gains < 1 mm or a **new** contact appeared (a
  descend-until-contact grasp must never be fought). Still ≥ 5 cm short with no new
  contact: the IK branch is pinned at a limit, so `_reconfigure_home` ramps to the
  limit-free home posture over 100 substeps and drives once more.

## Pose hold and the gripper

`ArmIKMink.substep` is a per-substep diff-IK closed on measured `qpos` (gain 0.3, `LEASH`
anti-windup): it stalls softly on contact, which deep grasps depend on. Waypoint motion
does not use it; `apply_action` / `update_gripper` hold a pose with it while the jaws move,
and `sim_env_server.probe_reachable` uses it for the reachability tint.

`update_gripper`: an open ramps over 8 steps; a close runs until the width and the
self-centring admittance are both still for 5 steps. The admittance is 1-DOF along the jaw
axis (`B·q̇ = F`, B = 2500 N·s/m), engaged after |F| > 10 N for 0.2 s (free objects drag
< 6 N, fixed ones splay ≥ 17 N), bounded to ±15 mm, referenced to the close-start pose so
disturbances never ratchet in.

`release_and_settle` (post-success finalize): open, then hold in place for
`settle_steps`; no homing sweep (a straight line home once yanked a half-open drawer fully
open). The caller pairs the re-check with `robot_clear()`.

## Constants

| Symbol | Value | Where |
|---|---|---|
| physics dt / substeps | 0.00025 s / 200 | `sim_env.PHYSICS_DT` or the pack's `physics.timestep`, pack `control_freq` |
| `_LIN_LIMS` v / a / j | 1 m/s, 5 m/s², 50 m/s³ | `control.py` |
| `_ROT_LIMS` v / a / j | 0.7, 2, 20 (rad) | `control.py` (a 90° flip at 1.5 rad/s shook a held part out) |
| `_JOINT_LIMS` v / a | 2 rad/s, 6 rad/s² | `control.py` |
| `DQ_MAX` / `LEASH` | 0.05 rad / 0.05 rad | `ArmIKMink` |
| `ARRIVE_TOL` / `ARRIVE_ROT_TOL` | 2 mm / 0.03 rad | `WaypointController` |
| `TRACK_TOL` | 1 cm | `WaypointController` |
| settle cap | 25 control steps | `_drive_to` |
| `waypoint_max_step` | pack `control` block | |

Verification: the per-task `tasks/<task>/test.py` files cover grip retention under jaw-axis
spins, small waypoints and articulated-object contacts.
