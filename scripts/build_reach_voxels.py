"""Precomputes each robot's reachable volume as a voxel grid.

Precompute a robot's reachable volume as a voxel grid in BASE frame (per robot model,
scene obstacles not baked in). A voxel is reachable if IK lands the fingertip site there
within ARRIVE_TOL at a probe orientation without new self-contact.

    python scripts/build_reach_voxels.py [--model franka_panda] [--step 0.02]

Writes assets/robots/<model>/reach.npz. Runtime ~1 min on 32 cores.
"""
import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from core.control import WaypointController, make_arm_ik
from core.tasks import ASSETS, robot_kinematics

ARRIVE_TOL = WaypointController.ARRIVE_TOL
_PER_PROCESS = {}


def _per_process(model_path, home, kin):
    if "ik" not in _PER_PROCESS:
        spec = mujoco.MjSpec.from_file(model_path)
        model = spec.compile()
        _PER_PROCESS["model"] = model
        _PER_PROCESS["data"] = mujoco.MjData(model)
        ik = _PER_PROCESS["ik"] = make_arm_ik(model, "", np.asarray(home, float), kin)
        d = _PER_PROCESS["data"]
        d.qpos[ik.qadr] = ik.q_home
        mujoco.mj_forward(model, d)
        _PER_PROCESS["pairs_at_home"] = {(min(c.geom1, c.geom2), max(c.geom1, c.geom2))
                                         for c in d.contact[: d.ncon]}
    return _PER_PROCESS["model"], _PER_PROCESS["data"], _PER_PROCESS["ik"], _PER_PROCESS["pairs_at_home"]


def _slab(args):
    model_path, home, kin, xs, ys, z = args
    model, data, ik, pairs_at_home = _per_process(model_path, home, kin)
    out = np.zeros((len(xs), len(ys)), dtype=bool)
    probe_eulers = kin["reach"].get("probe_euler") or kin["home_euler"]
    if not isinstance(probe_eulers[0], (list, tuple)):
        probe_eulers = [probe_eulers]
    goal_rots = [R.from_euler("xyz", e) for e in probe_eulers]
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            tgt = np.array([x, y, z])
            for goal_rot, seed in [(g, s) for g in goal_rots
                                   for s in kin["reach"]["seeds"]]:
                q = ik.solve_pose(np.asarray(seed, float), tgt, goal_rot)
                if not np.all(np.isfinite(q)):
                    continue
                data.qpos[ik.qadr] = q
                mujoco.mj_forward(model, data)
                if np.linalg.norm(data.site_xpos[ik.site_id] - tgt) >= ARRIVE_TOL:
                    continue
                pairs = {(min(c.geom1, c.geom2), max(c.geom1, c.geom2))
                         for c in data.contact[: data.ncon]}
                if not (pairs - pairs_at_home):
                    out[i, j] = True
                    break
    return z, out


def build(model_name: str, step: float, reach: float, z_lo: float, z_hi: float):
    kin = robot_kinematics(model_name)
    model_path = str(ASSETS / "robots" / model_name / kin["mjcf"])
    home = kin["home_qpos"]

    xs = np.arange(-reach, reach + 1e-9, step)
    ys = np.arange(-reach, reach + 1e-9, step)
    zs = np.arange(z_lo, z_hi + 1e-9, step)
    print(f"{model_name}: {len(xs)}x{len(ys)}x{len(zs)} = {len(xs)*len(ys)*len(zs):,} voxels")

    t0 = time.time()
    jobs = [(model_path, home, kin, xs, ys, float(z)) for z in zs]
    grid = np.zeros((len(xs), len(ys), len(zs)), dtype=bool)
    with ProcessPoolExecutor() as pool:
        for k, (z, slab) in enumerate(pool.map(_slab, jobs)):
            grid[:, :, int(round((z - z_lo) / step))] = slab
            print(f"  z={z:+.2f}  {slab.mean()*100:5.1f}% reachable   [{k+1}/{len(zs)}]", flush=True)

    if kin["reach"]["mirror_xz"]:
        grid |= grid[:, ::-1, :]

    out = ASSETS / "robots" / model_name / "reach.npz"
    np.savez_compressed(out, grid=np.packbits(grid), shape=grid.shape,
                        origin=[-reach, -reach, z_lo], step=step)
    print(f"\nwrote {out} ({out.stat().st_size/1024:.0f} KB) in {time.time()-t0:.0f}s")
    print(f"reachable fraction: {grid.mean()*100:.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="franka_panda")
    ap.add_argument("--step", type=float, default=0.02)
    ap.add_argument("--reach", type=float, default=0.90)
    ap.add_argument("--z-lo", type=float, default=-0.30)
    ap.add_argument("--z-hi", type=float, default=0.30)
    a = ap.parse_args()
    build(a.model, a.step, a.reach, a.z_lo, a.z_hi)
