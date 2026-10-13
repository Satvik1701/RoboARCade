"""Stress test of init randomization: re-roll a task and check nothing interpenetrates."""
import sys

import mujoco
import numpy as np

from core.sim_env import SimEnv


def worst_penetration_mm(env):
    m, d, fb = env.model, env.data, env._free_body_ids
    mujoco.mj_forward(m, d)
    worst = 0.0
    for c in d.contact[: d.ncon]:
        b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        if b1 != b2 and b1 in fb and b2 in fb:
            worst = min(worst, float(c.dist))
    return -worst * 1000


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "bowl_on_plate"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    env = SimEnv(task, seed=0)
    env.reset(seed=0)
    fails, worst = 0, 0.0
    for s in range(n):
        env._rng = np.random.default_rng(s)
        env._place_objects(True)
        env.data.qvel[:] = 0
        mujoco.mj_forward(env.model, env.data)
        env.settle_until_rest()
        pen = worst_penetration_mm(env)
        worst = max(worst, pen)
        if pen > 2.0:
            fails += 1
            print(f"  [seed {s}] INTERSECTION {pen:.1f}mm", flush=True)
        if (s + 1) % 100 == 0:
            print(f"{s + 1}/{n}: {fails} intersections, worst {worst:.2f}mm", flush=True)
    print(f"DONE {task}: {n} rerolls, {fails} intersections (>2mm), worst {worst:.2f}mm")


if __name__ == "__main__":
    main()
