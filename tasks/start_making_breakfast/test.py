"""Tests of the start_making_breakfast task.

"""
import mujoco
from core.grade_episode import make_checker
from core.sim_env import SimEnv


def test_on_site_fires_on_a_settled_placement():
    env = SimEnv("start_making_breakfast", seed=1)
    env.reset(seed=1)
    leaf = next(p for p in env.spec["success"]["of"] if p["pred"] == "on_site")
    check, _ = make_checker(leaf)
    assert check(env) is False

    adr = env._obj_qadr[leaf["obj"]]
    target = env.data.site(leaf["site"]).xpos.copy()
    env.data.qpos[adr : adr + 3] = [target[0], target[1], target[2] + 0.15]
    env.data.qpos[adr + 3 : adr + 7] = [1, 0, 0, 0]
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)
    for _ in range(2000):
        mujoco.mj_step(env.model, env.data)
    assert check(env) is True
