"""Tests of the tidy_counter task."""



def test_tidy_bowl_invariant_is_the_cabinet_volume_below_its_top():
    import mujoco
    import numpy as np
    from core.grade_episode import make_checker
    from core.sim_env import SimEnv
    env = SimEnv("tidy_counter", seed=0)
    env.reset(seed=0)
    inside, _ = make_checker({"pred": "over_site", "obj": "akita_black_bowl_1_main", "site": "wooden_cabinet_1_body_region",
                              "lo": [-0.127, -0.1105, -0.109], "hi": [0.127, 0.1105, 0.1]})
    m, d = env.model, env.data
    adr = m.joint("akita_black_bowl_1_joint0").qposadr[0]
    site = d.site("wooden_cabinet_1_body_region")
    R, c0 = site.xmat.reshape(3, 3).copy(), site.xpos.copy()

    def place(rel):
        d.qpos[adr:adr + 3] = c0 + R @ np.asarray(rel)
        d.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(m, d)

    place([0.0, 0.0, 0.05])
    assert inside(env)
    place([0.0, 0.0, 0.107])
    assert not inside(env)
    place([0.0, 0.0, 0.25])
    assert not inside(env)
    place([0.0, -0.25, 0.05])
    assert not inside(env)
