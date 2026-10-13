"""Tests of the set_up_pcr_run task."""
import numpy as np
from scipy.spatial.transform import Rotation as R
from core.sim_env import SimEnv



PCR_TASK = "set_up_pcr_run"


def test_nested_articulated_lid_is_one_payload_and_self_contact_is_not_an_obstacle():
    env = SimEnv(task=PCR_TASK, seed=1)
    env.reset(seed=1, randomize_init=True)
    env.max_len = 10**9
    arm, m, d = env.arms[0], env.model, env.data
    lid = m.body("lid").id
    corners = np.array([[i, j, k] for i in (-1, 0, 1)
                        for j in (-1, 0, 1) for k in (-1, 0, 1)])
    cloud = []
    for g in range(m.ngeom):
        if m.geom_group[g] == 3 and env._ancestor_in(int(m.geom_bodyid[g]), {lid}):
            centre, half = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
            cloud.append((centre + corners * half) @ d.geom_xmat[g].reshape(3, 3).T
                         + d.geom_xpos[g])
    points = np.vstack(cloud)
    start = np.array([-0.12, points[:, 1].max() - 0.028,
                      points[:, 2].max() - 0.021])
    down = R.from_euler("xyz", [np.pi, 0, 0]).as_euler("xyz")
    env.move_to(start + [0, 0, 0.06], down, 0)
    env.move_to(start, down, 0)
    for s in np.arange(0.01, 0.06, 0.01):
        env.move_to(start - [0, s, 0.30 * s], down, 0)

    touched = {f"{arm}_{side}_finger": set() for side in ("left", "right")}
    for contact in d.contact[: d.ncon]:
        for finger, other in ((contact.geom1, contact.geom2),
                              (contact.geom2, contact.geom1)):
            name = m.body(int(m.geom_bodyid[finger])).name or ""
            other_body = int(m.geom_bodyid[other])
            if name in touched and env._ancestor_in(other_body, {lid}):
                touched[name].add(m.body(other_body).name)
    assert "lid" in touched[f"{arm}_left_finger"]
    assert "lid_lever" in touched[f"{arm}_right_finger"]

    same_arm_geoms = {g for g in range(m.ngeom)
                      if (m.body(int(m.geom_bodyid[g])).name or "").startswith(arm + "_")}
    assert not env.ctl._arm_contacts(arm) & same_arm_geoms
    lid_geoms = {g for g in range(m.ngeom)
                 if env._ancestor_in(int(m.geom_bodyid[g]), {lid})}
    assert env.ctl._held_payload_geoms(arm) == lid_geoms


