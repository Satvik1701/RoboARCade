"""Physical and visual regression checks for the fitted mating interfaces."""

import mujoco
import numpy as np
import pytest

from core.tasks import TASKS, load_scene
from core.sim_env import SimEnv
from core.grade_episode import make_checker
from tasks.add_task.fit_assembly_collisions import CONFIG


@pytest.mark.parametrize("task", list(CONFIG))
def test_interface_surface_matches_visual_with_configured_clearance(task):
    cfg = CONFIG[task]
    model = load_scene(TASKS / task).compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for name in (cfg["target"], cfg["receiver"]):
        body = model.body(name).id
        model.geom_group[:] = 5
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] == body:
                model.geom_group[gid] = (
                    3 if model.geom_contype[gid] or model.geom_conaffinity[gid] else 1
                )
                model.geom_rgba[gid, 3] = 1
        target = name == cfg["target"]
        xy = np.zeros(2) if target else np.array(cfg["xy"])
        z0, z1 = (
            (cfg["peg_bottom"], cfg["peg_top"])
            if target
            else (cfg["hole_bottom"], cfg["hole_top"])
        )
        for z in np.linspace(z0 + 0.001, z1 - 0.001, 7):
            # Avoid casting exactly along polygon seams, where a ray/triangle
            # intersection can be numerically ambiguous.
            for angle in np.linspace(0, 2 * np.pi, 128, endpoint=False) + 0.012345:
                ray = np.array([np.cos(angle), np.sin(angle), 0])
                origin = np.array([*xy, z])
                if target:
                    origin += 0.02 * ray
                    ray = -ray
                rotation = data.xmat[body].reshape(3, 3)
                hits = []
                for group in (1, 3):
                    mask = np.zeros(6, dtype=np.uint8)
                    mask[group] = 1
                    hits.append(
                        mujoco.mj_ray(
                            model,
                            data,
                            data.xpos[body] + rotation @ origin,
                            rotation @ ray,
                            mask,
                            True,
                            -1,
                            np.array([-1], dtype=np.int32),
                        )
                    )
                assert min(hits) >= 0, "Missing visual or collision surface"
                # 30 um allows polygon approximation and source mesh rounding.
                assert abs(hits[1] - hits[0]) < cfg["erosion"] + 0.00003


def test_manifold_patch_preserves_chamfered_outer_corners():
    model = load_scene(TASKS / "cooling_manifold").compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    body = model.body("part1").id
    model.geom_group[:] = 5
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] == body:
            model.geom_group[gid] = (
                3 if model.geom_contype[gid] or model.geom_conaffinity[gid] else 1
            )
            model.geom_rgba[gid, 3] = 1
    cfg = CONFIG["cooling_manifold"]
    # Well outside the round hole, inside the replacement square, but beyond
    # the diagonal outside corner. Both visual and collision rays must miss.
    for x, y in [(0.0123, 0.0123), (-0.0123, -0.0123)]:
        origin = np.array([cfg["xy"][0] + x, cfg["xy"][1] + y, 0.06])
        rotation = data.xmat[body].reshape(3, 3)
        for group in (1, 3):
            mask = np.zeros(6, dtype=np.uint8)
            mask[group] = 1
            hit = mujoco.mj_ray(
                model,
                data,
                data.xpos[body] + rotation @ origin,
                rotation @ np.array([0.0, 0.0, -1.0]),
                mask,
                True,
                -1,
                np.array([-1], dtype=np.int32),
            )
            assert hit < 0, "Socket patch fills an empty chamfered corner"


@pytest.mark.parametrize("task", list(CONFIG))
def test_scripted_pickup_and_lift(task):
    from core.grade_episode import _grasped_by
    from tasks.cooling_manifold.test import _spawn_piece, _grasp_piece

    env = SimEnv(task, seed=0)
    env.reset(seed=0, randomize_init=False)
    env.max_len = 10**9
    piece, arm = CONFIG[task]["target"], env.arms[0]
    _spawn_piece(env, piece, 0.06, -0.12, 0.0)
    height = env.data.body(piece).xpos[2]
    _grasp_piece(env, arm, piece)
    assert _grasped_by(env, piece, arm)
    assert env.data.body(piece).xpos[2] > height + 0.05


@pytest.mark.parametrize("task", list(CONFIG))
@pytest.mark.parametrize("seed", range(1000000, 1000005))
def test_insertion_path_and_free_release(task, seed):
    cfg = CONFIG[task]
    env = SimEnv(task, seed=seed)
    env.reset(seed=seed)
    model, data = env.model, env.data
    target, base = model.body(cfg["target"]).id, model.body(cfg["receiver"]).id
    qadr = int(model.jnt_qposadr[model.body_jntadr[target]])
    vadr = int(model.jnt_dofadr[model.body_jntadr[target]])
    goal = np.array(env.spec["success"]["of"][0]["pos"])

    def place(relative):
        data.qpos[qadr : qadr + 3] = (
            data.xpos[base] + data.xmat[base].reshape(3, 3) @ relative
        )
        data.qpos[qadr + 3 : qadr + 7] = data.xquat[base]
        data.qvel[vadr : vadr + 6] = 0
        mujoco.mj_forward(model, data)

    def depths():
        return [
            c.dist
            for c in data.contact
            if {int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])}
            == {target, base}
        ]

    for dz in np.linspace(0.03, 0, 61):
        place(goal + [0, 0, dz])
        assert (
            min(depths(), default=0) > -0.00005
        ), "A collision hull blocks the aligned insertion"
    place(goal + [0.006, 0, 0])
    assert min(depths(), default=0) < -0.001, "Socket must block a misaligned insertion"
    place(goal + [0, 0, 0.003])
    mujoco.mj_step(model, data, nstep=round(2 / model.opt.timestep))
    assert np.all(np.isfinite(data.qpos))
    assert make_checker(env.spec["success"])[0](env)
    relative = data.xmat[base].reshape(3, 3).T @ (data.xpos[target] - data.xpos[base])
    assert np.linalg.norm(relative - goal) < 0.0015
