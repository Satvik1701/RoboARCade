"""Tests of the jenga_tower task."""
import json
import numpy as np
from core.tasks import CATEGORY_GAME, TASKS
from core.sim_env import SimEnv
import mujoco
from core.grade_episode import make_checker
from helpers import (
    _pack_env,
    zero_free_qvel,
)
from helpers import (
    ROYGBIV,
)


def test_spd_tower_randomizes_two_rigid_piles_and_keeps_build_strip_clear():
    task = "jenga_tower"
    spec = json.loads((TASKS / task / "task.json").read_text())
    negative = [f"jenga_{i}" for i in (1, 2, 3, 10, 11, 12, 13, 14, 15, 16, 17, 18)]
    positive = [f"jenga_{i}" for i in (4, 5, 6, 7, 8, 9)]
    blocks = [f"jenga_{i}" for i in range(1, 19)]
    rnd = spec["randomization"]

    assert spec["category"] == CATEGORY_GAME
    assert rnd["full_rand"] is True
    assert rnd["groups"] == [negative, positive]
    assert rnd["placement_regions"] == {
        "jenga_1": [{"x": [0.62, 0.74], "y": [-0.44, -0.36],
                     "yaw_offset_deg": [-180, 180]}],
        "jenga_4": [{"x": [0.62, 0.74], "y": [0.34, 0.39],
                     "yaw_offset_deg": [-180, 180]}],
    }
    assert rnd["lift_reach"] == {block: 0.14 for block in blocks}
    assert rnd["min_sep"] == [[a, b, 0.1] for a in negative for b in positive]

    env = SimEnv(task, seed=0)
    env.reset(seed=0, randomize_init=False)
    radii = []
    for pile in (negative, positive):
        pivot = np.mean([env.data.body(block).xpos[:2] for block in pile], axis=0)
        points = np.concatenate([env._xy_points(block) for block in pile])
        radii.append(float(np.linalg.norm(points - pivot, axis=1).max()))
    assert -0.36 + radii[0] < -0.10
    assert 0.34 - radii[1] > 0.10


def test_spd_tower_rejects_separate_columns_and_accepts_supported_tower():
    env = _pack_env("jenga_tower")
    blocks = [f"jenga_{i}" for i in range(1, 19)]
    table = 0.77921
    qx = [2 ** -0.5, 0, 2 ** -0.5, 0]
    qy = [0.5, -0.5, 0.5, 0.5]

    def put(body, pos, quat):
        qadr = env.model.joint(env.model.body(body).jntadr[0]).qposadr[0]
        env.data.qpos[qadr:qadr + 3] = pos
        env.data.qpos[qadr + 3:qadr + 7] = quat
        zero_free_qvel(env, body)

    assert make_checker(env.spec["success"], env.task)[0](env) is False
    for layer in range(6):
        quat = qx if layer % 2 == 0 else qy
        for column, body in enumerate(blocks[3 * layer:3 * layer + 3]):
            put(body, [0.55 + 0.20 * column, 0.0,
                       table + 0.0127 + 0.0254 * layer], quat)
    mujoco.mj_forward(env.model, env.data)
    assert make_checker(env.spec["success"], env.task)[0](env) is False

    for layer in range(4):
        along_x = layer % 2 == 0
        quat = qx if along_x else qy
        for offset, body in zip((-0.04064, 0.0, 0.04064),
                                blocks[3 * layer:3 * layer + 3]):
            pos = [0.75, 0.0, table + 0.0127 + 0.0254 * layer]
            pos[1 if along_x else 0] += offset
            put(body, pos, quat)
    for k, body in enumerate(blocks[12:]):
        put(body, [0.50 + 0.06 * k, -0.35, table + 0.0127], qy)
    mujoco.mj_forward(env.model, env.data)
    for _ in range(10):
        env._step_physics()
    assert make_checker(env.spec["success"], env.task)[0](env) is True


def _put(env, body, pos):
    j = env.model.joint(env.model.body(body).jntadr[0])
    env.data.qpos[j.qposadr[0]:j.qposadr[0] + 3] = pos
    env.data.qvel[j.dofadr[0]:j.dofadr[0] + 6] = 0.0
    mujoco.mj_forward(env.model, env.data)


def test_count_is_at_least_k_of_n():
    env = _pack_env("shell_game")
    cubes = ROYGBIV[:5]
    spec = {"pred": "count", "min": 3,
            "of": [{"pred": "lift_above", "obj": c, "z_above": 0.90} for c in cubes]}
    check, _ = make_checker(spec, "jenga_tower")
    assert check(env) is False
    for c in cubes[:2]:
        _put(env, c, [0.02, 0.0, 0.95])
    assert check(env) is False
    _put(env, cubes[2], [0.02, 0.1, 0.95])
    assert check(env) is True
