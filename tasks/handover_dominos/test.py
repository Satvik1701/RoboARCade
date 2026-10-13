"""Tests of the handover_dominos task."""
import json
import numpy as np
from core.tasks import CATEGORY_GAME, TASKS
from core.sim_env import SimEnv
import mujoco
from scipy.spatial.transform import Rotation as R
from core import grade_episode as predicates
from core.grade_episode import make_checker
from helpers import (
    _pack_env,
    set_free_body,
    zero_free_qvel,
)


def test_spd_handover_randomizes_one_rigid_pile_per_arm_side():
    task = "handover_dominos"
    spec = json.loads((TASKS / task / "task.json").read_text())
    negative = [f"jenga_{i}" for i in (1, 3, 4, 5, 6, 8, 11, 13, 17)]
    positive = [f"jenga_{i}" for i in (2, 7, 9, 10, 12, 14, 15, 16, 18)]
    blocks = [f"jenga_{i}" for i in range(1, 19)]
    rnd = spec["randomization"]

    assert spec["category"] == CATEGORY_GAME
    assert rnd["full_rand"] is True
    assert rnd["groups"] == [negative, positive]
    assert rnd["placement_regions"] == {
        "jenga_1": [{"x": [0.62, 0.78], "y": [-0.34, -0.32],
                     "yaw_offset_deg": [-180, 180]}],
        "jenga_2": [{"x": [0.62, 0.78], "y": [0.32, 0.34],
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
    assert -0.32 + radii[0] < -0.06
    assert 0.32 - radii[1] > 0.06


def test_spd_domino_pack_needs_a_line_and_propagated_cascade(monkeypatch):
    env = _pack_env("handover_dominos")
    blocks = [f"jenga_{i}" for i in range(1, 19)]
    line = blocks[:5]
    z_stand, z_down = 0.77921 + 0.0603, 0.77921 + 0.0127
    contact_pairs = []
    monkeypatch.setattr(
        predicates, "_contact_body_pairs", lambda _env: iter(contact_pairs))

    def put(body, pos, down=False):
        set_free_body(env, body, pos)
        if down:
            qadr = env.model.joint(env.model.body(body).jntadr[0]).qposadr[0]
            quat = R.from_euler("y", 90, degrees=True).as_quat()
            env.data.qpos[qadr + 3:qadr + 7] = [quat[3], quat[0], quat[1], quat[2]]
        zero_free_qvel(env, body)
        mujoco.mj_forward(env.model, env.data)

    for i, body in enumerate(blocks[5:]):
        put(body, [0.45 + 0.08 * (i % 7), -0.25 + 0.08 * (i // 7), z_down], True)
    for body, pos in zip(line, ([0.50, -0.20, z_stand], [0.62, 0.12, z_stand],
                                [0.75, -0.05, z_stand], [0.88, 0.20, z_stand],
                                [0.98, -0.18, z_stand])):
        put(body, pos)
    assert make_checker(env.spec["success"], env.task)[0](env) is False

    for i, body in enumerate(line):
        put(body, [0.55 + 0.06 * i, 0.0, z_stand])
    check, _ = make_checker(env.spec["success"], env.task)
    assert check(env) is False
    robot = env.model.body("move_link_6").id
    bids = {body: env.model.body(body).id for body in line}
    for i, body in enumerate(line):
        put(body, [0.55 + 0.06 * i, 0.0, z_stand], True)
        contact_pairs[:] = ([(robot, bids[body])] if i == 0 else
                            [(bids[line[i - 1]], bids[body])])
        assert check(env) is (i == len(line) - 1)


def test_domino_chain_requires_one_contact_propagated_knock(monkeypatch):
    env = _pack_env("handover_dominos")
    blocks = [f"jenga_{i}" for i in range(1, 6)]
    z = 0.77921 + 0.0603
    contact_pairs = []
    monkeypatch.setattr(
        predicates, "_contact_body_pairs", lambda _env: iter(contact_pairs))

    def pose(body, x, down=False):
        set_free_body(env, body, [x, 0.0, z])
        if down:
            qadr = env.model.joint(env.model.body(body).jntadr[0]).qposadr[0]
            quat = R.from_euler("y", 90, degrees=True).as_quat()
            env.data.qpos[qadr + 3:qadr + 7] = [quat[3], quat[0], quat[1], quat[2]]
        zero_free_qvel(env, body)
        mujoco.mj_forward(env.model, env.data)

    for i, body in enumerate(blocks):
        pose(body, 0.55 + 0.06 * i)

    spec = {"pred": "domino_chain", "bodies": blocks, "min": 5,
            "min_gap": 0.035, "max_gap": 0.105, "line_tol": 0.020}
    check, _ = make_checker(spec, "handover_dominos")
    assert check(env) is False
    robot = env.model.body("move_link_6").id
    bids = {body: env.model.body(body).id for body in blocks}

    pose(blocks[0], 0.55, down=True)
    contact_pairs[:] = [(robot, bids[blocks[0]])]
    assert check(env) is False
    for i, body in enumerate(blocks[1:], 1):
        pose(body, 0.55 + 0.06 * i, down=True)
        contact_pairs[:] = [(bids[blocks[i - 1]], bids[body])]
        assert check(env) is (i == len(blocks) - 1)

    contact_pairs.clear()
    for i, body in enumerate(blocks):
        pose(body, 0.55 + 0.06 * i)
    check, _ = make_checker(spec, "handover_dominos")
    assert check(env) is False
    pose(blocks[0], 0.55, down=True)
    contact_pairs[:] = [(robot, bids[blocks[0]])]
    assert check(env) is False
    pose(blocks[1], 0.61, down=True)
    contact_pairs[:] = [(robot, bids[blocks[1]]), (bids[blocks[0]], bids[blocks[1]])]
    assert check(env) is False and check.dead
    for i, body in enumerate(blocks[2:], 2):
        pose(body, 0.55 + 0.06 * i, down=True)
        contact_pairs[:] = [(bids[blocks[i - 1]], bids[body])]
        assert check(env) is False
