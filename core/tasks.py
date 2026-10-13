"""Task folders: ids, discovery and mesh locations.

Also adds the camera gallery every scene gets on top of the cameras a task authors."""
import functools
import importlib
import json
import os
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(os.environ.get("ROBOARCADE_ASSETS_DIR", ROOT / "assets"))
TASKS = ROOT / "tasks"

RING_N = 3
RING_PHASE = 90.0
EYELEVEL_ELEV_DEG = 10.0
FARSIDE_BACK_M = 0.11
FARSIDE_UP_M = 0.065
FARSIDE_FOVY = 50.0
FARSIDE_AIM_IN_M = 0.23
FARSIDE_AIM_UP_M = 0.016

CATEGORY_REAL_WORLD, CATEGORY_GAME, CATEGORY_IN_PROGRESS = "real_world", "game", "in_progress"
TASK_CATEGORIES = frozenset((CATEGORY_REAL_WORLD, CATEGORY_GAME, CATEGORY_IN_PROGRESS))


@functools.lru_cache(maxsize=None)
def robot_kinematics(model: str) -> dict:
    return json.loads((ASSETS / "robots" / model / "manifest.json").read_text())["kinematics"]


def camera_name(camera: dict) -> str:
    return (camera["name"] if "robot" not in camera
            else f'{camera["robot"]}_{camera["name"]}')


def list_tasks() -> list[str]:
    return sorted(p.parent.name for p in TASKS.glob("*/task.json"))


def resolve_task(task: str) -> Path:
    pack = TASKS / task
    if not (pack / "task.json").exists():
        raise ValueError(f"unknown task {task!r}; available: {list_tasks()}")
    return pack


def task_module(task: str, name: str):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return importlib.import_module(f"tasks.{task}.{name}")


def asset_dir(pack: Path) -> Path:
    pack = Path(pack).resolve()
    if pack.is_relative_to(ASSETS.resolve()):
        return pack
    spec = json.loads((pack / "task.json").read_text())
    return ASSETS / "tasks" / spec["family"] / spec["name"]


def load_scene(pack: Path) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(Path(pack) / "scene.xml"))
    root = asset_dir(pack)
    spec.meshdir, spec.texturedir = (os.path.normpath(root / d) for d in (spec.meshdir, spec.texturedir))
    return spec


def _look_at_wxyz(pos, target) -> list[float]:
    zc = np.asarray(pos, float) - np.asarray(target, float)
    zc /= np.linalg.norm(zc)
    xc = np.cross([0.0, 0.0, 1.0], zc)
    xc /= np.linalg.norm(xc)
    q = R.from_matrix(np.column_stack([xc, np.cross(zc, xc), zc])).as_quat()
    return [q[3], q[0], q[1], q[2]]


def _add_camera(mspec, name, pos, quat, fovy):
    if any(c.name == name for c in mspec.cameras):
        raise ValueError(
            f"generated camera {name!r} collides with one the pack's scene.xml declares; "
            f"rename the generated one in core/tasks.py")
    cam = mspec.worldbody.add_camera()
    cam.name, cam.pos, cam.quat, cam.fovy = name, pos, quat, fovy


def add_cameras(mspec, spec, task) -> list[str]:
    pc_names = {c["name"] for c in spec["cameras"] if c["role"] == "pc"}
    crop = spec["pc_crop"]
    cx, cy = float(np.mean(crop["x"])), float(np.mean(crop["y"]))
    center = np.array([cx, cy, float(np.mean(crop["z"]))])
    pc_cams = [c for c in mspec.cameras if c.name in pc_names]
    nested = [c.name for c in pc_cams if c.parent is not None and c.parent.name != "world"]
    if nested:
        raise ValueError(
            f"[{task}] pc cameras {nested} are mounted inside a body; the point-cloud "
            f"ring is built from their pos, which must be in WORLD coordinates. "
            f"Declare them directly under <worldbody>.")
    ring_names = []
    if pc_cams:
        radius = float(np.mean([np.hypot(c.pos[0] - cx, c.pos[1] - cy) for c in pc_cams]))
        height = float(np.mean([c.pos[2] for c in pc_cams]))
        fovy = float(np.mean([c.fovy for c in pc_cams]))
        for k in range(RING_N):
            a = np.radians(RING_PHASE) + k * 2 * np.pi / RING_N
            pos = np.array([cx + radius * np.cos(a), cy + radius * np.sin(a), height])
            ring_names.append(f"ring{k}")
            _add_camera(mspec, ring_names[-1], pos.tolist(), _look_at_wxyz(pos, center), fovy)
    bz_fovy = 60.0
    half = max(float(crop["x"][1]) - float(crop["x"][0]),
               float(crop["y"][1]) - float(crop["y"][0])) / 2 + 0.1
    for name, h in (("birdseye", half), ("zoomed_birdseye", half / 2)):
        _add_camera(mspec, name,
                    [cx, cy, float(crop["z"][1]) + h / np.tan(np.radians(bz_fovy / 2))],
                    [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)], bz_fovy)
    eye_d = half / np.tan(np.radians(bz_fovy / 2))
    eye_pos = center + eye_d * np.array(
        [np.cos(np.radians(EYELEVEL_ELEV_DEG)), 0.0, np.sin(np.radians(EYELEVEL_ELEV_DEG))])
    _add_camera(mspec, "eyelevel", eye_pos.tolist(), _look_at_wxyz(eye_pos, center), bz_fovy)
    table_z = float(spec.get("table_z", crop["z"][0]))
    side_pos = [cx, float(crop["y"][1]) + FARSIDE_BACK_M, table_z + FARSIDE_UP_M]
    side_target = np.array([cx, float(crop["y"][1]) - FARSIDE_AIM_IN_M, table_z + FARSIDE_AIM_UP_M])
    _add_camera(mspec, "farside", side_pos, _look_at_wxyz(np.array(side_pos), side_target),
                FARSIDE_FOVY)
    return ring_names
