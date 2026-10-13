"""Small shared helpers.

Interface package lookup and manifest, episode identity, logging setup, depth deprojection, camera
intrinsics and JSON reads that return None on failure."""
import json
import logging
import os
import sys
import urllib.request

from pathlib import Path

import numpy as np


def interface_dir(spec: str) -> Path:
    path = Path(spec)
    if os.path.sep in spec or path.is_dir():
        return path.resolve()
    return Path(__file__).resolve().parents[1] / "interfaces" / spec.replace("-", "_")


def interface_manifest(spec: str) -> dict:
    path = interface_dir(spec) / "manifest.json"
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid interface manifest: {path}") from exc


def interface_arm_count(spec: str) -> int:
    arms = interface_manifest(spec).get("arms", 1)
    if isinstance(arms, bool) or not isinstance(arms, int) or arms < 1:
        raise ValueError(f"invalid arms count in the manifest of interface {spec}")
    return arms


def episode_identity(folder: str | Path):
    episode_path = Path(folder) / "episode.json"
    try:
        episode = json.loads(episode_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"episode identity is missing or invalid: {episode_path}") from exc
    task = episode.get("task")
    seed = episode.get("seed")
    if not task or seed is None:
        raise ValueError(f"episode identity is missing task or seed: {episode_path}")
    return task, seed


def wrap_ruler(text: str, max_len=40):
    if len(text) > max_len:
        return text
    left = (max_len - len(text)) // 2
    return "=" * left + text + "=" * (max_len - len(text) - left)


LOG_FORMAT = "[%(asctime)s %(levelname)s %(filename)s:%(lineno)d] %(message)s"
DATE_FORMAT = "%H:%M:%S"
_QUIET_LOGGERS = ("asyncio", "websockets", "httpx", "httpcore", "PIL")


def setup_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    root = logging.getLogger()
    for existing in root.handlers[:]:
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def deproject(depth_m: np.ndarray, K: np.ndarray, cam_to_world: np.ndarray) -> np.ndarray:
    h, w = depth_m.shape
    i, j = np.indices((h, w))
    pix = np.stack([j.ravel(), i.ravel(), np.ones(h * w)], axis=0)
    pts = np.linalg.inv(K) @ pix * depth_m.ravel()
    pts = np.vstack([pts, np.ones(h * w)])
    return (cam_to_world @ pts).T[:, :3]


def camera_intrinsics(fovy_deg: float, h: int, w: int) -> np.ndarray:
    f = h / (2.0 * np.tan(np.radians(fovy_deg) / 2.0))
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])


def get_json(url: str, timeout: float = 1.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.load(resp)
    except Exception:
        return None


def get_json_file(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None
