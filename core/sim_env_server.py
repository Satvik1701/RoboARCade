"""The simulator as a server for one episode: API, episode loop, recording.

Runs sim_env.SimEnv, takes waypoints and the episode's end over its command WebSocket, serves its state as JSON
(env.json, robot.json, rgbd.json, camera frames), streams frames, and records the episode folder (video, saved states,
verdict.json). The browser page and its Chromium come from ui_for_agent.py; with --no-ui an interface uses the API alone.

    python -m core.sim_env_server --task <task> --teleop      # a person drives it from a browser
"""
from __future__ import annotations
import base64
import glob
import json
import logging
import os
import queue
import shutil
import signal
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import argparse
import asyncio
import websockets
from websockets.http11 import Response
from websockets.datastructures import Headers
from urllib.parse import parse_qs
from scipy.spatial.transform import Rotation as R
import msgpack
import mujoco
import numpy as np
import cv2

from core import grade_episode, render
from core import utils
from core import ui_for_agent
from core.initial_layout import init_regions, place_at_extreme, systematic_extremes
from core.control import make_arm_ik
from core.episode_recorder import EpisodeRecorder, ActMode
from core.tasks import list_tasks
from core.sim_env import Phase, SimEnv

logger = logging.getLogger(__name__)

BASE_PORT = int(os.environ.get("ROBOARCADE_BASE_PORT", "8100"))
HTTP_PORT = BASE_PORT
BROADCAST_PORT = BASE_PORT + 1
UI_LISTEN_PORT = BASE_PORT + 2

FINAL_SNAP_HOLD_S = 15.0



def _next_demo_idx(data_folder: str) -> int:
    idx = 0
    for entry in glob.glob(os.path.join(data_folder, "demo[0-9]*")):
        tail = os.path.basename(entry)[len("demo"):]
        if os.path.isdir(entry) and tail.isdigit():
            idx = max(idx, int(tail) + 1)
    return idx


def _empty(code: int, reason: str) -> Response:
    return Response(code, reason, Headers([("Content-Length", "0"),
                                           ("Access-Control-Allow-Origin", "*")]), b"")


def _ui_euler_to_robot(rot) -> np.ndarray:
    return np.array([rot[0] - np.pi, -rot[2], rot[1]])


class InteractiveBot:
    def __init__(
        self,
        task,
        num_point,
        stream_freq,
        data_root,
        seed,
        demo_folder: None | str,
        teleop: bool = False,
        dashboard: bool = False,
        record_state: bool = False,
        ui: bool = True,
        rgbd: bool = False,
    ):
        assert isinstance(task, str) and task, "must specify a single task string"

        self.num_point = num_point
        self.stream_freq = stream_freq
        self._data_root = data_root
        self._demo_folder_override = demo_folder
        self.task = task
        self.seed = seed
        self._teleop = teleop
        self._record_state = record_state
        self._record_demo = False
        self._crit_state = []
        self._rec_paused = False
        self._rec_stopped = False
        self._undo_stack = []
        self._input_log = []
        self._dashboard = dashboard
        self._ui = ui
        self._rgbd = rgbd
        self._browser = None
        self._task_start_time: float = 0.0
        self._task_end_time: float = 0.0
        self._timed_out: bool = False
        self._manual_quit: bool = False
        self._ended_by: str | None = None
        self._end_requested: bool = False
        self._episode_over: bool = False
        self._blender = render.enabled()
        self._hold_render: bool = False
        self.cmds: queue.Queue = queue.Queue(maxsize=1)
        self.frames: queue.Queue = queue.Queue(maxsize=1)
        self._stations: dict = {}
        self._ui_stations: dict | None = None
        self._systematic_k = -1
        self._qpos: list | None = None
        self._frame_ready = False
        self._probe_pool = ThreadPoolExecutor(max_workers=1)
        self._probe_cache = None

        self._build()

    def _build(self):
        if self._demo_folder_override:
            self.demo_folder = os.path.abspath(self._demo_folder_override)
            self.data_folder = os.path.dirname(self.demo_folder)
            base = os.path.basename(self.demo_folder)
            tail = base[len("demo"):]
            self.demo_idx = int(tail) if base.startswith("demo") and tail.isdigit() else 0
        else:
            self.data_folder = os.path.join(self._data_root, self.task.replace("/", "_"))
            self.demo_idx = _next_demo_idx(self.data_folder)
            self.demo_folder = os.path.join(self.data_folder, f"demo{self.demo_idx:05d}")

        os.makedirs(self.data_folder, exist_ok=True)
        self.screenshots_dir = os.path.join(self.demo_folder, "screenshots")
        os.makedirs(self.screenshots_dir, exist_ok=True)
        self.frames_dir = os.path.join(self.demo_folder, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)
        if self._ui and not self._teleop:
            self._browser = ui_for_agent.AgentBrowser(BASE_PORT, os.environ.get(
                "ROBOARCADE_CHROMIUM_LOG", os.path.join(self.demo_folder, "chromium_stderr.log")))
        logger.info(utils.wrap_ruler("save location"))
        logger.info(f"[save] demo {self.demo_idx:05d} -> {os.path.abspath(self.demo_folder)}/")

        self.env = SimEnv(self.task, seed=self.seed)
        self._ui_z_offset = float(self.env.pc["z_offset"])

        third = self.env.cameras_by_role("third_person")
        if len(third) != 1:
            raise ValueError(f"{self.task}: expected one third_person camera, got {third}")
        tp = third[0]
        wrists = self.env.cameras_by_role("wrist")
        pairs = [("agentview", tp)] + [
            ("wrist" if i == 0 else f"wrist_{self.env.camera_robot(c)}", c)
            for i, c in enumerate(wrists)]
        self._cam_feeds = [(f"{c}_image", label, c) for label, c in pairs]
        self._cam_K = {label: self.env.get_camera_intrinsics(c).ravel().tolist()
                       for _, label, c in self._cam_feeds}
        self._static_cams = ([('agentview', tp)]
                             + [(c, c) for c in self.env.cameras_by_role("pc")])
        self._camera_views = ([label for _, label, _ in self._cam_feeds]
                              + [label for label, _ in self._static_cams
                                 if label != "agentview"])

        self._reset_rgba_bank()
        self.recorder = EpisodeRecorder(
            self.demo_folder, video_name="cameras",
            state_fn=self._state_with_rgba if self._record_state else None,
            static_fn=self.env.static_poses if self._record_state else None)

    def _reset_rgba_bank(self):
        self._rgba_frames: list[int] = []
        self._rgba: list[np.ndarray] = []
        self._rgba_last: np.ndarray | None = None
        self._rgba_frame_count = 0

    def _state_with_rgba(self):
        rgba = self.env.model.geom_rgba
        if self._rgba_last is None or not np.array_equal(rgba, self._rgba_last):
            self._rgba_last = np.array(rgba, dtype=np.float32)
            self._rgba_frames.append(self._rgba_frame_count)
            self._rgba.append(self._rgba_last)
        self._rgba_frame_count += 1
        return self.env.sim_state()

    def _truncate_rgba_bank(self, frame_count):
        keep = sum(frame < frame_count for frame in self._rgba_frames)
        del self._rgba_frames[keep:]
        del self._rgba[keep:]
        self._rgba_last = self._rgba[-1] if self._rgba else None
        self._rgba_frame_count = frame_count

    def _write_rgba_bank(self):
        path = os.path.join(self.demo_folder, "cameras_states.npz")
        if not self._rgba or not os.path.exists(path):
            return
        with np.load(path) as saved:
            arrays = {name: saved[name] for name in saved.files}
        if len(arrays["states"]) != self._rgba_frame_count:
            raise RuntimeError(
                f"RGBA bank has {self._rgba_frame_count} frames for "
                f"{len(arrays['states'])} recorded states")
        arrays["rgba_frames"] = np.asarray(self._rgba_frames, dtype=np.int32)
        arrays["rgba"] = np.asarray(self._rgba, dtype=np.float32)
        tmp = path + ".rgba.tmp.npz"
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
        logger.info(f"saved {path} RGBA bank ({len(self._rgba)} snapshots for "
                    f"{self._rgba_frame_count} frames)")

    def _end_recording(self, save):
        self.recorder.end_episode(save=save)
        try:
            if save:
                self._write_rgba_bank()
        finally:
            self._reset_rgba_bank()

    def _publish_stations(self):
        decl = dict(self.env.spec["randomization"].get("placement_regions") or {})
        st = {}
        for body, regs in decl.items():
            out = []
            for reg in regs:
                cx, cy, clip = self.env.clipped_station(body, reg)
                out.append({"y": list(reg["y"]), "x": list(reg["x"]) if reg.get("x") else None,
                            "yaw_offset_deg": reg.get("yaw_offset_deg", [0, 0]),
                            "clipped_x": list(cx) if cx else None,
                            "clipped_y": list(cy), "was_clipped": bool(clip)})
            st[body] = out
        self._stations = st

    @property
    def _rec(self):
        return None if (self._rec_paused or self._rec_stopped) else self.recorder

    def _refresh_crit_state(self):
        if self._teleop:
            self._crit_state = grade_episode.describe_state(self.env.spec["success"], self.env)

    def _record_verdict(self):
        self._refresh_crit_state()
        elapsed = (self._task_end_time or time.time()) - self._task_start_time if self._task_start_time else 0.0
        verdict = {
            "task": self.task,
            "language": str(self.env.language_instruction or ""),
            "driver": "human" if self._teleop else "agent",
            "ended_by": self._ended_by,
            "timed_out": bool(self._timed_out),
            "sim_steps_used": int(self.env.num_step),
            "sim_steps_budget": int(self.env.max_len),
            "elapsed_seconds": round(elapsed, 1),
            "seed": int(self.seed),
            "full_rand": bool(self.env.full_rand),
            "init_poses": self._init_poses,
            "init_state": self._init_state,
            "final_poses": {n: [round(float(v), 5) for v in xyz] for n in self.env.init_bases
                            if (xyz := self.env._live_xyz(n)) is not None},
            "success_spec": self.env.spec["success"],
            "physics_timestep": float(self.env.model.opt.timestep),
            "control_freq": float(self.env.control_freq),
            "demo_idx": int(self.demo_idx),
            "demo_folder": os.path.abspath(self.demo_folder),
            "data_folder": os.path.abspath(self.data_folder),
            "cameras_mp4": os.path.join(self.demo_folder, "cameras.mp4"),
            "screenshots_dir": os.path.abspath(self.screenshots_dir),
        }
        with open(os.path.join(self.demo_folder, "verdict.json"), "w") as f:
            json.dump(verdict, f, indent=2, default=str)

        logger.info(f"[end] {self.task}: ended by {self._ended_by}, {self.env.num_step}/{int(self.env.max_len)} steps"
                    f"{', horizon hit' if self._timed_out else ''}, {elapsed:.0f}s wall")

    def _read_init_poses(self):
        self._init_poses = {n: [round(float(v), 5) for v in xyz]
                            for n in self.env.init_bases
                            if (xyz := self.env._live_xyz(n)) is not None}

    def reset(self):
        self.env.reset(seed=self.seed, randomize_init=not self._teleop)
        self._read_init_poses()
        self._init_state = {
            arm: {"eef": [round(float(v), 5) for v in self.env.observe_proprio(arm).eef_pos],
                  "joints": [round(float(self.env.data.qpos[a]), 5)
                             for a in self.env.ik[arm].qadr]}
            for arm in self.env.arms}

    def transform_robotframe_to_uiframe(self, waypoints):
        waypoints = np.array(waypoints) + np.array([0.0, 0.0, -self._ui_z_offset])
        transf = R.from_euler("x", -90, degrees=True)
        waypoints_ui = transf.apply(waypoints)
        waypoints_ui *= 10
        return waypoints_ui

    def transform_uiframe_to_robotframe(self, waypoints):
        waypoints_rob = waypoints.copy()
        waypoints_rob /= 10.0
        transf = R.from_euler("x", 90, degrees=True)
        waypoints_rob = transf.apply(waypoints_rob)
        waypoints_rob += np.array([0.0, 0.0, self._ui_z_offset])
        return waypoints_rob

    def probe_reachable(self, fingertip_ui, rotation_ui, arm=None) -> dict:
        env = self.env
        arm = arm or env.arms[0]
        p = arm + "_"
        ee_euler = _ui_euler_to_robot(rotation_ui)
        ee_pos = self.transform_uiframe_to_robotframe(
            np.asarray(fingertip_ui, float).reshape(1, 3)).squeeze()
        goal_rot = R.from_euler("xyz", ee_euler)

        cache = self._probe_cache or {}
        if arm not in cache:
            q_home = np.asarray(env.spec["robots"][env.arms.index(arm)]["home_qpos"], float)
            cache[arm] = (make_arm_ik(env.model, p, q_home, env.kin[arm]),
                          mujoco.MjData(env.model), q_home)
            self._probe_cache = cache
        solver, d, q_home = cache[arm]

        live_q = self._qpos
        if live_q is not None and len(live_q) == env.model.nq:
            d.qpos[:] = live_q
        else:
            d.qpos[:] = env.data.qpos
            d.qpos[solver.qadr] = q_home
        mujoco.mj_forward(env.model, d)
        q_live = d.qpos[solver.qadr].copy()

        def body_root(b):
            while b > 0 and b not in env._free_body_ids:
                b = int(env.model.body_parentid[b])
            return b

        def geom_root(g):
            return body_root(int(env.model.geom_bodyid[g]))
        held = {body_root(b) for b in env.held_bodies(arm, d)} - {0}

        def tracked(g):
            name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY,
                                     env.model.geom_bodyid[g]) or ""
            return name.startswith(p) or geom_root(g) in held

        def contact_pairs(dd):
            out = set()
            for c in dd.contact[: dd.ncon]:
                g1, g2 = int(c.geom1), int(c.geom2)
                if tracked(g1) or tracked(g2):
                    out.add((min(g1, g2), max(g1, g2)))
            return out

        baseline = contact_pairs(d)
        site = env._site[arm]
        id2name = {n: env.model.body(n).id for n in env._free_objects}
        rel = {}
        site_T = np.eye(4); site_T[:3, :3] = d.site_xmat[site].reshape(3, 3)
        site_T[:3, 3] = d.site_xpos[site]
        for name, bid in id2name.items():
            if bid in held:
                a = env._obj_qadr[name]
                obj_T = np.eye(4)
                obj_T[:3, :3] = R.from_quat(d.qpos[a + 3:a + 7][[1, 2, 3, 0]]).as_matrix()
                obj_T[:3, 3] = d.qpos[a:a + 3]
                rel[name] = np.linalg.inv(site_T) @ obj_T

        live_ok, live_pos, live_rot = env.ctl.reaches(
            arm, ee_pos, ee_euler, q0=q_live)
        q_target = solver.solve_pose(q_live, ee_pos, goal_rot)
        if np.all(np.isfinite(q_target)):
            d.qpos[solver.qadr] = q_target
            mujoco.mj_forward(env.model, d)

        def place_held():
            if not rel:
                return
            site_T[:3, :3] = d.site_xmat[site].reshape(3, 3)
            site_T[:3, 3] = d.site_xpos[site]
            for name, T in rel.items():
                obj_T = site_T @ T
                a = env._obj_qadr[name]
                d.qpos[a:a + 3] = obj_T[:3, 3]
                d.qpos[a + 3:a + 7] = R.from_matrix(obj_T[:3, :3]).as_quat()[[3, 0, 1, 2]]
            mujoco.mj_forward(env.model, d)

        place_held()
        live_hit = bool(contact_pairs(d) - baseline)
        return {"reachable": bool(live_ok),
                "pos_mm": round(live_pos * 1000, 1),
                "rot_deg": round(float(np.degrees(live_rot)), 1),
                "blocked": live_hit}

    def prepare_point_cloud(self, obs):
        points, colors = self.env.get_point_cloud(obs, bool(self.env.pc["crop_table"]))

        if len(points) > self.num_point:
            idxs = np.linspace(0, len(points) - 1, self.num_point, dtype=int)
            points = points[idxs]
            colors = colors[idxs]

        points_ui = self.transform_robotframe_to_uiframe(points).ravel().astype(np.float32)
        colors_out = colors.ravel().astype(np.float32)
        return points_ui.tobytes(), colors_out.tobytes()

    def collect_cam_info(self, obs: dict) -> dict:
        info = {}
        for obs_key, label, cam_name in self._cam_feeds:
            if obs_key not in obs:
                continue
            info[label] = {
                "K": self._cam_K[label],
                "E": self.env.get_camera_extrinsics(cam_name).ravel().tolist(),
                "img_size": self.env.image_size,
            }
        return info

    def collect_arm_poses(self) -> dict:
        if len(self.env.arms) < 2:
            return {}
        poses = {}
        for arm in self.env.arms:
            p = self.env.observe_proprio(arm)
            poses[arm] = {
                "fingertip_pos_ui":
                    self.transform_robotframe_to_uiframe(p.eef_pos.reshape(1, 3)).squeeze().tolist(),
                "ee_euler_ui": [p.eef_euler[0] + np.pi, p.eef_euler[1], p.eef_euler[2]],
                "gripper_open": float(p.gripper_open),
                "gripper_cmd_open": int(self.env.curr_gripper_open.get(arm, 1)),
            }
        return poses

    def collect_robot_state(self) -> dict:
        env = self.env
        arms = list(env.arms)
        sides = {}
        if len(arms) > 1:
            base_y = {a: float(rb["base_pos"][1]) for a, rb in zip(arms, env.spec["robots"])}
            sides = {a: ("right" if i == 0 else "left") for i, a in enumerate(sorted(arms, key=lambda a: -base_y[a]))}
        state = {"arms": {}, "wrist_cameras": {label: sides[env.camera_robot(cam)] for _, label, cam in self._cam_feeds
                                                if sides and env.camera_role(cam) == "wrist"}}
        for a in arms:
            p = env.observe_proprio(a)
            state["arms"][a] = {"side": sides.get(a), "position": [float(v) for v in p.eef_pos],
                                "euler": [float(v) for v in p.eef_euler],
                                "gripper_open": bool(env.curr_gripper_open.get(a, 1))}
        return state

    def collect_home_poses(self) -> dict:
        env = self.env
        poses = {}
        for arm in env.arms:
            pos, euler = env.ctl.home_pose(arm)
            poses[arm] = {
                "fingertip_pos_ui": self.transform_robotframe_to_uiframe(
                    pos.reshape(1, 3)).squeeze().tolist(),
                "ee_euler_ui": [float(euler[0] + np.pi), float(euler[1]), float(euler[2])],
            }
        return poses

    def cmd_open(self) -> int:
        return int(self.env.curr_gripper_open.get(self.env.arms[0], 1))

    def _send(self, frame, block=True):
        if not block and self.frames.full():
            return
        obs = frame["obs"]
        if self._rgbd and frame.get("update_ui"):
            from interfaces.rgbd_waypoint.sensors import capture
            frame = {**frame, "rgbd_snapshot": capture(
                self.env, obs, self._cam_feeds, self._static_cams)}
        self.frames.put(
            {**frame, "cam_info": self.collect_cam_info(obs),
             "gripper_cmd_open": self.cmd_open(),
             "arm_poses": self.collect_arm_poses(),
             "robot": self.collect_robot_state()},
            block=block)

    def _bank_frame(self, obs):
        self.recorder.record(ActMode.Dense, obs)

    def collect_workspaces(self) -> dict:
        if len(self.env.arms) < 2:
            return {}
        voxel_step_m = 0.02
        zs = [float(v) for v in self.env.rest_z.values()]
        z0 = float(np.mean(zs)) if zs else float(self._ui_z_offset) + 0.02
        c = self.env.pc
        xs = np.arange(np.ceil((float(c["x"][0]) - 0.3) / voxel_step_m) * voxel_step_m,
                       float(c["x"][1]) + 0.3, voxel_step_m)
        ys = np.arange(np.ceil((float(c["y"][0]) - 0.3) / voxel_step_m) * voxel_step_m,
                       float(c["y"][1]) + 0.3, voxel_step_m)
        return {arm: {"step": voxel_step_m, "z": z0,
                      "cells": [[float(x), float(y)] for x in xs for y in ys
                                if self.env.reachable([x, y, z0], arm)]}
                for arm in self.env.arms}

    def collect_init_regions(self) -> dict:
        env = self.env
        out = {}
        regions = init_regions(env)
        for name, (pos, _quat) in env.init_bases.items():
            r = regions.get(name, {})
            z = env.rest_z.get(name, pos[2])
            out[name] = {"base": [float(pos[0]), float(pos[1]), float(z)],
                         "color": env.object_color(name),
                         "face_robot": name in env._face_robot, **r}
        full = bool(env.full_rand)
        voxel_step_m = 0.02
        by_group = {}
        for name, r in out.items():
            if r.get("x") and r.get("group") is not None:
                by_group.setdefault(r["group"], []).append(name)
        masks = {}
        for names in by_group.values():
            px = float(np.mean([out[n]["base"][0] for n in names]))
            py = float(np.mean([out[n]["base"][1] for n in names]))
            pz = float(np.mean([out[n]["base"][2] for n in names]))
            gbox = env.group_center_box(names)
            if full:
                rx0, rx1, ry0, ry1 = gbox
            else:
                r = out[names[0]]
                rx0, rx1 = px - r["x"], px + r["x"]
                ry0, ry1 = py - r["y"], py + r["y"]
            xs = np.arange(np.ceil(rx0 / voxel_step_m) * voxel_step_m, rx1 + 1e-9, voxel_step_m)
            ys = np.arange(np.ceil(ry0 / voxel_step_m) * voxel_step_m, ry1 + 1e-9, voxel_step_m)
            stations = (env._placement_regions or {}).get(names[0])
            def _station(ay):
                for i, reg in enumerate(stations or ()):
                    lo, hi = reg["y"]
                    if lo <= ay <= hi:
                        return i
                return -1
            cells = []
            for ax in xs:
                for ay in ys:
                    if not env.center_fits(names, ax, ay, gbox):
                        continue
                    si = _station(ay)
                    if stations and si < 0:
                        continue
                    cells.append([float(ax - px), float(ay - py), si])
            masks[names[0]] = {"step": voxel_step_m, "z": pz, "cells": cells}
        zs = [m["z"] for m in masks.values()] or \
             [float(v) for v in env.rest_z.values()]
        sl = None
        if zs:
            z0 = float(np.mean(zs))
            c = env.pc["x"], env.pc["y"]
            g = voxel_step_m
            xs = np.arange(np.ceil((float(c[0][0]) - 0.3) / g) * g,
                           float(c[0][1]) + 0.3, g)
            ys = np.arange(np.ceil((float(c[1][0]) - 0.3) / g) * g,
                           float(c[1][1]) + 0.3, g)
            def cells_at(z):
                return [[float(x), float(y)] for x in xs for y in ys
                        if env.reachable([x, y, float(z)])]
            sl = {"step": g, "z": z0, "cells": cells_at(z0),
                  "slices": {f"{z:.4f}": cells_at(z) for z in
                             sorted({round(m["z"], 4) for m in masks.values()})}}
            sr = env._support_rect
            if sr is not None:
                sl["table"] = [float(v) for v in sr]
            c = env.pc
            sl["pc"] = [float(c["x"][0]), float(c["x"][1]),
                        float(c["y"][0]), float(c["y"][1])]
        reach = {"masks": masks, "slice": sl} if (masks or sl) else None
        station_bounds = {}
        for b in self._stations:
            station_bounds[b] = [float(v) for v in env.group_center_box([b])]
        return {"regions": out, "reach": reach, "stations": self._stations,
                "station_bounds": station_bounds, "full_rand": bool(env.full_rand)}

    def init_webcontent(self, obs: dict):
        ee_euler = obs["ee_euler"]
        fingertip_pos_ui = self.transform_robotframe_to_uiframe(
            obs["ee_pos"].reshape(1, 3)
        ).squeeze()
        ee_euler_ui = [ee_euler[0] + np.pi, ee_euler[1], ee_euler[2]]

        ui_mesh = self.env.kin[self.env.arms[0]]["ui_mesh"]
        self.ui_config = {
            "num_point": int(self.num_point),
            "fingertip_pos_ui": [round(float(v), 2) for v in fingertip_pos_ui],
            "ee_euler_ui": [round(float(v), 2) for v in ee_euler_ui],
            "gripper_name": ui_mesh["name"],
            "gripper_scale": float(ui_mesh["scale"]),
            "ui_z_offset": round(float(self._ui_z_offset), 4),
            "arms": [str(a) for a in self.env.arms],
            "home_poses": self.collect_home_poses(),
            "cams": [label for label, _ in self._static_cams],
            "static_cam_info": {
                label: {"K": self.env.get_camera_intrinsics(cam).ravel().tolist(),
                        "E": self.env.get_camera_extrinsics(cam).ravel().tolist(),
                        "img_size": self.env.image_size}
                for label, cam in self._static_cams},
            **({"robot_bases": {str(a): [float(v) for v in rb["base_pos"]]
                                for a, rb in zip(self.env.arms, self.env.spec["robots"])},
                "workspaces": self.collect_workspaces(),
                "wrist_arms": {label: self.env.camera_robot(cam)
                               for _, label, cam in self._cam_feeds
                               if self.env.camera_role(cam) == "wrist"}}
               if len(self.env.arms) > 1 else {}),
            "dashboard": bool(self._dashboard),
            "teleop": bool(self._teleop),
        }

    def _serve(self):
        cmds, frames, blender = self.cmds, self.frames, self._blender

        async def listen_ui(websocket):
            async for message in websocket:
                message = json.loads(message)
                if not len(message):
                    continue
                data = message[-1]

                if "inputs" in data:
                    for ev in data["inputs"]:
                        ev["waypoint"] = self.recorder.waypoint_idx + 1
                    self._input_log.extend(data["inputs"])
                    continue
                if "frame" in data:
                    frame = data["frame"]
                    path = os.path.join(self.frames_dir, f"{frame['i']:06d}.jpg")
                    with open(path, "wb") as f:
                        f.write(base64.b64decode(frame.pop("jpeg")))
                    frame["waypoint"] = self.recorder.waypoint_idx + 1
                    self._input_log.append(frame)
                    continue

                if not cmds.empty():
                    logger.warning("the cmds queue is not empty, dropping new UI command")
                    continue

                if "position" in data:
                    data["gripper_open"] = float(data["gripper_open"])
                    p, r = data["position"], data["orientation"]
                    logger.info(f"[ui_cmd] gripper_open: {data['gripper_open']}, "
                                f"done: {data['done']}, t={time.time():.3f}, "
                                f"pos_ui=({p['x']:.3f},{p['y']:.3f},{p['z']:.3f}), "
                                f"rot_ui=({r['x']:.2f},{r['y']:.2f},{r['z']:.2f}), "
                                f"arm={data.get('arm')}")
                cmds.put(data, block=True)

        async def main():
            env, loop = self.env, asyncio.get_running_loop()
            connected = set()
            latest_cam_jpegs: dict[str, bytes] = {}
            latest_frames: dict[str, np.ndarray] = {}
            last_update_msg: list = [None]
            latest_rgbd: list = [None]
            latest_robot: list = [None]

            async def on_connect(websocket):
                if last_update_msg[0] is not None:
                    try:
                        await websocket.send(last_update_msg[0])
                    except Exception:
                        return
                connected.add(websocket)
                try:
                    await websocket.wait_closed()
                finally:
                    connected.discard(websocket)

            def _body(body: bytes, ctype: str) -> Response:
                return Response(200, "OK", Headers([
                    ("Content-Type", ctype), ("Content-Length", str(len(body))),
                    ("Cache-Control", "no-store"), ("Access-Control-Allow-Origin", "*")]), body)

            def _json_response(payload: dict) -> Response:
                return _body(json.dumps(payload).encode(), "application/json")

            def env_json():
                return _json_response({
                    "task": self.task,
                    "task_name": str(env.spec.get("name", "") or ""),
                    "language": str(env.language_instruction or ""),
                    "arms": [str(a) for a in env.arms],
                    "cams": [label for label, _ in self._static_cams],
                    "camera_views": self._camera_views,
                    "ui_z_offset": float(self._ui_z_offset),
                    "robot_base": [float(v) for v in env.spec["robots"][0]["base_pos"]],
                    "demo_folder": os.path.abspath(self.demo_folder),
                    "screenshots_dir": os.path.abspath(self.screenshots_dir),
                    "sim_steps_budget": int(env.max_len),
                    "sim_steps_used": int(env.num_step),
                    "episode_over": self._episode_over,
                    "frame_ready": self._frame_ready,
                })

            async def reachable(query: str):
                qs = parse_qs(query)
                try:
                    fp = [float(qs[k][0]) for k in ("fx", "fy", "fz")]
                    ro = [float(qs[k][0]) for k in ("rx", "ry", "rz")]
                except (KeyError, ValueError):
                    return _json_response({"reachable": None})
                arm = qs.get("arm", [None])[0]
                try:
                    ok = await loop.run_in_executor(self._probe_pool, self.probe_reachable,
                                                    fp, ro, arm if arm in env.arms else None)
                except Exception:
                    logger.exception("[reachable] probe failed")
                    ok = {"reachable": None}
                return _json_response(ok)

            def objects():
                q = self._qpos
                out = {}
                if q:
                    for i, name in enumerate(env._free_objects, 1):
                        a = int(env._obj_qadr[name])
                        out[name] = {"num": i, "pose": [round(float(x), 6) for x in q[a:a + 7]]}
                return _json_response({"objects": out})

            async def init_regions():
                return _json_response(
                    await loop.run_in_executor(self._probe_pool, self.collect_init_regions))

            def cam_jpeg(name: str):
                body = latest_cam_jpegs.get(name)
                if body is None and name in latest_frames:
                    _, buf = cv2.imencode(".jpg", latest_frames[name][:, :, ::-1],
                                          [cv2.IMWRITE_JPEG_QUALITY, 85])
                    body = buf.tobytes()
                if body is None:
                    return _empty(503, "Service Unavailable")
                return _body(body, "image/jpeg")

            routes = {
                "/env.json": env_json,
                "/ui_config.json": lambda: _json_response(self.ui_config),
                "/robot.json": lambda: (_empty(503, "No frame yet") if latest_robot[0] is None
                                        else _json_response(latest_robot[0])),
                "/success_state": lambda: _json_response({"state": self._crit_state}),
                "/success_spec": lambda: _json_response(
                    {"rows": grade_episode.describe(env.spec["success"], env.task),
                     "preds": grade_episode.predicate_names(env.spec["success"])}),
                "/objects": objects,
                "/init_regions": init_regions,
            }
            if self._rgbd:
                routes["/rgbd.json"] = lambda: (_empty(503, "Service Unavailable") if latest_rgbd[0] is None
                                                else _json_response(latest_rgbd[0]))
            teleop_only = {"/objects", "/success_spec", "/success_state", "/init_regions"}

            async def process_request(connection, request):
                path, _, query = request.path.partition("?")
                if path.startswith("/cam/") and path.endswith(".jpg"):
                    return cam_jpeg(path[len("/cam/"):-len(".jpg")])
                if path == "/reachable":
                    return await reachable(query)
                if path in teleop_only and not self._teleop:
                    return _empty(403, "Forbidden")
                handler = routes.get(path)
                if handler is None:
                    return None
                resp = handler()
                return await resp if asyncio.iscoroutine(resp) else resp

            async def broadcast_loop():
                loop = asyncio.get_event_loop()
                update_seq = 0
                while True:
                    to_send = await loop.run_in_executor(None, frames.get)
                    if to_send is None:
                        return
                    obs = to_send["obs"]
                    fresh = bool(to_send.get("rendered")
                                 or not (blender or to_send.get("hold_render")))
                    if fresh:
                        for k, v in obs.items():
                            if k.endswith("_image") and getattr(v, "ndim", 0) == 3:
                                latest_frames[k[: -len("_image")]] = v
                    update_ui = to_send["update_ui"]
                    if update_ui:
                        update_seq += 1
                    if fresh and update_ui and "rgbd_snapshot" in to_send:
                        from interfaces.rgbd_waypoint.sensors import encode_snapshot
                        latest_rgbd[0] = encode_snapshot(to_send["rgbd_snapshot"], update_seq)
                    latest_robot[0] = {**to_send["robot"], "update_seq": update_seq}

                    if to_send.get("skip_pcl", False):
                        points_bytes, colors_bytes = b"", b""
                    else:
                        points_bytes, colors_bytes = self.prepare_point_cloud(obs)

                    ee_euler = obs["ee_euler"]
                    fingertip_pos_ui = (
                        self.transform_robotframe_to_uiframe(obs["ee_pos"].reshape(1, 3))
                        .squeeze()
                        .tolist()
                    )
                    ee_euler_ui = [ee_euler[0] + np.pi, ee_euler[1], ee_euler[2]]

                    cam_images = {}
                    if fresh:
                        for obs_key, label, _ in self._cam_feeds:
                            if obs_key in obs:
                                _, buf = cv2.imencode(
                                    ".jpg", obs[obs_key][:, :, ::-1],
                                    [cv2.IMWRITE_JPEG_QUALITY, 75]
                                )
                                jpeg_bytes = buf.tobytes()
                                cam_images[label] = jpeg_bytes
                                latest_cam_jpegs[label] = jpeg_bytes

                    data = {
                        "positions": points_bytes,
                        "colors": colors_bytes,
                        "rendered_colors": bool(to_send.get("rendered")),
                        "fingertip_pos_ui": fingertip_pos_ui,
                        "ee_euler_ui": ee_euler_ui,
                        "gripper_cmd_open": to_send["gripper_cmd_open"],
                        "gripper_width_frac": float(obs["gripper_open"].item()),
                        "update_ui": update_ui,
                        "update_ui_seq": update_seq if update_ui else None,
                        "cam_images": cam_images,
                        "cam_info": to_send.get("cam_info", {}),
                        "render_note": to_send.get("render_note"),
                    }
                    if to_send.get("arm_poses"):
                        data["arm_poses"] = to_send["arm_poses"]

                    msg = msgpack.packb(data)
                    if update_ui:
                        last_update_msg[0] = msg
                    dead = set()
                    for ws in list(connected):
                        try:
                            await ws.send(msg)
                        except Exception:
                            dead.add(ws)
                    connected.difference_update(dead)

            asyncio.create_task(broadcast_loop())
            async with websockets.serve(
                listen_ui, "localhost", UI_LISTEN_PORT, max_size=None, ping_timeout=120,
            ), websockets.serve(
                on_connect, "localhost", BROADCAST_PORT,
                process_request=process_request, ping_timeout=120,
            ):
                await asyncio.Future()

        threading.Thread(target=lambda: asyncio.run(main()), daemon=True).start()

    def teardown(self):
        try:
            self.frames.put(None, timeout=5)
        except queue.Full:
            pass
        if self._browser:
            self._browser.stop()

    def _stream_fn(self, obs):
        self._send({"obs": obs, "update_ui": False, "skip_pcl": True}, block=False)

    def apply_waypoint_mode(self, ui_cmd: dict):
        robot = ui_cmd.get("robot_waypoint")
        if robot is not None:
            arm, gripper_open = robot.get("arm"), float(robot["gripper_open"])
            ee_pos_cmd, ee_euler_cmd = np.asarray(robot["position"], float), np.asarray(robot["euler"], float)
        else:
            arm, gripper_open = ui_cmd.get("arm"), ui_cmd["gripper_open"]
            rot = ui_cmd["orientation"]
            ee_euler_cmd = _ui_euler_to_robot([rot["x"], rot["y"], rot["z"]])
            p = ui_cmd["position"]
            ee_pos_cmd = self.transform_uiframe_to_robotframe(
                np.array([[p["x"], p["y"], p["z"]]])).squeeze()
        if arm not in self.env.arms:
            arm = None

        obs = self.env.observe()
        rec = self._rec
        if rec is not None:
            rec.record(ActMode.Waypoint, obs)
        step0, t0 = int(self.env.num_step), time.time()
        home = bool(ui_cmd.get("home"))
        if home:
            self.env.ctl.home(arm, gripper_open, recorder=rec, stream_fn=self._stream_fn)
        else:
            self.env.move_to(
                ee_pos_cmd, ee_euler_cmd, gripper_open,
                recorder=rec, stream_fn=self._stream_fn, arm=arm,
            )
        self._log_waypoint(arm or self.env.arms[0], ee_pos_cmd, ee_euler_cmd,
                           gripper_open, home, step0, t0)

    def _log_waypoint(self, arm, pos_cmd, euler_cmd, gripper_open, home, step0, t0):
        env, m = self.env, self.env.model
        pro = env.observe_proprio(arm)
        now = R.from_matrix(env.data.site_xmat[env._site[arm]].reshape(3, 3))
        touching = env.ctl._contact_bodies(env.ctl._arm_contacts(arm))
        holding = env.ctl._payload_contact_baseline(arm)
        names = lambda ids: sorted({mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,
                                                      int(m.body_rootid[b])) or "" for b in ids})
        row = {"waypoint": self.recorder.waypoint_idx, "arm": arm, "home": home,
               "cmd_pos": [round(float(v), 5) for v in pos_cmd],
               "cmd_euler": [round(float(v), 5) for v in euler_cmd],
               "cmd_gripper_open": int(gripper_open),
               "step_start": step0, "step_end": int(env.num_step),
               "wall_s": round(time.time() - t0, 2),
               "eef_pos": [round(float(v), 5) for v in pro.eef_pos],
               "pos_err_mm": round(float(np.linalg.norm(pro.eef_pos - pos_cmd)) * 1000, 2),
               "rot_err_deg": round(float(np.degrees(
                   (R.from_euler("xyz", euler_cmd) * now.inv()).magnitude())), 2),
               "jaw_width_mm": round(float(env.gripper_width(arm)) * 1000, 2),
               "touching": names(touching), "holding": names(holding),
               "episode_over": bool(env.episode_over)}
        ncon, nefc, secs, steps = env.step_stats
        env.step_stats = [0, 0, 0.0, 0]
        row.update({"ncon_max": ncon, "nefc_max": nefc,
                    "physics_ms_per_step": round(1000 * secs / steps, 3) if steps else None})
        with open(os.path.join(self.demo_folder, "waypoints.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")

    def _save_task_demo(self):
        src = os.path.join(self.demo_folder, "cameras.mp4")
        if not os.path.isfile(src):
            logger.warning(f"[demo] no cameras.mp4 in {self.demo_folder}; demo not saved")
            return
        demos = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "task_demos")
        os.makedirs(demos, exist_ok=True)
        dst = os.path.join(demos, self.task.replace("/", "_") + ".mp4")
        try:
            import imageio_ffmpeg
            subprocess.run(
                [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", src, "-c:v", "libx264",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", dst + ".tmp.mp4"],
                check=True, capture_output=True)
            os.replace(dst + ".tmp.mp4", dst)
        except Exception as e:
            logger.warning(f"[demo] H.264 transcode failed ({e}); saving raw copy")
            shutil.copyfile(src, dst + ".tmp.mp4")
            os.replace(dst + ".tmp.mp4", dst)
        logger.info(f"[demo] task demo saved -> {os.path.abspath(dst)}")

    UNDO_DEPTH = 20

    def _snapshot_state(self):
        return self.env.checkpoint(), len(self.recorder._spans)

    def _undo_state(self):
        if not self._undo_stack:
            logger.info("[undo] nothing to undo")
            return
        s, spans = self._undo_stack.pop()
        while len(self.recorder._spans) > spans:
            self.recorder.discard_last_waypoint()
        self._truncate_rgba_bank(len(self.recorder._states))
        env = self.env
        env.restore(s)
        env.phase = Phase.RUNNING
        logger.info(f"[undo] reverted to pre-waypoint state (num_step={env.num_step}, "
                    f"{len(self._undo_stack)} more undo levels)")
        self._refresh_crit_state()
        self._send({"obs": env.observe(), "update_ui": True})

    def _blender_preview(self):
        t0 = time.perf_counter()
        obs = self.env.rerender(self.env.observe(), force=True)
        ok = bool(self.env._blender)
        self._hold_render = ok
        note = (f"Blender {(time.perf_counter() - t0):.1f}s" if ok
                else "Blender unavailable, still MuJoCo")
        logger.info(f"[preview] {note}")
        self._send({"obs": obs, "update_ui": False, "rendered": ok, "render_note": note})

    def _reroll_init(self, randomize, systematic=False, bodies=None, stations=None):
        if systematic:
            if stations is not None and stations != self._ui_stations:
                self.env._placement_regions = {
                    b: [{k: v for k, v in st.items() if v is not None} for st in sts]
                    for b, sts in stations.items() if sts}
                self.env.spec["randomization"]["placement_regions"] = \
                    dict(self.env._placement_regions)
                self._ui_stations = stations
                self._systematic_k = -1
            self._systematic_k += 1
            choice = place_at_extreme(self.env, self._systematic_k, bodies or None)
            if choice is None:
                self.env._place_objects(bool(randomize))
            else:
                n = len(systematic_extremes(self.env, bodies or None))
                logger.info(f"[systematic] {self._systematic_k % n + 1}/{n}: "
                            f"{choice['body']} station {choice['station']} "
                            f"y={choice['y']:+.3f} yaw_off={choice['yaw_offset_deg']:+.0f} deg")
        else:
            self.env._place_objects(bool(randomize))
        self.env.data.qvel[:] = 0.0
        self.env._dynamics("restore")
        mujoco.mj_forward(self.env.model, self.env.data)
        self.env.settle_until_rest()
        self.env.redeal_episode()
        self._read_init_poses()
        self._refresh_crit_state()
        self._send({"obs": self.env.observe(), "update_ui": True})

    def _restart_episode(self):
        self._end_recording(save=False)
        self.reset()
        self._publish_stations()
        self._refresh_crit_state()
        self.env.phase = Phase.RUNNING
        self._task_start_time = time.time()
        self._undo_stack.clear()
        self._rec_paused = self._rec_stopped = False
        self._ended_by = None
        obs = self.env.observe()
        self._send({"obs": obs, "update_ui": True})
        self._bank_frame(obs)
        logger.info("[demo] episode restarted")

    def _stop_recording(self):
        if self._rec_stopped:
            return
        self._end_recording(save=True)
        if self._record_demo:
            self._save_task_demo()
        self._record_demo = False
        self._rec_stopped, self._rec_paused = True, False
        logger.info(f"[demo] recording stopped -> {self.demo_folder}")

    def _new_demo(self):
        self.demo_idx += 1
        self.demo_folder = os.path.join(self.data_folder, f"demo{self.demo_idx:05d}")
        self.screenshots_dir = os.path.join(self.demo_folder, "screenshots")
        self.frames_dir = os.path.join(self.demo_folder, "frames")
        os.makedirs(self.screenshots_dir, exist_ok=True)
        os.makedirs(self.frames_dir, exist_ok=True)
        self._reset_rgba_bank()
        self.recorder = EpisodeRecorder(
            self.demo_folder, video_name="cameras",
            state_fn=self._state_with_rgba if self._record_state else None,
            static_fn=self.env.static_poses if self._record_state else None)
        self._rec_stopped = self._rec_paused = False
        self._bank_frame(self.env.observe())
        logger.info(f"[demo] new demo -> {self.demo_folder}")

    def record_demo(self):
        logger.info(utils.wrap_ruler(f"record session, task: {self.task}"))
        self.reset()
        self._publish_stations()
        self._refresh_crit_state()
        obs = self.env.observe()
        self.init_webcontent(obs)
        if self._ui:
            page_dir = ui_for_agent.package_ui_dir()
            if page_dir:
                logger.info(f"[ui] serving interface package page from {page_dir}")
            httpd = ui_for_agent.http_server(page_dir, HTTP_PORT)
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self._serve()

        def _finalize_and_exit(signum, _frame):
            self._end_recording(save=True)
            if self._browser:
                self._browser.stop()
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
        signal.signal(signal.SIGTERM, _finalize_and_exit)
        signal.signal(signal.SIGUSR1, lambda *_: setattr(self, "_end_requested", True))

        if self._teleop:
            logger.info(
                f"[teleop] no owned browser; open http://localhost:{HTTP_PORT}/?hud=1 "
                f"(over SSH: ssh -L {HTTP_PORT}:localhost:{HTTP_PORT} "
                f"-L {BROADCAST_PORT}:localhost:{BROADCAST_PORT} "
                f"-L {UI_LISTEN_PORT}:localhost:{UI_LISTEN_PORT} <machine>)"
            )
        if self._browser:
            self._browser.start()

        obs = self.env.rerender(obs)
        logger.info("reset done, sending first frame")
        self._send({"obs": obs, "update_ui": True, "rendered": True})
        self._frame_ready = True
        self._bank_frame(obs)

        logger.info(f"episode start! task: {self.task}")
        self.one_episode()

        self._record_verdict()
        if self._teleop and self._record_demo:
            self._save_task_demo()
        time.sleep(0.4 if (self._teleop and self._manual_quit) else
                   3.0 if self._teleop else FINAL_SNAP_HOLD_S)
        logger.info("demo complete; finalizing video, stopping UI servers")

    def one_episode(self):
        self._task_start_time = time.time()
        budget = int(self.env.max_len)
        self.env.phase = Phase.RUNNING

        while True:
            self._qpos = self.env.data.qpos.tolist()

            if self._end_requested:
                self._manual_quit = True
                self._ended_by = "harness"
                self._episode_over = True
                self._task_end_time = time.time()
                break

            try:
                ui_cmd = self.cmds.get(timeout=1 / self.stream_freq)
            except queue.Empty:
                self._send({"obs": self.env.observe(depth=False), "update_ui": False, "skip_pcl": True,
                            "hold_render": self._hold_render})
                continue
            self._hold_render = self._hold_render and bool(ui_cmd.get("blender_render"))
            if (ui_cmd.get("reroll_init") or "record_demo" in ui_cmd
                    or ui_cmd.get("undo") or ui_cmd.get("blender_render")
                    or ui_cmd.get("restart_episode") or "record_pause" in ui_cmd
                    or ui_cmd.get("record_stop")):
                if not self._teleop:
                    logger.info(f"[ui_cmd] ignored teleop-only command in agent episode: "
                                f"{sorted(ui_cmd)}")
                    continue
                if ui_cmd.get("blender_render"):
                    self._blender_preview()
                elif ui_cmd.get("reroll_init"):
                    self._reroll_init(ui_cmd.get("randomize"),
                                      systematic=bool(ui_cmd.get("systematic")),
                                      stations=ui_cmd.get("stations"),
                                      bodies=ui_cmd.get("bodies"))
                elif ui_cmd.get("restart_episode"):
                    self._restart_episode()
                elif "record_pause" in ui_cmd:
                    self._rec_paused = bool(ui_cmd["record_pause"])
                    logger.info(f"[demo] recording {'PAUSED' if self._rec_paused else 'resumed'}")
                elif ui_cmd.get("record_stop"):
                    self._stop_recording()
                elif "record_demo" in ui_cmd:
                    if ui_cmd["record_demo"] and self._rec_stopped:
                        self._new_demo()
                    self._record_demo = ui_cmd["record_demo"]
                    logger.info(f"[demo] task demo recording {'ARMED' if self._record_demo else 'off'}")
                else:
                    self._undo_state()
                continue
            if ui_cmd.get("done"):
                self._manual_quit = True
                self._ended_by = ui_cmd.get("done_by") or "ui"
                self.env.ctl.settle_for_verdict(recorder=self._rec, stream_fn=self._stream_fn)
                self._refresh_crit_state()
                self._episode_over = True
                self._task_end_time = time.time()
                self._send({"obs": self.env.rerender(self.env.observe()), "update_ui": True, "rendered": True})
                break

            self._undo_stack.append(self._snapshot_state())
            del self._undo_stack[:-self.UNDO_DEPTH]
            self.apply_waypoint_mode(ui_cmd)

            used = int(self.env.num_step)

            self._refresh_crit_state()
            if self.env.phase is Phase.DONE:
                self._ended_by = self._ended_by or ("scene" if self.env.scene_over else "sim")
                self._timed_out = used >= budget
                self._episode_over = True
                self._task_end_time = time.time()
                logger.info(f"[terminal] task {self.task} ended by the sim at "
                            f"{self.env.num_step}/{budget} steps")
            fresh_obs = self.env.rerender(self.env.observe())
            self._send({"obs": fresh_obs, "update_ui": True, "rendered": True})
            self._bank_frame(fresh_obs)
            if self.env.phase is Phase.DONE:
                break

        self._end_recording(save=True)
        if self._input_log:
            events = sorted(self._input_log, key=lambda ev: ev["t"])
            path = os.path.join(self.demo_folder, "inputs.msgpack")
            with open(path, "wb") as f:
                f.write(msgpack.packb(events))
            n_frames = len(os.listdir(self.frames_dir))
            logger.info(f"saved {path} ({len(events)} events, {n_frames} frames)")


def main():
    parser = argparse.ArgumentParser(
        description="One episode's simulator as a server. --task is a folder under tasks/ (--list-tasks lists them).",
    )
    parser.add_argument("--num_point", type=int, default=100_000)
    parser.add_argument("--stream_freq", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1, help="Seed for env.reset()'s initial state.")
    parser.add_argument("--task", type=str, default="bowl_on_plate",
                        help="a folder under tasks/")
    parser.add_argument("--data_root", type=str, default="data/dev1", help="<root>/<task>/demoNNN")
    parser.add_argument(
        "--demo_folder", type=str, default=None, help="overrides data_root's auto-numbered demoNNN"
    )
    parser.add_argument(
        "--teleop",
        action="store_true",
        help="Human teleop: serve the UI to a real browser instead of launching a "
        "headless Chromium/CDP endpoint. The verdict is tagged driver=human and never scored.",
    )
    parser.add_argument("--dashboard", action="store_true",
                        help="Launched from the add_task dashboard: adds a Dashboard button "
                             "(+ random-pose panel on debug packs).")
    parser.add_argument("--record-state", action="store_true",
                        help="Bank the full sim state per recorded frame to "
                             "cameras_states.npz (the eval driver sets this, so an episode can be "
                             "re-graded or restored without re-running it).")
    parser.add_argument("--no-ui", action="store_true",
                        help="serve no browser page and start no Chromium: for an interface whose manifest.json sets "
                             "\"ui\": false, which drives the simulator through its API alone")
    parser.add_argument("--rgbd", action="store_true",
                        help="publish /rgbd.json, calibrated RGB-D snapshots (manifest.json \"rgbd\": true)")
    parser.add_argument("--list-tasks", action="store_true")
    args = parser.parse_args()
    utils.setup_logging()

    if args.list_tasks:
        for t in list_tasks():
            print(t)
        return

    np.set_printoptions(precision=4, linewidth=100, suppress=True)

    robot = InteractiveBot(
        args.task,
        args.num_point,
        args.stream_freq,
        args.data_root,
        args.seed,
        demo_folder=args.demo_folder,
        teleop=args.teleop,
        dashboard=args.dashboard,
        record_state=args.record_state or args.teleop,
        ui=not args.no_ui,
        rgbd=args.rgbd,
    )

    try:
        robot.record_demo()
    finally:
        robot.teardown()


if __name__ == "__main__":
    main()
