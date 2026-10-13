"""Blender rendering of the agent-facing camera feeds.

On when `ROBOARCADE_RENDER=blender`, which run_seed sets for every eval simulator (run_eval --render, default
blender); off otherwise. Under the flag a Blender that cannot start or dies ends the simulator with
RENDER_FAILED_EXIT, never a MuJoCo substitute. Every camera is rendered, but only at
waypoint boundaries: frames streamed during motion use MuJoCo's renderer. Blender emits no Z pass, so
point-cloud depth stays with MuJoCo while taking its per-point colour from these frames.
"""
import atexit
import json
import logging
import os
import select
import shutil
import subprocess
import tempfile
import time

import imageio.v2 as imageio
import mujoco
import numpy as np

from core.render.state import RenderState

logger = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
BLENDER = os.environ.get(
    "ROBOARCADE_BLENDER", os.path.join(REPO, "tools", "blender-4.5.12-linux-x64", "blender"))
MARK = b"@@ROBOARCADE@@"
RENDERERS = ("mujoco", "blender")
START_TIMEOUT = 600.0
FIRST_RENDER_TIMEOUT = 900.0
RENDER_TIMEOUT = 300.0
RENDER_FAILED_EXIT = 75


def enabled() -> bool:
    r = os.environ.get("ROBOARCADE_RENDER", "mujoco").lower()
    if r not in RENDERERS:
        raise ValueError(f"ROBOARCADE_RENDER={r!r}; expected one of {RENDERERS}")
    return r == "blender"


def available() -> tuple[bool, str]:
    if not os.path.exists(BLENDER):
        return False, f"no blender at {BLENDER}"
    return True, ""


def start(env):
    if os.environ.get("ROBOARCADE_RENDER_GROUP_SOCKET"):
        from core.render.group import GroupRenderer
        return GroupRenderer(env)
    ok, why = available()
    if not ok:
        raise RuntimeError(why)
    return BlenderRenderer(env)


class WorkerPipe:
    def spawn(self, mjb, resolution, task, **env):
        with open(os.path.join(self.dir, "worker.err"), "wb") as err:
            self.proc = subprocess.Popen(
                [BLENDER, "-b", "-P", os.path.join(HERE, "worker.py"),
                 "--", str(mjb), str(resolution), task],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err,
                env=dict(os.environ, ROBOARCADE_BLENDER_OUT=self.dir, **env))
        self._buf = b""

    def stop(self):
        if self.proc is None:
            return
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        self.proc.stdin.close()
        self.proc.stdout.close()
        self.proc = None

    def _worker_err(self, lines=6):
        err = os.path.join(self.dir, "worker.err")
        tail = open(err, errors="replace").read().strip().splitlines() if os.path.exists(err) else []
        return " | ".join(t.strip()[:160] for t in tail[-lines:]) or "(no stderr)"

    def _readline(self, timeout_s):
        deadline = time.monotonic() + timeout_s
        fd = self.proc.stdout.fileno()
        while True:
            line, sep, rest = self._buf.partition(b"\n")
            if sep:
                self._buf = rest
                if line.startswith(MARK):
                    return json.loads(line[len(MARK):])
                continue
            if not select.select([fd], [], [], max(deadline - time.monotonic(), 0))[0]:
                raise TimeoutError(f"worker silent for {timeout_s:.0f} s")
            chunk = os.read(fd, 1 << 16)
            if not chunk:
                raise RuntimeError(f"worker closed its pipe: {self._worker_err()}")
            self._buf += chunk


class BlenderRenderer(RenderState, WorkerPipe):
    def __init__(self, env, cams=None):
        super().__init__(env, cams)
        self.proc = None
        self._first_render = True
        self._recoloured = set()
        self.dir = tempfile.mkdtemp(prefix="roboarcade_render_")
        atexit.register(self.close)
        try:
            self._start()
        except BaseException:
            self.close()
            raise

    def _start(self):
        mjb = os.path.join(self.dir, "scene.mjb")
        mujoco.mj_saveModel(self.env.model, mjb, None)
        self.spawn(mjb, self.env.image_size, self.env.task)
        hello = self._readline(START_TIMEOUT)
        if not hello.get("ready"):
            raise RuntimeError(f"worker did not start: {hello}")
        logger.info(f"[render] blender ready: {hello.get('objects')} objects, cycles "
                    f"on {hello.get('device')}, {len(self.cams)} cameras {self.cams}")

    def render(self) -> dict[str, np.ndarray]:
        req = self._request()
        try:
            self.proc.stdin.write((json.dumps(req) + "\n").encode())
            self.proc.stdin.flush()
            timeout_s = FIRST_RENDER_TIMEOUT if self._first_render else RENDER_TIMEOUT
            self._first_render = False
            rep = self._readline(timeout_s)
            if not rep.get("ok"):
                raise RuntimeError(f"worker error: {rep.get('error')}")
            if set(req["geom_rgba"]) != self._recoloured:
                self._recoloured = set(req["geom_rgba"])
                logger.info(f"[render] recoloured {len(self._recoloured)} geoms")
            return {c: imageio.imread(p) for c, p in rep["files"].items()}
        except Exception as e:
            why = f"{type(e).__name__}: {e}; worker stderr: {self._worker_err()}"
            self.close()
            raise RuntimeError(why) from e

    def close(self):
        self.stop()
        shutil.rmtree(self.dir, ignore_errors=True)
