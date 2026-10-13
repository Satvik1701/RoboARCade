"""Opt-in fixed seed groups sharing one Blender, owned by the evaluation driver."""
import argparse
import base64
import copy
import hashlib
import hmac
import io
import json
import logging
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import uuid

import imageio.v2 as imageio
import mujoco
import numpy as np

from core.render import (FIRST_RENDER_TIMEOUT, RENDER_TIMEOUT, START_TIMEOUT, WorkerPipe)
from core.render.lifecycle import follow_parent
from core.render.state import RenderState

logger = logging.getLogger(__name__)
MAX_SEEDS_PER_BLENDER = 5
MAX_BYTES = 64 * 1024 * 1024
SOCKET_ENV = "ROBOARCADE_RENDER_GROUP_SOCKET"
SEED_ENV = "ROBOARCADE_RENDER_GROUP_SEED"
TOKEN_ENV = "ROBOARCADE_RENDER_GROUP_TOKEN"


def lane_capacities(max_parallel, k):
    if max_parallel < 1 or not 1 <= k <= MAX_SEEDS_PER_BLENDER:
        raise ValueError(f"Positive concurrency and 1 <= seeds_per_blender <= {MAX_SEEDS_PER_BLENDER} required")
    return [min(k, max_parallel-i) for i in range(0, max_parallel, k)]


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def remaining(deadline):
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise TimeoutError("Render request deadline expired")
    return seconds


def receive(sock, deadline):
    def exact(size):
        chunks = []
        while size:
            sock.settimeout(remaining(deadline))
            part = sock.recv(min(size, 1 << 20))
            if not part:
                raise EOFError("Render connection closed")
            chunks.append(part)
            size -= len(part)
        return b"".join(chunks)
    size, = struct.unpack("!I", exact(4))
    if not 0 < size <= MAX_BYTES:
        raise ValueError("Invalid render message length")
    return json.loads(exact(size))


def send(sock, value, deadline):
    data = json.dumps(value).encode()
    if len(data) > MAX_BYTES:
        raise ValueError("Render message too large")
    sock.settimeout(remaining(deadline))
    sock.sendall(struct.pack("!I", len(data)) + data)


def call(path, value, deadline):
    with socket.socket(socket.AF_UNIX) as sock:
        sock.settimeout(remaining(deadline))
        sock.connect(str(path))
        send(sock, value, deadline)
        reply = receive(sock, deadline)
    if not reply.get("ok"):
        raise RuntimeError(f"Shared renderer: {reply.get('error', 'invalid response')}")
    return reply


def model_hash(model):
    data = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, data)
    return hashlib.sha256(data.tobytes()).hexdigest()


def colors(value, ngeom):
    rgba = np.asarray(value, dtype=float)
    if (rgba.shape != (ngeom, 4) or not np.isfinite(rgba).all() or
            np.any(rgba < 0) or np.any(rgba > 1)):
        raise ValueError("Invalid complete geometry colors")
    return rgba


class ImportBaseline:
    def __init__(self, model):
        self.rgba = colors(model.geom_rgba, model.ngeom).copy()
        self.static_hash = model_hash(model)

    def compatible(self, model, supported_geoms):
        rgba = colors(model.geom_rgba, len(self.rgba))
        changed = np.flatnonzero(np.any(rgba != self.rgba, axis=1))
        if (not set(changed) <= set(supported_geoms) or
                np.any(rgba[:, 3] != self.rgba[:, 3])):
            raise ValueError("Incompatible initial material colors; use dedicated rendering")
        comparison = copy.copy(model)
        comparison.geom_rgba[:] = self.rgba
        if model_hash(comparison) != self.static_hash:
            raise ValueError("Incompatible compiled scene/settings; use dedicated rendering")
        return changed.tolist()

    def delta(self, raw, seed_initial=None):
        if not isinstance(raw, dict) or set(raw) != {str(g) for g in range(len(self.rgba))}:
            raise ValueError("Incomplete geometry colors")
        rgba = colors([raw[str(g)] for g in range(len(self.rgba))], len(self.rgba))
        initial = self.rgba if seed_initial is None else seed_initial
        dynamic = np.any(rgba != initial, axis=1)
        initial_changed = np.any(initial != self.rgba, axis=1)
        imported = initial.astype(np.float16).astype(np.float32)
        return {str(g): (rgba[g] if dynamic[g] else imported[g]).tolist()
                for g in np.flatnonzero(dynamic | initial_changed)}


class GroupWorker(WorkerPipe):
    def __init__(self, directory, model_path, resolution, deadline, task):
        self.dir = str(directory)
        self.proc = None
        try:
            self.spawn(model_path, resolution, task, ROBOARCADE_RENDER_OWNER_PID=str(os.getpid()))
            self.hello = self._readline(min(START_TIMEOUT, remaining(deadline)))
            if not self.hello.get("ready") or self.hello.get("device") not in ("OPTIX", "CUDA"):
                raise RuntimeError("Shared Blender requires a ready GPU backend")
        except BaseException:
            self.stop()
            raise

    def render(self, state, seed, deadline):
        self.proc.stdin.write((json.dumps(dict(state, seed=str(seed))) + "\n").encode())
        self.proc.stdin.flush()
        result = self._readline(remaining(deadline))
        if not result.get("ok"):
            raise RuntimeError(f"Blender render failed: {result.get('error')}")
        return result


class Broker:
    def __init__(self, directory, tokens, worker_factory=GroupWorker):
        self.directory = Path(directory)
        self.tokens = {int(seed): token for seed, token in tokens.items()}
        self.worker_factory = worker_factory
        self.worker = None
        self.clients = {}
        self.key = None
        self.baseline = None
        self.failed = None
        self.last_finished = 0
        self.events = (self.directory / "requests.jsonl").open("x", buffering=1)

    def event(self, **value):
        self.events.write(json.dumps({"at": time.time(), **value}) + "\n")

    def authenticate(self, request):
        seed = request.get("seed")
        expected = self.tokens.get(seed)
        token = request.get("token")
        if expected is None or not isinstance(token, str) or not hmac.compare_digest(expected, token):
            raise PermissionError("Unregistered seed capability")
        if self.failed:
            raise RuntimeError(self.failed)
        return seed

    def register(self, request, pid, deadline):
        seed = self.authenticate(request)
        if seed in self.clients:
            raise ValueError("Seed already registered; start a fresh group to resume")
        path = self.directory / f"seed{seed}.mjb"
        if path.is_symlink() or not path.is_file():
            raise ValueError("Missing or symlinked seed scene")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != request["scene_hash"]:
            raise ValueError("Compiled scene changed before registration")
        model = mujoco.MjModel.from_binary_path(str(path))
        if model.ngeom != request["ngeom"]:
            raise ValueError("Wrong compiled geometry count")
        if not re.fullmatch(r"[a-z0-9_]+", request["task"]):
            raise ValueError("Bad task name")
        baseline = self.baseline or ImportBaseline(model)
        changed = baseline.compatible(model, self.worker.hello.get("seed_color_geoms", [])
                                      if self.worker is not None else [])
        key = (baseline.static_hash, request["resolution"], tuple(request["cameras"]),
               request["ngeom"], tuple(sorted(request["flex_sizes"].items())), request["task"])
        if self.key is not None and key != self.key:
            raise ValueError("Incompatible compiled scene/settings; use dedicated rendering")
        if self.worker is None:
            self.key = key
            self.baseline = baseline
            try:
                self.worker = self.worker_factory(self.directory, path, request["resolution"], deadline, request["task"])
            except Exception:
                self.failed = "Shared Blender startup failed; see worker.err"
                self.event(event="worker_failed", error=self.failed)
                raise RuntimeError(self.failed) from None
            self.event(event="worker_ready", scene_hash=actual, **self.worker.hello)
        self.clients[seed] = {"pid": pid, "seq": 0, "reply": None,
                              "initial_rgba": model.geom_rgba.copy()}
        self.event(event="registered", seed=seed, simulator_pid=pid, scene_hash=actual,
                   import_scene_hash=self.key[0], initial_color_overrides=changed)
        return {"ok": True, **self.worker.hello}

    def render(self, request, pid, deadline):
        seed = self.authenticate(request)
        client = self.clients.get(seed)
        if client is None or client["pid"] != pid:
            raise PermissionError("Render sender does not own this seed")
        seq, state = request["seq"], request["state"]
        remaining(deadline)
        if seq == client["seq"] and client["reply"] is not None:
            if state != client["state"]:
                raise ValueError("Duplicate sequence with different state")
            return dict(client["reply"], reused=True)
        if seq != client["seq"] + 1:
            raise ValueError("Nonsequential render request")
        if set(state["geoms"]) != {str(i) for i in range(self.key[3])}:
            raise ValueError("Incomplete geometry state")
        if any(len(v) != 12 for v in state["geoms"].values()):
            raise ValueError("Invalid geometry transform")
        rgba_delta = self.baseline.delta(state.get("geom_rgba"), client["initial_rgba"])
        if tuple(c["name"] for c in state["cams"]) != self.key[2]:
            raise ValueError("Wrong camera set/order")
        if tuple(sorted((k, len(v)) for k, v in state["flexes"].items())) != self.key[4]:
            raise ValueError("Incomplete deformable state")
        started = time.monotonic()
        try:
            result = self.worker.render(dict(state, geom_rgba=rgba_delta), seed, deadline)
            if set(result["files"]) != set(self.key[2]):
                raise RuntimeError("Blender did not return all cameras")
            images = {name: base64.b64encode(Path(path).read_bytes()).decode("ascii")
                      for name, path in result["files"].items()}
        except Exception:
            self.failed = "Shared Blender render failed; group is terminal (see worker.err)"
            self.worker.stop()
            self.event(event="worker_failed", seed=seed, seq=seq, error=self.failed)
            raise RuntimeError(self.failed) from None
        finished = time.monotonic()
        timing = {"queue_s": started - request["enqueued"], "service_s": finished - started,
                  "busy_wait_s": max(0, self.last_finished - request["enqueued"])}
        self.last_finished = finished
        self.event(event="render", seed=seed, seq=seq, exposure=result["exposure"], **timing)
        reply = {"ok": True, "images": images, "exposure": result["exposure"], **timing}
        client.update(seq=seq, reply=reply, state=state)
        return reply

    def serve(self):
        with socket.socket(socket.AF_UNIX) as listener:
            listener.bind(str(self.directory / "render.sock"))
            listener.listen(len(self.tokens))
            _write_json(self.directory / "listening.json", {"pid": os.getpid()})
            while True:
                with listener.accept()[0] as conn:
                    try:
                        pid, uid, _ = struct.unpack("3i", conn.getsockopt(
                            socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")))
                        if uid != os.getuid():
                            raise PermissionError("Foreign render uid")
                        request = receive(conn, time.monotonic() + 5)
                        deadline = min(request["deadline"], time.monotonic() + FIRST_RENDER_TIMEOUT)
                        remaining(deadline)
                        if request["op"] == "register":
                            reply = self.register(request, pid, deadline)
                        elif request["op"] == "render":
                            reply = self.render(request, pid, deadline)
                        else:
                            raise ValueError("Unknown render operation")
                    except Exception as exc:
                        reply = {"ok": False, "error": str(exc)}
                        self.event(event="request_rejected", error=str(exc))
                    try:
                        send(conn, reply, time.monotonic() + 5)
                    except (OSError, EOFError, TimeoutError):
                        self.event(event="client_disconnected")

    def close(self):
        if self.worker is not None:
            self.worker.stop()
        self.events.close()


class GroupRenderer(RenderState):
    def __init__(self, env):
        super().__init__(env)
        self.socket = Path(os.environ[SOCKET_ENV])
        self.seed = int(os.environ[SEED_ENV])
        self.token = os.environ[TOKEN_ENV]
        self.seq = 0
        self.lock = threading.Lock()
        deadline = time.monotonic() + FIRST_RENDER_TIMEOUT
        path = self.socket.parent / f"seed{self.seed}.mjb"
        if path.exists():
            raise RuntimeError("Seed scene slot already occupied")
        try:
            mujoco.mj_saveModel(env.model, str(path), None)
            state = self._request()
            result = call(self.socket, self.message("register", deadline,
                scene_hash=hashlib.sha256(path.read_bytes()).hexdigest(),
                resolution=env.image_size, task=env.task, cameras=self.cams, ngeom=env.model.ngeom,
                flex_sizes={k: len(v) for k, v in state["flexes"].items()}), deadline)
        finally:
            path.unlink(missing_ok=True)
        logger.info("[render] shared blender ready: cycles on %s, %s cameras", result["device"], len(self.cams))

    def message(self, op, deadline, **payload):
        return dict(payload, op=op, seed=self.seed, token=self.token, deadline=deadline)

    def _request(self):
        return super()._request(complete_rgba=True)

    def render(self):
        with self.lock:
            state = self._request()
            self.seq += 1
            stamp = time.monotonic()
            deadline = stamp + (FIRST_RENDER_TIMEOUT if self.seq == 1 else RENDER_TIMEOUT)
            result = call(self.socket, self.message("render", deadline, seq=self.seq,
                state=state, enqueued=stamp), deadline)
            images = {name: imageio.imread(io.BytesIO(base64.b64decode(value, validate=True)))
                      for name, value in result["images"].items()}
            if set(images) != set(self.cams) or any(
                    image.dtype.name != "uint8" or image.shape not in (
                        (self.env.image_size, self.env.image_size, 4),
                        (self.env.image_size, self.env.image_size, 3))
                    for image in images.values()):
                raise RuntimeError("Wrong render camera count/dimensions")
            return images


class RenderGroup:
    def __init__(self, seeds, log_root, startup_lock=None):
        if not seeds or len(set(seeds)) != len(seeds) or len(seeds) > MAX_SEEDS_PER_BLENDER:
            raise ValueError(f"A render group needs 1..{MAX_SEEDS_PER_BLENDER} distinct seeds")
        self.seeds = list(seeds)
        self.id = uuid.uuid4().hex[:12]
        self.directory = Path(tempfile.mkdtemp(prefix="roboarcade_group_", dir="/tmp"))
        self.log_dir = Path(log_root) / self.id
        self.log_dir.mkdir(parents=True)
        self.startup_lock = startup_lock if startup_lock is not None else threading.Lock()
        self.tokens = {seed: secrets.token_hex(32) for seed in seeds}
        self.proc = None
        self.closed = False
        try:
            _write_json(self.directory / "capabilities.json", self.tokens)
            env = {k: v for k, v in os.environ.items() if k in (
                "PATH", "HOME", "USER", "LANG", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES",
                "NVIDIA_VISIBLE_DEVICES", "ROBOARCADE_BLENDER", "ROBOARCADE_BLENDER_PY",
                "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")}
            env.update(PYTHONPATH=str(Path(__file__).resolve().parents[2]),
                       PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1")
            with (self.log_dir / "broker.log").open("x") as log:
                self.proc = subprocess.Popen([sys.executable, "-m", "core.render.group",
                    "--directory", str(self.directory), "--owner-pid", str(os.getpid())],
                    env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True)
            until = time.monotonic() + 30
            while not (self.directory / "listening.json").exists():
                if self.proc.poll() is not None or time.monotonic() > until:
                    raise RuntimeError(f"Render broker failed to start; see {self.log_dir / 'broker.log'}")
                time.sleep(.05)
        except BaseException:
            self.close()
            raise

    def environment(self, seed):
        return {SOCKET_ENV: str(self.directory / "render.sock"), SEED_ENV: str(seed),
                TOKEN_ENV: self.tokens[seed]}

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.proc is not None and self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                    self.proc.wait(timeout=5)
            for path in self.directory.iterdir():
                if path.name in ("requests.jsonl", "worker.err", "listening.json"):
                    shutil.copy2(path, self.log_dir / path.name)
        finally:
            shutil.rmtree(self.directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--owner-pid", type=int, required=True)
    args = parser.parse_args()
    def stop(*_):
        raise KeyboardInterrupt("Render owner stopped")
    signal.signal(signal.SIGTERM, stop)
    follow_parent(args.owner_pid)
    if args.directory.stat().st_mode & 0o077:
        raise PermissionError("Renderer directory must be private")
    tokens = json.loads((args.directory / "capabilities.json").read_text())
    broker = Broker(args.directory, tokens)
    try:
        broker.serve()
    finally:
        broker.close()


if __name__ == "__main__":
    main()
