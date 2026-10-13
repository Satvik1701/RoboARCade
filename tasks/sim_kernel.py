"""In-process sim kernel for scripted task tests."""
from __future__ import annotations

import numpy as np

from core.sim_env import Phase


class EvalModeError(RuntimeError):
    pass


class Kernel:
    def __init__(self, env, mode: str = "dev"):
        assert mode in ("dev", "eval")
        self.env = env
        self.mode = mode
        self._target_pos = {}
        self._target_euler = {}
        self._target_gripper = {}

    def _arm(self, arm):
        return arm or self.env.arms[0]

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        self.env.phase = Phase.RUNNING
        for a in self.env.arms:
            p = self.env.observe_proprio(a)
            self._target_pos[a] = p.eef_pos.copy()
            self._target_euler[a] = p.eef_euler.copy()
            self._target_gripper[a] = 1

    def depth(self, cam: str):
        return self.env.observe_camera()[f"{cam}_depth"].squeeze()

    def proprio(self, arm=None):
        return self.env.observe_proprio(arm)

    def raycast(self, cam: str, u: int, v: int):
        d = self.depth(cam)
        K = self.env.get_camera_intrinsics(cam)
        T = self.env.get_camera_extrinsics(cam)
        z = float(d[int(v), int(u)])
        p_cam = np.linalg.inv(K) @ [u, v, 1.0] * z
        return (T @ [*p_cam, 1.0])[:3]

    def set_ee_target(self, pos, euler, arm=None):
        arm = self._arm(arm)
        self._target_pos[arm] = np.asarray(pos, dtype=float)
        self._target_euler[arm] = np.asarray(euler, dtype=float)

    def set_gripper(self, open_frac: float, arm=None):
        self._target_gripper[self._arm(arm)] = int(round(open_frac))

    def execute(self, stream_fn=None, arm=None):
        arm = self._arm(arm)
        self.env.move_to(self._target_pos[arm], self._target_euler[arm],
                         self._target_gripper[arm], stream_fn=stream_fn, arm=arm)
        return self.proprio(arm)

    def _dev(self):
        if self.mode != "dev":
            raise EvalModeError("dev-only kernel call in eval mode")

    def object_poses(self) -> dict[str, np.ndarray]:
        self._dev()
        return {name: self.env.data.body(name).xpos.copy() for name in self.env.spec["objects"]}

    def save_state(self):
        self._dev()
        return self.env.checkpoint()

    def load_state(self, state):
        self._dev()
        self.env.restore(state)
