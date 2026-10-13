"""Lights Out: deals the board, toggles it on presses and paints the buttons.

The board lives in data.userdata so every saved state carries it: userdata[:n] is the board (1 for a blue button) and
userdata[n:2n] marks the buttons that were down at the last control step, so a held press toggles once. The reset RNG
deals a board that a unique press set in task.json's `lights_out` press band turns all red, which is the goal."""
import logging

import numpy as np

RGBA = np.array(((0.96, 0.26, 0.33, 1.0), (0.35, 0.55, 0.91, 1.0)))
DOWN_M = -0.02

logger = logging.getLogger(__name__)


def toggle_matrix(rows, cols):
    n = rows * cols
    matrix = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        r, c = divmod(i, cols)
        for dr, dc in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            if 0 <= r + dr < rows and 0 <= c + dc < cols:
                matrix[(r + dr) * cols + c + dc, i] = 1
    return matrix


def deal(rng, rows, cols, presses):
    n = rows * cols
    lo, hi = (int(v) for v in presses)
    if not 0 <= lo <= hi <= n:
        raise ValueError(f"invalid Lights Out press band [{lo}, {hi}] for {rows}x{cols}")
    k = int(rng.integers(lo, hi + 1))
    solution = np.zeros(n, dtype=np.uint8)
    solution[np.asarray(rng.choice(n, size=k, replace=False), dtype=int)] = 1
    return ((toggle_matrix(rows, cols) @ solution) & 1).tolist(), k


class Dynamics:
    def __init__(self, env):
        self.cfg = env.spec["lights_out"]
        rows, cols = self.cfg["rows"], self.cfg["cols"]
        self.n = rows * cols
        self.toggles = toggle_matrix(rows, cols)
        self.qadr = [env.model.joint(f"buttonbox_joint_{i}").qposadr[0] for i in range(self.n)]
        self.geoms = [env.model.geom(f"btngeom_{i}").id for i in range(self.n)]

    def down(self, env):
        return env.data.qpos[self.qadr] <= DOWN_M

    def deal(self, env):
        board, k = deal(env._rng, self.cfg["rows"], self.cfg["cols"], self.cfg["presses"])
        env.data.userdata[:self.n] = board
        env.data.userdata[self.n:2 * self.n] = self.down(env)
        logger.info(f"[lights_out] dealt {self.cfg['rows']}x{self.cfg['cols']} with unique {k}-press solution")
        self.paint(env)

    def control_step(self, env):
        board, was_down = env.data.userdata[:self.n], env.data.userdata[self.n:2 * self.n]
        down = self.down(env)
        board[:] = (board + self.toggles @ (down & (was_down == 0))) % 2
        was_down[:] = down
        self.paint(env)

    def paint(self, env):
        env.model.geom_rgba[self.geoms] = RGBA[env.data.userdata[:self.n].astype(int)]

    def restore(self, env):
        self.paint(env)
