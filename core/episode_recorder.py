"""Records an episode: camera video, per-waypoint frame spans and saved states.

Grading and replay read the saved per-frame sim states back."""
import json
import logging
import os
from enum import Enum

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ActMode(Enum):
    Waypoint = 0
    Dense = 1
    Interpolate = 3


class EpisodeRecorder:
    def __init__(self, episode_dir, video_name, vis_dim=(320, 240), state_fn=None,
                 static_fn=None):
        os.makedirs(episode_dir, exist_ok=True)
        self.episode_dir = episode_dir
        self.video_name = video_name
        self.vis_dim = vis_dim
        self.state_fn = state_fn
        self.static_fn = static_fn
        self._reset()

    def _reset(self):
        self.waypoint_idx = -1
        self._states: list = []
        self._state_wp: list[int] = []
        self._writer = None
        self._writer_path = None
        self._writer_failed = False
        self._spans: list[list[int]] = []
        self._undone: list[list[int]] = []
        self._frames_written = 0
        self._pre_wp_frames = 0

    def record(self, mode: ActMode, obs: dict[str, np.ndarray]):
        if mode == ActMode.Waypoint:
            self.waypoint_idx += 1
            self._spans.append([self._frames_written, self._frames_written])

        views = [
            (k[: -len("_image")], cv2.resize(v, self.vis_dim))
            for k, v in obs.items()
            if "image" in k
        ]
        if views:
            self._write_frame(self._tile_views(views))
            if self.state_fn is not None:
                self._states.append(np.asarray(self.state_fn(), dtype=np.float64))
                self._state_wp.append(self.waypoint_idx)
            if self.waypoint_idx < 0:
                self._pre_wp_frames += 1
            else:
                self._spans[-1][1] = self._frames_written

    def discard_last_waypoint(self):
        if not self._spans:
            return
        span = self._spans.pop()
        self._undone.append(span)
        del self._states[span[0]:]
        del self._state_wp[span[0]:]
        self.waypoint_idx -= 1

    def _cut_undone_frames(self):
        dead = set()
        for f0, f1 in self._undone:
            dead.update(range(f0, f1))
        cap = cv2.VideoCapture(self._writer_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        tmp = self._writer_path + ".cut.mp4"
        writer = self._open_writer(tmp, w, h)
        if writer is None:
            cap.release()
            return
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i not in dead:
                writer.write(frame)
            i += 1
        cap.release()
        writer.release()
        os.replace(tmp, self._writer_path)

        for span in self._spans:
            span[0] -= sum(f1 - f0 for f0, f1 in self._undone if f1 <= span[0])
            span[1] -= sum(f1 - f0 for f0, f1 in self._undone if f1 <= span[1])
        self._frames_written -= len(dead)

    def _write_segments(self):
        path = os.path.join(self.episode_dir, f"{self.video_name}_segments.json")
        with open(path, "w") as f:
            json.dump(
                {
                    "fps": 30,
                    "pre_waypoint_frames": self._pre_wp_frames,
                    "total_frames": self._frames_written,
                    "segments": [
                        {"waypoint": i, "start": f0, "end": f1}
                        for i, (f0, f1) in enumerate(self._spans)
                    ],
                },
                f,
                indent=2,
            )
        logger.info(f"saved {path}")

    @staticmethod
    def _tile_views(views):
        n = len(views)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        tiles = [EpisodeRecorder._label_tile(name, img) for name, img in views]
        tiles += [np.zeros_like(tiles[0])] * (rows * cols - n)
        return np.vstack([np.hstack(tiles[r * cols : (r + 1) * cols]) for r in range(rows)])

    @staticmethod
    def _label_tile(name, img):
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(img, (0, 0), (tw + 6, th + 8), (0, 0, 0), -1)
        cv2.putText(img, name, (3, th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    def _open_writer(self, path, w, h):
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
        if writer.isOpened():
            return writer
        writer.release()
        logger.warning(f"[recorder] no mp4v encoder; {self.video_name}.mp4 not written")
        return None

    def _write_frame(self, vis):
        if self._writer is None:
            if self._writer_failed:
                return
            H, W = vis.shape[:2]
            self._writer_path = os.path.join(self.episode_dir, f"{self.video_name}.mp4")
            self._writer = self._open_writer(self._writer_path, W, H)
            if self._writer is None:
                self._writer_failed = True
                return
        self._writer.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        self._frames_written += 1

    def end_episode(self, save):
        if self._writer is not None:
            self._writer.release()
            if save:
                if self._undone:
                    self._cut_undone_frames()
                logger.info(f"saved {self._writer_path}")
                self._write_segments()
                if self._states:
                    sp = os.path.join(self.episode_dir, f"{self.video_name}_states.npz")
                    static = dict(zip(("static_names", "static_poses"), self.static_fn())) if self.static_fn else {}
                    np.savez_compressed(sp, states=np.asarray(self._states),
                                        waypoint=np.asarray(self._state_wp, dtype=np.int32), **static)
                    logger.info(f"saved {sp} ({len(self._states)} frames, "
                                f"{os.path.getsize(sp) / 1e6:.2f} MB)")
            else:
                os.remove(self._writer_path)
                logger.info("Episode discarded")
        else:
            logger.info("no frames recorded")

        self._reset()
