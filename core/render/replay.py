"""Re-render one recorded episode's agent-facing cameras through Blender.

    source set_env.sh && python -m core.render.replay <episode_dir> \
        [--camera NAME]

Replays cameras_states.npz frame by frame (state i is mp4 frame i), so the output
keeps cameras_segments.json's numbering and the dashboard's waypoint seeking still
lands where it did. By default, writes the third-person and all wrist cameras as a
horizontal strip in <episode_dir>/blender_feeds.mp4. ``--camera`` keeps the original
single-camera behaviour and writes <episode_dir>/blender_<camera>.mp4.

Needs an episode recorded with --record-state (the eval driver sets it).
"""
import argparse
import json
import time
from pathlib import Path

import imageio_ffmpeg
import numpy as np

from core.render import BlenderRenderer
from core.sim_env import SimEnv
from core.utils import episode_identity


def _restore_recorded_rgba(model, frame, rgba_frames, rgba):
    bank = int(np.searchsorted(rgba_frames, frame, side="right") - 1)
    if bank < 0:
        raise ValueError(f"RGBA bank has no snapshot for frame {frame}")
    model.geom_rgba[:] = rgba[bank]
    return bank


def _default_cameras(spec):
    third = next((c for c in spec["cameras"] if c["role"] == "third_person"), None)
    if third is None:
        raise ValueError("task has no third_person camera")
    wrists = [c for c in spec["cameras"] if c["role"] == "wrist"]
    if not wrists:
        raise ValueError("task has no wrist camera")
    return [c["name"] if "robot" not in c else f'{c["robot"]}_{c["name"]}'
            for c in [third, *wrists]]


def _restore_initial_state(env, seed, states, static, rgba_frames, rgba):
    if states.ndim != 2 or not len(states):
        raise ValueError('recording must contain at least one simulation state')
    if static is None or rgba is None:
        env.reset(seed=seed)
    if env.sim_state().size != states.shape[1]:
        raise ValueError(f'state width {states.shape[1]} != this pack\'s '
                         f'{env.sim_state().size}: the pack changed since the recording')
    if static is not None:
        env.set_static_poses(*static)
    if rgba is not None:
        expected = (len(rgba_frames), env.model.ngeom, 4)
        if rgba.shape != expected or len(rgba_frames) == 0 or rgba_frames[0] != 0 \
                or np.any(np.diff(rgba_frames) <= 0) or rgba_frames[-1] >= len(states):
            raise ValueError('invalid recorded RGBA bank')
        _restore_recorded_rgba(env.model, 0, rgba_frames, rgba)
    env.reset_to_sim_state(states[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode", help="an episode folder under data/episodes")
    ap.add_argument("--camera", help="render one named camera instead of the default feed strip")
    a = ap.parse_args()

    roll = Path(a.episode)
    try:
        task, seed = episode_identity(roll)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    with np.load(roll / "cameras_states.npz") as z:
        states = z["states"]
        static = (z["static_names"], z["static_poses"]) if "static_poses" in z.files else None
        has_frames, has_rgba = "rgba_frames" in z.files, "rgba" in z.files
        if has_frames != has_rgba:
            raise SystemExit("cameras_states.npz has an incomplete RGBA bank")
        rgba_frames = z["rgba_frames"] if has_rgba else None
        rgba = z["rgba"] if has_rgba else None
    fps = json.loads((roll / "cameras_segments.json").read_text())["fps"]

    env = SimEnv(task, seed=seed)
    _restore_initial_state(env, seed, states, static, rgba_frames, rgba)
    cams = [a.camera] if a.camera else _default_cameras(env.spec)
    missing = [cam for cam in cams if cam not in env.cameras]
    if missing:
        raise SystemExit(f"camera(s) not present in compiled scene: {missing}")

    out = roll / (f"blender_{a.camera}.mp4" if a.camera else "blender_feeds.mp4")
    part = out.with_name(out.stem + ".part.mp4")
    n, res = len(states), env.image_size
    writer = imageio_ffmpeg.write_frames(
        str(part), (res * len(cams), res), fps=fps, codec="libx264", macro_block_size=1,
        output_params=["-x264-params", "keyint=1",
                       "-crf", "23", "-movflags", "+faststart"])
    writer.send(None)

    br = BlenderRenderer(env, cams)
    t0 = time.monotonic()
    try:
        for i, s in enumerate(states):
            env.reset_to_sim_state(s)
            if rgba is not None:
                _restore_recorded_rgba(env.model, i, rgba_frames, rgba)
            rendered = br.render()
            lost = [cam for cam in cams if cam not in rendered]
            if lost:
                raise SystemExit(f"blender did not render {lost} at frame {i}")
            writer.send(np.ascontiguousarray(
                np.concatenate([rendered[cam][..., :3] for cam in cams], axis=1)))
            if i % 100 == 0 or i == n - 1:
                dt = time.monotonic() - t0
                print(f"{i + 1}/{n} frames · {dt / (i + 1) * 1000:.0f} ms/frame · "
                      f"{dt / 60:.1f} min in · {(n - i - 1) * dt / (i + 1) / 60:.1f} min left",
                      flush=True)
    finally:
        br.close()
        writer.close()
    part.replace(out)
    dt = time.monotonic() - t0
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {n} frames at {fps} fps, "
          f"{dt / n * 1000:.0f} ms/frame, cameras={cams})")


if __name__ == "__main__":
    main()
