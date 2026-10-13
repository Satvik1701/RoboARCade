"""A thumbnail per task for the teleop dashboard."""
import itertools
import json
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

from core.render import BlenderRenderer
from core.tasks import list_tasks
from core.sim_env import SimEnv

OUT = Path("data/task_thumbs")
RES = 240


def _objects_coincide(env, tol=0.01):
    xy = [env.data.body(n).xpos[:2] for n in env._free_objects]
    return any(np.linalg.norm(a - b) < tol for a, b in itertools.combinations(xy, 2))


def slug(task):
    return task.replace("/", "_")




def third_person(env):
    return next((c["name"] for c in env.spec["cameras"] if c["role"] == "third_person"),
                "agentview")


def render(task, blender=False):
    env = SimEnv(task=task, seed=0)
    env.reset(seed=0, randomize_init=False)
    if _objects_coincide(env):
        env.reset(seed=0, randomize_init=True)
    tp = third_person(env)
    if blender:
        br = BlenderRenderer(env)
        br.cams = [tp]
        try:
            frames = br.render()
        finally:
            br.close()
        if tp not in frames:
            raise RuntimeError("blender returned no frame")
        img = cv2.resize(frames[tp], (RES, RES), interpolation=cv2.INTER_AREA)
    else:
        r = mujoco.Renderer(env.model, RES, RES)
        r.update_scene(env.data, tp)
        img = r.render()
        r.close()
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / f"{slug(task)}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def main():
    blender, force = "--blender" in sys.argv, "--force" in sys.argv
    tasks = all_tasks()
    done = 0
    for t in tasks:
        if (OUT / f"{slug(t)}.png").exists() and not force:
            continue
        try:
            render(t, blender)
            done += 1
            print(f"[{done}] {t}", flush=True)
        except Exception as e:
            print(f"[skip] {t}: {e}", flush=True)
    print(f"done: {done} new thumbnails, {len(tasks)} tasks total")


if __name__ == "__main__":
    main()
