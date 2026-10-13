"""Per-task facts for the teleop dashboard's Docs tab."""
import json
from pathlib import Path

from core.initial_layout import DEFAULT_INIT_REGION, init_regions
from core.tasks import list_tasks
from core.sim_env import SimEnv


def task_doc(task):
    env = SimEnv(task=task, seed=0)
    env.reset(seed=0, randomize_init=False)
    spec = env.spec
    reg = spec["randomization"].get("region") or DEFAULT_INIT_REGION
    regions = init_regions(env)
    objects = []
    for name, (pos, quat) in env.init_bases.items():
        r = regions.get(name, {})
        objects.append({"num": r.get("num"), "name": name,
                        "static": name in env._static_objects,
                        "base_pos": [round(float(x), 4) for x in pos],
                        "base_quat": [round(float(x), 4) for x in quat],
                        "x": r.get("x", 0.0), "y": r.get("y", 0.0),
                        "yaw": r.get("yaw", 0.0), "group": r.get("group")})
    return {
        "family": spec["family"], "name": spec["name"], "language": spec["language"],
        "success": spec["success"],
        "randomization": {"type": spec["randomization"].get("type"),
                          "region": reg,
                          "grouped": bool(spec["randomization"].get("groups")),
                          "objects": objects},
        "robots": [{"name": r["name"], "base_pos": r["base_pos"]} for r in spec["robots"]],
        "max_len": spec["termination"]["max_len"],
    }


def main():
    tasks = list_tasks()
    out = {}
    for t in tasks:
        try:
            out[t] = task_doc(t)
            print(f"[{len(out)}] {t}", flush=True)
        except Exception as e:
            print(f"[skip] {t}: {e}", flush=True)
    Path("data").mkdir(exist_ok=True)
    Path("data/task_docs.json").write_text(json.dumps(out, indent=1))
    print(f"wrote data/task_docs.json ({len(out)} tasks)")


if __name__ == "__main__":
    main()
