"""Viewer of a task's visual meshes next to its collision geoms."""
import argparse

import mujoco
import numpy as np
import trimesh
import viser

from core.tasks import list_tasks
from core.sim_env import SimEnv

ORANGE = (255, 133, 28)


def _geom_mesh(m, g):
    t, s = m.geom_type[g], m.geom_size[g]
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        did = m.geom_dataid[g]
        va, vn = m.mesh_vertadr[did], m.mesh_vertnum[did]
        fa, fn = m.mesh_faceadr[did], m.mesh_facenum[did]
        return m.mesh_vert[va:va + vn].copy(), m.mesh_face[fa:fa + fn].copy()
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        tm = trimesh.creation.box(extents=2 * s[:3])
    elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
        tm = trimesh.creation.icosphere(radius=float(s[0]))
    elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        tm = trimesh.creation.cylinder(radius=float(s[0]), height=float(2 * s[1]))
    elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        tm = trimesh.creation.capsule(radius=float(s[0]), height=float(2 * s[1]))
    else:
        return None
    return np.asarray(tm.vertices), np.asarray(tm.faces)


def _merge(chunks):
    verts, faces, off = [], [], 0
    for v, f in chunks:
        verts.append(v)
        faces.append(f + off)
        off += len(v)
    return (np.concatenate(verts), np.concatenate(faces)) if verts else None


def build(server: viser.ViserServer, task: str):
    env = SimEnv(task=task, seed=1)
    env.reset(randomize_init=False)
    m, d = env.model, env.data
    handles = []
    for b in range(m.nbody):
        name = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}").replace("/", "_")
        ga, ng = m.body_geomadr[b], m.body_geomnum[b]
        vis, coll, color = [], [], (150, 150, 150)
        for g in range(ga, ga + ng):
            vf = _geom_mesh(m, g)
            if vf is None:
                continue
            v, f = vf
            world = d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T
            if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
                vis.append((world, f))
                color = tuple(int(255 * c) for c in m.geom_rgba[g][:3])
            else:
                coll.append((world, f))
        mv = _merge(vis)
        if mv is not None:
            handles.append(server.scene.add_mesh_simple(
                f"/visual/{name}", mv[0], mv[1], color=color, flat_shading=True))
        mc = _merge(coll)
        if mc is not None:
            handles.append(server.scene.add_mesh_simple(
                f"/collision/{name}", mc[0], mc[1], color=ORANGE, opacity=0.6,
                side="double", flat_shading=True))
    return handles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8300)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    tasks = list_tasks()
    server = viser.ViserServer(host=args.host, port=args.port, label="roboarcade geometry")
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/grid", width=2.0, height=2.0, position=(0.0, 0.0, 0.0))

    default = "fabrica/beam/insert_0"
    task_dd = server.gui.add_dropdown("Task", tasks, initial_value=default if default in tasks else tasks[0])
    show_vis = server.gui.add_checkbox("Visual mesh", True)
    show_coll = server.gui.add_checkbox("Collision geoms", True)
    handles = []

    def reload():
        nonlocal handles
        for h in handles:
            h.remove()
        handles = build(server, task_dd.value)
        apply_visibility()

    def apply_visibility():
        for h in handles:
            h.visible = show_vis.value if h.name.startswith("/visual/") else show_coll.value

    task_dd.on_update(lambda _: reload())
    show_vis.on_update(lambda _: apply_visibility())
    show_coll.on_update(lambda _: apply_visibility())
    reload()
    server.sleep_forever()


if __name__ == "__main__":
    main()
