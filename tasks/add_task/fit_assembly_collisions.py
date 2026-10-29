"""Offline mjfit replacement of the two cylindrical assembly interfaces.

Run from the repo root with mjfit==0.1.0, scipy and trimesh installed:
    python -m tasks.add_task.fit_assembly_collisions gamepad cooling_manifold

Dimensions are measured in body coordinates from the unchanged Fabrica visual
meshes. Only the insertion stem and a square patch around its receiving hole
are replaced. Existing hulls crossing that patch are clipped, not discarded.
The old per-face erosion allowance and compiled body inertias are retained.
Generated vertices are inline; mjfit is not a simulation dependency.
"""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial import ConvexHull, QhullError

from core.tasks import TASKS, asset_dir, load_scene

CONFIG = {
    "gamepad": dict(
        target="part0",
        receiver="part3",
        xy=(-0.029322, -0.001709),
        peg_radius=0.0075,
        hole_radius=0.007875,
        peg_bottom=-0.02329483,
        peg_top=-0.00829483,
        hole_bottom=0.0045549,
        hole_top=0.0195549,
        patch_half_width=0.012,
        erosion=0.0005,
        floor=0.001,
    ),
    "cooling_manifold": dict(
        target="part5",
        receiver="part1",
        xy=(0.0750455, -0.075),
        peg_radius=0.009,
        hole_radius=0.00975,
        peg_bottom=-0.02938373,
        peg_top=-0.00688373,
        hole_bottom=-0.02005158,
        hole_top=0.00244842,
        patch_half_width=0.013,
        erosion=0.00025,
        floor=0,
    ),
}


def fmt(values):
    return " ".join(f"{float(x):.10g}" for x in np.asarray(values).ravel())


def geom_vertices(model, gid):
    mid = model.geom_dataid[gid]
    start, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, model.geom_quat[gid])
    return (
        model.mesh_vert[start : start + count] @ rotation.reshape(3, 3).T
        + model.geom_pos[gid]
    )


def clip(vertices, normal, offset):
    """Intersect a convex polyhedron with normal.dot(x) <= offset."""
    distance = vertices @ normal - offset
    if np.max(distance) <= 1e-10:
        return vertices
    if np.min(distance) >= -1e-10:
        return None
    hull = ConvexHull(vertices)
    edges = set()
    for a, b, c in hull.simplices:
        edges.update(tuple(sorted(e)) for e in ((a, b), (b, c), (c, a)))
    points = list(vertices[distance <= 0])
    for a, b in edges:
        if distance[a] * distance[b] < 0:
            points.append(
                vertices[a]
                + (vertices[b] - vertices[a])
                * distance[a]
                / (distance[a] - distance[b])
            )
    points = np.unique(np.round(points, 11), axis=0)
    try:
        hull = ConvexHull(points)
    except QhullError:
        return None
    return points[hull.vertices] if hull.volume > 1e-15 else None


def subtract(vertices, planes):
    """Disjoint convex pieces outside a convex clipping region."""
    # Do not split a hull that doesn't intersect the patch.
    if any(np.min(vertices @ n - d) >= -1e-9 for n, d in planes):
        return [vertices]
    pieces, rest = [], vertices
    for n, d in planes:
        if rest is None:
            break
        outside = clip(rest, -n, -d)
        if outside is not None:
            pieces.append(outside)
        rest = clip(rest, n, d)
    return pieces


def build(task):
    import mjfit
    import trimesh
    from shapely.geometry import Polygon, box

    pack = TASKS / task
    cfg = CONFIG[task]
    model = load_scene(pack).compile()
    root = ET.parse(pack / "scene.xml").getroot()
    asset = root.find("asset")
    mesh_lookup = {m.get("name"): m for m in asset.findall("mesh")}
    e, h = cfg["erosion"], cfg["patch_half_width"]
    xy = np.array(cfg["xy"])
    lo = cfg["hole_bottom"] + (e if not cfg["floor"] else -e)
    hi = cfg["hole_top"] - e
    stem_lo, stem_hi = cfg["peg_bottom"] + e, cfg["peg_top"] + e
    # The manifold's outside corners are chamfered: a square patch must not
    # fill those empty corners. Intersect the patch with the measured outline.
    receiver_id = model.body(cfg["receiver"]).id
    visual_gid = next(
        g
        for g in range(model.ngeom)
        if model.geom_bodyid[g] == receiver_id
        and not (model.geom_contype[g] or model.geom_conaffinity[g])
    )
    mid = model.geom_dataid[visual_gid]
    fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
    surface = trimesh.Trimesh(
        geom_vertices(model, visual_gid), model.mesh_face[fa : fa + fn]
    )
    section = surface.section([0, 0, 1], [0, 0, (lo + hi) / 2])
    outer = max(
        (Polygon(loop[:, :2] - xy) for loop in section.discrete), key=lambda p: p.area
    )
    patch = (
        outer.buffer(-e, join_style="mitre")
        .intersection(box(-h, -h, h, h))
        .simplify(1e-7)
    )
    plate_outline = np.asarray(patch.exterior.coords)[:-1]
    patch_planes = [
        (np.array([1.0, 0, 0]), xy[0] + h),
        (np.array([-1.0, 0, 0]), -xy[0] + h),
        (np.array([0.0, 1, 0]), xy[1] + h),
        (np.array([0.0, -1, 0]), -xy[1] + h),
    ]
    if cfg["floor"]:
        patch_planes.append((np.array([0.0, 0, -1]), -(lo - cfg["floor"])))
    geometry = {}
    bodies = {}

    def add(body, name, vertices, attrs):
        center = vertices.mean(axis=0)
        ET.SubElement(asset, "mesh", name=name, vertex=fmt(vertices - center))
        attrs = dict(attrs)
        for key in ("quat", "euler", "axisangle", "fromto", "size", "mass"):
            attrs.pop(key, None)
        attrs.update(name=name, type="mesh", mesh=name, pos=fmt(center), density="0")
        geom = ET.SubElement(body, "geom", attrs)
        geometry[name] = vertices
        return geom

    for name in (cfg["target"], cfg["receiver"]):
        body = root.find(f'.//body[@name="{name}"]')
        bodies[name] = body
        bid = model.body(name).id
        if body.find("inertial") is None:
            body.insert(
                0,
                ET.Element(
                    "inertial",
                    pos=fmt(model.body_ipos[bid]),
                    quat=fmt(model.body_iquat[bid]),
                    mass=fmt([model.body_mass[bid]]),
                    diaginertia=fmt(model.body_inertia[bid]),
                ),
            )
        gids = list(
            range(
                model.body_geomadr[bid],
                model.body_geomadr[bid] + model.body_geomnum[bid],
            )
        )
        for geom, gid in zip(list(body.findall("geom")), gids):
            if not (model.geom_contype[gid] or model.geom_conaffinity[gid]):
                continue
            geom_name = geom.get("name") or geom.get("mesh")
            if geom_name.startswith("fit_"):
                body.remove(geom)
                continue
            geom.set("name", geom_name)
            vertices = geom_vertices(model, gid)
            planes = (
                patch_planes
                if name == cfg["receiver"]
                else [(np.array([0.0, 0, 1]), stem_hi)]
            )
            pieces = subtract(vertices, planes)
            if len(pieces) == 1 and pieces[0] is vertices:
                geometry[geom_name] = vertices
                continue
            body.remove(geom)
            for i, piece in enumerate(pieces):
                add(body, f"{geom_name}_trim{i}", piece, geom.attrib)

    # Circumscribed 64-gons deviate by <13 um at these radii. Keep the
    # original nominal visual clearance plus the existing erosion allowance.
    hole_radius = cfg["hole_radius"] + e
    peg_radius = cfg["peg_radius"] - e
    generated = ET.fromstring(
        mjfit.peg(
            hole={"type": "circle", "radius": hole_radius, "n": 64},
            clearance=hole_radius - peg_radius,
            peg_length=stem_hi - stem_lo,
            plate=plate_outline,
            plate_thickness=hi - lo,
            bottom_thickness=cfg["floor"],
            standalone=False,
        )
    )
    generated_meshes = {
        m.get("name"): np.fromstring(m.get("vertex"), sep=" ").reshape(-1, 3)
        for m in generated.find("asset").findall("mesh")
    }
    attrs = dict(
        group="3",
        contype="1",
        conaffinity="1",
        condim="4",
        friction="0.6 0.3 0.0001",
        priority="1",
        solimp="0.998 0.998",
        solref="0.004 1",
        rgba="0.8 0.8 0.8 0",
    )
    for source, dest, shift, prefix in (
        ("plate", cfg["receiver"], [*xy, (lo + hi) / 2], "fit_socket_"),
        ("peg", cfg["target"], [0, 0, (stem_lo + stem_hi) / 2], "fit_stem_"),
    ):
        for i, geom in enumerate(
            generated.find(f'.//body[@name="{source}"]').findall("geom")
        ):
            vertices = generated_meshes[geom.get("mesh")] + np.array(shift)
            add(bodies[dest], prefix + str(i), vertices, attrs)

    if task == "cooling_manifold":
        contact = root.find("contact")
        for pair in list(contact.findall("pair")):
            contact.remove(pair)
        socket_geoms = [
            g
            for g in bodies[cfg["receiver"]].findall("geom")
            if g.get("name") in geometry
            and np.all(geometry[g.get("name")].max(axis=0)[:2] >= xy - 0.025)
            and np.all(geometry[g.get("name")].min(axis=0)[:2] <= xy + 0.025)
        ]
        for socket in socket_geoms:
            for peg in bodies[cfg["target"]].findall("geom"):
                if peg.get("name") not in geometry:
                    continue
                ET.SubElement(
                    contact,
                    "pair",
                    geom1=socket.get("name"),
                    geom2=peg.get("name"),
                    condim="4",
                    friction="0.6 0.6 0.002 0.0001 0.0001",
                    solref="0.004 1",
                    solimp="0.998 0.998",
                )
    used = {g.get("mesh") for g in root.iter("geom")}
    for mesh in list(asset.findall("mesh")):
        if mesh.get("name") not in used:
            asset.remove(mesh)
        elif mesh.get("name", "").startswith("fit_") and mesh is mesh_lookup.get(
            mesh.get("name")
        ):
            asset.remove(mesh)  # old generated mesh on a repeat build
    ET.indent(root, space="  ")
    # Keep the task XML's compact empty-element style to avoid unrelated diffs.
    (pack / "scene.xml").write_text(
        ET.tostring(root, encoding="unicode").replace(" />", "/>") + "\n"
    )
    manifest = json.loads((pack / "manifest.json").read_text())
    manifest["mjfit"] = dict(
        version="0.1.0",
        repo="https://github.com/younghyopark/mjfit",
        commit="5833f465896e1e06bc4be4f003be847f5183a4db",
        license="MIT",
        builder="tasks/add_task/fit_assembly_collisions.py",
        parameters=cfg,
        sides=64,
        notes="Only the mating stem/socket patch is replaced. Unchanged visual assets, body inertias, "
        "erosion allowances, contact coefficients, task initialization and grading.",
    )
    meshdir = (asset_dir(pack) / root.find("compiler").get("meshdir")).resolve()
    manifest["mjfit"]["visual_source_sha256"] = {
        name: hashlib.sha256(
            (meshdir / mesh_lookup[f"p{name[4:]}_visual"].get("file")).read_bytes()
        ).hexdigest()
        for name in (cfg["target"], cfg["receiver"])
    }
    if task == "cooling_manifold":
        manifest["contact"]["socket_torsion"]["pairs"] = len(
            root.find("contact").findall("pair")
        )
    manifest["files"]["scene.xml"] = hashlib.sha256(
        (pack / "scene.xml").read_bytes()
    ).hexdigest()[:16]
    (pack / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(
        task,
        {name: len(body.findall("geom")) - 1 for name, body in bodies.items()},
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tasks", nargs="+", choices=list(CONFIG))
    for task in parser.parse_args().tasks:
        build(task)
