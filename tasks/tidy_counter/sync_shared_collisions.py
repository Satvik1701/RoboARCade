"""Reuse bowl_on_plate's validated hulls for the same HF visual assets.

Run with python -m tasks.tidy_counter.sync_shared_collisions after rebuilding
the source task. The destination keeps its own mass, inertia and task setup.
"""

from copy import deepcopy
import hashlib
import json
import xml.etree.ElementTree as ET

from core.tasks import TASKS, asset_dir, load_scene
import numpy as np


def fmt(values):
    return " ".join(f"{float(x):.10g}" for x in np.asarray(values).ravel())


def sync():
    source, target = TASKS / "bowl_on_plate", TASKS / "tidy_counter"
    original = load_scene(target).compile()
    src, dst = (
        ET.parse(source / "scene.xml").getroot(),
        ET.parse(target / "scene.xml").getroot(),
    )
    counts = {}
    for stem, prefix, filename in [
        ("akita_black_bowl_1", "bowl_aligned_", "akita_black_bowl_vis.msh"),
        ("plate_1", "plate_aligned_", "model_vis.msh"),
    ]:
        assert (asset_dir(source) / "assets" / filename).read_bytes() == (
            asset_dir(target) / "assets" / filename
        ).read_bytes()
        visual_name = stem + (
            "_akita_black_bowl_vis" if stem.startswith("akita") else "_model_vis"
        )
        assert (
            src.find(f"./asset/mesh[@name='{visual_name}']").attrib
            == dst.find(f"./asset/mesh[@name='{visual_name}']").attrib
        )
        body = dst.find(f".//body[@name='{stem}_main']")
        if body.find("inertial") is None:
            b = original.body(stem + "_main").id
            body.insert(
                0,
                ET.Element(
                    "inertial",
                    pos=fmt(original.body_ipos[b]),
                    quat=fmt(original.body_iquat[b]),
                    mass=fmt([original.body_mass[b]]),
                    diaginertia=fmt(original.body_inertia[b]),
                ),
            )
        for geom in list(body.findall("geom")):
            if geom.get("name") != stem + "_g0":
                body.remove(geom)
        asset = dst.find("asset")
        for mesh in list(asset.findall("mesh")):
            if mesh.get("name", "").startswith(prefix):
                asset.remove(mesh)
        meshes = [
            m
            for m in src.findall("./asset/mesh")
            if m.get("name", "").startswith(prefix)
        ]
        asset.extend(deepcopy(meshes))
        body.extend(
            deepcopy(
                [
                    g
                    for g in src.findall(f".//body[@name='{stem}_main']/geom")
                    if g.get("name", "").startswith(prefix)
                ]
            )
        )
        # Rebind task-specific table/burner contact settings to every new hull.
        contact = dst.find("contact")
        if contact is not None:
            templates = {}
            for pair in list(contact.findall("pair")):
                for side, other in [("geom1", "geom2"), ("geom2", "geom1")]:
                    name = pair.get(side, "")
                    if (
                        name.startswith(stem + "_g") and name != stem + "_g0"
                    ) or name.startswith(prefix):
                        attrs = dict(pair.attrib)
                        attrs.pop(side)
                        templates[(side, tuple(sorted(attrs.items())))] = attrs
                        contact.remove(pair)
                        break
            for mesh in meshes:
                for (side, _), attrs in templates.items():
                    ET.SubElement(contact, "pair", **{side: mesh.get("name"), **attrs})
        counts[stem] = len(meshes)
    ET.indent(dst, space="  ")
    ET.ElementTree(dst).write(target / "scene.xml", encoding="unicode")
    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["collision_reused_from"] = "tasks/bowl_on_plate/scene.xml"
    manifest["collision_parts"] = counts
    manifest["collision_source_sha256"] = json.loads(
        (source / "manifest.json").read_text()
    )["collision_source_sha256"]
    manifest["collision_notes"] = (
        "Identical visual files and scales; reuse the validated bowl/plate convex hulls. Preserve tidy_counter mass, inertia, friction and task setup."
    )
    for name in ("scene.xml", "sync_shared_collisions.py"):
        manifest["files"][name] = hashlib.sha256(
            (target / name).read_bytes()
        ).hexdigest()[:16]
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    print(counts)


if __name__ == "__main__":
    sync()
