"""Fit the vortex knob collider to its visual surface offline.

Requires coacd==1.0.14 and trimesh in the build environment. Generated vertices
are embedded in scene.xml; simulation needs no extra dependencies.
"""

from pathlib import Path
import xml.etree.ElementTree as ET

import coacd
import numpy as np
import trimesh

from core.tasks import load_scene

PACK = Path(__file__).resolve().parent


def build():
    path = PACK / "scene.xml"
    root = ET.fromstring(path.read_text())
    mesh_path = (
        Path(load_scene(PACK).meshdir)
        / "instrument/vortex_mixer_genie_2/knob-visual-0.obj"
    )
    surface = trimesh.load(mesh_path, process=True)
    surface.merge_vertices(merge_norm=True, merge_tex=True)
    assert surface.is_watertight
    coacd.set_log_level("warn")
    parts = coacd.run_coacd(
        coacd.Mesh(surface.vertices, surface.faces),
        threshold=0.00025,
        real_metric=True,
        preprocess_mode="off",
        resolution=2000,
        mcts_iterations=40,
        mcts_nodes=15,
        seed=0,
    )
    asset = root.find("asset")
    body = root.find(".//body[@name='knob']")
    template = dict(body.find("geom[@class='collision']").attrib)
    for geom in list(body.findall("geom[@class='collision']")):
        body.remove(geom)
    for mesh in list(asset.findall("mesh")):
        if mesh.get("name", "").startswith("knob_aligned_"):
            asset.remove(mesh)
    for i, (vertices, _) in enumerate(parts):
        name = f"knob_aligned_{i}"
        ET.SubElement(
            asset,
            "mesh",
            name=name,
            vertex=" ".join(f"{x:.10g}" for x in np.asarray(vertices).ravel()),
        )
        attrs = dict(template)
        attrs.update(name=name, mesh=name, mass="0")
        ET.SubElement(body, "geom", **attrs)
    ET.indent(root, space="  ")
    path.write_text(ET.tostring(root, encoding="unicode") + "\n")
    print(f"Generated {len(parts)} knob collision hulls")


if __name__ == "__main__":
    build()
