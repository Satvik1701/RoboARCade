"""Build the machine's vector labels and physical keys (offline only).

Run from any directory with numpy, scipy, matplotlib and shapely installed.
Generated OBJ lettering lives in the Hugging Face assets tree; rendering needs
no installed fonts or build dependencies. The scene retains joints/materials.
Coordinates retain the original machine's button sites and Program 1 joint.
"""
from pathlib import Path
import os
import re
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from scipy.spatial.transform import Rotation
from shapely.geometry import Polygon
from shapely import constrained_delaunay_triangles

HERE = Path(__file__).resolve().parent
ASSETS = Path(os.environ.get("ROBOARCADE_ASSETS_DIR", HERE.parents[1] / "assets"))
PANEL = ASSETS / "tasks" / "autobio" / "start_incubation" / "panel"
RIGHT = np.array([0., 0., 1.])
UP = np.array([.787815, .615912, 0.])
UP /= np.linalg.norm(UP)
NORMAL = np.cross(RIGHT, UP)
QUAT = Rotation.from_matrix(np.column_stack([RIGHT, UP, NORMAL])).as_quat()[[3, 0, 1, 2]]
FONT = FontProperties(family="DejaVu Sans", weight="bold")


def numbers(values):
    return " ".join(f"{v:.9g}" for v in np.asarray(values).ravel())


def embed_xml(scene, kind, root, indent):
    start, end = f"<!-- BEGIN GENERATED PANEL {kind} -->", f"<!-- END GENERATED PANEL {kind} -->"
    ET.indent(root, space="  ")
    inner = ET.tostring(root, encoding="unicode").split('\n')[1:-1]
    block = '\n'.join([start, *[indent + line[2:] for line in inner], indent + end])
    if start in scene:
        return re.sub(re.escape(start) + r".*?" + re.escape(end), lambda _: block, scene, flags=re.S)
    return scene.replace(f'<include file="panel-{kind.lower()}.xml" />', block)


def build():
    PANEL.mkdir(parents=True, exist_ok=True)
    # Include the source font's license with the distributed glyph outlines.
    font_license = Path(matplotlib.get_data_path()) / "fonts/ttf/LICENSE_DEJAVU"
    (PANEL / "LICENSE_DEJAVU.txt").write_bytes(font_license.read_bytes())
    scene = ET.parse(HERE / "scene.xml").getroot()
    sites = {s.get("name"): np.fromstring(s.get("pos"), sep=" ")
             for s in scene.findall(".//site") if s.get("name", "").startswith("button_")}
    assets = ET.Element("mujocoinclude")
    controls = ET.Element("mujocoinclude")
    for name, rgba in [("panel", "0.88 0.90 0.92 1"), ("ink", "0.015 0.035 0.065 1"),
                       ("blue", "0.015 0.16 0.38 1"), ("white", "1 1 1 1")]:
        ET.SubElement(assets, "material", name="incubation_" + name, rgba=rgba,
                      specular="0", shininess="0", metallic="0", roughness="1")
    meshes = {}

    def label(parent, text, pos, width, height, color="ink", quat=QUAT):
        key = (text, width, height)
        if key not in meshes:
            path = TextPath((0, 0), text, size=1, prop=FONT)
            shape = Polygon()
            # XOR includes counter-shapes (holes in letters and digits).
            for contour in path.to_polygons():
                shape = shape.symmetric_difference(Polygon(contour))
            xmin, ymin, xmax, ymax = shape.bounds
            scale = min(width / (xmax - xmin), height / (ymax - ymin))
            centre = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2])
            vertices, faces = [], []
            for tri in constrained_delaunay_triangles(shape).geoms:
                points = (np.asarray(tri.exterior.coords)[:3] - centre) * scale
                a, b = points[1] - points[0], points[2] - points[0]
                if a[0] * b[1] - a[1] * b[0] < 0:
                    points = points[::-1]
                start = len(vertices)
                vertices.extend(np.column_stack([points, np.zeros(3)]))
                vertices.append([*points.mean(axis=0), 0.])
                faces.extend([[start + i, start + (i + 1) % 3, start + 3] for i in range(3)])
            name = f"incubation_text_{len(meshes)}"
            filename = f"label-{len(meshes):02d}.obj"
            (PANEL / filename).write_text(
                "# Generated vector lettering by tasks/start_incubation/build_panel.py\n"
                + "".join("v " + numbers(v) + "\n" for v in vertices)
                + "".join("f " + " ".join(str(int(i) + 1) for i in face) + "\n" for face in faces))
            ET.SubElement(assets, "mesh", name=name, inertia="shell",
                          file=f"../tasks/autobio/start_incubation/panel/{filename}")
            meshes[key] = name
        ET.SubElement(parent, "geom", name=f"label_{len(list(controls.iter('geom')))}",
                      type="mesh", mesh=meshes[key], material="incubation_" + color,
                      pos=numbers(pos), quat=numbers(quat), contype="0", conaffinity="0",
                      mass="0", group="2")

    # Cover only the printed control panel; retain the original machine housing,
    # branding, thermoblock and wells. Local X is reading-right, Y reading-up.
    origin = sites["button_prog_3"].copy()
    centre = origin - .032 * UP - .013 * RIGHT + .0007 * NORMAL
    ET.SubElement(controls, "geom", name="incubation_panel", type="box",
                  size=".096 .042 .00025", pos=numbers(centre), quat=numbers(QUAT),
                  material="incubation_panel", contype="0", conaffinity="0", mass="0", group="2")

    # A clean, static instrument readout, matching the original display values.
    display = sites["button_prog_3"] - .027 * UP + .0008 * NORMAL
    ET.SubElement(controls, "geom", name="incubation_display", type="box",
                  size=".049 .0115 .0003", pos=numbers(display), quat=numbers(QUAT),
                  material="incubation_white", contype="0", conaffinity="0", mass="0", group="2")
    for x, title, value in [(-.031, "TIME", "2:30"), (0., "TEMP °C", "45 / 24"), (.031, "RPM", "0")]:
        label(controls, title, display + x * RIGHT + .0065 * UP + .00035 * NORMAL, .027, .0033)
        label(controls, value, display + x * RIGHT - .0025 * UP + .00035 * NORMAL, .027, .006)

    # Keep the original two joints first, so their identifiers remain familiar.
    names = ["button_prog_1", "button_menu_enter"] + [n for n in sites if n not in {"button_prog_1", "button_menu_enter"}]
    arrows = {"up": "▲", "down": "▼", "left": "◀", "right": "▶"}
    for name in names:
        program = name.startswith("button_prog_")
        pos = sites[name] + .003 * NORMAL
        body = ET.SubElement(controls, "body", name=name + "_key", pos=numbers(pos), quat=numbers(QUAT))
        ET.SubElement(body, "joint", name=name + "_key", type="slide", axis="0 0 1",
                      range="-.006 0", stiffness="50", springref="0", damping="1", armature=".001")
        kind, size = ("box", ".010 .005 .003") if program else ("cylinder", ".0058 .003")
        ET.SubElement(body, "geom", name=name + "_cap", type=kind, size=size,
                      material="incubation_white" if program else "incubation_blue",
                      contype="0", conaffinity="0", mass="0", group="2")
        ET.SubElement(body, "geom", name=name + "_contact", type=kind, size=size,
                      density="1000", group="3", contype="0", conaffinity="1", condim="4",
                      friction="1 .002 .0001", solimp=".998 .998", solref=".004 1")
        if program:
            label(body, "Program " + name[-1], [0, 0, .00305], .0185, .0037, quat=[1, 0, 0, 0])
        elif name == "button_menu_enter":
            label(body, "MENU", [0, .002, .00305], .0085, .002, "white", [1, 0, 0, 0])
            label(body, "ENTER", [0, -.0015, .00305], .009, .002, "white", [1, 0, 0, 0])
        elif name == "button_start_stop":
            label(body, "START", [0, .002, .00305], .009, .0022, "white", [1, 0, 0, 0])
            label(body, "STOP", [0, -.0015, .00305], .0085, .0022, "white", [1, 0, 0, 0])
        elif name == "button_short":
            label(body, "SHORT", [0, 0, .00305], .0095, .0023, "white", [1, 0, 0, 0])
        else:
            label(body, arrows[name.rsplit('_', 1)[1]], [0, 0, .00305], .0055, .0055,
                  "white", [1, 0, 0, 0])
    # Embed declarations so scene hashing and frozen replay packs need no
    # special handling for XML includes. Referenced OBJ files ship on HF.
    scene_path = HERE / "scene.xml"
    source = embed_xml(scene_path.read_text(), "ASSETS", assets, "    ")
    scene_path.write_text(embed_xml(source, "CONTROLS", controls, "      "))


if __name__ == "__main__":
    build()
