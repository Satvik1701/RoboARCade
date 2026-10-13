"""Persistent Blender render worker.

    blender -b -P worker.py -- <scene.mjb> <res>

Owns the MuJoCo model: exports it to USD with mujoco's own exporter (Blender's bundled
pxr writes the .usda, so mujoco lives in tools/blender-py, provisioned once per node by
Blender's own pip with --no-deps:

    tools/blender-4.5.12-linux-x64/4.5/python/bin/python3.11 -m pip install \
        --target tools/blender-py --no-deps mujoco==3.10.0 pillow typing_extensions \
        etils absl-py glfw

), imports it once, then serves one JSON request per stdin line, replying one
MARK-prefixed JSON line per request on stdout:

    {"geoms": {"<gid>": [px,py,pz, r00..r22]},
     "geom_rgba": {"<changed-gid>": [r,g,b,a]},
     "cams": [{"name","pos","mat","fovy","clip"}]}
    -> {"ok": true, "files": {"<cam>": "/path.png"}, "ms": 137}
"""
import json
import math
import os
import re
import runpy
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
if os.environ.get("ROBOARCADE_RENDER_OWNER_PID"):
    from lifecycle import follow_parent
    follow_parent(int(os.environ["ROBOARCADE_RENDER_OWNER_PID"]))

import bmesh
import bpy
import numpy as np
from mathutils import Matrix

sys.path.insert(0, os.environ.get("ROBOARCADE_BLENDER_PY", os.path.join(
    os.path.dirname(os.path.dirname(HERE)), "tools", "blender-py")))
import mujoco                     # noqa: E402
from mujoco.usd import exporter   # noqa: E402
import materials                  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:]
MJB, RES, TASK = argv[0], int(argv[1]), argv[2]
SAMPLES = 32
AUTOEXPOSE_TARGET = 0.60
MARK = "@@ROBOARCADE@@"
OUTDIR = os.environ.get("ROBOARCADE_BLENDER_OUT", os.path.dirname(MJB))

model = mujoco.MjModel.from_binary_path(MJB)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
render = os.path.join(os.path.dirname(os.path.dirname(HERE)), "tasks", TASK, "render.py")
own = runpy.run_path(render) if os.path.isfile(render) else {}
LOOKS = {**materials.LOOKS, **own.get("LOOKS", {})}
WORDS = [*own.get("WORDS", {}).items(), *materials.WORDS.items()]
ROBOT_PREFIXES = tuple(dict.fromkeys(
    name.removesuffix("_grip_site")
    for i in range(model.nsite)
    if (name := model.site(i).name) and name.endswith("_grip_site")))

keep = np.isin(model.geom_group, (1, 2))
model.geom_group[:] = np.where(keep, 1, 4)
opt = mujoco.MjvOption()
opt.geomgroup[:] = 0
opt.geomgroup[1] = 1
opt.flexgroup[:] = 0
exp = exporter.USDExporter(model=model, output_directory=os.path.join(OUTDIR, "usd"),
                           output_directory_root="/", verbose=False)
exp.update_scene(data, scene_option=opt)
exp.save_scene(filetype="usda")
USD = os.path.join(OUTDIR, "usd", "frames", "frame_1.usda")

bpy.ops.wm.read_homefile(use_empty=True, use_factory_startup=True)
scene = bpy.context.scene

prefs = bpy.context.preferences.addons["cycles"].preferences
backend, gpu = "NONE", []
for kind in ("OPTIX", "CUDA"):
    prefs.compute_device_type = kind
    prefs.refresh_devices()
    found = [d for d in prefs.devices if d.type == kind]
    if found:
        backend, gpu = kind, found
        break
for d in prefs.devices:
    d.use = d.type == backend

scene.render.engine = "CYCLES"
scene.cycles.device = "GPU" if gpu else "CPU"
scene.cycles.samples = SAMPLES
scene.cycles.use_adaptive_sampling = True
scene.cycles.use_denoising = True
scene.cycles.denoiser = "OPENIMAGEDENOISE"
scene.cycles.denoising_use_gpu = bool(gpu and prefs.has_oidn_gpu_devices())
scene.cycles.max_bounces = 6
scene.cycles.caustics_reflective = scene.cycles.caustics_refractive = False
scene.render.use_persistent_data = True
scene.render.resolution_x = scene.render.resolution_y = RES
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.view_settings.view_transform = "Standard"

bpy.ops.wm.usd_import(filepath=USD, import_materials=True, import_cameras=False,
                      import_lights=False, scale=1.0,
                      apply_unit_conversion_scale=False,
                      create_world_material=False)


def flex_material(fid, name):
    rgba = [float(x) for x in model.flex_rgba[fid]]
    thin_film = int(model.flex_dim[fid]) == 2 and rgba[3] < 0.75
    mat = bpy.data.materials.new(f"{name}_material")
    mat.diffuse_color = rgba
    mat.use_nodes = True
    bsdf = next(n for n in mat.node_tree.nodes
                if n.bl_idname == "ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Roughness"].default_value = 0.18 if thin_film else 0.88
    bsdf.inputs["Alpha"].default_value = rgba[3]
    if thin_film:
        bsdf.inputs["Transmission Weight"].default_value = 0.45
        bsdf.inputs["IOR"].default_value = 1.46
    bsdf.inputs["Sheen Weight"].default_value = 0.0 if thin_film else 0.28
    bsdf.inputs["Sheen Roughness"].default_value = 0.75
    if int(model.flex_dim[fid]) == 2 and not thin_film:
        nodes, links = mat.node_tree.nodes, mat.node_tree.links
        geometry = nodes.new("ShaderNodeNewGeometry")
        pile_scale_3mm = nodes.new("ShaderNodeVectorMath")
        pile_scale_3mm.operation = "SCALE"
        pile_scale_3mm.inputs[3].default_value = 1.0 / 0.003
        noise = nodes.new("ShaderNodeTexNoise")
        noise.noise_dimensions = "3D"
        noise.inputs["Scale"].default_value = 1.0
        noise.inputs["Detail"].default_value = 5.0
        noise.inputs["Roughness"].default_value = 0.72
        noise.inputs["Distortion"].default_value = 0.12
        bump = nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = 0.38
        bump.inputs["Distance"].default_value = 0.00025
        links.new(geometry.outputs["Position"], pile_scale_3mm.inputs[0])
        links.new(pile_scale_3mm.outputs["Vector"], noise.inputs["Vector"])
        links.new(noise.outputs["Fac"], bump.inputs["Height"])
        links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


flex_objects = {}
for fid in range(model.nflex):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_FLEX, fid) or f"flex{fid}"
    va, vn = int(model.flex_vertadr[fid]), int(model.flex_vertnum[fid])
    vertices = np.asarray(data.flexvert_xpos[va:va + vn], float)
    mat = flex_material(fid, name)
    dim = int(model.flex_dim[fid])
    if dim == 1:
        curve = bpy.data.curves.new(f"{name}_rod", type="CURVE")
        curve.dimensions = "3D"
        curve.resolution_u = 1
        curve.bevel_depth = float(model.flex_radius[fid])
        curve.bevel_resolution = 3
        curve.fill_mode = "FULL"
        curve.use_fill_caps = True
        spline = curve.splines.new(type="POLY")
        spline.points.add(len(vertices) - 1)
        spline.points.foreach_set(
            "co", np.column_stack((vertices, np.ones(len(vertices)))).ravel())
        curve.materials.append(mat)
        obj = bpy.data.objects.new(f"{name}_flex", curve)
        scene.collection.objects.link(obj)
        flex_objects[name] = curve
        continue
    if dim == 2:
        ea, en = int(model.flex_elemdataadr[fid]), int(model.flex_elemnum[fid])
        faces = np.asarray(model.flex_elem, int)[ea:ea + 3 * en].reshape(-1, 3)
    else:
        sa, sn = int(model.flex_shelldataadr[fid]), int(model.flex_shellnum[fid])
        faces = np.asarray(model.flex_shell, int)[sa:sa + 3 * sn].reshape(-1, 3)
    mesh = bpy.data.meshes.new(f"{name}_flex")
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.materials.append(mat)
    mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
    mesh.update()
    obj = bpy.data.objects.new(f"{name}_flex", mesh)
    scene.collection.objects.link(obj)
    if dim == 2:
        solidify = obj.modifiers.new("cloth thickness", "SOLIDIFY")
        solidify.thickness = 2.0 * float(model.flex_radius[fid])
        solidify.offset = 0.0
    flex_objects[name] = obj.data


def drop_duplicate_triangles(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.normal_update()
    seen, duplicates = set(), []
    for face in bm.faces:
        if len(face.verts) != 3:
            continue
        points = tuple(sorted(tuple(round(float(x), 9) for x in v.co)
                              for v in face.verts))
        normal = tuple(round(float(x), 9) for x in face.normal)
        key = (face.material_index, normal, points)
        if key in seen:
            duplicates.append(face)
        else:
            seen.add(key)
    if duplicates:
        bmesh.ops.delete(bm, geom=duplicates, context="FACES")
        bm.to_mesh(obj.data)
        obj.data.update()
    bm.free()
    return len(duplicates)


for o in bpy.data.objects:
    if o.type == "MESH" and (removed := drop_duplicate_triangles(o)):
        print(f"[render] removed {removed} duplicate triangles from {o.name}",
              file=sys.stderr, flush=True)

for m in bpy.data.materials:
    if m.node_tree is None:
        continue
    for n in m.node_tree.nodes:
        if n.bl_idname == "ShaderNodeTexImage" and n.image:
            n.image.colorspace_settings.name = "sRGB"
            n.interpolation = "Cubic"

by_gid = {}
for o in bpy.data.objects:
    if o.type != "MESH":
        continue
    mm = re.search(r"_id(\d+)_geom$", o.name)
    gid = int(mm.group(1)) if mm else -1
    if mm:
        by_gid[gid] = o
    mid = int(model.geom_matid[gid]) if gid >= 0 else -1
    mat_name = model.mat(mid).name if mid >= 0 else ""
    reflectance = float(model.mat_reflectance[mid]) if mid >= 0 else 0.0
    ident = " ".join(filter(None, [model.geom(gid).name,
                                   model.body(int(model.geom_bodyid[gid])).name])) \
        if gid >= 0 else o.name
    for mat in o.data.materials:
        if not mat:
            continue
        if mid >= 0:
            mat["_texrepeat"] = [float(v) for v in model.mat_texrepeat[mid]]
            mat["_texuniform"] = bool(model.mat_texuniform[mid])
        if mat.get("_pbr"):
            continue
        bsdf = next((n for n in (mat.node_tree.nodes if mat.node_tree else [])
                     if n.bl_idname == "ShaderNodeBsdfPrincipled"), None)
        if bsdf is None:
            continue
        mat["_pbr"] = materials.apply(
            mat.node_tree, bsdf, ident, mat_name, reflectance, ROBOT_PREFIXES, LOOKS, WORDS)

for o in bpy.data.objects:
    if o.type != "MESH" or len(o.data.vertices) > 12:
        continue
    for m in o.data.materials:
        if not m or m.node_tree is None or m.get("_retiled"):
            continue
        tex = next((n for n in m.node_tree.nodes
                    if n.bl_idname == "ShaderNodeTexImage" and n.image), None)
        if tex is None:
            continue
        repeat = m.get("_texrepeat", (1.0, 1.0))
        uniform = bool(m.get("_texuniform", False))
        u, v = float(repeat[0]), float(repeat[1])
        coord = m.node_tree.nodes.new("ShaderNodeTexCoord")
        mp = m.node_tree.nodes.new("ShaderNodeMapping")
        mp.inputs["Scale"].default_value = (u, v, math.sqrt(u * v))
        source = coord.outputs["Object" if uniform else "Generated"]
        m.node_tree.links.new(source, mp.inputs["Vector"])
        m.node_tree.links.new(mp.outputs["Vector"], tex.inputs["Vector"])
        tex.projection, tex.projection_blend = "BOX", 0.2
        m["_retiled"] = True

for o in bpy.data.objects:
    if o.type == "MESH":
        smooth = len(o.data.polygons) >= 200
        o.data.polygons.foreach_set("use_smooth", [smooth] * len(o.data.polygons))
        o.data.update()

world = bpy.data.worlds.new("W")
scene.world = world
if world.node_tree is None:
    world.use_nodes = True
nt = world.node_tree
nt.nodes.clear()
bg = nt.nodes.new("ShaderNodeBackground")
bg.inputs["Color"].default_value = (0.62, 0.66, 0.72, 1.0)
bg.inputs["Strength"].default_value = 0.70
outw = nt.nodes.new("ShaderNodeOutputWorld")
nt.links.new(bg.outputs["Background"], outw.inputs["Surface"])
for name, loc, energy, size, tilt in (
        ("key", (0.30, -0.45, 2.60), 85.0, 3.2, 12),
        ("fill", (0.10, 0.75, 2.10), 38.0, 3.0, -10)):
    lt = bpy.data.lights.new(name, type="AREA")
    lt.energy, lt.size, lt.shape = energy, size, "SQUARE"
    ob = bpy.data.objects.new(name, lt)
    scene.collection.objects.link(ob)
    ob.location = loc
    ob.rotation_euler = (math.radians(tilt), 0, 0)

cam_data = bpy.data.cameras.new("cam")
cam_data.sensor_fit = "VERTICAL"
cam_obj = bpy.data.objects.new("cam", cam_data)
scene.collection.objects.link(cam_obj)
scene.camera = cam_obj


def set_cam(spec):
    R, p = spec["mat"], spec["pos"]
    cam_data.angle_y = math.radians(spec["fovy"])
    cam_data.clip_start, cam_data.clip_end = spec["clip"]
    cam_obj.matrix_world = Matrix(((R[0], R[1], R[2], p[0]),
                                   (R[3], R[4], R[5], p[1]),
                                   (R[6], R[7], R[8], p[2]), (0, 0, 0, 1)))


def autoexpose(spec):
    set_cam(spec)
    keep = (scene.render.resolution_x, scene.render.image_settings.file_format,
            scene.render.filepath)
    probe = os.path.join(OUTDIR, "_probe.exr")
    scene.render.resolution_x = scene.render.resolution_y = 64
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.view_settings.exposure = 0.0
    scene.render.filepath = probe
    bpy.ops.render.render(write_still=True)
    im = bpy.data.images.load(probe)
    px = np.array(im.pixels[:]).reshape(-1, 4)
    lum = 0.2126 * px[:, 0] + 0.7152 * px[:, 1] + 0.0722 * px[:, 2]
    ev = float(np.clip(math.log2(AUTOEXPOSE_TARGET / max(
        float(np.percentile(lum, 99)), 1e-4)), -3, 2))
    bpy.data.images.remove(im)
    os.remove(probe)
    scene.render.resolution_x = scene.render.resolution_y = keep[0]
    scene.render.image_settings.file_format = keep[1]
    scene.render.filepath = keep[2]
    return ev


dynamic_materials = {}


def _set_rgba(mat, rgba):
    rgba = tuple(float(v) for v in rgba)
    mat.diffuse_color = rgba
    if mat.node_tree is None:
        return
    for node in mat.node_tree.nodes:
        if node.bl_idname != "ShaderNodeBsdfPrincipled":
            continue
        node.inputs["Base Color"].default_value = rgba
        node.inputs["Alpha"].default_value = rgba[3]


def sync_geom_rgba(raw_changes):
    changes = {int(gid): rgba for gid, rgba in raw_changes.items()}
    for gid in set(dynamic_materials) - set(changes):
        o, slots = dynamic_materials.pop(gid)
        owned = {}
        for index, original_link, original, copy in slots:
            slot = o.material_slots[index]
            slot.link = original_link
            if original_link == "OBJECT":
                slot.material = original
            owned[copy.as_pointer()] = copy
        for mat in owned.values():
            if mat.users == 0:
                bpy.data.materials.remove(mat)

    for gid, rgba in changes.items():
        o = by_gid.get(gid)
        if o is None:
            continue
        if gid not in dynamic_materials:
            copies, slots = {}, []
            for index, slot in enumerate(o.material_slots):
                original = slot.material
                if original is None:
                    continue
                owned = copies.get(original.as_pointer())
                if owned is None:
                    owned = original.copy()
                    owned.name = f"{original.name}_geom_{gid}"
                    copies[original.as_pointer()] = owned
                original_link = slot.link
                slot.link = "OBJECT"
                slot.material = owned
                slots.append((index, original_link, original, owned))
            dynamic_materials[gid] = (o, slots)
        for _, _, _, mat in dynamic_materials[gid][1]:
            _set_rgba(mat, rgba)


def pose(req):
    for gid, m in req["geoms"].items():
        o = by_gid.get(int(gid))
        if o is None:
            continue
        o.matrix_world = Matrix(((m[3], m[4], m[5], m[0]),
                                 (m[6], m[7], m[8], m[1]),
                                 (m[9], m[10], m[11], m[2]), (0, 0, 0, 1)))
    for name, raw in req.get("flexes", {}).items():
        if name not in flex_objects:
            continue
        vertices = np.asarray(raw, dtype=float).reshape(-1, 3)
        data_block = flex_objects[name]
        if isinstance(data_block, bpy.types.Curve):
            points = data_block.splines[0].points
            points.foreach_set(
                "co", np.column_stack((vertices, np.ones(len(vertices)))).ravel())
            data_block.update_tag()
        else:
            data_block.vertices.foreach_set("co", vertices.ravel())
            data_block.update()
    bpy.context.view_layer.update()


def reply(obj):
    print(MARK + json.dumps(obj), flush=True)


def seed_color_geoms():
    supported = []
    for gid, obj in by_gid.items():
        rgba = model.geom_rgba[gid]
        if model.geom_matid[gid] >= 0 or rgba[3] != 1 or not obj.material_slots:
            continue
        valid = True
        for slot in obj.material_slots:
            mat = slot.material
            nodes = mat.node_tree.nodes if mat and mat.node_tree else []
            bsdfs = [n for n in nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"]
            if len(bsdfs) != 1:
                valid = False
                break
            bsdf = bsdfs[0]
            if (bsdf.inputs["Base Color"].is_linked or bsdf.inputs["Alpha"].is_linked or
                    not np.array_equal(bsdf.inputs["Base Color"].default_value,
                                       rgba.astype(np.float16).astype(np.float32)) or
                    bsdf.inputs["Alpha"].default_value != 1):
                valid = False
                break
        if valid:
            supported.append(gid)
    return sorted(supported)


exposures = {}
reply({"ready": True, "objects": len(by_gid) + len(flex_objects),
       "device": backend if gpu else "CPU", "seed_color_geoms": seed_color_geoms()})

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
        if req.get("quit"):
            break
        exposure = exposures.setdefault(str(req.get("seed", "dedicated")), {})
        t0 = time.perf_counter()
        sync_geom_rgba(req.get("geom_rgba", {}))
        pose(req)
        files = {}
        for spec in req["cams"]:
            if spec["name"] not in exposure:
                exposure[spec["name"]] = autoexpose(spec)
            scene.view_settings.exposure = exposure[spec["name"]]
            set_cam(spec)
            path = os.path.join(OUTDIR, f"{spec['name']}.png")
            scene.render.filepath = path
            bpy.ops.render.render(write_still=True)
            files[spec["name"]] = path
        reply({"ok": True, "files": files,
               "ms": round((time.perf_counter() - t0) * 1000),
               "exposure": dict(exposure)})
    except Exception as e:
        reply({"ok": False, "error": f"{type(e).__name__}: {e}"})
