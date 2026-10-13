"""Blender materials per MuJoCo geom.

One material table for the Blender render: a class per geom from its MuJoCo
names (bowl, knob, table_top, wood...) with reflectance as the
fallback for the unnamed, the Principled BSDF inputs it gets (Blender 4.x socket
spellings), and the procedural surface built on top of it.
"""
import re


def _coords(nt, scale, stretch=(1.0, 1.0, 1.0)):
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (scale * stretch[0], scale * stretch[1],
                                        scale * stretch[2])
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    return mp.outputs["Vector"]


def _noise(nt, vec, scale, detail=6.0, rough=0.5, distortion=0.0):
    n = nt.nodes.new("ShaderNodeTexNoise")
    n.inputs["Scale"].default_value = scale
    n.inputs["Detail"].default_value = detail
    n.inputs["Roughness"].default_value = rough
    n.inputs["Distortion"].default_value = distortion
    nt.links.new(vec, n.inputs["Vector"])
    return n.outputs["Fac"]


def _ramp(nt, fac, grey_stops):
    r = nt.nodes.new("ShaderNodeValToRGB")
    el = r.color_ramp.elements
    while len(el) > 1:
        el.remove(el[-1])
    el[0].position, el[0].color = grey_stops[0][0], (grey_stops[0][1],) * 3 + (1.0,)
    for pos, val in grey_stops[1:]:
        e = el.new(pos)
        e.color = (val,) * 3 + (1.0,)
    nt.links.new(fac, r.inputs["Fac"])
    return r.outputs["Color"]


def _bump(nt, height, strength, distance, normal_in=None):
    b = nt.nodes.new("ShaderNodeBump")
    b.inputs["Strength"].default_value = strength
    b.inputs["Distance"].default_value = distance
    nt.links.new(height, b.inputs["Height"])
    if normal_in is not None:
        nt.links.new(normal_in, b.inputs["Normal"])
    return b.outputs["Normal"]


def _tint(nt, bsdf, fac, lo, hi):
    base = bsdf.inputs["Base Color"]
    src = base.links[0].from_socket if base.is_linked else None
    m = nt.nodes.new("ShaderNodeMix")
    m.data_type = "RGBA"
    m.blend_type = "MULTIPLY"
    m.inputs["Factor"].default_value = 1.0
    shade = _ramp(nt, fac, [(0.0, lo), (1.0, hi)])
    if src is not None:
        nt.links.new(src, m.inputs[6])
    else:
        m.inputs[6].default_value = base.default_value
    nt.links.new(shade, m.inputs[7])
    nt.links.new(m.outputs[2], base)


def wood(nt, bsdf):
    v = _coords(nt, 1.0, stretch=(1.0, 0.06, 1.0))
    w = nt.nodes.new("ShaderNodeTexWave")
    w.wave_type = "BANDS"
    w.bands_direction = "Y"
    w.inputs["Scale"].default_value = 2.2
    w.inputs["Distortion"].default_value = 9.0
    w.inputs["Detail"].default_value = 5.0
    w.inputs["Detail Scale"].default_value = 1.6
    nt.links.new(v, w.inputs["Vector"])
    grain = w.outputs["Fac"]
    pores = _noise(nt, _coords(nt, 1.0, (1.0, 0.15, 1.0)), 90.0, detail=4.0)
    n1 = _bump(nt, grain, 0.22, 0.0012)
    n2 = _bump(nt, pores, 0.10, 0.0004, normal_in=n1)
    nt.links.new(n2, bsdf.inputs["Normal"])
    nt.links.new(_ramp(nt, grain, [(0.0, 0.30), (0.55, 0.52), (1.0, 0.60)]),
                 bsdf.inputs["Roughness"])
    _tint(nt, bsdf, grain, 0.80, 1.06)


def plaster(nt, bsdf):
    v = _coords(nt, 1.0)
    tooth = _noise(nt, v, 220.0, detail=3.0, rough=0.7)
    broad = _noise(nt, v, 9.0, detail=3.0)
    n1 = _bump(nt, tooth, 0.28, 0.0004)
    nt.links.new(_bump(nt, broad, 0.10, 0.0012, normal_in=n1), bsdf.inputs["Normal"])
    nt.links.new(_ramp(nt, broad, [(0.0, 0.82), (1.0, 0.94)]), bsdf.inputs["Roughness"])


def brushed(nt, bsdf):
    v = _coords(nt, 1.0, stretch=(1.0, 140.0, 1.0))
    lay = _noise(nt, v, 30.0, detail=2.0, rough=0.4)
    nt.links.new(_ramp(nt, lay, [(0.0, 0.16), (1.0, 0.38)]), bsdf.inputs["Roughness"])
    nt.links.new(_bump(nt, lay, 0.06, 0.0002), bsdf.inputs["Normal"])
    bsdf.inputs["Anisotropic"].default_value = 0.55


def cast_iron(nt, bsdf):
    v = _coords(nt, 1.0)
    pit = nt.nodes.new("ShaderNodeTexVoronoi")
    pit.inputs["Scale"].default_value = 160.0
    nt.links.new(v, pit.inputs["Vector"])
    season = _noise(nt, v, 26.0, detail=6.0)
    n1 = _bump(nt, pit.outputs["Distance"], 0.14, 0.0004)
    nt.links.new(_bump(nt, season, 0.10, 0.0008, normal_in=n1), bsdf.inputs["Normal"])
    nt.links.new(_ramp(nt, season, [(0.0, 0.28), (1.0, 0.52)]), bsdf.inputs["Roughness"])
    _tint(nt, bsdf, season, 0.85, 1.08)


def glaze(nt, bsdf):
    v = _coords(nt, 1.0)
    peel = _noise(nt, v, 55.0, detail=3.0)
    nt.links.new(_bump(nt, peel, 0.05, 0.0002), bsdf.inputs["Normal"])
    nt.links.new(_ramp(nt, peel, [(0.0, 0.08), (1.0, 0.16)]), bsdf.inputs["Roughness"])
    bsdf.inputs["Coat Weight"].default_value = 0.45
    bsdf.inputs["Coat Roughness"].default_value = 0.06


def molded(nt, bsdf):
    v = _coords(nt, 1.0)
    tool = _noise(nt, v, 180.0, detail=3.0, rough=0.55)
    nt.links.new(_bump(nt, tool, 0.10, 0.0002), bsdf.inputs["Normal"])
    nt.links.new(_ramp(nt, tool, [(0.0, 0.40), (1.0, 0.54)]), bsdf.inputs["Roughness"])
    bsdf.inputs["Coat Weight"].default_value = 0.0


def robot_shell(nt, bsdf):
    v = _coords(nt, 1.0)
    grain = _noise(nt, v, 320.0, detail=2.0, rough=0.5)
    wide = _noise(nt, v, 22.0, detail=3.0)
    n1 = _bump(nt, grain, 0.06, 0.0002)
    nt.links.new(_bump(nt, wide, 0.03, 0.0006, normal_in=n1), bsdf.inputs["Normal"])
    nt.links.new(_ramp(nt, wide, [(0.0, 0.28), (1.0, 0.40)]), bsdf.inputs["Roughness"])
    bsdf.inputs["Coat Weight"].default_value = 0.18
    bsdf.inputs["Coat Roughness"].default_value = 0.25


SURFACES = {f.__name__: f for f in (wood, plaster, brushed, cast_iron, glaze, molded, robot_shell)}
LOOKS = {
    "robot":     (dict(metallic=0.0, roughness=0.34, ior=1.47), "robot_shell"),
    "molded":    (dict(metallic=0.0, roughness=0.46, ior=1.46), "molded"),
    "liquid":    (dict(metallic=0.0, roughness=0.30, ior=1.33), None),
    "glass":     (dict(metallic=0.0, roughness=0.05, transmission=1.0, ior=1.45), None),
    "metal":     (dict(metallic=1.0, roughness=0.28, ior=1.45), "brushed"),
    "ceramic":   (dict(metallic=0.0, roughness=0.12, ior=1.5), "glaze"),
    "wood":      (dict(metallic=0.0, roughness=0.55, coat=0.05, ior=1.45), "wood"),
    "matte":     (dict(metallic=0.0, roughness=0.88, ior=1.45), "plaster"),
    "rubber":    (dict(metallic=0.0, roughness=0.78, ior=1.45), "plaster"),
    "plastic":   (dict(metallic=0.0, roughness=0.38, ior=1.45), "glaze"),
}
WORDS = {
    "molded":    ("table_top",),
    "liquid":    ("liquid",),
    "glass":     ("bottle",),
    "metal":     ("metal",),
    "ceramic":   ("bowl", "plate", "mug", "cup", "ceramic"),
    "wood":      ("wood",),
    "matte":     ("wall", "floor"),
    "rubber":    ("handle", "grip", "pad", "knob"),
}


def classify(geom_name="", mat_name="", reflectance=0.0, robot_prefixes=(), words=WORDS.items()):
    hay = f"{geom_name} {mat_name}".lower()
    if any(re.search(rf"(?:^|\s){re.escape(prefix.lower())}(?:_|/|$)", hay)
           for prefix in robot_prefixes):
        return "robot"
    for look, names in words:
        if any(re.search(rf"(?:^|[^a-z]){re.escape(w)}(?:[^a-z]|$)", hay) for w in names):
            return look
    return "metal" if reflectance >= 0.6 else "plastic"


def apply(nt, bsdf, geom_name, mat_name, reflectance, robot_prefixes=(), looks=LOOKS, words=WORDS.items()):
    look = classify(geom_name, mat_name, reflectance, robot_prefixes, words)
    p, surface = looks[look]
    i = bsdf.inputs
    i["Metallic"].default_value = p["metallic"]
    i["Roughness"].default_value = p["roughness"]
    i["IOR"].default_value = p["ior"]
    i["Coat Weight"].default_value = p.get("coat", 0.0)
    i["Sheen Weight"].default_value = 0.0
    if "transmission" in p:
        i["Transmission Weight"].default_value = p["transmission"]
    if "specular" in p:
        i["Specular IOR Level"].default_value = p["specular"]
    if "alpha" in p:
        i["Alpha"].default_value = p["alpha"]
    if p.get("base") and not i["Base Color"].is_linked:
        i["Base Color"].default_value = p["base"]
    if surface is not None:
        SURFACES[surface](nt, bsdf)
    return look
