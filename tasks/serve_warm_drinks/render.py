"""Blender looks for the names this scene coins; core/render/materials.py has the shared ones."""
LOOKS = {
    "smoked_glass": (dict(metallic=0.0, roughness=0.05, ior=1.5, specular=1.0, alpha=0.20, base=(0.4, 0.4, 0.4, 1.0)), None),
}
WORDS = {
    "smoked_glass": ("window",),
    "metal": ("fridge", "table_legs"),
    "wood":  ("table_texture",),
}
