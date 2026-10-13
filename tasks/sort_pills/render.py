"""Blender looks for the names this scene coins; core/render/materials.py has the shared ones."""
LOOKS = {
    "amber":        (dict(metallic=0.0, roughness=0.52, transmission=0.0, ior=1.47, specular=0.12, alpha=0.28), None),
    "pharmacy_cap": (dict(metallic=0.0, roughness=0.46, ior=1.47, specular=0.12), "molded"),
    "organizer":    (dict(metallic=0.0, roughness=0.48, ior=1.47, specular=0.15), "molded"),
    "pill":         (dict(metallic=0.0, roughness=0.38, ior=1.45), "glaze"),
}
WORDS = {
    "amber": ("capsule_bottle", "tablet_bottle"),
    "pharmacy_cap": ("capsule_lid", "tablet_lid"),
    "organizer": ("organizer",),
    "pill": ("capsule", "tablet"),
}
