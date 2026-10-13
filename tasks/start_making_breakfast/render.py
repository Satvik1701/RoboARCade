"""Blender looks for the names this scene coins; core/render/materials.py has the shared ones."""
LOOKS = {
    "dial":      (dict(metallic=0.8, roughness=0.42, ior=1.45, base=(0.055, 0.055, 0.062, 1.0)), None),
    "cast_iron": (dict(metallic=1.0, roughness=0.28, ior=1.45), "cast_iron"),
    "food":      (dict(metallic=0.0, roughness=0.45, coat=0.10, ior=1.45), "plaster"),
}
WORDS = {
    "dial":      ("dial_ring",),
    "cast_iron": ("frypan", "burner_plate"),
    "metal":     ("pot_body", "moka", "knob_metal", "fridge", "burner", "table_legs"),
    "wood":      ("table_texture",),
    "food":      ("bread",),
}
