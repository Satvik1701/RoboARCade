"""Subtasks of jenga tower.

Layers 2 to 4 laid block by block, each layer three blocks flat across the layer below, alternating
direction, released and supported (layer 1 stands at the start)."""
from core.grade_episode import Subtask
from core.grade_episode import _body_pos, _contact_body_pairs, _subtrees_touching

BLOCKS = [f"jenga_{i}" for i in range(1, 19)]
X, Y = [1, 0, 0], [0, 1, 0]
TABLE = 'play_table'


def layers(s):
    layer_bands_z_m = next(n["bands"] for n in s.env.spec["success"]["of"] if n.get("pred") == "tower_support")
    best = (0, 0)
    for first, second in ((X, Y), (Y, X)):
        below, complete, partial = [], 0, 0
        for k, (lo, hi) in enumerate(layer_bands_z_m):
            along = first if k % 2 == 0 else second
            supported = [b for b in BLOCKS
                         if s.lift_above(obj=b, z_above=lo) and s.z_below(obj=b, z_below=hi)
                         and s.axis_dot(obj=b, axis=[0, 0, 1], world_axis=along, abs=True, min_dot=0.9)
                         and s.axis_dot(obj=b, axis=[0, 0, 1], world_axis=[0, 0, 1], abs=True, min_dot=0.0, max_dot=0.15)
                         and s.env.robot_clear([b])
                         and (_subtrees_touching(s.env, b, TABLE) if k == 0
                              else sum(bool(_subtrees_touching(s.env, b, c)) for c in below) >= 2)]
            if len(supported) < 3:
                partial = len(supported)
                break
            complete += 1
            below = supported
        best = max(best, (complete, partial))
    return best


def tower_support(bodies, table, bands):
    """every block in one height layer, resting on {table} or on two distinct blocks of the layer below"""
    bodies, bands = tuple(bodies), tuple(tuple(band) for band in bands)

    def check(env):
        layers = [[] for _ in bands]
        for body in bodies:
            matches = [i for i, (lo, hi) in enumerate(bands)
                       if lo < float(_body_pos(env, body)[2]) < hi]
            if len(matches) != 1:
                return False
            layers[matches[0]].append(body)
        needed = {frozenset((body, table)) for body in layers[0]}
        needed |= {frozenset((body, below)) for i in range(1, len(layers))
                   for body in layers[i] for below in layers[i - 1]}
        m = env.model
        owners = {m.body(name).id: name for name in (*bodies, table)}
        def owner(bid):
            while bid > 0 and bid not in owners:
                bid = int(m.body_parentid[bid])
            return owners.get(bid)
        edges = {frozenset((a, b)) for b1, b2 in _contact_body_pairs(env)
                 if (a := owner(b1)) and (b := owner(b2)) and a != b}
        edges |= {pair for pair in needed - edges
                  if _subtrees_touching(env, *tuple(pair))}
        return (all(frozenset((body, table)) in edges for body in layers[0])
                and all(sum(frozenset((body, below)) in edges for below in layers[i - 1]) >= 2
                        for i in range(1, len(layers)) for body in layers[i]))
    return check


def subtasks():
    def laid(layer, n):
        return lambda s: (lambda c, p: c >= layer or c == layer - 1 and p >= n)(*layers(s))
    return [Subtask(f'{n} block{"s" * (n > 1)} of layer {layer} laid flat across layer {layer - 1}',
                    (lambda s: s.success()) if (layer, n) == (4, 3) else laid(layer, n), objects=tuple(BLOCKS))
            for layer in (2, 3, 4) for n in (1, 2, 3)]
