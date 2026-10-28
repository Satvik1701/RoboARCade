"""Offline collision generation from the unchanged HF visual meshes.

Install coacd==1.0.14 and trimesh in a build
environment, then run `python -m tasks.bowl_on_plate.build_collisions`.
Generated convex vertices are embedded in scene.xml; runtime needs no builder.
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import trimesh

from core.tasks import asset_dir, load_scene

PACK = Path(__file__).resolve().parent
OBJECTS = [('akita_black_bowl_1', 'bowl'), ('plate_1', 'plate')]


def fmt(values):
    return ' '.join(f'{float(x):.10g}' for x in np.asarray(values).ravel())


def visual_surface(model, name):
    geom = model.geom(name + '_g0').id
    mesh = int(model.geom_dataid[geom])
    va, vn = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    fa, fn = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
    vertices = model.mesh_vert[va:va+vn] @ rotation.reshape(3, 3).T + model.geom_pos[geom]
    surface = trimesh.Trimesh(vertices, model.mesh_face[fa:fa+fn], process=True)
    surface.fix_normals()
    return surface


def build(parts_dir=None):
    model = load_scene(PACK).compile()
    path = PACK/'scene.xml'
    root = ET.parse(path).getroot()
    asset = root.find('asset')
    counts = {}
    for stem, short in OBJECTS:
        body = root.find(f'.//body[@name="{stem}_main"]')
        bid = model.body(stem+'_main').id
        # Plate inertia was inferred from the old boxes. Pin it explicitly so
        # geometry improvements do not inadvertently change its mass/dynamics.
        if body.find('inertial') is None:
            body.insert(0, ET.Element('inertial', pos=fmt(model.body_ipos[bid]),
                quat=fmt(model.body_iquat[bid]), mass=fmt([model.body_mass[bid]]),
                diaginertia=fmt(model.body_inertia[bid])))
        if parts_dir:
            files = sorted((Path(parts_dir)/('collision-'+stem+'_g0')).glob('*.ply'))
            if not files:
                raise ValueError(f'No generated collision parts for {stem}')
            parts = [trimesh.load(p, force='mesh') for p in files]
        else:
            import coacd
            surface = visual_surface(model, stem)
            coacd.set_log_level('warn')
            pieces = coacd.run_coacd(coacd.Mesh(surface.vertices, surface.faces),
                threshold=.001, real_metric=True,
                preprocess_mode='off' if surface.is_watertight else 'auto',
                preprocess_resolution=150,
                resolution=2000, mcts_iterations=60, mcts_nodes=15, seed=0)
            parts = [trimesh.Trimesh(v, f) for v, f in pieces]
        prefix = f'{short}_aligned_'
        for mesh in list(asset.findall('mesh')):
            if mesh.get('name', '').startswith(prefix):
                asset.remove(mesh)
        for geom in list(body.findall('geom')):
            if geom.get('name') != stem+'_g0':
                body.remove(geom)
        for i, part in enumerate(parts):
            hull = part.convex_hull
            center = hull.vertices.mean(axis=0)
            name = prefix+str(i)
            ET.SubElement(asset, 'mesh', name=name, vertex=fmt(hull.vertices-center))
            ET.SubElement(body, 'geom', name=name, type='mesh', mesh=name,
                pos=fmt(center), group='3', contype='1', conaffinity='1',
                friction='0.95 0.3 0.1', solref='0.001', solimp='0.998 0.998',
                density='0', rgba='0.8 0.8 0.8 0')
        counts[short] = len(parts)
    ET.indent(root, space='  ')
    ET.ElementTree(root).write(path, encoding='unicode')
    manifest_path = PACK/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['collision_generated_by'] = 'tasks/bowl_on_plate/build_collisions.py'
    manifest['collision_parts'] = counts
    manifest['collision_source_sha256'] = {
        name: hashlib.sha256((asset_dir(PACK)/'assets'/name).read_bytes()).hexdigest()
        for name in ('akita_black_bowl_vis.msh', 'model_vis.msh')}
    manifest['collision_notes'] = ('Convex decomposition of unchanged visual meshes using CoACD 1.0.14; '
        '1 mm decomposition threshold, followed by ray and physical contact validation. '
        'Mass, inertia, friction, task placement and grading are retained.')
    for name in ('scene.xml', 'build_collisions.py'):
        manifest['files'][name] = hashlib.sha256((PACK/name).read_bytes()).hexdigest()[:16]
    manifest_path.write_text(json.dumps(manifest, indent=1)+'\n')
    print(counts, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parts-dir', type=Path, help='Reuse previously generated collision-<geom>/*.ply files')
    args = parser.parse_args()
    build(args.parts_dir)
