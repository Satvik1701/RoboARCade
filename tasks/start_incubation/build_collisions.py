"""Build thin convex collision patches from the authored machine surfaces.

Offline only: MuJoCo, numpy, scipy and trimesh. No convex hull spans a concavity
or fills a well. Coplanar faces merge only when their union is convex. Each patch
is thickened by 0.15 mm on either side, bounding surface error while preventing
zero-volume collision meshes. Source visual OBJ files remain in the HF assets.
"""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull
import trimesh

from core.tasks import ASSETS

PACK = Path(__file__).resolve().parent
HALF_THICKNESS = .00015


def patches(mesh):
    """Merge planar convex facets; retain triangles around concave boundaries."""
    used = set()
    for face_ids in mesh.facets:
        points = mesh.vertices[np.unique(mesh.faces[face_ids])]
        normal = mesh.face_normals[face_ids[0]]
        origin = points[0]
        u = points[np.argmax(np.linalg.norm(points-origin, axis=1))]-origin
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        uv = np.column_stack(((points-origin)@u, (points-origin)@v))
        if np.max(np.abs((points-origin)@normal)) > 2e-5:
            continue
        hull = ConvexHull(uv)
        area = mesh.area_faces[face_ids].sum()
        if abs(hull.volume-area) > max(1e-12, area*1e-5):
            continue
        yield points[hull.vertices], normal
        used.update(map(int, face_ids))
    for fid, triangle in enumerate(mesh.triangles):
        if fid not in used and mesh.area_faces[fid] > 1e-12:
            yield triangle, mesh.face_normals[fid]


def fmt(values):
    return ' '.join(f'{float(x):.10g}' for x in np.asarray(values).ravel())


def build():
    path = PACK/'scene.xml'
    root = ET.parse(path, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))).getroot()
    asset = root.find('asset')
    mixer = root.find('.//body[@name="mixer"]')
    block = mixer.find('body[@name="thermoblock"]')
    # Generated patches replace the five housing boxes and the old approximate
    # thermoblock. Existing inertia, button joints/caps and tube/rack stay intact.
    for geom in list(mixer.findall('geom')):
        if geom.get('class') == 'collision': mixer.remove(geom)
    for geom in list(block.findall('geom')): block.remove(geom)
    for mesh in list(asset.findall('mesh')):
        if mesh.get('name', '').startswith('aligned_collision_'): asset.remove(mesh)
    source = ASSETS/'autobio/instrument/thermal_mixer_eppendorf_c'
    meshes = [trimesh.load(p, force='mesh', process=True) for p in sorted(source.glob('body-visual-*.obj'))]
    surface = trimesh.util.concatenate(meshes)
    surface.merge_vertices()
    counts = dict(housing=0, thermoblock=0)
    for i, (points, normal) in enumerate(patches(surface)):
        center = points.mean(axis=0)
        name = f'aligned_collision_{i}'
        vertices = np.concatenate((points-center+HALF_THICKNESS*normal,
                                   points-center-HALF_THICKNESS*normal))
        ET.SubElement(asset, 'mesh', name=name, vertex=fmt(vertices))
        is_block = (center[1] > .054 and -.044 < center[0] < .08 and -.085 < center[2] < .075)
        parent, kind = (block, 'thermoblock') if is_block else (mixer, 'housing')
        ET.SubElement(parent, 'geom', name=f'{kind}_surface_{counts[kind]}',
            type='mesh', mesh=name, pos=fmt(center), **{'class':'collision'},
            condim='4', friction='1 0.002 0.0001', solimp='0.998 0.998', solref='0.001 1')
        counts[kind] += 1
    panel = mixer.find("geom[@name='incubation_panel']")
    ET.SubElement(mixer, 'geom', name='housing_panel_surface', type='mesh',
        mesh=panel.get('mesh'), pos=panel.get('pos'), quat=panel.get('quat'),
        **{'class': 'collision'}, condim='4', friction='1 0.002 0.0001')
    # Locate all 24 floor disks in the visual mesh; keep row/column names but
    # align each grading/reference site with the actual well floor and center.
    mask = ((surface.triangles_center[:, 1] > .065) & (surface.triangles_center[:, 1] < .068)
            & (np.abs(surface.face_normals[:, 1]) > .99))
    floors = surface.submesh([np.flatnonzero(mask)], append=True).split(only_watertight=False)
    assert len(floors) == 24
    centers = np.array([f.centroid for f in floors])
    assigned = set()
    for site in block.findall('site'):
        if not site.get('name', '').startswith('machine_well_'): continue
        old = np.fromstring(site.get('pos'), sep=' ')
        idx = int(np.argmin(np.linalg.norm(centers[:, [0, 2]]-old[[0, 2]], axis=1)))
        assert idx not in assigned
        assigned.add(idx)
        site.set('pos', fmt(centers[idx]))
    assert len(assigned) == 24
    ET.indent(root, space='  ')
    ET.ElementTree(root).write(path, encoding='unicode')
    print(counts)
    manifest_path = PACK/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['collision_generated_by'] = 'tasks/start_incubation/build_collisions.py'
    manifest['collision_source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                           for p in sorted(source.glob('body-visual-*.obj'))}
    manifest['collision_surface_tolerance_mm'] = HALF_THICKNESS*1000
    manifest['notes'] = manifest['notes'].replace('physical wells, rack, tubes and friction caps are retained.',
        'rack, tubes and friction caps are retained; machine-well reference sites follow the corrected surfaces.')
    manifest['notes'] = manifest['notes'].split(' Housing and thermoblock collision surfaces')[0] + ' Housing and thermoblock collision surfaces follow the source visual triangles, thickened by 0.15 mm; well sites follow the visual floor centers.'
    for name in ('scene.xml', 'build_collisions.py'):
        manifest['files'][name] = hashlib.sha256((PACK/name).read_bytes()).hexdigest()[:16]
    manifest_path.write_text(json.dumps(manifest, indent=1)+'\n')


if __name__ == '__main__':
    build()
