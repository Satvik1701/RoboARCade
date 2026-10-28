"""Inspect a task asset's appearance, collisions, joints and live dynamics.

Run as a module; see tasks/add_task/README.md. GUI callbacks enqueue commands:
only the simulation thread reads/writes MuJoCo or starts the Blender renderer.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import queue
import time

import mujoco
import numpy as np
from PIL import Image
import trimesh
import viser

from core.sim_env import SimEnv
from core.tasks import _look_at_wxyz, list_tasks
from tasks.add_task.geom_viewer import _geom_mesh, _merge


def descendants(model, body):
    root = model.body(body).id if body else 0
    selected = set()
    for bid in range(model.nbody):
        ancestor = bid
        while ancestor and ancestor != root:
            ancestor = int(model.body_parentid[ancestor])
        if ancestor == root:
            selected.add(bid)
    return selected


def collision_mesh(model, gid):
    """MuJoCo mesh collision uses the convex hull, not the rendered triangles."""
    vf = _geom_mesh(model, gid)
    if vf is not None and model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
        hull = trimesh.Trimesh(vertices=vf[0], faces=vf[1], process=False).convex_hull
        return np.asarray(hull.vertices), np.asarray(hull.faces)
    if vf is None and model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        tm = trimesh.creation.icosphere(subdivisions=2)
        return tm.vertices * model.geom_size[gid], tm.faces
    if vf is None and model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_PLANE:
        # Infinite planes are represented by a finite square in the viewer only.
        size = np.maximum(model.geom_size[gid][:2], 1.)
        return np.array([[-size[0], -size[1], 0], [size[0], -size[1], 0],
                         [size[0], size[1], 0], [-size[0], size[1], 0]]), np.array([[0, 1, 2], [0, 2, 3]])
    return vf


def visual_mesh(model, gid):
    vf = collision_mesh(model, gid) if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH else _geom_mesh(model, gid)
    if vf is None:
        return None
    vertices, faces = vf
    mid = int(model.geom_matid[gid])
    rgba = model.geom_rgba[gid]
    if mid >= 0 and np.allclose(rgba, [.5, .5, .5, 1]):
        rgba = model.mat_rgba[mid]
    texture = next((int(t) for t in model.mat_texid[mid] if t >= 0), -1) if mid >= 0 else -1
    if texture >= 0 and model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = int(model.geom_dataid[gid])
        fa, count, ta = model.mesh_faceadr[mesh], model.mesh_facenum[mesh], model.mesh_texcoordadr[mesh]
        indices = model.mesh_facetexcoord[fa:fa + count].ravel()
        if ta >= 0 and np.all(indices >= 0):
            height, width, channels = (int(x[texture]) for x in
                                       (model.tex_height, model.tex_width, model.tex_nchannel))
            start = model.tex_adr[texture]
            pixels = model.tex_data[start:start + height*width*channels].reshape(height, width, channels)[::-1]
            tm = trimesh.Trimesh(vertices=vertices[faces.ravel()],
                                 faces=np.arange(len(faces)*3).reshape(-1, 3), process=False)
            tm.visual = trimesh.visual.texture.TextureVisuals(
                uv=model.mesh_texcoord[ta + indices],
                material=trimesh.visual.material.PBRMaterial(baseColorTexture=Image.fromarray(pixels),
                                                             metallicFactor=0, roughnessFactor=.9))
            return tm, rgba
    return (vertices, faces), rgba


def state_array(env):
    kind = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(env.model, kind))
    mujoco.mj_getState(env.model, env.data, state, kind)
    return state


def restore_state(env, state):
    mujoco.mj_setState(env.model, env.data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
    env._dynamics('restore')
    mujoco.mj_forward(env.model, env.data)


class Inspector:
    def __init__(self, args):
        self.args = args
        self.commands = queue.Queue()
        self.renderer = None
        self.generation = 0
        self.server = viser.ViserServer(host=args.host, port=args.port, label='RoboARCade asset inspector')
        self.server.scene.set_up_direction('+z')
        self.server.scene.world_axes.visible = False
        self.load_task(args.task, args.body)
        self.server.on_client_connect(self.front)

    def load_task(self, task, body=None):
        # Load before replacing the visible scene, and only on the simulation thread.
        env = SimEnv(task, seed=self.args.seed)
        env.reset(seed=self.args.seed, randomize_init=False)
        selected = descendants(env.model, body)
        geoms = [g for g in range(env.model.ngeom) if int(env.model.geom_bodyid[g]) in selected]
        if not geoms:
            raise ValueError('Selected body has no geometry')
        self.env, self.m, self.d = env, env.model, env.data
        self.args.task, self.args.body = task, body
        self.selected, self.geoms = selected, geoms
        self.generation += 1
        # Viser retains name-keyed orientation messages after node removal.
        # New task geometry must not inherit transforms from reused geom IDs.
        self.scene_prefix = f'/task_{self.generation}'
        # Room walls obscure the workspace and distort automatic framing.
        self.room_geoms = {g for g in self.geoms
                           if self.m.geom(g).name == 'floor'
                           or (self.m.geom(g).name or '').startswith('wall_')}
        foreground = [g for g in self.geoms if g not in self.room_geoms]
        points = self.d.geom_xpos[foreground or self.geoms]
        self.origin = (points.min(axis=0) + points.max(axis=0))/2
        self.radius = max(.15, float(np.max(np.linalg.norm(points-self.origin, axis=1))))
        self.initial = state_array(self.env)
        self.saved = self.initial.copy()
        self.last_image = None
        self.pulse = None
        self.server.scene.reset()
        self.server.gui.reset()
        self.server.scene.world_axes.visible = False
        self.handles = []
        self.pose_cache = {}
        self.site_handles = []
        self.build_scene()
        self.handles_by_geom = {}
        for item in self.handles:
            self.handles_by_geom.setdefault(item[1], []).append(item)
        self.build_gui()
        self.refresh()
        for client in self.server.get_clients().values():
            self.front(client)

    def front(self, client):
        client.camera.fov = np.deg2rad(45)
        client.camera.position = (0, -self.radius*2.8, self.radius*1.7)
        client.camera.look_at = (0, 0, 0)
        client.camera.up_direction = (0, 0, 1)

    def pose(self, gid):
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, self.d.geom_xmat[gid])
        return dict(wxyz=quat, position=self.d.geom_xpos[gid]-self.origin)

    def build_scene(self):
        collision_groups = {}
        for gid in self.geoms:
            data = visual_mesh(self.m, gid)
            if data is not None and self.m.geom_group[gid] in (1, 2):
                mesh, rgba = data
                if isinstance(mesh, trimesh.Trimesh):
                    h = self.server.scene.add_mesh_trimesh(f'{self.scene_prefix}/visual/{gid}', mesh, **self.pose(gid))
                    colored = False
                else:
                    h = self.server.scene.add_mesh_simple(f'{self.scene_prefix}/visual/{gid}', *mesh,
                        color=tuple((255*np.clip(rgba[:3], 0, 1)).astype(int)), opacity=float(rgba[3]),
                        side='double', flat_shading=False, **self.pose(gid))
                    colored = True
                self.handles.append((h, gid, 'visual', colored, self.m.geom_rgba[gid].copy()))
                h.on_click(lambda event, g=gid, generation=self.generation: self.enqueue('poke',
                    (generation, g, np.array(event.ray_origin)+self.origin, np.array(event.ray_direction))))
            if self.m.geom_contype[gid] or self.m.geom_conaffinity[gid]:
                vf = collision_mesh(self.m, gid)
                if vf is not None:
                    collision_groups.setdefault((int(self.m.geom_bodyid[gid]), gid in self.room_geoms), []).append((gid, vf))
        # A body moves rigidly: batch its surface patches into one wireframe
        # instead of thousands of browser objects and per-frame messages.
        for (bid, room), chunks in collision_groups.items():
            anchor = chunks[0][0]
            rotation = self.d.geom_xmat[anchor].reshape(3, 3)
            origin = self.d.geom_xpos[anchor]
            local = []
            for gid, (vertices, faces) in chunks:
                world = self.d.geom_xpos[gid]+vertices@self.d.geom_xmat[gid].reshape(3, 3).T
                local.append(((world-origin)@rotation, faces))
            h = self.server.scene.add_mesh_simple(f'{self.scene_prefix}/collision/body{bid}_{room}', *_merge(local),
                color=(255, 140, 35), opacity=.4, wireframe=True, side='double', **self.pose(anchor))
            self.handles.append((h, anchor, 'collision', False, None))
        for sid in range(self.m.nsite):
            if int(self.m.site_bodyid[sid]) in self.selected:
                h = self.server.scene.add_frame(f'{self.scene_prefix}/sites/{sid}', axes_length=.015, axes_radius=.0006,
                                               show_axes=True, position=self.d.site_xpos[sid]-self.origin)
                label = self.server.scene.add_label(f'{self.scene_prefix}/site_labels/{sid}', self.m.site(sid).name or str(sid),
                                                   position=self.d.site_xpos[sid]-self.origin)
                self.site_handles.append((h, label, sid))
        self.contacts = self.server.scene.add_line_segments(f'{self.scene_prefix}/contacts', points=np.zeros((1, 2, 3)),
                                                           colors=(255, 35, 65), line_width=3)

    def enqueue(self, name, value=None):
        self.commands.put((name, value))

    def build_gui(self):
        gui = self.server.gui
        self.task_picker = gui.add_dropdown('Task', list_tasks(), initial_value=self.args.task)
        self.task_picker.on_update(lambda event: self.enqueue('task', event.target.value))
        gui.add_markdown(f'## {self.args.task}\nClick the asset to poke it or press a button. Drag to orbit; scroll to zoom.')
        self.show_visual = gui.add_checkbox('Visual meshes', True)
        self.show_collision = gui.add_checkbox('Show collisions', True)
        self.play = gui.add_checkbox('Run physics', False)
        gui.add_button('Reset').on_click(lambda _: self.enqueue('reset'))
        gui.add_button('Front view').on_click(lambda _: [self.front(c) for c in self.server.get_clients().values()])
        gui.add_button('Render in Blender').on_click(lambda event: self.enqueue('render',
            (np.array(event.client.camera.position)+self.origin,
             np.array(event.client.camera.look_at)+self.origin, float(event.client.camera.fov))))
        self.status = gui.add_markdown('Visual surfaces with orange collision outlines. Click an object to poke it.')
        self.readout = gui.add_markdown('')
        self.render_image = gui.add_image(np.zeros((16, 16, 3), dtype=np.uint8), label='Blender render')
        with gui.add_folder('More tools', expand_by_default=False):
            self.show_room = gui.add_checkbox('Room walls and floor', False)
            self.show_sites = gui.add_checkbox('Sites and axes', False)
            self.show_contacts = gui.add_checkbox('Contact forces', False)
            self.speed = gui.add_slider('Simulation speed', min=.1, max=5., step=.1, initial_value=1.)
            gui.add_button('Step 0.1 s').on_click(lambda _: self.enqueue('step', .1))
            gui.add_button('Advance 10 s').on_click(lambda _: self.enqueue('step', 10.))
            gui.add_button('Save state').on_click(lambda _: self.enqueue('save'))
            gui.add_button('Restore state').on_click(lambda _: self.enqueue('restore'))
            self.camera = gui.add_dropdown('Render camera', ['Inspector view', *self.env.cameras])
            self.isolate = gui.add_checkbox('Isolate selected body in Blender', bool(self.args.body))
            gui.add_button('Export state and report').on_click(lambda _: self.enqueue('export'))
            self.contact_info = gui.add_markdown('')

    def poke(self, value):
        generation, gid, origin, direction = value
        if generation != self.generation:
            return
        direction = direction / np.linalg.norm(direction)
        bid = int(self.m.geom_bodyid[gid])
        if self.m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
            distance = mujoco.mj_rayMesh(self.m, self.d, gid, origin, direction)
        else:
            distance = mujoco.mju_rayGeom(self.d.geom_xpos[gid], self.d.geom_xmat[gid],
                self.m.geom_size[gid], origin, direction, int(self.m.geom_type[gid]))
        point = origin + distance*direction if distance >= 0 else self.d.geom_xpos[gid].copy()
        # Small spring-loaded slide joints are pushbuttons. Press along their
        # travel axis so the same mouse gesture works from oblique views.
        button = None
        ancestor = bid
        while ancestor:
            for j in range(self.m.body_jntadr[ancestor], self.m.body_jntadr[ancestor]+self.m.body_jntnum[ancestor]):
                if (self.m.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE
                        and self.m.jnt_limited[j] and np.ptp(self.m.jnt_range[j]) <= .02):
                    button = j
                    break
            if button is not None: break
            ancestor = int(self.m.body_parentid[ancestor])
        self.pulse = dict(body=bid, point=point, force=direction*.7,
                          button=button, until=self.d.time+.3)
        label = self.m.body(ancestor if button is not None else bid).name
        label = label.removeprefix('button_').removesuffix('_key').replace('prog_', 'Program ').replace('_', ' ')
        self.status.content = f'Pressing {label}' if button is not None else f'Poking {label}'

    def advance(self, seconds):
        for _ in range(max(1, round(seconds/self.m.opt.timestep))):
            self.d.qfrc_applied[:] = 0
            if self.pulse:
                pulse = self.pulse
                if self.d.time < pulse['until']:
                    if pulse['button'] is not None:
                        j = pulse['button']
                        sign = -1 if abs(self.m.jnt_range[j, 0]) > abs(self.m.jnt_range[j, 1]) else 1
                        self.d.qfrc_applied[self.m.jnt_dofadr[j]] = sign*.4
                    else:
                        mujoco.mj_applyFT(self.m, self.d, pulse['force'], np.zeros(3),
                                         pulse['point'], pulse['body'], self.d.qfrc_applied)
                elif self.d.time >= pulse['until']+.25:
                    self.pulse = None
                    self.status.content = 'Released — click the asset to interact again.'
            self.env._step_physics()
        self.d.qfrc_applied[:] = 0
        self.env._dynamics('physics_step')
        mujoco.mj_forward(self.m, self.d)

    def report(self):
        readout = getattr(self.env.dynamics, 'readout', None)
        return dict(task=self.args.task, body=self.args.body, seed=self.args.seed,
            simulation_time=float(self.d.time), dynamics=readout(self.env) if readout else None,
            joints=[dict(name=self.m.joint(j).name, type=int(self.m.jnt_type[j]),
                         range=self.m.jnt_range[j].tolist(), limited=bool(self.m.jnt_limited[j]),
                         qpos=self.d.qpos[self.m.jnt_qposadr[j]:self.m.jnt_qposadr[j]+(
                             7 if self.m.jnt_type[j] == 0 else 4 if self.m.jnt_type[j] == 1 else 1)].tolist())
                    for j in range(self.m.njnt) if int(self.m.jnt_bodyid[j]) in self.selected],
            bodies=[dict(name=self.m.body(b).name, mass=float(self.m.body_mass[b]),
                         inertia=self.m.body_inertia[b].tolist()) for b in sorted(self.selected)],
            geoms=[dict(name=self.m.geom(g).name, group=int(self.m.geom_group[g]),
                        contype=int(self.m.geom_contype[g]), conaffinity=int(self.m.geom_conaffinity[g]),
                        friction=self.m.geom_friction[g].tolist(), size=self.m.geom_size[g].tolist()) for g in self.geoms])

    def render(self, view):
        from core.render import BlenderRenderer
        self.play.value = False
        self.status.content = 'Rendering current state with Blender…'
        camera = 'agentview' if self.camera.value == 'Inspector view' else self.camera.value
        cid = self.m.camera(camera).id
        old_pos, old_quat = self.m.cam_pos[cid].copy(), self.m.cam_quat[cid].copy()
        old_fovy, groups = self.m.cam_fovy[cid], self.m.geom_group.copy()
        try:
            if self.camera.value == 'Inspector view':
                pos, target, fov = view
                parent = int(self.m.cam_bodyid[cid])
                mat = self.d.xmat[parent].reshape(3, 3)
                world_quat = _look_at_wxyz(pos, target)
                world_mat = np.empty(9); mujoco.mju_quat2Mat(world_mat, world_quat)
                self.m.cam_pos[cid] = mat.T @ (pos-self.d.xpos[parent])
                mujoco.mju_mat2Quat(self.m.cam_quat[cid], (mat.T @ world_mat.reshape(3, 3)).ravel())
                self.m.cam_fovy[cid] = np.rad2deg(fov)
            if self.isolate.value:
                for gid in range(self.m.ngeom):
                    if int(self.m.geom_bodyid[gid]) not in self.selected:
                        self.m.geom_group[gid] = 4
            mujoco.mj_forward(self.m, self.d)
            # Rebuild so isolation/material changes and current dimensions are exact.
            self.env.image_size = self.args.resolution
            self.renderer = BlenderRenderer(self.env, [camera])
            self.last_image = self.renderer.render()[camera]
            self.render_image.image = self.last_image
            output = self.export()
            Image.fromarray(self.last_image).save(output/'blender.png')
            self.status.content = f'Blender render saved: `{output}/blender.png`'
        finally:
            if self.renderer:
                self.renderer.close(); self.renderer = None
            self.m.cam_pos[cid], self.m.cam_quat[cid], self.m.cam_fovy[cid] = old_pos, old_quat, old_fovy
            self.m.geom_group[:] = groups
            mujoco.mj_forward(self.m, self.d)

    def export(self):
        output = self.args.output/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        output.mkdir(parents=True)
        (output/'report.json').write_text(json.dumps(self.report(), indent=2)+'\n')
        np.savez_compressed(output/'state.npz', state=state_array(self.env),
                            state_spec=int(mujoco.mjtState.mjSTATE_INTEGRATION))
        mujoco.mj_saveModel(self.m, str(output/'scene.mjb'), None)
        return output

    def command(self, name, value):
        if name == 'task':
            if value != self.args.task:
                self.play.value = False
                self.status.content = f'Loading {value}…'
                self.load_task(value)
            return
        if name in ('reset', 'restore'):
            self.pulse = None
            restore_state(self.env, self.initial if name == 'reset' else self.saved)
        elif name == 'save':
            self.saved = state_array(self.env)
        elif name == 'step':
            self.play.value = False; self.advance(value)
        elif name == 'poke':
            self.poke(value); return
        elif name == 'render':
            self.render(value); return
        elif name == 'export':
            self.status.content = f'Exported to `{self.export()}`'; return
        self.status.content = f'{name.capitalize()} complete'

    def refresh(self):
        for h, gid, kind, colored, baseline in self.handles:
            pose = self.pose_cache.get(gid)
            current = np.concatenate((self.d.geom_xpos[gid], self.d.geom_xmat[gid]))
            if pose is None or not np.array_equal(pose, current):
                transform = self.pose(gid)
                for handle, index, *_ in self.handles_by_geom[gid]:
                    handle.position = transform['position']; handle.wxyz = transform['wxyz']
                self.pose_cache[gid] = current
            h.visible = ((self.show_visual.value if kind == 'visual' else self.show_collision.value)
                         and (gid not in self.room_geoms or self.show_room.value))
            if kind == 'collision':
                h.wireframe = self.show_visual.value
                h.opacity = .4 if self.show_visual.value else .85
            if colored:
                rgba = self.m.geom_rgba[gid]
                mid = int(self.m.geom_matid[gid])
                if mid >= 0 and np.allclose(rgba, [.5, .5, .5, 1]):
                    rgba = self.m.mat_rgba[mid]
                h.color = tuple((255*np.clip(rgba[:3], 0, 1)).astype(int))
                h.opacity = float(rgba[3])
        for h, label, sid in self.site_handles:
            h.visible = label.visible = self.show_sites.value
            h.position = label.position = self.d.site_xpos[sid]-self.origin
            quat = np.empty(4); mujoco.mju_mat2Quat(quat, self.d.site_xmat[sid]); h.wxyz = quat
        lines, contact_rows = [], []
        for idx, contact in enumerate(self.d.contact):
            if int(self.m.geom_bodyid[contact.geom1]) not in self.selected and int(self.m.geom_bodyid[contact.geom2]) not in self.selected:
                continue
            force = np.zeros(6); mujoco.mj_contactForce(self.m, self.d, idx, force)
            pos = contact.pos-self.origin
            lines.append([pos, pos+contact.frame[:3]*min(.1, .005+.005*abs(force[0]))])
            contact_rows.append(f'{self.m.geom(contact.geom1).name} ↔ {self.m.geom(contact.geom2).name}: {contact.dist*1000:.2f} mm, {force[0]:.3f} N')
        self.contacts.visible = self.show_contacts.value and bool(lines)
        if lines: self.contacts.points = np.asarray(lines)
        self.contact_info.content = f'**Contacts: {len(lines)}**\n\n'+'  \n'.join(contact_rows[:12])
        readout = getattr(self.env.dynamics, 'readout', None)
        values = readout(self.env) if readout else {'userdata': self.d.userdata.tolist()}
        if readout and {'program', 'running', 'remaining_seconds', 'actual_celsius', 'target_celsius', 'actual_rpm'} <= values.keys():
            seconds = max(0, round(values['remaining_seconds']))
            state = 'Running' if values['running'] else 'Done' if values.get('done') else 'Ready'
            self.readout.content = (f"**Program {values['program']} · {state}**  \n"
                f"Time: {seconds//60}:{seconds%60:02d}  \n"
                f"Temperature: {values['actual_celsius']:.0f} / {values['target_celsius']:.0f} °C  \n"
                f"Speed: {values['actual_rpm']:.0f} RPM")
        else:
            self.readout.content = f'Simulation time: {self.d.time:.1f} s'

    def run(self):
        try:
            while True:
                start = time.monotonic()
                try:
                    name, value = self.commands.get_nowait()
                    self.command(name, value)
                except queue.Empty:
                    if self.play.value or self.pulse: self.advance(self.speed.value/30)
                except Exception as exc:
                    self.play.value = False
                    self.status.content = f'Error: {type(exc).__name__}: {exc}'
                    print(self.status.content, flush=True)
                self.refresh()
                time.sleep(max(0., 1/30-(time.monotonic()-start)))
        finally:
            if self.renderer: self.renderer.close()
            self.server.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=list_tasks(), default='start_incubation',
                        help='Initial task; switch between all available tasks in the viewer')
    parser.add_argument('--body', help='Inspect this body and its descendants; omit for the whole task')
    parser.add_argument('--seed', type=int, default=1000000)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8410)
    parser.add_argument('--resolution', type=int, default=1024)
    parser.add_argument('--output', type=Path, default=Path('data/asset_inspector'))
    args = parser.parse_args()
    if not 64 <= args.resolution <= 4096: parser.error('--resolution must be between 64 and 4096')
    args.output = args.output.resolve()
    Inspector(args).run()


if __name__ == '__main__':
    main()
