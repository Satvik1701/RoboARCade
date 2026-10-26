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
from core.tasks import _look_at_wxyz
from tasks.add_task.geom_viewer import _geom_mesh


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
        self.env = SimEnv(args.task, seed=args.seed)
        self.env.reset(seed=args.seed, randomize_init=False)
        self.m, self.d = self.env.model, self.env.data
        self.selected = descendants(self.m, args.body)
        self.geoms = [g for g in range(self.m.ngeom) if int(self.m.geom_bodyid[g]) in self.selected]
        if not self.geoms:
            raise ValueError('Selected body has no geometry')
        points = self.d.geom_xpos[self.geoms]
        self.origin = (points.min(axis=0) + points.max(axis=0))/2
        self.radius = max(.15, float(np.max(np.linalg.norm(points-self.origin, axis=1))))
        self.initial = state_array(self.env)
        self.saved = self.initial.copy()
        self.commands = queue.Queue()
        self.renderer = None
        self.last_image = None
        self.pulse = None
        self.server = viser.ViserServer(host=args.host, port=args.port, label=f'Asset inspector · {args.task}')
        self.server.scene.set_up_direction('+z')
        self.server.scene.world_axes.visible = False
        self.handles = []
        self.pose_cache = {}
        self.site_handles = []
        self.build_scene()
        self.handles_by_geom = {}
        for item in self.handles:
            self.handles_by_geom.setdefault(item[1], []).append(item)
        self.build_gui()
        self.server.on_client_connect(self.front)

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
        for gid in self.geoms:
            data = visual_mesh(self.m, gid)
            if data is not None and self.m.geom_group[gid] in (1, 2):
                mesh, rgba = data
                if isinstance(mesh, trimesh.Trimesh):
                    h = self.server.scene.add_mesh_trimesh(f'/visual/{gid}', mesh, **self.pose(gid))
                    colored = False
                else:
                    h = self.server.scene.add_mesh_simple(f'/visual/{gid}', *mesh,
                        color=tuple((255*np.clip(rgba[:3], 0, 1)).astype(int)), opacity=float(rgba[3]),
                        side='double', flat_shading=False, **self.pose(gid))
                    colored = True
                self.handles.append((h, gid, 'visual', colored, self.m.geom_rgba[gid].copy()))
            if self.m.geom_contype[gid] or self.m.geom_conaffinity[gid]:
                vf = collision_mesh(self.m, gid)
                if vf is not None:
                    h = self.server.scene.add_mesh_simple(f'/collision/{gid}', *vf,
                        color=(255, 140, 35), opacity=.4, wireframe=True, side='double', **self.pose(gid))
                    self.handles.append((h, gid, 'collision', False, None))
        for sid in range(self.m.nsite):
            if int(self.m.site_bodyid[sid]) in self.selected:
                h = self.server.scene.add_frame(f'/sites/{sid}', axes_length=.015, axes_radius=.0006,
                                               show_axes=True, position=self.d.site_xpos[sid]-self.origin)
                label = self.server.scene.add_label(f'/site_labels/{sid}', self.m.site(sid).name or str(sid),
                                                   position=self.d.site_xpos[sid]-self.origin)
                self.site_handles.append((h, label, sid))
        self.contacts = self.server.scene.add_line_segments('/contacts', points=np.zeros((1, 2, 3)),
                                                           colors=(255, 35, 65), line_width=3)

    def enqueue(self, name, value=None):
        self.commands.put((name, value))

    def build_gui(self):
        gui = self.server.gui
        gui.add_markdown(f'## {self.args.task}\nBody: **{self.args.body or "whole scene"}**\n\nOrbit and zoom to inspect. Physics starts paused.')
        with gui.add_folder('Appearance and collisions'):
            self.show_visual = gui.add_checkbox('Visual meshes', True)
            self.show_collision = gui.add_checkbox('Collision hulls', False)
            self.show_sites = gui.add_checkbox('Sites and axes', False)
            self.show_contacts = gui.add_checkbox('Contact forces', True)
            gui.add_button('Front view').on_click(lambda _: [self.front(c) for c in self.server.get_clients().values()])
            gui.add_markdown('Orange shows collision hulls. Red lines show contact normals scaled by force (capped at 10 cm). Infinite planes use a finite preview square.')
        with gui.add_folder('Physics and state'):
            self.play = gui.add_checkbox('Run physics', False)
            self.speed = gui.add_slider('Simulation speed', min=.1, max=5., step=.1, initial_value=1.)
            gui.add_button('Step 0.1 s').on_click(lambda _: self.enqueue('step', .1))
            gui.add_button('Advance 10 s').on_click(lambda _: self.enqueue('step', 10.))
            gui.add_button('Reset asset').on_click(lambda _: self.enqueue('reset'))
            gui.add_button('Save state').on_click(lambda _: self.enqueue('save'))
            gui.add_button('Restore state').on_click(lambda _: self.enqueue('restore'))
            self.readout = gui.add_markdown('')
        joints = [self.m.joint(j).name for j in range(self.m.njnt)
                  if int(self.m.jnt_bodyid[j]) in self.selected and self.m.jnt_type[j] in
                  (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)]
        self.joint = None
        if joints:
            with gui.add_folder('Joint and button inspection'):
                self.joint = gui.add_dropdown('Joint', joints)
                self.force = gui.add_number('Force / torque (N / Nm)', initial_value=-.4, step=.1)
                self.duration = gui.add_number('Pulse duration (s)', initial_value=.3, min=.01, max=10., step=.1)
                gui.add_button('Apply force pulse').on_click(lambda _: self.enqueue('pulse',
                    (self.joint.value, float(self.force.value), float(self.duration.value))))
                self.position = gui.add_number('Inspection position (m / rad)', initial_value=0., step=.001)
                gui.add_button('Set inspection pose').on_click(lambda _: self.enqueue('pose',
                    (self.joint.value, float(self.position.value))))
                self.joint_info = gui.add_markdown('')
                gui.add_markdown('Force pulses advance real physics and exercise spring return. Set inspection pose directly changes the joint; it is not a contact test. Angles are radians.')
        with gui.add_folder('Blender and export'):
            self.camera = gui.add_dropdown('Render camera', ['Inspector view', *self.env.cameras])
            self.isolate = gui.add_checkbox('Isolate selected body in Blender', bool(self.args.body))
            gui.add_button('Render current state').on_click(lambda event: self.enqueue('render',
                (np.array(event.client.camera.position)+self.origin,
                 np.array(event.client.camera.look_at)+self.origin, float(event.client.camera.fov))))
            self.render_image = gui.add_image(np.zeros((16, 16, 3), dtype=np.uint8), label='Current Blender render')
            gui.add_button('Export state and report').on_click(lambda _: self.enqueue('export'))
            gui.add_markdown(f'Renders are {self.args.resolution} × {self.args.resolution}. Output: `{self.args.output}`. Rendering pauses physics; press Run physics to resume.')
        self.status = gui.add_markdown('Ready')
        self.contact_info = gui.add_markdown('')

    def advance(self, seconds):
        for _ in range(max(1, round(seconds/self.m.opt.timestep))):
            self.d.qfrc_applied[:] = 0
            if self.pulse:
                joint, force, until = self.pulse
                if self.d.time < until:
                    self.d.qfrc_applied[self.m.joint(joint).dofadr[0]] = force
                else:
                    self.pulse = None
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
        if name in ('reset', 'restore'):
            self.play.value = False; self.pulse = None
            restore_state(self.env, self.initial if name == 'reset' else self.saved)
        elif name == 'save':
            self.saved = state_array(self.env)
        elif name == 'step':
            self.play.value = False; self.advance(value)
        elif name == 'pulse':
            joint, force, duration = value
            self.pulse = (joint, force, self.d.time+duration)
            self.advance(duration+.25)
        elif name == 'pose':
            self.play.value = False
            joint, value = value; j = self.m.joint(joint)
            if self.m.jnt_limited[j.id]:
                value = np.clip(value, *self.m.jnt_range[j.id])
            self.d.qpos[j.qposadr[0]] = value; self.d.qvel[j.dofadr[0]] = 0
            self.env._dynamics('physics_step'); mujoco.mj_forward(self.m, self.d)
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
            h.visible = self.show_visual.value if kind == 'visual' else self.show_collision.value
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
        self.readout.content = f'**Simulation time: {self.d.time:.2f} s**\n```json\n{json.dumps(values, indent=2)}\n```'
        if self.joint:
            j = self.m.joint(self.joint.value)
            self.joint_info.content = (f'Position: **{self.d.qpos[j.qposadr[0]]:.5f}**  \n'
                f'Velocity: {self.d.qvel[j.dofadr[0]]:.5f}  \nRange: {self.m.jnt_range[j.id].tolist()}  \n'
                f'Stiffness: {self.m.jnt_stiffness[j.id]:g}; damping: {self.m.dof_damping[j.dofadr[0]]:g}')

    def run(self):
        try:
            while True:
                start = time.monotonic()
                try:
                    name, value = self.commands.get_nowait()
                    self.command(name, value)
                except queue.Empty:
                    if self.play.value: self.advance(self.speed.value/30)
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
    parser.add_argument('--task', required=True)
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
