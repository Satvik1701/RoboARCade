"""The vortex mixer's motor and display.

The vortex mixer: motor and display for the declared eccentric cup. The knob picks a speed, the switch runs it,
and a tube held pressed in the cup mixes; a lamp and a tilting liquid show the progress. task.json's `vortex`
declares the joints, geoms, speeds and thresholds."""
import mujoco
import numpy as np

from core.grade_episode import _grasped


class Dynamics:
    def __init__(self, env):
        self.cfg = c = env.spec["vortex"]
        m = env.model
        self.actuator = m.actuator(c["actuator"]).id
        self.switch = m.joint(c["switch"]).qposadr[0]
        self.knob = m.joint(c["knob"]).qposadr[0]
        self.velocity = m.joint(c["joint"]).dofadr[0]
        self.cup = set(map(int, env._subtree_geoms(c["cup"])))
        root = int(m.body_rootid[m.body(c["cup"]).id])
        self.instrument = set(map(int, env._subtree_geoms(m.body(root).name)))
        self.sample = set(map(int, env._subtree_geoms(c["sample"])))
        self.lamp = m.geom(c["lamp"]).id
        self.liquid = m.geom(c["liquid"]).id
        self.liquid_quat = m.geom_quat[self.liquid].copy()

    def state(self, env):
        start = self.cfg["state_index"]
        return env.data.userdata[start:start + 4]

    def reset(self, env):
        state = self.state(env)
        state[:] = [0, -1, 0, env.episode_t0]
        env.data.ctrl[self.actuator] = 0
        self.paint(env)

    def forces(self, env):
        total = sample = 0.0
        wrench = np.empty(6)
        for i, contact in enumerate(env.data.contact[:env.data.ncon]):
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if g1 in self.cup and g2 not in self.instrument:
                other = g2
            elif g2 in self.cup and g1 not in self.instrument:
                other = g1
            else:
                continue
            mujoco.mj_contactForce(env.model, env.data, i, wrench)
            normal = max(0.0, float(wrench[0]))
            total += normal
            if other in self.sample:
                sample += normal
        return total, sample

    def physics_step(self, env):
        c, d = self.cfg, env.data
        state = self.state(env)
        if state[3] != env.episode_t0:
            self.reset(env)
        tick = env.num_step + 1
        if state[1] == tick:
            return
        state[1] = tick
        switch = float(d.qpos[self.switch])
        on = c["on"][0] <= switch <= c["on"][1]
        touch = c["touch"][0] <= switch <= c["touch"][1]
        off = c["off"][0] <= switch <= c["off"][1]
        detent = int(np.clip(np.floor(d.qpos[self.knob] / c["detent_spacing"] + 0.5), 0, 10))
        rpm = float(np.interp(detent, [1, 10], c["rpm"])) if detent else 0.0
        total, sample = self.forces(env)
        d.ctrl[self.actuator] = rpm * np.pi / 30 if (
            on or touch and total >= c["touch_force"]) else 0.0
        speed = float(d.qvel[self.velocity])
        mixing = (on or touch) and sample >= c["touch_force"] and (
            speed >= c["mix_rpm"] * np.pi / 30) and _grasped(env, c["sample"])
        if off:
            state[0] = 0
        elif mixing:
            state[0] = min(c["mix_steps"], state[0] + 1)
        liquid_target = (float(np.clip(speed / (c["mix_rpm"] * np.pi / 30), 0, 1))
                         if sample >= c["touch_force"] else 0.0)
        decay = np.exp(-1 / (env.control_freq * c["liquid_relaxation"]))
        state[2] = liquid_target + (state[2] - liquid_target) * decay
        self.paint(env)

    def paint(self, env):
        c, m = self.cfg, env.model
        state = self.state(env)
        lit = state[0] >= c["mix_steps"]
        spinning = abs(env.data.qvel[self.velocity]) > c["stop_speed"]
        m.geom_rgba[self.lamp] = c["lamp_rgba"]["green" if lit else "amber" if spinning else "off"]
        angle = c["liquid_tilt"] * state[2]
        quat = np.array([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0])
        mujoco.mju_mulQuat(m.geom_quat[self.liquid], self.liquid_quat, quat)

    def deal(self, env):
        self.reset(env)
        env.forward()

    def restore(self, env):
        self.paint(env)
