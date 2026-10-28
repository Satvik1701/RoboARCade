"""ThermoMixer controls and deterministic, replayable instrument readout.

Time is simulation time. Temperature and RPM are modeled instrument states with
configurable ramp rates, not thermal sensing or an added thermoblock motor.
The existing task asks to start Program 1, not to wait for its timer to finish.
All settings, press latches and running state live in MuJoCo userdata.
"""
import math

import numpy as np

BUTTONS = (
    "prog_1", "prog_2", "prog_3", "prog_4", "prog_5", "menu_enter",
    "menu_up", "menu_down", "menu_left", "menu_right", "short", "start_stop",
    "time_up", "time_down", "temp_up", "temp_down", "speed_up", "speed_down",
)
INITIALIZED, LAST_TIME, PROGRAM, RUNNING, REMAINING, TEMPERATURE, RPM, DONE, EDITING, FIELD = range(10)
PRESETS = slice(10, 25)
LATCHES = slice(25, 43)
PAINT_TIME = 43
STATE_SIZE = 44
SEGMENTS = ("abcdef", "bc", "abdeg", "abcdg", "bcfg", "acdfg", "acdefg", "abc", "abcdefg", "abcdfg")
INK = (.04, .06, .08, 1.)
PAPER = (1., 1., 1., 1.)
BLUE = (.015, .22, .55, 1.)
GREEN = (.02, .48, .16, 1.)


class Dynamics:
    def __init__(self, env):
        self.cfg = env.spec["incubation_panel"]
        self.qadr = [env.model.joint(f"button_{name}_key").qposadr[0] for name in BUTTONS]
        self.digits = {field: np.array([[env.model.geom(f"lcd_{field}_{i}_{seg}").id
                                        for seg in range(7)] for i in range(4)])
                       for field in ("time", "temp", "rpm", "setrpm")}
        self.status = [env.model.geom(f"lcd_status_{word}").id for word in ("ready", "run", "done")]
        self.program_geoms = [env.model.geom(f"lcd_program_{i}").id for i in range(1, 6)]
        self.field_geoms = [env.model.geom(f"lcd_field_{i}").id for i in range(3)]
        self._painted = None
        self.restore(env)

    def state(self, env):
        return env.data.userdata[:STATE_SIZE]

    def reset(self, env):
        s = self.state(env)
        s[:] = 0
        s[INITIALIZED], s[LAST_TIME] = 1, env.data.time
        presets = np.array(self.cfg["programs"], dtype=float)
        if presets.shape != (5, 3):
            raise ValueError("incubation_panel.programs must hold five [seconds, celsius, rpm] presets")
        s[PRESETS] = presets.ravel()
        s[REMAINING] = presets[0, 0]
        s[TEMPERATURE] = self.cfg["ambient_celsius"]
        s[LATCHES] = env.data.qpos[self.qadr] <= -.0015
        self.paint(env, force=True)

    def deal(self, env):
        self.reset(env)

    def restore(self, env):
        if self.state(env)[INITIALIZED] != 1:
            self.reset(env)
        else:
            self.paint(env, force=True)

    def settings(self, s):
        return s[PRESETS].reshape(5, 3)[int(s[PROGRAM])]

    def adjust(self, s, field, direction):
        settings = self.settings(s)
        old = settings[field]
        low, high = self.cfg["limits"][field]
        settings[field] = np.clip(old + direction*self.cfg["increments"][field], low, high)
        if field == 0:
            s[REMAINING] = (max(0., s[REMAINING] + settings[0] - old)
                            if s[RUNNING] else settings[0])
            s[DONE] = 0

    def physics_step(self, env):
        s, d = self.state(env), env.data
        if s[INITIALIZED] != 1:
            self.reset(env)
        dt = max(0., float(d.time - s[LAST_TIME]))
        s[LAST_TIME] = d.time
        settings = self.settings(s)
        # Advance the state that was active during this elapsed interval, before
        # applying new button edges at the current simulation timestamp.
        active_dt = min(dt, s[REMAINING]) if s[RUNNING] else 0.
        if s[RUNNING]:
            s[REMAINING] = max(0., s[REMAINING] - dt)
            if s[REMAINING] <= 0:
                s[RUNNING], s[DONE] = 0, 1
        def slew(value, target, amount):
            return value + np.clip(target-value, -amount, amount)
        s[TEMPERATURE] = slew(s[TEMPERATURE], settings[1], self.cfg["heat_celsius_per_second"]*active_dt)
        s[TEMPERATURE] = slew(s[TEMPERATURE], self.cfg["ambient_celsius"], self.cfg["cool_celsius_per_second"]*(dt-active_dt))
        was_short = bool(s[LATCHES][BUTTONS.index("short")])
        if was_short:
            s[RPM] = slew(s[RPM], max(settings[2], self.cfg["short_rpm"]), self.cfg["rpm_per_second"]*dt)
        else:
            s[RPM] = slew(s[RPM], settings[2], self.cfg["rpm_per_second"]*active_dt)
            s[RPM] = slew(s[RPM], 0., self.cfg["rpm_per_second"]*(dt-active_dt))

        was_down = s[LATCHES].astype(bool)
        q = d.qpos[self.qadr]
        down = np.where(was_down, q <= -.00075, q <= -.0015)
        pressed = down & ~was_down
        s[LATCHES] = down
        for i in np.flatnonzero(pressed):
            name = BUTTONS[i]
            if name.startswith("prog_"):
                s[PROGRAM] = int(name[-1])-1
                s[REMAINING] = self.settings(s)[0]
                s[RUNNING], s[DONE] = 1, 0
            elif name == "start_stop":
                if s[RUNNING]:
                    s[RUNNING] = 0
                else:
                    if s[REMAINING] <= 0:
                        s[REMAINING] = self.settings(s)[0]
                    s[RUNNING], s[DONE] = 1, 0
            elif name == "menu_enter":
                s[EDITING] = not s[EDITING]
            elif name in ("menu_left", "menu_right"):
                s[FIELD] = (int(s[FIELD]) + (1 if name.endswith("right") else -1)) % 3
                s[EDITING] = 1
            elif name in ("menu_up", "menu_down"):
                s[EDITING] = 1
                self.adjust(s, int(s[FIELD]), 1 if name.endswith("up") else -1)
            elif name.startswith(("time_", "temp_", "speed_")):
                field = {"time": 0, "temp": 1, "speed": 2}[name.split('_')[0]]
                self.adjust(s, field, 1 if name.endswith("up") else -1)
        if pressed.any() or d.time-s[PAINT_TIME] >= .05:
            self.paint(env)
            s[PAINT_TIME] = d.time

    def readout(self, env):
        s = self.state(env)
        settings = self.settings(s)
        return dict(program=int(s[PROGRAM])+1, running=bool(s[RUNNING]), done=bool(s[DONE]),
                    remaining_seconds=float(s[REMAINING]), duration_seconds=float(settings[0]),
                    target_celsius=float(settings[1]), actual_celsius=float(s[TEMPERATURE]),
                    target_rpm=float(settings[2]), actual_rpm=float(s[RPM]),
                    short=bool(s[LATCHES][BUTTONS.index("short")]))

    def paint(self, env, force=False):
        s = self.state(env)
        settings = self.settings(s)
        seconds = int(math.ceil(max(0., s[REMAINING])-1e-8))
        values = {"time": f"{seconds//60:02d}{seconds%60:02d}",
                  "temp": f"{int(round(settings[1])):02d}{int(round(s[TEMPERATURE])):02d}",
                  "rpm": f"{int(round(s[RPM])):4d}", "setrpm": f"{int(round(settings[2])):4d}"}
        status = 1 if s[RUNNING] or s[LATCHES][BUTTONS.index("short")] else 2 if s[DONE] else 0
        key = (*values.values(), status, int(s[PROGRAM]), int(s[EDITING]), int(s[FIELD]))
        if key == self._painted and not force:
            return
        self._painted = key
        rgba = env.model.geom_rgba
        for field, text in values.items():
            for ids, char in zip(self.digits[field], text):
                on = SEGMENTS[int(char)] if char != ' ' else ''
                rgba[ids] = [INK if letter in on else PAPER for letter in 'abcdefg']
        rgba[self.status] = [GREEN if i == status else PAPER for i in range(3)]
        rgba[self.program_geoms] = [BLUE if i == int(s[PROGRAM]) else PAPER for i in range(5)]
        rgba[self.field_geoms] = [BLUE if s[EDITING] and i == int(s[FIELD]) else PAPER for i in range(3)]
