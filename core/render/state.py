"""Complete camera/geometry state shared by dedicated and grouped render clients."""
import mujoco
import numpy as np


class RenderState:
    def __init__(self, env, cams=None):
        self.env = env
        self.cams = list(env.cameras if cams is None else cams)
        self._geom_rgba_baseline = np.array(env.model.geom_rgba, copy=True)
        vis = env.model.vis.map
        self.clip = [float(vis.znear * env.model.stat.extent),
                     float(vis.zfar * env.model.stat.extent)]

    def _request(self, *, complete_rgba=False):
        d = self.env.data
        rgba = np.asarray(self.env.model.geom_rgba)
        changed = (range(self.env.model.ngeom) if complete_rgba else
                   np.flatnonzero(np.any(rgba != self._geom_rgba_baseline, axis=1)))
        cams = []
        for name in self.cams:
            cid = self.env.model.camera(name).id
            pos, mat = d.cam_xpos[cid], d.cam_xmat[cid]
            cams.append({"name": name, "fovy": float(self.env.model.cam(name).fovy[0]),
                         "clip": self.clip, "pos": [float(v) for v in pos],
                         "mat": [float(v) for v in mat]})
        return {
            "geoms": {str(g): [*map(float, d.geom_xpos[g]), *map(float, d.geom_xmat[g])]
                      for g in range(self.env.model.ngeom)},
            "geom_rgba": {str(g): [float(v) for v in rgba[g]] for g in changed},
            "flexes": {
                mujoco.mj_id2name(self.env.model, mujoco.mjtObj.mjOBJ_FLEX, f):
                np.asarray(d.flexvert_xpos[self.env.model.flex_vertadr[f]:
                                           self.env.model.flex_vertadr[f] + self.env.model.flex_vertnum[f]]).ravel().tolist()
                for f in range(self.env.model.nflex)
            },
            "cams": cams,
        }
