"""Public sensor snapshot encoder. No task identities, object poses, or grading.

sim_env_server captures on the simulator thread and publishes only complete
rendered observations; all cameras, depth and calibration share one frame id.
"""
import base64
import cv2
import numpy as np

DEPTH_NEAR = 0.05
DEPTH_FAR = 3.0
DEPTH_ENCODING = {"units":"m", "convention":"optical_axis_z", "near":DEPTH_NEAR,
                  "far":DEPTH_FAR, "invalid":0,
                  "decode":"z=0.05+(code-1)*2.95/254; valid codes 1..255; out-of-range=0"}


def encoded(ext, a):
    ok, buf = cv2.imencode(ext, a, [cv2.IMWRITE_JPEG_QUALITY, 85] if ext == '.jpg' else [])
    if not ok:
        raise ValueError('Image encoding failed')
    return base64.b64encode(buf).decode('ascii')


def capture(env, obs, feeds, static_cams):
    aliases = {label: cam for _, label, cam in feeds}
    aliases.update({label: cam for label, cam in static_cams})
    cameras = {}
    for label, cam in aliases.items():
        rgb, depth = obs.get(f'{cam}_image'), obs.get(f'{cam}_depth')
        if rgb is None or depth is None:
            continue
        cameras[label] = {'rgb':np.asarray(rgb).copy(), 'depth':np.asarray(depth).squeeze().copy(),
                          'K':env.get_camera_intrinsics(cam).copy(),
                          'E':env.get_camera_extrinsics(cam).copy()}
    return cameras


def encode_snapshot(cameras, frame_id):
    out = {}
    for name, c in cameras.items():
        rgb, depth = c['rgb'], c['depth']
        h,w = rgb.shape[:2]
        if depth.shape != (h,w):
            raise ValueError('RGB and depth dimensions differ')
        small_rgb = cv2.resize(rgb, (224,224), interpolation=cv2.INTER_AREA)
        small_d = cv2.resize(depth, (224,224), interpolation=cv2.INTER_NEAREST_EXACT)
        valid = np.isfinite(small_d) & (small_d >= DEPTH_NEAR) & (small_d <= DEPTH_FAR)
        gray = np.zeros((224,224), np.uint8)
        gray[valid] = 1 + np.rint((small_d[valid]-DEPTH_NEAR)*254/(DEPTH_FAR-DEPTH_NEAR)).astype(np.uint8)
        scale = np.array([[224/w,0,224/w*0.5-0.5],[0,224/h,224/h*0.5-0.5],[0,0,1]])
        out[name] = {'resolution':[224,224], 'native_resolution':[w,h],
                     'K':np.round(scale @ c['K'],6).tolist(),
                     'T_robot_from_camera':np.round(c['E'],6).tolist(),
                     'rgb':encoded('.jpg',small_rgb[:,:,::-1]),
                     'depth':encoded('.png',gray), 'raw':encoded('.jpg',rgb[:,:,::-1])}
    return {'frame_id':frame_id,'camera_frame':'x right,y down,z forward',
            'robot_frame':'RoboARCade command frame (sim world metres)',
            'depth_encoding':DEPTH_ENCODING,'views':out}
