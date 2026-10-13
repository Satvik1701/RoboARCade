"""Seven RGB-D waypoint tools over motion.py's immediate waypoint semantics."""
import importlib.util
import json
from pathlib import Path

import mcp.types as types
from core.kernel import fetch_sim_json, send_command, EPISODE_OVER

_spec = importlib.util.spec_from_file_location('rgbd_motion', Path(__file__).with_name('motion.py'))
motion = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(motion)


def text(s):
    return types.TextContent(type='text',text=s)


def packed(d):
    return json.dumps(d,separators=(',',':'))


async def snapshot():
    d = await fetch_sim_json('rgbd.json')
    if not d or not d.get('views'):
        raise RuntimeError('RGB-D snapshot unavailable. This interface requires the opt-in RGB-D sensor backend.')
    return d


def selected(d, views):
    unknown = [v for v in views if v not in d['views']]
    if unknown:
        raise ValueError(f'Unknown cameras {unknown}; available: {list(d["views"])}')
    return views


def default_views(d):
    return ['agentview'] + [n for n in d['views'] if n == 'wrist' or n.startswith('wrist_')]


async def arm_metadata(ctx):
    if len(getattr(ctx, 'arms', ())) > 1:
        r = await motion.robot()
        return [text(packed({'arms': {v['side']: a for a, v in r.get('arms', {}).items()},
                             'wrist_cameras': r.get('wrist_cameras', {})}))]
    return []


def image_blocks(d, views, *, depth=True, raw=False):
    blocks = [text(packed({'frame_id':d['frame_id'], **({'camera_frame':d['camera_frame'],
                       'robot_frame':d['robot_frame'],'depth_encoding':d['depth_encoding']} if depth else {})}))]
    for name in views:
        v = d['views'][name]
        meta = {'view':name,'resolution':v['native_resolution'] if raw else v['resolution']}
        if depth:
            meta.update(K=v['K'],T_robot_from_camera=v['T_robot_from_camera'])
        blocks.append(text(packed(meta)))
        blocks.extend([text(f'view: {name}_rgb'),types.ImageContent(type='image',
                       data=v['raw'] if raw else v['rgb'],mimeType='image/jpeg')])
        if depth:
            blocks.extend([text(f'view: {name}_depth'),types.ImageContent(type='image',data=v['depth'],mimeType='image/png')])
    return blocks


class ObservationContext:
    def __init__(self,ctx):
        object.__setattr__(self,'inner',ctx)
    def __getattr__(self,k):
        return getattr(self.inner,k)
    def __setattr__(self,k,v):
        setattr(self.inner,k,v)
    async def snap(self,note=None):
        d = await snapshot()
        views = selected(d,getattr(self.inner,'_rgbd_views',default_views(d)))
        return ([text(note)] if note else []) + await arm_metadata(self.inner) + image_blocks(d,views)


class Observe(motion.Tool):
    name='observe'
    description='Read synchronized calibrated RGB-D for selected cameras at224x224; remembers this ordered selection for action feedback. No movement. Defaults agentview and all wrist cameras; includes wrist-to-arm mapping for bimanual tasks.'
    input_schema={'type':'object','properties':{'views':{'type':'array','items':{'type':'string'},'minItems':1,'maxItems':9,'uniqueItems':True}},'additionalProperties':False}
    async def run(self,ctx,args):
        d=await snapshot()
        views=selected(d,args.get('views',default_views(d)))
        ctx._rgbd_views=list(views)
        return await arm_metadata(ctx) + image_blocks(d,views)


class CameraRaw(motion.Tool):
    name='camera_raw'
    description='Read one native RGB camera image, normally448x448, no resize/depth/UI. Does not change selected observation cameras.'
    input_schema={'type':'object','properties':{'view':{'type':'string'}},'required':['view'],'additionalProperties':False}
    async def run(self,ctx,args):
        d=await snapshot()
        return image_blocks(d,selected(d,[args['view']]),depth=False,raw=True)


class CameraViews(motion.Tool):
    name='camera_views'
    description='Survey all available cameras as individually labeled224x224 RGB images in stable order. No depth; does not change action-feedback selection.'
    async def run(self,ctx,args):
        d=await snapshot()
        return await arm_metadata(ctx) + [text(packed({'available_views':list(d['views'])}))]+image_blocks(d,list(d['views']),depth=False)


class SetPose(motion.SetPose):
    description=motion.SetPose.description.replace('screenshot','selected calibrated RGB-D images')
    async def run(self,ctx,args):
        return await super().run(ObservationContext(ctx),args)


class SetGripper(motion.SetGripper):
    description=motion.SetGripper.description.replace('screenshot','selected calibrated RGB-D images')
    async def run(self,ctx,args):
        return await super().run(ObservationContext(ctx),args)


class EndEpisode(motion.Tool):
    name='end_episode'
    description='Declare completion and end the episode. Returns confirmation only; no grading or success signal.'
    async def run(self,ctx,args):
        await send_command({'done': True, 'done_by': 'agent'})
        return [text(EPISODE_OVER)]


TOOLS = {t.name:t for t in (Observe(),CameraRaw(),CameraViews(),motion.GetProprio(),SetPose(),SetGripper(),EndEpisode())}
