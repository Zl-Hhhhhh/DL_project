"""Versioned, deterministic photo presets shared by UI and batch comparisons."""
import copy
import math

import numpy as np
from PIL import Image

from .composition_study import render_candidate, selective_color

POSITIONS = {'center':[.5,.5], 'top_left':[1/3,1/3], 'top_right':[2/3,1/3],
             'bottom_left':[1/3,2/3], 'bottom_right':[2/3,2/3]}
PRESETS = {
    'warm_film': {'name':'暖色胶片', 'description':'柔和对比，偏暖色调，保留原有构图。', 'requires_subject':False,
                  'color':{'saturation':.91, 'contrast':1.04, 'lift':.035, 'gain':[1.04,1.01,.94]}},
    'muted_vintage': {'name':'低饱和复古', 'description':'降低饱和度，抬起暗部，呈现褪色色调。', 'requires_subject':False,
                      'color':{'saturation':.60, 'contrast':.90, 'lift':.065, 'gain':[1.03,1.015,.96]}},
    'subject_pop': {'name':'主体突出', 'description':'主体位置与原色不变，降低背景饱和度。', 'requires_subject':True},
    'minimal_poster': {'name':'简洁海报', 'description':'移除原背景，保留原色主体与米白留白。', 'requires_subject':True},
}


def catalog():
    return [{'id':key, **{k:v for k,v in value.items() if k!='color'},
             'preview':f'/assets/preset_{key}.png'} for key,value in PRESETS.items()]


def parameters(preset_id, strength, size, content_box, position='center'):
    if preset_id not in PRESETS:
        raise ValueError('Unknown preset')
    if type(strength) not in (int, float) or not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('Preset strength must be between 0 and 1')
    if size not in (512, 1024) or position not in POSITIONS:
        raise ValueError('Unsupported size or position')
    return {'version':'photo-presets-v1', 'preset_id':preset_id, 'strength':strength,
            'size':size, 'content_box':list(content_box), 'position':position,
            'target_xy':list(POSITIONS[position]), 'background_rgb':[245,237,223],
            'color_parameters':copy.deepcopy(PRESETS[preset_id].get('color'))}


def render_preset(image, mask, params):
    if params.get('version') != 'photo-presets-v1':
        raise ValueError('Unsupported preset renderer version')
    key, strength, size = params['preset_id'], params['strength'], params['size']
    parameters(key, strength, size, params['content_box'], params['position'])
    if image.size != (size, size):
        raise ValueError('Preset source must match the output canvas')
    if PRESETS[key]['requires_subject'] and (mask is None or not np.asarray(mask).any()):
        raise ValueError('This preset needs a subject mask')
    if key == 'minimal_poster':
        # Strength controls whitespace, never crossfades detached subjects into their old position.
        output, visible, metrics = render_candidate(image, mask, {
            'mode':'poster', 'size':size, 'target_xy':params['target_xy'],
            'longest_edge_fraction':.60-.30*strength, 'background_rgb':params['background_rgb']})
        metrics['strength_meaning'] = 'whitespace; subject longest edge = 0.60 - 0.30 * strength'
    else:
        original = np.asarray(image.convert('RGB'))
        if key == 'subject_pop':
            output = selective_color(image, mask, 1-strength)
        elif strength == 0:
            output = image.convert('RGB').copy()
        else:
            color = params['color_parameters']
            rgb = original.astype(np.float32)/255
            gray = (rgb @ np.array([.2126,.7152,.0722],dtype=np.float32))[:,:,None]
            effect = gray+(rgb-gray)*color['saturation']
            effect = (effect-.5)*color['contrast']+.5
            effect = effect*(1-color['lift'])+color['lift']
            effect *= np.array(color['gain'],dtype=np.float32)
            result = np.rint(np.clip(rgb*(1-strength)+np.clip(effect,0,1)*strength,0,1)*255).astype(np.uint8)
            output = Image.fromarray(result)
        # Square-canvas padding is not part of the photo and remains unchanged.
        domain = np.zeros((size,size),bool)
        l,t,r,b = params['content_box']
        domain[t:b,l:r] = True
        rgb = np.asarray(output).copy()
        rgb[~domain] = original[~domain]
        output = Image.fromarray(rgb)
        visible = Image.fromarray((np.asarray(mask,dtype=bool)*255).astype(np.uint8)) if mask is not None else Image.new('L',(size,size))
        metrics = {'geometry_unchanged':True, 'padding_unchanged':True}
        if key == 'subject_pop':
            metrics['subject_rgb_exact'] = bool(np.array_equal(rgb[mask],original[mask]))
            metrics['background_saturation'] = 1-strength
    metrics.update(preset_id=key, strength=strength)
    return output, visible, metrics
