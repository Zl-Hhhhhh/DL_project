import hashlib
import itertools
import json
import math
import random

import numpy as np
from PIL import Image

TARGETS = {'center': (0.50, 0.50), 'left': (0.32, 0.50), 'right': (0.68, 0.50)}


def task_seed(seed, source_id, composition, style='duotone'):
    digest = hashlib.sha256(f'{seed}:{source_id}:{style}:{composition}'.encode()).digest()
    return int.from_bytes(digest[:4], 'big')


def parameter_table(seed, config, composition):
    if composition not in TARGETS:
        raise ValueError(f'Unknown composition: {composition}')
    style = config.get('render_style', 'duotone')
    if style not in {'duotone', 'original'}:
        raise ValueError(f'Unknown style: {style}')
    background_mode = config.get('background_mode', 'solid')
    if background_mode not in {'solid', 'source_masked'}:
        raise ValueError(f'Unknown background mode: {background_mode}')
    textures = [1.0] if style == 'original' else [0.0, 0.25, 0.50]
    table = list(itertools.product([0.50, 0.65], [-0.04, 0.0, 0.04], textures, range(len(config['palettes']))))
    random.Random(seed).shuffle(table)
    # Interleave scale levels so the initial candidate pool includes both.
    small = [row for row in table if row[0] == 0.50]
    large = [row for row in table if row[0] == 0.65]
    table = [row for pair in zip(small, large) for row in pair]
    target = TARGETS[composition]
    for index, (scale, dx, texture, palette_id) in enumerate(table[:config['max_attempts']]):
        params = {
            'schema_version': 1, 'renderer_version': f'{style}-v1', 'style': style,
            'seed': seed, 'attempt_index': index, 'composition': composition,
            'target_xy': list(target), 'center_xy': [target[0] + dx, target[1]],
            'longest_edge_fraction': scale, 'texture_fraction': texture,
            'palette': config['palettes'][palette_id], 'canvas_size': config['canvas_size'],
            'layer_order': ['solid_background', 'main_instance'], 'main_instance_id': 0,
            'minimum_visible_fraction': config['min_visible_fraction'],
        }
        if background_mode == 'source_masked':
            params.update(renderer_version=f'{style}-v2', background_mode=background_mode,
                          mask_fill=[245, 237, 223],
                          layer_order=['source_image', 'source_subject_mask', 'main_instance'])
        if config['canvas_size'] == 1024:
            params['renderer_version'] = f'{style}-v3'
        yield params


def render(image, mask, params):
    """Pure renderer: same pixels, mask and JSON parameters give identical output."""
    rgb = np.asarray(image.convert('RGB'))
    mask = np.asarray(mask) > 0
    if mask.shape != rgb.shape[:2]:
        raise ValueError('Image/mask size mismatch')
    if not mask.any():
        raise ValueError('empty_mask')
    background_mode = params.get('background_mode', 'solid')
    size = params['canvas_size']
    version = 'v3' if size == 1024 else ('v2' if background_mode == 'source_masked' else 'v1')
    if (background_mode not in {'solid', 'source_masked'} or params['style'] not in {'duotone', 'original'}
            or params['renderer_version'] != f"{params['style']}-{version}"):
        raise ValueError('Unsupported renderer version or style')
    if size not in {512, 1024}:
        raise ValueError('Supported canvas sizes are 512 and 1024')
    composition = params['composition']
    if composition not in TARGETS:
        raise ValueError('Unknown composition')
    ys, xs = np.where(mask)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    x0, y0, x1, y1 = bbox
    scale = size * params['longest_edge_fraction'] / max(x1 - x0, y1 - y0)
    width, height = max(1, round((x1-x0)*scale)), max(1, round((y1-y0)*scale))
    crop = np.asarray(Image.fromarray(rgb[y0:y1, x0:x1]).resize((width, height), Image.Resampling.LANCZOS), dtype=np.float32)
    crop_mask = Image.fromarray((mask[y0:y1, x0:x1] * 255).astype(np.uint8)).resize((width, height), Image.Resampling.NEAREST)
    alpha = np.asarray(crop_mask) > 0
    palette = params['palette']
    if params['style'] == 'original':
        texture = 1.0
        painted = crop.astype(np.uint8)
    else:
        luminance = (crop @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)) / 255.0
        dark, light = np.array(palette['dark']), np.array(palette['light'])
        duotone = dark + luminance[:, :, None] * (light - dark)
        texture = params['texture_fraction']
        painted = np.rint(np.clip((1-texture)*duotone + texture*crop, 0, 255)).astype(np.uint8)
    left = round(params['center_xy'][0] * size - width/2)
    top = round(params['center_xy'][1] * size - height/2)
    if background_mode == 'source_masked':
        if rgb.shape[:2] != (size, size):
            raise ValueError('Source background must match canvas size')
        background = rgb.copy()
        background[mask] = params['mask_fill']
        output = Image.fromarray(background)
    else:
        output = Image.new('RGB', (size, size), tuple(palette['background']))
    output.paste(Image.fromarray(painted), (left, top), crop_mask)
    visible_image = Image.new('L', (size, size), 0)
    visible_image.paste(crop_mask, (left, top))
    visible = np.asarray(visible_image) > 0
    before_area, after_area = int(alpha.sum()), int(visible.sum())
    visible_fraction = after_area / max(1, before_area)
    vy, vx = np.where(visible)
    center = [float(vx.mean()/size), float(vy.mean()/size)] if after_area else None
    blank_pixels = int(visible[:, int(size*0.75):].sum()) if composition == 'left' else (int(visible[:, :int(size*0.25)].sum()) if composition == 'right' else 0)
    position_error = math.dist(center, params['target_xy']) if center is not None else 1.0
    failures = []
    if not after_area:
        failures.append('empty_main_instance')
    if visible_fraction < params['minimum_visible_fraction']:
        failures.append('insufficient_visible_area')
    if blank_pixels:
        failures.append('blank_region_intrusion')
    metrics = {
        'valid': not failures, 'failures': failures, 'source_bbox_xyxy': bbox,
        'transformed_size': [width, height], 'paste_xy': [left, top],
        'transformed_area': before_area, 'visible_area': after_area,
        'visible_fraction': visible_fraction, 'visible_centroid_xy': center,
        'position_error': position_error, 'blank_intrusion_pixels': blank_pixels,
        'blank_intrusion_fraction': blank_pixels/max(1, after_area),
        'texture_fraction': texture,
    }
    return output, visible_image, metrics


def generate_candidates(image, mask, config, source_id, composition):
    seed = task_seed(config['seed'], source_id, composition, config.get('render_style', 'duotone'))
    accepted, rejected = [], []
    for params in parameter_table(seed, config, composition):
        output, visible, metrics = render(image, mask, params)
        if metrics['valid']:
            accepted.append({'candidate_id': f'c{len(accepted):02d}', 'params': params, 'image': output, 'visible_mask': visible, 'metrics': metrics})
        else:
            rejected.append({'params': params, 'metrics': metrics})
        if len(accepted) == config['candidates_per_task']:
            break
    # Fixed lexicographic baseline; no learned or aesthetic score.
    selected = min(accepted, key=lambda item: (item['metrics']['position_error'], -item['metrics']['visible_fraction'], abs(item['params']['texture_fraction']-0.25), item['candidate_id'])) if accepted else None
    return accepted, rejected, selected
