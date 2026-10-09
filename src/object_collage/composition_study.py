"""Deterministic composition experiments, separate from the frozen baseline."""
import math

import cv2
import numpy as np
from PIL import Image, ImageEnhance

TARGETS = [(1/3, 1/3), (2/3, 1/3), (1/3, 2/3), (2/3, 2/3)]


def bounds(mask):
    ys, xs = np.where(mask)
    if not len(xs):
        raise ValueError('Empty subject mask')
    return [int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1]


def selective_color(image, mask, saturation):
    if not 0 <= saturation <= 1:
        raise ValueError('Background saturation must be between zero and one')
    original = np.asarray(image.convert('RGB'))
    output = np.array(ImageEnhance.Color(image.convert('RGB')).enhance(saturation))
    output[mask] = original[mask]
    return Image.fromarray(output)


def plan_crop(mask, content_box, targets, tolerance=0.045):
    """Find an in-frame square crop keeping the entire binary subject + margin.

    Reference point is the bounding-box centre, explicitly an approximation.
    Prefer target accuracy and then a larger crop; no aesthetic score is claimed.
    """
    x0, y0, x1, y1 = bounds(mask)
    cx, cy = (x0+x1)/2, (y0+y1)/2
    l, t, r, b = content_box
    margin = max(2, round(min(r-l, b-t)*0.015))
    smallest = max(x1-x0, y1-y0) + 2*margin
    biggest = min(r-l, b-t)
    choices = []
    for side in range(biggest, smallest-1, -1):
        low_x, high_x = max(l, x1+margin-side), min(r-side, x0-margin)
        low_y, high_y = max(t, y1+margin-side), min(b-side, y0-margin)
        if low_x > high_x or low_y > high_y:
            continue
        for tx, ty in targets:
            left = int(np.clip(round(cx-tx*side), low_x, high_x))
            top = int(np.clip(round(cy-ty*side), low_y, high_y))
            error = math.dist(((cx-left)/side, (cy-top)/side), (tx, ty))
            if error <= tolerance:
                choices.append((error, -side, left, top, tx, ty))
    if not choices:
        return None
    error, neg_side, left, top, tx, ty = min(choices)
    side = -neg_side
    return {'box': [left, top, left+side, top+side], 'target_xy': [tx, ty],
            'source_reference': 'subject_bbox_center', 'position_error_before_resize': error,
            'subject_retained_fraction': 1.0, 'margin_pixels': margin,
            'target_tolerance': tolerance}


def render_candidate(image, mask, params):
    image = image.convert('RGB')
    mask = np.asarray(mask) > 0
    if mask.shape != (image.height, image.width):
        raise ValueError('Image/mask size mismatch')
    bounds(mask)
    size = params['size']
    if size not in (512, 1024):
        raise ValueError('Only 512 and 1024 are supported')
    mode = params['mode']
    if mode == 'poster':
        x0, y0, x1, y1 = bounds(mask)
        scale = size*params['longest_edge_fraction']/max(x1-x0, y1-y0)
        width, height = max(1, round((x1-x0)*scale)), max(1, round((y1-y0)*scale))
        patch = image.crop((x0, y0, x1, y1)).resize((width, height), Image.Resampling.LANCZOS)
        alpha = Image.fromarray(mask[y0:y1, x0:x1].astype(np.uint8)*255).resize((width, height), Image.Resampling.NEAREST)
        tx, ty = params['target_xy']
        left, top = round(tx*size-width/2), round(ty*size-height/2)
        if left < 0 or top < 0 or left+width > size or top+height > size:
            raise ValueError('Poster would clip the subject')
        output = Image.new('RGB', (size, size), tuple(params['background_rgb']))
        output.paste(patch, (left, top), alpha)
        visible = Image.new('L', (size, size))
        visible.paste(alpha, (left, top))
        domain = np.ones((size, size), bool)
        reference = None
    elif mode in ('original', 'color', 'crop'):
        box = params.get('crop_box', [0, 0, image.width, image.height])
        reference = image.crop(box).resize((size, size), Image.Resampling.LANCZOS)
        visible = Image.fromarray(mask.astype(np.uint8)*255).crop(box).resize((size, size), Image.Resampling.NEAREST)
        output = selective_color(reference, np.asarray(visible)>0, params.get('background_saturation', 1.0))
        content = Image.new('L', image.size)
        l, t, r, b = params['content_box']
        content.paste(255, (l, t, r, b))
        domain = np.asarray(content.crop(box).resize((size, size), Image.Resampling.NEAREST)) > 0
    else:
        raise ValueError('Unknown experiment mode')
    visible_array = np.asarray(visible)>0
    rgb = np.asarray(output)
    x0, y0, x1, y1 = bounds(visible_array)
    anchor = [(x0+x1)/(2*size), (y0+y1)/(2*size)]
    target = params.get('target_xy')
    side = 'right' if (target or anchor)[0] <= 0.5 else 'left'
    region = np.zeros_like(visible_array)
    if side == 'right':
        region[:, round(size*.75):] = True
    else:
        region[:, :round(size*.25)] = True
    region &= domain
    bg = region & ~visible_array
    # Exclude a narrow subject boundary; detect texture in the actual background.
    eroded = cv2.erode(bg.astype(np.uint8), np.ones((5,5), np.uint8)) > 0
    edges = cv2.Canny(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 80, 160) > 0
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    metrics = {'anchor_kind': 'subject_bbox_center', 'anchor_xy': anchor,
               'target_error': math.dist(anchor, target) if target else None,
               'subject_area_fraction': float(visible_array.mean()),
               'negative_space_side': side,
               'negative_space_subject_fraction': float(visible_array[region].mean()) if region.any() else None,
               'negative_space_edge_density': float(edges[eroded].mean()) if eroded.any() else None,
               'negative_space_evaluated_fraction': float(region.mean()),
               'background_mean_saturation': float(hsv[domain & ~visible_array, 1].mean()/255) if (domain & ~visible_array).any() else None,
               'subject_rgb_exact_vs_same_geometry': bool(np.array_equal(rgb[visible_array], np.asarray(reference)[visible_array])) if reference is not None else None}
    return output, visible, metrics


def candidate_specs(mask, content_box, size):
    common = {'version': 'priority-study-v1', 'size': size, 'content_box': content_box}
    specs = [('original', 'Original', dict(mode='original')),
             ('color_only', 'Original + low-saturation background', dict(mode='color', background_saturation=.15))]
    for key, title, targets in [('crop_center', 'Crop: centre', [(.5,.5)]),
                                 ('crop_thirds', 'Crop: thirds', TARGETS)]:
        plan = plan_crop(mask, content_box, targets)
        if plan is None:
            specs.append((key, title, {'unavailable': 'No square crop retains the full subject and reaches the target'}))
        else:
            specs.append((key, title, dict(mode='crop', crop_box=plan['box'],
                                           target_xy=plan['target_xy'], crop_plan=plan,
                                           background_saturation=1.0)))
        if key == 'crop_thirds':
            specs.append(('crop_thirds_color', 'Same thirds crop + colour emphasis',
                          dict(specs[-1][2], background_saturation=.15)))
    for key, title, target, scale in [
        ('poster_center', 'Collage: centre / compact', [.5,.5], .55),
        ('poster_thirds', 'Collage: thirds / compact', [1/3,1/3], .55),
        ('poster_airy', 'Collage: thirds / more space', [1/3,1/3], .38),
    ]:
        specs.append((key, title, dict(mode='poster', target_xy=target,
                                       longest_edge_fraction=scale, background_rgb=[245,237,223])))
    return [(key, title, dict(common, **params)) for key, title, params in specs]
