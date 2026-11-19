"""Execute a recorded image treatment recipe, carrying image and subject state."""
import gc
import hashlib
import io
import json
import math
import random
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from .composition_study import candidate_specs, render_candidate
from .external_pipeline import postprocess, run_process, stylize
from .io import letterbox, sha256, write_json
from .presets import PRESETS, parameters, render_preset
from .render import TARGETS, render
from .segmentation import Segmenter

COMPOSITIONS = {'crop_center', 'crop_thirds', 'crop_thirds_color',
                'poster_center', 'poster_thirds', 'poster_airy', 'color_only'}
COLLAGE_OPERATIONS = {'duotone_collage', 'original_collage'}
OPERATIONS = set(PRESETS) | COMPOSITIONS | COLLAGE_OPERATIONS | {'stylize', 'material_reconstruct', 'dragon_move'}
MATERIAL_SEEDS = {'newspaper_a': 42, 'newspaper_b': 43, 'newspaper_c': 44}
TEXTURE_IDS = {'wood_table', 'rough_linen', 'ceramic_tiles', 'silk_satin',
               'ceramic_iznik', 'silk_brocade', 'ceramic_blue_white', 'stained_glass'}
MATERIAL_IDS = set(MATERIAL_SEEDS) | TEXTURE_IDS


def validate_recipe(recipe):
    if not isinstance(recipe, dict) or recipe.get('schema_version') != 1:
        raise ValueError('Recipe requires schema_version=1')
    if 'subject_count' in recipe and (type(recipe['subject_count']) is not int or not 0 <= recipe['subject_count'] <= 10):
        raise ValueError('Confirmed subject_count must be an integer in [0, 10]')
    steps = recipe.get('steps')
    if not isinstance(steps, list) or not 1 <= len(steps) <= 3:
        raise ValueError('Recipe must contain one to three steps')
    for step in steps:
        if not isinstance(step, dict) or step.get('op') not in OPERATIONS:
            raise ValueError(f'Unknown treatment: {step!r}')
        if step['op'] == 'stylize' and not isinstance(step.get('reference_id'), str):
            raise ValueError('Stylize requires a fixed reference_id')
        if step['op'] == 'stylize':
            for key, default in (('style_strength', 1.0), ('structure_strength', .8)):
                value = step.get(key, default)
                if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1.5:
                    raise ValueError(f'{key} must be in [0, 1.5]')
        if step['op'] == 'material_reconstruct' and step.get('material_id') not in MATERIAL_IDS:
            raise ValueError(f'Material ID must be one of {sorted(MATERIAL_IDS)}')
        if step['op'] == 'material_reconstruct':
            strength = step.get('material_strength', 1.0)
            if type(strength) not in (int, float) or not math.isfinite(strength) or not 0 <= strength <= 1:
                raise ValueError('material_strength must be in [0, 1]')
        if step['op'] == 'dragon_move':
            if any(type(step.get(axis)) is not int or not -256 <= step[axis] <= 256 for axis in ('dx', 'dy')):
                raise ValueError('Dragon move requires integer dx and dy in [-256, 256]')
            if type(step.get('steps', 50)) is not int or not 1 <= step.get('steps', 50) <= 50:
                raise ValueError('Dragon move steps must be in [1, 50]')
        if step['op'] in PRESETS:
            strength = step.get('strength', 1.0)
            if type(strength) not in (int, float) or not math.isfinite(strength) or not 0 <= strength <= 1:
                raise ValueError('Preset strength must be between zero and one')
        if step['op'] in COLLAGE_OPERATIONS:
            if step.get('composition', 'center') not in TARGETS or step.get('background_mode', 'source_masked') not in {'solid', 'source_masked'}:
                raise ValueError('Invalid collage composition or background mode')
            for key, default, minimum, maximum in (('scale', .5, .3, .8), ('texture_fraction', .25, 0, 1)):
                value = step.get(key, default)
                if type(value) not in (int, float) or not math.isfinite(value) or not minimum <= value <= maximum:
                    raise ValueError(f'{key} must be in [{minimum}, {maximum}]')
        if 'prompt' in step and (not isinstance(step['prompt'], str) or len(step['prompt']) > 500):
            raise ValueError('Prompt must be a string of at most 500 characters')
    return steps


def validate_route(steps, subject_count):
    if subject_count == 0:
        if any(step['op'] not in {'stylize', 'material_reconstruct'} for step in steps):
            raise ValueError('Without a subject, use only stylize/material steps')
        return 'no_subject'
    if subject_count > 1:
        return 'multiple_subjects'
    return 'one_subject'


def style_reference(root, reference_id):
    source = json.loads((root/'data/style_refs/SOURCE.json').read_text(encoding='utf-8'))
    record = source['references'].get(reference_id)
    if record is None:
        raise ValueError(f'Unknown style reference: {reference_id}')
    image = root/'data/style_refs'/f'{reference_id}.jpg'
    if not image.is_file() or sha256(image) != record['sha256']:
        raise ValueError(f'Missing or modified style reference: {reference_id}')
    return image, record


def material_reference(root, material_id):
    if material_id in TEXTURE_IDS:
        folder = root/'data/material_refs'
        source = json.loads((folder/'SOURCE.json').read_text(encoding='utf-8'))['references'][material_id]
        if 'tiles' in source:
            for item in source['sources']:
                path = folder/item['file']
                if not path.is_file() or sha256(path) != item['sha256']:
                    raise ValueError(f'Missing or modified material source: {path}')
            tiles = []
            for item in source['tiles']:
                path = folder/item['file']
                if not path.is_file() or sha256(path) != item['sha256']:
                    raise ValueError(f'Missing or modified material tile: {path}')
                tiles.append(path.read_bytes())
            return tiles, {'material_id': material_id,
                           'source_sha256': [item['sha256'] for item in source['sources']],
                           'tile_sha256': [item['sha256'] for item in source['tiles']],
                           'asset_url': source['asset_url'], 'license': source['license']}
        path = folder/f'{material_id}.jpg'
        if not path.is_file() or sha256(path) != source['sha256']:
            raise ValueError(f'Missing or modified material reference: {material_id}')
        with Image.open(path) as raw:
            image = raw.convert('RGB')
        side = min(image.size)//2
        tiles = []
        for left, top in ((0, 0), (image.width-side, 0), (0, image.height-side),
                          (image.width-side, image.height-side)):
            stream = io.BytesIO()
            image.crop((left, top, left+side, top+side)).save(stream, format='PNG')
            tiles.append(stream.getvalue())
        return tiles, {'material_id': material_id, 'source_sha256': source['sha256'],
                       'tile_sha256': [hashlib.sha256(tile).hexdigest() for tile in tiles],
                       'asset_url': source['asset_url'], 'license': source['license']}
    folder = root/'third_party/repos/CollageRL/samples/materials/newspaper'
    source = json.loads((folder/'SOURCE.json').read_text(encoding='utf-8'))
    selected = random.Random(MATERIAL_SEEDS[material_id]).sample(source['images'], 8)
    paths = [folder/item['file'] for item in selected]
    if any(not path.is_file() or sha256(path) != item['sha256'] for path, item in zip(paths, selected)):
        raise ValueError(f'Missing or modified material reference: {material_id}')
    return [path.read_bytes() for path in paths], {'material_id': material_id,
        'selected_files': [item['file'] for item in selected], 'archive_sha256': source['archive_sha256']}


def _release_npu():
    gc.collect()
    import torch
    if hasattr(torch, 'npu') and torch.npu.is_available():
        torch.npu.empty_cache()


def _segment(root, config, image, mapping):
    segmenter = Segmenter(config, root)
    valid = np.ones((image.height, image.width), dtype=bool)
    mask, overlay, info = segmenter.predict(image, valid, mapping)
    del segmenter
    _release_npu()
    return mask, overlay, info


def blend_material(base, reconstructed, strength):
    return Image.blend(base, reconstructed, strength)


def run_sequence(root, image_path, recipe, output_dir, progress=lambda message: None,
                 initial_state=None, subject_selection_mode=None):
    root, image_path, output_dir = Path(root).resolve(), Path(image_path).resolve(), Path(output_dir).resolve()
    steps = validate_recipe(recipe)
    if output_dir.exists():
        raise FileExistsError(f'Output directory already exists: {output_dir}')
    config = yaml.safe_load((root/'configs/dev.yaml').read_text(encoding='utf-8'))
    if subject_selection_mode is not None:
        config['subject_selection_mode'] = subject_selection_mode
    if config['device'] != 'npu:3':
        raise ValueError('This local sequence executor is fixed to NPU 3')
    image, valid, mapping = letterbox(image_path, 512)
    output_dir.mkdir(parents=True)
    source_path = output_dir/'source.png'
    image.save(source_path)
    record = {'schema_version': 1, 'status': 'running', 'device': 'npu:3',
              'source': str(image_path), 'source_sha256': sha256(image_path),
              'recipe': recipe, 'steps': [], 'initial_mapping': mapping}
    write_json(output_dir/'sequence.json', record)
    try:
        progress('SAM2: analyzing source image')
        if initial_state is None:
            segmenter = Segmenter(config, root)
            mask, overlay, info = segmenter.predict(image, valid, mapping)
            del segmenter
            _release_npu()
        else:
            mask, overlay, info = initial_state
        overlay.save(output_dir/'detections.png')
        record['initial_segmentation'] = info
        count = recipe.get('subject_count', info['subject_count'])
        record['subject_count_used'] = count
        record['subject_count_source'] = 'confirmed_recipe' if 'subject_count' in recipe else 'sam2_estimate'
        route = validate_route(steps, count)
        record['route'] = route
        write_json(output_dir/'sequence.json', record)
        for index, step in enumerate(steps, 1):
            op = step['op']
            if op in PRESETS and PRESETS[op]['requires_subject'] and mask is None:
                raise ValueError(f'{op} requires a current subject mask')
            if op in COMPOSITIONS | COLLAGE_OPERATIONS and mask is None:
                raise ValueError(f'{op} requires a current subject mask')
            if op == 'dragon_move' and mask is None:
                raise ValueError('dragon_move requires a current subject mask')
            folder = output_dir/f'step_{index:02d}_{op}'
            folder.mkdir()
            current = folder/'source.png'
            image.save(current)
            step_record = {'index': index, 'operation': step, 'input_sha256': sha256(current),
                           'mapping_before': mapping, 'mask_before': mask is not None,
                           'status': 'running'}
            record['steps'].append(step_record)
            write_json(output_dir/'sequence.json', record)
            if mask is not None:
                Image.fromarray(mask.astype(np.uint8)*255).save(folder/'mask_before.png')
            progress(f'{index}/{len(steps)}: {op}')
            if op in PRESETS:
                params = parameters(op, step.get('strength', 1.0), 512,
                                    mapping['content_box_xyxy'], step.get('position', 'center'))
                image, visible, metrics = render_preset(image, mask, params)
                mask = np.asarray(visible) > 0 if mask is not None else None
                step_record.update(parameters=params, metrics=metrics)
                if op == 'minimal_poster':
                    mapping = _full_mapping(mapping)
            elif op in COMPOSITIONS:
                candidates = {key: params for key, _, params in
                              candidate_specs(mask, mapping['content_box_xyxy'], 512)}
                params = candidates[op]
                if 'unavailable' in params:
                    raise ValueError(f'{op}: {params["unavailable"]}')
                image, visible, metrics = render_candidate(image, mask, params)
                mask = np.asarray(visible) > 0
                mapping = _full_mapping(mapping)
                step_record.update(parameters=params, metrics=metrics)
            elif op in COLLAGE_OPERATIONS:
                palettes = {item['name']: item for item in config['palettes']}
                palette_name = step.get('palette', next(iter(palettes)))
                if palette_name not in palettes:
                    raise ValueError(f'Unknown collage palette: {palette_name}')
                composition = step.get('composition', 'center')
                style = 'duotone' if op == 'duotone_collage' else 'original'
                background_mode = step.get('background_mode', 'source_masked')
                params = {'schema_version': 1, 'renderer_version': f'{style}-v2' if background_mode == 'source_masked' else f'{style}-v1',
                    'style': style, 'seed': config['seed'], 'attempt_index': 0,
                    'composition': composition, 'target_xy': list(TARGETS[composition]),
                    'center_xy': list(TARGETS[composition]), 'longest_edge_fraction': step.get('scale', .5),
                    'texture_fraction': step.get('texture_fraction', .25), 'palette': palettes[palette_name],
                    'canvas_size': 512, 'layer_order': ['source_image', 'source_subject_mask', 'main_instance'] if background_mode == 'source_masked' else ['solid_background', 'main_instance'],
                    'main_instance_id': info['selected']['instance_id'] if info.get('selected') else 0,
                    'minimum_visible_fraction': config['min_visible_fraction'],
                    'background_mode': background_mode, 'mask_fill': [245, 237, 223]}
                image, visible, metrics = render(image, mask, params)
                if not metrics['valid']:
                    raise ValueError(f'Invalid collage geometry: {metrics["failures"]}')
                mask = np.asarray(visible) > 0
                mapping = _full_mapping(mapping)
                step_record.update(parameters=params, metrics=metrics)
            elif op == 'stylize':
                reference, source = style_reference(root, step['reference_id'])
                stylize(root, folder, {'style_data': reference.read_bytes(), 'prompt': step.get('prompt', 'a detailed image, high quality'),
                    'style_strength': step.get('style_strength', 1.0),
                    'structure_strength': step.get('structure_strength', .8), 'size': 512}, progress)
                image = Image.open(folder/'output.png').convert('RGB')
                mapping = _full_mapping(mapping)
                step_record['reference'] = {'id': step['reference_id'], **source}
                mask, overlay, info = _segment(root, config, image, mapping)
                overlay.save(folder/'detections_after.png')
                step_record['segmentation_after'] = info
            elif op == 'material_reconstruct':
                materials, reference = material_reference(root, step['material_id'])
                shutil.copyfile(current, folder/'output.png')
                step_record['reference'] = reference
                step_record['postprocess'] = postprocess(root, folder, materials, progress)
                with Image.open(folder/'output.png') as raw:
                    reconstructed = raw.convert('RGB')
                strength = step.get('material_strength', 1.0)
                image = blend_material(image, reconstructed, strength)
                if strength < 1:
                    image.save(folder/'output.png')
                step_record['material_blend'] = {'strength': strength, 'scope': 'full_image'}
                mapping = _full_mapping(mapping)
                mask, overlay, info = _segment(root, config, image, mapping)
                overlay.save(folder/'detections_after.png')
                step_record['segmentation_after'] = info
            else:
                command = [sys.executable, str(root/'third_party/repos/DragonDiffusion/run_local.py'),
                    '--image', str(current), '--mask', str(folder/'mask_before.png'),
                    '--prompt', step.get('prompt', 'a photograph'), '--dx', str(step['dx']),
                    '--dy', str(step['dy']), '--steps', str(step.get('steps', 50)),
                    '--seed', str(step.get('seed', 42)), '--output', str(folder/'dragon')]
                run_process(command, root, folder/'dragon.log', progress)
                shutil.copyfile(folder/'dragon/output.png', folder/'output.png')
                image = Image.open(folder/'dragon/output.png').convert('RGB')
                step_record['dragon_metrics'] = json.loads((folder/'dragon/metrics.json').read_text(encoding='utf-8'))
                mapping = _full_mapping(mapping)
                mask, overlay, info = _segment(root, config, image, mapping)
                overlay.save(folder/'detections_after.png')
                step_record['segmentation_after'] = info
            if op in PRESETS or op in COMPOSITIONS or op in COLLAGE_OPERATIONS:
                image.save(folder/'output.png')
            if mask is not None:
                Image.fromarray(mask.astype(np.uint8)*255).save(folder/'mask_after.png')
            step_record['output_sha256'] = sha256(folder/'output.png')
            step_record['mapping_after'] = mapping
            step_record['mask_after'] = mask is not None
            step_record['status'] = 'passed'
            write_json(output_dir/'sequence.json', record)
        shutil.copyfile(folder/'output.png', output_dir/'output.png')
        record['status'] = 'passed'
        record['output_sha256'] = sha256(output_dir/'output.png')
    except Exception:
        record['status'] = 'failed'
        record['error'] = traceback.format_exc()
        if record['steps'] and record['steps'][-1]['status'] == 'running':
            record['steps'][-1]['status'] = 'failed'
        raise
    finally:
        write_json(output_dir/'sequence.json', record)
    return record


def _full_mapping(mapping):
    result = dict(mapping)
    result['content_box_xyxy'] = [0, 0, 512, 512]
    result['offset_xy'] = [0, 0]
    result['resized_size'] = [512, 512]
    return result
