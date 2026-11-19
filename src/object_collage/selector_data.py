"""Frozen candidate batches for human preference labeling."""
import hashlib
import json
import random
import time
import traceback
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from .io import letterbox, read_json, sha256, validate_manifest, write_json
from .segmentation import Segmenter
from .sequence import _release_npu, run_sequence, validate_recipe, validate_route


def _move_for_mask(mask):
    if mask is None:
        return {'op': 'dragon_move', 'dx': 32, 'dy': 0, 'steps': 50, 'seed': 42}
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return {'op': 'dragon_move', 'dx': 32, 'dy': 0, 'steps': 50, 'seed': 42}
    center_x, center_y = (xs.min()+xs.max())/2, (ys.min()+ys.max())/2
    choices = [(-32, 0), (32, 0), (0, -32), (0, 32),
               (-16, 0), (16, 0), (0, -16), (0, 16)]
    feasible = [(dx, dy) for dx, dy in choices if 0 <= xs.min()+dx and xs.max()+dx < mask.shape[1]
                and 0 <= ys.min()+dy and ys.max()+dy < mask.shape[0]]
    dx, dy = min(feasible, key=lambda shift: ((center_x+shift[0]-256)**2+
                                               (center_y+shift[1]-256)**2, -abs(shift[0])-abs(shift[1]))) if feasible else (0, 0)
    return {'op': 'dragon_move', 'dx': dx, 'dy': dy, 'steps': 50, 'seed': 42}


def _reference_order(source_id):
    seed = int.from_bytes(hashlib.sha256(source_id.encode('utf-8')).digest()[:8], 'big')
    rng = random.Random(seed)
    styles = ['van_gogh_starry_night', 'da_vinci_mona_lisa',
              'van_gogh_sunflowers', 'cezanne_sainte_victoire']
    materials = ['newspaper_a', 'newspaper_b', 'newspaper_c', 'ceramic_blue_white', 'stained_glass']
    rng.shuffle(styles)
    rng.shuffle(materials)
    return seed, styles, materials


def _tempered_sample(rng, values, weights, count, temperature):
    remaining = list(zip(values, weights))
    chosen = []
    for _ in range(count):
        probabilities = [weight ** (1 / temperature) for _, weight in remaining]
        index = rng.choices(range(len(remaining)), weights=probabilities, k=1)[0]
        chosen.append(remaining.pop(index)[0])
    return chosen


def candidate_recipes(subject_count, source_id, mask=None, sampling_temperature=1.5):
    if not 0 < sampling_temperature <= 5:
        raise ValueError('Sampling temperature must be in (0, 5]')
    seed, styles, materials = _reference_order(source_id)
    rng = random.Random(seed ^ 0x5E1EC7)
    style = lambda index: {'op': 'stylize', 'reference_id': styles[index % len(styles)]}
    def material(index):
        material_id = materials[index % len(materials)]
        strength = {'ceramic_blue_white': .65, 'stained_glass': .6}.get(material_id, .45)
        return {'op': 'material_reconstruct', 'material_id': material_id,
                'material_strength': strength}
    move = _move_for_mask(mask)
    if subject_count == 0:
        sequences = ([[style(index)] for index in range(4)] +
                     [[material(index)] for index in range(5)] +
                     [[{**style(index), 'style_strength': strength}]
                      for strength in (.6, .85) for index in range(4)] +
                     [[{**style(index), 'structure_strength': .55}] for index in range(2)] +
                     [[material(index), style(index)] for index in range(5)] +
                     [[material(0), material(1)],
                      [material(0), style(0), material(1)],
                      [material(1), style(1), material(2)]])
    else:
        warm = {'op': 'warm_film', 'strength': .7}
        muted = {'op': 'muted_vintage', 'strength': .7}
        thirds = {'op': 'crop_thirds'}
        duotone = {'op': 'duotone_collage', 'composition': 'left'}
        exploration = [muted, {'op': 'subject_pop', 'strength': .7},
                       {'op': 'minimal_poster', 'strength': .7}, {'op': 'crop_center'},
                       thirds, {'op': 'crop_thirds_color'}, {'op': 'poster_center'},
                       {'op': 'poster_thirds'}, {'op': 'poster_airy'},
                       {'op': 'color_only'}, duotone,
                       {'op': 'original_collage', 'composition': 'right'}, move]
        exploration_weights = [.65, .65, .3, .4, .25, .8, .2,
                               .45, .2, .45, .6, .3, .55]
        selected = _tempered_sample(rng, exploration, exploration_weights,
                                    3 if subject_count > 1 else 2,
                                    sampling_temperature)
        singles = [[warm], [style(0)], [style(1)]] + [[step] for step in selected]
        double_variant = _tempered_sample(
            rng, [[warm, style(3)], [thirds, style(3)]], [.8, .6], 1,
            sampling_temperature)[0]
        doubles = [[material(0), warm], [material(1), warm],
                   [duotone, style(2)], double_variant]
        triples = [[duotone, muted, style(0)], [material(2), style(2), warm]]
        if subject_count == 1:
            triples.extend(_tempered_sample(
                rng, [[move, warm, style(3)], [thirds, style(3), muted],
                      [duotone, warm, style(3)]], [.45, .5, .7], 1,
                sampling_temperature))
        sequences = singles + doubles + triples
    if subject_count == 0:
        budgets = {1: 8, 2: 3, 3: 1}
        selected = []
        for length, budget in budgets.items():
            group = [steps for steps in sequences if len(steps) == length]
            anchors = (0, 4) if length == 1 else ()
            selected.extend(group[index] for index in anchors)
            choices = [steps for index, steps in enumerate(group) if index not in anchors]
            start = (seed >> (length*7)) % len(choices)
            selected.extend(choices[(start+index) % len(choices)]
                            for index in range(budget-len(anchors)))
        sequences = selected
    recipes = [{'schema_version': 1, 'steps': steps}
               for steps in sequences]
    for recipe in recipes:
        validate_route(validate_recipe(recipe), subject_count)
    return recipes


def _manifest(root):
    manifest = read_json(root/'data/photo120/manifest.json')
    validate_manifest(manifest)
    counts = {split: sum(row['split'] == split for row in manifest['images'])
              for split in ('train', 'val', 'test')}
    if counts != {'train': 80, 'val': 20, 'test': 20}:
        raise ValueError(f'photo120 split changed: {counts}')
    return manifest


def select_sources(manifest, split, limit=None, pilot20=False):
    if split == 'selector100':
        pilot_ids = {row['source_id'] for row in select_sources(manifest, 'train', pilot20=True)}
        return sorted((row for row in manifest['images'] if row['source_id'] not in pilot_ids),
                      key=lambda row: row['source_id'])
    rows = [row for row in manifest['images'] if split == 'all' or row['split'] == split]
    rows = sorted(rows, key=lambda row: row['source_id'])
    if pilot20:
        categories = sorted({row['category'] for row in rows})
        if len(categories) != 10:
            raise ValueError('Pilot requires ten frozen categories')
        rows = [row for category in categories
                for row in [item for item in rows if item['category'] == category][:2]]
        if len(rows) != 20:
            raise ValueError('Pilot requires two train sources per category')
    if limit is not None:
        rows = rows[:limit]
    return rows


def selector100_assignment(manifest, seed=2012):
    rows = select_sources(manifest, 'selector100')
    groups = {}
    for row in rows:
        groups.setdefault(row['category'], []).append(row['source_id'])
    if len(rows) != 100 or len(groups) != 10 or any(len(ids) != 10 for ids in groups.values()):
        raise ValueError('Expected ten remaining sources in each of ten photo120 categories')
    rng = random.Random(seed)
    assignment = {}
    for category in sorted(groups):
        ids = sorted(groups[category])
        rng.shuffle(ids)
        assignment.update({source_id: 'train' if index < 8 else 'val'
                           for index, source_id in enumerate(ids)})
    return assignment


def generate_batch(root, output_dir, split='train', limit=None, candidates_per_source=None,
                   progress=print, pilot20=False, source_offset=0,
                   sampling_temperature=1.5):
    root, output_dir = Path(root).resolve(), Path(output_dir).resolve()
    if split not in {'train', 'val', 'test', 'all', 'selector100'}:
        raise ValueError('Unknown split')
    if not 0 < sampling_temperature <= 5:
        raise ValueError('Sampling temperature must be in (0, 5]')
    if limit is not None and limit < 1:
        raise ValueError('limit must be positive')
    if source_offset < 0:
        raise ValueError('source_offset must be nonnegative')
    if candidates_per_source is not None and not 1 <= candidates_per_source <= 12:
        raise ValueError('candidates_per_source must be in [1, 12]')
    if pilot20 and (split != 'train' or candidates_per_source is not None):
        raise ValueError('pilot20 uses twelve candidates from each selected train source')
    config = yaml.safe_load((root/'configs/dev.yaml').read_text(encoding='utf-8'))
    if config['device'] != 'npu:3':
        raise ValueError('Candidate generation is fixed to NPU 3')
    manifest = _manifest(root)
    assignment = selector100_assignment(manifest) if split == 'selector100' else None
    rows = select_sources(manifest, split, pilot20=pilot20)[source_offset:]
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError('Source selection is empty')
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_path = output_dir/'batch.json'
    if batch_path.exists():
        batch = read_json(batch_path)
        if (batch['source_ids'] != [row['source_id'] for row in rows] or
                batch['split_filter'] != split or
                batch['candidates_per_source'] != candidates_per_source or
                batch.get('pilot20', False) != pilot20 or
                batch.get('source_offset', 0) != source_offset or
                batch.get('sampling_temperature', 1.5) != sampling_temperature or
                batch.get('split_assignment') != assignment):
            raise ValueError('Existing batch has different source selection or candidate budget')
        if batch.get('candidate_plan_version') != 14 and len(batch['sources']) < len(batch['source_ids']):
            raise ValueError('Existing partial batch uses an earlier candidate plan; start a new run')
        for entry in batch['sources'].values():
            entry.setdefault('subject_selection_mode', 'sam2_automatic')
        batch['subject_selection_mode_for_new_sources'] = config['subject_selection_mode']
        write_json(batch_path, batch)
    else:
        batch = {'schema_version': 1, 'dataset': 'photo120', 'split_filter': split,
                 'source_ids': [row['source_id'] for row in rows],
                 'candidates_per_source': candidates_per_source,
                 'source_offset': source_offset,
                 'pilot20': pilot20, 'sources': {}, 'candidate_plan_version': 14,
                 'sampling_temperature': sampling_temperature,
                 'split_assignment': assignment, 'split_seed': 2012 if assignment else None,
                 'subject_selection_mode_for_new_sources': config['subject_selection_mode'],
                 'selection_rule': 'SHA-frozen photo120; deterministic recipes; SAM2 estimated route'}
        write_json(batch_path, batch)
    for row in rows:
        source_id = row['source_id']
        image_path = root/row['path']
        if sha256(image_path) != row['sha256']:
            raise ValueError(f'Source hash mismatch: {source_id}')
        entry = batch['sources'].get(source_id)
        if entry is None:
            progress(f'{source_id}: SAM2 route estimate')
            route_start = time.perf_counter()
            image, valid, mapping = letterbox(image_path, 512)
            source_mode = batch['subject_selection_mode_for_new_sources']
            source_config = {**config, 'subject_selection_mode': source_mode}
            segmenter = Segmenter(source_config, root)
            mask, overlay, info = segmenter.predict(image, valid, mapping)
            del segmenter
            _release_npu()
            count = info['subject_count']
            recipes = candidate_recipes(count, source_id, mask, sampling_temperature)
            if candidates_per_source is not None:
                recipes = recipes[:candidates_per_source]
            entry = {'source_id': source_id,
                     'split': assignment[source_id] if assignment else row['split'],
                     'dataset_split': row['split'], 'source_path': row['path'],
                     'source_sha256': row['sha256'], 'estimated_subject_count': count,
                     'candidate_plan_version': 14,
                     'subject_selection_mode': source_mode,
                     'subject_count_source': 'sam2_estimate',
                     'route_elapsed_seconds': round(time.perf_counter()-route_start, 3),
                     'segmentation': info, 'candidates': [
                         {'candidate_id': f'c{index:02d}', 'recipe': recipe, 'status': 'pending',
                          'relative_dir': f'{source_id}/c{index:02d}'}
                         for index, recipe in enumerate(recipes)]}
            batch['sources'][source_id] = entry
            write_json(batch_path, batch)
        else:
            source_mode = entry['subject_selection_mode']
            source_config = {**config, 'subject_selection_mode': source_mode}
            info = entry['segmentation']
            overlay = mask = None
            for previous in entry['candidates']:
                previous_dir = output_dir/previous['relative_dir']
                detection_path = previous_dir/'detections.png'
                if previous['status'] == 'passed' and detection_path.is_file():
                    with Image.open(detection_path) as raw:
                        overlay = raw.convert('RGB')
                    first_op = previous['recipe']['steps'][0]['op']
                    mask_path = previous_dir/f'step_01_{first_op}'/'mask_before.png'
                    if mask_path.is_file():
                        with Image.open(mask_path) as raw:
                            mask = np.asarray(raw.convert('L')) > 0
                    break
            if overlay is None or (entry['estimated_subject_count'] > 0 and mask is None):
                image, valid, mapping = letterbox(image_path, 512)
                segmenter = Segmenter(source_config, root)
                mask, overlay, current_info = segmenter.predict(image, valid, mapping)
                del segmenter
                _release_npu()
                if current_info['subject_count'] != entry['estimated_subject_count']:
                    raise ValueError(f'SAM2 route changed for {source_id}; start a new batch')
        for candidate in entry['candidates']:
            if candidate['status'] in {'passed', 'failed'}:
                continue
            destination = output_dir/candidate['relative_dir']
            if destination.exists():
                sequence_path = destination/'sequence.json'
                sequence = read_json(sequence_path) if sequence_path.is_file() else {}
                if sequence.get('status') == 'passed' and (destination/'output.png').is_file():
                    candidate['status'] = 'passed'
                    candidate['output_sha256'] = sha256(destination/'output.png')
                    write_json(batch_path, batch)
                    continue
                suffix = 1
                backup = destination.with_name(f'{destination.name}_interrupted_{suffix}')
                while backup.exists():
                    suffix += 1
                    backup = destination.with_name(f'{destination.name}_interrupted_{suffix}')
                destination.rename(backup)
                candidate.setdefault('interrupted_dirs', []).append(str(backup.relative_to(output_dir)))
                write_json(batch_path, batch)
            progress(f"{source_id}/{candidate['candidate_id']}: {candidate['recipe']['steps']}")
            candidate_start = time.perf_counter()
            try:
                state = (mask, overlay, info) if info is not None else None
                result = run_sequence(root, image_path, candidate['recipe'], destination,
                                      progress, state, subject_selection_mode=source_mode)
                candidate['status'] = result['status']
                candidate['output_sha256'] = result['output_sha256']
            except Exception:
                candidate['status'] = 'failed'
                candidate['error'] = traceback.format_exc()
                progress(f"{source_id}/{candidate['candidate_id']}: failed")
            finally:
                candidate['elapsed_seconds'] = round(time.perf_counter()-candidate_start, 3)
                write_json(batch_path, batch)
    return batch


def load_batch(root, batch_path):
    root = Path(root).resolve()
    batch_path = Path(batch_path).resolve()
    batch = read_json(batch_path)
    manifest = _manifest(root)
    known = {row['source_id']: row for row in manifest['images']}
    if batch.get('dataset') != 'photo120' or batch.get('schema_version') != 1:
        raise ValueError('Unsupported candidate batch')
    assignment = (selector100_assignment(manifest)
                  if batch.get('split_filter') == 'selector100' else None)
    if assignment is not None and (batch.get('split_assignment') != assignment or
                                   batch.get('split_seed') != 2012 or
                                   not set(batch['source_ids']).issubset(assignment)):
        raise ValueError('Selector100 split provenance changed')
    for source_id, entry in batch['sources'].items():
        row = known[source_id]
        expected_split = assignment[source_id] if assignment is not None else row['split']
        if (entry['split'] != expected_split or
                entry.get('dataset_split', row['split']) != row['split'] or
                entry['source_path'] != row['path'] or entry['source_sha256'] != row['sha256']):
            raise ValueError(f'Batch source provenance mismatch: {source_id}')
        if sha256(root/row['path']) != row['sha256']:
            raise ValueError(f'Source content changed: {source_id}')
        for candidate in entry['candidates']:
            directory = (batch_path.parent/candidate['relative_dir']).resolve()
            if not directory.is_relative_to(batch_path.parent):
                raise ValueError('Candidate path escapes batch')
            if candidate['status'] == 'passed' and sha256(directory/'output.png') != candidate['output_sha256']:
                raise ValueError(f'Candidate hash mismatch: {candidate["candidate_id"]}')
    return batch


def augment_references(root, batch_path, progress=print):
    root, batch_path = Path(root).resolve(), Path(batch_path).resolve()
    batch = load_batch(root, batch_path)
    if batch['candidates_per_source'] != 5:
        raise ValueError('Reference augmentation requires a five-candidate base batch')
    if len(batch['sources']) != len(batch['source_ids']) or any(
        candidate['status'] not in {'passed', 'failed'}
        for entry in batch['sources'].values() for candidate in entry['candidates']):
        raise ValueError('Finish base candidate generation before augmenting references')
    if not batch.get('reference_augmentation_version'):
        for source_id in batch['source_ids']:
            entry = batch['sources'][source_id]
            if entry['estimated_subject_count'] == 0:
                continue
            _, styles, materials = _reference_order(source_id)
            for candidate_id, step in (('c05', {'op': 'stylize', 'reference_id': styles[1]}),
                                       ('c06', {'op': 'material_reconstruct', 'material_id': materials[1]})):
                recipe = {'schema_version': 1, 'steps': [step]}
                validate_route(validate_recipe(recipe), entry['estimated_subject_count'])
                entry['candidates'].append({'candidate_id': candidate_id, 'recipe': recipe,
                                            'status': 'pending', 'relative_dir': f'{source_id}/{candidate_id}'})
        batch['reference_augmentation_version'] = 1
        write_json(batch_path, batch)
    return generate_batch(root, batch_path.parent, batch['split_filter'], len(batch['source_ids']),
                          batch['candidates_per_source'], progress)


def pairs_for_source(entry):
    successful = [item for item in entry['candidates'] if item['status'] == 'passed']
    passed = [item['candidate_id'] for item in successful]
    if len(passed) < 2:
        return []
    pairs = [(passed[0], other) for other in passed[1:]]
    for operation in ('stylize', 'material_reconstruct'):
        comparable = [item['candidate_id'] for item in successful
                      if len(item.get('recipe', {}).get('steps', [])) == 1
                      and item['recipe']['steps'][0]['op'] == operation]
        if len(comparable) >= 2:
            pairs.extend((comparable[0], candidate_id) for candidate_id in comparable[1:])
    for item in successful:
        steps = item.get('recipe', {}).get('steps', [])
        if len(steps) > 1:
            first = next((other['candidate_id'] for other in successful
                          if len(other.get('recipe', {}).get('steps', [])) == 1
                          and other['recipe']['steps'][0] == steps[0]), None)
            if first:
                pairs.append((first, item['candidate_id']))
    return list(dict.fromkeys(pairs))
