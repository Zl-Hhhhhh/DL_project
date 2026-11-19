"""Frozen-CLIP features and an image-conditioned pairwise candidate ranker."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .io import sha256, write_json
from .presets import PRESETS
from .selector_data import load_batch, pairs_for_source
from .selector_label import read_labels, tasks_for_batch, validate_label, validate_ranking
from .sequence import MATERIAL_IDS, OPERATIONS


STYLE_IDS = ('van_gogh_cypresses', 'cezanne_sainte_victoire', 'hokusai_great_wave',
             'van_gogh_starry_night', 'da_vinci_mona_lisa', 'van_gogh_sunflowers')
VOCAB = sorted([f'op:{op}' for op in OPERATIONS] +
               [f'style:{item}' for item in STYLE_IDS] +
               [f'material:{item}' for item in MATERIAL_IDS] +
               [f'composition:{item}' for item in ('center', 'left', 'right')])
VISUAL_DIM = 1024
ACTION_DIM = 3*(len(VOCAB)+7)+2
INPUT_DIM = 3*VISUAL_DIM+ACTION_DIM
PREGEN_INPUT_DIM = 2*VISUAL_DIM+ACTION_DIM


class FeatureStore:
    def __init__(self, root, batch_dir):
        self.root = Path(root).resolve()
        self.folder = Path(batch_dir).resolve()/'features'
        self.folder.mkdir(exist_ok=True)
        self.model_path = self.root/'models/dragondiffusion/ip-adapter/models/image_encoder'
        digest = sha256(self.model_path/'model.safetensors')
        meta_path = self.folder/'meta.json'
        meta = {'schema_version': 1, 'encoder': 'CLIPVisionModelWithProjection',
                'path': str(self.model_path.relative_to(self.root)), 'weights_sha256': digest,
                'device': 'npu:3', 'preprocess': 'CLIPImageProcessor default; RGB source and masked source'}
        if meta_path.exists():
            old = json.loads(meta_path.read_text(encoding='utf-8'))
            if old != meta:
                raise ValueError('Feature cache encoder provenance changed')
        else:
            write_json(meta_path, meta)
        self.model = None
        self.processor = None

    def _load(self):
        import torch
        import torch_npu  # noqa: F401
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
        if not torch.npu.is_available() or torch.npu.device_count() <= 3:
            raise RuntimeError('NPU 3 is unavailable')
        self.processor = CLIPImageProcessor()
        self.model = CLIPVisionModelWithProjection.from_pretrained(
            self.model_path, local_files_only=True, torch_dtype=torch.float16
        ).eval().to('npu:3')

    def vector(self, image_path, mask_path=None):
        import torch
        from PIL import Image
        image_path = Path(image_path)
        key = sha256(image_path)
        if mask_path is not None:
            key += ':'+sha256(mask_path)
        filename = hashlib.sha256(key.encode()).hexdigest()+'.npy'
        destination = self.folder/filename
        if destination.exists():
            vector = np.load(destination, allow_pickle=False)
            if vector.shape != (VISUAL_DIM,) or not np.isfinite(vector).all():
                raise ValueError(f'Invalid cached visual feature: {destination}')
            return vector
        if self.model is None:
            self._load()
        with Image.open(image_path) as raw:
            image = raw.convert('RGB')
        if mask_path is not None:
            with Image.open(mask_path) as raw_mask:
                mask = raw_mask.convert('L')
            if mask.size != image.size:
                raise ValueError('Source mask dimensions do not match source image')
            white = Image.new('RGB', image.size, 'white')
            white.paste(image, (0, 0), mask)
            image = white
        pixel = self.processor(images=image, return_tensors='pt').pixel_values.to('npu:3', dtype=torch.float16)
        with torch.inference_mode():
            vector = self.model(pixel).image_embeds[0].float().cpu().numpy()
        vector = np.asarray(vector, dtype=np.float32)
        vector /= max(float(np.linalg.norm(vector)), 1e-8)
        if vector.shape != (VISUAL_DIM,) or not np.isfinite(vector).all():
            raise ValueError('CLIP produced invalid visual features')
        np.save(destination, vector, allow_pickle=False)
        return vector

    def close(self):
        if self.model is not None:
            del self.model
            self.model = None
            import gc
            import torch
            gc.collect()
            torch.npu.empty_cache()


def _actions(recipe, subject_count):
    vector = np.zeros(ACTION_DIM, dtype=np.float32)
    slot_size = len(VOCAB)+7
    for slot, step in enumerate(recipe['steps']):
        base = slot*slot_size
        tokens = [f"op:{step['op']}"]
        if 'reference_id' in step:
            tokens.append(f"style:{step['reference_id']}")
        if 'material_id' in step:
            tokens.append(f"material:{step['material_id']}")
        if 'composition' in step:
            tokens.append(f"composition:{step['composition']}")
        for token in tokens:
            vector[base+VOCAB.index(token)] = 1
        op = step['op']
        vector[base+len(VOCAB):base+slot_size] = [
            step.get('strength', 1) if op in PRESETS else 0,
            step.get('style_strength', 1) if op == 'stylize' else 0,
            step.get('structure_strength', .8) if op == 'stylize' else 0,
            step.get('material_strength', 1) if op == 'material_reconstruct' else 0,
            step.get('dx', 0)/256, step.get('dy', 0)/256,
            step.get('texture_fraction', .25) if op in {'duotone_collage', 'original_collage'} else 0]
    vector[-2:] = [len(recipe['steps'])/3, min(subject_count, 3)/3]
    return vector


def candidate_vector(store, batch_dir, entry, candidate):
    directory = Path(batch_dir)/candidate['relative_dir']
    source_path = directory/'source.png'
    if not source_path.is_file():
        raise ValueError(f'Missing source image: {source_path}')
    source = store.vector(source_path)
    first_step = candidate['recipe']['steps'][0]['op']
    mask_path = directory/f'step_01_{first_step}'/'mask_before.png'
    masked = store.vector(source_path, mask_path) if mask_path.is_file() else np.zeros(VISUAL_DIM, dtype=np.float32)
    output = store.vector(directory/'output.png')
    actions = _actions(candidate['recipe'], entry['estimated_subject_count'])
    return np.concatenate((source, masked, output, actions)).astype(np.float32)


def extract_features(root, batch_path, split='train', limit=None, progress=print):
    batch_path = Path(batch_path).resolve()
    batch = load_batch(root, batch_path)
    if split not in {'train', 'val', 'test', 'all'}:
        raise ValueError('Unknown split')
    rows = [entry for entry in batch['sources'].values() if split == 'all' or entry['split'] == split]
    if limit is not None:
        rows = rows[:limit]
    store = FeatureStore(root, batch_path.parent)
    count = 0
    try:
        for entry in rows:
            for candidate in entry['candidates']:
                if candidate['status'] == 'passed':
                    candidate_vector(store, batch_path.parent, entry, candidate)
                    count += 1
                    progress(f"features: {entry['source_id']}/{candidate['candidate_id']}")
    finally:
        store.close()
    return {'candidates': count, 'split': split, 'encoder': 'CLIPVisionModelWithProjection', 'device': 'npu:3'}


def _labeled_pairs(batch, label_paths):
    tasks = {task['task_id']: task for task in tasks_for_batch(batch)}
    seen = set()
    groups = {'train': [], 'val': [], 'test': []}
    excluded = {'route_mismatch_included': 0, 'repeat': 0, 'neither': 0}
    repeat_answers = {}
    for path in label_paths:
        for label in read_labels(path):
            task = tasks.get(label.get('task_id'))
            if task is None or label.get('source_id') != task['source_id'] or label.get('split') != task['split']:
                raise ValueError('Label provenance does not match candidate batch')
            validate_label(label, task, label.get('annotator'))
            key = (label['annotator'], label['task_id'])
            if key in seen:
                raise ValueError('Duplicate annotator task label')
            seen.add(key)
            answer = label['preference']
            if answer in {'left', 'right'}:
                answer = label[answer]
            pair_key = (label['annotator'], task['source_id'], tuple(sorted((task['left'], task['right']))))
            repeat_answers.setdefault(pair_key, {})[task['repeat']] = answer
            if task['repeat']:
                excluded['repeat'] += 1
                continue
            if label['subject_count'] != task['estimated_subject_count']:
                excluded['route_mismatch_included'] += 1
            if label['preference'] == 'neither':
                excluded['neither'] += 1
            groups[task['split']].append(label)
    checked = [answers for answers in repeat_answers.values() if len(answers) == 2]
    excluded['repeat_pairs_reviewed'] = len(checked)
    excluded['repeat_agreement'] = (sum(answers[False] == answers[True] for answers in checked)/len(checked)
                                    if checked else None)
    existing_pairs = {(row['annotator'], row['source_id'], tuple(sorted((row['left'], row['right']))))
                      for rows in groups.values() for row in rows}
    ranked_sources = set()
    for label_path in label_paths:
        rank_path = label_path.with_name(label_path.name.replace('labels_', 'ranks_', 1))
        if rank_path == label_path:
            continue
        for ranking in read_labels(rank_path):
            entry = batch['sources'].get(ranking.get('source_id'))
            if entry is None or ranking.get('split') != entry['split']:
                raise ValueError('Ranking provenance does not match candidate batch')
            validate_ranking(ranking, entry)
            key = (ranking['annotator'], ranking['source_id'])
            if key in ranked_sources:
                raise ValueError('Duplicate annotator source ranking')
            ranked_sources.add(key)
            if ranking['subject_count'] != entry['estimated_subject_count']:
                excluded['route_mismatch_included'] += 1
            positions = {candidate_id: index for index, candidate_id in enumerate(ranking['ranked_ids'])}
            passed = [item['candidate_id'] for item in entry['candidates'] if item['status'] == 'passed']
            comparisons = list(pairs_for_source(entry))
            comparisons.extend((left, right) for index, left in enumerate(passed)
                               for right in passed[index+1:]
                               if not (ranking['unacceptable'][left] and ranking['unacceptable'][right]))
            for left, right in dict.fromkeys(comparisons):
                pair_key = (ranking['annotator'], ranking['source_id'], tuple(sorted((left, right))))
                if pair_key in existing_pairs:
                    continue
                left_bad, right_bad = ranking['unacceptable'][left], ranking['unacceptable'][right]
                if left_bad and right_bad:
                    preference = 'neither'
                    excluded['neither'] += 1
                elif left_bad != right_bad:
                    preference = 'right' if left_bad else 'left'
                else:
                    preference = 'left' if positions[left] < positions[right] else 'right'
                groups[entry['split']].append({
                    'source_id': ranking['source_id'], 'split': entry['split'],
                    'annotator': ranking['annotator'], 'left': left, 'right': right,
                    'preference': preference, 'subject_count': ranking['subject_count'],
                    'unacceptable': {'left': left_bad, 'right': right_bad},
                    'identity': {'left': ranking['identity'][left], 'right': ranking['identity'][right]},
                    'derived_from_ranking': True})
                existing_pairs.add(pair_key)
    return groups, excluded


def _feature_matrix(root, batch_path, batch, labels):
    store = FeatureStore(root, batch_path.parent)
    keys = {(label['source_id'], side_id) for label in labels
            for side_id in (label['left'], label['right'])}
    features = {}
    try:
        for source_id, candidate_id in sorted(keys):
            entry = batch['sources'][source_id]
            candidate = next(c for c in entry['candidates'] if c['candidate_id'] == candidate_id)
            if candidate['status'] != 'passed':
                raise ValueError('Label refers to failed candidate')
            features[(source_id, candidate_id)] = candidate_vector(store, batch_path.parent, entry, candidate)
    finally:
        store.close()
    return features


def _network(hidden_dims=(128, 64), input_dim=INPUT_DIM):
    import torch
    first, second = hidden_dims
    return torch.nn.Sequential(torch.nn.Linear(input_dim, first), torch.nn.ReLU(),
                               torch.nn.Dropout(.25), torch.nn.Linear(first, second),
                               torch.nn.ReLU(), torch.nn.Linear(second, 2))


def _visual_mode(matrix, mode):
    if mode == 'pre_generation':
        return np.concatenate((matrix[:, :2*VISUAL_DIM], matrix[:, 3*VISUAL_DIM:]), axis=1)
    if mode == 'action_only':
        matrix[:, :3*VISUAL_DIM] = 0
    elif mode == 'output_only':
        matrix[:, :2*VISUAL_DIM] = 0
    elif mode != 'full':
        raise ValueError(f'Unknown visual mode: {mode}')
    return matrix


def _candidate_targets(labels):
    targets = {}
    for row in labels:
        for side in ('left', 'right'):
            key = (row['source_id'], row[side])
            target = {'source_id': key[0], 'candidate_id': key[1],
                      'acceptable': not row['unacceptable'][side],
                      'identity': row['identity'][side],
                      'subject_count': row['subject_count']}
            if key in targets and targets[key] != target:
                raise ValueError(f'Conflicting reviews for candidate: {key}')
            targets[key] = target
    return [targets[key] for key in sorted(targets)]


def _source_weights(rows):
    counts = {}
    for row in rows:
        counts[row['source_id']] = counts.get(row['source_id'], 0) + 1
    return np.array([1 / counts[row['source_id']] for row in rows], dtype=np.float32)


def _tensor_pairs(labels, features, mode, device):
    import torch
    left = _visual_mode(np.stack([features[(row['source_id'], row['left'])]
                                  for row in labels]), mode)
    right = _visual_mode(np.stack([features[(row['source_id'], row['right'])]
                                   for row in labels]), mode)
    target = np.array([1 if row['preference'] == 'left' else 0 if row['preference'] == 'right' else .5
                       for row in labels], dtype=np.float32)
    eligible = np.array([row['preference'] != 'neither' for row in labels], dtype=bool)
    decisive = [row for row in labels if row['preference'] != 'neither']
    weights = np.zeros(len(labels), dtype=np.float32)
    weights[eligible] = _source_weights(decisive)
    return tuple(torch.as_tensor(array, device=device)
                 for array in (left, right, target, eligible, weights))


def _tensor_candidates(rows, features, mode, device):
    import torch
    matrix = _visual_mode(np.stack([features[(row['source_id'], row['candidate_id'])]
                                    for row in rows]), mode)
    target = np.array([row['acceptable'] for row in rows], dtype=np.float32)
    weights = _source_weights(rows)
    return tuple(torch.as_tensor(array, device=device) for array in (matrix, target, weights))


def _source_metrics(predictions, rows, rankings):
    by_source = {}
    for row, prediction in zip(rows, predictions):
        by_source.setdefault(row['source_id'], []).append((row, prediction))
    results = []
    for ranking in rankings:
        candidates = by_source.get(ranking['source_id'], [])
        if not candidates:
            continue
        acceptable = [(row, output) for row, output in candidates if output[1] > 0]
        chosen = max(acceptable, key=lambda item: float(item[1][0]))[0] if acceptable else None
        ranked = ranking['ranked_ids']
        results.append((ranking, chosen, bool(ranked)))
    positive = [(ranking, chosen) for ranking, chosen, has_good in results if has_good]
    empty = [chosen for _, chosen, has_good in results if not has_good]
    selected = [(ranking, chosen) for ranking, chosen, _ in results if chosen is not None]
    return {
        'source_count': len(results),
        'source_decision_success': (sum((chosen is not None and
                                         not ranking['unacceptable'][chosen['candidate_id']])
                                        if ranked else chosen is None
                                        for ranking, chosen, ranked in results) / len(results)
                                    if results else None),
        'source_top1_hit': (sum(chosen is not None and chosen['candidate_id'] == ranking['ranked_ids'][0]
                                for ranking, chosen in positive) / len(positive) if positive else None),
        'source_top3_hit': (sum(chosen is not None and chosen['candidate_id'] in ranking['ranked_ids'][:3]
                                for ranking, chosen in positive) / len(positive) if positive else None),
        'all_unacceptable_sources': len(empty),
        'rejection_accuracy': sum(chosen is None for chosen in empty) / len(empty) if empty else None,
        'chosen_identity_preserved_rate': (sum(ranking['identity'][chosen['candidate_id']] == 'preserved'
                                               for ranking, chosen in selected if ranking['subject_count'] > 0) /
                                           sum(ranking['subject_count'] > 0 for ranking, _ in selected)
                                           if any(ranking['subject_count'] > 0 for ranking, _ in selected) else None),
    }


def _metrics(model, pairs, candidates, labels, candidate_rows, rankings):
    import torch
    left, right, target, eligible, _ = pairs
    matrix, acceptable, _ = candidates
    model.eval()
    with torch.inference_mode():
        ls, la = model(left).unbind(-1)
        rs, ra = model(right).unbind(-1)
        predictions = model(matrix).float().cpu().numpy()
        difference = (ls-rs).float().cpu().numpy()
        accept_left = (la > 0).cpu().numpy()
        accept_right = (ra > 0).cpu().numpy()
    choose_left = np.where(accept_left != accept_right, accept_left, difference >= 0)
    decisive = [index for index, row in enumerate(labels) if row['preference'] in {'left', 'right'}]
    correct = sum(bool(choose_left[index]) == (labels[index]['preference'] == 'left') for index in decisive)
    selected_bad = sum(labels[index]['unacceptable']['left' if choose_left[index] else 'right']
                       for index in decisive)
    true_accept = acceptable.cpu().numpy().astype(bool)
    acceptance_correct = int(((predictions[:, 1] > 0) == true_accept).sum())
    fixed = [(index, 'left' if row['left'] == 'c00' else 'right')
             for index, row in enumerate(labels) if 'c00' in (row['left'], row['right'])]
    fixed_decisive = [(index, side) for index, side in fixed
                      if labels[index]['preference'] in {'left', 'right'}]
    fixed_correct = sum(labels[index]['preference'] == side for index, side in fixed_decisive)
    identity = [labels[index]['identity']['left' if choose_left[index] else 'right']
                for index in decisive if labels[index]['subject_count'] > 0]
    random_bad = 1 - float(true_accept.mean()) if len(true_accept) else None
    return {'decisive_pairs': len(decisive), 'preference_accuracy': correct/len(decisive) if decisive else None,
            'selected_unacceptable_rate': selected_bad/len(decisive) if decisive else None,
            'acceptance_accuracy': acceptance_correct/len(candidate_rows) if candidate_rows else None,
            'selected_identity_preserved_rate': identity.count('preserved')/len(identity) if identity else None,
            'random_unacceptable_rate_expected': random_bad,
            'fixed_route_decisive_pairs': len(fixed_decisive),
            'fixed_route_preference_accuracy': fixed_correct/len(fixed_decisive) if fixed_decisive else None,
            'fixed_route_unacceptable_rate': sum(labels[index]['unacceptable'][side] for index, side in fixed)/len(fixed) if fixed else None,
            **_source_metrics(predictions, candidate_rows, rankings)}


def _selection_score(metrics):
    return tuple(-1 if metrics[key] is None else metrics[key]
                 for key in ('source_decision_success', 'source_top1_hit', 'preference_accuracy'))


def _training_loss(model, pairs, candidates, acceptance_weight=.3, positive_weight=1.0):
    import torch
    left, right, target, eligible, pair_weights = pairs
    candidate_matrix, acceptable, candidate_weights = candidates
    ls = model(left)[:, 0]
    rs = model(right)[:, 0]
    accept_logits = model(candidate_matrix)[:, 1]
    acceptance_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        accept_logits, acceptable, reduction='none',
        pos_weight=torch.as_tensor(positive_weight, device=acceptable.device))
    loss = acceptance_weight*(acceptance_loss*candidate_weights).sum()/candidate_weights.sum()
    if bool(eligible.any().item()):
        pair_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            (ls-rs)[eligible], target[eligible], reduction='none')
        loss = loss + (pair_loss*pair_weights[eligible]).sum()/pair_weights[eligible].sum()
    return loss


def _fit_one(train, val, train_candidates, val_candidates, train_labels, val_labels,
             train_rows, val_rows, train_rankings, val_rankings, hidden_dims, epochs, seed,
             input_dim=INPUT_DIM, history=None, acceptance_weight=.3,
             positive_weight=1.0):
    import torch
    torch.manual_seed(seed)
    model = _network(hidden_dims, input_dim).to('npu:3')
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.05)
    best_score, best_state, best_epoch = (-1, -1, -1), None, 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = _training_loss(model, train, train_candidates, acceptance_weight, positive_weight)
        loss.backward()
        optimizer.step()
        val_metrics = _metrics(model, val, val_candidates, val_labels, val_rows, val_rankings)
        if history is not None:
            history.append({'epoch': epoch+1, 'train_loss': float(loss.detach().cpu()),
                            'val_source_success': val_metrics['source_decision_success'],
                            'val_top1_hit': val_metrics['source_top1_hit'],
                            'val_pair_accuracy': val_metrics['preference_accuracy']})
        score = _selection_score(val_metrics)
        if score > best_score:
            best_score = score
            best_epoch = epoch+1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    return (model, best_epoch,
            _metrics(model, train, train_candidates, train_labels, train_rows, train_rankings),
            _metrics(model, val, val_candidates, val_labels, val_rows, val_rankings))


def train_ranker(root, batch_path, epochs=80, seed=2012):
    import torch
    import torch_npu  # noqa: F401
    root, batch_path = Path(root).resolve(), Path(batch_path).resolve()
    batch = load_batch(root, batch_path)
    if len(batch['sources']) != len(batch['source_ids']) or any(
        candidate['status'] not in {'passed', 'failed'}
        for entry in batch['sources'].values() for candidate in entry['candidates']):
        raise ValueError('Finish candidate generation before training')
    groups, excluded = _labeled_pairs(batch, [batch_path.parent/f'labels_{split}.jsonl'
                                        for split in ('train', 'val')])
    if not groups['train'] or not groups['val']:
        raise ValueError('Human labels are required in both train and val; test is held out')
    if not any(row['preference'] in {'left', 'right'} for row in groups['train']):
        raise ValueError('Training needs at least one decisive human preference')
    if not torch.npu.is_available() or torch.npu.device_count() <= 3:
        raise RuntimeError('NPU 3 is unavailable')
    if not 1 <= epochs <= 1000:
        raise ValueError('epochs must be in [1, 1000]')
    features = _feature_matrix(root, batch_path, batch, groups['train']+groups['val'])
    candidate_rows = {split: _candidate_targets(groups[split]) for split in ('train', 'val')}
    rankings = {split: read_labels(batch_path.parent/f'ranks_{split}.jsonl')
                for split in ('train', 'val')}
    results = {}
    states = {}
    variants = {
        'image_conditioned': ('full', (128, 64)),
        'image_conditioned_small': ('full', (64, 32)),
        'output_only': ('output_only', (128, 64)),
        'image_blind': ('action_only', (128, 64)),
    }
    for name, (mode, hidden_dims) in variants.items():
        train = _tensor_pairs(groups['train'], features, mode, 'npu:3')
        val = _tensor_pairs(groups['val'], features, mode, 'npu:3')
        train_candidates = _tensor_candidates(candidate_rows['train'], features, mode, 'npu:3')
        val_candidates = _tensor_candidates(candidate_rows['val'], features, mode, 'npu:3')
        model, best_epoch, train_metrics, val_metrics = _fit_one(
            train, val, train_candidates, val_candidates,
            groups['train'], groups['val'], candidate_rows['train'], candidate_rows['val'],
            rankings['train'], rankings['val'], hidden_dims, epochs, seed)
        results[name] = {'best_epoch': best_epoch, 'hidden_dims': list(hidden_dims),
                         'visual_mode': mode, 'train': train_metrics, 'val': val_metrics}
        if mode == 'full':
            states[name] = {key: value.cpu() for key, value in model.state_dict().items()}
        del model, train, val, train_candidates, val_candidates
        torch.npu.empty_cache()
    selected_name = max(states, key=lambda name: _selection_score(results[name]['val']))
    torch.save({'state_dict': states[selected_name], 'hidden_dims': variants[selected_name][1],
                'input_dim': INPUT_DIM, 'vocab': VOCAB, 'feature_weights_sha256':
                json.loads((batch_path.parent/'features/meta.json').read_text())['weights_sha256'],
                'batch_sha256': sha256(batch_path),
                'seed': seed, 'epochs': epochs}, batch_path.parent/'ranker.pt')
    report = {'schema_version': 1, 'training_split': 'train', 'selection_split': 'val',
              'test_evaluated': False, 'train_pairs': len(groups['train']),
              'val_pairs': len(groups['val']),
              'train_candidates': len(candidate_rows['train']),
              'val_candidates': len(candidate_rows['val']),
              'train_source_count': len({row['source_id'] for row in groups['train']}),
              'val_source_count': len({row['source_id'] for row in groups['val']}),
              'excluded': excluded,
              'random_preference_accuracy_expected': .5,
              'results': results, 'selected_model': selected_name, 'checkpoint': 'ranker.pt',
              'caution': 'Pair labels from a small number of source images have high variance; report source counts and do not infer population preference.'}
    write_json(batch_path.parent/'ranker_report.json', report)
    return report


def evaluate_ranker(root, batch_path, split='test'):
    import torch
    import torch_npu  # noqa: F401
    root, batch_path = Path(root).resolve(), Path(batch_path).resolve()
    if split != 'test':
        raise ValueError('This final evaluation command is reserved for the test split')
    batch = load_batch(root, batch_path)
    groups, excluded = _labeled_pairs(batch, [batch_path.parent/'labels_test.jsonl'])
    labels = groups['test']
    if not labels:
        raise ValueError('No human test labels available')
    checkpoint = torch.load(batch_path.parent/'ranker.pt', map_location='cpu', weights_only=True)
    if checkpoint['input_dim'] != INPUT_DIM or checkpoint['vocab'] != VOCAB:
        raise ValueError('Ranker action schema changed')
    if checkpoint['batch_sha256'] != sha256(batch_path):
        raise ValueError('Candidate batch changed since training')
    meta = json.loads((batch_path.parent/'features/meta.json').read_text())
    if checkpoint['feature_weights_sha256'] != meta['weights_sha256']:
        raise ValueError('Feature encoder changed since training')
    features = _feature_matrix(root, batch_path, batch, labels)
    model = _network(tuple(checkpoint.get('hidden_dims', (128, 64)))).to('npu:3')
    model.load_state_dict(checkpoint['state_dict'])
    candidate_rows = _candidate_targets(labels)
    rankings = read_labels(batch_path.parent/'ranks_test.jsonl')
    metrics = _metrics(model, _tensor_pairs(labels, features, 'full', 'npu:3'),
                       _tensor_candidates(candidate_rows, features, 'full', 'npu:3'),
                       labels, candidate_rows, rankings)
    report = {'schema_version': 1, 'split': 'test', 'pairs': len(labels),
              'source_count': len({row['source_id'] for row in labels}),
              'metrics': metrics, 'excluded': excluded}
    write_json(batch_path.parent/'test_report.json', report)
    return report


def pick_candidate(root, batch_path, source_id):
    import torch
    import torch_npu  # noqa: F401
    root, batch_path = Path(root).resolve(), Path(batch_path).resolve()
    batch = load_batch(root, batch_path)
    entry = batch['sources'].get(source_id)
    if entry is None:
        raise ValueError(f'Unknown source: {source_id}')
    if entry['split'] == 'test' and not (batch_path.parent/'test_report.json').is_file():
        raise ValueError('Test predictions remain sealed until final evaluation')
    checkpoint = torch.load(batch_path.parent/'ranker.pt', map_location='cpu', weights_only=True)
    if checkpoint['input_dim'] != INPUT_DIM or checkpoint['vocab'] != VOCAB or checkpoint['batch_sha256'] != sha256(batch_path):
        raise ValueError('Ranker does not match candidate batch or action schema')
    store = FeatureStore(root, batch_path.parent)
    meta = json.loads((store.folder/'meta.json').read_text())
    if checkpoint['feature_weights_sha256'] != meta['weights_sha256']:
        raise ValueError('Feature encoder changed since training')
    candidates = [candidate for candidate in entry['candidates'] if candidate['status'] == 'passed']
    if not candidates:
        raise ValueError('No successful candidates for source')
    try:
        matrix = np.stack([candidate_vector(store, batch_path.parent, entry, candidate)
                           for candidate in candidates])
    finally:
        store.close()
    model = _network(tuple(checkpoint.get('hidden_dims', (128, 64)))).to('npu:3')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    with torch.inference_mode():
        prediction = model(torch.as_tensor(matrix, device='npu:3')).float().cpu().numpy()
    acceptable = [index for index, row in enumerate(prediction) if row[1] > 0]
    if not acceptable:
        return {'source_id': source_id, 'status': 'rejected',
                'successful_candidates_scored': len(candidates)}
    best = max(acceptable, key=lambda index: float(prediction[index, 0]))
    chosen = candidates[best]
    return {'source_id': source_id, 'status': 'selected', 'candidate_id': chosen['candidate_id'],
            'recipe': chosen['recipe'],
            'output': str(batch_path.parent/chosen['relative_dir']/'output.png'),
            'preference_score': float(prediction[best, 0]),
            'acceptability_logit': float(prediction[best, 1]),
            'successful_candidates_scored': len(candidates)}
