"""CPU-only, source-disjoint evaluation for the course paper."""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import torch

from object_collage.io import sha256, write_json
from object_collage.selector_data import load_batch
from object_collage.selector_label import read_labels
from object_collage.selector_train import (FeatureStore, _candidate_targets, _labeled_pairs,
                                           _network, _visual_mode, candidate_vector)


ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT/'experiments/selector100_t15_v14_20261028/batch.json'
THRESHOLDS = (-float('inf'), -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)
MODES = ('full', 'pre_generation', 'action_only')


class CachedOnlyFeatures(FeatureStore):
    def _load(self):
        raise RuntimeError('Missing cached CLIP feature; paper evaluation must stay on CPU')


def load_data(batch_path):
    batch = load_batch(ROOT, batch_path)
    labels_by_split, _ = _labeled_pairs(batch, [batch_path.parent/f'labels_{split}.jsonl'
                                           for split in ('train', 'val')])
    labels = labels_by_split['train'] + labels_by_split['val']
    rankings = {row['source_id']: row for split in ('train', 'val')
                for row in read_labels(batch_path.parent/f'ranks_{split}.jsonl')}
    rows = _candidate_targets(labels)
    expected = {(sid, candidate['candidate_id']) for sid, entry in batch['sources'].items()
                for candidate in entry['candidates'] if candidate['status'] == 'passed'}
    actual = {(row['source_id'], row['candidate_id']) for row in rows}
    if actual != expected or len(rankings) != len(batch['source_ids']):
        raise ValueError('Rankings or cached candidate targets are incomplete')
    store = CachedOnlyFeatures(ROOT, batch_path.parent)
    features = []
    for row in rows:
        entry = batch['sources'][row['source_id']]
        candidate = next(item for item in entry['candidates']
                         if item['candidate_id'] == row['candidate_id'])
        features.append(candidate_vector(store, batch_path.parent, entry, candidate))
    return batch, labels, rankings, rows, np.stack(features)


def make_training_data(ids, mode_matrix, labels, rows):
    selected = [i for i, row in enumerate(rows) if row['source_id'] in ids]
    local = {(rows[i]['source_id'], rows[i]['candidate_id']): j
             for j, i in enumerate(selected)}
    pairs = [row for row in labels if row['source_id'] in ids
             and row['preference'] in ('left', 'right')]
    pair_counts = defaultdict(int)
    candidate_counts = defaultdict(int)
    for pair in pairs:
        pair_counts[pair['source_id']] += 1
    for i in selected:
        candidate_counts[rows[i]['source_id']] += 1
    pair_indices = torch.tensor([[local[(row['source_id'], row['left'])],
                                  local[(row['source_id'], row['right'])]] for row in pairs])
    pair_targets = torch.tensor([float(row['preference'] == 'left') for row in pairs])
    pair_weights = torch.tensor([1 / pair_counts[row['source_id']] / len(pair_counts)
                                 for row in pairs])
    candidate_targets = torch.tensor([float(rows[i]['acceptable']) for i in selected])
    candidate_weights = torch.tensor([1 / candidate_counts[rows[i]['source_id']] / len(candidate_counts)
                                      for i in selected])
    positives = int(candidate_targets.sum())
    positive_weight = max(1.0, (len(selected)-positives)/max(positives, 1))
    return (torch.from_numpy(mode_matrix[selected]), pair_indices, pair_targets,
            pair_weights, candidate_targets, candidate_weights, positive_weight)


def train(ids, matrix, labels, rows, seed, epochs):
    torch.manual_seed(seed)
    x, pair_indices, pair_targets, pair_weights, targets, candidate_weights, positive_weight = (
        make_training_data(ids, matrix, labels, rows))
    model = _network((64, 32), x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.05)
    losses = []
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        prediction = model(x)
        preference = prediction[:, 0]
        differences = preference[pair_indices[:, 0]] - preference[pair_indices[:, 1]]
        pair_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            differences, pair_targets, reduction='none')
        candidate_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            prediction[:, 1], targets, reduction='none',
            pos_weight=torch.tensor(positive_weight))
        loss = (pair_loss*pair_weights).sum() + (candidate_loss*candidate_weights).sum()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return model, losses


def predict(model, matrix, ids, rows):
    selected = [i for i, row in enumerate(rows) if row['source_id'] in ids]
    model.eval()
    with torch.inference_mode():
        values = model(torch.from_numpy(matrix[selected])).numpy()
    return {(rows[i]['source_id'], rows[i]['candidate_id']): values[j].tolist()
            for j, i in enumerate(selected)}


def evaluate(ids, predictions, rankings, batch, threshold):
    details = []
    for sid in sorted(ids):
        ranking = rankings[sid]
        eligible = [(key[1], value) for key, value in predictions.items()
                    if key[0] == sid and value[1] > threshold]
        chosen = max(eligible, key=lambda item: item[1][0])[0] if eligible else None
        acceptable = bool(chosen is not None and not ranking['unacceptable'][chosen])
        has_good = bool(ranking['ranked_ids'])
        candidate = next((c for c in batch['sources'][sid]['candidates']
                          if c['candidate_id'] == chosen), None)
        details.append({'source_id': sid, 'chosen': chosen,
                        'correct': acceptable if has_good else chosen is None,
                        'acceptable': acceptable, 'has_good': has_good,
                        'top1': chosen == ranking['ranked_ids'][0] if has_good else False,
                        'top3': chosen in ranking['ranked_ids'][:3],
                        'elapsed_seconds': candidate['elapsed_seconds'] if candidate else None})
    selected = [row for row in details if row['chosen'] is not None]
    positives = [row for row in details if row['has_good']]
    return {'correct': sum(row['correct'] for row in details), 'total': len(details),
            'acceptable_selections': sum(row['acceptable'] for row in details),
            'bad_selections': sum(not row['acceptable'] for row in selected),
            'false_rejections': sum(row['has_good'] and row['chosen'] is None for row in details),
            'correct_rejections': sum(not row['has_good'] and row['chosen'] is None for row in details),
            'selection_count': len(selected), 'top1': sum(row['top1'] for row in positives),
            'top3': sum(row['top3'] for row in positives),
            'selected_elapsed_seconds': sum(row['elapsed_seconds'] for row in selected),
            'details': details}


def fixed_position(train_ids, val_ids, rankings, batch):
    positions = [f'c{i:02d}' for i in range(12)]
    def score(ids, position):
        predictions = {(sid, item['candidate_id']): [float(item['candidate_id'] == position), 1.0]
                       for sid in ids for item in batch['sources'][sid]['candidates']
                       if item['status'] == 'passed' and item['candidate_id'] == position}
        return evaluate(ids, predictions, rankings, batch, 0.0)
    chosen = max(positions, key=lambda position: score(train_ids, position)['correct'])
    return chosen, score(val_ids, chosen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, default=BATCH)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--fold-limit', type=int, default=5)
    parser.add_argument('--output', type=Path, default=BATCH.parent/'paper_eval_cpu/report.json')
    args = parser.parse_args()
    if not 1 <= args.fold_limit <= 5 or not 1 <= args.epochs <= 1000:
        parser.error('fold-limit must be 1-5 and epochs must be positive')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    batch_path = args.batch.resolve()
    batch, labels, rankings, rows, full = load_data(batch_path)
    old_report = json.loads((batch_path.parent/'cross_validation_balanced/report.json').read_text())
    if old_report['batch_sha256'] != sha256(batch_path):
        raise ValueError('Frozen cross-validation folds do not match the batch')
    matrices = {mode: _visual_mode(full.copy(), mode) for mode in MODES}
    folds = []
    for fold_index, old_fold in enumerate(old_report['folds'][:args.fold_limit], 1):
        outer_val = set(old_fold['val_source_ids'])
        outer_train = set(old_fold['train_source_ids'])
        inner_cal = set(old_report['folds'][fold_index % 5]['val_source_ids'])
        inner_train = outer_train - inner_cal
        if len(inner_train) != 60 or len(inner_cal) != 20 or outer_val & outer_train:
            raise ValueError('Invalid nested source split')
        position, position_metrics = fixed_position(outer_train, outer_val, rankings, batch)
        fold = {'fold': fold_index, 'inner_calibration_sources': sorted(inner_cal),
                'fixed_position': position, 'models': {}, 'fixed_position_metrics': position_metrics}
        for mode in MODES:
            print(f'fold {fold_index}/{args.fold_limit}: {mode}, inner fit', flush=True)
            matrix = matrices[mode]
            inner_model, inner_loss = train(inner_train, matrix, labels, rows,
                                            2012+fold_index, args.epochs)
            inner_predictions = predict(inner_model, matrix, inner_cal, rows)
            threshold = max(THRESHOLDS, key=lambda value: (
                evaluate(inner_cal, inner_predictions, rankings, batch, value)['correct'],
                -abs(value) if np.isfinite(value) else -1e6))
            print(f'fold {fold_index}/{args.fold_limit}: {mode}, outer fit, threshold {threshold}',
                  flush=True)
            model, losses = train(outer_train, matrix, labels, rows, 2012+fold_index,
                                  args.epochs)
            predictions = predict(model, matrix, outer_val, rows)
            metrics = evaluate(outer_val, predictions, rankings, batch, threshold)
            ungated = evaluate(outer_val, predictions, rankings, batch, -float('inf'))
            fold['models'][mode] = {'threshold': threshold, 'metrics': metrics,
                                    'ungated': ungated, 'train_loss': losses[-1],
                                    'inner_train_loss': inner_loss[-1]}
            if mode == 'pre_generation':
                rng = np.random.default_rng(2012+fold_index)
                donor = {sid: full[next(i for i, row in enumerate(rows)
                                        if row['source_id'] == sid), :2048]
                         for sid in outer_val}
                shuffled = []
                ids = sorted(outer_val)
                for _ in range(30):
                    permuted = ids.copy()
                    rng.shuffle(permuted)
                    modified = matrix.copy()
                    for sid, replacement in zip(ids, permuted):
                        modified[[i for i, row in enumerate(rows)
                                  if row['source_id'] == sid], :2048] = donor[replacement]
                    variant = predict(model, modified, outer_val, rows)
                    shuffled.append(evaluate(outer_val, variant, rankings, batch,
                                              threshold)['correct'])
                fold['models'][mode]['permuted_correct_30'] = shuffled
        folds.append(fold)
        partial = {'schema_version': 1, 'device': 'cpu', 'epochs': args.epochs,
                   'folds': folds, 'note': 'Outer folds are development estimates, not independent test data.'}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, partial)
    summary = {'fixed_position': sum(f['fixed_position_metrics']['correct'] for f in folds)}
    for mode in MODES:
        summary[mode] = {key: sum(f['models'][mode]['metrics'][key] for f in folds)
                         for key in ('correct', 'acceptable_selections', 'bad_selections',
                                     'false_rejections', 'correct_rejections', 'selection_count',
                                     'top1', 'top3', 'selected_elapsed_seconds')}
        summary[mode]['ungated_correct'] = sum(f['models'][mode]['ungated']['correct'] for f in folds)
    summary['pre_generation_permuted_correct_30'] = [sum(f['models']['pre_generation']
        ['permuted_correct_30'][i] for f in folds) for i in range(30)]
    write_json(args.output, {**partial, 'summary': summary,
              'method': 'Five outer source folds; a disjoint 60/20 split within each 80-source '
                        'training fold calibrates the acceptance threshold; all models refit '
                        'on 80 sources for a fixed number of epochs. Candidate features '
                        'come exclusively from existing CLIP caches.'})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
