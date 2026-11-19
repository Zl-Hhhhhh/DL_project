"""Five source-disjoint folds for rendered-candidate and pre-generation selectors."""
import json
import statistics
from pathlib import Path

from .io import read_json, sha256, write_json
from .selector_data import load_batch
from .selector_label import read_labels
from .selector_route import source_folds
from .selector_train import (INPUT_DIM, PREGEN_INPUT_DIM, VOCAB, _candidate_targets,
                             _feature_matrix, _fit_one, _labeled_pairs, _network,
                             _tensor_candidates, _tensor_pairs, _training_loss)


MODELS = {
    'rendered_ranker': ('full', INPUT_DIM),
    'recipe_ranker': ('pre_generation', PREGEN_INPUT_DIM),
    'action_only': ('action_only', INPUT_DIM),
}
HIDDEN_DIMS = (64, 32)


def _positive_weight(rows):
    positives = sum(row['acceptable'] for row in rows)
    return max(1.0, (len(rows)-positives)/max(positives, 1))


def fixed_epoch_summary(folds, epoch):
    result = {}
    for name in MODELS:
        correct = round(sum(fold['models'][name]['history'][epoch-1]['val_source_success']*
                            len(fold['val_source_ids']) for fold in folds))
        total = sum(len(fold['val_source_ids']) for fold in folds)
        result[name] = {'sources_correct': correct, 'sources_total': total,
                        'source_decision_success': correct/total}
    return result


def _train_final(features, labels, epochs, seed):
    import torch

    rows = _candidate_targets(labels)
    pairs = _tensor_pairs(labels, features, 'pre_generation', 'npu:3')
    candidates = _tensor_candidates(rows, features, 'pre_generation', 'npu:3')
    torch.manual_seed(seed)
    model = _network(HIDDEN_DIMS, PREGEN_INPUT_DIM).to('npu:3')
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.05)
    history = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = _training_loss(model, pairs, candidates, 1.0, _positive_weight(rows))
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu()))
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, history


def cross_validate_selectors(root, batch_path, epochs=80, seed=2012, progress=print):
    import torch
    import torch_npu  # noqa: F401

    root, batch_path = Path(root).resolve(), Path(batch_path).resolve()
    if not 1 <= epochs <= 1000 or not torch.npu.is_available() or torch.npu.device_count() <= 3:
        raise ValueError('Training requires 1-1000 epochs and NPU 3')
    batch = load_batch(root, batch_path)
    if len(batch['sources']) != 100 or any(candidate['status'] not in {'passed', 'failed'}
            for entry in batch['sources'].values() for candidate in entry['candidates']):
        raise ValueError('Five-fold evaluation requires the completed 100-source batch')
    manifest = read_json(root/'data/photo120/manifest.json')
    folds = source_folds(batch, manifest, seed)
    groups, excluded = _labeled_pairs(batch, [batch_path.parent/f'labels_{split}.jsonl'
                                        for split in ('train', 'val')])
    labels = groups['train']+groups['val']
    rankings = [row for split in ('train', 'val')
                for row in read_labels(batch_path.parent/f'ranks_{split}.jsonl')]
    if len(rankings) != 100 or len({row['source_id'] for row in rankings}) != 100:
        raise ValueError('Every source needs exactly one human ranking')
    features = _feature_matrix(root, batch_path, batch, labels)
    destination = batch_path.parent/'cross_validation_balanced'
    destination.mkdir(exist_ok=True)
    fold_reports = []
    for fold_number, held_out in enumerate(folds, 1):
        val_ids = set(held_out)
        train_ids = set(batch['source_ids'])-val_ids
        assert len(train_ids) == 80 and len(val_ids) == 20 and not train_ids.intersection(val_ids)
        train_labels = [row for row in labels if row['source_id'] in train_ids]
        val_labels = [row for row in labels if row['source_id'] in val_ids]
        train_rows, val_rows = _candidate_targets(train_labels), _candidate_targets(val_labels)
        positive_weight = _positive_weight(train_rows)
        train_rankings = [row for row in rankings if row['source_id'] in train_ids]
        val_rankings = [row for row in rankings if row['source_id'] in val_ids]
        if len(train_rankings) != 80 or len(val_rankings) != 20:
            raise ValueError('Ranking source leakage or missing labels')
        fold_result = {'fold': fold_number, 'train_source_ids': sorted(train_ids),
                       'val_source_ids': sorted(val_ids), 'train_pairs': len(train_labels),
                       'val_pairs': len(val_labels), 'acceptance_positive_weight': positive_weight,
                       'models': {}}
        for name, (mode, input_dim) in MODELS.items():
            progress(f'fold {fold_number}/5: {name}')
            train = _tensor_pairs(train_labels, features, mode, 'npu:3')
            val = _tensor_pairs(val_labels, features, mode, 'npu:3')
            train_candidates = _tensor_candidates(train_rows, features, mode, 'npu:3')
            val_candidates = _tensor_candidates(val_rows, features, mode, 'npu:3')
            history = []
            model, best_epoch, train_metrics, val_metrics = _fit_one(
                train, val, train_candidates, val_candidates, train_labels, val_labels,
                train_rows, val_rows, train_rankings, val_rankings, HIDDEN_DIMS,
                epochs, seed+fold_number, input_dim=input_dim, history=history,
                acceptance_weight=1.0, positive_weight=positive_weight)
            checkpoint_path = destination/f'fold_{fold_number:02d}_{name}.pt'
            torch.save({'state_dict': {key: value.cpu() for key, value in model.state_dict().items()},
                        'model_kind': name, 'visual_mode': mode, 'input_dim': input_dim,
                        'hidden_dims': HIDDEN_DIMS, 'vocab': VOCAB,
                        'batch_sha256': sha256(batch_path), 'val_source_ids': sorted(val_ids),
                        'best_epoch': best_epoch}, checkpoint_path)
            fold_result['models'][name] = {'best_epoch': best_epoch,
                                           'train': train_metrics, 'val': val_metrics,
                                           'history': history,
                                           'checkpoint': checkpoint_path.name}
            del model, train, val, train_candidates, val_candidates
            torch.npu.empty_cache()
        fold_reports.append(fold_result)
        write_json(destination/'folds_partial.json', fold_reports)
    summary = {}
    for name in MODELS:
        metrics = [fold['models'][name]['val'] for fold in fold_reports]
        summary[name] = {key: {'mean': statistics.mean(row[key] for row in metrics),
                               'std': statistics.stdev(row[key] for row in metrics)}
                         for key in ('source_decision_success', 'source_top1_hit',
                                     'source_top3_hit', 'preference_accuracy',
                                     'acceptance_accuracy')}
        summary[name]['sources_correct'] = round(sum(row['source_decision_success']*20
                                                      for row in metrics))
    chosen_epochs = [fold['models']['recipe_ranker']['best_epoch'] for fold in fold_reports]
    final_epochs = round(statistics.median(chosen_epochs))
    progress(f'final recipe model: all 100 sources, {final_epochs} epochs')
    state, final_losses = _train_final(features, labels, final_epochs, seed)
    feature_meta = json.loads((batch_path.parent/'features/meta.json').read_text())
    final_checkpoint = batch_path.parent/'recipe_ranker.pt'
    torch.save({'state_dict': state, 'model_kind': 'recipe_ranker',
                'visual_mode': 'pre_generation', 'input_dim': PREGEN_INPUT_DIM,
                'hidden_dims': HIDDEN_DIMS, 'vocab': VOCAB,
                'feature_weights_sha256': feature_meta['weights_sha256'],
                'batch_sha256': sha256(batch_path), 'epochs': final_epochs,
                'seed': seed, 'fold_best_epochs': chosen_epochs}, final_checkpoint)
    report = {'schema_version': 1, 'batch_sha256': sha256(batch_path),
              'fold_seed': seed, 'fold_count': 5, 'epochs_per_fold': epochs,
              'acceptance_loss_weight': 1.0, 'positive_class_weight': 'negative/positive in each training fold',
              'folds': fold_reports, 'summary': summary,
              'fixed_epoch_summary': fixed_epoch_summary(fold_reports, epochs),
              'excluded': excluded,
              'final_recipe_checkpoint': str(final_checkpoint.relative_to(batch_path.parent)),
              'final_recipe_epochs': final_epochs, 'final_recipe_train_loss': final_losses,
              'limitation': 'Each held-out fold also selects its best epoch, so CV metrics have selection optimism; obtain a new independent test set before deployment.'}
    write_json(destination/'report.json', report)
    return report
