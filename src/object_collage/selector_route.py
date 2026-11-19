"""Source-level subject route audit with held-out detector-box calibration."""
import random
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from .io import letterbox, read_json, sha256, write_json
from .segmentation import _load_detector, detector_route
from .selector_data import load_batch
from .selector_label import read_labels, validate_ranking


def source_folds(batch, manifest, seed=2012):
    categories = {row['source_id']: row['category'] for row in manifest['images']}
    groups = defaultdict(list)
    for source_id in batch['source_ids']:
        groups[categories[source_id]].append(source_id)
    if len(batch['source_ids']) != 100 or len(groups) != 10 or any(len(ids) != 10 for ids in groups.values()):
        raise ValueError('Five-fold selector evaluation expects ten sources in each of ten categories')
    rng = random.Random(seed)
    folds = [[] for _ in range(5)]
    for category in sorted(groups):
        ids = sorted(groups[category])
        rng.shuffle(ids)
        for index, source_id in enumerate(ids):
            folds[index // 2].append(source_id)
    return [sorted(fold) for fold in folds]


def route_class(count):
    return min(int(count), 2)


def _detector_outputs(root, batch, config):
    import torch
    from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
    from torchvision.ops import nms
    from torchvision.transforms.functional import to_tensor

    checkpoint = root/config['subject_detector_path']
    if sha256(checkpoint) != config['subject_detector_sha256']:
        raise ValueError('Subject detector checkpoint hash mismatch')
    model = _load_detector(str(checkpoint))
    categories = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT.meta['categories']
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    output = {}
    for source_id in batch['source_ids']:
        entry = batch['sources'][source_id]
        image, valid, mapping = letterbox(root/entry['source_path'], 512)
        with torch.inference_mode():
            prediction = model([to_tensor(image.convert('RGB'))])[0]
        scores = prediction['scores']
        eligible = torch.nonzero(scores >= .25).flatten()
        kept = nms(prediction['boxes'][eligible], scores[eligible], .65)[:30]
        x0, y0, x1, y1 = mapping['content_box_xyxy']
        rows = []
        for index in eligible[kept].tolist():
            bx0, by0, bx1, by1 = prediction['boxes'][index].tolist()
            area = max(0, min(bx1, x1)-max(bx0, x0))*max(0, min(by1, y1)-max(by0, y0))
            rows.append({'category': categories[int(prediction['labels'][index])],
                         'score': float(scores[index]),
                         'area_fraction': area/max(1, int(valid.sum())),
                         'bbox_xyxy': [bx0, by0, bx1, by1]})
        output[source_id] = rows
    return output


def _accuracy(ids, predicted, truth):
    confusion = Counter((predicted[source_id], truth[source_id]) for source_id in ids)
    return {'correct': sum(predicted[source_id] == truth[source_id] for source_id in ids),
            'total': len(ids),
            'accuracy': sum(predicted[source_id] == truth[source_id] for source_id in ids)/len(ids),
            'confusion': {f'{estimate}->{human}': count
                          for (estimate, human), count in sorted(confusion.items())}}


def audit_routes(root, batch_path):
    root, batch_path = Path(root).resolve(), Path(batch_path).resolve()
    batch = load_batch(root, batch_path)
    manifest = read_json(root/'data/photo120/manifest.json')
    folds = source_folds(batch, manifest)
    rankings = [row for split in ('train', 'val')
                for row in read_labels(batch_path.parent/f'ranks_{split}.jsonl')]
    if len(rankings) != 100 or len({row['source_id'] for row in rankings}) != 100:
        raise ValueError('All 100 sources need one ranking before route audit')
    for row in rankings:
        validate_ranking(row, batch['sources'][row['source_id']])
    truth = {row['source_id']: route_class(row['subject_count']) for row in rankings}
    baseline = {source_id: route_class(entry['estimated_subject_count'])
                for source_id, entry in batch['sources'].items()}
    cache_path = batch_path.parent/'route_detections.json'
    if cache_path.exists():
        cache = read_json(cache_path)
        if cache['batch_sha256'] != sha256(batch_path):
            raise ValueError('Route detector cache does not match candidate batch')
        detections = cache['detections']
    else:
        config = yaml.safe_load((root/'configs/dev.yaml').read_text(encoding='utf-8'))
        detections = _detector_outputs(root, batch, config)
        write_json(cache_path, {'schema_version': 1, 'batch_sha256': sha256(batch_path),
                                'model_path': config['subject_detector_path'],
                                'model_sha256': config['subject_detector_sha256'],
                                'detections': detections})
    grid = [(score, area, mode)
            for score in (.35, .5, .65, .8)
            for area in (.01, .025, .05)
            for mode in ('all', 'living_preferred')]
    predictions = {params: {source_id: detector_route(rows, *params)
                            for source_id, rows in detections.items()} for params in grid}
    fold_reports = []
    all_ids = set(batch['source_ids'])
    out_of_fold = {}
    for index, held_out in enumerate(folds):
        training = sorted(all_ids-set(held_out))
        selected = max(grid, key=lambda params: (
            _accuracy(training, predictions[params], truth)['accuracy'],
            _accuracy(training, predictions[params], truth)['correct'],
            -grid.index(params)))
        out_of_fold.update({source_id: predictions[selected][source_id] for source_id in held_out})
        fold_reports.append({'fold': index+1, 'source_ids': held_out,
                             'parameters': {'score_threshold': selected[0],
                                            'area_threshold': selected[1], 'mode': selected[2]},
                             'raw': _accuracy(held_out, baseline, truth),
                             'detector': _accuracy(held_out, predictions[selected], truth)})
    report = {'schema_version': 1, 'batch_sha256': sha256(batch_path),
              'raw': _accuracy(batch['source_ids'], baseline, truth),
              'five_fold': fold_reports,
              'detector_cv_accuracy': sum(row['detector']['correct'] for row in fold_reports)/100,
              'mode_mismatches': {mode: sum(route_class(batch['sources'][row['source_id']]['estimated_subject_count']) !=
                                           route_class(row['subject_count']) for row in rankings
                                           if batch['sources'][row['source_id']]['subject_selection_mode'] == mode)
                                  for mode in ('sam2_automatic', 'detector_box_v1')},
              'per_source': [
                  {'source_id': source_id,
                   'category': next(row['category'] for row in manifest['images'] if row['source_id'] == source_id),
                   'selection_mode': batch['sources'][source_id]['subject_selection_mode'],
                   'human_count': next(row['subject_count'] for row in rankings if row['source_id'] == source_id),
                   'old_count': batch['sources'][source_id]['estimated_subject_count'],
                   'detector_route': out_of_fold[source_id],
                   'detections': detections[source_id]}
                  for source_id in batch['source_ids']],
              'note': 'Each detector rule is selected on 80 sources and tested on the other 20.'}
    write_json(batch_path.parent/'route_audit.json', report)
    return report
