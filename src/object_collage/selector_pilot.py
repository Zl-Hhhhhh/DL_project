"""Automatic audit of the 20-source training pilot."""
import html
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import numpy as np
from PIL import Image

from .io import read_json, write_json
from .selector_data import load_batch, pairs_for_source
from .selector_label import read_labels, tasks_for_batch


def _dhash(path):
    with Image.open(path) as raw:
        pixels = np.asarray(raw.convert('L').resize((9, 8), Image.Resampling.LANCZOS))
    return sum(int(left > right) << index for index, (left, right) in
               enumerate(zip(pixels[:, :-1].flat, pixels[:, 1:].flat)))


def _percentiles(values):
    if not values:
        return None
    return {'count': len(values), 'median_seconds': round(median(values), 2),
            'p90_seconds': round(float(np.percentile(values, 90)), 2)}


def build_pilot_report(root, batch_path):
    root, batch_path = Path(root).resolve(), Path(batch_path).resolve()
    batch = load_batch(root, batch_path)
    if not batch.get('pilot20') or len(batch['sources']) != 20:
        raise ValueError('Expected a complete pilot20 candidate batch')
    if any(candidate['status'] not in {'passed', 'failed'}
           for entry in batch['sources'].values() for candidate in entry['candidates']):
        raise ValueError('Finish pilot generation before reporting')
    manifest = {row['source_id']: row for row in
                json.loads((root/'data/photo120/manifest.json').read_text())['images']}
    category_counts = Counter(manifest[source_id]['category'] for source_id in batch['source_ids'])
    if len(category_counts) != 10 or set(category_counts.values()) != {2} or any(
        batch['sources'][source_id]['split'] != 'train' for source_id in batch['source_ids']):
        raise ValueError('Pilot source balance or split changed')
    status = Counter()
    step_counts = Counter()
    by_operation = defaultdict(lambda: {'passed': 0, 'failed': 0, 'seconds': [],
                                        'post_sam2_checked': 0, 'post_sam2_subject_found': 0})
    route_counts = Counter()
    source_rows = []
    total_near_duplicates = 0
    all_seconds = []
    for source_id in batch['source_ids']:
        entry = batch['sources'][source_id]
        route_counts['none' if entry['estimated_subject_count'] == 0 else
                     'one' if entry['estimated_subject_count'] == 1 else 'multiple'] += 1
        expected = 12
        if len(entry['candidates']) != expected:
            raise ValueError(f'Candidate count changed for {source_id}: {len(entry["candidates"])}')
        hashes = []
        for candidate in entry['candidates']:
            state = candidate['status']
            status[state] += 1
            step_counts[len(candidate['recipe']['steps'])] += 1
            op = '+'.join(step['op'] for step in candidate['recipe']['steps'])
            by_operation[op][state] += 1
            seconds = candidate.get('elapsed_seconds')
            if seconds is not None:
                by_operation[op]['seconds'].append(seconds)
                all_seconds.append(seconds)
            if state == 'passed':
                image_path = batch_path.parent/candidate['relative_dir']/'output.png'
                hashes.append((candidate['candidate_id'], _dhash(image_path)))
                sequence = read_json(batch_path.parent/candidate['relative_dir']/'sequence.json')
                post_segmentations = [step['segmentation_after'] for step in sequence['steps']
                                      if 'segmentation_after' in step]
                if post_segmentations:
                    by_operation[op]['post_sam2_checked'] += 1
                    by_operation[op]['post_sam2_subject_found'] += int(
                        post_segmentations[-1]['subject_count'] > 0)
        near = [(left_id, right_id, (left_hash ^ right_hash).bit_count())
                for index, (left_id, left_hash) in enumerate(hashes)
                for right_id, right_hash in hashes[index+1:]
                if (left_hash ^ right_hash).bit_count() <= 4]
        total_near_duplicates += len(near)
        source_rows.append({'source_id': source_id, 'category': manifest[source_id]['category'],
                            'estimated_subject_count': entry['estimated_subject_count'],
                            'planned_candidates': len(entry['candidates']),
                            'passed': sum(c['status'] == 'passed' for c in entry['candidates']),
                            'failed': sum(c['status'] == 'failed' for c in entry['candidates']),
                            'label_pairs': len(pairs_for_source(entry)),
                            'route_elapsed_seconds': entry.get('route_elapsed_seconds'),
                            'candidate_elapsed_seconds': round(sum(c.get('elapsed_seconds', 0)
                                                                   for c in entry['candidates']), 2),
                            'near_duplicate_pairs': near})
    labels = read_labels(batch_path.parent/'labels_train.jsonl')
    rankings = read_labels(batch_path.parent/'ranks_train.jsonl')
    report = {'schema_version': 1, 'pilot': 'two train sources per category; 20 total',
              'source_count': 20, 'category_counts': dict(sorted(category_counts.items())),
              'sam2_route_counts': dict(route_counts), 'candidate_status': dict(status),
              'candidate_step_counts': dict(sorted(step_counts.items())),
              'candidate_seconds': _percentiles(all_seconds),
              'total_generation_seconds': round(sum(all_seconds)+sum(
                  entry.get('route_elapsed_seconds', 0) for entry in batch['sources'].values()), 2),
              'route_seconds': _percentiles([entry['route_elapsed_seconds'] for entry in batch['sources'].values()
                                            if 'route_elapsed_seconds' in entry]),
              'operation_results': {op: {'passed': values['passed'], 'failed': values['failed'],
                                         'duration': _percentiles(values['seconds']),
                                         'post_sam2_checked': values['post_sam2_checked'],
                                         'post_sam2_subject_found': values['post_sam2_subject_found']}
                                    for op, values in sorted(by_operation.items())},
              'near_duplicate_pairs_total': total_near_duplicates,
              'planned_label_tasks': len(tasks_for_batch(batch)),
              'human_labels_recorded': len(labels),
              'planned_ranking_sources': 20,
              'human_rankings_recorded': len(rankings),
              'aesthetic_conclusion': 'pending human review',
              'post_sam2_note': 'SAM2 rediscovery is an automatic diagnostic, not a human identity or quality label',
              'near_duplicate_rule': '64-bit dHash Hamming distance <=4; only a redundancy flag, not an aesthetic score',
              'sources': source_rows}
    write_json(batch_path.parent/'pilot_report.json', report)
    sections = []
    for row in source_rows:
        entry = batch['sources'][row['source_id']]
        original = os.path.relpath(root/entry['source_path'], batch_path.parent)
        tiles = [f'<figure><img src="{html.escape(original)}" alt="原图"><figcaption>原图 · {html.escape(row["category"])}</figcaption></figure>']
        for candidate in entry['candidates']:
            op = ' + '.join(step['op'] for step in candidate['recipe']['steps'])
            detail = f"{candidate['candidate_id']} · {op} · {candidate['status']} · {candidate.get('elapsed_seconds', '?')}s"
            if candidate['status'] == 'passed':
                image = f"{candidate['relative_dir']}/output.png"
                tiles.append(f'<figure><img src="{html.escape(image)}" alt="候选"><figcaption>{html.escape(detail)}</figcaption></figure>')
            else:
                tiles.append(f'<figure class="failed"><div>生成失败</div><figcaption>{html.escape(detail)}</figcaption></figure>')
        sections.append(f'<section><h2>{html.escape(row["source_id"])} · {html.escape(row["category"])} · {row["passed"]}/{row["planned_candidates"]}</h2><div class="grid">{"".join(tiles)}</div></section>')
    page = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>20 图候选试验</title><style>body{font:14px system-ui,sans-serif;color:#202b2a;background:#f6f7f5;margin:0}'
            'header{padding:18px max(16px,calc((100vw - 1500px)/2));background:white;border-bottom:1px solid #d9dfda}'
            'h1{font-size:21px;margin:0 0 5px}p{margin:0;color:#52615b}main{max-width:1500px;margin:auto;padding:16px}'
            'section{border-top:1px solid #ced8d2;padding:15px 0 24px}h2{font-size:16px;margin:0 0 12px}'
            '.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(165px,1fr));gap:10px}figure{margin:0;min-width:0}'
            'img,.failed div{width:100%;aspect-ratio:1;object-fit:contain;background:#e3e8e4;display:block}.failed div{display:grid;place-items:center;color:#9a3c31}'
            'figcaption{font-size:11px;line-height:1.35;overflow-wrap:anywhere;padding:5px 0}</style>'
            '<header><h1>20 图候选试验</h1><p>训练集 · 每类两张 · 自动生成检查，审美评价等待人工标注</p></header>'
            f'<main>{"".join(sections)}</main></html>')
    (batch_path.parent/'pilot_gallery.html').write_text(page, encoding='utf-8')
    return report
