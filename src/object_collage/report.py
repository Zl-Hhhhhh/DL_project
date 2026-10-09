"""Static, self-contained experiment report; no server or remote assets."""
import base64
import csv
import html
from collections import Counter
from pathlib import Path

from .io import read_json, write_json


def image_tag(path, label):
    path = Path(path)
    if not path.is_file():
        return f'<div class="missing">{html.escape(label)}: unavailable</div>'
    encoded = base64.b64encode(path.read_bytes()).decode('ascii')
    return f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(label)}"><figcaption>{html.escape(label)}</figcaption></figure>'


def build_report(run_dir, destination=None):
    run_dir = Path(run_dir)
    run = read_json(run_dir / 'run.json')
    records = [read_json(path) for path in sorted(run_dir.glob('*/result.json'))]
    review_path = run_dir / 'inspection_notes.json'
    reviews = read_json(review_path).get('images', {}) if review_path.exists() else {}
    statuses = dict(Counter(row['status'] for row in records))
    successful = [row for row in records if row['status'] == 'ok']
    summary = {
        'run_id': run_dir.name, 'source_count': len(records), 'success_count': len(successful),
        'success_fraction': len(successful)/max(1, len(records)), 'statuses': statuses,
        'accepted_candidates': sum(row.get('candidate_count', 0) for row in records),
        'model_load_seconds': run.get('model_load_seconds'),
        'cold_mask_seconds': [row['segmentation']['seconds'] for row in records if row.get('segmentation', {}).get('cold_prediction')],
        'warm_mask_mean_seconds': _mean([row['segmentation']['seconds'] for row in records if row.get('segmentation') and not row['segmentation']['cold_prediction']]),
        'render_mean_seconds': _mean([row['timings']['render_seconds'] for row in records if 'render_seconds' in row.get('timings', {})]),
        'peak_cuda_allocated_bytes': max([row.get('segmentation', {}).get('peak_cuda_allocated_bytes', 0) for row in records] or [0]),
        'quality_note': 'Success means automatic segmentation and geometric constraints passed. Mask quality, subject identity and aesthetic quality still require human review.',
        'stages_not_run': ['feature_extraction', 'learned_ranking', 'photography_inpainting'],
    }
    write_json(run_dir / 'summary.json', summary)
    fields = ['source_id', 'split', 'category', 'composition', 'status', 'candidate_count', 'mask_seconds', 'render_seconds', 'peak_cuda_allocated_bytes', 'selected_candidate']
    with (run_dir / 'metrics.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in records:
            seg = row.get('segmentation', {})
            writer.writerow({**{key: row.get(key, '') for key in fields}, 'mask_seconds': seg.get('seconds', ''), 'render_seconds': row.get('timings', {}).get('render_seconds', ''), 'peak_cuda_allocated_bytes': seg.get('peak_cuda_allocated_bytes', '')})
    sections = []
    for row in records:
        folder = run_dir / row['source_id']
        status = html.escape(row['status'])
        figures = ''.join(image_tag(folder / name, label) for name, label in [('source.png', 'Letterboxed input'), ('detections.png', 'Instance IDs and automatic selection'), ('mask.png', 'Selected mask'), ('output.png', 'Fixed-rule selection')])
        candidate_figures = ''.join(image_tag(path, path.stem) for path in sorted((folder / 'candidates').glob('c??.png')))
        provenance = row.get('source_metadata', {})
        license_info = provenance.get('license', {})
        details = html.escape(str({'requested_category': row.get('category'), 'selected': row.get('segmentation', {}).get('selected'), 'composition': row.get('composition'), 'status': row['status'], 'human_mask_review': 'pending', 'assistant_visual_inspection': reviews.get(row['source_id']), 'error': row.get('error')}))
        source_url = html.escape(provenance.get('flickr_url') or provenance.get('download_url') or '', quote=True)
        source_link = f'<a href="{source_url}">source</a>' if source_url else 'local input'
        sections.append(f'<section><h2>{html.escape(row["source_id"])} · {status}</h2><p>{source_link} · {html.escape(license_info.get("name", "license not supplied"))}</p><div class="grid">{figures}</div><details><summary>8-candidate pool / available candidates</summary><div class="grid">{candidate_figures}</div></details><pre>{details}</pre></section>')
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Object Collage — development report</title><style>body{font:16px system-ui,sans-serif;background:#f5f4f0;color:#222;margin:0 auto;padding:32px;max-width:1400px}h1{font-size:32px}section{background:white;padding:20px;margin:24px 0;border:1px solid #ddd}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}figure{margin:0}img{width:100%;height:auto;display:block}figcaption{padding:8px 0;color:#555}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}.missing{display:grid;place-items:center;min-height:180px;background:#eee}summary{cursor:pointer;margin:16px 0}a{color:#24547a}</style>'
    page += f'<h1>Object-aware collage · development run</h1><p>{len(successful)}/{len(records)} sources passed automatic checks. All failures are retained. No learned ranker or aesthetic claims.</p><pre>{html.escape(str(summary))}</pre>' + ''.join(sections) + '</html>'
    destination = Path(destination) if destination else run_dir / 'index.html'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(page, encoding='utf-8')
    return summary, destination


def _mean(values):
    return sum(values)/len(values) if values else None
