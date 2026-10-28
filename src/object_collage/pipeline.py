import importlib.metadata
import platform
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .io import letterbox, read_json, sha256, validate_manifest, write_json
from .render import generate_candidates, render
from .report import build_report
from .segmentation import Segmenter


def validate_config(config):
    if config['canvas_size'] != 512 or config['styles'] != ['duotone']:
        raise ValueError('v0.1 supports only 512px duotone; other planned styles are not implemented yet.')
    if not 0 < config['min_confidence'] <= config['confidence'] <= 1:
        raise ValueError('Invalid confidence thresholds')
    if config['candidates_per_task'] != 8 or config['max_attempts'] != 32:
        raise ValueError('The development protocol fixes 8 candidates and at most 32 attempts.')
    if not config['palettes']:
        raise ValueError('No palettes defined')


def environment_info():
    import torch
    packages = {name: importlib.metadata.version(name) for name in ['torch', 'torchvision', 'ultralytics', 'numpy', 'Pillow', 'opencv-python', 'PyYAML']}
    return {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(), 'packages': packages, 'cuda_runtime': torch.version.cuda, 'cuda_available': torch.cuda.is_available(), 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def run_batch(root, config, manifest, run_dir, explicit_instance=None):
    root, run_dir = Path(root), Path(run_dir)
    validate_config(config)
    validate_manifest(manifest)
    # Development IDs are never accepted in a formal split, including across manifests.
    dev_path = root / 'data/dev/manifest.json'
    if dev_path.exists():
        dev = read_json(dev_path)['images']
        ids, hashes = {r['source_id'] for r in dev}, {r['sha256'] for r in dev}
        for row in manifest['images']:
            if row['split'] != 'dev' and (row['source_id'] in ids or row['sha256'] in hashes):
                raise ValueError(f"Development source leaked into {row['split']}: {row['source_id']}")
    # source_id becomes a directory, so forbid separators and traversal in user manifests.
    for row in manifest['images']:
        source_id = row['source_id']
        if not source_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in source_id):
            raise ValueError('source_id must contain only letters, digits, underscores and hyphens')
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / 'config.json', config)
    write_json(run_dir / 'manifest.json', manifest)
    run = {'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(), 'config': config, 'environment': environment_info(), 'source_code_sha256': {path.name: sha256(path) for path in Path(__file__).parent.glob('*.py')}, 'status': 'initializing', 'selection_method': 'fixed_rule', 'feature_extraction': 'not_run', 'learned_ranking': 'not_run'}
    write_json(run_dir / 'run.json', run)
    try:
        segmenter = Segmenter(config, root)
    except Exception as error:
        run.update(status='initialization_error', error=str(error))
        write_json(run_dir / 'run.json', run)
        raise
    run.update(model=segmenter.provenance, model_load_seconds=segmenter.load_seconds, status='running')
    write_json(run_dir / 'run.json', run)
    for row in manifest['images']:
        folder = run_dir / row['source_id']
        folder.mkdir()
        start = time.perf_counter()
        result = {'schema_version': 1, 'source_id': row['source_id'], 'split': row['split'], 'category': row.get('category'), 'composition': row.get('composition', 'center'), 'source_metadata': row, 'model': segmenter.provenance, 'style': 'duotone', 'selection_method': 'fixed_rule', 'status': 'processing', 'candidate_count': 0, 'timings': {}}
        try:
            source_path = root / row['path']
            if sha256(source_path) != row['sha256']:
                raise ValueError('Source hash mismatch; original was changed')
            image, alpha, mapping = letterbox(source_path, config['canvas_size'])
            image.save(folder / 'source.png')
            result['canvas_mapping'] = mapping
            result['timings']['input_seconds'] = time.perf_counter()-start
            mask, overlay, segmentation = segmenter.predict(image, alpha, mapping, row.get('category'), explicit_instance)
            result['segmentation'] = segmentation
            result['status'] = segmentation['status']
            overlay.save(folder / 'detections.png')
            if mask is not None:
                Image.fromarray((mask * 255).astype(np.uint8)).save(folder / 'mask.png')
                foreground = image.convert('RGBA')
                foreground.putalpha(Image.fromarray((mask*255).astype(np.uint8)))
                foreground.save(folder / 'subject.png')
                gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
                background = (~mask) & alpha
                result['background_edge_energy'] = float(np.abs(cv2.Laplacian(gray, cv2.CV_32F))[background].mean()) if background.any() else None
            if segmentation['status'] == 'ok':
                render_start = time.perf_counter()
                accepted, rejected, selected = generate_candidates(image, mask, config, row['source_id'], result['composition'])
                result['timings']['render_seconds'] = time.perf_counter()-render_start
                result['candidate_count'] = len(accepted)
                result['status'] = 'ok' if len(accepted) == 8 else ('partial_candidates' if accepted else 'no_valid_candidates')
                result['selected_candidate'] = selected['candidate_id'] if selected else None
                candidate_dir = folder / 'candidates'
                candidate_dir.mkdir()
                write_json(folder / 'rejected.json', rejected)
                for candidate in accepted:
                    cid = candidate['candidate_id']
                    candidate['image'].save(candidate_dir / f'{cid}.png')
                    candidate['visible_mask'].save(candidate_dir / f'{cid}_visible.png')
                    replay_record = {'source_id': row['source_id'], 'split': row['split'], 'candidate_id': cid, 'source': '../source.png', 'mask': '../mask.png', 'source_sha256': sha256(folder / 'source.png'), 'mask_sha256': sha256(folder / 'mask.png'), 'output_sha256': sha256(candidate_dir / f'{cid}.png'), 'visible_mask_sha256': sha256(candidate_dir / f'{cid}_visible.png'), 'params': candidate['params'], 'metrics': candidate['metrics'], 'model_sha256': segmenter.provenance['sha256'], 'mask_source': segmentation['mask_source'], 'selected_instance': segmentation['selected'], 'canvas_mapping': mapping, 'selection_method': 'fixed_rule', 'environment': run['environment']['packages']}
                    write_json(candidate_dir / f'{cid}.json', replay_record)
                if selected:
                    cid = selected['candidate_id']
                    shutil.copyfile(candidate_dir / f'{cid}.png', folder / 'output.png')
                    shutil.copyfile(candidate_dir / f'{cid}_visible.png', folder / 'visible_mask.png')
                    # Output parameters keep paths relative to their own location.
                    chosen = read_json(candidate_dir / f'{cid}.json')
                    chosen.update(source='source.png', mask='mask.png')
                    write_json(folder / 'params.json', chosen)
                    result['selected_metrics'] = selected['metrics']
        except (OSError, ValueError) as error:
            result.update(status='input_or_parameter_error', error=str(error))
        except Exception as error:
            result.update(status='runtime_error', error=f'{type(error).__name__}: {error}')
            with (run_dir / 'errors.log').open('a', encoding='utf-8') as stream:
                stream.write(f"\n{row['source_id']}\n{traceback.format_exc()}")
        result['timings']['total_seconds'] = time.perf_counter()-start
        write_json(folder / 'result.json', result)
        print(f"{row['source_id']}: {result['status']} ({result['candidate_count']}/8 candidates)", flush=True)
    run['status'] = 'complete'
    write_json(run_dir / 'run.json', run)
    summary, report = build_report(run_dir)
    run['status'] = 'complete' if summary['success_count'] == summary['source_count'] else 'complete_with_failures'
    write_json(run_dir / 'run.json', run)
    return summary, report


def replay(record_path, output_path):
    record_path, output_path = Path(record_path), Path(output_path)
    record = read_json(record_path)
    source, mask_path = record_path.parent / record['source'], record_path.parent / record['mask']
    if output_path.exists():
        raise ValueError('Replay output already exists; choose a new file to preserve existing artifacts')
    if sha256(source) != record['source_sha256'] or sha256(mask_path) != record['mask_sha256']:
        raise ValueError('Replay input hash mismatch')
    with Image.open(source) as image, Image.open(mask_path) as mask:
        if record['params'].get('version') == 'photo-presets-v1':
            from .presets import render_preset
            output, visible, metrics = render_preset(image, np.asarray(mask)>0, record['params'])
        elif record['params'].get('version') == 'priority-study-v1':
            from .composition_study import render_candidate
            output, visible, metrics = render_candidate(image, np.asarray(mask)>0, record['params'])
        else:
            output, visible, metrics = render(image, np.asarray(mask) > 0, record['params'])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(output_path, format='PNG')
    verified = sha256(output_path) == record['output_sha256']
    if not verified:
        raise ValueError('Replay pixel encoding differs; check pinned environment and renderer version')
    return {'output': str(output_path), 'sha256': sha256(output_path), 'exact_match': verified, 'metrics': metrics}
