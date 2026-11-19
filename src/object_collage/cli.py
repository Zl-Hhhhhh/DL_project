import argparse
import json
from datetime import datetime
from pathlib import Path

import yaml

from .io import configure_local_runtime, read_json, sha256


def main():
    parser = argparse.ArgumentParser(description='DSAA2012 object-aware collage development prototype')
    parser.add_argument('--root', type=Path, default=Path.cwd(), help='Project root (default: current directory)')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('doctor', help='Print available accelerator information')
    ui = sub.add_parser('ui', help='Open the local image editor service')
    ui.add_argument('--port', type=int, default=7860)
    ui.add_argument('--device', help='cpu, npu:<index>, or cuda:<index>')
    run = sub.add_parser('run', help='Run segmentation and duotone candidates on a frozen manifest or one image')
    run.add_argument('--config', default='configs/dev.yaml')
    sources = run.add_mutually_exclusive_group()
    sources.add_argument('--manifest', default=None)
    sources.add_argument('--image', type=Path)
    run.add_argument('--category', help='Not supported by automatic SAM 2 masks')
    run.add_argument('--instance', type=int, help='Explicit prediction instance ID shown in detections.png')
    run.add_argument('--composition', choices=['center', 'left', 'right'], default='center')
    run.add_argument('--device', help='cpu, npu:<index>, or cuda:<index>')
    run.add_argument('--allow-truncated', action='store_true', help='Explicit override, recorded in result metadata')
    run.add_argument('--run-id', default=None)
    sequence = sub.add_parser('sequence', help='Run a recorded treatment sequence on one image')
    sequence.add_argument('--image', required=True, type=Path)
    sequence.add_argument('--recipe', required=True, type=Path, help='JSON recipe with fixed reference IDs')
    sequence.add_argument('--run-id', default=None)
    candidates = sub.add_parser('selector-generate', help='Generate a resumable photo120 candidate batch')
    candidates.add_argument('--split', choices=['train', 'val', 'test', 'all', 'selector100'], default='train')
    candidates.add_argument('--pilot20', action='store_true', help='Two frozen train images per category, 12 candidates each')
    candidates.add_argument('--source-offset', type=int, default=0, help='Skip this many sources in the frozen selection')
    candidates.add_argument('--limit', type=int, help='Number of sources after the offset')
    candidates.add_argument('--candidates-per-source', type=int, help='First N deterministic recipes per source')
    candidates.add_argument('--sampling-temperature', type=float, default=1.5,
                            help='Tempered exploration sampling; selector100 defaults to 1.5')
    candidates.add_argument('--run-id', required=True)
    augment = sub.add_parser('selector-augment', help='Add a second style and material reference to a completed five-candidate batch')
    augment.add_argument('--batch', required=True, type=Path)
    label = sub.add_parser('selector-label', help='Serve blind candidate ranking and pairwise labeling locally')
    label.add_argument('--batch', required=True, type=Path)
    label.add_argument('--port', type=int, default=7862)
    label.add_argument('--host', default='127.0.0.1', help='Listen address; use 0.0.0.0 for LAN access')
    features = sub.add_parser('selector-features', help='Cache frozen CLIP visual features on NPU 3')
    features.add_argument('--batch', required=True, type=Path)
    features.add_argument('--split', choices=['train', 'val', 'test', 'all'], default='train')
    features.add_argument('--limit', type=int)
    train = sub.add_parser('selector-train', help='Train the human-preference ranker')
    train.add_argument('--batch', required=True, type=Path)
    train.add_argument('--epochs', type=int, default=80)
    route_audit = sub.add_parser('selector-audit-routes', help='Audit subject-count routing against human labels')
    route_audit.add_argument('--batch', required=True, type=Path)
    cross_validate = sub.add_parser('selector-cross-validate', help='Five source-disjoint folds for rendered and recipe rankers')
    cross_validate.add_argument('--batch', required=True, type=Path)
    cross_validate.add_argument('--epochs', type=int, default=80)
    evaluate = sub.add_parser('selector-evaluate', help='Evaluate the frozen ranker on held-out test labels')
    evaluate.add_argument('--batch', required=True, type=Path)
    pick = sub.add_parser('selector-pick', help='Choose among already rendered candidates')
    pick.add_argument('--batch', required=True, type=Path)
    pick.add_argument('--source-id', required=True)
    recommend = sub.add_parser('selector-recommend', help='Score treatment recipes before rendering')
    recommend.add_argument('--batch', required=True, type=Path)
    recommend.add_argument('--image', required=True, type=Path)
    recommend.add_argument('--run-id', required=True)
    recommend.add_argument('--execute', action='store_true', help='Render the best acceptable recipe, retrying if it fails')
    pilot_report = sub.add_parser('selector-pilot-report', help='Audit 20-source candidate success, redundancy and cost')
    pilot_report.add_argument('--batch', required=True, type=Path)
    replay = sub.add_parser('replay', help='Render saved parameters and verify the original output SHA-256')
    replay.add_argument('record', type=Path)
    replay.add_argument('--output', required=True, type=Path)
    report = sub.add_parser('report', help='Rebuild summary, CSV and self-contained HTML from result logs')
    report.add_argument('run_dir', type=Path)
    report.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    configure_local_runtime(root)
    if args.command == 'ui':
        from .web import serve
        serve(root, args.port, args.device)
    elif args.command == 'doctor':
        from .pipeline import environment_info
        import torch
        info = environment_info()
        if info['npu_available']:
            device = 'npu:3'
            if info['npu_count'] <= 3:
                raise RuntimeError('NPU 3 is unavailable')
            x = torch.ones(16, 16, device=device)
            assert (x @ x)[0, 0].item() == 16
            info['npu3_matmul'] = 'passed'
        if torch.cuda.is_available():
            x = torch.ones(16, 16, device='cuda')
            assert torch.equal(x @ x, torch.full_like(x, 16))
            info['cuda_matmul'] = 'passed'
        print(json.dumps(info, indent=2))
    elif args.command == 'run':
        from .pipeline import run_batch
        if args.category:
            parser.error('SAM 2 automatic masks do not support category filtering; omit --category.')
        config = yaml.safe_load((root / args.config).read_text(encoding='utf-8'))
        if args.device:
            config['device'] = args.device
        if args.allow_truncated:
            config['allow_truncated'] = True
        if args.image:
            image = args.image.resolve()
            manifest = {'schema_version': 1, 'images': [{'source_id': 'local_' + sha256(image)[:12], 'path': str(image), 'sha256': sha256(image), 'split': 'dev', 'category': args.category, 'composition': args.composition}]}
        else:
            manifest = read_json(root / (args.manifest or 'data/dev/manifest.json'))
        run_id = args.run_id or datetime.now().strftime('dev_%Y%m%d_%H%M%S')
        if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in run_id):
            parser.error('run-id must contain only letters, digits, underscores and hyphens')
        summary, report = run_batch(root, config, manifest, root / 'experiments' / run_id, args.instance)
        print(json.dumps(summary, indent=2))
        print(f'Report: {report}')
        if summary['success_count'] == 0:
            raise SystemExit(2)
    elif args.command == 'sequence':
        from .sequence import run_sequence
        run_id = args.run_id or datetime.now().strftime('sequence_%Y%m%d_%H%M%S')
        if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in run_id):
            parser.error('run-id must contain only letters, digits, underscores and hyphens')
        record = run_sequence(root, args.image, read_json(args.recipe), root/'experiments'/run_id, print)
        print(json.dumps({'status': record['status'], 'route': record['route'],
                          'output': str(root/'experiments'/run_id/'output.png')}, indent=2))
    elif args.command == 'selector-generate':
        from .selector_data import generate_batch
        if not args.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_id):
            parser.error('run-id must contain only letters, digits, underscores and hyphens')
        batch = generate_batch(root, root/'experiments'/args.run_id, args.split,
                               args.limit, args.candidates_per_source, pilot20=args.pilot20,
                               source_offset=args.source_offset,
                               sampling_temperature=args.sampling_temperature)
        print(json.dumps({'batch': str(root/'experiments'/args.run_id/'batch.json'),
                          'sources': len(batch['sources']),
                          'passed': sum(c['status'] == 'passed' for s in batch['sources'].values() for c in s['candidates']),
                          'failed': sum(c['status'] == 'failed' for s in batch['sources'].values() for c in s['candidates'])}, indent=2))
    elif args.command == 'selector-augment':
        from .selector_data import augment_references
        batch = augment_references(root, args.batch)
        print(json.dumps({'batch': str(args.batch.resolve()),
                          'sources': len(batch['sources']),
                          'passed': sum(c['status'] == 'passed' for s in batch['sources'].values() for c in s['candidates']),
                          'failed': sum(c['status'] == 'failed' for s in batch['sources'].values() for c in s['candidates'])}, indent=2))
    elif args.command == 'selector-label':
        from .selector_label import serve_labels
        serve_labels(root, args.batch, args.port, args.host)
    elif args.command == 'selector-features':
        from .selector_train import extract_features
        print(json.dumps(extract_features(root, args.batch, args.split, args.limit), indent=2))
    elif args.command == 'selector-train':
        from .selector_train import train_ranker
        print(json.dumps(train_ranker(root, args.batch, args.epochs), indent=2))
    elif args.command == 'selector-audit-routes':
        from .selector_route import audit_routes
        result = audit_routes(root, args.batch)
        print(json.dumps({'raw': result['raw'],
                          'detector_cv_accuracy': result['detector_cv_accuracy'],
                          'report': str(args.batch.resolve().parent/'route_audit.json')}, indent=2))
    elif args.command == 'selector-cross-validate':
        from .selector_cv import cross_validate_selectors
        result = cross_validate_selectors(root, args.batch, args.epochs)
        print(json.dumps({'folds': result['fold_count'], 'summary': result['summary'],
                          'recipe_checkpoint': result['final_recipe_checkpoint']}, indent=2))
    elif args.command == 'selector-evaluate':
        from .selector_train import evaluate_ranker
        print(json.dumps(evaluate_ranker(root, args.batch), indent=2))
    elif args.command == 'selector-pick':
        from .selector_train import pick_candidate
        print(json.dumps(pick_candidate(root, args.batch, args.source_id), ensure_ascii=False, indent=2))
    elif args.command == 'selector-recommend':
        from .selector_recommend import recommend_recipes
        if any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_id):
            parser.error('run-id must contain only letters, digits, underscores and hyphens')
        result = recommend_recipes(root, args.batch, args.image,
                                   root/'experiments'/args.run_id, args.execute)
        print(json.dumps({'status': result['status'], 'selected': result['selected'],
                          'output': result.get('output'),
                          'report': str(root/'experiments'/args.run_id/'recommendation.json')},
                         ensure_ascii=False, indent=2))
    elif args.command == 'selector-pilot-report':
        from .selector_pilot import build_pilot_report
        result = build_pilot_report(root, args.batch)
        print(json.dumps({key: value for key, value in result.items() if key != 'sources' and key != 'operation_results'}, ensure_ascii=False, indent=2))
    elif args.command == 'replay':
        from .pipeline import replay
        print(json.dumps(replay(args.record, args.output), indent=2))
    else:
        from .report import build_report
        summary, destination = build_report(args.run_dir, args.output)
        print(json.dumps(summary, indent=2))
        print(destination)


if __name__ == '__main__':
    main()
