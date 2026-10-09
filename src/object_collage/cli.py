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
    sub.add_parser('doctor', help='Print environment and test a CUDA operation')
    ui = sub.add_parser('ui', help='Open the local image editor service')
    ui.add_argument('--port', type=int, default=7860)
    run = sub.add_parser('run', help='Run segmentation and duotone candidates on a frozen manifest or one image')
    run.add_argument('--config', default='configs/dev.yaml')
    sources = run.add_mutually_exclusive_group()
    sources.add_argument('--manifest', default=None)
    sources.add_argument('--image', type=Path)
    run.add_argument('--category', help='Optional COCO class name for single-image mode')
    run.add_argument('--instance', type=int, help='Explicit prediction instance ID shown in detections.png')
    run.add_argument('--composition', choices=['center', 'left', 'right'], default='center')
    run.add_argument('--device', help='0 (CUDA) or cpu; no silent CPU fallback')
    run.add_argument('--allow-truncated', action='store_true', help='Explicit override, recorded in result metadata')
    run.add_argument('--run-id', default=None)
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
        serve(root, args.port)
    elif args.command == 'doctor':
        from .pipeline import environment_info
        import torch
        info = environment_info()
        if torch.cuda.is_available():
            x = torch.ones(16, 16, device='cuda')
            assert torch.equal(x @ x, torch.full_like(x, 16))
            info['cuda_matmul'] = 'passed'
        print(json.dumps(info, indent=2))
    elif args.command == 'run':
        from .pipeline import run_batch
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
