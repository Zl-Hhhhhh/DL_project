"""Download the complete pinned DragonDiffusion weights, without running models.

SD 1.5 uses the publisher's fp16 variant; load with variant='fp16'.
Rerunning resumes partial downloads and verifies existing files.
"""
import argparse
import fnmatch
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

from prepare_instantstyle import ranged_download

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'models' / 'dragondiffusion'
MODELS = [
    ('stable-diffusion-v1-5/stable-diffusion-v1-5',
     '451f4fe16113bff5a5d2269ed5ad43b0592e9a14', 'sd15', [
         'model_index.json', 'scheduler/scheduler_config.json',
         'tokenizer/*', 'text_encoder/config.json',
         'text_encoder/model.fp16.safetensors', 'unet/config.json',
         'unet/diffusion_pytorch_model.fp16.safetensors', 'vae/config.json',
         'vae/diffusion_pytorch_model.fp16.safetensors',
     ]),
    ('h94/IP-Adapter', '018e402774aeeddd60609b4ecdb7e298259dc729',
     'ip-adapter', ['models/image_encoder/config.json',
                    'models/image_encoder/model.safetensors']),
    ('Adapter/DragonDiffusion', 'eb744a24b896db24ad6751242a07d60cb5995986',
     'auxiliary', ['model/ip_sd15_64.bin',
                   'model/efficient_sam_vits.pt',
                   'model/shape_predictor_68_face_landmarks.dat']),
]


def metadata(repo, revision, patterns):
    response = requests.get(
        f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true',
        timeout=60,
    )
    response.raise_for_status()
    siblings = response.json()['siblings']
    for pattern in patterns:
        if not any(fnmatch.fnmatch(f['rfilename'], pattern) for f in siblings):
            raise ValueError(f'No upstream match: {repo}/{pattern}')
    return [f for f in siblings if any(
        fnmatch.fnmatch(f['rfilename'], p) for p in patterns)]


def verify(folder, files):
    records = []
    for item in files:
        path = folder / item['rfilename']
        if not path.is_file() or path.stat().st_size != item['size']:
            raise ValueError(f'Missing or incorrect size: {path}')
        sha = hashlib.sha256()
        git = hashlib.sha1(f'blob {item["size"]}\0'.encode())
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                sha.update(block)
                git.update(block)
        lfs = item.get('lfs')
        actual = sha.hexdigest() if lfs else git.hexdigest()
        expected = lfs['sha256'] if lfs else item['blobId']
        if actual != expected:
            raise ValueError(f'Publisher hash mismatch: {path}')
        records.append({'path': path.relative_to(DEST).as_posix(),
                        'bytes': item['size'], 'sha256': sha.hexdigest(),
                        'publisher_hash': expected,
                        'publisher_hash_algorithm': 'sha256' if lfs else 'git-blob-sha1'})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    all_records = []
    for repo, revision, subdir, patterns in MODELS:
        folder = DEST / subdir
        files = metadata(repo, revision, patterns)
        print(f'{repo}: {sum(f["size"] for f in files):,} bytes, '
              f'{len(files)} files, revision {revision}', flush=True)
        if not args.verify_only:
            ranged_download(repo, revision, folder, patterns)
        records = verify(folder, files)
        source = {'repository': repo, 'revision': revision,
                  'patterns': patterns, 'files': records}
        (folder / 'SOURCE.json').write_text(
            json.dumps(source, indent=2), encoding='utf-8')
        all_records.extend(records)
    manifest = {
        'verified_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'complete', 'inference_tested': False,
        'sd15_variant': 'fp16',
        'total_bytes': sum(f['bytes'] for f in all_records),
        'files': all_records,
    }
    (DEST / 'MANIFEST.json').write_text(
        json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'COMPLETE: {len(all_records)} verified files, '
          f'{manifest["total_bytes"]:,} bytes in {DEST}', flush=True)


if __name__ == '__main__':
    main()
