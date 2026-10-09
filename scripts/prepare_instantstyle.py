"""Download only the pinned SDXL fp16 and original InstantStyle adapter weights."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ['HF_HOME'] = str(ROOT / '.cache/huggingface')
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '120'

from huggingface_hub import snapshot_download


def ranged_download(repo, revision, folder, patterns):
    """Resumable ranged downloads, verified against the publisher's file hashes."""
    import concurrent.futures
    import fnmatch
    import hashlib
    import json
    import requests
    response = requests.get(f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true', timeout=30)
    response.raise_for_status()
    files = [item for item in response.json()['siblings'] if any(fnmatch.fnmatch(item['rfilename'], pattern) for pattern in patterns)]
    def digest(path, lfs):
        value = hashlib.sha256() if lfs else hashlib.sha1()
        if not lfs:
            value.update(f'blob {path.stat().st_size}\0'.encode())
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                value.update(block)
        return value.hexdigest()
    for item in files:
        target = folder/item['rfilename']
        target.parent.mkdir(parents=True, exist_ok=True)
        size = item['size']
        lfs = item.get('lfs')
        expected = lfs['sha256'] if lfs else item['blobId']
        if target.exists() and target.stat().st_size == size and digest(target, lfs) == expected:
            print(f'Verified existing {item["rfilename"]}', flush=True)
            continue
        url = f'https://huggingface.co/{repo}/resolve/{revision}/{item["rfilename"]}'
        partial = target.with_suffix(target.suffix+'.part')
        journal = target.with_suffix(target.suffix+'.chunks.json')
        chunk = 16*1024*1024
        done = set(json.loads(journal.read_text())) if partial.exists() and partial.stat().st_size == size and journal.exists() else set()
        if not done:
            with partial.open('wb') as stream:
                stream.truncate(size)
        def fetch(start):
            end = min(size, start+chunk)-1
            for attempt in range(3):
                try:
                    with requests.get(url, headers={'Range':f'bytes={start}-{end}'}, stream=True, timeout=(20,60)) as result:
                        result.raise_for_status()
                        if result.status_code != 206 or result.headers.get('Content-Range') != f'bytes {start}-{end}/{size}':
                            raise ValueError('Server did not honor requested range')
                        count = 0
                        with partial.open('r+b') as stream:
                            stream.seek(start)
                            for block in result.iter_content(1024*1024):
                                stream.write(block)
                                count += len(block)
                        if count != end-start+1:
                            raise ValueError('Incomplete range')
                    return start
                except Exception:
                    if attempt == 2:
                        raise
        # Small text files may not support Range responses.
        if not lfs:
            result = requests.get(url, timeout=60)
            result.raise_for_status()
            partial.write_bytes(result.content)
        else:
            offsets = list(range(0, size, chunk))
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
                futures = [pool.submit(fetch, start) for start in offsets if start not in done]
                for future in concurrent.futures.as_completed(futures):
                    done.add(future.result())
                    journal.write_text(json.dumps(sorted(done)))
                    if len(done)%8 == 0 or len(done) == len(offsets):
                        print(f'{item["rfilename"]}: {len(done)}/{len(offsets)} chunks', flush=True)
        if digest(partial, lfs) != expected:
            raise ValueError(f'Publisher hash mismatch: {target}')
        partial.replace(target)
        print(f'Verified {item["rfilename"]}: {expected}', flush=True)

MODELS = [
    ('stabilityai/stable-diffusion-xl-base-1.0', '462165984030d82259a11f4367a4eed129e94a7b',
     ROOT / 'models/instantstyle/sdxl-base',
     ['model_index.json', 'scheduler/*', 'tokenizer/*', 'tokenizer_2/*',
      'text_encoder/config.json', 'text_encoder/model.fp16.safetensors',
      'text_encoder_2/config.json', 'text_encoder_2/model.fp16.safetensors',
      'unet/config.json', 'unet/diffusion_pytorch_model.fp16.safetensors',
      'vae/config.json', 'vae/diffusion_pytorch_model.fp16.safetensors']),
    ('h94/IP-Adapter', '018e402774aeeddd60609b4ecdb7e298259dc729',
     ROOT / 'models/instantstyle/ip-adapter',
     ['sdxl_models/image_encoder/config.json', 'sdxl_models/image_encoder/model.safetensors',
      'sdxl_models/ip-adapter_sdxl.bin']),
    ('diffusers/controlnet-canny-sdxl-1.0', 'eb115a19a10d14909256db740ed109532ab1483c',
     ROOT / 'models/instantstyle/controlnet-canny',
     ['config.json', 'diffusion_pytorch_model.fp16.safetensors']),
]

if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--ranged', action='store_true', help='Use six resumable connections and verify publisher hashes')
    parser.add_argument('--controlnet-only', action='store_true')
    args = parser.parse_args()
    for repo, revision, folder, patterns in MODELS:
        if args.controlnet_only and 'controlnet' not in repo:
            continue
        print(f'Downloading {repo} at {revision}', flush=True)
        if args.ranged:
            ranged_download(repo, revision, folder, patterns)
        else:
            snapshot_download(repo, revision=revision, local_dir=folder, allow_patterns=patterns, max_workers=3)
        (folder/'SOURCE.json').write_text(json.dumps({'repository':repo,'revision':revision,'patterns':patterns},indent=2),encoding='utf-8')
    print('InstantStyle model downloads complete.', flush=True)
