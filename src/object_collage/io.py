import hashlib
import json
import os
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def download(url, destination):
    destination = Path(destination)
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + '.part')
    with urllib.request.urlopen(url, timeout=120) as response, partial.open('wb') as stream:
        for chunk in iter(lambda: response.read(1024 * 1024), b''):
            stream.write(chunk)
    partial.replace(destination)


def letterbox(path, size=512):
    """No center crop; alpha is composited on white, original is never modified."""
    with Image.open(path) as raw:
        original_size = list(raw.size)
        oriented = ImageOps.exif_transpose(raw)
        rgba = oriented.convert('RGBA')
        rgb = Image.alpha_composite(Image.new('RGBA', rgba.size, 'white'), rgba).convert('RGB')
    width, height = rgb.size
    scale = min(size / width, size / height)
    resized = (max(1, round(width * scale)), max(1, round(height * scale)))
    left, top = (size - resized[0]) // 2, (size - resized[1]) // 2
    canvas = Image.new('RGB', (size, size), 'white')
    canvas.paste(rgb.resize(resized, Image.Resampling.LANCZOS), (left, top))
    alpha = Image.new('L', (size, size), 0)
    alpha.paste(rgba.getchannel('A').resize(resized, Image.Resampling.NEAREST), (left, top))
    mapping = {
        'file_size_before_exif': original_size, 'oriented_size': [width, height],
        'canvas_size': [size, size], 'resized_size': list(resized), 'offset_xy': [left, top],
        'scale_xy': [resized[0] / width, resized[1] / height],
        'content_box_xyxy': [left, top, left + resized[0], top + resized[1]],
        'alpha_policy': 'composite_on_white; exclude fully transparent pixels from subject mask',
    }
    return canvas, np.asarray(alpha) > 0, mapping


def validate_manifest(manifest):
    """Reject duplicated IDs/content and cross-split group leakage before running."""
    seen_ids, seen_hashes, groups = {}, {}, {}
    for row in manifest['images']:
        split = row['split']
        if split not in {'dev', 'train', 'val', 'test', 'photography'}:
            raise ValueError(f'Unknown split: {split}')
        source = row['source_id']
        if source in seen_ids:
            raise ValueError(f'Duplicate source_id: {source}')
        seen_ids[source] = split
        digest = row['sha256']
        if digest in seen_hashes:
            raise ValueError(f'Duplicate image content: {source}')
        seen_hashes[digest] = split
        group = row.get('capture_group', source)
        if group in groups and groups[group] != split:
            raise ValueError(f'Capture-group leakage: {group}')
        groups[group] = split


def configure_local_runtime(root):
    # Keep library settings/cache in this workspace; do not edit user settings or require accounts.
    root = Path(root)
    (root / '.cache/ultralytics').mkdir(parents=True, exist_ok=True)
    (root / '.cache/matplotlib').mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('YOLO_CONFIG_DIR', str(root / '.cache/ultralytics'))
    os.environ.setdefault('MPLCONFIGDIR', str(root / '.cache/matplotlib'))
    os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
