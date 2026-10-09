"""Freeze 12 COCO train2017 development images before running any model.

Only the official annotation archive and the selected images are downloaded.
Uses the standard library so preparation can run while the ML env is installing.
"""
import argparse
import hashlib
import json
import random
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Path-style HTTPS for the same COCO S3 bucket; the dotted custom host has a TLS hostname mismatch.
COCO_BASE = 'https://s3.amazonaws.com/images.cocodataset.org'
ANNOTATIONS = COCO_BASE + '/annotations/annotations_trainval2017.zip'


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else _hash(stream)


def _hash(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(chunk)
    return digest.hexdigest()


def download(url, path):
    path = Path(path)
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.part')
    print(f'Downloading {url}', flush=True)
    request = urllib.request.Request(url, headers={'User-Agent': 'DSAA2012-development/0.1'})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open('wb') as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    partial.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=2012)
    args = parser.parse_args()
    manifest_path = ROOT / 'data/dev/manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        for row in manifest['images']:
            path = ROOT / row['path']
            download(row['download_url'], path)
            if sha256(path) != row['sha256']:
                raise ValueError(f"Source hash mismatch: {row['source_id']}")
        print('Existing development manifest verified; selection was not changed.')
        return
    archive = ROOT / 'data/cache/annotations_trainval2017.zip'
    download(ANNOTATIONS, archive)
    print('Reading official COCO instance metadata...', flush=True)
    with zipfile.ZipFile(archive) as zf:
        with zf.open('annotations/instances_train2017.json') as stream:
            coco = json.load(stream)
    images = {image['id']: image for image in coco['images']}
    categories = {cat['id']: cat['name'] for cat in coco['categories']}
    licenses = {item['id']: item for item in coco['licenses']}
    pools = defaultdict(list)
    exclusions = Counter()
    for annotation in coco['annotations']:
        image = images[annotation['image_id']]
        x, y, w, h = annotation['bbox']
        fraction = annotation['area'] / (image['width'] * image['height'])
        if annotation['iscrowd']:
            exclusions['crowd'] += 1
            continue
        if not 0.12 <= fraction <= 0.65 or min(w, h) < 100:
            exclusions['small_or_overlarge_subject'] += 1
            continue
        edge = min(x, y, image['width'] - x - w, image['height'] - y - h) < 5
        record = {
            'image_id': image['id'], 'annotation_id': annotation['id'],
            'category': categories[annotation['category_id']],
            'bbox_xywh': annotation['bbox'], 'area_fraction': fraction,
            'annotation_touches_border': edge,
        }
        pools[(record['category'], edge)].append(record)
    # The last slot deliberately probes border truncation; failures remain in reports.
    slots = [('person', False)] * 4 + [(c, False) for c in ['cat', 'dog', 'bird', 'horse', 'chair', 'cup', 'potted plant']] + [('teddy bear', True)]
    rng = random.Random(args.seed)
    selected, used = [], set()
    for category, edge in slots:
        pool = sorted(pools[(category, edge)], key=lambda item: (item['image_id'], item['annotation_id']))
        rng.shuffle(pool)
        choice = next((row for row in pool if row['image_id'] not in used), None)
        if choice is None:
            raise ValueError(f'No eligible source for fixed slot {category}, edge={edge}')
        used.add(choice['image_id'])
        selected.append(choice)
    manifest = {
        'schema_version': 1, 'split': 'dev', 'seed': args.seed,
        'dataset': 'COCO train2017', 'source': 'https://cocodataset.org/#download',
        'annotation_url': ANNOTATIONS, 'annotation_sha256': sha256(archive),
        'selection': 'Fixed category slots; seeded shuffle of non-crowd instances with area fraction 0.12–0.65 and bbox sides >=100 px; 11 interior and 1 border case. Frozen before segmentation. No quality-based replacement.',
        'excluded_annotation_counts': dict(exclusions),
        'restrictions': ['Development only; exclude these source IDs and near duplicates from formal train/val/test and photography sets.', 'COCO-pretrained models may have seen these images.'],
        'images': [],
    }
    for index, selection in enumerate(selected):
        meta = images[selection['image_id']]
        relative = f"data/dev/images/{meta['file_name']}"
        url = f"{COCO_BASE}/train2017/{meta['file_name']}"
        download(url, ROOT / relative)
        manifest['images'].append({
            **selection, 'source_id': f"coco_train2017_{meta['id']:012d}",
            'split': 'dev', 'path': relative, 'sha256': sha256(ROOT / relative),
            'download_url': url, 'flickr_url': meta.get('flickr_url'),
            'license': licenses[meta['license']], 'width': meta['width'], 'height': meta['height'],
            'group': 'person' if selection['category'] == 'person' else ('animal' if selection['category'] in ['cat', 'dog', 'bird', 'horse'] else 'object'),
            'composition': ['center', 'left', 'right'][index % 3],
        })
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Frozen {len(selected)} development sources: {manifest_path}', flush=True)


if __name__ == '__main__':
    main()
