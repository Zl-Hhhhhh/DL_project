"""Prepare a reproducible COCO photo-editing dataset, without running models.

Default: 120 new images, 80/20/20 source-level train/val/test split.
Only CC BY / CC BY-SA images are selected. Windows and Linux supported.
"""
import argparse
import hashlib
import json
import random
import time
import struct
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://s3.amazonaws.com/images.cocodataset.org'
ANNOTATION_URL = 'https://s3.us-east-1.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip'
CATEGORIES = ['person', 'cat', 'dog', 'horse', 'chair', 'bus', 'bowl', 'laptop', 'umbrella', 'elephant']


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def download(url, path):
    path = Path(path)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.part')
    for attempt in range(4):
        try:
            # The regional endpoint serves the same official bucket without the
            # long-lived proxy stalls observed on the global S3 endpoint.
            endpoint=url.replace('https://s3.amazonaws.com/', 'https://s3.us-east-1.amazonaws.com/')
            with requests.get(endpoint, stream=True, timeout=(15, 30)) as response:
                response.raise_for_status()
                with partial.open('wb') as stream:
                    for block in response.iter_content(32 * 1024):
                        stream.write(block)
            partial.replace(path)
            return
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(1 + attempt)


def dhash(path):
    with Image.open(path) as raw:
        image = np.asarray(ImageOps.exif_transpose(raw).convert('L').resize((9, 8)))
    return sum(int(bit) << i for i, bit in enumerate((image[:, 1:] > image[:, :-1]).ravel()))


def byte_range(start, end):
    for attempt in range(4):
        try:
            response = requests.get(ANNOTATION_URL+f'?range={start}-{end}',
                                    headers={'Range':f'bytes={start}-{end}'},timeout=(15,60))
            response.raise_for_status()
            if response.status_code != 206 or len(response.content)!=end-start+1:
                raise ValueError('Incorrect remote ZIP byte range')
            return response.content
        except (requests.RequestException,ValueError):
            if attempt==3:
                raise
            time.sleep(attempt+1)


def annotation_document(coco_split):
    """Read one compressed JSON member from the official ZIP, not the entire archive."""
    member='annotations/instances_'+coco_split+'.json'
    cached=ROOT/'data/cache'/('instances_'+coco_split+'.json')
    provenance=cached.with_suffix('.source.json')
    if cached.exists() and provenance.exists():
        source=json.loads(provenance.read_text(encoding='utf-8'))
        if digest(cached)!=source['extracted_sha256']:
            raise ValueError('Cached annotation JSON changed')
        return json.loads(cached.read_text(encoding='utf-8')),source
    # Fixed central directory of the published 2017 archive; each member has ZIP CRC validation.
    central=byte_range(252906877,252907540)
    cursor=0
    while central[cursor:cursor+4]==b'PK\x01\x02':
        fields=struct.unpack_from('<4s6H3L5H2L',central,cursor)
        n,e,c=fields[10:13]
        name=central[cursor+46:cursor+46+n].decode('utf-8')
        if name==member:
            length,offset,expected_crc=fields[8],fields[-1],fields[7]
            break
        cursor+=46+n+e+c
    else:
        raise ValueError('Official archive layout changed; review its central directory')
    header=byte_range(offset,offset+29)
    n,e=struct.unpack_from('<HH',header,26)
    start=offset+30+n+e
    chunk=64*1024
    parts=ROOT/'data/cache'/('instances_'+coco_split+'.parts')
    parts.mkdir(parents=True,exist_ok=True)
    def fetch(index):
        a=start+index*chunk;b=min(start+length-1,a+chunk-1)
        path=parts/str(index)
        if not path.exists() or path.stat().st_size!=b-a+1:
            path.write_bytes(byte_range(a,b))
        if index%16==0:
            print(f'Annotation metadata chunk {index+1}/{(length+chunk-1)//chunk}',flush=True)
        return path
    print(f'Downloading only {member}: {length:,} compressed bytes',flush=True)
    with ThreadPoolExecutor(max_workers=8) as executor:
        paths=list(executor.map(fetch,range((length+chunk-1)//chunk)))
    decoded=zlib.decompress(b''.join(p.read_bytes() for p in paths),-15)
    if zlib.crc32(decoded)!=expected_crc:
        raise ValueError('Annotation ZIP member CRC mismatch')
    cached.write_bytes(decoded)
    source={'archive_url':ANNOTATION_URL,'member':member,'member_crc32':f'{expected_crc:08x}',
            'extracted_sha256':digest(cached),'verification':'ZIP member CRC32 and locally computed SHA-256; full archive not downloaded'}
    provenance.write_text(json.dumps(source,indent=2)+'\n',encoding='utf-8')
    for path in paths:
        path.unlink()
    parts.rmdir()
    return json.loads(decoded),source


def candidate_pools(coco, excluded_ids, seed):
    images = {row['id']: row for row in coco['images']}
    names = {row['id']: row['name'] for row in coco['categories']}
    licenses = {row['id']: row for row in coco['licenses']}
    pools = {name: {} for name in CATEGORIES}
    for ann in coco['annotations']:
        category = names[ann['category_id']]
        image = images[ann['image_id']]
        if category not in pools or image['id'] in excluded_ids or image['license'] not in (4, 5) or ann['iscrowd']:
            continue
        x, y, w, h = ann['bbox']
        fraction = ann['area'] / (image['width'] * image['height'])
        if not .04 <= fraction <= .70 or min(w, h) < 50:
            continue
        if min(x, y, image['width'] - x - w, image['height'] - y - h) < 4:
            continue
        row = dict(image_id=image['id'], annotation_id=ann['id'], category=category,
                   bbox_xywh=ann['bbox'], area_fraction=fraction,
                   width=image['width'], height=image['height'], license=licenses[image['license']],
                   flickr_url=image.get('flickr_url'), file_name=image['file_name'])
        previous = pools[category].get(image['id'])
        if previous is None or row['area_fraction'] > previous['area_fraction']:
            pools[category][image['id']] = row
    rng = random.Random(seed)
    result = {}
    for name in CATEGORIES:
        result[name] = sorted(pools[name].values(), key=lambda r: r['image_id'])
        rng.shuffle(result[name])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'data/photo120')
    parser.add_argument('--per-category', type=int, default=12, help='Multiple of 6; 12 gives 120 images, 120 gives 1200.')
    parser.add_argument('--coco-split', choices=['val2017','train2017'], default='val2017')
    parser.add_argument('--seed', type=int, default=20261027)
    args = parser.parse_args()
    if args.per_category < 6 or args.per_category % 6:
        parser.error('--per-category must be a positive multiple of 6')
    output = args.output.resolve()
    manifest_path = output/'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['seed'] != args.seed or manifest['per_category'] != args.per_category or manifest['coco_split'] != args.coco_split:
            raise ValueError('Existing frozen dataset differs; choose a new output directory.')
        for row in manifest['images']:
            path = output/row['relative_path']
            download(row['download_url'], path)
            if digest(path) != row['sha256']:
                raise ValueError('Dataset hash mismatch: ' + row['source_id'])
        print(f'Verified {len(manifest["images"])} frozen images.', flush=True)
        return
    dev = json.loads((ROOT/'data/dev/manifest.json').read_text(encoding='utf-8'))
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output/'selection.json'
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding='utf-8'))
        if plan['seed'] != args.seed or plan['per_category'] != args.per_category or plan['coco_split'] != args.coco_split:
            raise ValueError('Selection plan differs; choose a new output directory.')
    else:
        coco, annotation_source = annotation_document(args.coco_split)
        pools = candidate_pools(coco, {row['image_id'] for row in dev['images']}, args.seed)
        insufficient={name:len(pool) for name,pool in pools.items() if len(pool)<args.per_category}
        if insufficient:
            raise ValueError(f'Insufficient licensed sources: {insufficient}; use train2017 or a smaller per-category count.')
        # Freeze oversampled reserves before inference. Replacements below only remove duplicates.
        plan = dict(seed=args.seed, per_category=args.per_category, coco_split=args.coco_split, annotation_source=annotation_source,
                    categories={name: pool[:args.per_category*4] for name, pool in pools.items()})
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        del coco, pools
    previous_hashes = {r['sha256'] for r in dev['images']}
    hashes = [dhash(ROOT/r['path']) for r in dev['images']]
    used_ids, selected, skipped = set(), [], []
    for category, pool in plan['categories'].items():
        candidates = [r for r in pool if r['image_id'] not in used_ids]
        initial = candidates[:args.per_category]
        def fetch(row):
            download(BASE+'/'+args.coco_split+'/'+row['file_name'], output/'images'/row['file_name'])
        with ThreadPoolExecutor(max_workers=6) as executor:
            list(executor.map(fetch, initial))
        chosen = []
        for row in candidates:
            path = output/'images'/row['file_name']
            fetch(row)
            sha, perceptual = digest(path), dhash(path)
            if sha in previous_hashes or any(bin(perceptual ^ h).count('1') <= 4 for h in hashes):
                skipped.append(dict(image_id=row['image_id'], reason='exact_or_dhash_near_duplicate'))
                continue
            previous_hashes.add(sha)
            hashes.append(perceptual)
            used_ids.add(row['image_id'])
            with Image.open(path) as image:
                image.verify()
            chosen.append({**row, 'sha256':sha, 'dhash64':f'{perceptual:016x}'})
            if len(chosen) == args.per_category:
                break
        if len(chosen) != args.per_category:
            raise ValueError('Insufficient distinct sources: ' + category)
        rng = random.Random(str(args.seed)+category)
        rng.shuffle(chosen)
        n = args.per_category
        splits = ['train']*(n*2//3)+['val']*(n//6)+['test']*(n//6)
        for row, split in zip(chosen, splits):
            sid = f'coco_{args.coco_split}_{row["image_id"]:012d}'
            relative = 'images/'+row['file_name']
            selected.append({**row, 'source_id':sid, 'capture_group':sid, 'split':split,
                             'relative_path':relative, 'path':(output/relative).relative_to(ROOT).as_posix(),
                             'download_url':BASE+'/'+args.coco_split+'/'+row['file_name']})
        print(category, len(chosen), 'images prepared', flush=True)
    manifest = dict(schema_version=1, dataset='COCO '+args.coco_split, source='https://cocodataset.org/#download',
                    seed=args.seed, per_category=args.per_category, coco_split=args.coco_split, annotation_source=plan['annotation_source'],
                    selection='Seeded, category-balanced, interior non-crowd annotated subjects; CC BY / CC BY-SA only; frozen before model inference.',
                    duplicate_screen='Exact SHA-256 and 64-bit dHash distance <=4 against development and all selected sources; not a guarantee against all related captures.',
                    restrictions=['Existing 12 development sources excluded.', 'Validation/test images are reserved; not used for preview or fixed comparisons.', 'COCO-pretrained models may have seen these sources.'],
                    excluded_duplicates=skipped, images=selected)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    summary = dict(total=len(selected), splits=dict(Counter(r['split'] for r in selected)),
                   categories=dict(Counter(r['category'] for r in selected)),
                   image_bytes=sum((output/r['relative_path']).stat().st_size for r in selected),
                   development_overlap=0, perceptual_duplicates_rejected=len(skipped), inference_performed=False)
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    credits = ['# Image sources', '', 'Selection uses CC BY 2.0 and CC BY-SA 2.0 images. Attribution follows each linked original source; preserve applicable share-alike terms for derivatives.', '']
    credits += [f'- {r["source_id"]}: {r.get("flickr_url")} | {r["license"]["name"]} | {r["license"]["url"]}' for r in selected]
    (output/'ATTRIBUTION.md').write_text('\n'.join(credits)+'\n', encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
