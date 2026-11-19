"""Verify and prepare fixed material references for reconstruction."""
import hashlib
import io
import json
import urllib.request
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REFERENCES = {
    'wood_table': {
        'title': 'Wood Table', 'asset': 'wood_table',
        'url': 'https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/wood_table/wood_table_diff_1k.jpg',
        'md5': '45af5766b300f7e940ee8984c5fabd4f',
    },
    'rough_linen': {
        'title': 'Rough Linen', 'asset': 'rough_linen',
        'url': 'https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/rough_linen/rough_linen_diff_1k.jpg',
        'md5': '663db789fb075462b685c1f740e58930',
    },
    'ceramic_tiles': {
        'title': 'Interior Tiles', 'asset': 'interior_tiles',
        'url': 'https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/interior_tiles/interior_tiles_diff_1k.jpg',
        'md5': '94ba3714450bb6bbe6a57c9a26e8dac5',
    },
    'silk_satin': {
        'title': 'Crepe Satin', 'asset': 'crepe_satin',
        'url': 'https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/crepe_satin/crepe_satin_diff_1k.jpg',
        'md5': '671816ccd82b5b619cd64bcb77cc9bfd',
    },
}
PATTERN_REFERENCES = {
    'ceramic_iznik': {
        'title': 'Iznik Tile Panel', 'object_id': 454352,
        'url': 'https://images.metmuseum.org/CRDImages/is/web-large/LC-17_190_2088.jpg',
        'source_sha256': '98fe6a194eecf5898dd93648b112b8b6eec5ced22313f022fcf0e7553142225d',
        'crop_box': [100, 35, 500, 435],
    },
    'silk_brocade': {
        'title': 'Rectangular Silk Fragment', 'object_id': 451100,
        'url': 'https://images.metmuseum.org/CRDImages/is/web-large/wb-52.20.20.JPG',
        'source_sha256': '7bea137e7e8b656e416d25e2be6b9e676bb19849c83d1168cdae279d584bdf6b',
        'crop_box': [35, 110, 435, 510],
    },
}
USER_REFERENCES = {
    'ceramic_blue_white': {
        'title': 'Blue and white porcelain',
        'sources': [
            {'file': 'main-image.jpg',
             'sha256': '00f25bbfe8fc3da76f045af1363ed011fb15d2c3739c29e03c666c8590da41f2',
             'crops': [[300, 170, 600, 470], [600, 170, 900, 470],
                       [300, 470, 600, 770], [600, 470, 900, 770]]},
            {'file': 'hk1147-c27cm-1-t2-calendar.jpg',
             'sha256': '7741496b495305a4c3973fb8c08c1174e64db42de3706776ca5ecd81f0745f85',
             'crops': [[650, 1400, 1350, 2100], [1250, 1450, 1950, 2150],
                       [1900, 1400, 2600, 2100], [1050, 600, 1750, 1300]]},
        ],
    },
    'stained_glass': {
        'title': 'Colored stained glass',
        'sources': [
            {'file': '21-23-1-20-6-51-24m.jpg',
             'sha256': '9733cd70cfef2d1f75a78e30899b89799bb24652275a30f91ccc13ccff21db09',
             'crops': [[0, 0, 288, 288], [288, 0, 576, 288],
                       [0, 432, 288, 720], [288, 432, 576, 720]]},
            {'file': 'il_1080xN.5662307901_iqpd.avif',
             'sha256': '3574a3eeadda4332c329219d8b6ca84321f68f9f2a9df363f25c0353aead251c',
             'crops': [[300, 200, 550, 450], [550, 200, 800, 450],
                       [300, 600, 550, 850], [550, 600, 800, 850]]},
        ],
    },
}


def digest(path, algorithm):
    value = hashlib.new(algorithm)
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def download(url, path):
    partial = path.with_suffix(path.suffix+'.part')
    request = urllib.request.Request(url, headers={'User-Agent': 'ObjectCollage/0.1'})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open('wb') as output:
        for block in iter(lambda: response.read(1024*1024), b''):
            output.write(block)
    partial.replace(path)


def main():
    folder = ROOT/'data/material_refs'
    folder.mkdir(parents=True, exist_ok=True)
    catalog = {}
    for reference_id, source in REFERENCES.items():
        path = folder/f'{reference_id}.jpg'
        if not path.is_file():
            download(source['url'], path)
        if digest(path, 'md5') != source['md5']:
            raise ValueError(f'Cached publisher MD5 mismatch: {reference_id}')
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if min(image.size) < 512:
                raise ValueError(f'Texture resolution is too small: {reference_id}')
            size = list(image.size)
        catalog[reference_id] = {**source, 'asset_url': f'https://polyhaven.com/a/{source["asset"]}',
            'license': 'CC0', 'license_url': 'https://polyhaven.com/license',
            'sha256': digest(path, 'sha256'), 'size': size,
            'collagerl_sources': 'four deterministic square quadrant crops of the diffuse map'}
        print(f'Verified {reference_id}: {size}, {catalog[reference_id]["sha256"]}', flush=True)
    for reference_id, source in PATTERN_REFERENCES.items():
        source_path = folder/f'{reference_id}.source.jpg'
        if not source_path.is_file():
            download(source['url'], source_path)
        if digest(source_path, 'sha256') != source['source_sha256']:
            raise ValueError(f'Pattern source hash mismatch: {reference_id}')
        with Image.open(source_path) as image:
            image.load()
            cropped = image.convert('RGB').crop(source['crop_box'])
        if cropped.size != (400, 400):
            raise ValueError(f'Unexpected pattern crop: {reference_id}')
        stream = io.BytesIO()
        cropped.resize((1024, 1024), Image.Resampling.LANCZOS).save(stream, format='JPEG', quality=95)
        path = folder/f'{reference_id}.jpg'
        if path.is_file():
            if path.read_bytes() != stream.getvalue():
                raise ValueError(f'Pattern reference changed: {reference_id}')
        else:
            path.write_bytes(stream.getvalue())
        catalog[reference_id] = {**source,
            'asset_url': f'https://www.metmuseum.org/art/collection/search/{source["object_id"]}',
            'license': 'The Met Open Access; public domain',
            'license_url': 'https://www.metmuseum.org/hubs/open-access',
            'sha256': digest(path, 'sha256'), 'size': [1024, 1024],
            'collagerl_sources': 'four deterministic square quadrant crops of the cropped artwork'}
        print(f'Verified {reference_id}: [1024, 1024], {catalog[reference_id]["sha256"]}', flush=True)
    for reference_id, reference in USER_REFERENCES.items():
        tiles = []
        for item in reference['sources']:
            source_path = folder/item['file']
            if not source_path.is_file() or digest(source_path, 'sha256') != item['sha256']:
                raise ValueError(f'Missing or modified user reference: {source_path}')
            if source_path.suffix.lower() == '.avif':
                import pillow_avif  # noqa: F401
            with Image.open(source_path) as raw:
                image = raw.convert('RGB')
            for crop_box in item['crops']:
                left, top, right, bottom = crop_box
                if not (0 <= left < right <= image.width and 0 <= top < bottom <= image.height
                        and right-left == bottom-top):
                    raise ValueError(f'Invalid user reference crop: {reference_id}')
                stream = io.BytesIO()
                image.crop(crop_box).resize((512, 512), Image.Resampling.LANCZOS).save(
                    stream, format='JPEG', quality=95)
                filename = f'{reference_id}_{len(tiles):02d}.jpg'
                tile_path = folder/filename
                if tile_path.is_file():
                    if tile_path.read_bytes() != stream.getvalue():
                        raise ValueError(f'User reference tile changed: {tile_path}')
                else:
                    tile_path.write_bytes(stream.getvalue())
                tiles.append({'file': filename, 'sha256': digest(tile_path, 'sha256'),
                              'source_file': item['file'], 'crop_box': crop_box})
        catalog[reference_id] = {'title': reference['title'], 'sources': reference['sources'],
            'tiles': tiles, 'license': 'user-provided', 'asset_url': None,
            'collagerl_sources': 'eight fixed crops from two user-provided images'}
        print(f'Verified {reference_id}: {len(tiles)} tiles', flush=True)
    (folder/'SOURCE.json').write_text(json.dumps({'schema_version': 1, 'references': catalog}, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
