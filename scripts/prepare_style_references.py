"""Install pinned public-domain artwork references from The Met and Commons."""
import hashlib
import io
import json
from pathlib import Path

import requests
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / 'data/style_refs'
OBJECTS = {
    'van_gogh_cypresses': 437980,
    'cezanne_sainte_victoire': 435878,
    'hokusai_great_wave': 36491,
}
COMMONS = {
    'van_gogh_starry_night': ('Van Gogh - Starry Night - Google Art Project.jpg',
                             'The Starry Night', 'Vincent van Gogh',
                             'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg'),
    'da_vinci_mona_lisa': ('Mona Lisa, by Leonardo da Vinci, from C2RMF retouched.jpg',
                          'Mona Lisa', 'Leonardo da Vinci',
                          'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg/1280px-Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg'),
    'van_gogh_sunflowers': ('Vincent Willem van Gogh 127.jpg',
                           'Sunflowers', 'Vincent van Gogh',
                           'https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Vincent_Willem_van_Gogh_127.jpg/960px-Vincent_Willem_van_Gogh_127.jpg'),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def request(url, params=None):
    for attempt in range(3):
        try:
            response = requests.get(url, params=params,
                                    headers={'User-Agent': 'ObjectCollage/0.1 (art reference installer)'},
                                    timeout=(30, 90))
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 2:
                raise


def download(url):
    return request(url).content


def verify_image(data, reference_id):
    with Image.open(io.BytesIO(data)) as image:
        if image.format != 'JPEG' or min(image.size) < 300:
            raise ValueError(f'Unexpected artwork image: {reference_id}')
        return list(image.size)


def commons_record(reference_id, filename, title, artist, image_url, cached_data=None):
    data = cached_data if cached_data is not None else download(image_url)
    size = verify_image(data, reference_id)
    return data, {'title': title, 'artist': artist,
        'object_url': f'https://commons.wikimedia.org/wiki/File:{filename.replace(" ", "_")}',
        'image_url': image_url, 'rights': 'Wikimedia Commons; public domain',
        'sha256': digest(data), 'size': size}


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    manifest_path = DESTINATION / 'SOURCE.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {
        'schema_version': 1, 'references': {}}
    references = manifest['references']
    if set(references) - (set(OBJECTS) | set(COMMONS)):
        raise ValueError('Style reference catalog contains unknown IDs')
    for reference_id, object_id in OBJECTS.items():
        if reference_id in references:
            continue
        record = json.loads(download(f'https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}'))
        if record['objectID'] != object_id or not record['isPublicDomain'] or not record['primaryImageSmall']:
            raise ValueError(f'Artwork is not available through Open Access: {object_id}')
        data = download(record['primaryImageSmall'])
        size = verify_image(data, reference_id)
        (DESTINATION / f'{reference_id}.jpg').write_bytes(data)
        references[reference_id] = {
            'object_id': object_id,
            'title': record['title'],
            'artist': record['artistDisplayName'],
            'object_url': record['objectURL'],
            'image_url': record['primaryImageSmall'],
            'rights': 'The Met Open Access; public domain',
            'sha256': digest(data),
            'size': size,
        }
        print(f'Downloaded {reference_id}: {size}', flush=True)
    for reference_id, (filename, title, artist, image_url) in COMMONS.items():
        if reference_id in references:
            continue
        path = DESTINATION / f'{reference_id}.jpg'
        data, references[reference_id] = commons_record(reference_id, filename, title, artist,
                                                         image_url, path.read_bytes() if path.exists() else None)
        if not path.exists():
            path.write_bytes(data)
        print(f'Downloaded {reference_id}: {references[reference_id]["size"]}', flush=True)
    for reference_id, row in references.items():
        path = DESTINATION / f'{reference_id}.jpg'
        if not path.exists():
            path.write_bytes(download(row['image_url']))
        data = path.read_bytes()
        if digest(data) != row['sha256'] or verify_image(data, reference_id) != row['size']:
            raise ValueError(f'Style reference mismatch: {reference_id}')
    manifest_path.write_text(json.dumps({'schema_version': 1, 'references': references}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Verified {len(references)} style references.', flush=True)


if __name__ == '__main__':
    main()
