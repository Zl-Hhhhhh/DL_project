"""Install the UCI newspaper materials recommended by CollageRL, as small JPEGs."""
import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PureWindowsPath

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
DATASET_URL = 'https://archive.ics.uci.edu/dataset/306/newspaper+and+magazine+images+segmentation+dataset'
DOWNLOAD_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00306/dataset_segmentation.rar'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Reuse an already downloaded official RAR or ZIP')
    parser.add_argument('--unrar', default=shutil.which('unrar') or 'C:/Program Files/WinRAR/UnRAR.exe')
    args = parser.parse_args()
    destination = ROOT/'CollageRL/samples/materials/newspaper'
    manifest = destination/'SOURCE.json'
    if manifest.exists():
        source = json.loads(manifest.read_text(encoding='utf-8'))
        for item in source['images']:
            if Path(item['file']).name != item['file'] or digest(destination/item['file']) != item['sha256']:
                raise ValueError('Existing library failed verification')
        print(f'Verified {len(source["images"])} existing materials.')
        return
    if destination.exists():
        raise FileExistsError('Incomplete library exists; inspect it before retrying.')
    if not Path(args.unrar).is_file():
        raise FileNotFoundError('UnRAR is required; pass --unrar with its executable path.')
    cache = ROOT/'.cache/materials-download'
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='prepare-', dir=cache) as temp:
        staging = Path(temp).resolve()
        if staging.parent != cache.resolve() or not staging.is_relative_to(ROOT):
            raise ValueError('Temporary directory must be within this project')
        archive = args.archive.resolve() if args.archive else staging/'uci-306.rar'
        if not args.archive:
            print('Downloading the official UCI dataset...', flush=True)
            with urllib.request.urlopen(DOWNLOAD_URL, timeout=120) as response, archive.open('wb') as output:
                shutil.copyfileobj(response, output, 1024*1024)
        rar = archive
        if archive.suffix.lower() == '.zip':
            with zipfile.ZipFile(archive) as bundle:
                members = [item for item in bundle.infolist() if item.filename.lower().endswith('.rar')]
                if len(members) != 1:
                    raise ValueError('Expected one RAR archive in the official ZIP')
                rar = staging/'dataset.rar'
                with bundle.open(members[0]) as source, rar.open('wb') as output:
                    shutil.copyfileobj(source, output, 1024*1024)
        encoding = 'utf-16-le' if os.name == 'nt' else 'utf-8'
        names = subprocess.check_output([args.unrar,'lb','-scu','-p-',str(rar)]).decode(encoding).splitlines()
        pages = [name for name in names if PureWindowsPath(name).suffix.lower() in {'.jpg','.jpeg','.bmp','.tif','.tiff','.png'}
                 and not PureWindowsPath(name).stem.lower().endswith('_m')]
        basenames = set()
        for name in pages:
            member = PureWindowsPath(name)
            if member.is_absolute() or member.drive or '..' in member.parts or member.name.casefold() in basenames:
                raise ValueError('Unexpected archive path or duplicate image name')
            basenames.add(member.name.casefold())
        if len(pages) != 101:
            raise ValueError(f'Expected 101 source pages, found {len(pages)}')
        extracted = staging/'pages'
        extracted.mkdir()
        subprocess.run([args.unrar,'e','-idq','-y','-o-','-p-',str(rar),*pages,str(extracted)+'\\'],check=True)
        package = staging/'library'
        package.mkdir()
        entries = []
        for index, name in enumerate(sorted(pages)):
            original = extracted/PureWindowsPath(name).name
            if original.is_symlink() or original.resolve().parent != extracted.resolve():
                raise ValueError('Extracted image is outside the staging directory')
            with Image.open(original) as photo:
                original_size = list(photo.size)
                image = ImageOps.exif_transpose(photo).convert('RGB')
                image.thumbnail((1024,1024),Image.Resampling.LANCZOS)
                output = package/f'page-{index+1:03d}.jpg'
                image.save(output,quality=90,optimize=True)
            entries.append({'file':output.name,'sha256':digest(output),'original_member':name,
                            'original_sha256':digest(original),'original_size':original_size,'size':list(image.size)})
        preview = Image.new('RGB',(720,440),'#f0f2f5')
        for index, item in enumerate(random.Random(42).sample(entries,8)):
            with Image.open(package/item['file']) as photo:
                thumb = ImageOps.contain(photo,(172,212))
                preview.paste(thumb,((index%4)*180+(180-thumb.width)//2,(index//4)*220+(220-thumb.height)//2))
        preview.save(package/'preview.jpg',quality=85,optimize=True)
        record = {'dataset_url':DATASET_URL,'download_url':DOWNLOAD_URL,'archive_sha256':digest(archive),
                  'creators':['Aleksey Vilkin','Ilia Safonov'],'license':'CC BY 4.0',
                  'license_url':'https://creativecommons.org/licenses/by/4.0/',
                  'doi':'10.24432/C5N60V','source':'Material dataset linked by CollageRL README',
                  'modifications':'Source JPG/BMP/TIFF pages retained; segmentation masks excluded; EXIF orientation applied; converted to RGB; aspect ratio preserved; maximum edge 1024; JPEG quality 90.',
                  'images':entries}
        (package/'SOURCE.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
        (package/'ATTRIBUTION.md').write_text(
            '# Newspaper and magazine images segmentation dataset\n\n'
            'Aleksey Vilkin and Ilia Safonov. UCI Machine Learning Repository. DOI: 10.24432/C5N60V.\n\n'
            f'Source: {DATASET_URL}\n\nLicense: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/\n\n'
            'Derived copies: resized to maximum edge 1024, RGB JPEG quality 90. Segmentation masks omitted. '
            'Source and derivative hashes are recorded in SOURCE.json.\n',encoding='utf-8')
        if package.resolve().parent != staging or destination.resolve().parent != (ROOT/'CollageRL/samples/materials').resolve():
            raise ValueError('Unexpected library move path')
        package.rename(destination)
    print(f'Installed {len(entries)} pages, {sum(p.stat().st_size for p in destination.iterdir())/1024**2:.2f} MiB.',flush=True)


if __name__ == '__main__':
    main()
