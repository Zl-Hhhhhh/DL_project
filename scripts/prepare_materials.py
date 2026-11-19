"""Install the UCI newspaper materials recommended by CollageRL, as small JPEGs."""
import argparse
import hashlib
import json
import random
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PureWindowsPath

from PIL import Image, ImageOps
import libarchive

ROOT = Path(__file__).resolve().parents[1]
DATASET_URL = 'https://archive.ics.uci.edu/dataset/306/newspaper+and+magazine+images+segmentation+dataset'
DOWNLOAD_URL = 'https://archive.ics.uci.edu/static/public/306/newspaper+and+magazine+images+segmentation+dataset.zip'
SEVENZIP_URL = 'https://github.com/ip7z/7zip/releases/download/26.05/7z2605-linux-arm64.tar.xz'
SEVENZIP_SHA256 = '0b71f6cfc34669b3bf69f2347efd3429f77020665003a13b9cdc53b784270421'
SEVENZIP_BINARY_SHA256 = '17925963fe96bd0b441d1a139dbd2780b8894de0446fe2650ccdc65372167415'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def ensure_sevenzip(path):
    if path.is_file():
        if path == ROOT/'.cache/tools/7zz' and digest(path) != SEVENZIP_BINARY_SHA256:
            raise ValueError('Cached 7-Zip binary SHA-256 mismatch')
        return path
    if path != ROOT/'.cache/tools/7zz':
        raise FileNotFoundError(f'7-Zip is missing: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = path.parent/'7z2605-linux-arm64.tar.xz'
    if not bundle.is_file():
        with urllib.request.urlopen(SEVENZIP_URL, timeout=120) as source, bundle.open('wb') as output:
            shutil.copyfileobj(source, output)
    if digest(bundle) != SEVENZIP_SHA256:
        raise ValueError('7-Zip download SHA-256 mismatch')
    with tarfile.open(bundle, 'r:xz') as archive:
        member = archive.getmember('7zz')
        if not member.isfile():
            raise ValueError('7-Zip archive does not contain a regular 7zz binary')
        with archive.extractfile(member) as source, path.open('wb') as output:
            shutil.copyfileobj(source, output)
    path.chmod(0o755)
    if digest(path) != SEVENZIP_BINARY_SHA256:
        raise ValueError('Extracted 7-Zip binary SHA-256 mismatch')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Reuse an already downloaded official RAR or ZIP')
    parser.add_argument('--sevenzip', type=Path, default=ROOT/'.cache/tools/7zz',
                        help='7-Zip ARM64 binary for extracting the official RAR')
    args = parser.parse_args()
    destination = ROOT/'third_party/repos/CollageRL/samples/materials/newspaper'
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
    cache = ROOT/'.cache/materials-download'
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='prepare-', dir=cache) as temp:
        staging = Path(temp).resolve()
        if staging.parent != cache.resolve() or not staging.is_relative_to(ROOT):
            raise ValueError('Temporary directory must be within this project')
        archive = args.archive.resolve() if args.archive else staging/'uci-306.zip'
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
        with libarchive.file_reader(str(rar)) as bundle:
            names = [entry.pathname for entry in bundle]
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
        sevenzip = ensure_sevenzip(args.sevenzip.resolve())
        subprocess.run([str(sevenzip), 'e', str(rar), f'-o{extracted}', '-y', '-bb0', *pages],
                       check=True, stdout=subprocess.DEVNULL)
        if {path.name.casefold() for path in extracted.iterdir()} != basenames:
            raise ValueError('Extracted files do not match the 101 expected pages')
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
        if package.resolve().parent != staging or destination.resolve().parent != (ROOT/'third_party/repos/CollageRL/samples/materials').resolve():
            raise ValueError('Unexpected library move path')
        package.rename(destination)
    print(f'Installed {len(entries)} pages, {sum(p.stat().st_size for p in destination.iterdir())/1024**2:.2f} MiB.',flush=True)


if __name__ == '__main__':
    main()
