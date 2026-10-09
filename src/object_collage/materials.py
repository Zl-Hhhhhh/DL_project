"""A local, attributed material library recommended by the CollageRL README."""
import json
import random
from pathlib import Path

from .io import sha256


def builtin_catalog(root):
    folder = Path(root) / 'CollageRL/samples/materials/newspaper'
    manifest = folder / 'SOURCE.json'
    if not manifest.is_file():
        return folder, [], None
    source = json.loads(manifest.read_text(encoding='utf-8'))
    entries = source['images']
    for item in entries:
        name = item['file']
        if not isinstance(name, str) or Path(name).name != name or not name.endswith('.jpg'):
            raise ValueError('Invalid built-in material filename')
        path = folder / name
        if path.is_symlink() or path.resolve().parent != folder.resolve() or not path.is_file():
            raise ValueError('Built-in material is missing or outside its library')
    return folder, entries, source


def load_builtin(root):
    folder, entries, source = builtin_catalog(root)
    if not entries:
        raise ValueError('内置素材尚未安装，请运行 scripts/prepare_materials.py 或选择自定义上传。')
    selected = random.Random(42).sample(entries, min(8, len(entries)))
    data = []
    for item in selected:
        path = folder / item['file']
        if sha256(path) != item['sha256']:
            raise ValueError('内置素材校验失败，请重新准备素材库。')
        data.append(path.read_bytes())
    return data, {'source':'builtin', 'library':'uci-newspaper-306',
                  'dataset_url':source['dataset_url'], 'license':source['license'],
                  'selection_seed':42, 'selected_files':[item['file'] for item in selected]}
