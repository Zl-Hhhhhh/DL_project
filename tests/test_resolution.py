import numpy as np
import pytest
import yaml
from pathlib import Path
from PIL import Image

from object_collage.io import sha256, write_json
from object_collage.pipeline import replay
from object_collage.render import generate_candidates


@pytest.mark.parametrize('style', ['duotone', 'original'])
def test_1024_candidates_and_replay_preserve_fine_detail(tmp_path, style):
    config = yaml.safe_load(Path('configs/dev.yaml').read_text())
    config.update(canvas_size=1024, render_style=style, background_mode='source_masked')
    yy, xx = np.mgrid[:1024, :1024]
    rgb = np.stack([xx % 256, yy % 256, (xx+yy) % 256], axis=-1).astype(np.uint8)
    rgb[0:40, :, :] = ((xx[0:40, :, None] % 2)*255)
    image = Image.fromarray(rgb)
    mask = ((xx-512)/160)**2 + ((yy-512)/320)**2 < 1
    accepted, _, chosen = generate_candidates(image, mask, config, 'high-resolution', 'right')
    assert chosen is not None and all(c['metrics']['valid'] for c in accepted)
    assert chosen['image'].size == (1024, 1024)
    # One-pixel stripes survive: the renderer must not shrink to 512 and upscale.
    assert np.array_equal(np.asarray(chosen['image'])[:40], rgb[:40])
    image.save(tmp_path/'source.png')
    Image.fromarray(mask.astype(np.uint8)*255).save(tmp_path/'mask.png')
    chosen['image'].save(tmp_path/'output.png')
    record = {'source':'source.png', 'mask':'mask.png',
              'source_sha256':sha256(tmp_path/'source.png'), 'mask_sha256':sha256(tmp_path/'mask.png'),
              'output_sha256':sha256(tmp_path/'output.png'), 'params':chosen['params']}
    write_json(tmp_path/'params.json', record)
    assert replay(tmp_path/'params.json', tmp_path/'replay.png')['exact_match']
