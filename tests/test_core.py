import copy
import json

import numpy as np
import pytest
import yaml
from PIL import Image

from object_collage.io import letterbox, sha256, validate_manifest, write_json
from object_collage.pipeline import replay, run_batch
from object_collage.render import generate_candidates, parameter_table, render
from object_collage.report import build_report
from object_collage.segmentation import choose_instance


@pytest.fixture
def config():
    return yaml.safe_load(__import__('pathlib').Path('configs/dev.yaml').read_text())


@pytest.fixture
def subject():
    yy, xx = np.mgrid[:512, :512]
    mask = ((xx-256)/100)**2 + ((yy-256)/170)**2 < 1
    image = Image.fromarray(np.stack([xx % 256, yy % 256, (xx+yy) % 256], axis=-1).astype(np.uint8))
    return image, mask


@pytest.mark.parametrize('dimensions', [(1000, 100), (100, 1000), (256, 256)])
def test_letterbox_preserves_bounds(tmp_path, dimensions):
    path = tmp_path / 'input.png'
    Image.new('RGB', dimensions, 'red').save(path)
    original_hash = sha256(path)
    canvas, alpha, mapping = letterbox(path)
    assert canvas.size == (512, 512)
    x0, y0, x1, y1 = mapping['content_box_xyxy']
    sx, sy = mapping['scale_xy']
    assert x1-x0 == pytest.approx(dimensions[0]*sx)
    assert y1-y0 == pytest.approx(dimensions[1]*sy)
    assert alpha.sum() == (x1-x0)*(y1-y0)
    assert sha256(path) == original_hash


def test_alpha_and_corrupt_input(tmp_path):
    path = tmp_path / 'transparent.png'
    Image.new('RGBA', (512, 512), (255, 0, 0, 0)).save(path)
    image, alpha, _ = letterbox(path)
    assert not alpha.any()
    assert np.all(np.asarray(image) == 255)
    path.write_bytes(b'not an image')
    with pytest.raises(OSError):
        letterbox(path)


def test_selection_rules_and_explicit_override():
    records = [dict(instance_id=0, category='person', confidence=0.9, mask_area=100), dict(instance_id=1, category='person', confidence=0.8, mask_area=300), dict(instance_id=2, category='cat', confidence=0.95, mask_area=900)]
    assert choose_instance(records, 0.5, 'person')[0]['instance_id'] == 1
    assert choose_instance(records, 0.5, 'person', 0)[0]['instance_id'] == 0
    assert choose_instance(records, 0.99, 'person')[1] == 'low_confidence'
    assert choose_instance(records, 0.5, 'dog')[1] == 'category_not_detected'
    assert choose_instance(records, 0.5, 'person', 99)[1] == 'invalid_instance'
    assert choose_instance([], 0.5)[1] == 'no_subject'


@pytest.mark.parametrize('composition', ['center', 'left', 'right'])
def test_eight_candidates_satisfy_constraints_and_repeat(config, subject, composition):
    image, mask = subject
    first, rejected, chosen = generate_candidates(image, mask, config, 'synthetic_subject', composition)
    second, _, selected_again = generate_candidates(image, mask, config, 'synthetic_subject', composition)
    assert len(first) == len(second) == 8
    assert len(first)+len(rejected) <= 32
    assert chosen['candidate_id'] == selected_again['candidate_id']
    expected = min(first, key=lambda c: (c['metrics']['position_error'], -c['metrics']['visible_fraction'], abs(c['params']['texture_fraction']-0.25), c['candidate_id']))
    assert chosen['candidate_id'] == expected['candidate_id']
    for a, b in zip(first, second):
        assert a['params'] == b['params']
        assert np.array_equal(a['image'], b['image'])
        assert a['metrics']['visible_fraction'] >= 0.90
        assert a['metrics']['blank_intrusion_pixels'] == 0


def test_holes_remain_transparent_and_impossible_position_fails(config, subject):
    image, mask = subject
    mask[220:290, 220:290] = False
    params = next(parameter_table(2012, config, 'center'))
    output, visible, metrics = render(image, mask, params)
    left, top = metrics['paste_xy']
    width, height = metrics['transformed_size']
    assert np.asarray(visible)[top+height//2, left+width//2] == 0
    params['center_xy'] = [2, 2]
    _, _, metrics = render(image, mask, params)
    assert not metrics['valid']
    assert 'empty_main_instance' in metrics['failures']
    with pytest.raises(ValueError, match='empty_mask'):
        render(image, np.zeros_like(mask), params)


@pytest.mark.parametrize('style', ['duotone', 'original'])
@pytest.mark.parametrize('background_mode', ['solid', 'source_masked'])
def test_replay_hash_and_input_integrity(config, subject, tmp_path, style, background_mode):
    config['render_style'] = style
    config['background_mode'] = background_mode
    image, mask = subject
    params = next(parameter_table(2012, config, 'center'))
    image.save(tmp_path / 'source.png')
    Image.fromarray((mask*255).astype(np.uint8)).save(tmp_path / 'mask.png')
    output, _, _ = render(image, mask, params)
    output.save(tmp_path / 'output.png')
    record = {'source': 'source.png', 'mask': 'mask.png', 'source_sha256': sha256(tmp_path/'source.png'), 'mask_sha256': sha256(tmp_path/'mask.png'), 'output_sha256': sha256(tmp_path/'output.png'), 'params': params}
    write_json(tmp_path / 'params.json', record)
    assert replay(tmp_path / 'params.json', tmp_path / 'replay.png')['exact_match']
    Image.new('RGB', (512, 512), 'black').save(tmp_path/'source.png')
    with pytest.raises(ValueError, match='input hash mismatch'):
        replay(tmp_path / 'params.json', tmp_path / 'changed.png')


def test_original_style_preserves_subject_colors(config):
    config['render_style'] = 'original'
    config['palettes'] = [{'name': 'original', 'background': [245, 237, 223]}]
    pixels = np.zeros((512, 512, 3), dtype=np.uint8)
    pixels[128:384, 128:256] = [231, 17, 52]
    pixels[128:384, 256:384] = [12, 149, 238]
    mask = np.zeros((512, 512), dtype=bool)
    mask[128:384, 128:384] = True
    params = next(parameter_table(2012, config, 'center'))
    params.update(longest_edge_fraction=0.5, center_xy=[0.5, 0.5])
    output, visible, _ = render(Image.fromarray(pixels), mask, params)
    assert np.array_equal(np.asarray(output)[mask], pixels[mask])
    assert np.array_equal(np.asarray(visible) > 0, mask)
    assert tuple(np.asarray(output)[0, 0]) == (245, 237, 223)


@pytest.mark.parametrize('style', ['duotone', 'original'])
@pytest.mark.parametrize('size', [512, 1024])
def test_source_background_changes_only_cutout_and_destination(config, subject, style, size):
    image, mask = subject
    # A hole inside the subject's bounding box must retain the original background.
    mask[230:280, 230:280] = False
    if size == 1024:
        image = image.resize((size, size), Image.Resampling.LANCZOS)
        mask = np.repeat(np.repeat(mask, 2, axis=0), 2, axis=1)
    config.update(render_style=style, background_mode='source_masked', canvas_size=size)
    params = next(parameter_table(2012, config, 'left'))
    output, visible, _ = render(image, mask, params)
    pixels = np.asarray(output)
    assert output.size == (size, size)
    destination = np.asarray(visible) > 0
    untouched = ~(mask | destination)
    vacated = mask & ~destination
    assert untouched.any() and vacated.any()
    assert np.array_equal(pixels[untouched], np.asarray(image)[untouched])
    assert np.all(pixels[vacated] == params['mask_fill'])


def test_split_leakage_and_duplicates():
    rows = [{'source_id': 'a', 'sha256': 'hash_a', 'split': 'train', 'capture_group': 'group1'}, {'source_id': 'b', 'sha256': 'hash_b', 'split': 'val', 'capture_group': 'group2'}]
    validate_manifest({'images': rows})
    same_group = copy.deepcopy(rows)
    same_group[1]['capture_group'] = 'group1'
    with pytest.raises(ValueError, match='leakage'):
        validate_manifest({'images': same_group})
    same_content = copy.deepcopy(rows)
    same_content[1]['sha256'] = 'hash_a'
    with pytest.raises(ValueError, match='Duplicate image content'):
        validate_manifest({'images': same_content})
    with pytest.raises(ValueError, match='Duplicate source_id'):
        validate_manifest({'images': [rows[0], rows[0]]})


def test_dev_manifest_is_excluded_from_formal_splits(tmp_path, config):
    row = {'source_id': 'dev_a', 'sha256': 'abc', 'split': 'dev', 'path': 'unused.png'}
    write_json(tmp_path/'data/dev/manifest.json', {'images': [row]})
    formal = dict(row, split='test')
    with pytest.raises(ValueError, match='Development source leaked'):
        run_batch(tmp_path, config, {'images': [formal]}, tmp_path/'experiments/test')
    assert not (tmp_path/'experiments/test').exists()


def test_failure_report_and_resummary_are_repeatable(tmp_path):
    write_json(tmp_path/'run.json', {'model_load_seconds': 0.25})
    for sid, status in [('a', 'ok'), ('b', 'no_subject')]:
        write_json(tmp_path/sid/'result.json', {'source_id': sid, 'split': 'dev', 'status': status, 'candidate_count': 8 if status == 'ok' else 0})
    first, report = build_report(tmp_path)
    csv_before = (tmp_path/'metrics.csv').read_bytes()
    second, _ = build_report(tmp_path)
    assert first == second
    assert first['success_count'] == 1 and first['source_count'] == 2
    assert first['statuses']['no_subject'] == 1
    assert (tmp_path/'metrics.csv').read_bytes() == csv_before
    assert 'no_subject' in report.read_text(encoding='utf-8')
