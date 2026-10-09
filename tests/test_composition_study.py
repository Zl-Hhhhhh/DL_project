import numpy as np
import pytest
from PIL import Image

from object_collage.composition_study import plan_crop, render_candidate, selective_color, TARGETS


def fixture_image():
    yy, xx = np.mgrid[:512, :512]
    rgb = np.stack([xx % 256, yy % 256, (xx+yy) % 256], -1).astype(np.uint8)
    mask = np.zeros((512, 512), bool)
    mask[100:200, 80:180] = True
    return Image.fromarray(rgb), mask


def test_color_changes_only_background_and_zero_is_gray():
    image, mask = fixture_image()
    result = np.asarray(selective_color(image, mask, 0))
    np.testing.assert_array_equal(result[mask], np.asarray(image)[mask])
    np.testing.assert_array_equal(result[~mask, 0], result[~mask, 1])
    np.testing.assert_array_equal(result[~mask, 1], result[~mask, 2])
    assert not np.array_equal(result[~mask], np.asarray(image)[~mask])


def test_crop_keeps_subject_and_stays_in_content():
    _, mask = fixture_image()
    plan = plan_crop(mask, [20, 30, 490, 470], TARGETS)
    assert plan is not None
    l,t,r,b = plan['box']
    assert r-l == b-t
    assert 20 <= l < r <= 490 and 30 <= t < b <= 470
    assert mask[t:b,l:r].sum() == mask.sum()
    assert plan['position_error_before_resize'] <= plan['target_tolerance']


def test_impossible_crop_does_not_clip_or_invent_background():
    mask = np.zeros((512,512), bool)
    mask[15:497,15:497] = True
    assert plan_crop(mask, [0,0,512,512], TARGETS) is None


@pytest.mark.parametrize('size', [512,1024])
def test_poster_targets_and_more_space(size):
    image, mask = fixture_image()
    params = dict(mode='poster', size=size, target_xy=[1/3,1/3],
                  longest_edge_fraction=.55, background_rgb=[245,237,223])
    output, visible, compact = render_candidate(image, mask, params)
    assert output.size == (size,size)
    assert compact['target_error'] < 2/size
    assert compact['negative_space_subject_fraction'] == 0
    assert compact['negative_space_edge_density'] == 0
    _,_,airy = render_candidate(image, mask, dict(params, longest_edge_fraction=.38))
    assert airy['subject_area_fraction'] < compact['subject_area_fraction']
    outside = np.asarray(output)[np.asarray(visible)==0]
    assert np.all(outside == [245,237,223])


def test_padding_excluded_from_negative_space_metrics():
    image, mask = fixture_image()
    _,_,metrics = render_candidate(image, mask, dict(mode='original', size=512,
                                   content_box=[0,0,300,512]))
    # Target defaults to the upper-left subject, so assess the rightmost strip.
    assert metrics['negative_space_evaluated_fraction'] == 0
    assert metrics['negative_space_edge_density'] is None
