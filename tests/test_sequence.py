import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from object_collage.segmentation import count_distinct_subjects
from object_collage.sequence import blend_material, material_reference, validate_recipe, validate_route


def test_route_limits_match_subject_count():
    style = {'op': 'stylize', 'reference_id': 'van_gogh_cypresses'}
    material = {'op': 'material_reconstruct', 'material_id': 'newspaper_a'}
    assert validate_route(validate_recipe({'schema_version': 1, 'steps': [style, material]}), 0) == 'no_subject'
    assert validate_route(validate_recipe({'schema_version': 1, 'steps': [style, material, style]}), 0) == 'no_subject'
    with pytest.raises(ValueError, match='Without a subject'):
        validate_route([{'op': 'warm_film'}], 0)
    assert validate_route([{'op': 'warm_film'}, {'op': 'subject_pop'}], 1) == 'one_subject'
    assert validate_route(validate_recipe({'schema_version': 1, 'steps': [style, material, style]}), 2) == 'multiple_subjects'
    assert validate_recipe({'schema_version': 1, 'subject_count': 2, 'steps': [style]})
    with pytest.raises(ValueError, match='Confirmed subject_count'):
        validate_recipe({'schema_version': 1, 'subject_count': 1.5, 'steps': [style]})


def test_distinct_count_ignores_nested_sam_proposals():
    main = np.zeros((32, 32), dtype=bool)
    main[2:14, 2:14] = True
    nested = np.zeros_like(main)
    nested[4:10, 4:10] = True
    separate = np.zeros_like(main)
    separate[18:30, 18:30] = True
    masks = {0: main, 1: nested, 2: separate}
    records = [{'instance_id': key, 'mask_area': int(mask.sum()), 'confidence': .95,
                'touches_content_border': False} for key, mask in masks.items()]
    assert count_distinct_subjects(records, masks, records[0], .7, 20, 500) == 2
    assert count_distinct_subjects(records, masks, None, .7, 20, 500) == 0


def test_recipe_rejects_unbounded_or_unknown_choices():
    with pytest.raises(ValueError, match='one to three'):
        validate_recipe({'schema_version': 1, 'steps': []})
    with pytest.raises(ValueError, match='Unknown treatment'):
        validate_recipe({'schema_version': 1, 'steps': [{'op': 'unregistered'}]})
    with pytest.raises(ValueError, match='Material ID'):
        validate_recipe({'schema_version': 1, 'steps': [{'op': 'material_reconstruct', 'material_id': 'arbitrary_file'}]})
    with pytest.raises(ValueError, match='integer dx and dy'):
        validate_recipe({'schema_version': 1, 'steps': [{'op': 'dragon_move', 'dx': 10, 'dy': 1.5}]})
    assert validate_recipe({'schema_version': 1, 'steps': [{'op': 'material_reconstruct', 'material_id': 'wood_table'}]})
    assert validate_recipe({'schema_version': 1, 'steps': [{'op': 'duotone_collage', 'composition': 'left'}]})
    for strength in (True, -0.1, 1.1, float('nan')):
        with pytest.raises(ValueError, match='material_strength'):
            validate_recipe({'schema_version': 1, 'steps': [{
                'op': 'material_reconstruct', 'material_id': 'wood_table',
                'material_strength': strength}]})


def test_material_blend_retains_source_structure():
    base = Image.new('RGB', (32, 32), (220, 40, 30))
    material = Image.new('RGB', (32, 32), (20, 80, 180))
    output = blend_material(base, material, .4)
    assert output.getpixel((16, 16)) == (140, 56, 90)
    assert blend_material(base, material, 0).getpixel((16, 16)) == base.getpixel((16, 16))


@pytest.mark.parametrize('material_id', ['ceramic_blue_white', 'stained_glass'])
def test_user_material_uses_both_source_images(material_id):
    tiles, provenance = material_reference(Path(__file__).parents[1], material_id)
    assert len(tiles) == len(provenance['tile_sha256']) == 8
    assert len(provenance['source_sha256']) == 2
    assert all(Image.open(io.BytesIO(tile)).size == (512, 512) for tile in tiles)
