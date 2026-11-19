import numpy as np
import pytest
import json
from collections import Counter
from pathlib import Path

from object_collage.selector_data import (_move_for_mask, candidate_recipes, pairs_for_source,
                                          select_sources, selector100_assignment)
from object_collage.selector_label import (source_ids_for_split, tasks_for_batch,
                                           validate_label, validate_ranking)
from object_collage.segmentation import detector_route
from object_collage.selector_route import source_folds
from object_collage.selector_train import (ACTION_DIM, INPUT_DIM, PREGEN_INPUT_DIM, VISUAL_DIM, _actions,
                                          _candidate_targets, _labeled_pairs, _source_metrics,
                                          _source_weights, _visual_mode)
from object_collage.sequence import OPERATIONS, validate_recipe, validate_route


@pytest.mark.parametrize('count,expected_steps', [
    (0, {1: 8, 2: 3, 3: 1}),
    (1, {1: 5, 2: 4, 3: 3}),
    (2, {1: 6, 2: 4, 3: 2}),
])
def test_candidate_bank_obeys_route(count, expected_steps):
    recipes = candidate_recipes(count, 'coco_val2017_000000123456')
    assert len(recipes) == 12
    assert Counter(len(recipe['steps']) for recipe in recipes) == expected_steps
    for recipe in recipes:
        assert 'subject_count' not in recipe
        assert validate_route(validate_recipe(recipe), count)
    assert recipes == candidate_recipes(count, 'coco_val2017_000000123456')
    if count:
        assert [recipe['steps'][0]['op'] for recipe in recipes[:2]] == [
            'warm_film', 'stylize']
        assert all(recipe['steps'][0]['op'] != 'material_reconstruct'
                   for recipe in recipes if len(recipe['steps']) == 1)
    if count == 0:
        assert {step['op'] for recipe in recipes for step in recipe['steps']} == {
            'stylize', 'material_reconstruct'}
    else:
        root = Path(__file__).parents[1]
        manifest = json.loads((root/'data/photo120/manifest.json').read_text())
        sources = select_sources(manifest, 'selector100')
        assert {step['op'] for row in sources
                for recipe in candidate_recipes(count, row['source_id'])
                for step in recipe['steps']} == OPERATIONS
    for recipe in recipes:
        steps = recipe['steps']
        if len(steps) > 1:
            assert steps[0]['op'] != 'stylize'
            if any(step['op'] == 'material_reconstruct' for step in steps):
                assert steps[0]['op'] == 'material_reconstruct'
        for step in steps:
            if step['op'] == 'material_reconstruct':
                assert 0 < step['material_strength'] < 1


def test_pairs_only_include_successful_candidates_and_repeats_are_reversed():
    entry = {'split': 'train', 'estimated_subject_count': 1,
             'candidates': [{'candidate_id': f'c{i:02d}', 'status': 'failed' if i == 2 else 'passed'}
                            for i in range(7)]}
    pairs = pairs_for_source(entry)
    assert all('c02' not in pair for pair in pairs)
    batch = {'source_ids': ['source'], 'sources': {'source': entry}}
    tasks = tasks_for_batch(batch)
    first = next(task for task in tasks if task['task_id'].endswith(':0'))
    repeat = next(task for task in tasks if task['task_id'].endswith(':1'))
    assert (first['left'], first['right']) == (repeat['right'], repeat['left'])
    incomplete = {'source_ids': ['source'], 'sources': {'source': {
        **entry, 'candidates': entry['candidates'] + [{'candidate_id': 'c07', 'status': 'pending'}]}}}
    assert tasks_for_batch(incomplete) == []


def test_reference_pairs_compare_same_type_on_one_source():
    entry = {'candidates': [
        {'candidate_id': 'c00', 'status': 'passed', 'recipe': {'steps': [{'op': 'warm_film'}]}},
        {'candidate_id': 'c01', 'status': 'passed', 'recipe': {'steps': [{'op': 'stylize', 'reference_id': 'van_gogh_cypresses'}]}},
        {'candidate_id': 'c02', 'status': 'passed', 'recipe': {'steps': [{'op': 'material_reconstruct', 'material_id': 'wood_table'}]}},
        {'candidate_id': 'c05', 'status': 'passed', 'recipe': {'steps': [{'op': 'stylize', 'reference_id': 'hokusai_great_wave'}]}},
        {'candidate_id': 'c06', 'status': 'passed', 'recipe': {'steps': [{'op': 'material_reconstruct', 'material_id': 'rough_linen'}]}}]}
    pairs = pairs_for_source(entry)
    assert ('c01', 'c05') in pairs
    assert ('c02', 'c06') in pairs


def test_label_requires_provenance_and_valid_quality_flags():
    task = {'task_id': 'source:c00:c01:0', 'left': 'c00', 'right': 'c01'}
    label = {'task_id': task['task_id'], 'annotator': 'reviewer', 'left': 'c00', 'right': 'c01',
             'subject_count': 1, 'subject_count_confirmed': True, 'preference': 'left',
             'unacceptable': {'left': False, 'right': True},
             'identity': {'left': 'preserved', 'right': 'changed'}}
    validate_label(label, task, 'reviewer')
    with pytest.raises(ValueError, match='Candidate order'):
        validate_label({**label, 'left': 'c01'}, task, 'reviewer')
    with pytest.raises(ValueError, match='Preferred candidate'):
        validate_label({**label, 'unacceptable': {'left': True, 'right': True}}, task, 'reviewer')
    with pytest.raises(ValueError, match='reviews'):
        validate_label({**label, 'identity': []}, task, 'reviewer')


def test_ranking_requires_every_successful_candidate_and_human_route():
    entry = {'source_id': 'source', 'estimated_subject_count': 1,
             'candidates': [{'candidate_id': 'c00', 'status': 'passed'},
                            {'candidate_id': 'c01', 'status': 'passed'},
                            {'candidate_id': 'c02', 'status': 'failed'}]}
    ranking = {'source_id': 'source', 'annotator': 'reviewer', 'ranked_ids': ['c01'],
               'subject_count': 1, 'subject_count_confirmed': True,
               'unacceptable': {'c00': True, 'c01': False},
               'identity': {'c00': 'preserved', 'c01': 'changed'}}
    validate_ranking(ranking, entry)
    validate_ranking({**ranking, 'ranked_ids': [],
                      'unacceptable': {'c00': True, 'c01': True}}, entry)
    with pytest.raises(ValueError, match='at most once'):
        validate_ranking({**ranking, 'ranked_ids': ['c01', 'c01']}, entry)
    with pytest.raises(ValueError, match='at most once'):
        validate_ranking({**ranking, 'ranked_ids': [{'bad': True}, 'c01']}, entry)
    with pytest.raises(ValueError, match='unselected candidates'):
        validate_ranking({**ranking, 'unacceptable': {'c00': False, 'c01': False}}, entry)
    with pytest.raises(ValueError, match='Confirm subject count'):
        validate_ranking({**ranking, 'subject_count_confirmed': False}, entry)
    with pytest.raises(ValueError, match='acceptability'):
        validate_ranking({**ranking, 'unacceptable': {'c00': False}}, entry)
    with pytest.raises(ValueError, match='identity'):
        validate_ranking({**ranking, 'identity': {'c00': {}, 'c01': 'changed'}}, entry)


def test_ranked_sources_become_bounded_training_pairs_without_test_leakage(tmp_path):
    entry = {'source_id': 'source', 'split': 'train', 'estimated_subject_count': 1,
             'candidates': [{'candidate_id': f'c{i:02d}', 'status': 'passed'} for i in range(3)]}
    batch = {'source_ids': ['source'], 'sources': {'source': entry}}
    ranking = {'source_id': 'source', 'split': 'train', 'annotator': 'reviewer',
               'ranked_ids': ['c02', 'c01'], 'subject_count': 1,
               'subject_count_confirmed': True,
               'unacceptable': {'c00': True, 'c01': False, 'c02': False},
               'identity': {'c00': 'changed', 'c01': 'preserved', 'c02': 'preserved'}}
    (tmp_path/'ranks_train.jsonl').write_text(json.dumps(ranking)+'\n')
    groups, _ = _labeled_pairs(batch, [tmp_path/'labels_val.jsonl'])
    assert not any(groups.values())
    groups, _ = _labeled_pairs(batch, [tmp_path/'labels_train.jsonl'])
    assert len(groups['train']) == 3
    assert all(row['preference'] == 'right' for row in groups['train'])
    assert all(row['derived_from_ranking'] for row in groups['train'])
    assert {(row['left'], row['right']) for row in groups['train']} == {
        ('c00', 'c01'), ('c00', 'c02'), ('c01', 'c02')}
    assert len(_candidate_targets(groups['train'])) == 3


def test_source_weights_and_rejection_are_per_image():
    rows = [{'source_id': 'a', 'candidate_id': 'c00', 'acceptable': False},
            {'source_id': 'a', 'candidate_id': 'c01', 'acceptable': True},
            {'source_id': 'b', 'candidate_id': 'c00', 'acceptable': False}]
    assert _source_weights(rows).tolist() == [.5, .5, 1.]
    rankings = [
        {'source_id': 'a', 'ranked_ids': ['c01'],
         'unacceptable': {'c00': True, 'c01': False},
         'identity': {'c00': 'changed', 'c01': 'preserved'}, 'subject_count': 1},
        {'source_id': 'b', 'ranked_ids': [], 'unacceptable': {'c00': True},
         'identity': {'c00': 'unclear'}, 'subject_count': 0},
    ]
    predictions = np.array([[.8, -.2], [.9, .2], [.5, -.1]])
    metrics = _source_metrics(predictions, rows, rankings)
    assert metrics['source_decision_success'] == 1
    assert metrics['source_top1_hit'] == 1
    assert metrics['rejection_accuracy'] == 1


def test_pre_generation_features_exclude_rendered_output():
    first = np.arange(INPUT_DIM, dtype=np.float32)[None, :]
    changed = first.copy()
    changed[:, 2*VISUAL_DIM:3*VISUAL_DIM] = -1000
    projected = _visual_mode(first, 'pre_generation')
    assert projected.shape == (1, PREGEN_INPUT_DIM)
    assert np.array_equal(projected, _visual_mode(changed, 'pre_generation'))


def test_five_folds_are_source_disjoint_and_category_balanced():
    images = [{'source_id': f'{category}_{index}', 'category': category}
              for category in range(10) for index in range(10)]
    batch = {'source_ids': [row['source_id'] for row in images]}
    folds = source_folds(batch, {'images': images})
    assert len(folds) == 5
    assert set().union(*(set(fold) for fold in folds)) == set(batch['source_ids'])
    assert all(len(fold) == 20 for fold in folds)
    assert all(Counter(source_id.split('_')[0] for source_id in fold) ==
               {str(category): 2 for category in range(10)} for fold in folds)


def test_detector_route_prefers_complete_living_subjects():
    detections = [
        {'category': 'person', 'score': .9, 'area_fraction': .2},
        {'category': 'chair', 'score': .8, 'area_fraction': .2},
        {'category': 'person', 'score': .6, 'area_fraction': .07},
        {'category': 'dog', 'score': .95, 'area_fraction': .01},
    ]
    assert detector_route(detections) == 2
    assert detector_route(detections, score_threshold=.7) == 1
    assert detector_route(detections, mode='all') == 2


def test_action_features_encode_order_and_reference():
    style = {'op': 'stylize', 'reference_id': 'van_gogh_cypresses'}
    material = {'op': 'material_reconstruct', 'material_id': 'wood_table'}
    first = _actions({'steps': [style, material]}, 1)
    reversed_order = _actions({'steps': [material, style]}, 1)
    assert first.shape == (ACTION_DIM,)
    assert INPUT_DIM == 3*VISUAL_DIM+ACTION_DIM
    assert np.isfinite(first).all()
    assert not np.array_equal(first, reversed_order)
    weaker = _actions({'steps': [style, {**material, 'material_strength': .35}]}, 1)
    assert not np.array_equal(first, weaker)


def test_dragon_move_stays_inside_subject_bounds():
    mask = np.zeros((512, 512), dtype=bool)
    mask[200:350, 343:488] = True
    move = _move_for_mask(mask)
    assert move['dx'] < 0
    ys, xs = np.nonzero(mask)
    assert 0 <= xs.min()+move['dx'] and xs.max()+move['dx'] < 512
    assert 0 <= ys.min()+move['dy'] and ys.max()+move['dy'] < 512


def test_source_seed_rotates_references_and_treatments():
    recipes = [candidate_recipes(1, f'coco_val2017_{index:012d}') for index in range(30)]
    assert len({rows[1]['steps'][0]['reference_id'] for rows in recipes}) == 4
    materials = [step for rows in recipes for recipe in rows for step in recipe['steps']
                 if step['op'] == 'material_reconstruct']
    assert {step['material_id'] for step in materials} == {
        'newspaper_a', 'newspaper_b', 'newspaper_c', 'ceramic_blue_white', 'stained_glass'}
    assert all(step['material_strength'] == {
        'ceramic_blue_white': .65, 'stained_glass': .6}.get(step['material_id'], .45)
        for step in materials)
    assert len({rows[3]['steps'][0]['op'] for rows in recipes}) >= 5


def test_pilot_selects_two_train_sources_per_category():
    root = Path(__file__).parents[1]
    manifest = json.loads((root/'data/photo120/manifest.json').read_text())
    rows = select_sources(manifest, 'train', pilot20=True)
    assert len(rows) == 20
    assert set(row['split'] for row in rows) == {'train'}
    assert set(Counter(row['category'] for row in rows).values()) == {2}


def test_selector100_is_seeded_balanced_and_excludes_pilot():
    root = Path(__file__).parents[1]
    manifest = json.loads((root/'data/photo120/manifest.json').read_text())
    pilot = {row['source_id'] for row in select_sources(manifest, 'train', pilot20=True)}
    rows = select_sources(manifest, 'selector100')
    assignment = selector100_assignment(manifest)
    assert len(rows) == len(assignment) == 100
    assert not pilot.intersection(assignment)
    assert Counter(assignment.values()) == {'train': 80, 'val': 20}
    assert selector100_assignment(manifest) == assignment
    for category in {row['category'] for row in rows}:
        ids = [row['source_id'] for row in rows if row['category'] == category]
        assert Counter(assignment[source_id] for source_id in ids) == {'train': 8, 'val': 2}


def test_label_queue_uses_batch_split_assignment():
    rows = [{'source_id': 'a', 'split': 'train'}, {'source_id': 'b', 'split': 'val'}]
    batch = {'source_ids': ['a', 'b'], 'split_assignment': {'a': 'val', 'b': 'train'}}
    assert source_ids_for_split(batch, 'train', rows) == ['b']
    assert source_ids_for_split(batch, 'val', rows) == ['a']
    assert source_ids_for_split({'source_ids': ['a', 'b']}, 'train', rows) == ['a']


def test_human_route_disagreement_keeps_ranked_preference(tmp_path):
    entry = {'source_id': 'source', 'split': 'train', 'estimated_subject_count': 1,
             'candidates': [{'candidate_id': f'c{i:02d}', 'status': 'passed'} for i in range(2)]}
    ranking = {'source_id': 'source', 'split': 'train', 'annotator': 'reviewer',
               'ranked_ids': ['c01'], 'subject_count': 2,
               'subject_count_confirmed': True,
               'unacceptable': {'c00': True, 'c01': False},
               'identity': {'c00': 'unclear', 'c01': 'preserved'}}
    (tmp_path/'ranks_train.jsonl').write_text(json.dumps(ranking)+'\n')
    groups, counts = _labeled_pairs({'source_ids': ['source'], 'sources': {'source': entry}},
                                    [tmp_path/'labels_train.jsonl'])
    assert len(groups['train']) == 1
    assert groups['train'][0]['preference'] == 'right'
    assert counts['route_mismatch_included'] == 1


def test_training_loader_does_not_read_test_label_file(tmp_path):
    batch = {'source_ids': ['source'], 'sources': {'source': {'split': 'test',
             'estimated_subject_count': 1, 'candidates': [
                 {'candidate_id': 'c00', 'status': 'passed'},
                 {'candidate_id': 'c01', 'status': 'passed'}]}}}
    task = tasks_for_batch(batch)[0]
    label = {'task_id': task['task_id'], 'source_id': task['source_id'], 'split': 'test',
             'annotator': 'reviewer', 'left': task['left'], 'right': task['right'],
             'subject_count': 1, 'subject_count_confirmed': True, 'preference': 'left',
             'unacceptable': {'left': False, 'right': False},
             'identity': {'left': 'preserved', 'right': 'preserved'}}
    test_path = tmp_path/'labels_test.jsonl'
    test_path.write_text(json.dumps(label)+'\n')
    groups, _ = _labeled_pairs(batch, [tmp_path/'labels_train.jsonl', tmp_path/'labels_val.jsonl'])
    assert not any(groups.values())
    groups, _ = _labeled_pairs(batch, [test_path])
    assert len(groups['test']) == 1
