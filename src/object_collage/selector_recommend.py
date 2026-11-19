"""Score treatment recipes before rendering any candidate output."""
import gc
import math
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from .io import letterbox, read_json, sha256, write_json
from .segmentation import Segmenter
from .selector_data import candidate_recipes
from .selector_train import (PREGEN_INPUT_DIM, VISUAL_DIM, VOCAB,
                             FeatureStore, _actions, _network)
from .sequence import _release_npu, run_sequence


def recommend_recipes(root, batch_path, image_path, output_dir, execute=False, progress=print):
    import torch
    import torch_npu  # noqa: F401

    root, batch_path = Path(root).resolve(), Path(batch_path).resolve()
    image_path, output_dir = Path(image_path).resolve(), Path(output_dir).resolve()
    if not image_path.is_file() or output_dir.exists():
        raise ValueError('Input image must exist and the recommendation output directory must be new')
    config = yaml.safe_load((root/'configs/dev.yaml').read_text(encoding='utf-8'))
    if config['device'] != 'npu:3':
        raise ValueError('Recipe recommendation is fixed to NPU 3')
    checkpoint_path = batch_path.parent/'recipe_ranker.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if (checkpoint['input_dim'] != PREGEN_INPUT_DIM or checkpoint['vocab'] != VOCAB or
            checkpoint['batch_sha256'] != sha256(batch_path) or
            checkpoint['visual_mode'] != 'pre_generation'):
        raise ValueError('Recipe scorer does not match the candidate batch or feature schema')
    image, valid, mapping = letterbox(image_path, 512)
    segmenter = Segmenter(config, root)
    mask, overlay, info = segmenter.predict(image, valid, mapping)
    del segmenter
    _release_npu()
    source_id = 'local_'+sha256(image_path)[:12]
    recipes = candidate_recipes(info['subject_count'], source_id, mask)
    output_dir.mkdir(parents=True)
    source_path = output_dir/'source.png'
    image.save(source_path)
    overlay.save(output_dir/'detections.png')
    mask_path = output_dir/'mask.png'
    if mask is not None:
        Image.fromarray(mask.astype(np.uint8)*255).save(mask_path)
    store = FeatureStore(root, batch_path.parent)
    try:
        meta = read_json(store.folder/'meta.json')
        if checkpoint['feature_weights_sha256'] != meta['weights_sha256']:
            raise ValueError('Recipe scorer and visual encoder weights differ')
        source = store.vector(source_path)
        masked = store.vector(source_path, mask_path) if mask is not None else np.zeros(VISUAL_DIM, dtype=np.float32)
    finally:
        store.close()
    matrix = np.stack([np.concatenate((source, masked, _actions(recipe, info['subject_count'])))
                       for recipe in recipes]).astype(np.float32)
    assert matrix.shape == (len(recipes), PREGEN_INPUT_DIM)
    model = _network(tuple(checkpoint['hidden_dims']), PREGEN_INPUT_DIM).to('npu:3')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    with torch.inference_mode():
        predictions = model(torch.as_tensor(matrix, device='npu:3')).float().cpu().numpy()
    del model
    gc.collect()
    torch.npu.empty_cache()
    order = sorted(range(len(recipes)), key=lambda index: float(predictions[index, 0]), reverse=True)
    eligible = [index for index in order if predictions[index, 1] > 0]
    scored = [{'candidate_id': f'c{index:02d}', 'recipe': recipes[index],
               'preference_score': float(predictions[index, 0]),
               'acceptability_logit': float(predictions[index, 1]),
               'acceptability_probability': 1/(1+math.exp(-float(predictions[index, 1])))}
              for index in order]
    result = {'schema_version': 1, 'source_id': source_id, 'source_path': str(image_path),
              'source_sha256': sha256(image_path), 'status': 'recommended' if eligible else 'rejected',
              'segmentation': info, 'canvas_mapping': mapping,
              'recipe_checkpoint_sha256': sha256(checkpoint_path),
              'scored_recipes': scored,
              'selected': scored[order.index(eligible[0])] if eligible else None,
              'executed': False}
    write_json(output_dir/'recommendation.json', result)
    if execute and eligible:
        failures = []
        for attempt, index in enumerate(eligible, 1):
            progress(f'attempt {attempt}: {recipes[index]["steps"]}')
            destination = output_dir/f'attempt_{attempt:02d}'
            try:
                sequence = run_sequence(root, image_path, recipes[index], destination,
                                        progress, (mask, overlay, info),
                                        subject_selection_mode=config['subject_selection_mode'])
                result['status'] = sequence['status']
                result['selected'] = scored[order.index(index)]
                result['output'] = str(destination/'output.png')
                result['executed'] = True
                break
            except Exception as error:
                failures.append({'candidate_id': f'c{index:02d}', 'error': str(error)})
        else:
            result['status'] = 'failed'
        result['failed_attempts'] = failures
        write_json(output_dir/'recommendation.json', result)
    return result
