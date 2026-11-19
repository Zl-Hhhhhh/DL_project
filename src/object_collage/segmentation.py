"""SAM 2 masks with automatic or whole-object box prompted selection."""
import time
from functools import lru_cache

import cv2
import numpy as np
from PIL import ImageDraw

from .io import configure_local_runtime, sha256


LIVING_CATEGORIES = frozenset({
    'person', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe',
})


def detector_route(detections, score_threshold=.5, area_threshold=.05,
                   mode='living_preferred'):
    accepted = [item for item in detections if item['score'] >= score_threshold
                and item['area_fraction'] >= area_threshold]
    if mode == 'living_preferred':
        living = [item for item in accepted if item['category'] in LIVING_CATEGORIES]
        accepted = living or accepted
    elif mode != 'all':
        raise ValueError(f'Unknown route mode: {mode}')
    return min(len(accepted), 2)


@lru_cache(maxsize=2)
def _load_detector(path):
    import torch
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn

    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, weights_backbone=None)
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    return model.eval()


def choose_box_subject(records, confidence, min_area, max_area):
    eligible = [item for item in records if min_area <= item['mask_area'] <= max_area
                and item['confidence'] >= confidence]
    if not eligible:
        return None
    living = [item for item in eligible if item['category'] in LIVING_CATEGORIES]
    pool = living or eligible
    return max(pool, key=lambda item: (
        item['mask_area'] * item['confidence'] *
        (.85 if item['touches_content_border'] else 1),
        item['detection_confidence'], -item['instance_id']))


def count_box_subjects(records, masks, selected, confidence, min_area, max_area):
    if selected is None:
        return 0
    accepted = [selected['instance_id']]
    candidates = sorted(records, key=lambda item: -item['detection_confidence'])
    if selected['category'] in LIVING_CATEGORIES:
        candidates = [item for item in candidates if item['category'] in LIVING_CATEGORIES]
    for item in candidates:
        index = item['instance_id']
        if index in accepted or item['confidence'] < confidence or not min_area <= item['mask_area'] <= max_area:
            continue
        if any(np.logical_and(masks[index], masks[other]).sum() /
               min(item['mask_area'], int(masks[other].sum())) > .5 for other in accepted):
            continue
        accepted.append(index)
    return len(accepted)


def choose_subject(instances, confidence, min_area, max_area, allow_truncated=False, instance_id=None):
    if not instances:
        return None, 'no_subject'
    if instance_id is not None:
        candidates = [item for item in instances if item['instance_id'] == instance_id]
        if not candidates:
            return None, 'invalid_instance'
    else:
        candidates = [item for item in instances if min_area <= item['mask_area'] <= max_area]
    if not candidates:
        return None, 'no_subject'
    confident = [item for item in candidates if item['confidence'] >= confidence]
    if not confident:
        return None, 'low_confidence'
    interior = [item for item in confident if not item['touches_content_border']]
    pool = interior or confident
    selected = max(pool, key=lambda item: (item['mask_area'], item['confidence'],
                                            item['stability_score'], -item['instance_id']))
    return selected, 'ok' if interior or allow_truncated else 'subject_truncated'


def count_distinct_subjects(records, masks, selected, confidence, min_area, max_area):
    if selected is None:
        return 0
    accepted = [selected['instance_id']]
    # Automatic SAM proposals often split a person into clothes, face and limbs.
    # Count another subject only when it is a substantial, separate region.
    min_distinct_area = max(min_area, round(masks[selected['instance_id']].size * .08))
    candidates = sorted(records, key=lambda item: (-item['mask_area'], -item['confidence']))
    for item in candidates:
        index = item['instance_id']
        if index in accepted or item['confidence'] < confidence or not min_distinct_area <= item['mask_area'] <= max_area:
            continue
        if item['touches_content_border']:
            continue
        proposal = masks[index]
        if any(np.logical_and(proposal, masks[other]).sum() / min(item['mask_area'], int(masks[other].sum())) > .5
               for other in accepted):
            continue
        accepted.append(index)
    return len(accepted)


class Segmenter:
    def __init__(self, config, root):
        configure_local_runtime(root)
        import torch
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2

        device = str(config['device'])
        if device.isdigit():
            device = f'cuda:{device}'
        if device.startswith('npu'):
            import torch_npu  # noqa: F401 - registers the NPU device with PyTorch
            if not torch.npu.is_available():
                raise RuntimeError('Ascend NPU is unavailable; source the CANN environment or use --device cpu.')
        elif device.startswith('cuda'):
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA is unavailable; use --device cpu or npu:<index>.')
        elif device != 'cpu':
            raise ValueError(f'Unsupported device: {device}')

        path = root / config['model_path']
        if not path.is_file():
            raise FileNotFoundError(f'SAM 2 checkpoint is missing: {path}')
        digest = sha256(path)
        if digest != config['model_sha256']:
            raise ValueError(f'SAM 2 checkpoint hash mismatch: {path}')
        self.provenance = {
            'model': 'SAM 2.1', 'path': config['model_path'], 'config': config['model_config'],
            'url': config['model_url'], 'sha256': digest,
            'selection': 'automatic masks; category labels are metadata only',
        }
        self.selection_mode = config.get('subject_selection_mode', 'sam2_automatic')
        if self.selection_mode not in {'sam2_automatic', 'detector_box_v1'}:
            raise ValueError(f'Unknown subject selection mode: {self.selection_mode}')
        if self.selection_mode == 'detector_box_v1':
            detector_path = root / config['subject_detector_path']
            if not detector_path.is_file():
                raise FileNotFoundError(f'Subject detector checkpoint is missing: {detector_path}')
            detector_digest = sha256(detector_path)
            if detector_digest != config['subject_detector_sha256']:
                raise ValueError(f'Subject detector checkpoint hash mismatch: {detector_path}')
            self.detector = _load_detector(str(detector_path))
            self.provenance['selection'] = 'detector boxes prompt SAM 2 whole-object masks'
            self.provenance['detector'] = {
                'path': config['subject_detector_path'], 'sha256': detector_digest,
                'url': config['subject_detector_url'],
            }
        self.torch = torch
        self.config = config
        self.device = torch.device(device)
        start = time.perf_counter()
        model = build_sam2(config['model_config'], str(path), device=device)
        self.generator = SAM2AutomaticMaskGenerator(
            model, points_per_side=config['sam2_points_per_side'],
            points_per_batch=config['sam2_points_per_batch'],
            pred_iou_thresh=config['min_confidence'],
            stability_score_thresh=config['sam2_stability_threshold'],
        )
        self.load_seconds = time.perf_counter() - start
        self.calls = 0

    def _synchronize(self):
        if self.device.type == 'cuda':
            self.torch.cuda.synchronize(self.device)
        elif self.device.type == 'npu':
            self.torch.npu.synchronize(self.device)

    def _box_proposals(self, image, valid_alpha, mapping):
        from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
        from torchvision.ops import nms
        from torchvision.transforms.functional import to_tensor

        rgb = image.convert('RGB')
        categories = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT.meta['categories']
        with self.torch.inference_mode():
            detected = self.detector([to_tensor(rgb)])[0]
        x0, y0, x1, y1 = mapping['content_box_xyxy']
        margin = self.config['border_margin']
        records, masks = [], {}
        eligible = self.torch.nonzero(detected['scores'] >= .25).flatten()
        kept = nms(detected['boxes'][eligible], detected['scores'][eligible],
                   self.config['subject_route_nms_iou'])[:30]
        route_detections = []
        for index in eligible[kept].tolist():
            bx0, by0, bx1, by1 = detected['boxes'][index].tolist()
            area = max(0, min(bx1, x1)-max(bx0, x0))*max(0, min(by1, y1)-max(by0, y0))
            route_detections.append({
                'category': categories[int(detected['labels'][index])],
                'score': float(detected['scores'][index]),
                'area_fraction': area/max(1, int(valid_alpha.sum())),
                'bbox_xyxy': [bx0, by0, bx1, by1],
            })
        route_count = detector_route(
            route_detections, self.config['subject_route_score_threshold'],
            self.config['subject_route_area_threshold'], self.config['subject_route_mode'])
        self.generator.predictor.set_image(np.array(rgb, copy=True))
        for box, label, score in zip(detected['boxes'], detected['labels'], detected['scores']):
            score = float(score)
            if score < self.config['subject_detector_min_score'] or len(records) >= 12:
                break
            box = box.numpy()
            with self.torch.inference_mode():
                proposals, qualities, _ = self.generator.predictor.predict(
                    box=box.astype(np.float32), multimask_output=True)
            best = int(np.argmax(qualities))
            mask = (proposals[best] > 0) & valid_alpha
            area = int(mask.sum())
            if not area:
                continue
            index = len(records)
            masks[index] = mask
            x, y, width, height = cv2.boundingRect(mask.astype(np.uint8))
            ys, xs = np.where(mask)
            components, _ = cv2.connectedComponents(mask.astype(np.uint8))
            records.append({
                'instance_id': index, 'category': categories[int(label)], 'class_id': int(label),
                'confidence': float(np.clip(qualities[best], 0, 1)),
                'detection_confidence': score, 'stability_score': None,
                'detector_bbox_xyxy': [float(value) for value in box],
                'bbox_xyxy': [x, y, x + width, y + height], 'mask_area': area,
                'connected_components': components - 1,
                'touches_content_border': bool(xs.min() < x0 + margin or ys.min() < y0 + margin
                                               or xs.max() >= x1 - margin or ys.max() >= y1 - margin),
            })
        return records, masks, route_count, route_detections

    def predict(self, image, valid_alpha, mapping, category=None, instance_id=None):
        self.generator.pred_iou_thresh = self.config['min_confidence']
        self._synchronize()
        start = time.perf_counter()
        records, masks = [], {}
        content_area = int(valid_alpha.sum())
        min_area = round(content_area * self.config['sam2_min_area_fraction'])
        max_area = round(content_area * self.config['sam2_max_area_fraction'])
        use_boxes = self.selection_mode == 'detector_box_v1' and instance_id is None
        route_count, route_detections = None, None
        if use_boxes:
            records, masks, route_count, route_detections = self._box_proposals(image, valid_alpha, mapping)
            selected = choose_box_subject(records, self.config['confidence'], min_area, max_area)
            status = 'ok' if selected else 'no_subject'
        if not use_boxes or selected is None:
            x0, y0, x1, y1 = mapping['content_box_xyxy']
            margin = self.config['border_margin']
            with self.torch.inference_mode():
                proposals = self.generator.generate(np.array(image.convert('RGB'), copy=True))
            records, masks = [], {}
            for index, proposal in enumerate(proposals):
                mask = proposal['segmentation'] & valid_alpha
                area = int(mask.sum())
                if not area:
                    continue
                masks[index] = mask
                x, y, width, height = cv2.boundingRect(mask.astype(np.uint8))
                ys, xs = np.where(mask)
                touches_border = bool(xs.min() < x0 + margin or ys.min() < y0 + margin
                                      or xs.max() >= x1 - margin or ys.max() >= y1 - margin)
                components, _ = cv2.connectedComponents(mask.astype(np.uint8))
                records.append({
                    'instance_id': index, 'category': 'subject', 'class_id': None,
                    'confidence': float(np.clip(proposal['predicted_iou'], 0, 1)),
                    'stability_score': float(proposal['stability_score']),
                    'bbox_xyxy': [x, y, x + width, y + height], 'mask_area': area,
                    'connected_components': components - 1,
                    'touches_content_border': touches_border,
                })
            selected, status = choose_subject(
                records, self.config['confidence'], min_area, max_area,
                self.config['allow_truncated'], instance_id)
        self._synchronize()
        elapsed = time.perf_counter() - start
        box_selected = use_boxes and selected is not None and 'detection_confidence' in selected
        selected_mask = masks[selected['instance_id']] if selected else None
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        for record in records:
            color = '#1d9e63' if selected and record['instance_id'] == selected['instance_id'] else '#d85038'
            draw.rectangle(record['bbox_xyxy'], outline=color, width=3)
        info = {
            'status': status, 'instances': records, 'selected': selected,
            'subject_count': (max(1, route_count) if box_selected else
                              count_distinct_subjects(records, masks, selected, self.config['confidence'], min_area, max_area)),
            'subject_count_rule': ('detector_box_route_v1; at least one for a selected mask; verify manually for training labels' if box_selected else
                                   'conservative SAM2 mask estimate; verify manually for training labels'),
            'selection_rule': ('whole_object_detector_box_prompt' if box_selected else
                               'explicit_instance' if instance_id is not None else
                               'largest_interior_sam2_mask_with_area_and_quality_limits'),
            'requested_category': category, 'category_conditioned': False,
            'mask_source': ('sam2.1_box_prompt' if box_selected else
                            'sam2.1_automatic_fallback' if use_boxes else 'sam2.1_automatic'),
            'subject_selection_mode': self.selection_mode,
            'route_detections': route_detections,
            'touches_content_border': bool(selected and selected['touches_content_border']),
            'truncation_override': self.config['allow_truncated'],
            'seconds': elapsed, 'cold_prediction': self.calls == 0,
            'peak_cuda_allocated_bytes': self.torch.cuda.max_memory_allocated() if self.device.type == 'cuda' else 0,
            'peak_npu_allocated_bytes': self.torch.npu.max_memory_allocated() if self.device.type == 'npu' else 0,
        }
        self.calls += 1
        return selected_mask, overlay, info
