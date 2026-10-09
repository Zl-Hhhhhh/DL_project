import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .io import configure_local_runtime, download, sha256, write_json


def choose_instance(instances, confidence, category=None, instance_id=None):
    if not instances:
        return None, 'no_subject'
    candidates = [item for item in instances if category is None or item['category'] == category]
    if instance_id is not None:
        candidates = [item for item in candidates if item['instance_id'] == instance_id]
        if not candidates:
            return None, 'invalid_instance'
    if not candidates:
        return None, 'category_not_detected'
    confident = [item for item in candidates if item['confidence'] >= confidence]
    if not confident:
        return None, 'low_confidence'
    selected = max(confident, key=lambda item: (item['mask_area'], item['confidence'], -item['instance_id']))
    if selected['mask_area'] == 0:
        return None, 'empty_mask'
    return selected, 'ok'


class Segmenter:
    def __init__(self, config, root):
        configure_local_runtime(root)
        import torch
        from ultralytics import YOLO, settings

        settings.update({'sync': False})
        self.torch = torch
        self.config = config
        self.device = config['device']
        if self.device != 'cpu' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable. Use --device cpu explicitly, or install the pinned CUDA environment.')
        path = root / config['model_path']
        download(config['model_url'], path)
        digest = sha256(path)
        provenance_path = path.with_suffix('.json')
        if provenance_path.exists():
            from .io import read_json
            if read_json(provenance_path)['sha256'] != digest:
                raise ValueError('Segmentation weight hash changed; use a new model/config instead of silently replacing it.')
        self.provenance = {'model': 'YOLOv8n-seg', 'path': config['model_path'], 'url': config['model_url'], 'sha256': digest, 'pretraining': 'COCO; development overlap is disclosed', 'hash_note': 'Locally computed identity hash, not an upstream-signed checksum.'}
        write_json(provenance_path, self.provenance)
        start = time.perf_counter()
        self.model = YOLO(str(path), task='segment')
        self.load_seconds = time.perf_counter() - start
        self.calls = 0

    def predict(self, image, valid_alpha, mapping, category=None, instance_id=None):
        if self.device != 'cpu':
            self.torch.cuda.synchronize()
            self.torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        result = self.model.predict(image, device=self.device, conf=self.config['min_confidence'], imgsz=self.config['inference_size'], retina_masks=True, verbose=False)[0]
        if self.device != 'cpu':
            self.torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        records, masks = [], []
        if result.masks is not None:
            for index, (mask, box) in enumerate(zip(result.masks.data.cpu().numpy(), result.boxes)):
                mask = cv2.resize(mask, image.size, interpolation=cv2.INTER_NEAREST) > 0.5
                mask &= valid_alpha
                masks.append(mask)
                count, _ = cv2.connectedComponents(mask.astype(np.uint8))
                records.append({'instance_id': index, 'category': self.model.names[int(box.cls.item())], 'class_id': int(box.cls.item()), 'confidence': float(box.conf.item()), 'bbox_xyxy': box.xyxy[0].cpu().tolist(), 'mask_area': int(mask.sum()), 'connected_components': count - 1})
        selected, status = choose_instance(records, self.config['confidence'], category, instance_id)
        selected_mask = None
        touches_border = False
        if selected:
            selected_mask = masks[selected['instance_id']]
            x0, y0, x1, y1 = mapping['content_box_xyxy']
            margin = self.config['border_margin']
            ys, xs = np.where(selected_mask)
            touches_border = bool(xs.min() < x0 + margin or ys.min() < y0 + margin or xs.max() >= x1 - margin or ys.max() >= y1 - margin)
            if touches_border and not self.config['allow_truncated']:
                status = 'subject_truncated'
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        for record in records:
            color = '#1d9e63' if selected and record['instance_id'] == selected['instance_id'] else '#d85038'
            draw.rectangle(record['bbox_xyxy'], outline=color, width=3)
            draw.text(record['bbox_xyxy'][:2], f"#{record['instance_id']} {record['category']} {record['confidence']:.2f}", fill=color, stroke_width=1, stroke_fill='white')
        info = {
            'status': status, 'instances': records, 'selected': selected,
            'selection_rule': 'explicit_instance' if instance_id is not None else 'largest_mask_above_confidence_within_requested_category',
            'requested_category': category, 'mask_source': 'automatic_yolov8n_seg',
            'touches_content_border': touches_border, 'truncation_override': self.config['allow_truncated'],
            'seconds': elapsed, 'cold_prediction': self.calls == 0,
            'peak_cuda_allocated_bytes': self.torch.cuda.max_memory_allocated() if self.device != 'cpu' else 0,
        }
        self.calls += 1
        return selected_mask, overlay, info
