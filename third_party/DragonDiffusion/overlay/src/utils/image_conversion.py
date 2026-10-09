"""Image conversions used by the local CLI (BasicSR-compatible BGR defaults)."""
import numpy as np
import torch


def img2tensor(img, bgr2rgb=True, float32=True):
    if isinstance(img, list):
        return [img2tensor(i, bgr2rgb, float32) for i in img]
    if img.ndim == 2:
        img = img[..., None]
    if bgr2rgb and img.shape[2] == 3:
        img = img[..., ::-1]
    value = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
    return value.float() if float32 else value


def tensor2img(tensor, rgb2bgr=True):
    if isinstance(tensor, list):
        return [tensor2img(t, rgb2bgr) for t in tensor]
    value = tensor.detach().float().cpu().clamp(0, 1)
    if value.ndim == 4:
        if value.shape[0] != 1:
            raise ValueError('Local image conversion requires batch size 1')
        value = value[0]
    value = value.permute(1, 2, 0).numpy()
    if rgb2bgr and value.shape[2] == 3:
        value = value[..., ::-1]
    return np.round(value * 255).astype(np.uint8)
