"""Fixed Ascend device for the local CollageRL inference environment."""
import torch
import torch_npu  # noqa: F401 - registers the Ascend backend
import numpy as np  # exported by the upstream module's wildcard imports


def get_free_device():
    if not torch.npu.is_available() or torch.npu.device_count() <= 3:
        raise RuntimeError('Ascend NPU 3 is required for local CollageRL inference.')
    torch.npu.set_device('npu:3')
    return torch.device('npu:3')


device = get_free_device()
