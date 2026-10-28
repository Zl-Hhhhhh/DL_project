import copy
import importlib.util
from pathlib import Path

import pytest


spec=importlib.util.spec_from_file_location('hpc_environment',Path('scripts/hpc/check_environment.py'))
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def compatible_report():
    return {'platform':'Linux','python':'3.10.14','torch':'2.1.0+cu121',
            'torchvision':'0.16.0+cu121','numpy':'1.26.4','cuda_available':True,
            'gpu':{'name':'NVIDIA A800-SXM4-80GB','total_memory_bytes':80*1024**3},
            'checks':{key:'passed' for key in ('numpy_bridge','cuda_matmul','torchvision_cuda_nms','project_imports')}}


def test_matching_cuda_local_versions_are_accepted(compatible_report):
    assert not module.check_contract(compatible_report)


@pytest.mark.parametrize('field,value,expected',[
    ('python','3.11.0','Python 3.10'),('torch','2.5.1+cu121','torch must be 2.1.0'),
    ('torchvision','0.20.1+cu121','torchvision must be 0.16.0'),('numpy','2.1.0','NumPy 1.x'),
    ('cuda_available',False,'CUDA is unavailable')])
def test_incompatible_runtime_is_rejected(compatible_report,field,value,expected):
    compatible_report[field]=value
    assert any(expected in problem for problem in module.check_contract(compatible_report))


def test_wrong_gpu_small_partition_and_missing_operator_are_rejected(compatible_report):
    report=copy.deepcopy(compatible_report)
    report['gpu']['name']='NVIDIA A40'
    assert any('A800' in p for p in module.check_contract(report))
    report=copy.deepcopy(compatible_report)
    report['gpu']['total_memory_bytes']=10*1024**3
    assert any('80 GB' in p for p in module.check_contract(report))
    report=copy.deepcopy(compatible_report)
    del report['checks']['torchvision_cuda_nms']
    assert any('torchvision_cuda_nms' in p for p in module.check_contract(report))
    assert module.check_contract({})
