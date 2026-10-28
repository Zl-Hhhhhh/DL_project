"""Validate the observed runtime for UniScheduler's pytorch:py310-2.1.0 image."""
import argparse
import json
import os
import platform
from pathlib import Path

TARGET = {'image':'pytorch:py310-2.1.0', 'gpu':'NVIDIA A800-SXM4-80GB',
          'python':'3.10', 'torch':'2.1.0', 'torchvision':'0.16.0'}


def check_contract(report):
    problems = []
    if report.get('platform') != 'Linux':
        problems.append('The HPC job must run in the Linux image.')
    if not str(report.get('python','')).startswith('3.10.'):
        problems.append('Python 3.10 is required.')
    for package in ('torch','torchvision'):
        if str(report.get(package,'')).split('+')[0] != TARGET[package]:
            problems.append(f'{package} must be {TARGET[package]} with a matching CUDA build.')
    if not str(report.get('numpy','')).startswith('1.'):
        problems.append('Use NumPy 1.x; requirements-hpc.txt pins 1.26.4.')
    if report.get('cuda_available') is not True:
        problems.append('CUDA is unavailable; run inside an allocated GPU job.')
    gpu = report.get('gpu',{})
    name = str(gpu.get('name','')).upper()
    if not all(token in name for token in ('A800','SXM4','80GB')):
        problems.append('Allocated GPU must be NVIDIA A800-SXM4-80GB.')
    if gpu.get('total_memory_bytes',0) < 75_000_000_000:
        problems.append('The job does not expose an 80 GB class GPU allocation.')
    for name in ('numpy_bridge','cuda_matmul','torchvision_cuda_nms','project_imports'):
        if report.get('checks',{}).get(name) != 'passed':
            problems.append(f'Runtime check failed: {name}.')
    problems.extend(report.get('import_errors',[]))
    return problems


def collect_runtime():
    report = {'platform':platform.system(), 'python':platform.python_version(), 'checks':{}, 'import_errors':[]}
    root=Path(__file__).resolve().parents[2]
    for key,folder in [('YOLO_CONFIG_DIR','ultralytics'),('MPLCONFIGDIR','matplotlib')]:
        path=root/'.cache'/folder
        path.mkdir(parents=True,exist_ok=True)
        os.environ.setdefault(key,str(path))
    os.environ.setdefault('YOLO_AUTOINSTALL','false')
    try:
        import numpy as np
        import torch
        import torchvision
    except Exception as error:
        report['import_errors'].append(f'{type(error).__name__}: {error}')
        return report
    report.update(numpy=np.__version__,torch=torch.__version__,torchvision=torchvision.__version__,
                  cuda_runtime=torch.version.cuda,cuda_available=torch.cuda.is_available())
    try:
        assert torch.from_numpy(np.ones(2,dtype=np.float32)).tolist()==[1.,1.]
        report['checks']['numpy_bridge']='passed'
    except Exception as error:
        report['checks']['numpy_bridge']=f'{type(error).__name__}: {error}'
    try:
        import cv2
        import PIL
        import yaml
        import requests
        import ultralytics
        report['packages']={'opencv':cv2.__version__,'pillow':PIL.__version__,
                            'pyyaml':yaml.__version__,'requests':requests.__version__,'ultralytics':ultralytics.__version__}
        report['checks']['project_imports']='passed'
    except Exception as error:
        report['checks']['project_imports']=f'{type(error).__name__}: {error}'
    if torch.cuda.is_available():
        try:
            props=torch.cuda.get_device_properties(0)
        except Exception as error:
            report['import_errors'].append(f'CUDA device properties: {type(error).__name__}: {error}')
            return report
        report['gpu']={'name':props.name,'total_memory_bytes':props.total_memory,
                       'compute_capability':[props.major,props.minor]}
        try:
            x=torch.ones((16,16),device='cuda')
            assert torch.equal(x @ x,torch.full_like(x,16))
            torch.cuda.synchronize()
            report['checks']['cuda_matmul']='passed'
        except Exception as error:
            report['checks']['cuda_matmul']=f'{type(error).__name__}: {error}'
        try:
            boxes=torch.tensor([[0.,0.,10.,10.],[1.,1.,9.,9.]],device='cuda')
            scores=torch.tensor([.9,.8],device='cuda')
            assert torchvision.ops.nms(boxes,scores,.5).tolist()==[0]
            report['checks']['torchvision_cuda_nms']='passed'
        except Exception as error:
            report['checks']['torchvision_cuda_nms']=f'{type(error).__name__}: {error}'
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,help='Optional JSON environment report')
    args=parser.parse_args()
    report=collect_runtime()
    report['target']=TARGET
    report['problems']=check_contract(report)
    report['status']='ready' if not report['problems'] else 'incompatible'
    text=json.dumps(report,indent=2,ensure_ascii=False)+'\n'
    print(text,end='')
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text,encoding='utf-8')
    return 0 if report['status']=='ready' else 2


if __name__=='__main__':
    raise SystemExit(main())
