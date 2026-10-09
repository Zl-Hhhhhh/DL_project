"""Run optional GPU stages in isolated processes so model memory is released."""
import gc
import shutil
import subprocess
import sys
from pathlib import Path
from PIL import Image

from .io import letterbox, sha256, write_json


def run_process(command, cwd, log_path, progress):
    with Path(log_path).open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding='utf-8', errors='replace')
        for line in process.stdout:
            log.write(line)
            log.flush()
            if 'PROGRESS:' in line:
                progress(line.partition('PROGRESS:')[2].strip())
        if process.wait() != 0:
            raise RuntimeError(f'GPU stage failed; see {log_path}')


def release_models(editor):
    editor.segmenter = None
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def stylize(root, folder, options, progress):
    (folder/'style.image').write_bytes(options['style_data'])
    command = [sys.executable, '-X', 'utf8', str(root/'InstantStyle/run_controlnet.py'),
        '--content', str(folder/'source.png'), '--style', str(folder/'style.image'),
        '--output', str(folder/'output.png'), '--size', str(options.get('size',1024)), '--prompt='+options['prompt'],
        '--style-strength', str(options['style_strength']), '--structure-strength', str(options['structure_strength'])]
    run_process(command, root, folder/'stylize.log', progress)


def postprocess(root, folder, materials, progress):
    with Image.open(folder/'output.png') as base:
        size = base.width
        if base.size not in {(512,512), (1024,1024)}:
            raise ValueError('CollageRL input must be 512x512 or 1024x1024')
    material_dir = folder/'materials'
    material_dir.mkdir()
    for index, data in enumerate(materials):
        path = material_dir/f'{index}.image'
        path.write_bytes(data)
        image, _, _ = letterbox(path, size)
        image.save(material_dir/f'{index}.png')
    shutil.copyfile(folder/'output.png', folder/'base_output.png')
    output_dir = folder/'collagerl'
    command = [sys.executable, '-X', 'utf8', 'infer.py', '--scale', 'small',
        '--weight_path', 'weights/agents/dtd2imgnet10S_random0110', '--num_multi_env', '1',
        '--wmin', '0.1', '--wmax', '1', '--hmin', '0.1', '--hmax', '1', '--learned_max_steps', '10',
        '--goal', (folder/'base_output.png').as_posix(), '--source_dir', material_dir.as_posix(),
        '--source_load_limit', '8', '--source_sample_size', '4', '--model_based', '--algo', 'sac', '--noop',
        '--window_ratio', '1', '--fixed_t', '9', '--paper_like', '--goal_resolution', str(size),
        '--goal_resolution_fit', 'square', '--source_resolution_ratio', '1', '--min_source_complexity', '0',
        '--scale_order', '256', '128', '64', '--num_cycles', '4', '--min_scrap_size', '0', '--sensitivity', '3',
        '--output_dir', output_dir.as_posix(), '--no_video']
    progress('正在进行 CollageRL 材质重构')
    run_process(command, root/'CollageRL', folder/'collagerl.log', progress)
    shutil.copyfile(output_dir/'output.png', folder/'output.png')
    record = {'enabled':True, 'size':size, 'input':'base_output.png', 'input_sha256':sha256(folder/'base_output.png'),
              'output_sha256':sha256(folder/'output.png'), 'material_count':len(materials), 'command':command}
    write_json(folder/'postprocess.json', record)
    return record
