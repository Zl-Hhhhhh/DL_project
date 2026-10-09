"""Run this checkout's original IPAdapterXL with the parent project's venv."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ['HF_HOME'] = str(ROOT / '.cache/huggingface')
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'


def main():
    parser = argparse.ArgumentParser(description='Local InstantStyle SDXL inference (CPU offload for 8 GB GPUs).')
    parser.add_argument('--image', type=Path, default=ROOT/'InstantStyle/assets/0.jpg')
    parser.add_argument('--prompt', default='a cat, masterpiece, best quality, high quality')
    parser.add_argument('--output', type=Path, default=ROOT/'experiments/instantstyle/result.png')
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.steps < 1 or args.size < 256 or args.size % 64:
        parser.error('steps must be positive; size must be >=256 and a multiple of 64')
    if args.output.exists():
        parser.error('output already exists; choose a new path')
    import torch
    from diffusers import StableDiffusionXLPipeline
    from PIL import Image
    from ip_adapter import IPAdapterXL

    class OffloadedIPAdapterXL(IPAdapterXL):
        @torch.inference_mode()
        def get_image_embeds(self, *values, **options):
            self.image_encoder.to(self.device)
            try:
                return super().get_image_embeds(*values, **options)
            finally:
                self.image_encoder.to('cpu')
                torch.cuda.empty_cache()

    if not torch.cuda.is_available():
        raise RuntimeError('This local configuration requires CUDA.')
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model_root = ROOT/'models/instantstyle'
    print('Loading SDXL fp16 on CPU...', flush=True)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_root/'sdxl-base', torch_dtype=torch.float16, variant='fp16',
        use_safetensors=True, add_watermarker=False, local_files_only=True,
    )
    pipe.enable_vae_tiling()
    pipe.enable_vae_slicing()
    print('Loading original InstantStyle IPAdapterXL...', flush=True)
    adapter = OffloadedIPAdapterXL(
        pipe, str(model_root/'ip-adapter/sdxl_models/image_encoder'),
        str(model_root/'ip-adapter/sdxl_models/ip-adapter_sdxl.bin'),
        'cpu', target_blocks=['up_blocks.0.attentions.1'],
    )
    adapter.device = 'cuda'
    adapter.image_proj_model.to('cuda')
    pipe.enable_model_cpu_offload()
    with Image.open(args.image) as source:
        reference = source.convert('RGB')
    print(f'Generating {args.size}px / {args.steps} steps...', flush=True)
    images = adapter.generate(
        pil_image=reference, prompt=args.prompt,
        negative_prompt='text, watermark, lowres, low quality, deformed, blurry',
        scale=1.0, guidance_scale=5.0, num_samples=1,
        num_inference_steps=args.steps, seed=args.seed, height=args.size, width=args.size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(args.output)
    record = {'python':sys.executable, 'torch':torch.__version__, 'device':torch.cuda.get_device_name(),
              'input':str(args.image), 'output':str(args.output), 'prompt':args.prompt,
              'seed':args.seed, 'steps':args.steps, 'size':args.size,
              'target_blocks':['up_blocks.0.attentions.1'], 'cpu_offload':True,
              'seconds':time.perf_counter()-started, 'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()}
    args.output.with_suffix('.json').write_text(json.dumps(record,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(record,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
