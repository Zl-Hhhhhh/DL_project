"""Content image + style image, original InstantStyle adapter with alternating GPU residency."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ['HF_HOME'] = str(ROOT/'.cache/huggingface')
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--content', type=Path, required=True)
    parser.add_argument('--style', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prompt', default='a detailed image, high quality')
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--size', type=int, choices=[512,1024], default=1024)
    parser.add_argument('--style-strength', type=float, default=1.0)
    parser.add_argument('--structure-strength', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.steps <= 50:
        parser.error('Use a new output path and 1-50 steps.')
    import cv2
    import numpy as np
    import torch
    from PIL import Image, ImageOps
    from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
    from ip_adapter import IPAdapterXL
    from object_collage.io import letterbox

    class OffloadedAdapter(IPAdapterXL):
        @torch.inference_mode()
        def get_image_embeds(self, *values, **options):
            self.image_encoder.to(self.device)
            try:
                return super().get_image_embeds(*values, **options)
            finally:
                # This worker generates once. Drop the encoder after embedding
                # instead of retaining another ~1.8 GB of weights in system RAM.
                self.image_encoder = None
                import gc
                gc.collect()
                torch.cuda.empty_cache()

    started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this local configuration.')
    torch.cuda.reset_peak_memory_stats()
    folder = ROOT/'models/instantstyle'
    print('PROGRESS:正在加载风格化模型', flush=True)
    controlnet = ControlNetModel.from_pretrained(folder/'controlnet-canny', variant='fp16', torch_dtype=torch.float16, use_safetensors=True, local_files_only=True)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(folder/'sdxl-base', controlnet=controlnet, variant='fp16', torch_dtype=torch.float16, use_safetensors=True, local_files_only=True, add_watermarker=False)
    pipe.enable_vae_tiling()
    pipe.enable_vae_slicing()
    pipe.set_progress_bar_config(disable=True)
    adapter = OffloadedAdapter(pipe, str(folder/'ip-adapter/sdxl_models/image_encoder'), str(folder/'ip-adapter/sdxl_models/ip-adapter_sdxl.bin'), 'cpu', target_blocks=['up_blocks.0.attentions.1'])
    adapter.device = 'cuda'
    adapter.image_proj_model.to('cuda')
    pipe.enable_model_cpu_offload()
    # Standard pipeline offload keeps ControlNet resident during UNet. On 8 GB,
    # explicitly release each model before the other runs, keeping residuals on GPU.
    def before_control(module, inputs):
        pipe.unet.to('cpu')
        torch.cuda.empty_cache()
    def after_control(module, inputs, output):
        module.to('cpu')
        torch.cuda.empty_cache()
    controlnet.register_forward_pre_hook(before_control)
    controlnet.register_forward_hook(after_control)
    content, _, mapping = letterbox(args.content, args.size)
    with Image.open(args.style) as image:
        style = ImageOps.exif_transpose(image).convert('RGB')
    edges = Image.fromarray(cv2.Canny(np.asarray(content), 50, 200)).convert('RGB')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    edges.save(args.output.parent/'control_edges.png')
    def progress(pipeline, step, timestep, values):
        print(f'PROGRESS:正在风格化 {step+1}/{args.steps}', flush=True)
        return values
    image = adapter.generate(pil_image=style, prompt=args.prompt, negative_prompt='text, watermark, low quality, deformed, blurry',
        scale=args.style_strength, guidance_scale=5.0, num_samples=1, num_inference_steps=args.steps, seed=args.seed,
        height=args.size, width=args.size, image=edges, controlnet_conditioning_scale=args.structure_strength,
        callback_on_step_end=progress)[0]
    image.save(args.output)
    record = {'python':sys.executable, 'mode':'stylize', 'content':str(args.content), 'style':str(args.style),
        'size':args.size, 'steps':args.steps, 'seed':args.seed, 'prompt':args.prompt,
        'style_strength':args.style_strength, 'structure_strength':args.structure_strength,
        'seconds':time.perf_counter()-started, 'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
        'cpu_offload':'alternating_controlnet_unet', 'mapping':mapping}
    args.output.with_suffix('.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print('PROGRESS:风格化完成',flush=True)


if __name__ == '__main__':
    main()
