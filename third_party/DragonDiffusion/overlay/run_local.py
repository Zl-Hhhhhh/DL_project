"""Offline 512px object-move verification with staged CPU offload.

Uses the upstream inversion, feature guidance, memory bank and D+ sampler.
All coordinates refer to the square image saved as input.png in the output folder.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
os.environ['HF_HOME'] = str(ROOT / '.cache/huggingface')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.models.modeling_utils import ModelMixin
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPImageProcessor
from src.models.dragondiff import DragonPipeline
from src.models.Sampler import Sampler
from src.unet.unet_2d_condition import DragonUNet2DConditionModel
from src.unet.estimator import MyUNet2DConditionModel
from src.utils.utils import process_move


class LocalPipeline(DragonPipeline):
    def __init__(self, weights, steps):
        self.precision = torch.float16
        self.NUM_DDIM_STEPS = steps
        common = dict(torch_dtype=self.precision, local_files_only=True,
                      variant='fp16', use_safetensors=True)
        sd = weights / 'sd15'
        # The upstream override predates variant loading; use the installed loader.
        unet = ModelMixin.from_pretrained.__func__(
            DragonUNet2DConditionModel, sd, subfolder='unet', **common)
        vae = AutoencoderKL.from_pretrained(sd, subfolder='vae', **common)
        text = CLIPTextModel.from_pretrained(sd, subfolder='text_encoder', **common)
        tokenizer = CLIPTokenizer.from_pretrained(sd, subfolder='tokenizer', local_files_only=True)
        scheduler = DDIMScheduler.from_pretrained(sd, subfolder='scheduler', local_files_only=True)
        self.pipe = Sampler(vae=vae, text_encoder=text, tokenizer=tokenizer,
                            unet=unet, scheduler=scheduler, safety_checker=None,
                            feature_extractor=None, requires_safety_checker=False)
        self.pipe.estimator = MyUNet2DConditionModel.from_pretrained(sd, subfolder='unet', **common)
        self.pipe.vae.enable_tiling()
        self.pipe.vae.enable_slicing()
        self.num_tokens = 64
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            weights / 'ip-adapter/models/image_encoder', torch_dtype=self.precision,
            local_files_only=True, use_safetensors=True)
        self.clip_image_processor = CLIPImageProcessor()
        self.image_proj_model = self.init_proj(self.precision)
        self.ip_id = str(weights / 'auxiliary/model/ip_sd15_64.bin')
        self.load_adapter(self.ip_id, 0.1)
        self.pipe.unet.to('cpu')  # Adapter modules initially load onto CUDA upstream.
        for model in (unet, vae, text, self.pipe.estimator, self.image_encoder, self.image_proj_model):
            model.eval().requires_grad_(False)
        self.active = None
        # Move only between complete forwards; the estimator remains on CUDA
        # through its latent-gradient calculations. No offload inside autograd.
        for name in ('unet', 'estimator'):
            getattr(self.pipe, name).register_forward_pre_hook(
                lambda module, args, name=name: self.activate(name))
        self.pipe.text_encoder.to('cuda')

    def activate(self, name):
        if name == self.active:
            return
        if self.active:
            getattr(self.pipe, self.active).to('cpu')
        torch.cuda.empty_cache()
        getattr(self.pipe, name).to('cuda')
        self.active = name

    def release_active(self):
        if self.active:
            getattr(self.pipe, self.active).to('cpu')
            self.active = None
        torch.cuda.empty_cache()


def segment(image, box, weights):
    from sam.efficient_sam.efficient_sam import build_efficient_sam
    model = build_efficient_sam(
        encoder_patch_embed_dim=384, encoder_num_heads=6,
        checkpoint=str(weights / 'auxiliary/model/efficient_sam_vits.pt'),
    ).eval().to('cuda')
    value = torch.from_numpy(np.array(image)).permute(2, 0, 1).float().cuda() / 255
    points = torch.tensor(box, device='cuda').reshape(1, 1, 2, 2).float()
    labels = torch.tensor([2, 3], device='cuda').reshape(1, 1, 2)
    with torch.inference_mode():
        logits, scores = model(value[None], points, labels)
        best = scores[0, 0].argmax().item()
        mask = (logits[0, 0, best] >= 0).cpu().numpy().astype(np.uint8) * 255
        score = float(scores[0, 0, best])
    del model, value, logits, scores
    gc.collect()
    torch.cuda.empty_cache()
    return mask, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True, type=Path)
    parser.add_argument('--mask', type=Path)
    parser.add_argument('--box', type=int, nargs=4, default=[95, 215, 225, 390])
    parser.add_argument('--prompt', default='a photo of a cup')
    parser.add_argument('--dx', type=int, default=128)
    parser.add_argument('--dy', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--weights', type=Path, default=ROOT / 'models/dragondiffusion')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output directory must not already exist')
    if not 1 <= args.steps <= 50:
        parser.error('steps must be in [1, 50]')
    if not torch.cuda.is_available():
        parser.error('CUDA is required')
    args.output.mkdir(parents=True)
    report = {'status': 'running', 'mode': 'move', 'algorithm': 'D+', 'size': [512, 512],
              'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'python': sys.executable, 'torch': torch.__version__,
              'gpu': torch.cuda.get_device_name(), 'offload': 'alternate-unet-estimator',
              'attention': 'pytorch-sdpa', 'phases': {}}
    start = time.perf_counter()
    phase_start = start
    phase_name = 'segmentation'

    def end_phase(next_name):
        nonlocal phase_start, phase_name
        torch.cuda.synchronize()
        report['phases'][phase_name] = {
            'seconds': round(time.perf_counter() - phase_start, 3),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        }
        print(f'PHASE {phase_name}: {report["phases"][phase_name]}', flush=True)
        phase_name = next_name
        phase_start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()

    try:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.set_num_threads(4)
        image = ImageOps.fit(Image.open(args.image).convert('RGB'), (512, 512), method=Image.Resampling.LANCZOS)
        image.save(args.output / 'input.png')
        if args.mask:
            mask = np.array(Image.open(args.mask).convert('L').resize((512, 512), Image.Resampling.NEAREST))
            mask = (mask > 127).astype(np.uint8) * 255
        else:
            mask, score = segment(image, args.box, args.weights)
            report['sam_predicted_iou'] = score
        ys, xs = np.nonzero(mask)
        if not len(xs) or len(xs) == 512 * 512:
            raise ValueError('Segmentation must contain foreground and background')
        if min(xs) + args.dx < 0 or max(xs) + args.dx >= 512 or min(ys) + args.dy < 0 or max(ys) + args.dy >= 512:
            raise ValueError('Movement would put the subject outside the image')
        Image.fromarray(mask).save(args.output / 'mask.png')
        report['mask_pixels'] = int(len(xs))
        end_phase('model_loading')
        editor = LocalPipeline(args.weights, args.steps)
        end_phase('encoding')
        editor.image_encoder.to('cuda')
        emb_im, emb_uncond = editor.get_image_embeds(image.resize((256, 256)))
        # These encoders are never used again in this single-image process.
        del editor.image_encoder, editor.image_proj_model
        gc.collect()
        torch.cuda.empty_cache()
        editor.pipe.vae.to('cuda')
        value = torch.from_numpy(np.array(image)).permute(2, 0, 1)[None].to('cuda', dtype=torch.float16) / 127.5 - 1
        latent = editor.image2latent(value)
        editor.pipe.vae.to('cpu')
        del value
        torch.cuda.empty_cache()
        end_phase('inversion')
        history = editor.ddim_inv(latent, args.prompt)
        latent_in = history[-1].squeeze(2)
        mask_rgb = np.repeat(mask[:, :, None], 3, axis=2)
        kwargs = process_move(mask_rgb, h=512, w=512, dx=args.dx, dy=args.dy,
                              scale=4, input_scale=1, resize_scale=1, up_scale=2,
                              up_ft_index=[1, 2], w_edit=4, w_content=6,
                              w_contrast=0.2, w_inpaint=0.8, precision=torch.float16)
        mask_latent = F.interpolate(torch.from_numpy(mask.copy())[None, None].float().cuda(), latent_in.shape[-2:]) > 0
        shift = (int(args.dy / 8), int(args.dx / 8))
        mask_target = torch.roll(mask_latent, shift, (-2, -1)).to(latent_in.dtype)
        latent_in = latent_in * (1 - mask_target) + torch.roll(latent_in, shift, (-2, -1)) * mask_target
        end_phase('editing')
        edited = editor.pipe.edit(prompt=args.prompt, mode='move', emb_im=emb_im,
                                  emb_im_uncond=emb_uncond, latent=latent_in,
                                  num_inference_steps=args.steps, start_time=args.steps,
                                  guidance_scale=4, energy_scale=500,
                                  SDE_strength=0.4, latent_noise_ref=history,
                                  edit_kwargs=kwargs, alg='D+')
        if not torch.isfinite(edited).all():
            raise ValueError('Non-finite edited latent')
        editor.release_active()
        end_phase('decoding')
        editor.pipe.vae.to('cuda')
        result = editor.decode_latents(edited)[:, :, ::-1]
        if result.std() < 1:
            raise ValueError('Nearly constant output image')
        output_file = args.output / 'output.png'
        Image.fromarray(result).save(output_file)
        panel = Image.new('RGB', (1024, 512))
        panel.paste(image, (0, 0))
        panel.paste(Image.fromarray(result), (512, 0))
        panel.save(args.output / 'comparison.png')
        report['output_sha256'] = hashlib.sha256(output_file.read_bytes()).hexdigest()
        report['mean_absolute_pixel_change'] = float(np.abs(result.astype(float) - np.array(image)).mean())
        report['status'] = 'passed'
        end_phase('complete')
    except Exception:
        report['status'] = 'failed'
        report['failed_phase'] = phase_name
        report['error'] = traceback.format_exc()
        end_phase('failed')
        raise
    finally:
        report['total_seconds'] = round(time.perf_counter() - start, 3)
        report['peak_allocated_bytes'] = max((p['peak_allocated_bytes'] for p in report['phases'].values()), default=0)
        report['peak_reserved_bytes'] = max((p['peak_reserved_bytes'] for p in report['phases'].values()), default=0)
        (args.output / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f'RESULT: {report["status"]}, {report["total_seconds"]} seconds', flush=True)


if __name__ == '__main__':
    main()
