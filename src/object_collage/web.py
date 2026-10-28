"""Loopback-only image editor, backed by the existing segmentation and renderer."""
import base64
import binascii
import copy
import io
import json
import secrets
import socket
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import yaml
from PIL import Image, UnidentifiedImageError

from .io import configure_local_runtime, letterbox, sha256, write_json
from .render import TARGETS, generate_candidates
from .segmentation import Segmenter
from .presets import PRESETS, POSITIONS, catalog as preset_catalog, parameters as preset_parameters, render_preset

MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_REQUEST_BYTES = 80 * 1024 * 1024
MAX_PIXELS = 25_000_000
ASSETS = Path(__file__).parent / 'web_assets'
STYLE_NAMES = {'navy_coral': '靛蓝珊瑚', 'plum_yellow': '复古黄紫', 'pine_mint': '松绿薄荷', 'original': '保持原色'}
FAILURES = {
    'no_subject': '没有找到清晰的主体，请换一张人物、动物或常见物品的照片。',
    'low_confidence': '主体不够清晰，请尝试轮廓更明确的照片。',
    'empty_mask': '未能提取有效主体，请换一张图片。',
    'subject_truncated': '主体接触了照片边缘。可以换一张完整照片，或使用当前可见主体继续。',
}


class InputError(ValueError):
    pass


def validate_options(payload, detect_only=False):
    mode = payload.get('mode', 'collage')
    if mode not in ('collage', 'stylize'):
        raise InputError('请选择拼接或风格化。')
    size = payload.get('size', 1024)
    if type(size) is not int or size not in (512, 1024):
        raise InputError('输出尺寸请选择 512×512 或 1024×1024。')
    options = {'mode':mode, 'postprocess':False, 'materials':[], 'size':size}
    if detect_only:
        return options
    if mode == 'collage':
        effect = payload.get('collage_effect', 'existing')
        if effect not in ('existing', 'color_focus', 'poster', 'presets'):
            raise InputError('请选择页面中提供的拼接效果。')
        if effect != 'existing':
            options['collage_effect'] = effect
            if effect == 'presets':
                preset = payload.get('preset_id', 'warm_film')
                strength = payload.get('preset_strength', .7)
                position = payload.get('preset_position', 'center')
                if not isinstance(preset, str) or preset not in PRESETS:
                    raise InputError('请选择页面中提供的风格预设。')
                if type(strength) not in (int, float) or not 0 <= strength <= 1:
                    raise InputError('预设强度应在 0–100% 之间。')
                if not isinstance(position, str) or position not in POSITIONS:
                    raise InputError('请选择居中或三分交点。')
                options.update(preset_id=preset, preset_strength=strength, preset_position=position)
            elif effect == 'color_focus':
                value = payload.get('background_saturation', 0.15)
                if type(value) not in (int, float) or not 0 <= value <= 1:
                    raise InputError('背景饱和度应在 0–100% 之间。')
                options['background_saturation'] = value
            else:
                position = payload.get('poster_position', 'center')
                if position not in ('center', 'top_left', 'top_right', 'bottom_left', 'bottom_right'):
                    raise InputError('请选择居中或三分交点。')
                value = payload.get('subject_size', 0.55)
                if type(value) not in (int, float) or not 0.30 <= value <= 0.60:
                    raise InputError('主体最长边应占画布的 30–60%。')
                options.update(poster_position=position, subject_size=value)
    enabled = payload.get('postprocess', False)
    if not isinstance(enabled, bool):
        raise InputError('后处理选项不正确。')
    options['postprocess'] = enabled
    if mode == 'stylize':
        options['style_data'] = validate_upload({'image':payload.get('style_image'), 'style':'original'}, {'original'})[0]
        prompt = payload.get('prompt', 'a detailed image, high quality')
        if not isinstance(prompt, str) or len(prompt) > 500:
            raise InputError('画面描述不能超过 500 个字符。')
        options['prompt'] = prompt.strip() or 'a detailed image, high quality'
        for key, default in [('style_strength',1.0), ('structure_strength',0.8)]:
            value = payload.get(key, default)
            if type(value) not in (int, float) or not 0 <= value <= 1.5:
                raise InputError('强度应在 0–1.5 之间。')
            options[key] = value
    if enabled:
        material_source = payload.get('material_source', 'custom')
        if material_source not in ('builtin', 'custom'):
            raise InputError('请选择使用内置素材或自定义上传。')
        options['material_source'] = material_source
        if material_source == 'builtin':
            return options
        materials = payload.get('materials')
        if not isinstance(materials, list) or not 1 <= len(materials) <= 8:
            raise InputError('启用 CollageRL 后，请上传 1–8 张素材图片。')
        for encoded in materials:
            data = validate_upload({'image':encoded,'style':'original'}, {'original'})[0]
            if len(data) > 4*1024*1024:
                raise InputError('每张 CollageRL 素材不能超过 4 MB。')
            options['materials'].append(data)
    return options


def validate_upload(payload, style_ids):
    if not isinstance(payload, dict):
        raise InputError('请求格式不正确，请重新选择图片。')
    style = payload.get('style')
    composition = payload.get('composition', 'center')
    if not isinstance(style, str) or style not in style_ids:
        raise InputError('请选择页面中提供的样式。')
    if not isinstance(composition, str) or composition not in TARGETS:
        raise InputError('构图选项不正确。')
    if not isinstance(payload.get('allow_truncated', False), bool):
        raise InputError('主体边缘选项不正确。')
    try:
        encoded = payload.get('image', '')
        if not isinstance(encoded, str) or len(encoded) > MAX_REQUEST_BYTES:
            raise InputError('图片过大，请使用 12 MB 以内的图片。')
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise InputError('图片数据不完整，请重新上传。') from error
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise InputError('请使用 12 MB 以内的图片。')
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in {'JPEG', 'PNG', 'WEBP'}:
                raise InputError('目前支持 JPG、PNG 和 WebP 图片。')
            if image.width * image.height > MAX_PIXELS:
                raise InputError('图片尺寸过大，请缩小到 2500 万像素以内。')
            if getattr(image, 'n_frames', 1) > 1:
                raise InputError('请上传静态图片。')
            image.verify()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise InputError('无法读取这张图片，请尝试其他 JPG、PNG 或 WebP 文件。') from error
    return data, style, composition, payload.get('allow_truncated', False)


class Editor:
    def __init__(self, root):
        self.root = Path(root).resolve()
        configure_local_runtime(self.root)
        self.config = yaml.safe_load((self.root/'configs/dev.yaml').read_text(encoding='utf-8'))
        self.config['canvas_size'] = 1024
        self.config['border_margin'] *= 2
        self.palettes = {p['name']: p for p in self.config['palettes']}
        self.palettes['original'] = {'name': 'original', 'background': [245, 237, 223]}
        self.token = secrets.token_urlsafe(32)
        self.jobs = {}
        self.state_lock = threading.Lock()
        self.busy = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='collage-render')
        self.segmenter = None

    def public_config(self):
        from .materials import builtin_catalog
        try:
            _, entries, _ = builtin_catalog(self.root)
        except (ValueError, OSError, KeyError):
            entries = []
        return {'token': self.token, 'max_bytes': MAX_IMAGE_BYTES, 'canvas_size': self.config['canvas_size'], 'output_sizes':[512,1024], 'conversion_modes':['collage','stylize'],
                'collage_effects':['existing','color_focus','poster','presets'], 'presets':preset_catalog(),
                'material_sources':['builtin','custom'], 'builtin_materials':{'count':len(entries), 'used_count':min(8,len(entries)), 'preview':'/assets/builtin-materials.jpg'},
                'styles': [{'id': key, 'name': STYLE_NAMES.get(key, key), 'preview': f'/assets/{key}.png'} for key in self.palettes]}

    def update(self, job_id, **values):
        with self.state_lock:
            self.jobs[job_id].update(values)

    def get(self, job_id):
        with self.state_lock:
            job = self.jobs.get(job_id)
            return copy.deepcopy(job) if job else None

    def submit(self, payload, detect_only=False):
        data, style, composition, allow_truncated = validate_upload(payload, self.palettes)
        options = validate_options(payload, detect_only)
        if options.get('material_source') == 'builtin':
            from .materials import load_builtin
            try:
                options['materials'], options['materials_info'] = load_builtin(self.root)
            except (ValueError, OSError, KeyError) as error:
                raise InputError(str(error)) from error
        if not self.busy.acquire(blocking=False):
            raise BlockingIOError('正在处理上一张图片，请稍后再试。')
        job_id = uuid.uuid4().hex
        try:
            with self.state_lock:
                # Results on disk remain available; bound the live in-memory history.
                while len(self.jobs) >= 64:
                    del self.jobs[next(iter(self.jobs))]
                self.jobs[job_id] = {'id': job_id, 'state': 'working', 'stage': '正在准备图片', 'style': style, 'composition': composition}
            self.executor.submit(self.generate, job_id, data, style, composition, allow_truncated, detect_only, options)
        except Exception:
            self.busy.release()
            raise
        return job_id

    def generate(self, job_id, data, style, composition, allow_truncated, detect_only=False, options=None):
        options = options or {'mode':'collage','postprocess':False}
        folder = self.root/'experiments/ui'/job_id
        try:
            folder.mkdir(parents=True, exist_ok=False)
            upload = folder/'upload.image'
            upload.write_bytes(data)
            config = copy.deepcopy(self.config)
            config['canvas_size'] = options.get('size', 1024)
            config['border_margin'] = round(self.config['border_margin'] * config['canvas_size'] / self.config['canvas_size'])
            config['allow_truncated'] = allow_truncated
            config['palettes'] = [self.palettes[style]]
            config['render_style'] = 'original' if style == 'original' else 'duotone'
            config['background_mode'] = 'source_masked'
            if options.get('collage_effect'):
                config['interactive_effect'] = {key:value for key,value in options.items() if key in ('collage_effect','background_saturation','poster_position','subject_size','preset_id','preset_strength','preset_position')}
            if detect_only:
                config['min_confidence'] = 0.01
            write_json(folder/'config.json', config)
            image, alpha, mapping = letterbox(upload, config['canvas_size'])
            image.save(folder/'source.png')
            if (not detect_only and options.get('collage_effect') == 'presets'
                    and not PRESETS[options['preset_id']]['requires_subject']):
                self.generate_preset(job_id, folder, image, None, mapping, options)
                return
            if options['mode'] == 'stylize' and not detect_only:
                from .external_pipeline import release_models, stylize
                release_models(self)
                stylize(self.root, folder, options, lambda stage:self.update(job_id,stage=stage))
                self.finish(job_id, folder, options, {'status':'ok','mode':'stylize','source':'source.png'})
                return
            self.update(job_id, stage='首次加载模型，稍等片刻' if self.segmenter is None else '正在提取图片主体')
            if self.segmenter is None:
                self.segmenter = Segmenter(config, self.root)
            self.segmenter.config = config
            self.update(job_id, stage='正在提取图片主体')
            mask, overlay, segmentation = self.segmenter.predict(image, alpha, mapping)
            if detect_only:
                detections = [{'confidence': item['confidence'], 'bbox_xyxy': item['bbox_xyxy']} for item in segmentation['instances']]
                write_json(folder/'result.json', {'status': 'ok', 'mode': 'detection_preview', 'size':config['canvas_size'], 'detections': detections})
                self.update(job_id, state='succeeded', stage='识别完成', source=f'/api/jobs/{job_id}/source.png', size=config['canvas_size'], detections=detections)
                return
            overlay.save(folder/'detections.png')
            if mask is not None:
                Image.fromarray((mask*255).astype(np.uint8)).save(folder/'mask.png')
            if segmentation['status'] != 'ok':
                code = segmentation['status']
                write_json(folder/'result.json', {'status': code, 'segmentation': segmentation, 'canvas_mapping': mapping, 'model': self.segmenter.provenance})
                self.update(job_id, state='failed', code=code, message=FAILURES.get(code, '暂时无法提取这张图片的主体，请换一张图片。'))
                return
            self.update(job_id, stage='正在生成拼贴')
            if options.get('collage_effect') == 'presets':
                self.generate_preset(job_id, folder, image, mask, mapping, options, segmentation)
                return
            if options.get('collage_effect') in ('color_focus', 'poster'):
                from .composition_study import render_candidate
                effect = options['collage_effect']
                params = {'version':'priority-study-v1', 'size':config['canvas_size'],
                          'content_box':mapping['content_box_xyxy']}
                if effect == 'color_focus':
                    params.update(mode='color', background_saturation=options['background_saturation'])
                else:
                    positions = {'center':[.5,.5], 'top_left':[1/3,1/3], 'top_right':[2/3,1/3],
                                 'bottom_left':[1/3,2/3], 'bottom_right':[2/3,2/3]}
                    params.update(mode='poster', target_xy=positions[options['poster_position']],
                                  longest_edge_fraction=options['subject_size'], background_rgb=[245,237,223])
                output, visible, metrics = render_candidate(image, mask, params)
                output.save(folder/'output.png')
                visible.save(folder/'visible_mask.png')
                record = {'source':'source.png', 'mask':'mask.png', 'source_sha256':sha256(folder/'source.png'),
                          'mask_sha256':sha256(folder/'mask.png'), 'output_sha256':sha256(folder/'output.png'),
                          'params':params, 'metrics':metrics, 'model':self.segmenter.provenance,
                          'segmentation':segmentation, 'canvas_mapping':mapping, 'mode':'interactive',
                          'style':'original', 'collage_effect':effect, 'selection_method':'user_parameters', 'candidate_count':1}
                write_json(folder/'params.json', record)
                self.finish(job_id, folder, options, {'status':'ok', **record}, segmentation['touches_content_border'])
                return
            candidates, rejected, selected = generate_candidates(image, mask, config, sha256(upload), composition)
            if selected is None:
                self.update(job_id, state='failed', code='no_valid_candidates', message='当前构图无法保留主体，请换成居中构图重试。')
                return
            selected['image'].save(folder/'output.png')
            selected['visible_mask'].save(folder/'visible_mask.png')
            record = {'source': 'source.png', 'mask': 'mask.png', 'source_sha256': sha256(folder/'source.png'), 'mask_sha256': sha256(folder/'mask.png'), 'output_sha256': sha256(folder/'output.png'), 'params': selected['params'], 'metrics': selected['metrics'], 'model': self.segmenter.provenance, 'segmentation': segmentation, 'canvas_mapping': mapping, 'mode': 'interactive', 'style': style, 'selection_method': 'fixed_rule', 'candidate_count': len(candidates)}
            write_json(folder/'params.json', record)
            self.finish(job_id, folder, options, {'status':'ok', **record}, segmentation['touches_content_border'])
        except Exception:
            detail = traceback.format_exc()
            print(detail, flush=True)
            folder.mkdir(parents=True, exist_ok=True)
            (folder/'error.log').write_text(detail, encoding='utf-8')
            self.update(job_id, state='failed', code='generation_error', message='生成遇到问题，请重新尝试；如果持续失败，请检查启动窗口中的错误信息。')
        finally:
            self.busy.release()

    def generate_preset(self, job_id, folder, image, mask, mapping, options, segmentation=None):
        self.update(job_id, stage='正在应用风格预设')
        params = preset_parameters(options['preset_id'], options['preset_strength'], options['size'],
                                   mapping['content_box_xyxy'], options['preset_position'])
        output, visible, metrics = render_preset(image, mask, params)
        output.save(folder/'output.png')
        visible.save(folder/'visible_mask.png')
        stored_mask = Image.fromarray((np.asarray(mask,dtype=bool)*255).astype(np.uint8)) if mask is not None else Image.new('L',image.size)
        stored_mask.save(folder/'mask.png')
        record = {'source':'source.png', 'mask':'mask.png', 'source_sha256':sha256(folder/'source.png'),
                  'mask_sha256':sha256(folder/'mask.png'), 'output_sha256':sha256(folder/'output.png'),
                  'params':params, 'metrics':metrics, 'canvas_mapping':mapping, 'mode':'interactive',
                  'collage_effect':'presets', 'preset_id':options['preset_id'], 'selection_method':'user_preset',
                  'segmentation':segmentation or {'status':'not_required'}, 'candidate_count':1}
        write_json(folder/'params.json', record)
        self.finish(job_id, folder, options, {'status':'ok', **record}, bool(segmentation and segmentation['touches_content_border']))

    def finish(self, job_id, folder, options, record, truncated=False):
        if options['postprocess']:
            from .external_pipeline import release_models, postprocess
            release_models(self)
            record['postprocess'] = postprocess(self.root, folder, options['materials'], lambda stage:self.update(job_id,stage=stage))
            record['postprocess']['materials_info'] = options.get('materials_info', {'source':'custom'})
            write_json(folder/'postprocess.json', record['postprocess'])
            record['base_output'] = 'base_output.png'
        record['output_sha256'] = sha256(folder/'output.png')
        record['conversion_mode'] = options['mode']
        record['size'] = options.get('size', 1024)
        write_json(folder/'result.json', record)
        self.update(job_id, state='succeeded', stage='生成完成', mode=options['mode'], postprocess=options['postprocess'], size=record['size'],
                    output=f'/api/jobs/{job_id}/output.png', source=f'/api/jobs/{job_id}/source.png',
                    download=f'/api/jobs/{job_id}/download', truncated=truncated)


def handler_for(editor):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, data, content_type='application/json; charset=utf-8', download=False):
            if isinstance(data, dict):
                data = json.dumps(data, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' blob: data:; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
            if download:
                self.send_header('Content-Disposition', 'attachment; filename="collage.png"')
            self.end_headers()
            self.wfile.write(data)

        def valid_host(self):
            return self.headers.get('Host') in {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}

        def do_GET(self):
            if not self.valid_host():
                self.reply(403, {'message': '仅允许从本机页面访问。'})
                return
            path = urlparse(self.path).path
            if path == '/api/config':
                self.reply(200, editor.public_config())
                return
            if path == '/':
                self.reply(200, (ASSETS/'index.html').read_bytes(), 'text/html; charset=utf-8')
                return
            if path.startswith('/assets/'):
                filename = path.removeprefix('/assets/')
                if filename == 'builtin-materials.jpg':
                    preview = editor.root/'CollageRL/samples/materials/newspaper/preview.jpg'
                    if preview.is_file():
                        self.reply(200, preview.read_bytes(), 'image/jpeg')
                        return
                allowed = {'app.js': 'text/javascript; charset=utf-8', 'style.css': 'text/css; charset=utf-8', 'example.jpg': 'image/jpeg', **{f'{key}.png': 'image/png' for key in editor.palettes}, **{f'preset_{key}.png':'image/png' for key in PRESETS}}
                if filename in allowed and (ASSETS/filename).is_file():
                    self.reply(200, (ASSETS/filename).read_bytes(), allowed[filename])
                    return
            parts = path.strip('/').split('/')
            if len(parts) in {3, 4} and parts[:2] == ['api', 'jobs']:
                job = editor.get(parts[2])
                if job is not None:
                    if len(parts) == 3:
                        self.reply(200, job)
                        return
                    if job['state'] == 'succeeded' and (parts[3] == 'source.png' or (parts[3] in {'output.png', 'download'} and 'output' in job)):
                        filename = 'output.png' if parts[3] == 'download' else parts[3]
                        self.reply(200, (editor.root/'experiments/ui'/job['id']/filename).read_bytes(), 'image/png', parts[3] == 'download')
                        return
            self.reply(404, {'message': '内容不存在或已过期，请返回编辑页面重新生成。'})

        def do_POST(self):
            origin = self.headers.get('Origin')
            expected = {f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}'}
            if not self.valid_host() or (origin is not None and origin not in expected) or not secrets.compare_digest(self.headers.get('X-Collage-Token', ''), editor.token):
                self.reply(403, {'message': '页面验证已失效，请刷新页面再试。'})
                return
            route = urlparse(self.path).path
            if route not in {'/api/generate', '/api/detect'}:
                self.reply(404, {'message': '请求地址不存在。'})
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= MAX_REQUEST_BYTES:
                    self.reply(413, {'message': '图片过大，请使用 12 MB 以内的图片。'})
                    return
                if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                    self.reply(415, {'message': '请求格式不正确。'})
                    return
                self.connection.settimeout(30)
                payload = json.loads(self.rfile.read(length))
                job_id = editor.submit(payload, detect_only=True) if route == '/api/detect' else editor.submit(payload)
                self.reply(202, {'id': job_id})
            except BlockingIOError as error:
                self.reply(409, {'message': str(error)})
            except (InputError, ValueError, UnicodeError) as error:
                self.reply(400, {'message': str(error) if isinstance(error, InputError) else '无法读取上传内容，请重新选择图片。'})

        def log_message(self, format, *args):
            if self.command != 'GET' or self.path == '/':
                super().log_message(format, *args)

    return Handler


class LocalServer(ThreadingHTTPServer):
    allow_reuse_address = False

    def server_bind(self):
        if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def serve(root, port=7860):
    editor = Editor(root)
    server = LocalServer(('127.0.0.1', port), handler_for(editor))
    print(f'拼贴工作室已启动：http://127.0.0.1:{port}', flush=True)
    print('保持此窗口运行；按 Ctrl+C 停止。', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        editor.executor.shutdown(wait=True)
