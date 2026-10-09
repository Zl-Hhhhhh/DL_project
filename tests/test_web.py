import base64
import io
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest
from PIL import Image

from object_collage.web import InputError, handler_for, validate_upload, validate_options


def payload(style='pine_mint', composition='left'):
    output = io.BytesIO()
    Image.new('RGB', (32, 32), 'red').save(output, format='PNG')
    return {'image': base64.b64encode(output.getvalue()).decode(), 'style': style, 'composition': composition}


def test_upload_validation_checks_actual_content_and_options():
    raw, style, composition, allow = validate_upload(payload(), {'pine_mint'})
    assert raw.startswith(b'\x89PNG') and style == 'pine_mint' and composition == 'left' and allow is False
    for invalid in [None, {'style': []}, payload('missing'), payload(composition='outside'), {**payload(), 'image': 'broken base64'}, {**payload(), 'image': base64.b64encode(b'not a photo').decode()}, {**payload(), 'allow_truncated': 'yes'}]:
        with pytest.raises(InputError):
            validate_upload(invalid, {'pine_mint'})


def test_conversion_modes_require_their_own_inputs_and_explicit_postprocess():
    assert validate_options(payload()) == {'mode':'collage','postprocess':False,'materials':[], 'size':1024}
    valid = {**payload(), 'mode':'stylize', 'style_image':payload()['image'], 'postprocess':True, 'materials':[payload()['image']]}
    result = validate_options(valid)
    assert result['mode']=='stylize' and result['style_data'] and len(result['materials'])==1
    for invalid in [{**payload(),'mode':'combined'}, {**payload(),'mode':'stylize'},
                    {**payload(),'postprocess':True}, {**valid,'materials':[]},
                    {**valid,'materials':[payload()['image']]*9}, {**valid,'postprocess':'true'},
                    {**valid,'structure_strength':float('nan')}, {**valid,'style_strength':2},
                    {**valid,'prompt':'a'*501}]:
        with pytest.raises(InputError):
            validate_options(invalid)


def test_output_size_validation_also_applies_to_detection():
    for detection in [False, True]:
        for size in [512,1024]:
            assert validate_options({**payload(),'size':size},detect_only=detection)['size']==size
        for size in [256,768,2048,'512',512.0,True,None]:
            with pytest.raises(InputError):
                validate_options({**payload(),'size':size},detect_only=detection)


def test_priority_effect_options_and_ranges():
    color = validate_options({**payload(), 'collage_effect':'color_focus', 'background_saturation':0})
    assert color['background_saturation'] == 0
    for position in ['center','top_left','top_right','bottom_left','bottom_right']:
        result = validate_options({**payload(), 'collage_effect':'poster', 'poster_position':position, 'subject_size':.38})
        assert result['poster_position'] == position and result['subject_size'] == .38
    invalid = [{'collage_effect':'bad'}, {'collage_effect':[]},
               {'collage_effect':'poster','poster_position':'left'},
               {'collage_effect':'poster','subject_size':.8},
               {'collage_effect':'poster','subject_size':True},
               {'collage_effect':'color_focus','background_saturation':float('nan')},
               {'collage_effect':'color_focus','background_saturation':1.1},
               {'collage_effect':'color_focus','background_saturation':'0.15'}]
    for options in invalid:
        with pytest.raises(InputError):
            validate_options({**payload(), **options})


@pytest.mark.parametrize('size', [512,1024])
def test_external_stages_use_requested_canvas_size(tmp_path, monkeypatch, size):
    from object_collage import external_pipeline
    calls = []
    def run_process(command, cwd, log_path, progress):
        calls.append(command)
        if '--output_dir' in command:
            from pathlib import Path
            output = Path(command[command.index('--output_dir')+1])
            output.mkdir()
            assert Image.open(tmp_path/'materials/0.png').size==(size,size)
            Image.new('RGB',(size,size),'blue').save(output/'output.png')
    monkeypatch.setattr(external_pipeline,'run_process',run_process)
    options = {'size':size,'style_data':b'style','prompt':'test','style_strength':1,'structure_strength':0.8}
    external_pipeline.stylize(tmp_path,tmp_path,options,lambda stage:None)
    assert calls[0][calls[0].index('--size')+1]==str(size)
    Image.new('RGB',(size,size),'red').save(tmp_path/'output.png')
    record = external_pipeline.postprocess(tmp_path,tmp_path,[base64.b64decode(payload()['image'])],lambda stage:None)
    assert calls[1][calls[1].index('--goal_resolution')+1]==str(size)
    assert record['size']==size and Image.open(tmp_path/'output.png').size==(size,size)


def test_builtin_material_source_needs_no_upload_and_rejects_unknown_source():
    result = validate_options({**payload(), 'postprocess':True, 'material_source':'builtin'})
    assert result['material_source'] == 'builtin' and result['materials'] == []
    with pytest.raises(InputError):
        validate_options({**payload(), 'postprocess':True, 'material_source':'../outside'})
    with pytest.raises(InputError):
        validate_options({**payload(), 'postprocess':True, 'material_source':'custom'})


def test_builtin_library_checks_hashes_and_rejects_paths_outside_library(tmp_path):
    from object_collage.materials import load_builtin
    from object_collage.io import sha256
    folder = tmp_path/'CollageRL/samples/materials/newspaper'
    folder.mkdir(parents=True)
    photo = folder/'page.jpg'
    Image.new('RGB',(16,16),'red').save(photo)
    source = {'dataset_url':'https://archive.ics.uci.edu/dataset/306', 'license':'CC BY 4.0',
              'images':[{'file':'page.jpg','sha256':sha256(photo)}]}
    manifest = folder/'SOURCE.json'
    manifest.write_text(json.dumps(source),encoding='utf-8')
    data, info = load_builtin(tmp_path)
    assert data == [photo.read_bytes()] and info['selected_files'] == ['page.jpg']
    photo.write_bytes(b'changed')
    with pytest.raises(ValueError, match='校验失败'):
        load_builtin(tmp_path)
    source['images'][0]['file'] = '../outside.jpg'
    manifest.write_text(json.dumps(source),encoding='utf-8')
    with pytest.raises(ValueError, match='filename'):
        load_builtin(tmp_path)


@pytest.mark.parametrize('mode', ['collage','stylize'])
def test_postprocess_runs_after_conversion_and_publishes_final_hash(tmp_path, monkeypatch, mode):
    from object_collage.web import Editor
    from object_collage.io import sha256
    from object_collage import external_pipeline
    editor = Editor(__import__('pathlib').Path.cwd())
    editor.jobs['test'] = {'id':'test','state':'working'}
    Image.new('RGB',(16,16),'red').save(tmp_path/'output.png')
    original = sha256(tmp_path/'output.png')
    calls = []
    def postprocess(root, folder, materials, progress):
        assert sha256(folder/'output.png') == original
        assert materials == [b'material']
        (folder/'base_output.png').write_bytes((folder/'output.png').read_bytes())
        Image.new('RGB',(16,16),'blue').save(folder/'output.png')
        calls.append('postprocess')
        return {'enabled':True}
    monkeypatch.setattr(external_pipeline,'release_models',lambda editor:None)
    monkeypatch.setattr(external_pipeline,'postprocess',postprocess)
    try:
        editor.finish('test',tmp_path,{'mode':mode,'postprocess':True,'materials':[b'material']},{'status':'ok'})
        result=json.loads((tmp_path/'result.json').read_text())
        assert calls==['postprocess'] and result['output_sha256']!=original
        assert sha256(tmp_path/'base_output.png')==original
        assert editor.get('test')['postprocess'] is True
        assert editor.get('test')['state']=='succeeded'
    finally:
        editor.executor.shutdown()


def test_local_api_requires_session_token_and_blocks_foreign_origins(tmp_path):
    class FakeEditor:
        token = 'session-token'
        palettes = {'pine_mint': {}}
        root = tmp_path
        submissions = []

        def public_config(self):
            return {'token': self.token}

        def submit(self, data, detect_only=False):
            validate_upload(data, self.palettes)
            self.submissions.append((data, detect_only))
            return 'test-job'

        def get(self, job):
            return None

    editor = FakeEditor()
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(editor))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    data = json.dumps(payload()).encode()
    try:
        for headers in [{'Content-Type': 'application/json'}, {'Content-Type': 'application/json', 'X-Collage-Token': editor.token, 'Origin': 'http://outside.example'}]:
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(urllib.request.Request(base+'/api/generate', data=data, headers=headers))
            assert error.value.code == 403
        response = urllib.request.urlopen(urllib.request.Request(base+'/api/generate', data=data, headers={'Content-Type': 'application/json', 'X-Collage-Token': editor.token, 'Origin': base}))
        assert response.status == 202 and json.load(response)['id'] == 'test-job'
        assert len(editor.submissions) == 1
        response = urllib.request.urlopen(urllib.request.Request(base+'/api/detect', data=data, headers={'Content-Type': 'application/json', 'X-Collage-Token': editor.token, 'Origin': base}))
        assert response.status == 202 and editor.submissions[-1][1] is True
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(base+'/assets/../../configs/dev.yaml')
        assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
