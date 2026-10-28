import base64
import io

import numpy as np
import pytest
from PIL import Image

from object_collage.io import sha256, write_json
from object_collage.pipeline import replay
from object_collage.presets import PRESETS, parameters, render_preset
from object_collage.web import Editor, InputError, validate_options


@pytest.mark.parametrize('size',[512,1024])
def test_presets_keep_padding_and_subject_and_replay(size,tmp_path):
    yy,xx=np.mgrid[:size,:size]
    rgb=np.stack([xx%256,yy%256,(xx+yy)%256],-1).astype(np.uint8)
    mask=np.zeros((size,size),bool);mask[size//3:2*size//3,size//3:2*size//3]=True
    image=Image.fromarray(rgb)
    image.save(tmp_path/'source.png');Image.fromarray(mask.astype(np.uint8)*255).save(tmp_path/'mask.png')
    for preset in PRESETS:
        params=parameters(preset,.7,size,[0,40,size,size-40],'top_right')
        output,visible,metrics=render_preset(image,mask,params)
        assert output.size==(size,size)
        if preset!='minimal_poster':
            np.testing.assert_array_equal(np.asarray(output)[:40],rgb[:40])
            unchanged,_,_=render_preset(image,mask,parameters(preset,0,size,[0,40,size,size-40]))
            np.testing.assert_array_equal(unchanged,image)
        if preset=='subject_pop':
            np.testing.assert_array_equal(np.asarray(output)[mask],rgb[mask])
        output.save(tmp_path/(preset+'.png'))
        record={'source':'source.png','mask':'mask.png','source_sha256':sha256(tmp_path/'source.png'),
                'mask_sha256':sha256(tmp_path/'mask.png'),'params':params,
                'output_sha256':sha256(tmp_path/(preset+'.png'))}
        write_json(tmp_path/(preset+'.json'),record)
        assert replay(tmp_path/(preset+'.json'),tmp_path/(preset+'_replay.png'))['exact_match']
    _,_,compact=render_preset(image,mask,parameters('minimal_poster',0,size,[0,0,size,size]))
    _,_,airy=render_preset(image,mask,parameters('minimal_poster',1,size,[0,0,size,size]))
    assert airy['subject_area_fraction']<compact['subject_area_fraction']


@pytest.mark.parametrize('invalid',[{'preset_id':'bad'},{'preset_id':[]},{'preset_strength':float('nan')},
                                  {'preset_strength':True},{'preset_strength':1.01},{'preset_position':[]},
                                  {'preset_position':'outside'}])
def test_preset_api_rejects_invalid_parameters(invalid):
    with pytest.raises(InputError):
        validate_options({'mode':'collage','collage_effect':'presets',**invalid})


def test_global_filter_works_without_loading_segmentation(monkeypatch,tmp_path):
    from object_collage import web
    from pathlib import Path
    (tmp_path/'configs').mkdir()
    (tmp_path/'configs/dev.yaml').write_bytes(Path('configs/dev.yaml').read_bytes())
    editor=Editor(tmp_path)
    output=io.BytesIO();Image.new('RGB',(32,24),'steelblue').save(output,format='PNG')
    encoded=base64.b64encode(output.getvalue()).decode()
    seen=[]
    def no_detector(*args,**kwargs):
        pytest.fail('Global color presets must not require object detection')
    monkeypatch.setattr(web,'Segmenter',no_detector)
    # Avoid creating a permanent UI job; assert the early routing receives no mask.
    def generated(job,folder,image,mask,mapping,options,segmentation=None):
        assert mask is None
        seen.append(options['preset_id'])
    monkeypatch.setattr(editor,'generate_preset',generated)
    job=editor.submit({'image':encoded,'style':'original','collage_effect':'presets','preset_id':'warm_film'})
    editor.executor.shutdown(wait=True)
    assert seen==['warm_film']
    assert editor.segmenter is None
