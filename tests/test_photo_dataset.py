import importlib.util
from pathlib import Path


def load_script(name):
    spec=importlib.util.spec_from_file_location(name,Path('scripts')/(name+'.py'))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_selection_excludes_development_and_non_derivative_licenses():
    module=load_script('prepare_photo_dataset')
    photos=[{'id':i,'license':license,'width':500,'height':500,'file_name':str(i)+'.jpg'}
            for i,license in [(1,4),(2,5),(3,6),(4,4)]]
    annotations=[{'id':i,'image_id':i,'category_id':1,'iscrowd':0,'bbox':[50,50,200,200],'area':40000}
                 for i in range(1,5)]
    coco={'images':photos,'categories':[{'id':1,'name':'person'}],
          'licenses':[{'id':i,'name':str(i)} for i in (4,5,6)],'annotations':annotations}
    first=module.candidate_pools(coco,{4},2012)
    assert {r['image_id'] for r in first['person']}=={1,2}
    assert first==module.candidate_pools(coco,{4},2012)


def test_comparison_sources_do_not_consume_heldout_images():
    module=load_script('run_preset_study')
    manifest={'images':[{'source_id':str(i),'category':category,'split':split}
                        for i,category,split in [(0,'cat','test'),(1,'cat','val'),(2,'cat','train'),
                                                 (3,'dog','train'),(4,'dog','test')]]}
    chosen=module.fixed_sources(manifest)
    assert {r['source_id'] for r in chosen}=={'2','3'}
    assert all(r['split']=='train' for r in chosen)
