"""Generate fixed, source-attributed training-only comparisons for photo presets."""
import argparse
import csv
import html
import json
import platform
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from object_collage.io import letterbox, read_json, sha256, write_json, validate_manifest
from object_collage.pipeline import replay
from object_collage.presets import PRESETS, parameters, render_preset
from object_collage.segmentation import Segmenter

ROOT = Path(__file__).resolve().parents[1]
LABELS = {'original':'Original', 'warm_film':'Warm film', 'muted_vintage':'Muted vintage',
          'subject_pop':'Subject pop', 'minimal_poster':'Poster: center', 'poster_thirds':'Poster: thirds'}


def font(size):
    for name in ('DejaVuSans.ttf', 'Arial.ttf', 'C:/Windows/Fonts/arial.ttf'):
        try:
            return ImageFont.truetype(name,size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def fixed_sources(manifest):
    training = sorted([r for r in manifest['images'] if r['split']=='train'],key=lambda r:r['source_id'])
    categories = sorted({r['category'] for r in training})
    chosen = [next(r for r in training if r['category']==category) for category in categories]
    chosen += [r for r in training if r not in chosen][:2]
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=ROOT/'data/photo120/manifest.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--size',type=int,choices=[512,1024],default=512)
    parser.add_argument('--device',default='0',help='CUDA index, or explicitly cpu on HPC login nodes')
    parser.add_argument('--write-previews',action='store_true')
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    validate_manifest(manifest)
    sources = fixed_sources(manifest)
    args.output.mkdir(parents=True,exist_ok=False)
    # Freeze source list before loading the detector or examining output quality.
    write_json(args.output/'sources.json', {'selection':'first training source per category, then first two unused training sources, sorted by source_id',
                                          'heldout_used':False, 'images':sources})
    config = yaml.safe_load((ROOT/'configs/dev.yaml').read_text(encoding='utf-8'))
    config.update(canvas_size=args.size,device=args.device,border_margin=round(2*args.size/512))
    segmenter = Segmenter(config,ROOT)
    results, metrics_rows, preview_source = [], [], None
    for row in sources:
        folder = args.output/row['source_id']
        folder.mkdir()
        path = args.manifest.parent/row['relative_path']
        if sha256(path)!=row['sha256']:
            raise ValueError('Source hash changed')
        image,alpha,mapping = letterbox(path,args.size)
        image.save(folder/'source.png')
        mask,overlay,segmentation = segmenter.predict(image,alpha,mapping,category=row['category'])
        overlay.save(folder/'detections.jpg')
        usable = segmentation['status']=='ok'
        stored_mask = mask if mask is not None else np.zeros((args.size,args.size),bool)
        Image.fromarray(stored_mask.astype(np.uint8)*255).save(folder/'mask.png')
        sheet = Image.new('RGB',(3*300,2*344+64),'#f5f2ec')
        draw = ImageDraw.Draw(sheet)
        records, previews = [], {}
        for index,key in enumerate(LABELS):
            x,y=(index%3)*300,(index//3)*344
            draw.text((x+12,y+10),LABELS[key],font=font(16),fill='#27323a')
            preset = 'minimal_poster' if key=='poster_thirds' else key
            strength = .7
            position = 'top_left' if key=='poster_thirds' else 'center'
            if key=='original':
                output=image
                draw.text((x+12,y+319),'Input control',font=font(12),fill='#657078')
            elif PRESETS[preset]['requires_subject'] and not usable:
                reason=segmentation['status']
                draw.multiline_text((x+16,y+110),'Not generated\n'+reason,font=font(15),fill='#865a4a',spacing=8)
                records.append({'candidate_id':key,'status':'unavailable','reason':reason})
                continue
            else:
                params = parameters(preset,strength,args.size,mapping['content_box_xyxy'],position)
                output,visible,metrics = render_preset(image,stored_mask,params)
                output.save(folder/(key+'.png'))
                visible.save(folder/(key+'_mask.png'))
                record = {'candidate_id':key,'status':'generated','source':'source.png','mask':'mask.png',
                          'source_sha256':sha256(folder/'source.png'),'mask_sha256':sha256(folder/'mask.png'),
                          'params':params,'metrics':metrics,'output_sha256':sha256(folder/(key+'.png'))}
                write_json(folder/(key+'.json'),record)
                checked = replay(folder/(key+'.json'),folder/(key+'_replay.png'))
                record['replay_verified']=checked['exact_match']
                # The checked copy is redundant; retain only the recorded original output.
                (folder/(key+'_replay.png')).unlink()
                write_json(folder/(key+'.json'),record)
                records.append(record)
                metrics_rows.append({'source_id':row['source_id'],'category':row['category'],'candidate_id':key,
                                     'strength':strength,'segmentation_status':segmentation['status'],
                                     'output_sha256':record['output_sha256'],'replay_verified':True})
                draw.text((x+12,y+319),'70% strength' if preset!='minimal_poster' else '70% whitespace',font=font(12),fill='#657078')
                if key!='poster_thirds':
                    previews[key]=output.copy()
            sheet.paste(output.resize((276,276),Image.Resampling.LANCZOS),(x+12,y+36))
        draw.text((12,698),row['source_id']+' | '+row['category']+' | train only',font=font(14),fill='#35434b')
        sheet.save(folder/'comparison.jpg',quality=94)
        write_json(folder/'result.json',{'source':row,'segmentation':segmentation,'model':segmenter.provenance,'candidates':records})
        results.append({'source_id':row['source_id'],'category':row['category'],'segmentation_status':segmentation['status'],
                        'generated':sum(r['status']=='generated' for r in records),
                        'unavailable':sum(r['status']=='unavailable' for r in records)})
        if len(previews)==4 and (preview_source is None or (row['category']=='person' and preview_source['source']['category']!='person')):
            preview_source={'source':row,'images':previews}
        print(results[-1],flush=True)
    with (args.output/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(metrics_rows[0]))
        writer.writeheader();writer.writerows(metrics_rows)
    if args.write_previews:
        if preview_source is None:
            raise ValueError('No usable subject for preset previews; inspect the recorded failures')
        assets=ROOT/'src/object_collage/web_assets'
        from PIL import ImageOps
        for key,image in preview_source['images'].items():
            ImageOps.pad(image,(320,240),color='#f5eddf').save(assets/f'preset_{key}.png')
        write_json(assets/'preset_previews_source.json',{'source':preview_source['source'],
                   'selection':'first usable person training example; otherwise first usable fixed training example', 'strength':.7})
    summary={'size':args.size,'source_count':len(sources),'sources':results,
             'generated_edits':len(metrics_rows),'original_controls':len(sources),'replay_verified':len(metrics_rows),
             'heldout_sources_used':0,'style_inference_performed':False,'aesthetic_evaluation':'not yet performed',
             'preset_code_sha256':sha256(ROOT/'src/object_collage/presets.py'),'python':platform.python_version()}
    write_json(args.output/'summary.json',summary)
    cards=''.join(f'<section><h2>{html.escape(r["category"])} · {r["source_id"]}</h2><a href="{r["source_id"]}/comparison.jpg"><img loading="lazy" src="{r["source_id"]}/comparison.jpg" alt="Six fixed comparison versions"></a><p>Subject extraction: {r["segmentation_status"]}. <a href="{r["source_id"]}/result.json">Parameters and image attribution</a></p></section>' for r in results)
    (args.output/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Photo preset comparisons</title><style>body{max-width:1000px;margin:32px auto;padding:0 20px;background:#f5f2ec;color:#27323a;font:16px/1.6 system-ui}section{margin:40px 0}img{width:100%;height:auto}h1{line-height:1.2}h2{font-size:18px}a{color:#315d97}</style><h1>Fixed photo preset comparisons</h1><p>Training sources only. Original, warm film, muted vintage, subject pop, and two poster layouts. Failed subject extractions remain visible. These are technical comparisons, not aesthetic ratings.</p>'+cards+'</html>',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='sources'}),flush=True)


if __name__=='__main__':
    main()
