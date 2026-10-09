"""Generate comparisons from every successful frozen development segmentation."""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from object_collage.composition_study import candidate_specs, render_candidate
from object_collage.io import read_json, write_json, sha256

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, choices=[512,1024], default=512)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources, skipped = [], []
    for folder in sorted((ROOT/'experiments/dev_baseline').iterdir()):
        if not (folder/'result.json').exists():
            continue
        record = read_json(folder/'result.json')
        if record['status'] != 'ok':
            skipped.append({'source_id':folder.name, 'reason':record['status']})
            continue
        saved = read_json(folder/'params.json')
        for name in ['source', 'mask']:
            if sha256(folder/f'{name}.png') != saved[f'{name}_sha256']:
                raise ValueError(f'Frozen input changed: {folder}/{name}')
        sources.append((folder.name, folder/'source.png', folder/'mask.png', record['canvas_mapping']['content_box_xyxy']))
    cup = ROOT/'experiments/dragondiffusion_check/smoke-512'
    if (cup/'mask.png').exists():
        sources.append(('official_cup', cup/'input.png', cup/'mask.png', [0,0,512,512]))
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 15)
    small_font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 12)
    rows, summaries = [], []
    for sid, source, mask_path, content in sources:
        folder = args.output/sid
        folder.mkdir()
        image = Image.open(source).convert('RGB')
        mask = np.asarray(Image.open(mask_path))>0
        image.save(folder/'source.png')
        Image.fromarray(mask.astype(np.uint8)*255).save(folder/'mask.png')
        sheet = Image.new('RGB', (3*340, 3*384), '#e9edf1')
        draw = ImageDraw.Draw(sheet)
        records = []
        for index, (key, title, params) in enumerate(candidate_specs(mask, content, args.size)):
            x, y = (index%3)*340, (index//3)*384
            draw.text((x+10,y+8), title, font=font, fill='#17212b')
            record = {'candidate_id':key, 'title':title, 'params':params,
                      'source_sha256':sha256(folder/'source.png'), 'mask_sha256':sha256(folder/'mask.png')}
            if 'unavailable' in params:
                record.update(status='infeasible', reason=params['unavailable'])
                draw.multiline_text((x+18,y+130), 'No feasible crop\nFull subject retained as a hard constraint', font=font, fill='#6b4750', spacing=8)
            else:
                output, visible, metrics = render_candidate(image, mask, params)
                output.save(folder/f'{key}.png')
                visible.save(folder/f'{key}_mask.png')
                # Validate replay determinism using the serialized parameter boundary.
                write_json(folder/f'{key}.json', record)
                replay, replay_mask, replay_metrics = render_candidate(image, mask, read_json(folder/f'{key}.json')['params'])
                if replay.tobytes()!=output.tobytes() or replay_mask.tobytes()!=visible.tobytes() or replay_metrics!=metrics:
                    raise ValueError(f'Replay mismatch: {sid}/{key}')
                record.update(status='generated', metrics=metrics, output_sha256=sha256(folder/f'{key}.png'), replay_verified=True)
                sheet.paste(output.resize((320,320), Image.Resampling.LANCZOS), (x+10,y+34))
                density = metrics['negative_space_edge_density']
                caption = f"Subject {metrics['subject_area_fraction']:.1%} | background edges {density:.1%}" if density is not None else 'No evaluable background region'
                draw.text((x+10,y+361), caption, font=small_font, fill='#465261')
                rows.append({'source_id':sid, 'candidate_id':key, **metrics})
            write_json(folder/f'{key}.json', record)
            records.append(record)
        draw.multiline_text((700,820), f'{sid}\nPhotography: crop / colour only\nCollage: solid background\nMetrics are not aesthetic scores', font=font, fill='#243444', spacing=10)
        sheet.save(folder/'comparison.jpg', quality=95)
        write_json(folder/'candidates.json', records)
        summaries.append({'source_id':sid, 'generated':sum(r['status']=='generated' for r in records), 'infeasible':[r['candidate_id'] for r in records if r['status']=='infeasible']})
        print(summaries[-1], flush=True)
    with (args.output/'metrics.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.output/'summary.json', {'size':args.size, 'sources':summaries, 'skipped_sources':skipped,
               'selection':'all baseline sources with status ok, plus the previously tested official cup',
               'generated_count':len(rows), 'aesthetic_ranking':'not performed', 'inference':'none; cached masks only',
               'code_sha256':sha256(ROOT/'src/object_collage/composition_study.py')})


if __name__ == '__main__':
    main()
