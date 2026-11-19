"""Local blind pairwise annotation for a frozen candidate batch."""
import hashlib
import json
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .io import read_json
from .selector_data import load_batch, pairs_for_source


def tasks_for_batch(batch):
    tasks = []
    for source_id in batch['source_ids']:
        entry = batch['sources'].get(source_id)
        if not entry or any(item['status'] not in {'passed', 'failed'} for item in entry['candidates']):
            continue
        pairs = pairs_for_source(entry)
        for index, (a, b) in enumerate(pairs):
            repeat = index == 0 and len(pairs) >= 5
            for round_index in range(2 if repeat else 1):
                task_id = f'{source_id}:{a}:{b}:{round_index}'
                flip = int(hashlib.sha256(task_id.encode()).hexdigest(), 16) % 2 == 1
                if round_index == 1:
                    flip = not (int(hashlib.sha256(f'{source_id}:{a}:{b}:0'.encode()).hexdigest(), 16) % 2 == 1)
                tasks.append({'task_id': task_id, 'source_id': source_id, 'split': entry['split'],
                              'left': b if flip else a, 'right': a if flip else b,
                              'repeat': round_index == 1, 'estimated_subject_count': entry['estimated_subject_count']})
    return tasks


def read_labels(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def source_ids_for_split(batch, split, photo_rows):
    original_splits = {row['source_id']: row['split'] for row in photo_rows}
    assignment = batch.get('split_assignment')
    return [source_id for source_id in batch['source_ids']
            if (assignment[source_id] if assignment is not None else
                original_splits[source_id]) == split]


def validate_label(value, task, annotator):
    if not isinstance(value, dict) or value.get('task_id') != task['task_id']:
        raise ValueError('Unknown task')
    if value.get('annotator') != annotator or not isinstance(annotator, str) or not annotator or len(annotator) > 64:
        raise ValueError('Invalid annotator')
    if value.get('preference') not in {'left', 'right', 'tie', 'neither'}:
        raise ValueError('Choose left, right, tie or neither')
    count = value.get('subject_count')
    if type(count) is not int or not 0 <= count <= 10 or value.get('subject_count_confirmed') is not True:
        raise ValueError('Confirm subject count in [0, 10]')
    if not isinstance(value.get('unacceptable'), dict) or not isinstance(value.get('identity'), dict):
        raise ValueError('Candidate quality and identity reviews are required')
    for side in ('left', 'right'):
        if value.get(side) != task[side]:
            raise ValueError('Candidate order mismatch')
        if type(value.get('unacceptable', {}).get(side)) is not bool:
            raise ValueError('Unacceptable flag required for each candidate')
        if value.get('identity', {}).get(side) not in {'preserved', 'changed', 'unclear'}:
            raise ValueError('Identity review required for each candidate')
    if value['preference'] == 'left' and value['unacceptable']['left']:
        raise ValueError('Preferred candidate is marked unacceptable')
    if value['preference'] == 'right' and value['unacceptable']['right']:
        raise ValueError('Preferred candidate is marked unacceptable')
    if value['preference'] == 'neither' and not all(value['unacceptable'].values()):
        raise ValueError('Mark both candidates unacceptable when choosing neither')
    if value['preference'] == 'tie' and any(value['unacceptable'].values()):
        raise ValueError('A tie requires both candidates to be acceptable')


def validate_ranking(value, entry):
    if not isinstance(value, dict) or value.get('source_id') != entry['source_id']:
        raise ValueError('Unknown source')
    annotator = value.get('annotator')
    if not isinstance(annotator, str) or not annotator or len(annotator) > 64:
        raise ValueError('Invalid annotator')
    if any(item['status'] not in {'passed', 'failed'} for item in entry['candidates']):
        raise ValueError('Source candidates are still generating')
    ids = [item['candidate_id'] for item in entry['candidates'] if item['status'] == 'passed']
    ranking = value.get('ranked_ids')
    if (not isinstance(ranking, list) or len(ranking) > len(ids) or
            not all(isinstance(item, str) for item in ranking) or
            len(set(ranking)) != len(ranking) or not set(ranking).issubset(ids)):
        raise ValueError('Ranking must contain each selected successful candidate at most once')
    count = value.get('subject_count')
    if type(count) is not int or not 0 <= count <= 10 or value.get('subject_count_confirmed') is not True:
        raise ValueError('Confirm subject count in [0, 10]')
    unacceptable, identity = value.get('unacceptable'), value.get('identity')
    if not isinstance(unacceptable, dict) or set(unacceptable) != set(ids) or any(type(flag) is not bool for flag in unacceptable.values()):
        raise ValueError('Review acceptability for every candidate')
    selected = set(ranking)
    if any(unacceptable[candidate_id] == (candidate_id in selected) for candidate_id in ids):
        raise ValueError('Selected candidates must be acceptable; unselected candidates must be unacceptable')
    if not isinstance(identity, dict) or set(identity) != set(ids) or any(
        not isinstance(result, str) or result not in {'preserved', 'changed', 'unclear'}
        for result in identity.values()):
        raise ValueError('Review subject identity for every candidate')


def serve_labels(root, batch_path, port=7862, host='127.0.0.1'):
    root, batch_path = Path(root).resolve(), Path(batch_path).resolve()
    batch = load_batch(root, batch_path)
    tasks = tasks_for_batch(batch)
    lock = threading.Lock()
    state = {'batch': batch, 'tasks': tasks,
             'task_by_id': {task['task_id']: task for task in tasks},
             'mtime_ns': batch_path.stat().st_mtime_ns}

    def snapshot():
        nonlocal state
        mtime = batch_path.stat().st_mtime_ns
        if mtime == state['mtime_ns']:
            return state
        with lock:
            if mtime != state['mtime_ns']:
                try:
                    updated = read_json(batch_path)
                except json.JSONDecodeError:
                    return state
                if updated['source_ids'] != state['batch']['source_ids']:
                    raise ValueError('Candidate source selection changed while labeling')
                updated_tasks = tasks_for_batch(updated)
                state = {'batch': updated, 'tasks': updated_tasks,
                         'task_by_id': {task['task_id']: task for task in updated_tasks},
                         'mtime_ns': mtime}
        return state

    class Handler(BaseHTTPRequestHandler):
        def _json(self, value, status=200):
            body = json.dumps(value, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            query = parse_qs(url.query)
            if url.path == '/':
                body = (Path(__file__).parent/'web_assets/selector_rank.html').read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(body)
                return
            if url.path == '/pairs':
                body = (Path(__file__).parent/'web_assets/selector_label.html').read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(body)
                return
            if url.path == '/api/rank-next':
                current = snapshot()
                split = query.get('split', ['train'])[0]
                annotator = query.get('annotator', [''])[0]
                if split not in {'train', 'val', 'test'} or not annotator or len(annotator) > 64:
                    self._json({'error': 'Invalid split or annotator'}, 400)
                    return
                selected_ids = source_ids_for_split(current['batch'], split, _photo_rows)
                with lock:
                    completed = {row['source_id'] for row in read_labels(batch_path.parent/f'ranks_{split}.jsonl')
                                 if row['annotator'] == annotator}
                ready = [current['batch']['sources'][source_id] for source_id in selected_ids
                         if source_id in current['batch']['sources'] and all(
                             item['status'] in {'passed', 'failed'}
                             for item in current['batch']['sources'][source_id]['candidates'])]
                entry = next((row for row in ready if row['source_id'] not in completed and
                              sum(item['status'] == 'passed' for item in row['candidates']) >= 2), None)
                if entry is None:
                    self._json({'source': None, 'done': len(completed), 'ready': len(ready),
                                'total': len(selected_ids), 'generating': len(ready) < len(selected_ids)})
                    return
                source_id = entry['source_id']
                candidates = [item for item in entry['candidates'] if item['status'] == 'passed']
                candidates.sort(key=lambda item: hashlib.sha256(
                    f"{annotator}:{source_id}:{item['candidate_id']}".encode()).digest())
                self._json({'source': {'source_id': source_id, 'category': _photo_by_id[source_id]['category'],
                                       'source_url': f'/image/{source_id}/source',
                                       'estimated_subject_count': entry['estimated_subject_count'],
                                       'candidates': [{'id': item['candidate_id'],
                                                       'url': f"/image/{source_id}/{item['candidate_id']}"}
                                                      for item in candidates]},
                            'done': len(completed), 'ready': len(ready), 'total': len(selected_ids),
                            'generating': len(ready) < len(selected_ids)})
                return
            if url.path == '/api/next':
                current = snapshot()
                split = query.get('split', ['train'])[0]
                annotator = query.get('annotator', [''])[0]
                if split not in {'train', 'val', 'test'} or not annotator or len(annotator) > 64:
                    self._json({'error': 'Invalid split or annotator'}, 400)
                    return
                with lock:
                    completed = {row['task_id'] for row in read_labels(batch_path.parent/f'labels_{split}.jsonl')
                                 if row['annotator'] == annotator}
                available = [task for task in current['tasks'] if task['split'] == split]
                generating = (len(current['batch']['sources']) < len(current['batch']['source_ids']) or
                              any(item['status'] not in {'passed', 'failed'}
                                  for entry in current['batch']['sources'].values() for item in entry['candidates']))
                pending = next((task for task in available if task['task_id'] not in completed), None)
                if pending is None:
                    self._json({'task': None, 'done': len(completed.intersection(t['task_id'] for t in available)),
                                'total': len(available), 'generating': generating})
                    return
                result = dict(pending)
                result['source_url'] = f"/image/{pending['source_id']}/source"
                result['left_url'] = f"/image/{pending['source_id']}/{pending['left']}"
                result['right_url'] = f"/image/{pending['source_id']}/{pending['right']}"
                result['category'] = next(row['category'] for row in
                                          _photo_rows if row['source_id'] == pending['source_id'])
                self._json({'task': result, 'done': len(available)-sum(t['task_id'] not in completed for t in available),
                            'total': len(available), 'generating': generating})
                return
            if url.path.startswith('/image/'):
                current = snapshot()
                parts = url.path.split('/')
                if len(parts) != 4 or parts[2] not in current['batch']['sources']:
                    self.send_error(404)
                    return
                entry = current['batch']['sources'][parts[2]]
                if parts[3] == 'source':
                    path = root/entry['source_path']
                else:
                    candidate = next((c for c in entry['candidates'] if c['candidate_id'] == parts[3] and c['status'] == 'passed'), None)
                    if candidate is None:
                        self.send_error(404)
                        return
                    path = batch_path.parent/candidate['relative_dir']/'output.png'
                body = path.read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', 'image/png' if path.suffix == '.png' else 'image/jpeg')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(404)

        def do_POST(self):
            if self.path not in {'/api/label', '/api/rank'}:
                self.send_error(404)
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 4096:
                    raise ValueError('Invalid request size')
                value = json.loads(self.rfile.read(length))
                if not isinstance(value, dict):
                    raise ValueError('Expected a label object')
                current = snapshot()
                if self.path == '/api/rank':
                    source_id = value.get('source_id')
                    if not isinstance(source_id, str):
                        raise ValueError('Unknown source')
                    entry = current['batch']['sources'].get(source_id)
                    if entry is None:
                        raise ValueError('Unknown source')
                    validate_ranking(value, entry)
                    value['split'] = entry['split']
                    label_path = batch_path.parent/f"ranks_{entry['split']}.jsonl"
                    duplicate_key = 'source_id'
                else:
                    task_id = value.get('task_id')
                    if not isinstance(task_id, str):
                        raise ValueError('Unknown task')
                    task = current['task_by_id'].get(task_id)
                    if task is None:
                        raise ValueError('Unknown task')
                    validate_label(value, task, value.get('annotator'))
                    value['source_id'] = task['source_id']
                    value['split'] = task['split']
                    label_path = batch_path.parent/f"labels_{task['split']}.jsonl"
                    duplicate_key = 'task_id'
                value['created_utc'] = datetime.now(timezone.utc).isoformat()
                with lock:
                    if any(row[duplicate_key] == value[duplicate_key] and row['annotator'] == value['annotator']
                           for row in read_labels(label_path)):
                        raise ValueError('Source or task already labeled by this annotator')
                    with label_path.open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')
                        stream.flush()
                self._json({'saved': True})
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                self._json({'error': str(exc)}, 400)

    _photo_rows = json.loads((root/'data/photo120/manifest.json').read_text(encoding='utf-8'))['images']
    _photo_by_id = {row['source_id']: row for row in _photo_rows}
    server = ThreadingHTTPServer((host, port), Handler)
    print(f'Labeling: http://{host}:{port}/ ({len(tasks)} tasks)', flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
