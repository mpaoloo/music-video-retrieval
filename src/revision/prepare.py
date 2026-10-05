"""Загрузка фиксированной выборки с журналом ошибок и продолжением после сбоя."""
import argparse
import hashlib
import json
import random
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, parse_qs

import cv2
import pandas as pd
from huggingface_hub import hf_hub_download
from .common import ROOT, DATA, RUN, config, ffmpeg, save_json, split_for, status


def youtube_id(url):
    u = urlparse(url)
    if u.netloc.endswith('youtu.be'):
        return u.path.strip('/')
    if '/shorts/' in u.path:
        return u.path.split('/shorts/')[1].split('/')[0]
    return parse_qs(u.query).get('v', [''])[0]


def inspect(path):
    cap = cv2.VideoCapture(str(path))
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ok, _ = cap.read()
    cap.release()
    seconds = frames / fps if fps > 0 else 0
    if not ok or seconds < 2 or seconds > 180:
        raise ValueError(f'Invalid video or unsupported duration: {seconds:.2f}s')
    return seconds


def download(row):
    row = dict(row)
    path = DATA / 'videos' / f"{row['video_id']}.mp4"
    try:
        if not path.exists():
            cmd = [sys.executable, '-m', 'yt_dlp', '--no-playlist', '--no-progress',
                   '--retries', '1', '--fragment-retries', '1', '--socket-timeout', '15',
                   '--ffmpeg-location', ffmpeg(), '--merge-output-format', 'mp4',
                   '-f', 'bv*[vcodec^=avc1][height<=480]+ba[ext=m4a]/b[vcodec^=avc1][height<=480]',
                   '--max-filesize', '80M', '-o', str(path), row['url']]
            if shutil.which('node'):
                cmd[3:3] = ['--js-runtimes', 'node']
            run = subprocess.run(cmd, text=True, capture_output=True, timeout=100)
            if run.returncode:
                raise RuntimeError(run.stderr[-1000:])
        row['duration'] = inspect(path)
        row['video_path'] = str(path.relative_to(ROOT))
        row['download_ok'] = True
        row['error'] = ''
    except Exception as error:
        row.update(download_ok=False, video_path='', duration=0, error=str(error)[-1200:])
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    cfg = config()
    limit = args.limit or cfg['limit']
    (DATA / 'videos').mkdir(parents=True, exist_ok=True)
    source = hf_hub_download('Zzitang/HarmonySet', 'HarmonySet_Train.json',
                            repo_type='dataset', revision=cfg['dataset_revision'])
    items = json.loads(open(source).read())
    random.Random(cfg['seed']).shuffle(items)
    rows, seen = [], set()
    for item in items:
        vid = youtube_id(item.get('video', ''))
        if not re.fullmatch(r'[A-Za-z0-9_-]{11}', vid) or vid in seen:
            continue
        seen.add(vid)
        annotation = next((m.get('value', '') for m in item.get('conversations', [])
                           if m.get('from') == 'model'), '')
        rows.append(dict(pair_id=str(item['id']), video_id=vid, url=item['video'],
                         annotation=annotation, split=split_for(vid)))
        if len(rows) >= limit:
            break
    pd.DataFrame(rows).to_csv(DATA / 'requested.csv', index=False)
    results = []
    status('downloading', requested=len(rows), completed=0)
    with ThreadPoolExecutor(max_workers=cfg['download_workers']) as pool:
        futures = [pool.submit(download, row) for row in rows]
        for future in as_completed(futures):
            results.append(future.result())
            if len(results) % 10 == 0 or len(results) == len(rows):
                table = pd.DataFrame(results).sort_values('video_id')
                table.to_csv(DATA / 'downloads.csv', index=False)
                status('downloading', requested=len(rows), completed=len(results),
                       successful=int(table.download_ok.sum()))
            # При массовой блокировке YouTube прекращаем попытки, а не ждём часами.
            if len(results) == 30 and not any(r['download_ok'] for r in results):
                for f in futures:
                    f.cancel()
                raise RuntimeError('0/30 videos downloaded. See downloads.csv; stop and fix access first.')
    table = pd.DataFrame(results)
    valid = table[table.download_ok].copy()
    # Точные копии файлов не должны оказаться по разные стороны разбиения.
    valid['sha256'] = [hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                       for p in valid.video_path]
    valid = valid.drop_duplicates('sha256').sort_values('video_id').reset_index(drop=True)
    valid.to_csv(DATA / 'manifest.csv', index=False)
    save_json(RUN / 'dataset_summary.json', dict(requested=len(rows), downloaded=int(table.download_ok.sum()),
              unique_files=len(valid), split_counts=valid.split.value_counts().to_dict(),
              dataset_revision=cfg['dataset_revision']))
    status('download_complete', valid=len(valid))


if __name__ == '__main__':
    main()
