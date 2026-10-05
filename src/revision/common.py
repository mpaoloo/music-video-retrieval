"""Пути и небольшие общие функции нового эксперимента."""
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/revision'
RUN = ROOT / 'runs/revision'


def config():
    return json.loads((ROOT / 'configs/revision.json').read_text())


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def status(stage, **details):
    record = dict(stage=stage, time=datetime.now(timezone.utc).isoformat(), **details)
    save_json(RUN / 'status.json', record)
    print(json.dumps(record, ensure_ascii=False), flush=True)


def split_for(video_id):
    # Одинаковый YouTube id всегда попадает в один split, даже при дозагрузке.
    value = int(hashlib.sha256(('42:' + video_id).encode()).hexdigest()[:8], 16) % 100
    return 'train' if value < 70 else 'val' if value < 85 else 'test'


def cache_key(cfg):
    fields = ['audio_model', 'audio_revision', 'frame_count', 'max_seconds',
              'audio_windows', 'window_seconds', 'sample_rate']
    payload = {'version': 2, 'resnet': 'IMAGENET1K_V2', **{k: cfg[k] for k in fields}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def device():
    import torch
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def ffmpeg():
    import shutil
    import imageio_ffmpeg
    return shutil.which('ffmpeg') or imageio_ffmpeg.get_ffmpeg_exe()


def checkpoint_backup():
    """Копируем маленькие результаты в Drive, если путь указан в ноутбуке."""
    import shutil
    target = os.environ.get('MVR_BACKUP_DIR')
    if target:
        shutil.copytree(RUN, Path(target) / 'runs/revision', dirs_exist_ok=True)
        for name in ['manifest.csv', 'features.csv', 'audit.csv']:
            if (DATA / name).exists():
                dest = Path(target) / 'data/revision' / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(DATA / name, dest)
