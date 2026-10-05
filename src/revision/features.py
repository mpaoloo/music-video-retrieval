"""Замороженные ResNet50/AST и старые mel-признаки для честного сравнения."""
import argparse
import gc
import json
import re
import subprocess
import tempfile

import cv2
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
from transformers import ASTFeatureExtractor, ASTForAudioClassification

from .common import ROOT, DATA, RUN, config, cache_key, device, ffmpeg, save_json, status


def semantic_warning(annotation):
    # Только слабая эвристика. Она не заменяет разметку совместимости человеком.
    words = r'\bmismatch\b|\bclash(?:es|ing)?\b|\bdissonance\b|\bdisjointed\b|\bdisconnected\b|contrasts sharply'
    sentences = re.split(r'[.!?\n]+', str(annotation).lower())
    return sum(bool(re.search(words, sentence)) for sentence in sentences) >= 2


def gates(music, speech, cfg):
    music = np.asarray(music)
    speech = np.asarray(speech)
    fraction = float(np.mean(music >= cfg['music_threshold']))
    # Число 0.67 означает примерно две трети, поэтому допуск нужен для 2/3.
    music_ok = fraction + .005 >= cfg['min_music_fraction']
    clean = music_ok and float(np.mean(speech)) <= cfg['speech_threshold']
    return music_ok, clean, fraction


def frames(path, count, max_seconds):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total <= 0:
        cap.release()
        raise ValueError('Unreadable video')
    last = min(total - 1, int(fps * max_seconds) - 1)
    images = []
    for index in np.linspace(0, last, count).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if ok:
            images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    if len(images) < max(2, count // 2):
        raise ValueError(f'Only {len(images)} frames decoded')
    return images


def waveform(path, cfg):
    with tempfile.TemporaryDirectory() as temp:
        wav = temp + '/audio.wav'
        subprocess.run([ffmpeg(), '-v', 'error', '-y', '-i', str(path),
                        '-t', str(cfg['max_seconds']), '-vn', '-ac', '1',
                        '-ar', str(cfg['sample_rate']), wav], check=True,
                       capture_output=True, timeout=60)
        audio, sr = sf.read(wav, dtype='float32')
    if len(audio) < sr or not np.isfinite(audio).all():
        raise ValueError('Audio too short or non-finite')
    return audio


def mel_feature(audio, sr):
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=96, n_fft=1024, hop_length=512)
    db = librosa.power_to_db(mel, ref=np.max)
    return np.concatenate([db.mean(1), db.std(1)]).astype('float32')


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    cfg = config()
    torch.set_num_threads(4)
    dev = device()
    key = cache_key(cfg)
    folder = DATA / 'features' / key
    folder.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(DATA / 'manifest.csv', dtype={'pair_id': str})
    if args.limit:
        table = table.head(args.limit)
    weights = ResNet50_Weights.IMAGENET1K_V2
    visual = resnet50(weights=weights).eval().to(dev)
    visual.fc = torch.nn.Identity()
    transform = weights.transforms()
    extractor = ASTFeatureExtractor.from_pretrained(cfg['audio_model'], revision=cfg['audio_revision'])
    audio_model = ASTForAudioClassification.from_pretrained(
        cfg['audio_model'], revision=cfg['audio_revision'], use_safetensors=True).eval().to(dev)
    labels = {str(v).lower(): int(k) for k, v in audio_model.config.id2label.items()}
    music_idx, speech_idx = labels['music'], labels['speech']
    save_json(RUN / 'feature_protocol.json', dict(cache_key=key, config=cfg, device=str(dev),
              music_label=music_idx, speech_label=speech_idx,
              note='AST scores are uncalibrated audio-event scores, not ground-truth labels.'))
    records, errors = [], []
    for index, row in enumerate(table.to_dict('records')):
        dest = folder / f"{row['video_id']}.npz"
        try:
            if not dest.exists():
                path = ROOT / row['video_path']
                batch = torch.stack([transform(im) for im in frames(path, cfg['frame_count'], cfg['max_seconds'])])
                video = visual(batch.to(dev)).mean(0).cpu().numpy()
                audio = waveform(path, cfg)
                size = cfg['sample_rate'] * cfg['window_seconds']
                starts = np.unique(np.linspace(0, max(0, len(audio) - size), cfg['audio_windows']).astype(int))
                windows = [audio[start:start + size] for start in starts]
                ast_features, probabilities = [], []
                for segment in windows:
                    inputs = extractor(segment, sampling_rate=cfg['sample_rate'], return_tensors='pt')
                    result = audio_model.audio_spectrogram_transformer(**{k: v.to(dev) for k, v in inputs.items()})
                    embedding = result.pooler_output
                    probs = audio_model.classifier(embedding).sigmoid()
                    ast_features.append(embedding.cpu().numpy()[0])
                    probabilities.append(probs.cpu().numpy()[0])
                probs = np.stack(probabilities)
                values = dict(video=video.astype('float32'), ast=np.mean(ast_features, 0).astype('float32'),
                              mel=mel_feature(audio, cfg['sample_rate']), music=probs[:, music_idx],
                              speech=probs[:, speech_idx], rms=np.array(float(np.sqrt(np.mean(audio ** 2)))))
                if not all(np.isfinite(v).all() for v in values.values()):
                    raise ValueError('Non-finite feature')
                temp = dest.with_suffix('.tmp.npz')
                np.savez_compressed(temp, **values)
                temp.replace(dest)
            with np.load(dest) as f:
                music_ok, clean, fraction = gates(f['music'], f['speech'], cfg)
                rms = float(f['rms'])
                warning = semantic_warning(row['annotation'])
                records.append(dict(**row, feature_path=str(dest.relative_to(ROOT)),
                                    music_mean=float(f['music'].mean()), speech_mean=float(f['speech'].mean()),
                                    music_fraction=fraction, music_ok=bool(music_ok and rms > 1e-5),
                                    audio_clean=bool(clean and rms > 1e-5), semantic_warning=warning,
                                    train_clean=bool(clean and rms > 1e-5 and not warning)))
        except Exception as error:
            errors.append(dict(video_id=row['video_id'], error=str(error)))
            print(f"Feature error {row['video_id']}: {error}", flush=True)
        if (index + 1) % 10 == 0 or index + 1 == len(table):
            pd.DataFrame(records).to_csv(DATA / 'features.csv', index=False)
            pd.DataFrame(errors, columns=['video_id', 'error']).to_csv(DATA / 'feature_errors.csv', index=False)
            status('features', completed=index + 1, total=len(table), successful=len(records), device=str(dev))
    if not records:
        raise RuntimeError('No features extracted; inspect feature_errors.csv')
    table = pd.DataFrame(records)
    # Случайная выборка для проверки автоматических правил, а не только удачные кейсы.
    audit = table.sample(min(30, len(table)), random_state=cfg['seed']).copy()
    for col in ['human_music', 'human_speech', 'human_pair_fit', 'comment']:
        audit[col] = ''
    if not (DATA / 'audit.csv').exists():
        audit.to_csv(DATA / 'audit.csv', index=False)
    counts = table.groupby('split')[['music_ok', 'audio_clean', 'train_clean']].sum().to_dict()
    save_json(RUN / 'filter_counts.json', counts)
    status('features_complete', successful=len(table), cache_key=key)


if __name__ == '__main__':
    main()
