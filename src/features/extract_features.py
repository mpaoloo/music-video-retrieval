from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import cv2
import librosa
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm

from src.utils import load_config, make_dir, project_path, set_seed


def build_frame_model(device: torch.device) -> tuple[torch.nn.Module, object]:
    """Create a pretrained ResNet18 that returns a 512-dimensional vector."""
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(device)
    return model, weights.transforms()


def read_sampled_frames(video_path: str | Path, frame_count: int) -> list[Image.Image]:
    """Read a few evenly spaced frames from a video."""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        cap.release()
        return []

    indices = np.linspace(0, max(total_frames - 1, 0), frame_count).astype(int)
    frames = []

    for frame_index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))

    cap.release()
    return frames


@torch.no_grad()
def extract_video_feature(
    video_path: str | Path,
    model: torch.nn.Module,
    preprocess: object,
    device: torch.device,
    frame_count: int,
) -> np.ndarray:
    """Extract one averaged visual vector for a video."""
    frames = read_sampled_frames(video_path, frame_count)
    if not frames:
        raise RuntimeError(f"Could not read frames from {video_path}")

    batch = torch.stack([preprocess(frame) for frame in frames]).to(device)
    features = model(batch)
    features = features.mean(dim=0)
    return features.cpu().numpy().astype("float32")


def extract_audio_to_wav(video_path: str | Path, wav_path: str | Path, sample_rate: int) -> None:
    """Use ffmpeg to extract mono wav audio from a video file."""
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(wav_path),
    ]
    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {video_path}")


def extract_audio_feature(
    video_path: str | Path,
    sample_rate: int,
    n_mels: int,
    max_audio_seconds: int,
) -> np.ndarray:
    """Extract a simple audio vector based on log-mel statistics.

    This is deliberately simple: for every mel band we take mean and standard
    deviation over time. It gives a compact description of tempo/energy/timbre
    without adding heavy audio models.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = Path(tmp_dir) / "audio.wav"
        extract_audio_to_wav(video_path, wav_path, sample_rate)

        audio, _ = librosa.load(
            wav_path,
            sr=sample_rate,
            mono=True,
            duration=max_audio_seconds,
        )

    if len(audio) == 0:
        raise RuntimeError(f"No audio was extracted from {video_path}")

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=1024,
        hop_length=512,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    mean = log_mel.mean(axis=1)
    std = log_mel.std(axis=1)
    feature = np.concatenate([mean, std]).astype("float32")
    return feature


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])

    manifest_path = project_path(config["paths"]["manifest_path"])
    feature_dir = make_dir(config["paths"]["feature_dir"])
    video_feature_dir = make_dir(feature_dir / "video")
    audio_feature_dir = make_dir(feature_dir / "audio")
    features_csv_path = project_path(config["paths"]["train_features_path"])

    manifest = pd.read_csv(manifest_path)
    if args.limit is not None:
        manifest = manifest.head(args.limit).copy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    frame_model, preprocess = build_frame_model(device)

    rows = []
    for row in tqdm(manifest.to_dict("records"), desc="Extracting features"):
        pair_id = str(row["pair_id"])
        video_path = row["video_path"]
        video_out = video_feature_dir / f"{pair_id}.npy"
        audio_out = audio_feature_dir / f"{pair_id}.npy"

        try:
            if not video_out.exists():
                video_feature = extract_video_feature(
                    video_path=video_path,
                    model=frame_model,
                    preprocess=preprocess,
                    device=device,
                    frame_count=int(config["features"]["frame_count"]),
                )
                np.save(video_out, video_feature)

            if not audio_out.exists():
                audio_feature = extract_audio_feature(
                    video_path=video_path,
                    sample_rate=int(config["features"]["sample_rate"]),
                    n_mels=int(config["features"]["n_mels"]),
                    max_audio_seconds=int(config["features"]["max_audio_seconds"]),
                )
                np.save(audio_out, audio_feature)

            rows.append(
                {
                    **row,
                    "video_feature_path": str(video_out),
                    "audio_feature_path": str(audio_out),
                }
            )
        except Exception as error:
            print(f"Skipped pair_id={pair_id}: {error}")

    features = pd.DataFrame(rows)
    features.to_csv(features_csv_path, index=False)
    print(f"Saved feature table to {features_csv_path}")
    print(f"Feature rows: {len(features)}")


if __name__ == "__main__":
    main()
