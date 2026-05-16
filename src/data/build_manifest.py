from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.utils import load_config, make_dir, project_path, set_seed


def inspect_video(video_path: str | Path) -> dict:
    """Read basic technical information about a video file."""
    path = Path(video_path)
    if not path.exists():
        return {"is_valid": False, "duration": 0.0, "fps": 0.0, "frames": 0}

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"is_valid": False, "duration": 0.0, "fps": 0.0, "frames": 0}

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    duration = frames / fps if fps > 0 else 0.0
    is_valid = frames > 0 and duration > 1.0
    return {"is_valid": is_valid, "duration": duration, "fps": fps, "frames": frames}


def add_split(df: pd.DataFrame, train_ratio: float, val_ratio: float, seed: int) -> pd.DataFrame:
    """Add train/val/test split in a deterministic way."""
    rng = np.random.default_rng(seed)
    indices = np.arange(len(df))
    rng.shuffle(indices)

    train_end = int(len(df) * train_ratio)
    val_end = train_end + int(len(df) * val_ratio)

    split = np.array(["test"] * len(df), dtype=object)
    split[indices[:train_end]] = "train"
    split[indices[train_end:val_end]] = "val"

    result = df.copy()
    result["split"] = split
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])

    metadata_path = project_path(config["paths"]["metadata_path"])
    manifest_path = project_path(config["paths"]["manifest_path"])
    make_dir(manifest_path.parent)

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file was not found: {metadata_path}. "
            "Run `python -m src.data.download_harmonyset` from the project root first "
            "(it creates this CSV after loading Hugging Face metadata; wait until you see "
            "'Saved initial metadata' or downloads finishing)."
        )

    metadata = pd.read_csv(metadata_path)
    metadata = metadata[metadata["download_ok"] == True].copy()  # noqa: E712

    rows = []
    for row in metadata.to_dict("records"):
        info = inspect_video(row["video_path"])
        if not info["is_valid"]:
            continue

        rows.append(
            {
                "pair_id": row["pair_id"],
                "url": row["url"],
                "video_path": row["video_path"],
                "annotation": row.get("annotation", ""),
                "duration": info["duration"],
                "fps": info["fps"],
                "frames": info["frames"],
            }
        )

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise RuntimeError("No valid videos found. Try downloading a larger subset.")

    max_video_seconds = float(config["data"]["max_video_seconds"])
    manifest = manifest[manifest["duration"] <= max_video_seconds].reset_index(drop=True)

    manifest = add_split(
        manifest,
        train_ratio=float(config["data"]["train_ratio"]),
        val_ratio=float(config["data"]["val_ratio"]),
        seed=int(config["seed"]),
    )
    manifest.to_csv(manifest_path, index=False)

    print(f"Saved manifest to {manifest_path}")
    print(manifest["split"].value_counts().to_string())
    print(f"Average duration: {manifest['duration'].mean():.1f} seconds")


if __name__ == "__main__":
    main()
