from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from yt_dlp import YoutubeDL

from src.utils import load_config, make_dir, project_path, set_seed


def get_model_answer(row: dict) -> str:
    """Extract the long annotation text from a HarmonySet row."""
    conversations = row.get("conversations", [])
    if not conversations:
        return ""

    for message in conversations:
        if message.get("from") == "model":
            return str(message.get("value", ""))
    return ""


def load_harmonyset_metadata(dataset_name: str, limit: int, seed: int) -> pd.DataFrame:
    """Load a small subset of HarmonySet metadata from Hugging Face.

    HarmonySet contains YouTube links and text annotations. The actual videos
    are not stored inside the dataset, so we download them in the next step.

    The Hub repo ships plain JSON files (HarmonySet_Train.json) without a
    dataset loading script, so ``load_dataset(dataset_name)`` fails on recent
    ``datasets`` versions. We load the train split explicitly as JSON.
    """
    train_json_url = (
        f"https://huggingface.co/datasets/{dataset_name}/resolve/main/HarmonySet_Train.json"
    )
    dataset = load_dataset("json", data_files={"train": train_json_url}, split="train")
    dataset = dataset.shuffle(seed=seed)

    rows = []
    for item in dataset.select(range(min(limit, len(dataset)))):
        rows.append(
            {
                "pair_id": str(item.get("id")),
                "url": item.get("video", ""),
                "annotation": get_model_answer(item),
            }
        )

    return pd.DataFrame(rows)


def find_downloaded_file(raw_dir: Path, pair_id: str) -> Path | None:
    """Find a downloaded media file for one HarmonySet pair."""
    candidates = list(raw_dir.glob(f"{pair_id}.*"))
    if not candidates:
        return None
    return candidates[0]


def download_one_video(url: str, pair_id: str, raw_dir: Path) -> tuple[bool, str, str]:
    """Download one YouTube video.

    Returns:
        success flag, local path, error text
    """
    output_template = str(raw_dir / f"{pair_id}.%(ext)s")
    options = {
        "format": "best[ext=mp4][height<=480]/best[height<=480]/best",
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
        "noplaylist": True,
    }

    try:
        with YoutubeDL(options) as ydl:
            ydl.download([url])
    except Exception as error:
        return False, "", str(error)

    downloaded_file = find_downloaded_file(raw_dir, pair_id)
    if downloaded_file is None:
        return False, "", "yt-dlp did not create a file"

    return True, str(downloaded_file), ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])

    limit = args.limit or int(config["data"]["subset_size"])
    raw_dir = make_dir(config["paths"]["raw_dir"])
    metadata_path = project_path(config["paths"]["metadata_path"])
    make_dir(metadata_path.parent)

    print(f"Loading HarmonySet metadata, limit={limit}")
    metadata = load_harmonyset_metadata(
        dataset_name=config["data"]["dataset_name"],
        limit=limit,
        seed=config["seed"],
    )

    if args.metadata_only:
        metadata.to_csv(metadata_path, index=False)
        print(f"Saved metadata to {metadata_path}")
        return

    results = []
    for row in tqdm(metadata.to_dict("records"), desc="Downloading videos"):
        pair_id = row["pair_id"]
        url = row["url"]

        existing_file = find_downloaded_file(raw_dir, pair_id)
        if existing_file is not None:
            success, video_path, error = True, str(existing_file), ""
        else:
            success, video_path, error = download_one_video(url, pair_id, raw_dir)

        results.append(
            {
                **row,
                "download_ok": success,
                "video_path": video_path,
                "download_error": error,
            }
        )

    result_df = pd.DataFrame(results)
    result_df.to_csv(metadata_path, index=False)

    ok_count = int(result_df["download_ok"].sum())
    print(f"Saved metadata to {metadata_path}")
    print(f"Downloaded {ok_count}/{len(result_df)} videos")
    print("If many links fail, reduce the scope and mention this as a data limitation.")


if __name__ == "__main__":
    main()
