from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.evaluate import load_trained_model
from src.utils import load_config, project_path


@torch.no_grad()
def encode_one_video(model, video_feature_path: str, device: torch.device) -> np.ndarray:
    video_feature = np.load(video_feature_path).astype("float32")
    video_tensor = torch.from_numpy(video_feature).unsqueeze(0).to(device)
    return model.encode_video(video_tensor).cpu().numpy()[0]


@torch.no_grad()
def encode_all_audio(model, table: pd.DataFrame, device: torch.device) -> np.ndarray:
    audio_embeddings = []
    for row in table.to_dict("records"):
        audio_feature = np.load(row["audio_feature_path"]).astype("float32")
        audio_tensor = torch.from_numpy(audio_feature).unsqueeze(0).to(device)
        audio_embedding = model.encode_audio(audio_tensor).cpu().numpy()[0]
        audio_embeddings.append(audio_embedding)
    return np.vstack(audio_embeddings)


def short_text(text: str, max_len: int = 180) -> str:
    text = str(text).replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def choose_query_row(table: pd.DataFrame, row_index: int | None, pair_id: str | None) -> pd.Series:
    if pair_id is not None:
        selected = table[table["pair_id"].astype(str) == str(pair_id)]
        if selected.empty:
            raise ValueError(f"pair_id={pair_id} не найден в таблице признаков")
        return selected.iloc[0]

    row_index = 0 if row_index is None else row_index
    if row_index < 0 or row_index >= len(table):
        raise ValueError(f"row_index должен быть между 0 и {len(table) - 1}.")
    return table.iloc[row_index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--row-index", type=int, default=0) # индекс строки с видео в таблице если не задано video-id или pair-id)
    parser.add_argument("--video-id", type=int, default=None)
    parser.add_argument("--pair-id", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    config = load_config(args.config)
    features_path = project_path(config["paths"]["train_features_path"])
    model_path = Path(args.model_path) if args.model_path else project_path(config["paths"]["model_dir"]) / "best_model.pt"

    table = pd.read_csv(features_path)
    if args.split != "all":
        table = table[table["split"] == args.split].reset_index(drop=True)

    if table.empty:
        raise RuntimeError("Выбранный сплит пуст")

    row_index = args.video_id if args.video_id is not None else args.row_index
    query = choose_query_row(table, row_index=row_index, pair_id=args.pair_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_trained_model(model_path, device)

    query_embedding = encode_one_video(model, query["video_feature_path"], device)
    audio_embeddings = encode_all_audio(model, table, device)
    scores = audio_embeddings @ query_embedding
    order = np.argsort(-scores)[: args.top_k]

    print("Запрошенное видео")
    print(f"pair_id: {query['pair_id']}")
    print(f"path: {query['video_path']}")
    print()
    print(f"Топ-{args.top_k} рекомендованных аудио")

    for rank, index in enumerate(order, start=1):
        row = table.iloc[index]
        marker = "верный pair_id" if str(row["pair_id"]) == str(query["pair_id"]) else ""
        print(
            f"{rank}. pair_id={row['pair_id']} "
            f"score={scores[index]:.3f}{marker}"
        )
        print(f"   video/audio source: {row['video_path']}")
        print(f"   анотация: {short_text(row.get('annotation', ''))}")


if __name__ == "__main__":
    main()
