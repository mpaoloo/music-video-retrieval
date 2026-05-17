from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.model import TwoTowerModel
from src.utils import load_config, make_dir, project_path, set_seed


@torch.no_grad()
def embed_split(
    model: TwoTowerModel,
    table: pd.DataFrame,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Закодируем все видео и аудио признаки из одного датасета"""
    model.eval()
    video_embeddings = []
    audio_embeddings = []

    for row in table.to_dict("records"):
        video_feature = np.load(row["video_feature_path"]).astype("float32")
        audio_feature = np.load(row["audio_feature_path"]).astype("float32")

        video_tensor = torch.from_numpy(video_feature).unsqueeze(0).to(device)
        audio_tensor = torch.from_numpy(audio_feature).unsqueeze(0).to(device)

        video_embedding, audio_embedding = model(video_tensor, audio_tensor)
        video_embeddings.append(video_embedding.cpu().numpy()[0])
        audio_embeddings.append(audio_embedding.cpu().numpy()[0])

    return np.vstack(video_embeddings), np.vstack(audio_embeddings)


def compute_retrieval_metrics(
    video_embeddings: np.ndarray,
    audio_embeddings: np.ndarray,
    top_k: int = 5,
) -> dict[str, float]:
    """Вычисляем метрики Hit@1, Hit@K и MRR
    Строки - это видео, столбцы - это кандидаты аудио треков
    """
    scores = video_embeddings @ audio_embeddings.T
    ranks = []

    for i in range(scores.shape[0]):
        order = np.argsort(-scores[i])
        rank = int(np.where(order == i)[0][0]) + 1
        ranks.append(rank)

    ranks = np.array(ranks)
    requested_top_k = top_k
    effective_top_k = min(top_k, scores.shape[1])

    return {
        "hit_at_1": float(np.mean(ranks <= 1)),
        f"hit_at_{requested_top_k}": float(np.mean(ranks <= effective_top_k)),
        "mrr": float(np.mean(1.0 / ranks)),
        "mean_rank": float(np.mean(ranks)),
        "num_queries": int(scores.shape[0]),
        "num_candidates": int(scores.shape[1]),
        "effective_top_k": int(effective_top_k),
    }


def random_baseline(num_candidates: int, top_k: int = 5) -> dict[str, float]:
    """Ожидаемый случайный baseline для одного правильного элемента среди N кандидатов"""
    requested_top_k = top_k
    effective_top_k = min(top_k, num_candidates)
    harmonic = sum(1.0 / rank for rank in range(1, num_candidates + 1))
    return {
        "hit_at_1": 1.0 / num_candidates,
        f"hit_at_{requested_top_k}": effective_top_k / num_candidates,
        "mrr": harmonic / num_candidates,
        "effective_top_k": int(effective_top_k),
    }


def load_trained_model(model_path: str | Path, device: torch.device) -> tuple[TwoTowerModel, dict]:
    checkpoint = torch.load(model_path, map_location=device)
    cfg = checkpoint.get("config") or {}
    training_cfg = cfg.get("training") or {}
    hidden_dim = int(
        checkpoint.get("tower_hidden_dim") or training_cfg.get("tower_hidden_dim", 512)
    )
    dropout = float(
        checkpoint.get("tower_dropout") or training_cfg.get("tower_dropout", 0.25)
    )
    model = TwoTowerModel(
        video_dim=int(checkpoint["video_dim"]),
        audio_dim=int(checkpoint["audio_dim"]),
        embedding_dim=int(checkpoint["embedding_dim"]),
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def save_metrics_plot(metrics: dict[str, float], baseline: dict[str, float], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    names = ["hit_at_1", "hit_at_5", "mrr"]
    model_values = [metrics.get(name, np.nan) for name in names]
    baseline_values = [baseline.get(name, np.nan) for name in names]

    x = np.arange(len(names))
    width = 0.35

    plt.figure(figsize=(7, 4))
    plt.bar(x - width / 2, model_values, width, label="model")
    plt.bar(x + width / 2, baseline_values, width, label="random")
    plt.xticks(x, names)
    plt.ylim(0, 1)
    plt.title("Retrieval metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--no-plots", action="store_true", help="Пропустить графики, полезно для быстрой проверки")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config["seed"]))

    features_path = project_path(config["paths"]["train_features_path"])
    model_path = Path(args.model_path) if args.model_path else project_path(config["paths"]["model_dir"]) / "best_model.pt"
    report_dir = make_dir(config["paths"]["report_dir"])

    features = pd.read_csv(features_path)
    table = features[features["split"] == args.split].reset_index(drop=True)
    if len(table) < 2:
        raise RuntimeError(f"'{args.split}'должен содержать минимум 2 объекта")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_trained_model(model_path, device)

    video_embeddings, audio_embeddings = embed_split(model, table, device)
    metrics = compute_retrieval_metrics(video_embeddings, audio_embeddings, top_k=5)
    baseline = random_baseline(num_candidates=len(table), top_k=5)

    metrics_table = pd.DataFrame(
        [
            {"name": "model", **metrics},
            {"name": "random_baseline", **baseline},
        ]
    )
    metrics_path = report_dir / f"metrics_{args.split}.csv"
    metrics_table.to_csv(metrics_path, index=False)
    if not args.no_plots:
        save_metrics_plot(metrics, baseline, report_dir / f"metrics_{args.split}.png")

    print(f"Evaluation split: {args.split}")
    print(metrics_table.to_string(index=False))
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
