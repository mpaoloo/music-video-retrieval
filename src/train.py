from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.evaluate import compute_retrieval_metrics, embed_split
from src.models.model import TwoTowerModel, similarity_matrix
from src.utils import load_config, make_dir, project_path, set_seed


class FeatureDataset(Dataset):
    """Dataset that reads already extracted .npy features."""

    def __init__(self, table: pd.DataFrame) -> None:
        self.table = table.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.table.iloc[index]
        video_feature = np.load(row["video_feature_path"]).astype("float32")
        audio_feature = np.load(row["audio_feature_path"]).astype("float32")
        return torch.from_numpy(video_feature), torch.from_numpy(audio_feature)


def infer_feature_dims(table: pd.DataFrame) -> tuple[int, int]:
    """Read one pair of features to understand input dimensions."""
    first = table.iloc[0]
    video_dim = int(np.load(first["video_feature_path"]).shape[0])
    audio_dim = int(np.load(first["audio_feature_path"]).shape[0])
    return video_dim, audio_dim


def train_one_epoch(
    model: TwoTowerModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    temperature: float,
) -> float:
    model.train()
    losses = []

    for video_features, audio_features in tqdm(loader, desc="Training", leave=False):
        video_features = video_features.to(device)
        audio_features = audio_features.to(device)

        video_embeddings, audio_embeddings = model(video_features, audio_features)
        logits = similarity_matrix(video_embeddings, audio_embeddings, temperature)
        labels = torch.arange(logits.shape[0], device=device)

        # Two directions make training a bit more stable:
        # video -> audio and audio -> video.
        loss_video = torch.nn.functional.cross_entropy(logits, labels)
        loss_audio = torch.nn.functional.cross_entropy(logits.T, labels)
        loss = (loss_video + loss_audio) / 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else 0.0


def save_loss_plot(history: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    plt.plot(history["epoch"], history["train_loss"], label="train loss")
    plt.plot(history["epoch"], history["val_mrr"], label="val MRR")
    plt.xlabel("Epoch")
    plt.title("Training dynamics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--no-plots", action="store_true", help="Skip png plots. Useful for quick technical checks.")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config["seed"]))

    features_path = project_path(config["paths"]["train_features_path"])
    model_dir = make_dir(config["paths"]["model_dir"])
    report_dir = make_dir(config["paths"]["report_dir"])

    features = pd.read_csv(features_path)
    train_table = features[features["split"] == "train"].reset_index(drop=True)
    val_table = features[features["split"] == "val"].reset_index(drop=True)

    if len(train_table) < 2:
        raise RuntimeError("Need at least 2 training examples for contrastive learning.")
    if len(val_table) < 2:
        print("Validation split is too small, using train split for validation metrics.")
        val_table = train_table.copy()

    video_dim, audio_dim = infer_feature_dims(train_table)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TwoTowerModel(
        video_dim=video_dim,
        audio_dim=audio_dim,
        embedding_dim=int(config["training"]["embedding_dim"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    loader = DataLoader(
        FeatureDataset(train_table),
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    best_mrr = -1.0
    history_rows = []
    best_path = model_dir / "best_model.pt"
    temperature = float(config["training"]["temperature"])

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        train_loss = train_one_epoch(model, loader, optimizer, device, temperature)

        video_embeddings, audio_embeddings = embed_split(model, val_table, device)
        val_metrics = compute_retrieval_metrics(video_embeddings, audio_embeddings, top_k=5)
        val_mrr = val_metrics["mrr"]

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_hit_at_1": val_metrics["hit_at_1"],
                "val_hit_at_5": val_metrics["hit_at_5"],
                "val_mrr": val_mrr,
            }
        )

        print(
            f"Epoch {epoch:02d}: "
            f"loss={train_loss:.4f}, "
            f"val_hit@1={val_metrics['hit_at_1']:.3f}, "
            f"val_hit@5={val_metrics['hit_at_5']:.3f}, "
            f"val_mrr={val_mrr:.3f}"
        )

        if val_mrr > best_mrr:
            best_mrr = val_mrr
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "video_dim": video_dim,
                    "audio_dim": audio_dim,
                    "embedding_dim": int(config["training"]["embedding_dim"]),
                    "temperature": temperature,
                    "config": config,
                },
                best_path,
            )

    history = pd.DataFrame(history_rows)
    history_path = report_dir / "training_history.csv"
    history.to_csv(history_path, index=False)
    if not args.no_plots:
        save_loss_plot(history, report_dir / "training_dynamics.png")

    print(f"Saved best model to {best_path}")
    print(f"Saved training history to {history_path}")


if __name__ == "__main__":
    main()
