from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoTowerModel(nn.Module):
    """Two linear towers + L2 normalization for contrastive video–audio retrieval."""

    def __init__(self, video_dim: int, audio_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.video_tower = nn.Sequential(
            nn.Linear(video_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.audio_tower = nn.Sequential(
            nn.Linear(audio_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def encode_video(self, video_features: torch.Tensor) -> torch.Tensor:
        z = self.video_tower(video_features)
        return F.normalize(z, p=2, dim=-1)

    def encode_audio(self, audio_features: torch.Tensor) -> torch.Tensor:
        z = self.audio_tower(audio_features)
        return F.normalize(z, p=2, dim=-1)

    def forward(
        self, video_features: torch.Tensor, audio_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_video(video_features), self.encode_audio(audio_features)


def similarity_matrix(
    video_embeddings: torch.Tensor,
    audio_embeddings: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Scaled dot-product similarity (B, B) for a batch of aligned pairs."""
    logits = video_embeddings @ audio_embeddings.T
    return logits / temperature
