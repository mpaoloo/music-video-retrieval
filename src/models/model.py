from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_tower(input_dim: int, hidden_dim: int, embedding_dim: int, dropout: float) -> nn.Sequential:
    """Две полносвязных ступени с LayerNorm и GELU между входом и эмбеддингом."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, embedding_dim),
    )


class TwoTowerModel(nn.Module):
    """Две башни видео / аудио + L2-нормализация для contrastive retrieval"""

    def __init__(
        self,
        video_dim: int,
        audio_dim: int,
        embedding_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.video_tower = _make_tower(video_dim, hidden_dim, embedding_dim, dropout)
        self.audio_tower = _make_tower(audio_dim, hidden_dim, embedding_dim, dropout)

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
    """Измеряем сходство между видео и аудио эмбеддингами косинусным расстоянием"""
    logits = video_embeddings @ audio_embeddings.T
    return logits / temperature
