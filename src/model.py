"""
Two-Tower Retrieval Model — src/model.py
========================================
Industry-standard retrieval model used by companies like YouTube, Pinterest,
and Airbnb for candidate generation at scale.

Architecture:
    user_id ─► UserTower (Embedding → LayerNorm → MLP) ─► L2-norm ─►┐
                                                                       ├─► dot product ─► logit
    item_id ─► ItemTower (Embedding → LayerNorm → MLP) ─► L2-norm ─►┘

Training: BCEWithLogitsLoss with in-batch negatives
Serving:  Pre-compute all item embeddings → FAISS ANN index
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Shared MLP block: Linear → LayerNorm → GELU → Dropout."""

    def __init__(self, in_dim: int, hidden_dims: list[int], dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        self.net = nn.Sequential(*layers)
        self.out_dim = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Tower(nn.Module):
    """Single tower: Embedding → LayerNorm → MLP → L2-normalised output."""

    def __init__(
        self,
        num_entities: int,
        embedding_dim: int,
        hidden_dims: list[int],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_entities, embedding_dim, padding_idx=0)
        nn.init.trunc_normal_(self.embedding.weight, std=0.02)
        self.norm = nn.LayerNorm(embedding_dim)
        self.mlp = MLP(embedding_dim, hidden_dims, dropout)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.norm(self.embedding(ids))   # (B, D)
        x = self.mlp(x)                      # (B, out_dim)
        return F.normalize(x, p=2, dim=-1)   # L2-norm → cosine-sim ready


class TwoTowerModel(nn.Module):
    """
    Two-Tower retrieval model.

    Score  = temperature × (user_emb · item_emb)   [cosine similarity × scale]
    Loss   = BCEWithLogitsLoss
    Serving = pre-compute item embeddings → FAISS inner-product index
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [128, 64]
        self.user_tower = Tower(num_users, embedding_dim, hidden_dims, dropout)
        self.item_tower = Tower(num_items, embedding_dim, hidden_dims, dropout)
        # Learnable temperature (like in SimCLR / SentenceBERT)
        self.log_temperature = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(min=1.0, max=100.0)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Return logits for BCEWithLogitsLoss."""
        u = self.user_tower(user_ids)   # (B, D)
        v = self.item_tower(item_ids)   # (B, D)
        return (u * v).sum(-1) * self.temperature   # (B,)

    def forward_in_batch(
        self, user_ids: torch.Tensor, item_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        In-batch negative training (more efficient than pre-sampled negatives).
        Returns a (B, B) similarity matrix — diagonal = positives, off-diagonal = negatives.
        Used with cross-entropy loss over the row.
        """
        u = self.user_tower(user_ids)   # (B, D)
        v = self.item_tower(item_ids)   # (B, D)
        return (u @ v.T) * self.temperature   # (B, B)

    @torch.no_grad()
    def get_user_embeddings(self, user_ids: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.user_tower(user_ids)

    @torch.no_grad()
    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.item_tower(item_ids)
