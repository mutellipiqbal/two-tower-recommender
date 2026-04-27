"""
Retrieval Evaluation — src/evaluate.py
========================================
Builds a FAISS inner-product (cosine) index over all item embeddings,
then measures Recall@K and NDCG@K — the standard retrieval metrics.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ─── FAISS helper — checked at call time, not import time ─────────────────────

def _get_faiss():
    """
    Import faiss fresh every call so installing it mid-session works correctly.
    Raises a clear error if not installed instead of silently returning None.
    """
    try:
        import faiss
        return faiss
    except ImportError:
        raise ImportError(
            "\nfaiss-cpu is not installed. Run:\n"
            "    !pip install faiss-cpu==1.13.2\n"
            "No runtime restart needed — then re-run this cell."
        )


# ─── Build FAISS index ────────────────────────────────────────────────────────

def build_faiss_index(item_embeddings: np.ndarray):
    """
    Builds a brute-force inner-product (cosine) FAISS index.
    For production scale, swap with faiss.IndexIVFPQ for approximate search.
    """
    faiss = _get_faiss()
    item_embeddings = item_embeddings.astype(np.float32)
    faiss.normalize_L2(item_embeddings)
    d     = item_embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(item_embeddings)
    print(f"FAISS index built: {index.ntotal:,} items, dim={d}")
    return index


# ─── Compute item embeddings ──────────────────────────────────────────────────

@torch.no_grad()
def compute_all_item_embeddings(
    model: nn.Module,
    num_items: int,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    """Embed every item ID once and return as float32 numpy array."""
    model.eval()
    item_ids = torch.arange(1, num_items, dtype=torch.long)  # skip padding_idx=0
    loader   = DataLoader(TensorDataset(item_ids), batch_size=batch_size)
    embs     = []
    for (batch,) in loader:
        embs.append(model.get_item_embeddings(batch.to(device)).cpu().numpy())
    return np.vstack(embs).astype(np.float32)


# ─── Recall@K / NDCG@K ───────────────────────────────────────────────────────

def recall_at_k(retrieved: np.ndarray, relevant: set, k: int) -> float:
    hits = sum(1 for r in retrieved[:k] if r in relevant)
    return hits / min(len(relevant), k)


def ndcg_at_k(retrieved: np.ndarray, relevant: set, k: int) -> float:
    dcg = sum(
        1.0 / np.log2(rank + 2)
        for rank, r in enumerate(retrieved[:k])
        if r in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


# ─── Main evaluation function ─────────────────────────────────────────────────

def evaluate_retrieval(
    model: nn.Module,
    test_df,
    num_items: int,
    device: torch.device,
    k_values: list[int] | None = None,
    n_users: int = 500,
) -> dict[str, float]:
    """
    For a sample of users, retrieve top-K items via FAISS and measure
    Recall@K and NDCG@K against their held-out positive interactions.
    """
    faiss    = _get_faiss()   # raises clear error if not installed
    k_values = k_values or [10, 20, 50]

    # Build item index
    item_embs = compute_all_item_embeddings(model, num_items, device)
    index     = build_faiss_index(item_embs.copy())

    # Ground-truth: positives in test set
    pos_test       = test_df[test_df["label"] == 1]
    user_positives = pos_test.groupby("user_idx")["item_idx"].apply(set).to_dict()

    sampled_users = list(user_positives.keys())[:n_users]
    metrics: dict[str, list[float]] = {f"recall@{k}": [] for k in k_values}
    metrics.update({f"ndcg@{k}": [] for k in k_values})

    max_k = max(k_values)
    for uid in sampled_users:
        user_tensor = torch.tensor([uid], dtype=torch.long)
        u_emb = (
            model.get_user_embeddings(user_tensor.to(device))
            .cpu().numpy().astype(np.float32)
        )
        faiss.normalize_L2(u_emb)

        _, top_indices = index.search(u_emb, max_k)  # (1, max_k)
        # +1 because item IDs start at 1 (padding_idx=0 was skipped)
        retrieved = top_indices[0] + 1

        relevant = user_positives[uid]
        for k in k_values:
            metrics[f"recall@{k}"].append(recall_at_k(retrieved, relevant, k))
            metrics[f"ndcg@{k}"].append(ndcg_at_k(retrieved, relevant, k))

    return {name: float(np.mean(vals)) for name, vals in metrics.items()}
