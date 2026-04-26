"""
Dataset utilities — src/dataset.py
===================================
Downloads MovieLens 1M, encodes IDs, builds positive/negative interaction pairs.
"""

from __future__ import annotations
import os
import zipfile
import urllib.request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
DATA_DIR = "data/ml-1m"


# ─── Download ──────────────────────────────────────────────────────────────────

def download_movielens(data_dir: str = DATA_DIR) -> None:
    if os.path.exists(data_dir):
        print("Dataset already present.")
        return
    os.makedirs("data", exist_ok=True)
    print("Downloading MovieLens 1M (~6 MB)...")
    urllib.request.urlretrieve(ML1M_URL, "data/ml-1m.zip")
    with zipfile.ZipFile("data/ml-1m.zip") as z:
        z.extractall("data")
    os.remove("data/ml-1m.zip")
    print("Done.")


# ─── Loading & encoding ────────────────────────────────────────────────────────

def load_and_encode(
    data_dir: str = DATA_DIR,
    pos_threshold: float = 4.0,
) -> tuple[pd.DataFrame, dict, dict, int, int]:
    """
    Returns:
        ratings     – DataFrame with user_idx, item_idx, rating columns
        user2idx    – raw user_id → 0-based index
        item2idx    – raw item_id → 0-based index
        num_users   – total unique users (for Embedding table size)
        num_items   – total unique items
    """
    ratings = pd.read_csv(
        f"{data_dir}/ratings.dat",
        sep="::", engine="python",
        names=["user_id", "item_id", "rating", "timestamp"],
    )
    # Encode IDs starting at 1 (0 reserved for padding_idx)
    user2idx = {u: i + 1 for i, u in enumerate(sorted(ratings["user_id"].unique()))}
    item2idx = {m: i + 1 for i, m in enumerate(sorted(ratings["item_id"].unique()))}

    ratings["user_idx"] = ratings["user_id"].map(user2idx)
    ratings["item_idx"] = ratings["item_id"].map(item2idx)

    num_users = max(user2idx.values()) + 1   # +1 for padding idx=0
    num_items = max(item2idx.values()) + 1

    print(f"Loaded {len(ratings):,} ratings | Users: {num_users:,} | Items: {num_items:,}")
    return ratings, user2idx, item2idx, num_users, num_items


# ─── Negative sampling ─────────────────────────────────────────────────────────

def build_interaction_df(
    ratings: pd.DataFrame,
    pos_threshold: float = 4.0,
    neg_ratio: int = 4,
    seed: int = 42,
    num_items: int | None = None,
) -> pd.DataFrame:
    """
    Positive: rating >= pos_threshold
    Negative: random (user, item) pairs NOT in the user's history
    """
    pos = ratings[ratings["rating"] >= pos_threshold][["user_idx", "item_idx"]].copy()
    pos["label"] = 1.0

    pos_set = set(zip(pos["user_idx"], pos["item_idx"]))
    n_neg = len(pos) * neg_ratio
    _max_item = num_items or int(ratings["item_idx"].max()) + 1

    rng = np.random.default_rng(seed)
    neg_u, neg_v = [], []
    while len(neg_u) < n_neg:
        us = rng.integers(1, _max_item, size=n_neg * 2)
        vs = rng.integers(1, _max_item, size=n_neg * 2)
        for u, v in zip(us, vs):
            if (u, v) not in pos_set:
                neg_u.append(u)
                neg_v.append(v)
            if len(neg_u) >= n_neg:
                break

    neg = pd.DataFrame({"user_idx": neg_u, "item_idx": neg_v, "label": 0.0})
    df = pd.concat([pos, neg], ignore_index=True).sample(frac=1, random_state=seed)
    print(f"Positives: {len(pos):,} | Negatives: {len(neg):,} | Total: {len(df):,}")
    return df


def split_data(
    df: pd.DataFrame, val_frac: float = 0.1, test_frac: float = 0.1, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, temp = train_test_split(df, test_size=val_frac + test_frac, random_state=seed)
    val, test = train_test_split(temp, test_size=test_frac / (val_frac + test_frac), random_state=seed)
    print(f"Split → Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
    return train, val, test


# ─── PyTorch Dataset ───────────────────────────────────────────────────────────

class InteractionDataset(Dataset):
    """Simple (user_idx, item_idx, label) dataset."""

    def __init__(self, df: pd.DataFrame):
        self.users  = torch.as_tensor(df["user_idx"].values, dtype=torch.long)
        self.items  = torch.as_tensor(df["item_idx"].values, dtype=torch.long)
        self.labels = torch.as_tensor(df["label"].values,    dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.users[idx], self.items[idx], self.labels[idx]


def make_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = 4096,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    pin = torch.cuda.is_available()
    kw = dict(num_workers=num_workers, pin_memory=pin)
    train_loader = DataLoader(InteractionDataset(train_df), batch_size=batch_size, shuffle=True,  **kw)
    val_loader   = DataLoader(InteractionDataset(val_df),   batch_size=batch_size, shuffle=False, **kw)
    test_loader  = DataLoader(InteractionDataset(test_df),  batch_size=batch_size, shuffle=False, **kw)
    return train_loader, val_loader, test_loader
