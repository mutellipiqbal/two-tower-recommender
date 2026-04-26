"""
Trainer — src/trainer.py
=========================
Training loop with MLflow 3.x experiment tracking.
Uses AdamW + CosineAnnealingLR (industry standard).
"""

from __future__ import annotations
import time
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import mlflow
import mlflow.pytorch
import numpy as np


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    loss_fn: nn.Module,
    device: torch.device,
    use_in_batch_negatives: bool = True,
) -> float:
    model.train()
    total_loss = 0.0

    for users, items, labels in loader:
        users  = users.to(device, non_blocking=True)
        items  = items.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if use_in_batch_negatives:
            # In-batch negatives: (B, B) logit matrix → cross-entropy on row
            logits = model.forward_in_batch(users, items)   # (B, B)
            targets = torch.arange(len(users), device=device)
            loss = nn.CrossEntropyLoss()(logits, targets)
        else:
            logits = model(users, items)   # (B,)
            loss = loss_fn(logits, labels)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0

    for users, items, labels in loader:
        users  = users.to(device, non_blocking=True)
        items  = items.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(users, items)
        loss   = loss_fn(logits, labels)
        total_loss += loss.item()

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()
    probs      = torch.sigmoid(torch.tensor(all_logits)).numpy()
    auc        = roc_auc_score(all_labels, probs)

    return {"loss": total_loss / len(loader), "auc": auc}


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    device: torch.device,
    run_name: str = "two_tower",
) -> nn.Module:
    """Full training loop with MLflow 3.x tracking."""

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    total_steps = cfg["epochs"] * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=cfg["lr"] * 0.01
    )
    loss_fn = nn.BCEWithLogitsLoss()

    mlflow.set_experiment(cfg["mlflow_experiment"])

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(cfg)
        best_val_auc = 0.0

        for epoch in range(1, cfg["epochs"] + 1):
            t0 = time.time()
            train_loss = train_epoch(
                model, train_loader, optimizer, scheduler, loss_fn, device,
                use_in_batch_negatives=cfg.get("in_batch_negatives", True),
            )
            val_metrics = evaluate(model, val_loader, loss_fn, device)
            elapsed     = time.time() - t0

            print(
                f"Epoch {epoch:>3}/{cfg['epochs']} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_auc={val_metrics['auc']:.4f} | "
                f"time={elapsed:.1f}s"
            )

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss":   val_metrics["loss"],
                    "val_auc":    val_metrics["auc"],
                    "lr":         scheduler.get_last_lr()[0],
                },
                step=epoch,
            )

            if val_metrics["auc"] > best_val_auc:
                best_val_auc = val_metrics["auc"]
                torch.save(model.state_dict(), "best_two_tower.pt")
                mlflow.log_artifact("best_two_tower.pt")

        mlflow.log_metric("best_val_auc", best_val_auc)
        print(f"\nBest val AUC: {best_val_auc:.4f}")

    # Reload best weights
    model.load_state_dict(torch.load("best_two_tower.pt", map_location=device))
    return model
