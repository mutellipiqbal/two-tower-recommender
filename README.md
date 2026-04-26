# Two-Tower Retrieval Model 🗼🗼

> **Stage in the recommendation funnel:** Candidate Generation (Retrieval)
> Narrows millions of items down to ~hundreds of candidates per user.

Standalone PyTorch port of the [Databricks Two-Tower reference](https://docs.databricks.com/aws/en/machine-learning/train-recommender-models).
**No Spark. No Databricks. Runs free on Google Colab T4 or Kaggle P100.**

---

## Architecture

```
user_id ─► UserTower ─► L2-norm ─────────────────────┐
           [Emb → LayerNorm → MLP]                    ├─► dot product ─► P(click)
item_id ─► ItemTower ─► L2-norm ─────────────────────┘
           [Emb → LayerNorm → MLP]
```

- **Training objective:** In-batch negative cross-entropy (industry standard — used by Google, Meta, Pinterest)
- **Inference:** Pre-compute all item embeddings → FAISS ANN index → top-K retrieval
- **Metrics:** Recall@K, NDCG@K (retrieval metrics), AUC (ranking metric)

---

## Library Versions (April 2026)

| Library | Version | Why |
|---|---|---|
| `torch` | 2.11.0 | `torch.compile` for ~20% GPU speedup |
| `mlflow` | 3.11.1 | Experiment tracking, model registry |
| `faiss-cpu` | 1.13.2 | ANN index for retrieval evaluation |
| `scikit-learn` | 1.8.0 | AUC metric, train/val/test split |
| `pandas` | 3.0.2 | Data loading |

---

## Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/two-tower-recommender.git
cd two-tower-recommender
pip install -r requirements.txt
jupyter notebook two_tower_recommender.ipynb
```

Or open directly in Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/two-tower-recommender/blob/main/two_tower_recommender.ipynb)

---

## Project Structure

```
two-tower-recommender/
├── two_tower_recommender.ipynb   ← Main notebook (start here)
├── requirements.txt
├── src/
│   ├── model.py      ← TwoTowerModel, Tower, MLP
│   ├── dataset.py    ← MovieLens download, encoding, DataLoader
│   ├── trainer.py    ← Training loop with MLflow 3.x
│   └── evaluate.py   ← FAISS index, Recall@K, NDCG@K
└── README.md
```

---

## What was changed from the Databricks original

| Databricks | This repo |
|---|---|
| `TorchDistributor` (requires PySpark cluster) | Standard `torch.compile` + DataParallel |
| `StreamingDataset` (requires S3/DBFS) | `torch.utils.data.DataLoader` |
| `dbutils` / `spark.sql` | Removed |
| Databricks-hosted MLflow | Open-source `mlflow==3.11.1` (local) |
| Delta Lake | MovieLens 1M CSV (auto-downloaded) |
| `TorchRec` sharded embeddings | `nn.Embedding` (sufficient for <10M entities) |

---

## Free GPU Platforms

| Platform | GPU | Free Quota | Best for |
|---|---|---|---|
| Google Colab | T4 (16 GB) | ~12 hrs/session | Quick experiments |
| Kaggle Notebooks | P100 (16 GB) | 30 hrs/week | Reproducible runs |
| Paperspace Gradient | M4000 (8 GB) | Free tier | Longer runs |
| Lightning.ai | T4 | 22 hrs/month | MLflow UI |

---

## Production Path

```
[Two-Tower] ──► FAISS / Vertex Matching Engine / Pinecone / Milvus
                         │
                    top-100 candidates
                         │
               [DLRM Re-ranker] ──► top-10 final recommendations
```

See the companion project: **`dlrm-recommender`**
