# ─────────────────────────────────────────────────────────────────────────────
# train_and_push.py  —  Two-Tower
# Run this in your Colab/Kaggle notebook AFTER training.
# Pushes the trained model + FAISS index to Hugging Face Hub so the
# Spaces deployment can load them at startup.
#
# Usage:
#   1. pip install huggingface_hub faiss-cpu
#   2. huggingface-cli login          (paste your HF write token)
#   3. python deploy/train_and_push.py
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
import numpy as np
import torch
import faiss
from huggingface_hub import HfApi, create_repo

# ── Config (edit these) ───────────────────────────────────────────────────────
HF_USERNAME   = "YOUR_HF_USERNAME"          # e.g. "ikbal"
REPO_NAME     = "two-tower-movie-recommender"
MODEL_PT      = "best_two_tower.pt"         # saved by trainer.py
ITEM_EMBS_NPY = "item_embeddings.npy"       # saved by evaluate.py
FAISS_INDEX   = "item_index.faiss"
META_JSON     = "meta.json"

# ── Step 1: Build FAISS index from saved embeddings ───────────────────────────
print("Building FAISS index...")
item_embs = np.load(ITEM_EMBS_NPY).astype(np.float32)
faiss.normalize_L2(item_embs)

d     = item_embs.shape[1]
index = faiss.IndexFlatIP(d)
index.add(item_embs)
faiss.write_index(index, FAISS_INDEX)
print(f"  Index: {index.ntotal:,} items, dim={d}  →  {FAISS_INDEX}")

# ── Step 2: Save metadata (needed by the Spaces app) ─────────────────────────
# Edit num_users / num_items to match what your training printed
meta = {
    "num_users":    6041,    # from load_and_encode() output
    "num_items":    3953,    # from load_and_encode() output
    "embedding_dim": 64,
    "hidden_dims":  [256, 128, 64],
    "dropout":       0.1,
}
with open(META_JSON, "w") as f:
    json.dump(meta, f, indent=2)
print(f"  Meta saved: {META_JSON}")

# ── Step 3: Push all files to HF Hub ─────────────────────────────────────────
repo_id = f"{HF_USERNAME}/{REPO_NAME}"
print(f"\nPushing to https://huggingface.co/{repo_id} ...")

create_repo(repo_id, repo_type="model", exist_ok=True, private=False)
api = HfApi()

for path in [MODEL_PT, ITEM_EMBS_NPY, FAISS_INDEX, META_JSON]:
    if not os.path.exists(path):
        print(f"  SKIP (not found): {path}")
        continue
    api.upload_file(path_or_fileobj=path, path_in_repo=path, repo_id=repo_id)
    size_mb = os.path.getsize(path) / 1e6
    print(f"  ✓  {path}  ({size_mb:.1f} MB)")

print(f"\nDone! Model available at: https://huggingface.co/{repo_id}")
print("Update HF_REPO in deploy/app.py with this repo ID, then push the Space.")
