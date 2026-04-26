"""
deploy/app.py  —  Two-Tower recommender  (Hugging Face Spaces / Gradio)
========================================================================
Loads trained model + FAISS index from HF Hub at startup.
Users enter a user ID and get their top-N movie recommendations.

Deploy steps:
  1. Create a new HF Space:  https://huggingface.co/new-space
     - SDK: Gradio
     - Hardware: CPU Basic (free)
  2. Clone the Space repo and copy this file as app.py
  3. Add a requirements.txt (see deploy/requirements.txt)
  4. Set HF_REPO below to your model repo
  5. git push → Space auto-rebuilds
"""

from __future__ import annotations
import json
import os
import sys
import urllib.request

import numpy as np
import faiss
import torch
import gradio as gr
from huggingface_hub import hf_hub_download

# ── Edit this ─────────────────────────────────────────────────────────────────
HF_REPO = "YOUR_HF_USERNAME/two-tower-movie-recommender"
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, ".")
from src.model import TwoTowerModel   # bundled alongside app.py in the Space repo

# ── Load artefacts at startup ─────────────────────────────────────────────────
print("Loading model artefacts from Hub...")

meta      = json.loads(open(hf_hub_download(HF_REPO, "meta.json")).read())
model_pt  = hf_hub_download(HF_REPO, "best_two_tower.pt")
index_bin = hf_hub_download(HF_REPO, "item_index.faiss")

model = TwoTowerModel(
    num_users    = meta["num_users"],
    num_items    = meta["num_items"],
    embedding_dim= meta["embedding_dim"],
    hidden_dims  = meta["hidden_dims"],
    dropout      = meta["dropout"],
)
model.load_state_dict(torch.load(model_pt, map_location="cpu", weights_only=True))
model.eval()

index = faiss.read_index(index_bin)
print(f"Ready — {index.ntotal:,} items indexed.")

# ── Optional: MovieLens title lookup ─────────────────────────────────────────
# Download movies.dat for display names (small file, ~170 KB)
MOVIES_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
ITEM_TITLES: dict[int, str] = {}
try:
    import zipfile, io
    data = urllib.request.urlopen(MOVIES_URL, timeout=10).read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        with z.open("ml-1m/movies.dat") as f:
            for line in f.read().decode("latin-1").splitlines():
                parts = line.strip().split("::")
                if len(parts) >= 2:
                    ITEM_TITLES[int(parts[0])] = parts[1]
    print(f"Loaded {len(ITEM_TITLES):,} movie titles.")
except Exception as e:
    print(f"Title lookup unavailable: {e}")


# ── Inference function ─────────────────────────────────────────────────────────
def recommend(user_id: int, top_k: int) -> str:
    """Retrieve top-K items for a user via FAISS ANN search."""
    if not (1 <= user_id < meta["num_users"]):
        return f"User ID must be between 1 and {meta['num_users'] - 1}."

    uid_tensor = torch.tensor([user_id], dtype=torch.long)
    with torch.no_grad():
        u_emb = model.get_user_embeddings(uid_tensor).numpy().astype(np.float32)
    faiss.normalize_L2(u_emb)

    scores, indices = index.search(u_emb, top_k)   # (1, K)
    indices = indices[0]   # +1 because item IDs start at 1 (padding_idx=0)
    scores  = scores[0]

    lines = [f"**Top-{top_k} recommendations for User {user_id}**\n"]
    for rank, (idx, score) in enumerate(zip(indices, scores), 1):
        item_id = int(idx) + 1
        title   = ITEM_TITLES.get(item_id, f"Item {item_id}")
        lines.append(f"{rank:>2}. {title}  *(score: {score:.3f})*")

    return "\n".join(lines)


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="Two-Tower Movie Recommender") as demo:
    gr.Markdown(
        "# 🎬 Two-Tower Movie Recommender\n"
        "Retrieval model trained on MovieLens 1M. "
        "Enter a user ID (1 – 6040) to get personalised recommendations."
    )
    with gr.Row():
        user_input = gr.Number(label="User ID", value=1, minimum=1, maximum=6040, precision=0)
        topk_input = gr.Slider(label="Top K", minimum=5, maximum=50, step=5, value=10)
    btn    = gr.Button("Get Recommendations", variant="primary")
    output = gr.Markdown()
    btn.click(fn=recommend, inputs=[user_input, topk_input], outputs=output)

    gr.Examples(
        examples=[[1, 10], [42, 20], [999, 15]],
        inputs=[user_input, topk_input],
    )

if __name__ == "__main__":
    demo.launch()
