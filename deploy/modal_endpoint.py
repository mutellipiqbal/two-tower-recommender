"""
deploy/modal_endpoint.py  —  Two-Tower  (Modal.com serverless)
================================================================
Serverless GPU endpoint — free $30/month credit on Modal.
Spins up a container on demand, serves recommendations via HTTP.

Setup:
  pip install modal
  modal token new        # authenticate
  modal deploy deploy/modal_endpoint.py

Then call via:
  curl https://YOUR_MODAL_URL/recommend?user_id=42&top_k=10

Modal free tier: ~$30 credit/month.
Each T4 GPU request costs ~$0.000225/sec.  A recommendation call takes ~0.1s
so you get ~130,000 free inference calls/month.
"""

import io
import json
import sys

import modal

# ── Modal app definition ───────────────────────────────────────────────────────
app = modal.App("two-tower-recommender")

# Docker image with required packages
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.11.0", "faiss-cpu==1.13.2",
        "huggingface_hub", "numpy==2.2.5",
    )
)

HF_REPO = "YOUR_HF_USERNAME/two-tower-movie-recommender"  # ← edit this


# ── Model class (loaded once per container, cached) ────────────────────────────
@app.cls(image=image, gpu="T4", container_idle_timeout=300)
class TwoTowerEndpoint:
    """
    Modal Cls — model loads once when the container starts,
    then serves many requests without re-loading.
    """

    @modal.enter()
    def load_model(self):
        import json
        import numpy as np
        import faiss
        import torch
        from huggingface_hub import hf_hub_download

        # Import src.model bundled in the container (add to Modal mounts if needed)
        sys.path.insert(0, "/root")
        from src.model import TwoTowerModel

        meta      = json.loads(open(hf_hub_download(HF_REPO, "meta.json")).read())
        model_pt  = hf_hub_download(HF_REPO, "best_two_tower.pt")
        index_bin = hf_hub_download(HF_REPO, "item_index.faiss")

        self.meta = meta
        self.model = TwoTowerModel(
            num_users    = meta["num_users"],
            num_items    = meta["num_items"],
            embedding_dim= meta["embedding_dim"],
            hidden_dims  = meta["hidden_dims"],
            dropout      = meta["dropout"],
        )
        self.model.load_state_dict(
            torch.load(model_pt, map_location="cuda", weights_only=True)
        )
        self.model.eval().cuda()
        self.index = faiss.read_index(index_bin)

        # Move index to GPU for faster search (optional, useful at scale)
        res         = faiss.StandardGpuResources()
        self.index  = faiss.index_cpu_to_gpu(res, 0, self.index)
        print(f"Ready: {self.index.ntotal:,} items indexed on GPU.")

    @modal.web_endpoint(method="GET")
    def recommend(self, user_id: int = 1, top_k: int = 10) -> dict:
        import numpy as np
        import faiss
        import torch

        if not (1 <= user_id < self.meta["num_users"]):
            return {"error": f"user_id must be 1–{self.meta['num_users'] - 1}"}
        if not (1 <= top_k <= 100):
            return {"error": "top_k must be 1–100"}

        uid_t = torch.tensor([user_id], dtype=torch.long, device="cuda")
        with torch.no_grad():
            u_emb = self.model.get_user_embeddings(uid_t).cpu().numpy().astype(np.float32)
        faiss.normalize_L2(u_emb)

        scores, indices = self.index.search(u_emb, top_k)
        item_ids = (indices[0] + 1).tolist()   # shift from 0-based to 1-based
        item_scores = scores[0].tolist()

        return {
            "user_id":    user_id,
            "top_k":      top_k,
            "item_ids":   item_ids,
            "scores":     [round(s, 4) for s in item_scores],
        }
