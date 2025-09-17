import faiss
import numpy as np
import os, json
from src.config import IMAGE_DIR, INDEX_PATH, MAPPING_PATH
from typing import List

class ImageIndexer:
    def __init__(self, embedder, dim=512):
        self.embedder = embedder
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim) 
        self.id_to_path = []

    def build_index(self, image_folder=IMAGE_DIR, rebuild=False):
        if os.path.exists(INDEX_PATH) and os.path.exists(MAPPING_PATH) and not rebuild:
            self.load_index()
            return

        paths = []
        for fname in sorted(os.listdir(image_folder)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(image_folder, fname))
        if not paths:
            raise ValueError("No images found in " + image_folder)

        # compute embeddings (batch)
        embs = self.embedder.embed_images_batch(paths)
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embs)
        self.id_to_path = paths
        self.save_index()

    def add_image(self, image_path: str):
        emb = self.embedder.embed_image(image_path)
        self.index.add(emb)
        self.id_to_path.append(image_path)
        self.save_index()

    def search(self, query: str, top_k=5):
        q_emb = self.embedder.embed_text(query)
        D, I = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            path = self.id_to_path[idx]
            results.append({"path": path, "score": float(score)})
        return results

    def save_index(self, index_path=INDEX_PATH, mapping_path=MAPPING_PATH):
        faiss.write_index(self.index, index_path)
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(self.id_to_path, f)

    def load_index(self, index_path=INDEX_PATH, mapping_path=MAPPING_PATH):
        self.index = faiss.read_index(index_path)
        with open(mapping_path, "r", encoding="utf-8") as f:
            self.id_to_path = json.load(f)
