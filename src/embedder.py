import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from typing import List

class Embedder:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.dim = self.model.config.projection_dim

    def _normalize(self, vec: np.ndarray):
        norms = np.linalg.norm(vec, axis=1, keepdims=True) + 1e-10
        return vec / norms

    def embed_text(self, text: str) -> np.ndarray:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            emb = self.model.get_text_features(**inputs)
        emb = emb.cpu().numpy().astype("float32")
        return self._normalize(emb)

    def embed_image(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            emb = self.model.get_image_features(**inputs)
        emb = emb.cpu().numpy().astype("float32")
        return self._normalize(emb)

    def embed_images_batch(self, paths: List[str], batch_size=8) -> np.ndarray:
        all_embs = []
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i+batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                emb = self.model.get_image_features(**inputs)
            emb = emb.cpu().numpy().astype("float32")
            all_embs.append(emb)
        all_embs = np.concatenate(all_embs, axis=0)
        return self._normalize(all_embs)
