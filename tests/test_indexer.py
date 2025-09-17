import os
import numpy as np
from src.indexer import ImageIndexer

class DummyEmbedder:
    def __init__(self, dim=512):
        self.dim = dim
    def embed_images_batch(self, paths, batch_size=8):
        return np.random.rand(len(paths), self.dim).astype("float32")
    def embed_image(self, path):
        return np.random.rand(1, self.dim).astype("float32")
    def embed_text(self, text):
        return np.random.rand(1, self.dim).astype("float32")

def test_index_build_and_search(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(5):
        p = img_dir / f"img_{i}.jpg"
        p.write_text("dummy")
    embedder = DummyEmbedder()
    indexer = ImageIndexer(embedder, dim=512)
    indexer.build_index(image_folder=str(img_dir))
    results = indexer.search("a test", top_k=3)
    assert isinstance(results, list)
