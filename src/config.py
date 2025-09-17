import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "Data")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
MODEL_NAME = os.environ.get("MODEL_NAME", "openai/clip-vit-base-patch32")
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.idx")
MAPPING_PATH = os.path.join(DATA_DIR, "id_to_path.json")
TOP_K_DEFAULT = 5
