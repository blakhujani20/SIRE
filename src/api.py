import os, time
from fastapi import FastAPI, Query, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import logging
from src.embedder import Embedder
from src.indexer import ImageIndexer
from src.config import IMAGE_DIR, TOP_K_DEFAULT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sir")

app = FastAPI(title="Semantic Image Retrieval (V0)")

if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)
app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")

embedder = Embedder()
indexer = ImageIndexer(embedder)
try:
    indexer.build_index()
    logger.info("Index built/loaded successfully.")
except Exception as e:
    logger.exception("Failed to build/load index: %s", e)


@app.get("/")
def root():
    return {"message": "Semantic Image Retrieval API is running. Use /search or /upload."}

@app.get("/search")
def search(query: str = Query(...), top_k: int = TOP_K_DEFAULT):
    t0 = time.time()
    results = indexer.search(query, top_k=top_k)
    logger.info("search q=%s top_k=%d took=%.3f", query, top_k, time.time()-t0)
    out = []
    for r in results:
        fname = os.path.basename(r["path"])
        url = f"http://127.0.0.1:8000/images/{fname}"
        out.append({"filename": fname, "score": r["score"], "url": url})
    return JSONResponse({"query": query, "results": out})

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    save_path = os.path.join(IMAGE_DIR, file.filename)
    with open(save_path, "wb") as f:
        f.write(await file.read())
    indexer.add_image(save_path)
    return {"status": "ok", "path": save_path}
