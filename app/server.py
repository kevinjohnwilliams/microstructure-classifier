"""
FastAPI web server for microstructure image retrieval.

Endpoints:
    POST /api/search         - Upload image, get similar microstructures
    GET  /api/image/{index}  - Serve a micrograph image by its index in the FAISS db
    GET  /api/health         - Health check
    GET  /api/stats          - Index statistics
    GET  /                   - Web UI

Usage:
    uvicorn app.server:app --reload --port 8000
"""
import io
import sys
import base64
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.embed import get_embedder
from src.index import MicrostructureIndex

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Microstructure Classifier",
    description="Upload a micrograph to find similar microstructures and identify phases.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load models on startup ───────────────────────────────────────────────────
embedder = None
index = None


@app.on_event("startup")
async def load_models():
    global embedder, index

    try:
        embedder = get_embedder()
        print(f"Loaded {config.EMBEDDING_MODEL} embedder")
    except Exception as e:
        print(f"Warning: Could not load embedder: {e}")

    try:
        index = MicrostructureIndex.load()
        print(f"Loaded index with {index.size} vectors")
    except FileNotFoundError:
        print("Warning: No index found. Run `python scripts/build_index.py` first.")


# ── API Endpoints ────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model": config.EMBEDDING_MODEL,
        "index_loaded": index is not None,
        "index_size": index.size if index else 0,
    }


@app.get("/api/stats")
async def stats():
    if not index:
        raise HTTPException(503, "Index not loaded")

    # Count by source and label
    sources = {}
    labels = {}
    for meta in index.metadata:
        src = meta.get("source", "unknown")
        lbl = meta.get("label", "unknown")
        sources[src] = sources.get(src, 0) + 1
        labels[lbl] = labels.get(lbl, 0) + 1

    return {
        "index_size": index.size,
        "embedding_dim": index.embedding_dim,
        "metric": index.metric,
        "model": config.EMBEDDING_MODEL,
        "sources": sources,
        "labels": labels,
    }


@app.post("/api/search")
async def search_by_image(
    file: UploadFile = File(...),
    top_k: int = Query(default=8, ge=1, le=50),
):
    """Upload a micrograph image and get the most similar microstructures."""
    if not index or not embedder:
        raise HTTPException(503, "Model or index not loaded")

    contents = await file.read()

    # Validate image
    try:
        img = Image.open(io.BytesIO(contents))
        img.verify()
    except Exception:
        raise HTTPException(400, "Invalid image file")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        query_embedding = embedder.embed_single(tmp_path)
        results = index.search(query_embedding, top_k=top_k + 1)  # +1 to skip self-match

        # Filter out exact self-match (distance ~0)
        results = [r for r in results if r["distance"] > 0.001][:top_k]

        # Encode query image as base64 thumbnail for response
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img.thumbnail((300, 300))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        query_thumb = base64.b64encode(buf.getvalue()).decode()

        return {
            "query": file.filename,
            "query_thumbnail": query_thumb,
            "top_k": top_k,
            "results": results,
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/api/image/{idx}")
async def get_image(idx: int, size: int = Query(default=300, ge=50, le=1200)):
    """Serve a micrograph image by its index in the database."""
    if not index:
        raise HTTPException(503, "Index not loaded")

    if idx < 0 or idx >= len(index.metadata):
        raise HTTPException(404, f"Image index {idx} not found")

    meta = index.metadata[idx]
    image_path = Path(meta["path"])

    if not image_path.exists():
        raise HTTPException(404, f"Image file not found: {image_path}")

    # Generate thumbnail for faster serving
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((size, size))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    # Encode as base64 and return in JSON (simpler than file serving for thumbnails)
    b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        "index": idx,
        "thumbnail": b64,
        "metadata": meta,
    }


@app.get("/api/image/{idx}/full")
async def get_full_image(idx: int):
    """Serve the full-resolution micrograph image."""
    if not index:
        raise HTTPException(503, "Index not loaded")

    if idx < 0 or idx >= len(index.metadata):
        raise HTTPException(404, f"Image index {idx} not found")

    meta = index.metadata[idx]
    image_path = Path(meta["path"])

    if not image_path.exists():
        raise HTTPException(404, f"Image file not found: {image_path}")

    return FileResponse(image_path)


# ── Web UI ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def web_ui():
    ui_path = Path(__file__).parent / "index.html"
    if ui_path.exists():
        return HTMLResponse(ui_path.read_text())
    return HTMLResponse("<h1>UI not found</h1><p>Place index.html in the app/ directory.</p>")
