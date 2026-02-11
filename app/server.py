"""
FastAPI web server for microstructure image retrieval.

Endpoints:
    POST /api/search          - Upload image, get similar microstructures
    POST /api/search-text     - Text query (CLIP only), get matching microstructures
    GET  /api/image/{index}   - Serve a micrograph thumbnail by index
    GET  /api/image/{idx}/full - Serve full-resolution image
    GET  /api/health          - Health check
    GET  /api/stats           - Index statistics
    GET  /                    - Web UI

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
from pydantic import BaseModel
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.embed import get_embedder
from src.index import MicrostructureIndex

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Microstructure Classifier",
    description="Upload a micrograph to find similar microstructures and identify phases.",
    version="0.2.0",
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
    has_text_search = hasattr(embedder, "embed_text") if embedder else False
    return {
        "status": "ok",
        "model": config.EMBEDDING_MODEL,
        "index_loaded": index is not None,
        "index_size": index.size if index else 0,
        "text_search": has_text_search,
    }


@app.get("/api/stats")
async def stats():
    if not index:
        raise HTTPException(503, "Index not loaded")

    sources = {}
    labels = {}
    for meta in index.metadata:
        src = meta.get("source", "unknown")
        lbl = meta.get("label", "unknown")
        sources[src] = sources.get(src, 0) + 1
        labels[lbl] = labels.get(lbl, 0) + 1

    has_text_search = hasattr(embedder, "embed_text") if embedder else False

    return {
        "index_size": index.size,
        "embedding_dim": index.embedding_dim,
        "metric": index.metric,
        "model": config.EMBEDDING_MODEL,
        "text_search": has_text_search,
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
        results = index.search(query_embedding, top_k=top_k + 1)

        # Filter out exact self-match (distance ~0 for L2, ~1.0 for cosine)
        if config.SIMILARITY_METRIC == "cosine":
            results = [r for r in results if r["distance"] < 0.999][:top_k]
        else:
            results = [r for r in results if r["distance"] > 0.001][:top_k]

        # Encode query image as base64 thumbnail
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img.thumbnail((300, 300))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        query_thumb = base64.b64encode(buf.getvalue()).decode()

        return {
            "query": file.filename,
            "query_thumbnail": query_thumb,
            "query_type": "image",
            "top_k": top_k,
            "metric": config.SIMILARITY_METRIC,
            "results": results,
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)


class TextSearchRequest(BaseModel):
    query: str
    top_k: int = 8


@app.post("/api/search-text")
async def search_by_text(request: TextSearchRequest):
    """
    Search microstructures by text description (CLIP only).
    
    Examples:
        "pearlite with lamellar structure"
        "coarse martensite laths"
        "ferritic microstructure"
        "bainite with retained austenite"
    """
    if not index or not embedder:
        raise HTTPException(503, "Model or index not loaded")

    if not hasattr(embedder, "embed_text"):
        raise HTTPException(
            400,
            "Text search requires CLIP model. Rebuild index with: "
            "python scripts/build_index.py --model clip"
        )

    query_text = request.query.strip()
    if not query_text:
        raise HTTPException(400, "Empty query")

    top_k = max(1, min(request.top_k, 50))

    try:
        query_embedding = embedder.embed_text(query_text)
        results = index.search(query_embedding, top_k=top_k)

        return {
            "query": query_text,
            "query_type": "text",
            "top_k": top_k,
            "metric": config.SIMILARITY_METRIC,
            "results": results,
        }
    except Exception as e:
        raise HTTPException(500, f"Search failed: {str(e)}")


@app.get("/api/image/{idx}")
async def get_image(idx: int, size: int = Query(default=300, ge=50, le=1200)):
    """Serve a micrograph thumbnail by its index in the database."""
    if not index:
        raise HTTPException(503, "Index not loaded")

    if idx < 0 or idx >= len(index.metadata):
        raise HTTPException(404, f"Image index {idx} not found")

    meta = index.metadata[idx]
    image_path = Path(meta["path"])

    if not image_path.exists():
        raise HTTPException(404, f"Image file not found: {image_path}")

    img = Image.open(image_path).convert("RGB")
    img.thumbnail((size, size))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

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
