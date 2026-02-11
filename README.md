# Microstructure Classifier

A proof-of-concept system for **microstructure image retrieval and phase classification** in metals (primarily steel).

Upload a micrograph → get back the closest matching microstructures from a database and predicted phase classifications.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Upload      │────▶│  Embedding   │────▶│  Vector Search   │
│  Micrograph  │     │  (ResNet/CLIP)│     │  (FAISS)         │
└─────────────┘     └──────┬───────┘     └────────┬────────┘
                           │                       │
                           ▼                       ▼
                    ┌──────────────┐     ┌─────────────────┐
                    │  Phase        │     │  Top-K Matches   │
                    │  Classifier   │     │  + Metadata      │
                    └──────────────┘     └─────────────────┘
```

## Project Structure

```
microstructure-classifier/
├── README.md
├── requirements.txt
├── config.py                  # Paths, model settings, hyperparams
├── data/
│   ├── raw/                   # Drop downloaded datasets here
│   │   ├── uhcs/              # UHCS micrographs + microstructures.sqlite
│   │   └── aachen/            # Aachen-Heerlen SEM images + annotations
│   ├── processed/             # Resized/normalized images
│   └── embeddings/            # Pre-computed embedding vectors
├── src/
│   ├── __init__.py
│   ├── ingest.py              # Load datasets, normalize, prepare metadata
│   ├── embed.py               # Generate image embeddings (ResNet/CLIP)
│   ├── index.py               # Build and query FAISS vector index
│   ├── classify.py            # Phase classification model (Phase 2)
│   └── utils.py               # Image preprocessing, visualization helpers
├── scripts/
│   ├── build_index.py         # One-shot: embed all images, build FAISS index
│   ├── query.py               # CLI: upload image, get matches
│   └── evaluate.py            # Evaluate retrieval quality
├── app/
│   └── server.py              # FastAPI web server (Phase 3)
├── notebooks/
│   └── exploration.ipynb      # Dataset exploration and prototyping
└── tests/
    └── test_pipeline.py
```

## Quick Start

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Add data

Download the UHCS dataset from [NIST](https://materialsdata.nist.gov/handle/11256/940):
- Place micrograph images in `data/raw/uhcs/micrographs/`
- Place `microstructures.sqlite` in `data/raw/uhcs/`

### 3. Build the embedding index

```bash
python scripts/build_index.py
```

This will:
- Load and preprocess all micrographs
- Generate embeddings using the configured model
- Build a FAISS index and save it to `data/embeddings/`

### 4. Query with a new image

```bash
python scripts/query.py --image path/to/your/micrograph.png --top-k 5
```

### 5. Run the web server (Phase 3)

```bash
uvicorn app.server:app --reload
```

## Datasets

| Dataset | Images | Type | Labels | Source |
|---------|--------|------|--------|--------|
| UHCS (CMU/NIST) | ~600 | Optical | Processing metadata | [NIST](https://materialsdata.nist.gov/handle/11256/940) |
| Aachen-Heerlen | 1,705 | SEM | MA island polygons | [Nature](https://www.nature.com/articles/s41597-021-00926-7) |
| ASM Micrograph DB | Thousands | Mixed | Material + phase | Subscription required |

## Roadmap

- [x] Project structure
- [ ] Phase 1: Image similarity search (embedding + FAISS)
- [ ] Phase 2: Phase classification (fine-tuned CNN)
- [ ] Phase 3: Web interface
- [ ] Phase 4: ASM database integration (if access available)
