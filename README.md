# Microstructure Classifier

[![Live Demo](https://img.shields.io/badge/🔬_Live_Demo-Try_It_Now-blue?style=for-the-badge)](https://kevwill-microstructure-classifier.hf.space)
[![Python](https://img.shields.io/badge/Python-3.11-green?style=flat-square&logo=python)](https://python.org)
[![CLIP](https://img.shields.io/badge/Model-CLIP_ViT--B--32-orange?style=flat-square)](https://openai.com/research/clip)
[![FAISS](https://img.shields.io/badge/Search-FAISS-purple?style=flat-square)](https://github.com/facebookresearch/faiss)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

A proof-of-concept system for **microstructure image retrieval and phase classification** in metals (primarily steel).

Upload a micrograph → get back the closest matching microstructures from a database of 2,500+ images using deep learning embeddings and vector search.

> **[🔬 Try the live demo →](https://kevwill-microstructure-classifier.hf.space)** — No installation needed. Works on desktop and mobile.

---

## What It Does

- **Image Similarity Search**: Drop in a micrograph and find the most visually similar microstructures from a database of 2,500+ images using CLIP embeddings and FAISS vector search.
- **Text-Based Search**: Describe a microstructure in plain English (e.g., *"pearlite with lamellar structure"*) and find matching images using CLIP's multimodal capabilities.
- **Web Interface**: Browser-based UI with drag-and-drop querying, ranked results, distance metrics, and full-resolution viewing.
- **Multi-Dataset Support**: Ingests and indexes images from multiple research datasets (Aachen-Heerlen SEM, Kaggle alloy micrographs, UHCS optical).
- **Phase Classification** *(planned)*: Fine-tuned CNN to identify ferrite, pearlite, martensite, bainite, and austenite phases.

---

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Upload      │────▶│  Embedding   │────▶│  Vector Search   │
│  Micrograph  │     │  (CLIP)      │     │  (FAISS)         │
└─────────────┘     └──────┬───────┘     └────────┬────────┘
                           │                       │
                           ▼                       ▼
                    ┌──────────────┐     ┌─────────────────┐
                    │  Phase        │     │  Top-K Matches   │
                    │  Classifier   │     │  + Metadata      │
                    └──────────────┘     └─────────────────┘
```

**Embedding Pipeline**: Images are passed through CLIP ViT-B-32 to extract 512-dim feature vectors. These embeddings capture high-level visual features — texture, morphology, contrast patterns — that correspond to microstructural similarity. CLIP also enables text-to-image search via its shared embedding space.

**Vector Search**: Embeddings are stored in a FAISS index for sub-millisecond nearest-neighbor retrieval using cosine similarity.

---

## Project Structure

```
microstructure-classifier/
├── README.md
├── requirements.txt
├── config.py                  # Paths, model settings, hyperparams
├── data/
│   ├── raw/                   # Downloaded datasets (not committed)
│   │   ├── aachen/            # Aachen-Heerlen SEM images
│   │   │   └── PNG/           # 1,705 annotated micrographs
│   │   ├── kaggle/            # Kaggle steel alloy datasets
│   │   │   ├── cpj_alloys/    # 430 images
│   │   │   ├── hr_alloys.../  # 311 images
│   │   │   └── p92_alloys/    # 96 images
│   │   └── uhcs/              # UHCS micrographs (when available)
│   ├── processed/             # Resized/normalized images
│   └── embeddings/            # FAISS index + metadata pickle
├── src/
│   ├── ingest.py              # Dataset loaders (Aachen, Kaggle, UHCS, generic)
│   ├── embed.py               # CLIP and ResNet50 embedding extractors
│   ├── index.py               # FAISS vector index wrapper
│   ├── classify.py            # Phase classification model (stub)
│   └── utils.py               # Image preprocessing, visualization helpers
├── scripts/
│   ├── build_index.py         # Embed all images and build FAISS index
│   └── query.py               # CLI similarity search with visualization
├── app/
│   ├── server.py              # FastAPI web server
│   └── index.html             # Web UI (single-page app)
├── notebooks/
└── tests/
```

---

## Quick Start

### 1. Set up environment

On Windows with long path issues (e.g., no admin rights), create a venv at a short path:

```
python -m venv C:\mcvenv
C:\mcvenv\Scripts\Activate.ps1
```

### 2. Install dependencies

```
pip install torch torchvision faiss-cpu scikit-image sqlalchemy fastapi python-multipart h5py open-clip-torch tqdm uvicorn Pillow pandas matplotlib
```

### 3. Add data

Download datasets and place them in `data/raw/`:

- **Aachen-Heerlen**: [Figshare Collection](https://figshare.com/collections/Aachen-Heerlen_Annotated_Steel_Microstructure_Dataset/5185004) → `data/raw/aachen/PNG/`
- **Kaggle**: Steel microstructure datasets → `data/raw/kaggle/`
- **UHCS** *(optional)*: [NIST](https://materialsdata.nist.gov/handle/11256/940) → `data/raw/uhcs/`

### 4. Build the embedding index

```
python scripts/build_index.py --model clip --include-excluded
```

This will load all images, generate CLIP embeddings, build a FAISS index, and save everything to `data/embeddings/`. Takes ~90 seconds on CPU for ~2,500 images.

### 5. Query via CLI

```
python scripts/query.py --image path/to/micrograph.png --top-k 5
python scripts/query.py --text "ferritic microstructure with pearlite"
python scripts/query.py --image path/to/micrograph.png --top-k 5 --visualize
```

### 6. Launch the web UI

```
python -m uvicorn app.server:app --reload --port 8000
```

Open [http://localhost:8000](http://localhost:8000) — drag and drop a micrograph to search.

Or just use the **[live demo](https://kevwill-microstructure-classifier.hf.space)** — no setup needed.

---

## Datasets

| Dataset | Images | Imaging | Labels | Source |
| --- | --- | --- | --- | --- |
| Aachen-Heerlen | 1,705 | SEM | MA island annotations | [Figshare](https://figshare.com/collections/Aachen-Heerlen_Annotated_Steel_Microstructure_Dataset/5185004) |
| Kaggle (CPJ/HR/P92) | 837 | Optical | Alloy type subfolders | [Kaggle](https://www.kaggle.com) |
| UHCS (CMU/NIST) | ~600 | Optical | Processing metadata + SQLite | [NIST](https://materialsdata.nist.gov/handle/11256/940) |

---

## API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `POST` | `/api/search?top_k=8` | Upload image, get similar microstructures |
| `POST` | `/api/search-text` | Text query (CLIP), get matching microstructures |
| `GET` | `/api/image/{idx}` | Get thumbnail of indexed image |
| `GET` | `/api/image/{idx}/full` | Get full-resolution image |
| `GET` | `/api/stats` | Index statistics and metadata counts |
| `GET` | `/api/health` | Health check |

---

## Roadmap

- [x] Project scaffolding and dataset ingestion
- [x] Image similarity search (ResNet50 embeddings + FAISS)
- [x] CLI query tool with visualization
- [x] Web interface (FastAPI + drag-and-drop UI)
- [x] CLIP model support (text-based search: "pearlite colonies")
- [x] Hosted live demo (Hugging Face Spaces)
- [ ] Phase classification (fine-tuned CNN for ferrite/pearlite/martensite/bainite/austenite)
- [ ] UHCS dataset integration (pending NIST availability)
- [ ] ASM Micrograph Database integration

---

## References

- DeCost et al. (2017). [UHCSDB: UltraHigh Carbon Steel Micrograph DataBase](https://doi.org/10.1007/s40192-017-0098-z)
- Azimi et al. (2018). [Advanced Steel Microstructural Classification by Deep Learning Methods](https://doi.org/10.1038/s41598-018-20037-5). 93.94% accuracy with FCNN.
- Iren et al. (2021). [Aachen-Heerlen Annotated Steel Microstructure Dataset](https://doi.org/10.1038/s41597-021-00926-7)

## License

MIT
