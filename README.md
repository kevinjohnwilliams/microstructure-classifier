# Microstructure Classifier

A proof-of-concept system for microstructure image retrieval and phase classification in metals (primarily steel). Upload a micrograph and get back the closest matching microstructures from a database of 2,500+ images using CLIP embeddings and FAISS vector search.

**[Try the live demo](https://kevwill-microstructure-classifier.hf.space)** — no installation needed, works on desktop and mobile.

## How It Works

Images are passed through CLIP ViT-B/32 to extract 512-dimensional feature vectors that capture texture, morphology, and contrast patterns. These embeddings are indexed with FAISS for sub-millisecond nearest-neighbor retrieval using cosine similarity.

CLIP's shared embedding space also enables text-to-image search — describe a microstructure in plain English (e.g., "pearlite with lamellar structure") and find matching images.

**Features:**
- Image similarity search via drag-and-drop
- Text-based search using CLIP's multimodal embeddings
- Ranked results with distance metrics and full-resolution viewing
- Multi-dataset support (Aachen-Heerlen SEM, Kaggle alloy micrographs, UHCS optical)

**Phase classification** (ferrite, pearlite, martensite, bainite, austenite) is planned but not yet implemented.

## Datasets

| Dataset | Images | Imaging | Source |
| --- | --- | --- | --- |
| Aachen-Heerlen | 1,705 | SEM | [Figshare](https://figshare.com/collections/Aachen-Heerlen_Annotated_Steel_Microstructure_Dataset/5185004) |
| Kaggle (CPJ/HR/P92) | 837 | Optical | [Kaggle](https://www.kaggle.com) |
| UHCS (CMU/NIST) | ~600 | Optical | [NIST](https://materialsdata.nist.gov/handle/11256/940) |

## Quick Start

```bash
# Install
pip install torch torchvision faiss-cpu open-clip-torch fastapi uvicorn python-multipart Pillow tqdm

# Download datasets into data/raw/ (see Datasets table above)

# Build the embedding index (~90 seconds on CPU for ~2,500 images)
python scripts/build_index.py --model clip --include-excluded

# Query via CLI
python scripts/query.py --image path/to/micrograph.png --top-k 5
python scripts/query.py --text "ferritic microstructure with pearlite"

# Or launch the web UI
python -m uvicorn app.server:app --reload --port 8000
```

## API

| Method | Endpoint | Description |
| --- | --- | --- |
| `POST` | `/api/search?top_k=8` | Upload image, get similar microstructures |
| `POST` | `/api/search-text` | Text query, get matching microstructures |
| `GET` | `/api/image/{idx}` | Thumbnail of indexed image |
| `GET` | `/api/image/{idx}/full` | Full-resolution image |
| `GET` | `/api/stats` | Index statistics and metadata |

## Roadmap

- [x] Image similarity search (CLIP + FAISS)
- [x] Text-based search
- [x] Web interface with drag-and-drop
- [x] Hosted live demo on Hugging Face Spaces
- [ ] Phase classification (fine-tuned CNN)
- [ ] UHCS dataset integration
- [ ] ASM Micrograph Database integration

## References

- DeCost et al. (2017). [UHCSDB: UltraHigh Carbon Steel Micrograph DataBase](https://doi.org/10.1007/s40192-017-0098-z)
- Azimi et al. (2018). [Advanced Steel Microstructural Classification by Deep Learning Methods](https://doi.org/10.1038/s41598-018-20037-5)
- Iren et al. (2021). [Aachen-Heerlen Annotated Steel Microstructure Dataset](https://doi.org/10.1038/s41597-021-00926-7)
