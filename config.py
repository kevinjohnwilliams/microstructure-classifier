"""
Central configuration for the microstructure classifier.
Adjust paths and model settings here.
"""
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# Dataset-specific paths
UHCS_DIR = RAW_DIR / "uhcs"
UHCS_MICROGRAPHS = UHCS_DIR / "micrographs"
UHCS_SQLITE = UHCS_DIR / "microstructures.sqlite"

AACHEN_DIR = RAW_DIR / "aachen"

# Index files
FAISS_INDEX_PATH = EMBEDDINGS_DIR / "microstructure.index"
METADATA_PATH = EMBEDDINGS_DIR / "metadata.pkl"

# ── Model Settings ───────────────────────────────────────────────────────────
# Options: "resnet50", "clip"
EMBEDDING_MODEL = "clip"

# ResNet settings
RESNET_WEIGHTS = "IMAGENET1K_V2"
EMBEDDING_DIM = 2048

# CLIP settings
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"

# ── Image Preprocessing ─────────────────────────────────────────────────────
IMAGE_SIZE = 224
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# ── Retrieval Settings ───────────────────────────────────────────────────────
DEFAULT_TOP_K = 5
SIMILARITY_METRIC = "cosine"      # "L2" or "cosine" (use cosine for CLIP)

# ── Phase Classification (Phase 2) ──────────────────────────────────────────
PHASE_CLASSES = [
    "ferrite",
    "pearlite",
    "martensite",
    "bainite",
    "austenite",
    "cementite",
]
NUM_CLASSES = len(PHASE_CLASSES)

# ── Ensure directories exist ─────────────────────────────────────────────────
for d in [RAW_DIR, PROCESSED_DIR, EMBEDDINGS_DIR, UHCS_DIR, AACHEN_DIR]:
    d.mkdir(parents=True, exist_ok=True)
