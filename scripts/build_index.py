#!/usr/bin/env python3
"""
Build the FAISS similarity index from all available microstructure datasets.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --model clip
    python scripts/build_index.py --data-dir /path/to/custom/images
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.ingest import load_all_datasets, load_generic_directory
from src.embed import get_embedder
from src.index import MicrostructureIndex


def main():
    parser = argparse.ArgumentParser(description="Build microstructure search index")
    parser.add_argument(
        "--model",
        choices=["resnet50", "clip"],
        default=config.EMBEDDING_MODEL,
        help="Embedding model to use",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Custom directory of images (overrides default dataset loading)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation",
    )
    parser.add_argument(
        "--include-excluded",
        action="store_true",
        help="Include excluded/low-quality images (more data for similarity search)",
    )
    args = parser.parse_args()
    
    # ── Load data ────────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Loading datasets")
    print("=" * 60)
    
    if args.data_dir:
        records = load_generic_directory(args.data_dir)
    else:
        records = load_all_datasets(include_excluded=args.include_excluded)
    
    if not records:
        print("\nERROR: No images found!")
        print("Please download datasets first. See README.md for instructions.")
        print(f"Expected data in: {config.RAW_DIR}")
        sys.exit(1)
    
    # ── Generate embeddings ──────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"STEP 2: Generating embeddings ({args.model})")
    print("=" * 60)
    
    embedder = get_embedder(args.model)
    image_paths = [r["path"] for r in records]
    
    embeddings = embedder.embed_batch(image_paths, batch_size=args.batch_size)
    print(f"Generated {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]}")
    
    # ── Build index ──────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("STEP 3: Building FAISS index")
    print("=" * 60)
    
    index = MicrostructureIndex(embedding_dim=embeddings.shape[1])
    
    # Prepare metadata (strip non-serializable items)
    metadata_list = []
    for r in records:
        meta = {
            "path": str(r["path"]),
            "filename": r["filename"],
            "source": r.get("source", "unknown"),
            "label": r.get("label", ""),
        }
        # Include any extra metadata from the dataset
        if "metadata" in r and isinstance(r["metadata"], dict):
            for k, v in r["metadata"].items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
        metadata_list.append(meta)
    
    index.add(embeddings, metadata_list)
    
    # ── Save ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("STEP 4: Saving index")
    print("=" * 60)
    
    index.save()
    
    print(f"\nDone! Index contains {index.size} microstructures.")
    print(f"Query with: python scripts/query.py --image <path>")


if __name__ == "__main__":
    main()
