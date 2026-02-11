#!/usr/bin/env python3
"""
Query the microstructure index with a new image.

Usage:
    python scripts/query.py --image path/to/micrograph.png
    python scripts/query.py --image path/to/micrograph.png --top-k 10
    python scripts/query.py --image path/to/micrograph.png --model clip
    python scripts/query.py --text "ferritic microstructure with pearlite"  # CLIP only
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.embed import get_embedder
from src.index import MicrostructureIndex


def main():
    parser = argparse.ArgumentParser(description="Query microstructure similarity index")
    parser.add_argument("--image", type=Path, help="Path to query micrograph")
    parser.add_argument(
        "--text",
        type=str,
        help="Text description to search (CLIP model only)",
    )
    parser.add_argument(
        "--top-k", type=int, default=config.DEFAULT_TOP_K, help="Number of results"
    )
    parser.add_argument(
        "--model",
        choices=["resnet50", "clip"],
        default=config.EMBEDDING_MODEL,
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Display results with matplotlib"
    )
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()
    
    if not args.image and not args.text:
        parser.error("Provide either --image or --text")
    
    if args.text and args.model != "clip":
        parser.error("Text search requires --model clip")
    
    # ── Load index ───────────────────────────────────────────────────────
    index = MicrostructureIndex.load()
    
    # ── Generate query embedding ─────────────────────────────────────────
    embedder = get_embedder(args.model)
    
    if args.image:
        if not args.image.exists():
            print(f"Error: Image not found: {args.image}")
            sys.exit(1)
        
        print(f"Query image: {args.image}")
        query_embedding = embedder.embed_single(args.image)
    else:
        print(f"Query text: {args.text}")
        query_embedding = embedder.embed_text(args.text)
    
    # ── Search ───────────────────────────────────────────────────────────
    results = index.search(query_embedding, top_k=args.top_k)
    
    # ── Display results ──────────────────────────────────────────────────
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"\nTop {len(results)} matches:")
        print("-" * 70)
        for r in results:
            print(
                f"  #{r['rank']:2d} | "
                f"dist: {r['distance']:.4f} | "
                f"source: {r.get('source', '?'):10s} | "
                f"{r.get('filename', 'unknown')}"
            )
            if r.get("label"):
                print(f"       label: {r['label']}")
            if r.get("primary_microconstituent"):
                print(f"       phase: {r['primary_microconstituent']}")
        print("-" * 70)
    
    if args.visualize and args.image:
        from src.utils import display_results
        display_results(str(args.image), results)


if __name__ == "__main__":
    main()
