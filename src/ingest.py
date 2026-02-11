"""
Dataset ingestion: load micrograph images and metadata from supported datasets.

Supports:
    - Aachen-Heerlen SEM images (PNG/TIFF from Figshare)
    - Kaggle steel microstructure dataset
    - UHCS dataset (when NIST comes back online)
    - Any generic folder of images
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.utils import collect_image_paths


def load_aachen_images(aachen_dir: Path = None) -> list[dict]:
    """
    Load Aachen-Heerlen SEM micrographs.

    Expected folder structure (flexible — handles multiple layouts):
        data/raw/aachen/
            TIFF/               # Raw SEM TIFF files
            excluded_PNG/       # Excluded images (suboptimal quality)
            PNG/                # Main annotated images (preferred)
    
    We prefer PNGs over TIFFs (identical content, faster to process).
    Excluded images are loaded but flagged.
    """
    aachen_dir = aachen_dir or config.AACHEN_DIR

    if not aachen_dir.exists():
        print(f"Aachen directory not found: {aachen_dir}")
        return []

    records = []

    # Scan all subdirectories and categorize
    for subdir in sorted(aachen_dir.iterdir()):
        if not subdir.is_dir():
            continue

        images = collect_image_paths(subdir)
        if not images:
            continue

        # Determine if these are excluded images
        dir_name = subdir.name.lower()
        is_excluded = "excluded" in dir_name

        # Determine image type
        is_tiff = "tiff" in dir_name or "tif" in dir_name

        print(f"  {subdir.name}: {len(images)} images "
              f"{'(excluded)' if is_excluded else ''}"
              f"{'(TIFF)' if is_tiff else '(PNG)'}")

        for path in images:
            records.append({
                "path": path,
                "filename": path.name,
                "stem": path.stem,
                "source": "aachen",
                "label": "bainite_ma",
                "is_excluded": is_excluded,
                "image_format": "tiff" if is_tiff else "png",
                "metadata": {
                    "dataset": "Aachen-Heerlen",
                    "imaging": "SEM",
                    "material": "bainitic steel",
                    "feature": "martensite-austenite islands",
                    "subfolder": subdir.name,
                },
            })

    # If we have both PNG and TIFF of the same images, prefer PNG
    png_stems = {r["stem"] for r in records if r["image_format"] == "png"}
    if png_stems:
        before = len(records)
        records = [
            r for r in records
            if r["image_format"] == "png" or r["stem"] not in png_stems
        ]
        dropped = before - len(records)
        if dropped:
            print(f"  Dropped {dropped} duplicate TIFFs (PNG versions exist)")

    print(f"  Aachen total: {len(records)} images")
    return records


def load_kaggle_images(kaggle_dir: Path = None) -> list[dict]:
    """
    Load Kaggle steel microstructure dataset.

    Handles both labeled subfolders and flat image directories.
    """
    kaggle_dir = kaggle_dir or (config.RAW_DIR / "kaggle")

    if not kaggle_dir.exists():
        print(f"Kaggle directory not found: {kaggle_dir}")
        return []

    records = []

    # Check if organized in subfolders (labeled) or flat
    subdirs = [d for d in kaggle_dir.iterdir() if d.is_dir()]

    if subdirs:
        for subdir in sorted(subdirs):
            images = collect_image_paths(subdir)
            label = subdir.name.lower().replace("_", " ").strip()

            if images:
                print(f"  {subdir.name}: {len(images)} images (label: {label})")

            for path in images:
                records.append({
                    "path": path,
                    "filename": path.name,
                    "stem": path.stem,
                    "source": "kaggle",
                    "label": label,
                    "is_excluded": False,
                    "metadata": {
                        "dataset": "Kaggle",
                        "class_folder": subdir.name,
                    },
                })

    # Also grab any images directly in the kaggle folder
    direct_images = [
        p for p in collect_image_paths(kaggle_dir)
        if p.parent == kaggle_dir
    ]

    if direct_images:
        print(f"  kaggle root: {len(direct_images)} images (unlabeled)")

    for path in direct_images:
        records.append({
            "path": path,
            "filename": path.name,
            "stem": path.stem,
            "source": "kaggle",
            "label": "unknown",
            "is_excluded": False,
            "metadata": {"dataset": "Kaggle"},
        })

    print(f"  Kaggle total: {len(records)} images")
    return records


def load_uhcs_images(uhcs_dir: Path = None) -> list[dict]:
    """
    Load UHCS micrographs and metadata (when NIST is back online).
    """
    uhcs_dir = uhcs_dir or config.UHCS_DIR
    micrographs_dir = uhcs_dir / "micrographs"

    if not micrographs_dir.exists():
        for alt in [uhcs_dir, uhcs_dir / "static" / "micrographs"]:
            if alt.exists() and any(collect_image_paths(alt)):
                micrographs_dir = alt
                break
        else:
            print(f"UHCS micrographs not found in {uhcs_dir}")
            return []

    images = collect_image_paths(micrographs_dir)
    if not images:
        return []

    # Try loading SQLite metadata
    meta_lookup = {}
    sqlite_path = uhcs_dir / "microstructures.sqlite"
    if sqlite_path.exists():
        try:
            from sqlalchemy import create_engine, text
            engine = create_engine(f"sqlite:///{sqlite_path}")
            with engine.connect() as conn:
                df = pd.read_sql(text("SELECT * FROM micrographs"), conn)
                for _, row in df.iterrows():
                    key = str(row.get("path", row.get("micrograph_id", "")))
                    stem = Path(key).stem if "/" in str(key) else key
                    meta_lookup[stem] = row.to_dict()
            print(f"  Loaded UHCS metadata: {len(meta_lookup)} entries")
        except Exception as e:
            print(f"  Warning: Could not load UHCS metadata: {e}")

    records = []
    for path in images:
        meta = meta_lookup.get(path.stem, {})
        records.append({
            "path": path,
            "filename": path.name,
            "stem": path.stem,
            "source": "uhcs",
            "label": str(meta.get("primary_microconstituent", "unknown")),
            "is_excluded": False,
            "metadata": {
                "dataset": "UHCS",
                "imaging": "optical",
                "material": "ultrahigh carbon steel",
                **{k: v for k, v in meta.items()
                   if isinstance(v, (str, int, float, bool))},
            },
        })

    print(f"  UHCS total: {len(records)} images")
    return records


def load_generic_directory(directory: str | Path, label: str = None) -> list[dict]:
    """
    Load images from any directory. Uses parent folder name as label
    if images are organized in subfolders.
    """
    directory = Path(directory)
    image_paths = collect_image_paths(directory)

    records = []
    for path in image_paths:
        folder_label = path.parent.name if path.parent != directory else "unknown"
        records.append({
            "path": path,
            "filename": path.name,
            "stem": path.stem,
            "source": directory.name,
            "label": label or folder_label,
            "is_excluded": False,
            "metadata": {"source_dir": str(directory)},
        })

    print(f"  {directory.name}: {len(records)} images")
    return records


def load_all_datasets(include_excluded: bool = False) -> list[dict]:
    """
    Load all available datasets and combine into a unified list.

    Args:
        include_excluded: If True, include excluded/low-quality images.
                         For similarity search, True gives more data.
                         For classification training, use False.
    """
    all_records = []

    print("=" * 60)
    print("Loading datasets...")
    print("=" * 60)

    # Aachen-Heerlen
    print("\n[Aachen-Heerlen]")
    aachen = load_aachen_images()
    all_records.extend(aachen)

    # Kaggle
    print("\n[Kaggle]")
    kaggle = load_kaggle_images()
    all_records.extend(kaggle)

    # UHCS (if available)
    print("\n[UHCS]")
    uhcs = load_uhcs_images()
    all_records.extend(uhcs)

    # Any other directories in raw/
    known_dirs = {"aachen", "kaggle", "uhcs"}
    for subdir in sorted(config.RAW_DIR.iterdir()):
        if subdir.is_dir() and subdir.name.lower() not in known_dirs:
            print(f"\n[{subdir.name}]")
            extra = load_generic_directory(subdir)
            all_records.extend(extra)

    # Filter excluded if requested
    if not include_excluded:
        before = len(all_records)
        all_records = [r for r in all_records if not r.get("is_excluded", False)]
        excluded_count = before - len(all_records)
        if excluded_count:
            print(f"\nFiltered out {excluded_count} excluded images")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Total: {len(all_records)} images")

    sources = {}
    for r in all_records:
        src = r.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1
    for src, count in sorted(sources.items()):
        print(f"  {src}: {count}")

    labels = {}
    for r in all_records:
        lbl = r.get("label", "unknown")
        labels[lbl] = labels.get(lbl, 0) + 1
    if len(labels) <= 20:
        print("Labels:")
        for lbl, count in sorted(labels.items(), key=lambda x: -x[1]):
            print(f"  {lbl}: {count}")

    print("=" * 60)
    return all_records
