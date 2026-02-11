"""
Image preprocessing and visualization utilities for micrographs.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def get_transform(augment: bool = False) -> transforms.Compose:
    """
    Get the image transform pipeline.
    
    Args:
        augment: If True, add data augmentation (for training only).
    """
    steps = []
    
    if augment:
        steps.extend([
            transforms.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
        ])
    else:
        steps.extend([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        ])
    
    steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD),
    ])
    
    return transforms.Compose(steps)


def load_image(path: str | Path) -> Image.Image:
    """
    Load an image and convert to RGB.
    Handles grayscale micrographs by converting to 3-channel.
    """
    img = Image.open(path)
    
    # Many micrographs are grayscale — convert to RGB for model compatibility
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    return img


def load_and_preprocess(path: str | Path, augment: bool = False) -> torch.Tensor:
    """Load an image and apply preprocessing transforms."""
    img = load_image(path)
    transform = get_transform(augment=augment)
    return transform(img).unsqueeze(0)  # Add batch dimension


def collect_image_paths(directory: str | Path, extensions: set = None) -> list[Path]:
    """
    Recursively collect all image files from a directory.
    
    Args:
        directory: Root directory to search.
        extensions: Set of valid extensions. Defaults to common image formats.
    """
    if extensions is None:
        extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    
    directory = Path(directory)
    paths = []
    
    for ext in extensions:
        paths.extend(directory.rglob(f"*{ext}"))
        paths.extend(directory.rglob(f"*{ext.upper()}"))
    
    # Deduplicate and sort for reproducibility
    paths = sorted(set(paths))
    return paths


def display_results(query_path: str, results: list[dict], cols: int = 3):
    """
    Display query image alongside top-k retrieval results.
    Requires matplotlib (optional import for CLI use).
    """
    import matplotlib.pyplot as plt
    
    n = len(results) + 1
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.array(axes).flatten()
    
    # Show query image
    query_img = load_image(query_path)
    axes[0].imshow(query_img, cmap="gray")
    axes[0].set_title("QUERY", fontweight="bold", fontsize=14)
    axes[0].axis("off")
    
    # Show results
    for i, result in enumerate(results):
        img = load_image(result["path"])
        axes[i + 1].imshow(img, cmap="gray")
        
        title = f"#{i+1} | dist: {result['distance']:.3f}"
        if "label" in result:
            title += f"\n{result['label']}"
        
        axes[i + 1].set_title(title, fontsize=10)
        axes[i + 1].axis("off")
    
    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].axis("off")
    
    plt.tight_layout()
    plt.savefig("query_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Results saved to query_results.png")
