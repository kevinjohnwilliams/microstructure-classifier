"""
Image embedding generation using pre-trained vision models.

Supports:
- ResNet50 (default): Fast, good features, ImageNet pre-trained
- CLIP: Better semantic understanding, heavier model, enables text search
  and zero-shot phase classification

The embedding is extracted from the penultimate layer (before classification head),
giving a rich feature vector that captures visual patterns in the micrograph.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.utils import load_and_preprocess


class ResNetEmbedder:
    """Extract feature embeddings from ResNet50 (penultimate layer)."""
    
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pre-trained ResNet50
        weights = getattr(models, f"ResNet50_Weights", None)
        if weights:
            model = models.resnet50(weights=weights.DEFAULT)
        else:
            model = models.resnet50(pretrained=True)
        
        # Remove the final classification layer to get embeddings
        # ResNet50 avgpool output = 2048-dim vector
        self.model = nn.Sequential(*list(model.children())[:-1])
        self.model.to(self.device)
        self.model.eval()
        
        self.embedding_dim = config.EMBEDDING_DIM  # 2048
    
    @torch.no_grad()
    def embed_single(self, image_path: str | Path) -> np.ndarray:
        """Generate embedding for a single image."""
        tensor = load_and_preprocess(image_path).to(self.device)
        embedding = self.model(tensor)
        return embedding.squeeze().cpu().numpy()
    
    @torch.no_grad()
    def embed_batch(self, image_paths: list[Path], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a batch of images.
        
        Returns: np.ndarray of shape (N, embedding_dim)
        """
        all_embeddings = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Embedding"):
            batch_paths = image_paths[i : i + batch_size]
            tensors = []
            
            for path in batch_paths:
                try:
                    t = load_and_preprocess(path)
                    tensors.append(t)
                except Exception as e:
                    print(f"Warning: Failed to load {path}: {e}")
                    tensors.append(torch.zeros(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE))
            
            batch_tensor = torch.cat(tensors, dim=0).to(self.device)
            embeddings = self.model(batch_tensor)
            embeddings = embeddings.squeeze(-1).squeeze(-1)  # Remove spatial dims
            all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)


class CLIPEmbedder:
    """Extract feature embeddings using OpenCLIP."""
    
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            import open_clip
        except ImportError:
            raise ImportError(
                "open-clip-torch is required for CLIP embeddings. "
                "Install with: pip install open-clip-torch"
            )
        
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            config.CLIP_MODEL_NAME,
            pretrained=config.CLIP_PRETRAINED,
            device=self.device,
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(config.CLIP_MODEL_NAME)
        
        # ViT-B-32 gives 512-dim embeddings
        self.embedding_dim = 512

        # Cache for zero-shot text prototypes (computed once, reused)
        self._zeroshot_cache = {}
    
    @torch.no_grad()
    def embed_single(self, image_path: str | Path) -> np.ndarray:
        """Generate CLIP image embedding for a single image."""
        from src.utils import load_image
        
        img = load_image(image_path)
        tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        embedding = self.model.encode_image(tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.squeeze().cpu().numpy()

    @torch.no_grad()
    def embed_image_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Generate CLIP image embedding directly from raw bytes (no temp file)."""
        from PIL import Image as PILImage
        import io

        img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        embedding = self.model.encode_image(tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.squeeze().cpu().numpy()
    
    @torch.no_grad()
    def embed_batch(self, image_paths: list[Path], batch_size: int = 32) -> np.ndarray:
        """Generate CLIP embeddings for a batch of images."""
        from src.utils import load_image
        
        all_embeddings = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="CLIP Embedding"):
            batch_paths = image_paths[i : i + batch_size]
            tensors = []
            
            for path in batch_paths:
                try:
                    img = load_image(path)
                    t = self.preprocess(img).unsqueeze(0)
                    tensors.append(t)
                except Exception as e:
                    print(f"Warning: Failed to load {path}: {e}")
                    tensors.append(torch.zeros(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE))
            
            batch_tensor = torch.cat(tensors, dim=0).to(self.device)
            embeddings = self.model.encode_image(batch_tensor)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate CLIP text embedding. Useful for searching by description, e.g.:
        "ferritic microstructure with pearlite colonies"
        """
        tokens = self.tokenizer([text]).to(self.device)
        embedding = self.model.encode_text(tokens)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.squeeze().cpu().numpy()

    @torch.no_grad()
    def classify_zero_shot(
        self,
        image_embedding: np.ndarray,
        phase_prompts: dict[str, list[str]],
    ) -> dict[str, float]:
        """
        Zero-shot phase classification using CLIP text-image similarity.

        For each phase, we encode multiple text descriptions (prompts) and
        average their embeddings to create a robust text prototype. Then we
        compute cosine similarity between the image embedding and each
        phase prototype, and convert to probabilities via softmax.

        The prompts are cached after first computation so subsequent calls
        are fast (just a matrix multiply).

        Args:
            image_embedding: L2-normalized image embedding (512-dim)
            phase_prompts: Dict mapping phase key -> list of text descriptions
                e.g. {"pearlite": ["a micrograph showing pearlite ...", ...]}

        Returns:
            Dict mapping phase key -> probability (softmax-normalized)
        """
        cache_key = tuple(sorted(phase_prompts.keys()))

        if cache_key not in self._zeroshot_cache:
            prototypes = {}
            for phase_key, prompts in phase_prompts.items():
                tokens = self.tokenizer(prompts).to(self.device)
                text_embeddings = self.model.encode_text(tokens)
                text_embeddings = text_embeddings / text_embeddings.norm(
                    dim=-1, keepdim=True
                )
                # Average all prompt embeddings into one prototype per phase
                prototype = text_embeddings.mean(dim=0)
                prototype = prototype / prototype.norm()
                prototypes[phase_key] = prototype

            phase_keys = list(prototypes.keys())
            proto_matrix = torch.stack([prototypes[k] for k in phase_keys])
            self._zeroshot_cache[cache_key] = (phase_keys, proto_matrix)

        phase_keys, proto_matrix = self._zeroshot_cache[cache_key]

        # Cosine similarity: image embedding dot each text prototype
        img_tensor = torch.from_numpy(image_embedding).float().to(self.device)
        img_tensor = img_tensor / img_tensor.norm()
        similarities = proto_matrix @ img_tensor

        # Temperature-scaled softmax
        # Lower temperature = more peaked distribution (more confident)
        # Higher temperature = more spread out
        # CLIP's own logit scale is ~100, but that makes probs too peaked
        # 20 gives reasonable spread for our number of classes
        temperature = 20.0
        probs = torch.softmax(similarities * temperature, dim=0)

        return {
            phase_keys[i]: float(probs[i])
            for i in range(len(phase_keys))
        }


def get_embedder(model_name: str = None):
    """Factory function to get the configured embedder."""
    model_name = model_name or config.EMBEDDING_MODEL
    
    if model_name == "resnet50":
        return ResNetEmbedder()
    elif model_name == "clip":
        return CLIPEmbedder()
    else:
        raise ValueError(f"Unknown model: {model_name}. Use 'resnet50' or 'clip'.")