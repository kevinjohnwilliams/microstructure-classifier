"""
FAISS-based vector index for microstructure similarity search.

Stores embeddings and supports fast nearest-neighbor retrieval.
Metadata (paths, labels, dataset source) is stored alongside the index.
"""
import sys
import pickle
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class MicrostructureIndex:
    """
    Vector index for microstructure image retrieval.
    
    Wraps FAISS with metadata storage so query results include
    image paths, labels, and other information.
    """
    
    def __init__(self, embedding_dim: int = None, metric: str = None):
        self.embedding_dim = embedding_dim or config.EMBEDDING_DIM
        self.metric = metric or config.SIMILARITY_METRIC
        
        # Build FAISS index
        if self.metric == "cosine":
            # Normalize vectors + use inner product = cosine similarity
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        else:
            # L2 (Euclidean) distance
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Parallel metadata store (ordered same as embeddings)
        self.metadata: list[dict] = []
    
    def add(self, embeddings: np.ndarray, metadata_list: list[dict]):
        """
        Add embeddings and their associated metadata to the index.
        
        Args:
            embeddings: np.ndarray of shape (N, embedding_dim), float32
            metadata_list: List of dicts with keys like 'path', 'label', 'source'
        """
        assert len(embeddings) == len(metadata_list), (
            f"Mismatch: {len(embeddings)} embeddings vs {len(metadata_list)} metadata"
        )
        
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        
        if self.metric == "cosine":
            # Normalize for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            embeddings = embeddings / norms
        
        self.index.add(embeddings)
        self.metadata.extend(metadata_list)
        
        print(f"Index now contains {self.index.ntotal} vectors")
    
    def search(self, query_embedding: np.ndarray, top_k: int = None) -> list[dict]:
        """
        Find the top-k most similar microstructures to the query.
        
        Args:
            query_embedding: np.ndarray of shape (embedding_dim,)
            top_k: Number of results to return.
        
        Returns:
            List of dicts with keys: 'path', 'distance', 'rank', + any metadata
        """
        top_k = top_k or config.DEFAULT_TOP_K
        top_k = min(top_k, self.index.ntotal)
        
        query = np.ascontiguousarray(
            query_embedding.reshape(1, -1), dtype=np.float32
        )
        
        if self.metric == "cosine":
            norms = np.linalg.norm(query, axis=1, keepdims=True)
            norms[norms == 0] = 1
            query = query / norms
        
        distances, indices = self.index.search(query, top_k)
        
        results = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            
            result = {
                "rank": rank + 1,
                "distance": float(dist),
                "index": int(idx),
            }
            # Merge metadata
            if idx < len(self.metadata):
                result.update(self.metadata[idx])
                # Convert Path to string for serialization
                if "path" in result and isinstance(result["path"], Path):
                    result["path"] = str(result["path"])
            
            results.append(result)
        
        return results
    
    def save(self, index_path: str | Path = None, metadata_path: str | Path = None):
        """Save the FAISS index and metadata to disk."""
        index_path = Path(index_path or config.FAISS_INDEX_PATH)
        metadata_path = Path(metadata_path or config.METADATA_PATH)
        
        index_path.parent.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.index, str(index_path))
        with open(metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
        
        print(f"Saved index ({self.index.ntotal} vectors) to {index_path}")
        print(f"Saved metadata to {metadata_path}")
    
    @classmethod
    def load(cls, index_path: str | Path = None, metadata_path: str | Path = None):
        """Load a previously saved index from disk."""
        index_path = Path(index_path or config.FAISS_INDEX_PATH)
        metadata_path = Path(metadata_path or config.METADATA_PATH)
        
        if not index_path.exists():
            raise FileNotFoundError(f"No index found at {index_path}")
        
        index = faiss.read_index(str(index_path))
        
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
        
        # Reconstruct the object
        obj = cls.__new__(cls)
        obj.index = index
        obj.metadata = metadata
        obj.embedding_dim = index.d
        obj.metric = config.SIMILARITY_METRIC
        
        print(f"Loaded index with {index.ntotal} vectors")
        return obj
    
    @property
    def size(self) -> int:
        return self.index.ntotal
