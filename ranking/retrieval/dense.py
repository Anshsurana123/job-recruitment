"""
Dense semantic retrieval using local SentenceTransformer bi-encoders.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np

from ..config import RetrievalConfig
from ..models import CandidateRecord, ProcessedCandidateFields


class DenseRetriever:
    """
    Manages dense candidate representations, query encoding, and semantic similarity scoring.
    """
    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or RetrievalConfig()
        self.model = None
        self._loaded_model_path = None
        self.id_to_embedding: Dict[str, np.ndarray] = {}

    def _ensure_model(self):
        if self.model is None:
            model_path = Path(self.config.dense_model_path)
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Dense model cache not found at {model_path}. "
                    "Ensure models are downloaded in model_cache/."
                )
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(str(model_path))
            self._loaded_model_path = str(model_path)

    def load_cached_embeddings(self, cache_path: Union[str, Path]) -> int:
        """Loads precomputed candidate embeddings from a pickle file."""
        path = Path(cache_path)
        if not path.exists():
            return 0
        try:
            with open(path, "rb") as f:
                loaded = pickle.load(f)
            if isinstance(loaded, dict):
                self.id_to_embedding.update(loaded)
                return len(loaded)
        except Exception:
            pass
        return 0

    def encode_query(self, query_text: str) -> np.ndarray:
        """Encodes query using BGE instruction prefix."""
        self._ensure_model()
        bge_query = "Represent this sentence for searching relevant passages: " + query_text
        return self.model.encode(
            bge_query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    def encode_candidates(
        self,
        candidate_fields: List[ProcessedCandidateFields],
        batch_size: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """Encodes candidate records that are missing from the cache."""
        bs = batch_size or self.config.dense_batch_size
        uncached = [f for f in candidate_fields if f.candidate_id not in self.id_to_embedding]
        
        if uncached:
            self._ensure_model()
            texts = [f.full_text for f in uncached]
            embeddings = self.model.encode(
                texts,
                batch_size=bs,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            for idx, f in enumerate(uncached):
                self.id_to_embedding[f.candidate_id] = embeddings[idx].astype(np.float32)
                
        return {f.candidate_id: self.id_to_embedding[f.candidate_id] for f in candidate_fields}

    def score_candidates(
        self,
        query_text: str,
        candidate_fields: List[ProcessedCandidateFields]
    ) -> Dict[str, float]:
        """
        Encodes query, computes cosine similarity against all candidates, and returns a score mapping.
        """
        if not candidate_fields:
            return {}
            
        self.encode_candidates(candidate_fields)
        query_vec = self.encode_query(query_text)
        
        doc_ids = [f.candidate_id for f in candidate_fields]
        mat = np.array([self.id_to_embedding[cid] for cid in doc_ids], dtype=np.float32)
        
        # Dot product with unit normalized embeddings gives cosine similarity
        sims = np.dot(mat, query_vec)
        return {doc_ids[i]: float(sims[i]) for i in range(len(doc_ids))}
