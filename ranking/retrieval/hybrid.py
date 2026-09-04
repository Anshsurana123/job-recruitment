"""
Hybrid retrieval fusion combining lexical (BM25F) and semantic (Dense) representations.
"""

from typing import Dict, List, Optional, Tuple

from ..config import RetrievalConfig
from ..models import CandidateRecord, RetrievalMatch


class HybridFusion:
    """
    Fuses lexical and dense retrieval signals via min-max score normalization and weighted linear combination.
    """
    def __init__(self, config: Optional[RetrievalConfig] = None):
        cfg = config or RetrievalConfig()
        self.bm25_weight, self.dense_weight = cfg.hybrid_weights
        self.top_k = cfg.top_k_retrieval

    def fuse(
        self,
        records: List[CandidateRecord],
        bm25_scores: Dict[str, float],
        dense_scores: Dict[str, float],
        top_k: Optional[int] = None
    ) -> List[RetrievalMatch]:
        """
        Normalizes both score spaces to [0, 100] and fuses with configured weights.
        Returns top-K sorted RetrievalMatch instances.
        """
        k = top_k or self.top_k
        if not records:
            return []

        bm25_vals = [bm25_scores.get(r.candidate_id, 0.0) for r in records]
        dense_vals = [dense_scores.get(r.candidate_id, 0.0) for r in records]

        min_b, max_b = min(bm25_vals), max(bm25_vals)
        range_b = max_b - min_b if max_b > min_b else 1.0

        min_d, max_d = min(dense_vals), max(dense_vals)
        range_d = max_d - min_d if max_d > min_d else 1.0

        matches: List[RetrievalMatch] = []
        for r in records:
            cid = r.candidate_id
            raw_b = bm25_scores.get(cid, 0.0)
            raw_d = dense_scores.get(cid, 0.0)

            norm_b = 100.0 * (raw_b - min_b) / range_b
            norm_d = 100.0 * (raw_d - min_d) / range_d

            hybrid = self.bm25_weight * norm_b + self.dense_weight * norm_d
            matches.append(RetrievalMatch(
                candidate_id=cid,
                hybrid_score=round(hybrid, 4),
                bm25_score=raw_b,
                semantic_score=raw_d,
                record=r
            ))

        # Sort descending by hybrid_score, tie-break by candidate_id ascending
        matches.sort(key=lambda m: (-m.hybrid_score, m.candidate_id))
        return matches[:k]
