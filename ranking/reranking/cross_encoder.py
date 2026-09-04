"""
Cross-Encoder re-ranking stage for contextual fine-grained candidate ranking.
"""

from pathlib import Path
from typing import List, Optional

from ..config import RerankingConfig
from ..models import ScoredCandidate


class CrossEncoderReranker:
    """
    Re-ranks top candidates using a deep cross-encoder model cached locally.
    """
    def __init__(self, config: Optional[RerankingConfig] = None):
        self.config = config or RerankingConfig()
        self.model = None

    def _load_model(self) -> bool:
        if self.model is not None:
            return True
        model_path = Path(self.config.cross_encoder_path)
        if not model_path.exists():
            return False
        try:
            from sentence_transformers.cross_encoder import CrossEncoder
            self.model = CrossEncoder(str(model_path))
            return True
        except Exception:
            return False

    def rerank(
        self,
        scored_candidates: List[ScoredCandidate],
        query_text: str,
        top_k: Optional[int] = None
    ) -> List[ScoredCandidate]:
        if not self.config.enabled or not scored_candidates:
            return scored_candidates

        k = top_k or self.config.cross_encoder_top_k
        top_pool = scored_candidates[:k]
        remaining = scored_candidates[k:]

        if not self._load_model():
            # Graceful fallback: return without reranking if model missing
            return scored_candidates

        pairs = []
        for item in top_pool:
            cand = item.record
            p = cand.profile
            skills_str = ", ".join(s.get("name", "") for s in cand.skills[:20])
            
            recent_roles = []
            for job in cand.career_history[:3]:
                jt = job.get("title", "")
                comp = job.get("company", "")
                desc = job.get("description", "")
                r_parts = []
                if jt:
                    if comp: r_parts.append(f"{jt} at {comp}")
                    else: r_parts.append(jt)
                if desc:
                    r_parts.append(f"({desc[:120]}...)")
                if r_parts:
                    recent_roles.append(" ".join(r_parts))
            career_str = ". ".join(recent_roles)
            cand_text = f"{p.current_title}. {p.headline}. {p.summary}. Skills: {skills_str}. Recent: {career_str}."
            pairs.append((query_text, cand_text))

        ce_scores = self.model.predict(pairs, batch_size=self.config.batch_size, show_progress_bar=False)

        ce_min, ce_max = float(min(ce_scores)), float(max(ce_scores))
        ce_range = ce_max - ce_min if ce_max > ce_min else 1.0

        fit_scores = [c.final_score for c in top_pool]
        fit_min, fit_max = min(fit_scores), max(fit_scores)
        fit_range = fit_max - fit_min if fit_max > fit_min else 1.0

        alpha_fit = self.config.alpha_fit
        alpha_ce = self.config.alpha_ce

        for i, c in enumerate(top_pool):
            norm_fit = (c.final_score - fit_min) / fit_range if fit_range > 0 else 1.0
            norm_ce = (float(ce_scores[i]) - ce_min) / ce_range
            blended = alpha_fit * norm_fit + alpha_ce * norm_ce

            c.final_score = round(fit_min + blended * fit_range, 4)
            c.cross_encoder_score = float(ce_scores[i])

        top_pool.sort(key=lambda x: (-x.final_score, x.candidate_id))
        return top_pool + remaining
