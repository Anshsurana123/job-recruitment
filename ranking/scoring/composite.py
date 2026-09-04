"""
Multi-factor composite scoring engine.
Evaluates domain fit, experience bands, pedigree, and applies behavioral multipliers and honeypot filters.
"""

import math
from collections import Counter
from typing import Dict, List, Optional, Set

from ..config import DateConfig, ScoringConfig
from ..models import CandidateRecord, RetrievalMatch, ScoredCandidate
from .honeypots import detect_honeypots
from .signals import calculate_availability_multiplier


SENIORITY_MAP = {
    "intern": 0, "junior": 1, "associate": 1, "mid": 2,
    "senior": 3, "lead": 4, "principal": 5, "staff": 5, "director": 6
}

JD_RELEVANT_SKILLS = {
    "pytorch", "tensorflow", "keras", "jax", "cuda", "triton", "deep learning",
    "neural networks", "llms", "large language models", "transformers", "fine-tuning",
    "peft", "lora", "qlora", "bert", "gpt", "cnn", "rnn",
    "pinecone", "weaviate", "qdrant", "milvus", "faiss", "opensearch", "elasticsearch",
    "vector search", "semantic search", "hybrid search", "retrieval", "ranking",
    "reranking", "information retrieval", "recommendation", "recommendation systems",
    "recsys", "collaborative filtering", "rag",
    "ndcg", "mrr", "map", "a/b testing", "offline evaluation", "online evaluation",
    "python", "machine learning", "data science", "mlops", "feature engineering",
    "llama", "mistral", "huggingface", "sentence-transformers", "cross-encoder",
    "bi-encoder", "dense retrieval", "sparse retrieval", "hnsw", "annoy", "scann",
    "learning to rank", "ltr", "matrix factorization", "two-tower", "ab testing",
    "mean average precision", "ctr prediction", "recommender systems"
}

ML_DOMAIN_COMPANIES = {
    "google", "deepmind", "meta", "facebook", "openai", "anthropic", "microsoft",
    "amazon", "aws", "apple", "nvidia", "uber", "airbnb", "netflix", "spotify",
    "linkedin", "twitter", "x", "pinterest", "snap", "bytedance", "tiktok",
    "stripe", "shopify", "databricks", "snowflake", "palantir", "confluent",
    "hugging face", "huggingface", "cohere", "stability ai", "midjourney",
    "samsung research", "adobe", "salesforce", "oracle", "ibm research",
    "flipkart", "swiggy", "zomato", "meesho", "phonepe", "razorpay", "cred",
    "dream11", "juspay", "ola", "myntra", "paytm", "zerodha",
    "atlas ml", "weights & biases", "wandb", "anyscale", "ray", "modal",
    "arize", "tecton", "feast", "mlflow",
    "sarvam ai", "sarvam", "krutrim", "glance", "observe.ai", "observe", "niramai",
    "mad street den", "mad street", "rephrase.ai", "rephrase", "aganitha", "saarthi.ai",
    "saarthi", "wysa", "haptik", "genpact ai", "genpact"
}

CONSULTING_FIRMS = [
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "mphasis", "tech mahindra", "mindtree"
]

NON_TECH_TITLES = {
    "marketing manager", "accountant", "hr manager",
    "operations manager", "sales executive", "customer support"
}


def infer_seniority_level(title: str) -> int:
    t_lower = (title or "").lower()
    if "director" in t_lower: return 6
    if "principal" in t_lower or "staff" in t_lower: return 5
    if "lead" in t_lower or "head" in t_lower: return 4
    if "senior" in t_lower or "sr" in t_lower: return 3
    if "junior" in t_lower or "jr" in t_lower: return 1
    if "intern" in t_lower or "co-op" in t_lower: return 0
    if "associate" in t_lower: return 1
    return 2


class CompositeScorer:
    """
    Deterministic domain scoring engine for candidates matching the Senior AI Engineer profile.
    """
    def __init__(
        self,
        scoring_config: Optional[ScoringConfig] = None,
        date_config: Optional[DateConfig] = None
    ):
        self.config = scoring_config or ScoringConfig()
        self.date_config = date_config or DateConfig()

    def score(
        self,
        retrieval_matches: List[RetrievalMatch],
        reference_date: Optional[any] = None
    ) -> List[ScoredCandidate]:
        if not retrieval_matches:
            return []

        ref_date = self.date_config.resolve_reference_date(reference_date)
        
        # Skill IDF computation across retrieved candidates
        skill_df = Counter()
        for m in retrieval_matches:
            c_skills = {s.get("name", "").lower() for s in m.record.skills if s.get("name")}
            for sk in c_skills:
                skill_df[sk] += 1
                
        total_retrieved = len(retrieval_matches)
        skill_idf = {}
        for sk, df in skill_df.items():
            skill_idf[sk] = math.log(total_retrieved / df) if df > 0 else 0.0
        raw_max = max(skill_idf.values()) if skill_idf else 1.0
        max_idf = raw_max if raw_max > 0.0 else 1.0
        skill_idf_norm = {sk: v / max_idf for sk, v in skill_idf.items()}

        # Min-max range of hybrid score for normalization
        hybrid_vals = [m.hybrid_score for m in retrieval_matches]
        min_h, max_h = min(hybrid_vals), max(hybrid_vals)
        h_range = max_h - min_h if max_h > min_h else 1.0

        scored_list: List[ScoredCandidate] = []

        for m in retrieval_matches:
            cand = m.record
            p = cand.profile
            career = cand.career_history
            skills = cand.skills
            education = cand.education
            signals = cand.redrob_signals

            # 1. Hybrid score contribution
            norm_hybrid = 100.0 * (m.hybrid_score - min_h) / h_range
            fit_score = self.config.hybrid_score_weight * norm_hybrid

            # 2. Skill overlap & depth
            skills_lower = {s.get("name", "").lower() for s in skills if s.get("name")}
            career_text = " ".join([
                (job.get("title", "") + " " + job.get("description", "")).lower()
                for job in career
            ])

            ml_skills = {"pytorch", "tensorflow", "keras", "jax", "cuda", "triton", "deep learning", "neural networks", "transformers", "fine-tuning", "peft", "lora"}
            ir_skills = {"pinecone", "weaviate", "qdrant", "milvus", "faiss", "opensearch", "elasticsearch", "vector search", "semantic search", "hybrid search", "retrieval", "ranking", "reranking", "rag"}
            eval_skills = {"ndcg", "mrr", "map", "a/b testing", "offline evaluation", "online evaluation", "evaluation framework"}

            matched_ml = skills_lower & ml_skills
            matched_ir = skills_lower & ir_skills
            matched_eval = skills_lower & eval_skills

            # Boost for verified ML/IR skills with assessment scores
            assess_scores = signals.get("skill_assessment_scores", {})
            for sk in matched_ml | matched_ir:
                idf_wt = skill_idf_norm.get(sk, 0.5)
                score_boost = 3.0 * idf_wt
                if sk in assess_scores and assess_scores[sk] >= 70:
                    score_boost *= 1.3
                fit_score += score_boost

            for sk in matched_eval:
                fit_score += 4.0 * skill_idf_norm.get(sk, 0.5)

            # 3. Experience band calibration
            years_exp = p.years_of_experience
            if self.config.min_exp <= years_exp <= self.config.max_exp:
                fit_score += 15.0  # Optimal experience band
            elif years_exp < 3.0:
                fit_score -= 20.0  # Heavy penalty for junior profiles
            elif years_exp < self.config.min_exp:
                fit_score += 5.0
            elif years_exp > 12.0 and not (matched_ml and matched_ir):
                fit_score -= 10.0

            # 4. Title relevance
            title_lower = p.current_title.lower()
            if any(t in title_lower for t in ["ai engineer", "machine learning engineer", "ml engineer", "search engineer", "recommendation"]):
                fit_score += 12.0
            elif any(t in title_lower for t in ["data scientist", "applied scientist"]):
                fit_score += 8.0
            elif any(t in title_lower for t in NON_TECH_TITLES):
                fit_score -= 50.0  # Disqualify non-tech roles

            # 5. Pedigree & Company Background
            has_ml_domain = False
            ml_comp_name = ""
            for job in career:
                c_name = job.get("company", "").lower()
                for dom_c in ML_DOMAIN_COMPANIES:
                    if dom_c in c_name:
                        has_ml_domain = True
                        ml_comp_name = job.get("company", "")
                        break
                if has_ml_domain:
                    break

            if has_ml_domain:
                fit_score += 10.0

            # Consulting-only penalty
            is_all_consulting = len(career) > 0 and all(
                any(cf in job.get("company", "").lower() for cf in CONSULTING_FIRMS)
                for job in career
            )
            if is_all_consulting:
                fit_score -= 25.0

            # Education prestige
            has_tier1 = any(edu.get("tier") == "tier_1" for edu in education)
            has_masters = any(any(d in edu.get("degree", "").lower() for d in ["master", "m.sc", "m.tech", "m.s.", "ms", "ph.d", "phd"]) for edu in education)
            if has_tier1:
                fit_score += 6.0
            elif has_masters:
                fit_score += 3.0

            fit_score = max(0.0, fit_score)

            # 6. Availability & Behavioral Multiplier
            avail_multiplier = calculate_availability_multiplier(
                p, signals, self.config.target_cities, reference_date=ref_date
            )
            final_score = fit_score * avail_multiplier

            # 7. Honeypot check
            is_honeypot = False
            disqual_reason = ""
            if self.config.calibrated_honeypots:
                is_honeypot, disqual_reason = detect_honeypots(
                    career,
                    skills,
                    reference_date=ref_date,
                    foundation_dates=self.date_config.foundation_dates,
                    duration_tolerance_months=self.config.honeypot_duration_tolerance_months
                )

            if is_honeypot:
                final_score = 0.0

            # Domain specific flags
            jd_matching_count = len(skills_lower & JD_RELEVANT_SKILLS)
            total_skills = max(len(skills), 1)
            is_cv_speech = any(k in career_text for k in ["computer vision", "speech recognition", "whisper", "asr", "image segmentation"])
            has_nlp_ir = len(matched_ir | matched_ml) >= 2

            scored_list.append(ScoredCandidate(
                candidate_id=cand.candidate_id,
                record=cand,
                final_score=round(final_score, 4),
                fit_score=round(fit_score, 4),
                avail_multiplier=round(avail_multiplier, 4),
                is_honeypot=is_honeypot,
                disqualification_reason=disqual_reason,
                has_ml_domain_company=has_ml_domain,
                ml_domain_company_name=ml_comp_name,
                jd_matching_count=jd_matching_count,
                skill_concentration=jd_matching_count / total_skills,
                is_cv_speech_primary=is_cv_speech,
                has_nlp_ir_compensation=has_nlp_ir
            ))

        # Sort descending by final_score, tie-break by candidate_id ascending
        scored_list.sort(key=lambda x: (-x.final_score, x.candidate_id))
        return scored_list
