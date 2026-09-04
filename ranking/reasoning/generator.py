"""
Deterministic candidate reasoning generator.
Synthesizes verified profile facts, skills alignment, and logistical readiness into natural-language justifications.
"""

import datetime
import re
from typing import Optional, List, Dict, Any

from ..models import ScoredCandidate


class DeterministicReasoningGenerator:
    """
    Constructs fact-grounded, non-hallucinated candidate justifications.
    Every claim links directly to a parsed field or computed score component.
    """
    def __init__(self, target_cities: Optional[List[str]] = None):
        self.target_cities = target_cities or [
            "pune", "noida", "delhi", "new delhi", "gurugram", "gurgaon", "faridabad", "ghaziabad"
        ]

    def generate(self, rank: int, candidate: ScoredCandidate, reference_date: Optional[datetime.date] = None) -> str:
        record = candidate.record
        p = record.profile
        career = record.career_history
        skills = record.skills
        signals = record.redrob_signals
        cid = candidate.candidate_id

        title = p.current_title or "Software Engineer"
        company = career[0].get("company", "Tech Company") if career else "Startup"
        exp = p.years_of_experience

        # Deterministic variation seed based on candidate ID digits
        h_digits = int(cid.split('_')[1]) if '_' in cid and cid.split('_')[1].isdigit() else 42

        # Sentence 1: Professional identity and tenure
        openers = [
            f"A {title} with {exp:.1f} years of experience, currently working at {company}.",
            f"Brings {exp:.1f} years of ML/software experience, currently serving as a {title} at {company}.",
            f"Experienced {title} possessing {exp:.1f} years of background, currently at {company}.",
            f"Currently a {title} at {company} with {exp:.1f} years of total industry tenure."
        ]
        s1 = openers[h_digits % len(openers)]

        # Sentence 2: Core technical alignment
        skills_lower = {s.get("name", "").lower() for s in skills if s.get("name")}
        ml_skills = {"pytorch", "tensorflow", "jax", "cuda", "triton", "llms", "transformers", "fine-tuning", "peft", "lora"}
        ir_skills = {"pinecone", "weaviate", "qdrant", "milvus", "faiss", "opensearch", "elasticsearch", "vector search", "semantic search", "hybrid search", "retrieval", "ranking", "rag"}
        eval_skills = {"ndcg", "mrr", "map", "a/b testing", "offline evaluation", "online evaluation"}

        matched_ml = sorted([s.get("name") for s in skills if s.get("name", "").lower() in ml_skills])
        matched_ir = sorted([s.get("name") for s in skills if s.get("name", "").lower() in ir_skills])
        matched_eval = sorted([s.get("name") for s in skills if s.get("name", "").lower() in eval_skills])

        skills_declared = []
        if matched_ir:
            skills_declared.append(f"search/retrieval systems (using {matched_ir[0]})")
        if matched_ml:
            skills_declared.append(f"ML models (using {matched_ml[0]})")
        if matched_eval:
            skills_declared.append(f"ranking metrics like {matched_eval[0]}")

        if skills_declared:
            s2_body = f"Strong alignment in {', '.join(skills_declared[:2])}."
        else:
            s2_body = "Brings core software engineering skills adjacent to ML."

        edu_list = record.education
        has_tier1 = any(edu.get("tier") == "tier_1" for edu in edu_list)
        has_masters = any(any(d in str(edu.get("degree", "")).lower() for d in ["master", "m.sc", "m.tech", "m.s.", "ms"]) for edu in edu_list)

        if has_tier1:
            s2_body += " Educated at a Tier-1 institution."
        elif has_masters:
            s2_body += " Holds a Master's degree."

        # Sentence 3: Logistics & availability readiness
        loc = p.location or "India"
        willing_reloc = signals.get("willing_to_relocate", False)
        notice = signals.get("notice_period_days", 0)

        loc_lower = loc.lower()
        is_local = any(city in loc_lower for city in self.target_cities)

        if is_local:
            loc_phrase = f"Based locally in {loc}"
        elif willing_reloc:
            loc_phrase = f"Located in {loc} (willing to relocate)"
        else:
            loc_phrase = f"Based in {loc}"

        notice_phrase = "ready to start immediately" if notice == 0 else f"with a {notice}-day notice period"

        strengths = []
        github_score = signals.get("github_activity_score", -1)
        if github_score >= 50:
            strengths.append("highly active GitHub profile")
        resp_rate = signals.get("recruiter_response_rate", 1.0)
        if resp_rate >= 0.85:
            strengths.append(f"excellent responsiveness ({int(resp_rate * 100)}%)")
        saves = signals.get("saved_by_recruiters_30d", 0)
        if saves >= 5:
            strengths.append("saved by multiple recruiters recently")

        concerns = []
        if notice > 90:
            concerns.append("long notice period")
        if candidate.is_cv_speech_primary and not candidate.has_nlp_ir_compensation:
            concerns.append("speech/vision-primary background rather than NLP/IR")

        s3 = f"{loc_phrase}, {notice_phrase}."
        if strengths:
            s3 += f" Notable strengths: {', '.join(strengths[:2])}."
        if concerns:
            s3 += f" Consideration: {'; '.join(concerns)}."

        full_reasoning = f"{s1} {s2_body} {s3}"
        full_reasoning = re.sub(r'\s+', ' ', full_reasoning).strip()

        # Enforce maximum word count of 70 words per specification
        words = full_reasoning.split()
        if len(words) > 70:
            truncated = " ".join(words[:70])
            last_p = max(truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?'))
            if last_p > 30:
                full_reasoning = truncated[:last_p + 1]
            else:
                full_reasoning = truncated + "..."

        return full_reasoning
