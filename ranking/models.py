"""
Strongly typed data structures and domain models for the IR and Ranking pipeline.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CandidateProfile:
    candidate_id: str
    current_title: str = ""
    headline: str = ""
    summary: str = ""
    years_of_experience: float = 0.0
    location: str = ""
    country: str = "India"
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateRecord:
    candidate_id: str
    profile: CandidateProfile
    career_history: List[Dict[str, Any]] = field(default_factory=list)
    education: List[Dict[str, Any]] = field(default_factory=list)
    skills: List[Dict[str, Any]] = field(default_factory=list)
    certifications: List[Dict[str, Any]] = field(default_factory=list)
    languages: List[Dict[str, Any]] = field(default_factory=list)
    redrob_signals: Dict[str, Any] = field(default_factory=dict)
    resume_text: str = ""
    content_hash: str = ""


@dataclass
class ProcessedCandidateFields:
    candidate_id: str
    title_headline: str
    skills: str
    career_titles: str
    summary: str
    career_descriptions: str
    education: str
    other: str
    full_text: str


@dataclass
class RetrievalMatch:
    candidate_id: str
    hybrid_score: float
    bm25_score: float
    semantic_score: float
    record: CandidateRecord


@dataclass
class ScoredCandidate:
    candidate_id: str
    record: CandidateRecord
    final_score: float
    fit_score: float
    avail_multiplier: float
    is_honeypot: bool = False
    disqualification_reason: str = ""
    has_credibility_concern: bool = False
    credibility_warning_skills: List[str] = field(default_factory=list)
    has_salary_inversion: bool = False
    has_ml_domain_company: bool = False
    ml_domain_company_name: str = ""
    jd_matching_count: int = 0
    skill_concentration: float = 0.0
    is_cv_speech_primary: bool = False
    has_nlp_ir_compensation: bool = False
    has_skills_stuffing_concern: bool = False
    has_skills_duration_mismatch: bool = False
    cross_encoder_score: Optional[float] = None


@dataclass
class RankedCandidate:
    candidate_id: str
    rank: int
    score: float
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)
