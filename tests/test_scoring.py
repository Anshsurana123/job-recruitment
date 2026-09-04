"""
Tests for CompositeScorer, Honeypot detection, and Availability signal multipliers.
"""

import datetime
import pytest
from ranking.config import DateConfig, ScoringConfig
from ranking.models import CandidateProfile, CandidateRecord, RetrievalMatch
from ranking.scoring.composite import CompositeScorer, infer_seniority_level
from ranking.scoring.honeypots import detect_honeypots
from ranking.scoring.signals import calculate_availability_multiplier


def test_infer_seniority_level():
    assert infer_seniority_level("Director of Engineering") == 6
    assert infer_seniority_level("Principal Architect") == 5
    assert infer_seniority_level("Staff ML Engineer") == 5
    assert infer_seniority_level("Lead Developer") == 4
    assert infer_seniority_level("Senior Python Dev") == 3
    assert infer_seniority_level("Software Engineer") == 2
    assert infer_seniority_level("Junior Associate") == 1
    assert infer_seniority_level("Engineering Intern") == 0


def test_honeypot_duration_mismatch():
    ref_date = datetime.date(2026, 5, 20)
    foundations = {"krutrim": datetime.date(2023, 4, 1)}
    
    # Career claims 48 months, but 2024-01-01 to 2025-01-01 is only 12 months
    career = [{"title": "Eng", "company": "Co", "start_date": "2024-01-01", "end_date": "2025-01-01", "duration_months": 48}]
    skills = [{"name": "Python", "proficiency": "mid", "duration_months": 12}]
    
    is_hp, reason = detect_honeypots(career, skills, reference_date=ref_date, foundation_dates=foundations)
    assert is_hp
    assert "mismatch" in reason.lower()


def test_honeypot_foundation_date():
    ref_date = datetime.date(2026, 5, 20)
    foundations = {"krutrim": datetime.date(2023, 4, 1)}
    
    # Candidate claims to have worked at Krutrim in 2021 before it was founded in April 2023
    career = [{"title": "Eng", "company": "Krutrim AI", "start_date": "2021-01-01", "end_date": "2023-01-01", "duration_months": 24}]
    skills = [{"name": "Python", "proficiency": "mid", "duration_months": 24}]
    
    is_hp, reason = detect_honeypots(career, skills, reference_date=ref_date, foundation_dates=foundations)
    assert is_hp
    assert "krutrim" in reason.lower()
    assert "foundation" in reason.lower()


def test_honeypot_expert_zero_duration():
    ref_date = datetime.date(2026, 5, 20)
    foundations = {}
    
    career = [{"title": "Eng", "company": "Tech Corp", "start_date": "2022-01-01", "end_date": "2024-01-01", "duration_months": 24}]
    # Expert proficiency claimed with 0 duration
    skills = [{"name": "PyTorch", "proficiency": "expert", "duration_months": 0}]
    
    is_hp, reason = detect_honeypots(career, skills, reference_date=ref_date, foundation_dates=foundations)
    assert is_hp
    assert "expert skill" in reason.lower()


def test_availability_multiplier_location_and_notice():
    ref_date = datetime.date(2026, 5, 20)
    target_cities = ["pune", "noida", "delhi"]
    
    # Local candidate in Pune with 15 day notice
    p_local = CandidateProfile(candidate_id="C1", location="Pune, Maharashtra", country="India")
    sig_local = {"notice_period_days": 15, "last_active_date": "2026-05-18", "willing_to_relocate": False, "open_to_work_flag": True}
    mult_local = calculate_availability_multiplier(p_local, sig_local, target_cities, ref_date)

    # Remote international candidate unwilling to relocate with 120 day notice
    p_remote = CandidateProfile(candidate_id="C2", location="Berlin, Germany", country="Germany")
    sig_remote = {"notice_period_days": 120, "last_active_date": "2025-01-01", "willing_to_relocate": False}
    mult_remote = calculate_availability_multiplier(p_remote, sig_remote, target_cities, ref_date)

    assert mult_local > 1.0
    assert mult_remote < 0.1
    assert mult_local > mult_remote * 10


def test_composite_scorer_ranking_differentiation():
    ref_date = datetime.date(2026, 5, 20)
    scorer = CompositeScorer(ScoringConfig(), DateConfig(evaluation_date=ref_date))
    
    # Candidate A: Senior AI Engineer with 7 years, PyTorch, FAISS, NDCG, at Google
    cand_a = CandidateRecord(
        candidate_id="CAND_TOP",
        profile=CandidateProfile(candidate_id="CAND_TOP", current_title="Senior AI Engineer", years_of_experience=7.0, location="Noida, India"),
        career_history=[{"title": "Senior AI Engineer", "company": "Google", "start_date": "2020-01-01", "end_date": "Present", "duration_months": 76}],
        skills=[{"name": "PyTorch", "proficiency": "expert", "duration_months": 60}, {"name": "FAISS", "proficiency": "mid", "duration_months": 36}, {"name": "NDCG", "proficiency": "mid", "duration_months": 24}],
        education=[{"degree": "Master of Technology", "tier": "tier_1"}],
        redrob_signals={"notice_period_days": 30, "last_active_date": "2026-05-15", "recruiter_response_rate": 0.95}
    )

    # Candidate B: Marketing Manager with 1 year, no ML skills
    cand_b = CandidateRecord(
        candidate_id="CAND_LOW",
        profile=CandidateProfile(candidate_id="CAND_LOW", current_title="Marketing Manager", years_of_experience=1.0, location="Paris, France"),
        career_history=[{"title": "Marketing Associate", "company": "AdAgency", "start_date": "2024-01-01", "end_date": "Present", "duration_months": 12}],
        skills=[{"name": "SEO", "proficiency": "mid", "duration_months": 12}],
        education=[],
        redrob_signals={"notice_period_days": 90, "last_active_date": "2025-01-01", "recruiter_response_rate": 0.10}
    )

    matches = [
        RetrievalMatch(candidate_id="CAND_TOP", hybrid_score=95.0, bm25_score=25.0, semantic_score=0.88, record=cand_a),
        RetrievalMatch(candidate_id="CAND_LOW", hybrid_score=10.0, bm25_score=0.0, semantic_score=0.20, record=cand_b),
    ]

    scored = scorer.score(matches, reference_date=ref_date)
    assert len(scored) == 2
    assert scored[0].candidate_id == "CAND_TOP"
    assert scored[0].final_score > 50.0
    assert scored[1].candidate_id == "CAND_LOW"
    assert scored[1].final_score < 5.0
