"""
Comprehensive stress and edge-case testing for the IR and Ranking pipeline.
Ensures predictable, graceful handling of pathological inputs.
"""

import datetime
import pytest
from ranking.config import PipelineConfig, RerankingConfig
from ranking.dates import calculate_years_span, calculate_total_experience, parse_date_string
from ranking.ingestion import normalize_candidate_record, extract_processed_fields
from ranking.models import CandidateProfile, CandidateRecord, ScoredCandidate, RetrievalMatch
from ranking.pipeline import RankingPipeline
from ranking.reranking.cross_encoder import CrossEncoderReranker
from ranking.scoring.composite import CompositeScorer


# =====================================================================
# 1. Experience & Date Edge Cases
# =====================================================================

def test_experience_zero_years():
    ref_date = datetime.date(2026, 5, 20)
    history = [{"title": "Intern", "start_date": "2026-05-20", "end_date": "2026-05-20"}]
    total = calculate_total_experience(history, reference_date=ref_date)
    assert total == 0.0


def test_experience_future_dates():
    ref_date = datetime.date(2026, 5, 20)
    # Start date in the future
    history = [{"title": "Engineer", "start_date": "2028-01-01", "end_date": "2029-01-01"}]
    total = calculate_total_experience(history, reference_date=ref_date)
    # 2028-01-01 to 2029-01-01 is 366 days
    assert pytest.approx(total, 0.01) == 366 / 365.25


def test_experience_invalid_dates():
    ref_date = datetime.date(2026, 5, 20)
    assert calculate_years_span("corrupt-date", "another-corrupt-date", reference_date=ref_date) == 1.0
    assert calculate_years_span(None, None, reference_date=ref_date) == 1.0


def test_experience_micro_tenures():
    ref_date = datetime.date(2026, 5, 20)
    # 5 days tenure
    span = calculate_years_span("2025-01-01", "2025-01-06", reference_date=ref_date)
    assert 0.0 < span < 0.02


def test_multiple_concurrent_present_roles():
    ref_date = datetime.date(2026, 5, 20)
    history = [
        {"title": "Role 1", "start_date": "2024-01-01", "end_date": "Present"},
        {"title": "Role 2", "start_date": "2024-06-01", "end_date": "Present"},
        {"title": "Role 3", "start_date": "2025-01-01", "end_date": "Present"}
    ]
    # Merged tenure should simply be 2024-01-01 to 2026-05-20 (870 days)
    total = calculate_total_experience(history, reference_date=ref_date, merge_overlaps=True)
    assert pytest.approx(total, 0.01) == 870 / 365.25


# =====================================================================
# 2. Candidate Content Edge Cases
# =====================================================================

def test_empty_candidate_profile():
    raw = {"candidate_id": "CAND_EMPTY"}
    record = normalize_candidate_record(raw)
    assert record.candidate_id == "CAND_EMPTY"
    assert record.profile.years_of_experience == 0.0
    assert record.skills == []
    fields = extract_processed_fields(record)
    assert isinstance(fields.full_text, str)
    assert "CAND_EMPTY" in fields.candidate_id


def test_huge_resume_content():
    huge_text = ("Senior ML Engineer specialized in PyTorch and embeddings. " * 5000)  # ~300,000 characters
    raw = {
        "candidate_id": "CAND_HUGE",
        "resume_text": huge_text,
        "profile": {"current_title": "ML Engineer", "summary": huge_text[:1000]}
    }
    record = normalize_candidate_record(raw)
    fields = extract_processed_fields(record)
    assert len(record.content_hash) == 64
    assert len(fields.full_text) > 0


def test_unusual_formatting_and_unicode():
    strange_text = "Senior \x00 ML Engineer \u200b with \U0001F680 PyTorch & C++ // \t\n\r"
    raw = {
        "candidate_id": "CAND_UNICODE",
        "profile": {"current_title": strange_text, "summary": "Tested with emojis and nulls \x00"}
    }
    record = normalize_candidate_record(raw)
    fields = extract_processed_fields(record)
    assert "ML Engineer" in fields.title_headline


def test_missing_critical_fields():
    raw = {
        "candidate_id": "CAND_MISSING",
        # missing: profile, skills, education, career_history, redrob_signals
    }
    record = normalize_candidate_record(raw)
    scorer = CompositeScorer()
    match = RetrievalMatch(candidate_id="CAND_MISSING", hybrid_score=10.0, bm25_score=1.0, semantic_score=0.1, record=record)
    scored = scorer.score([match], reference_date=datetime.date(2026, 5, 20))
    assert len(scored) == 1
    assert scored[0].final_score >= 0.0


# =====================================================================
# 3. Ranking & Pipeline Edge Cases
# =====================================================================

def test_zero_candidates_in_pipeline():
    pipeline = RankingPipeline()
    results = pipeline.run(candidates_input=[], top_n=10)
    assert results == []


def test_single_candidate_in_pipeline():
    pipeline = RankingPipeline()
    cand = {
        "candidate_id": "CAND_SOLO",
        "profile": {"current_title": "AI Engineer", "years_of_experience": 6.0, "location": "Pune"},
        "skills": [{"name": "PyTorch", "proficiency": "expert", "duration_months": 36}],
        "career_history": [{"title": "AI Engineer", "company": "Product Co", "start_date": "2020-01-01", "end_date": "Present"}]
    }
    results = pipeline.run(candidates_input=[cand], top_n=10)
    assert len(results) == 1
    assert results[0].candidate_id == "CAND_SOLO"
    assert results[0].rank == 1
    assert results[0].score == 1.0000


def test_identical_scores_tie_breaking():
    """Ties must strictly break by candidate_id ascending."""
    pipeline = RankingPipeline()
    
    # Three candidates with identical profiles except candidate_id
    cand_z = {
        "candidate_id": "CAND_ZZZ",
        "profile": {"current_title": "AI Engineer", "years_of_experience": 6.0, "location": "Pune"},
        "skills": [{"name": "PyTorch", "proficiency": "mid", "duration_months": 24}]
    }
    cand_a = {
        "candidate_id": "CAND_AAA",
        "profile": {"current_title": "AI Engineer", "years_of_experience": 6.0, "location": "Pune"},
        "skills": [{"name": "PyTorch", "proficiency": "mid", "duration_months": 24}]
    }
    cand_m = {
        "candidate_id": "CAND_MMM",
        "profile": {"current_title": "AI Engineer", "years_of_experience": 6.0, "location": "Pune"},
        "skills": [{"name": "PyTorch", "proficiency": "mid", "duration_months": 24}]
    }
    
    results = pipeline.run(candidates_input=[cand_z, cand_a, cand_m], top_n=10)
    assert len(results) == 3
    # Scores must be non-increasing
    assert results[0].score >= results[1].score >= results[2].score
    # Ties must be sorted alphabetically by candidate_id
    if results[0].score == results[1].score == results[2].score:
        assert [r.candidate_id for r in results] == ["CAND_AAA", "CAND_MMM", "CAND_ZZZ"]


def test_missing_reranker_model_graceful_fallback():
    # Configure an invalid path for cross-encoder
    reranker = CrossEncoderReranker(RerankingConfig(cross_encoder_path="./nonexistent_model_dir"))
    record = CandidateRecord(candidate_id="C1", profile=CandidateProfile(candidate_id="C1"))
    candidates = [ScoredCandidate(candidate_id="C1", record=record, final_score=50.0, fit_score=50.0, avail_multiplier=1.0)]
    
    # Must not crash, should return candidates intact
    reranked = reranker.rerank(candidates, query_text="query", top_k=10)
    assert len(reranked) == 1
    assert reranked[0].final_score == 50.0
