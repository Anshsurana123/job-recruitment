"""
Tests for duplicate detection, content repetition, and train/eval data leakage validation.
"""

import pytest
from ranking.leakage import (
    compute_content_hash,
    detect_duplicate_content,
    detect_exact_duplicates,
    compute_shingle_jaccard,
    validate_splits_leakage,
)


def test_content_hashing():
    t1 = "Senior ML Engineer at Google.\nSkills: Python, PyTorch."
    t2 = "  senior ml engineer at google.   skills: python, pytorch.  "
    assert compute_content_hash(t1) == compute_content_hash(t2)


def test_detect_duplicate_content():
    unique_text = "This is a unique sentence containing diverse words to describe high-caliber engineering skills."
    is_dup, ratio = detect_duplicate_content(unique_text)
    assert not is_dup
    assert ratio == 0.0

    # Pattern of exactly 200 chars repeated 10 times
    pattern = "A" * 200
    repeated = pattern * 10
    is_dup, ratio = detect_duplicate_content(repeated)
    assert is_dup
    assert ratio >= 0.8


def test_detect_exact_duplicates():
    candidates = [
        {"candidate_id": "CAND_001", "resume_text": "ML engineer with NLP background"},
        {"candidate_id": "CAND_002", "resume_text": "Frontend developer with React"},
        {"candidate_id": "CAND_001", "resume_text": "Duplicate ID record"},
        {"candidate_id": "CAND_003", "resume_text": "ML engineer with NLP background"}  # Duplicate content
    ]
    
    dups = detect_exact_duplicates(candidates)
    assert "CAND_001" in dups["duplicate_ids"]
    assert len(dups["duplicate_content_groups"]) >= 1


def test_validate_splits_leakage_clean():
    train = [
        {"candidate_id": "CAND_001", "resume_text": "Software engineer with 5 years in backend Python"},
        {"candidate_id": "CAND_002", "resume_text": "Data scientist working on computer vision and CNNs"}
    ]
    val = [
        {"candidate_id": "CAND_003", "resume_text": "Frontend engineer specialized in React and CSS"},
        {"candidate_id": "CAND_004", "resume_text": "DevOps architect focusing on Kubernetes and AWS"}
    ]
    violations = validate_splits_leakage(train, val)
    assert len(violations) == 0


def test_validate_splits_leakage_detected():
    train = [
        {"candidate_id": "CAND_001", "resume_text": "Software engineer with 5 years in backend Python"},
        {"candidate_id": "CAND_002", "resume_text": "Identical resume text between splits"}
    ]
    val = [
        {"candidate_id": "CAND_001", "resume_text": "Different text but duplicate ID"},
        {"candidate_id": "CAND_003", "resume_text": "Identical resume text between splits"}
    ]
    violations = validate_splits_leakage(train, val)
    assert len(violations) == 2
    assert any("candidate_id" in v for v in violations)
    assert any("content leakage" in v.lower() for v in violations)


def test_shingle_jaccard():
    text1 = "senior machine learning engineer with deep pytorch experience"
    text2 = "senior machine learning engineer with deep tensorflow experience"
    jaccard = compute_shingle_jaccard(text1, text2, k=3)
    assert jaccard > 0.4  # High overlap of 3-shingles
    
    text3 = "frontend ui designer working on figma and css animations"
    jaccard_unrelated = compute_shingle_jaccard(text1, text3, k=3)
    assert jaccard_unrelated == 0.0
