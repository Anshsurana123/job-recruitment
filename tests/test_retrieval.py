"""
Tests for retrieval components: BM25F lexical index, Dense bi-encoder retriever, and Hybrid fusion.
"""

import pytest
from ranking.config import RetrievalConfig
from ranking.models import CandidateRecord, CandidateProfile, ProcessedCandidateFields
from ranking.retrieval.bm25f import BM25FIndex, tokenize
from ranking.retrieval.hybrid import HybridFusion


@pytest.fixture
def sample_candidate_fields():
    return [
        ProcessedCandidateFields(
            candidate_id="CAND_001",
            title_headline="Senior AI Engineer",
            skills="Python PyTorch Transformers FAISS",
            career_titles="Senior AI Engineer ML Engineer",
            summary="Building search engines and large scale vector retrieval systems.",
            career_descriptions="Deployed hybrid search using BM25 and vector embeddings.",
            education="M.S. Computer Science",
            other="Google",
            full_text="Senior AI Engineer. Python PyTorch Transformers. Vector retrieval systems."
        ),
        ProcessedCandidateFields(
            candidate_id="CAND_002",
            title_headline="Frontend Developer",
            skills="React TypeScript HTML CSS",
            career_titles="Frontend Developer UI Engineer",
            summary="Specialized in building responsive web applications and component libraries.",
            career_descriptions="Implemented responsive UI components using TailwindCSS.",
            education="B.S. Design",
            other="Startup Inc",
            full_text="Frontend Developer. React TypeScript HTML CSS."
        ),
        ProcessedCandidateFields(
            candidate_id="CAND_003",
            title_headline="Recommendation Systems Engineer",
            skills="Python Deep Learning PyTorch Weaviate RecSys",
            career_titles="ML Engineer Recommendation Specialist",
            summary="Shipped recommendation ranking algorithms and collaborative filtering.",
            career_descriptions="Trained deep retrieval two-tower models for candidate generation.",
            education="B.Tech Computer Science",
            other="CRED",
            full_text="Recommendation Systems Engineer. PyTorch Weaviate RecSys."
        )
    ]


def test_tokenize_concept_expansion():
    tokens = tokenize("Working on deep learning with PyTorch and Pinecone vector search")
    assert "concept_deep_learning" in tokens
    assert "concept_vector_search" in tokens
    assert "concept_ml_framework" in tokens
    assert "pytorch" in tokens
    assert "pinecone" in tokens


def test_bm25f_scoring_relevance(sample_candidate_fields):
    index = BM25FIndex()
    for f in sample_candidate_fields:
        index.add_candidate(f)
    index.finalize()

    query_tokens = tokenize("PyTorch vector retrieval search engine")
    scores = index.get_scores(query_tokens)

    # CAND_001 has highest relevance to vector retrieval and search engine
    assert scores["CAND_001"] > scores["CAND_002"]
    assert scores["CAND_001"] > scores["CAND_003"]
    # Frontend developer should have ~0 relevance
    assert scores["CAND_002"] == 0.0


def test_hybrid_fusion():
    fusion = HybridFusion(RetrievalConfig(hybrid_weights=(0.60, 0.40)))
    
    r1 = CandidateRecord(candidate_id="CAND_001", profile=CandidateProfile(candidate_id="CAND_001"))
    r2 = CandidateRecord(candidate_id="CAND_002", profile=CandidateProfile(candidate_id="CAND_002"))
    
    records = [r1, r2]
    bm25_scores = {"CAND_001": 20.0, "CAND_002": 0.0}
    dense_scores = {"CAND_001": 0.85, "CAND_002": 0.35}

    matches = fusion.fuse(records, bm25_scores, dense_scores, top_k=2)
    assert len(matches) == 2
    assert matches[0].candidate_id == "CAND_001"
    assert matches[0].hybrid_score > matches[1].hybrid_score
    # CAND_001 has max BM25 and max dense -> normalized hybrid should be 100.0
    assert pytest.approx(matches[0].hybrid_score, 0.1) == 100.0
    # CAND_002 has min BM25 and min dense -> normalized hybrid should be 0.0
    assert pytest.approx(matches[1].hybrid_score, 0.1) == 0.0
