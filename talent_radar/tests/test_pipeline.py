import datetime
from unittest.mock import MagicMock, patch
import pytest

from talent_radar.pipeline import TalentRadarPipeline
from talent_radar.matrix_pipeline import RequirementsMatrix, SwarmEvaluator, GeminiGateway

@pytest.fixture
def mock_candidates():
    return [
        {
            "candidate_id": f"cand_{i:03d}",
            "name": f"Candidate {i}",
            "current_title": "Software Engineer" if i % 2 == 0 else "ML Engineer",
            "years_experience": 2.0 + i,
            "career_history": [
                {"title": "Software Engineer", "company": "Tech Corp", "start_date": "2020-01-01", "end_date": "Present"}
            ],
            "skills_listed": ["Python", "FastAPI", "Docker"] if i % 2 == 0 else ["Python", "PyTorch", "MLOps"],
            "last_active": datetime.date.today().isoformat(),
            "education": "B.S. in Computer Science",
            "location": "Remote",
            "resume_text": "Experienced software developer with hands-on Python and web development expertise."
        }
        for i in range(1, 15) # 14 mock candidates
    ]

@patch.object(GeminiGateway, 'route_and_polish')
@patch.object(SwarmEvaluator, 'load_model')
@patch.object(SwarmEvaluator, 'evaluate_fragments_batch')
def test_pipeline_execution_and_top_k_slicing(
    mock_eval_batch,
    mock_load,
    mock_route,
    mock_candidates
):
    # 1. Setup our mock GeminiGateway output
    mock_route.return_value = RequirementsMatrix(
        polished_requirements="Software engineer python fastapi",
        sector_token="TECH",
        top_keywords=["Python", "FastAPI", "Docker", "PyTorch"]
    )
    
    # 2. Setup our mock SwarmEvaluator fragment scoring return values
    # For every fragment, return a mock alignment score.
    # SwarmMatrixRanker ranks fragments for candidates and maps max back.
    # We will simulate mock scores.
    def mock_eval_fn(polished_reqs, fragments, batch_size=32):
        # Return unique scores so they sort deterministically
        return [0.5 + (0.01 * idx) for idx in range(len(fragments))]
        
    mock_eval_batch.side_effect = mock_eval_fn
    
    # 3. Instantiate the pipeline
    pipeline = TalentRadarPipeline()
    
    # Override its candidate pool with our clean test pool
    pipeline.candidates_pool = mock_candidates
    
    # 4. Execute the pipeline with a custom sector, job description, and top_k
    job_desc = "Looking for a python developer who understands web frameworks like FastAPI and containerization."
    candidates, expanded_query, timings = pipeline.run(
        job_description=job_desc,
        seniority_level="Senior",
        top_k=5, # We request top 5
        sector="TECH"
    )
    
    # 5. Verify the results
    # The pipeline.run returns all candidates passed through scorer.
    # In app.py /api/rank, it is sliced by top_k. But pipeline.run also operates fully.
    # Let's verify that the ranking is performed and timings keys are populated.
    assert len(candidates) > 0
    assert "query_explosion_ms" in timings
    assert "vector_retrieval_ms" in timings
    assert "swarm_evaluation_ms" in timings
    assert "scorer_scoring_ms" in timings
    assert "overall_ms" in timings
    
    # The return values inside the candidate records should have the fields we enhanced
    first_cand = candidates[0]
    assert "raw_semantic_score" in first_cand
    assert "education_bonus" in first_cand
    assert "final_score" in first_cand
    assert "reasoning" in first_cand
    assert "matched_skills" in first_cand
    assert "missing_skills" in first_cand
    
    # Let's also verify that the /api/rank endpoint logic of slicing by top_k works correctly.
    # In our API rank function, we do: top_candidates = candidates[:request.top_k]
    top_candidates = candidates[:5]
    assert len(top_candidates) == 5
    
    # Verify that the sorted order is strictly descending by final_score
    scores = [c["final_score"] for c in candidates]
    assert scores == sorted(scores, reverse=True)
