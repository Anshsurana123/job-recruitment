import pytest
from talent_radar.question_generator import TechnicalQuestionGenerator, PhoneScreenGuide
from talent_radar.comparison import CandidateComparator, ComparisonSynthesis
from talent_radar.exporter import BriefExporter

@pytest.fixture
def sample_candidates():
    return [
        {
            "candidate_id": "cand_001",
            "name": "Jane Doe",
            "current_title": "Senior ML Engineer",
            "years_experience": 8.0,
            "career_history": [
                {"title": "Senior ML Engineer", "company": "Tech Corp", "start_date": "2022-01-01", "end_date": "Present"},
                {"title": "ML Engineer", "company": "Startup Inc", "start_date": "2018-01-01", "end_date": "2022-01-01"}
            ],
            "skills_listed": ["Python", "PyTorch", "Transformers", "SQL"],
            "last_active": "2026-05-15",
            "education": "M.S. in Computer Science",
            "location": "San Francisco, CA",
            "resume_text": "Experienced Senior Machine Learning Engineer specialized in PyTorch, NLP Transformers, and scalable inference architectures.",
            "matched_skills": ["Python", "PyTorch"],
            "missing_skills": ["Triton", "CUDA"],
            "semantic_score": 92.5,
            "velocity_score": 8.5,
            "freshness_label": "Active Now",
            "status_label": "Top Hidden Gem 🚀",
            "final_score": 89.2,
            "reasoning": "Elite candidate showing exceptional career velocity."
        },
        {
            "candidate_id": "cand_002",
            "name": "Bob Builder",
            "current_title": "Software Developer",
            "years_experience": 3.0,
            "career_history": [
                {"title": "Software Developer", "company": "Build Co", "start_date": "2023-01-01", "end_date": "Present"},
                {"title": "Junior Developer", "company": "Web Co", "start_date": "2021-01-01", "end_date": "2023-01-01"}
            ],
            "skills_listed": ["JavaScript", "HTML", "CSS"],
            "last_active": "2026-05-10",
            "education": "B.A. in History",
            "location": "Boston, MA",
            "resume_text": "Front-end developer focused on HTML, CSS, and modern JavaScript frameworks.",
            "matched_skills": [],
            "missing_skills": ["React", "TypeScript"],
            "semantic_score": 50.0,
            "velocity_score": 3.0,
            "freshness_label": "Recent",
            "status_label": "Potential Fit ⭐",
            "final_score": 56.4,
            "reasoning": "Steady career progression and solid competence."
        }
    ]

def test_technical_question_generator(sample_candidates):
    generator = TechnicalQuestionGenerator()
    cand = sample_candidates[0]
    
    # Run dynamic questions generation (will cascade to local fallback since no Gemini API key in test environment)
    guide = generator.generate_questions(cand, "Looking for ML optimization expert", "TECH")
    
    assert isinstance(guide, PhoneScreenGuide)
    assert guide.candidate_name == "Jane Doe"
    assert len(guide.questions) == 5
    
    # Assert question structure and content
    assert len(guide.questions[0].expected_answer) > 10
    assert guide.questions[1].question != ""
    assert guide.questions[1].rationale != ""

def test_candidate_comparator(sample_candidates):
    comparator = CandidateComparator()
    
    # Run comparison (will cascade to local fallback)
    synthesis = comparator.compare_candidates(sample_candidates, "Looking for ML optimization expert")
    
    assert isinstance(synthesis, ComparisonSynthesis)
    assert len(synthesis.metrics) == 2
    assert synthesis.metrics[0].candidate_name == "Jane Doe"
    assert synthesis.metrics[1].candidate_name == "Bob Builder"
    assert len(synthesis.strengths_synthesis) > 10
    assert len(synthesis.recruiter_recommendation) > 10

def test_brief_exporters(sample_candidates):
    # Test Markdown brief generation
    md_brief = BriefExporter.generate_markdown_brief(
        "Looking for ML optimization expert",
        "Senior",
        "TECH",
        sample_candidates,
        top_k=2
    )
    
    assert "# 🧠 SwarmMatrix AI — Talent Radar Search Brief" in md_brief
    assert "Jane Doe" in md_brief
    assert "Bob Builder" in md_brief
    assert "Final Score" in md_brief
    assert "Skill Analysis" in md_brief
    
    # Test CSV brief generation
    csv_brief = BriefExporter.generate_csv_brief(sample_candidates)
    
    assert "Jane Doe" in csv_brief
    assert "Bob Builder" in csv_brief
    assert "Final Composite Score" in csv_brief

def test_custom_scorer_weights(sample_candidates):
    from talent_radar.scorer import CandidateScorer
    
    # 1. High Semantic weight
    scorer_sem = CandidateScorer(semantic_weight=0.80, velocity_weight=0.10, freshness_weight=0.10)
    scored_sem = scorer_sem.score_candidates([c.copy() for c in sample_candidates])

    # 2. High Velocity weight
    scorer_vel = CandidateScorer(semantic_weight=0.10, velocity_weight=0.80, freshness_weight=0.10)
    scored_vel = scorer_vel.score_candidates([c.copy() for c in sample_candidates])
    
    # Assert that the final composite scores differ because different weights were applied
    assert scored_sem[1]["final_score"] != scored_vel[1]["final_score"]

def test_ingest_deduplication():
    from talent_radar.smart_ingest import _build_candidate_record, SmartIngestJob
    
    existing = [
        {
            "candidate_id": "cand_0001",
            "name": "Alice Smith",
            "current_title": "Software Engineer",
            "years_experience": 2.0,
            "skills_listed": ["Python"],
            "career_history": [],
            "last_active": "2026-01-01",
            "education": "B.S.",
            "location": "Boston",
            "resume_text": "Alice resume text"
        }
    ]
    existing_ids = {"cand_0001"}
    job = SmartIngestJob()
    
    # 1. Ingest same candidate (Alice Smith) -> Should merge in-place
    parsed = {
        "name": "Alice Smith",
        "name_not_extracted": False,
        "current_title": "Senior Software Engineer",
        "years_experience": 4.0,
        "skills_listed": ["Python", "PyTorch"],
        "career_history": [{"title": "Senior Engineer", "company": "Tech", "start_date": "2024-01-01", "end_date": "Present"}],
        "education": "M.S.",
        "location": "Boston, MA",
        "last_active": "2026-05-01"
    }
    
    record = _build_candidate_record(parsed, "Alice new resume", "alice.pdf", existing, existing_ids, job)
    
    # Assert return is None (merged in-place)
    assert record is None
    # Assert Alice in existing pool is updated
    alice = existing[0]
    assert alice["current_title"] == "Senior Software Engineer"
    assert alice["years_experience"] == 4.0
    assert set(alice["skills_listed"]) == {"Python", "PyTorch"}
    assert alice["education"] == "M.S."
    assert alice["location"] == "Boston, MA"
    assert len(alice["career_history"]) == 1
    
    # 2. Ingest a new candidate -> Should create new record with standard 4-digit ID
    parsed_new = {
        "name": "Charlie Brown",
        "name_not_extracted": False,
        "current_title": "Developer",
        "years_experience": 1.0,
        "skills_listed": ["Java"],
        "career_history": []
    }
    record_new = _build_candidate_record(parsed_new, "Charlie resume text", "charlie.pdf", existing, existing_ids, job)
    assert record_new is not None
    assert record_new["candidate_id"] == "cand_0002"
    assert record_new["name"] == "Charlie Brown"

def test_api_endpoints_mock():
    from talent_radar.app import api_candidate_similar, api_candidate_outreach, OutreachRequest
    from fastapi import HTTPException
    
    # 1. Test Similar Candidate cloner endpoint with invalid candidate
    with pytest.raises(HTTPException) as exc_info:
        api_candidate_similar("invalid_id")
    assert exc_info.value.status_code == 404
    
    # 2. Test Outreach email generator endpoint with invalid candidate
    req = OutreachRequest(job_description="Looking for Python developer", tone="casual")
    with pytest.raises(HTTPException) as exc_info:
        api_candidate_outreach("invalid_id", req)
    assert exc_info.value.status_code == 404
