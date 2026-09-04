import datetime
import pytest
from talent_radar.scorer import (
    CandidateScorer,
    infer_seniority_level,
    calculate_years_span,
    detect_duplicate_content,
    TODAY
)
from talent_radar.llm_parser import validate_name, extract_name_from_filename

def test_validate_name():
    # Valid names
    assert validate_name("John Doe") is True
    assert validate_name("Alice Smith") is True
    assert validate_name("Alan Turing") is True
    
    # Placeholder names / generic markers
    assert validate_name("Not specified") is False
    assert validate_name("Unknown") is False
    assert validate_name("N/A") is False
    assert validate_name("None") is False
    assert validate_name("Unknown Candidate") is False
    assert validate_name("null") is False
    assert validate_name("n.a.") is False
    
    # Name matches title
    assert validate_name("Software Engineer", "Software Engineer") is False
    assert validate_name("lead architect", "Lead Architect") is False
    
    # Job indicators / titles exceeding 50% words
    assert validate_name("Senior Software Engineer") is False
    assert validate_name("CTO") is False
    assert validate_name("Intern") is False
    
    # Combined job titles containing separators
    assert validate_name("Developer/Architect") is False
    assert validate_name("SRE | DevOps") is False

def test_extract_name_from_filename():
    assert extract_name_from_filename("John_Doe_Resume.pdf") == "John Doe"
    assert extract_name_from_filename("cv_jane_smith_2026.docx") == "Jane Smith"
    assert extract_name_from_filename("Ansh_Surana.txt") == "Ansh Surana"
    assert extract_name_from_filename("resume.pdf") == ""
    assert extract_name_from_filename("cv_2026.docx") == ""
    assert extract_name_from_filename("CTO_profile.pdf") == ""
    assert extract_name_from_filename("") == ""

def test_infer_seniority_level():
    assert infer_seniority_level("Director of Engineering") == 6
    assert infer_seniority_level("Principal Architect") == 5
    assert infer_seniority_level("Staff ML Engineer") == 5
    assert infer_seniority_level("Lead Developer") == 4
    assert infer_seniority_level("Senior Python Dev") == 3
    assert infer_seniority_level("Junior Associate") == 1
    assert infer_seniority_level("Engineering Intern") == 0
    assert infer_seniority_level("Software Engineer") == 2

def test_calculate_years_span():
    # 2 years span
    assert pytest.approx(calculate_years_span("2020-01-01", "2022-01-01"), 0.05) == 2.0
    
    # Current active role ("Present" end date)
    five_years_ago = (datetime.date.today() - datetime.timedelta(days=5 * 365.25)).isoformat()
    assert pytest.approx(calculate_years_span(five_years_ago, "Present", reference_date=datetime.date.today()), 0.05) == 5.0
    
    # Fallback default for invalid dates
    assert calculate_years_span("invalid-date", "Present") == 1.0

def test_detect_duplicate_content():
    unique_text = "This is a unique sentence containing diverse words to describe high-caliber engineering skills and dynamic trajectories."
    is_dup, ratio = detect_duplicate_content(unique_text)
    assert is_dup is False
    assert ratio == 0.0
    
    # Highly repetitive content with exactly 200 character pattern length
    pattern = "This is a chunk of text that is precisely two hundred characters in length. It repeats perfectly so that when chunked into slices of two hundred characters, every single chunk is exactly identical.   "
    pattern = pattern[:200]
    assert len(pattern) == 200
    
    repeated_chunk = pattern * 10
    is_dup, ratio = detect_duplicate_content(repeated_chunk)
    assert is_dup is True
    assert ratio == 0.9

def test_scorer_composite_and_normalization():
    candidates = [
        {
            "candidate_id": "cand_001",
            "name": "Jane Doe",
            "current_title": "Senior Machine Learning Engineer",
            "years_experience": 8.0,
            "career_history": [
                {"title": "Senior ML Engineer", "company": "Tech Corp", "start_date": "2022-01-01", "end_date": "Present"},
                {"title": "ML Engineer", "company": "Startup Inc", "start_date": "2018-01-01", "end_date": "2022-01-01"}
            ],
            "skills_listed": ["Python", "PyTorch", "Transformers", "SQL"],
            "last_active": datetime.date.today().isoformat(),
            "education": "M.S. in Computer Science",
            "location": "San Francisco, CA",
            "resume_text": "Experienced Senior Machine Learning Engineer specialized in PyTorch, NLP Transformers, and scalable inference architectures."
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
            "last_active": (datetime.date.today() - datetime.timedelta(days=10)).isoformat(),
            "education": "B.A. in History",
            "location": "Boston, MA",
            "resume_text": "Front-end developer focused on HTML, CSS, and modern JavaScript frameworks."
        }
    ]
    
    # Set different semantic depth scores to test normalization
    candidates[0]["semantic_depth_score"] = 0.90
    candidates[1]["semantic_depth_score"] = 0.50
    
    scorer = CandidateScorer(seniority_level="Senior", target_keywords=["PyTorch", "Transformers", "Python"])
    scored = scorer.score_candidates(candidates)
    
    # The highest raw semantic depth score should normalize to 1.0, lowest to 0.0
    assert scored[0]["raw_semantic_score"] == 0.90
    assert scored[0]["semantic_depth_score"] == 1.0
    
    assert scored[1]["raw_semantic_score"] == 0.50
    assert scored[1]["semantic_depth_score"] == 0.0
    
    # Jane has CS degree -> education bonus of +5 should be awarded
    assert scored[0]["education_bonus"] == 5.0
    # Bob has History degree -> no education bonus
    assert scored[1]["education_bonus"] == 0.0

def test_scorer_keyword_stuffing_guardrail():
    # Candidate with 20+ skills but very short resume text (under 200 words)
    stuffer = {
        "candidate_id": "cand_stuffer",
        "name": "Keyword Stuffer",
        "current_title": "DevOps Engineer",
        "years_experience": 5.0,
        "career_history": [
            {"title": "DevOps Engineer", "company": "Cloud Corp", "start_date": "2023-01-01", "end_date": "Present"},
            {"title": "SysAdmin", "company": "IT Inc", "start_date": "2021-01-01", "end_date": "2023-01-01"}
        ],
        "skills_listed": [f"Skill{i}" for i in range(25)], # 25 skills
        "last_active": datetime.date.today().isoformat(),
        "education": "B.S. in Computer Science",
        "location": "Remote",
        "resume_text": "DevOps engineer. Python AWS Docker Kubernetes CI/CD Bash Linux Git Jenkins Terraform Ansible Cloud." # Very short!
    }
    
    stuffer["semantic_depth_score"] = 0.80
    
    scorer = CandidateScorer(seniority_level="Senior")
    scored = scorer.score_candidates([stuffer])
    
    assert scored[0]["guardrail_keyword_penalty_applied"] is True
    # The semantic depth score should be penalized (original * 0.85)
    # Since there's only 1 candidate, max-min range is 0. But wait, in scorer.py,
    # if s_range is 0, normalized_s is set to 1.0. So semantic_depth_score becomes 1.0,
    # semantic_score is round(1.0 * 100 * 0.85, 1) = 85.0.
    assert scored[0]["semantic_score"] == 85.0

def test_scorer_seniority_mismatch_guardrail():
    # Seniority target is "Senior" or higher, but candidate has < 2 years experience
    junior_candidate = {
        "candidate_id": "cand_junior",
        "name": "Fast Learner",
        "current_title": "Junior Developer",
        "years_experience": 1.2, # < 2.0 years
        "career_history": [
            {"title": "Junior Developer", "company": "Small Shop", "start_date": "2025-01-01", "end_date": "Present"}
        ],
        "skills_listed": ["Python", "PyTorch"],
        "last_active": datetime.date.today().isoformat(),
        "education": "B.S. in Computer Science",
        "location": "Remote",
        "resume_text": "Ambitious junior developer skilled in basic Python programming and deep learning packages."
    }
    
    junior_candidate["semantic_depth_score"] = 0.95
    
    scorer = CandidateScorer(seniority_level="Senior")
    scored = scorer.score_candidates([junior_candidate])
    
    assert scored[0]["guardrail_seniority_cap_applied"] is True
    # Composite score must be capped at 65.0
    assert scored[0]["final_score"] <= 65.0

def test_scorer_tie_breaking_sorting():
    # Two candidates with very similar composite scores (within 0.5)
    # The tie-breaker should choose the one with the higher freshness_score (descending)
    cand_dormant = {
        "candidate_id": "cand_dormant",
        "name": "Dormant Match",
        "current_title": "Software Engineer",
        "years_experience": 5.0,
        "career_history": [
            {"title": "Engineer", "company": "Old Corp", "start_date": "2021-01-01", "end_date": "2025-01-01"}
        ],
        "skills_listed": ["Python", "Django"],
        "last_active": "2024-01-01", # Dormant
        "education": "Self-taught",
        "location": "Austin, TX",
        "resume_text": "Python Django backend software engineer with multiple years of production experience."
    }
    
    cand_active = {
        "candidate_id": "cand_active",
        "name": "Active Match",
        "current_title": "Software Engineer",
        "years_experience": 5.0,
        "career_history": [
            {"title": "Engineer", "company": "New Corp", "start_date": "2021-01-01", "end_date": "2025-01-01"}
        ],
        "skills_listed": ["Python", "Django"],
        "last_active": datetime.date.today().isoformat(), # Active Now
        "education": "Self-taught",
        "location": "Austin, TX",
        "resume_text": "Python Django backend software engineer with multiple years of production experience."
    }
    
    # Add a third candidate to prevent s_range = 0 or full scaling to 1.0 and 0.0
    cand_dummy = {
        "candidate_id": "cand_dummy",
        "name": "Dummy Candidate",
        "current_title": "Intern",
        "years_experience": 1.0,
        "career_history": [
            {"title": "Intern", "company": "Co", "start_date": "2024-01-01", "end_date": "Present"}
        ],
        "skills_listed": ["None"],
        "last_active": datetime.date.today().isoformat(),
        "education": "None",
        "location": "Remote",
        "resume_text": "Dummy candidate for min-max alignment."
    }
    
    # Assign specific semantic depth scores so they result in identical composite scores
    cand_dormant["semantic_depth_score"] = 0.80
    cand_active["semantic_depth_score"] = 0.60
    cand_dummy["semantic_depth_score"] = 0.00
    
    scorer = CandidateScorer(seniority_level="Mid")
    scored = scorer.score_candidates([cand_dormant, cand_active, cand_dummy])
    
    # Check that final scores are identical or extremely close
    diff = abs(scored[0]["final_score"] - scored[1]["final_score"])
    assert diff <= 0.5
    
    # The active candidate must be ranked first due to the freshness tie-breaker
    assert scored[0]["candidate_id"] == "cand_active"
    assert scored[1]["candidate_id"] == "cand_dormant"

def test_skills_gap_matching():
    candidate = {
        "candidate_id": "cand_gap",
        "name": "Gap Candidate",
        "current_title": "Backend Developer",
        "years_experience": 4.0,
        "career_history": [
            {"title": "Backend Dev", "company": "API Inc", "start_date": "2022-01-01", "end_date": "Present"}
        ],
        "skills_listed": ["Python", "PostgreSQL", "FastAPI", "Docker"],
        "last_active": datetime.date.today().isoformat(),
        "education": "BS in CS",
        "location": "New York, NY",
        "resume_text": "FastAPI python backend dev with PostgreSQL and Docker deployment experience."
    }
    
    candidate["semantic_depth_score"] = 0.70
    
    # Target keywords are: ["Python", "FastAPI", "Kubernetes", "Redis"]
    scorer = CandidateScorer(seniority_level="Mid", target_keywords=["Python", "FastAPI", "Kubernetes", "Redis"])
    scored = scorer.score_candidates([candidate])
    
    assert "Python" in scored[0]["matched_skills"]
    assert "FastAPI" in scored[0]["matched_skills"]
    assert "Kubernetes" in scored[0]["missing_skills"]
    assert "Redis" in scored[0]["missing_skills"]

def test_sector_specific_education_bonus():
    cand_cs = {
        "candidate_id": "cand_cs",
        "name": "Tech Graduate",
        "current_title": "Software Engineer",
        "years_experience": 3.0,
        "career_history": [{"title": "SE", "company": "Co", "start_date": "2023-01-01", "end_date": "Present"}],
        "skills_listed": ["Python"],
        "last_active": datetime.date.today().isoformat(),
        "education": "B.S. in Computer Science",
        "location": "Remote",
        "resume_text": "Experienced coder."
    }
    
    cand_law = {
        "candidate_id": "cand_law",
        "name": "Legal Practitioner",
        "current_title": "Corporate Counsel",
        "years_experience": 3.0,
        "career_history": [{"title": "Counsel", "company": "Co", "start_date": "2023-01-01", "end_date": "Present"}],
        "skills_listed": ["Contracts"],
        "last_active": datetime.date.today().isoformat(),
        "education": "Juris Doctor (J.D.)",
        "location": "Remote",
        "resume_text": "Experienced corporate law practitioner."
    }
    
    cand_cs["semantic_depth_score"] = 0.5
    cand_law["semantic_depth_score"] = 0.5
    
    # 1. Tech Sector Evaluation
    scorer_tech = CandidateScorer(seniority_level="Mid", sector="TECH")
    scored_tech = scorer_tech.score_candidates([cand_cs, cand_law])
    
    # In TECH sector: CS grad gets +5.0 bonus, Law grad gets 0.0
    cand_cs_scored = next(c for c in scored_tech if c["candidate_id"] == "cand_cs")
    cand_law_scored = next(c for c in scored_tech if c["candidate_id"] == "cand_law")
    assert cand_cs_scored["education_bonus"] == 5.0
    assert cand_law_scored["education_bonus"] == 0.0
    
    # 2. Legal Sector Evaluation
    scorer_legal = CandidateScorer(seniority_level="Mid", sector="LEGAL")
    scored_legal = scorer_legal.score_candidates([cand_cs, cand_law])
    
    # In LEGAL sector: Law grad gets +5.0 bonus, CS grad gets 0.0
    cand_cs_scored_l = next(c for c in scored_legal if c["candidate_id"] == "cand_cs")
    cand_law_scored_l = next(c for c in scored_legal if c["candidate_id"] == "cand_law")
    assert cand_cs_scored_l["education_bonus"] == 0.0
    assert cand_law_scored_l["education_bonus"] == 5.0

def test_scorer_sector_reasoning_sentence():
    # Setup candidate with elite score to trigger the leadership profiling reasoning
    elite_cand = {
        "candidate_id": "cand_elite",
        "name": "Alex Mercer",
        "current_title": "Department Head",
        "years_experience": 8.0,
        "career_history": [
            {"title": "Department Head", "company": "Global Corp", "start_date": "2020-01-01", "end_date": "Present"},
            {"title": "Associate", "company": "Local Shop", "start_date": "2018-01-01", "end_date": "2020-01-01"}
        ],
        "skills_listed": ["Leadership", "Strategy"],
        "last_active": datetime.date.today().isoformat(),
        "education": "Master Degree",
        "location": "New York, NY",
        "resume_text": "Highly accomplished executive with extensive leadership and operational capability."
    }
    
    # 1. Evaluate elite candidate under LEGAL sector
    scorer_legal = CandidateScorer(seniority_level="Senior", sector="LEGAL")
    elite_cand["semantic_depth_score"] = 0.95
    scored_legal = scorer_legal.score_candidates([elite_cand])
    assert "legal professional profile" in scored_legal[0]["reasoning"]
    assert "tech lead" not in scored_legal[0]["reasoning"]

    # 2. Evaluate elite candidate under FIN sector
    scorer_fin = CandidateScorer(seniority_level="Senior", sector="FIN")
    elite_cand["semantic_depth_score"] = 0.95
    scored_fin = scorer_fin.score_candidates([elite_cand])
    assert "finance leader profile" in scored_fin[0]["reasoning"]
    assert "tech lead" not in scored_fin[0]["reasoning"]

    # 3. Evaluate elite candidate under TECH sector
    scorer_tech = CandidateScorer(seniority_level="Senior", sector="TECH")
    elite_cand["semantic_depth_score"] = 0.95
    scored_tech = scorer_tech.score_candidates([elite_cand])
    assert "tech lead profile" in scored_tech[0]["reasoning"]

