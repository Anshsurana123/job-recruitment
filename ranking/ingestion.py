"""
Candidate ingestion and schema normalization.
Handles nested JSON/JSONL (candidate_schema.json) and flat formats.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Union

from .models import CandidateProfile, CandidateRecord, ProcessedCandidateFields
from .leakage import compute_content_hash


def load_candidates_from_file(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Loads candidate records from a JSON array or JSONL file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Candidates file not found: {path}")

    candidates: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if content.startswith("[") and content.endswith("]"):
            try:
                candidates = json.loads(content)
            except Exception:
                pass
                
        if not candidates:
            f.seek(0)
            for line in f:
                line_str = line.strip()
                if line_str:
                    try:
                        candidates.append(json.loads(line_str))
                    except Exception:
                        pass
    return candidates


def normalize_candidate_record(raw_candidate: Dict[str, Any]) -> CandidateRecord:
    """
    Normalizes a candidate dictionary (nested or flat) into a structured CandidateRecord.
    """
    cid = raw_candidate.get("candidate_id", "UNKNOWN")
    
    # Check for nested schema vs flat schema
    if "profile" in raw_candidate and isinstance(raw_candidate["profile"], dict):
        p_dict = raw_candidate["profile"]
        profile = CandidateProfile(
            candidate_id=cid,
            current_title=p_dict.get("current_title", ""),
            headline=p_dict.get("headline", ""),
            summary=p_dict.get("summary", ""),
            years_of_experience=float(p_dict.get("years_of_experience", 0.0) or 0.0),
            location=p_dict.get("location", ""),
            country=p_dict.get("country", "India"),
            raw_data=p_dict
        )
        career = raw_candidate.get("career_history", [])
        education = raw_candidate.get("education", [])
        skills = raw_candidate.get("skills", [])
        certifications = raw_candidate.get("certifications", [])
        languages = raw_candidate.get("languages", [])
        signals = raw_candidate.get("redrob_signals", {})
        resume_text = raw_candidate.get("resume_text", "")
    else:
        # Flat schema (e.g. talent_radar/candidates.json)
        profile = CandidateProfile(
            candidate_id=cid,
            current_title=raw_candidate.get("current_title", ""),
            headline="",
            summary=raw_candidate.get("resume_text", "")[:200],
            years_of_experience=float(raw_candidate.get("years_experience", 0.0) or 0.0),
            location=raw_candidate.get("location", ""),
            country="India",
            raw_data=raw_candidate
        )
        career = raw_candidate.get("career_history", [])
        education_val = raw_candidate.get("education", "")
        education = [{"degree": str(education_val), "field_of_study": "", "institution": ""}] if education_val else []
        skills_listed = raw_candidate.get("skills_listed", [])
        skills = [{"name": s, "proficiency": "mid", "duration_months": 12} for s in skills_listed]
        certifications = []
        languages = []
        signals = raw_candidate.get("redrob_signals", {})
        if not signals and "last_active" in raw_candidate:
            signals = {
                "last_active_date": raw_candidate.get("last_active"),
                "recruiter_response_rate": 1.0,
                "notice_period_days": 30,
                "willing_to_relocate": True
            }
        resume_text = raw_candidate.get("resume_text", "")

    # Build content hash
    content_basis = resume_text if resume_text else f"{profile.current_title} {profile.summary} {' '.join(s.get('name', '') for s in skills)}"
    c_hash = compute_content_hash(content_basis)

    return CandidateRecord(
        candidate_id=cid,
        profile=profile,
        career_history=career if isinstance(career, list) else [],
        education=education if isinstance(education, list) else [],
        skills=skills if isinstance(skills, list) else [],
        certifications=certifications if isinstance(certifications, list) else [],
        languages=languages if isinstance(languages, list) else [],
        redrob_signals=signals if isinstance(signals, dict) else {},
        resume_text=resume_text,
        content_hash=c_hash
    )


def extract_processed_fields(record: CandidateRecord) -> ProcessedCandidateFields:
    """Extracts structured text fields for indexing and embedding."""
    p = record.profile
    
    parts_title = []
    if p.current_title:
        parts_title.append(p.current_title)
    if p.headline:
        parts_title.append(p.headline)
    title_headline = " ".join(parts_title)
    
    skills_str = " ".join([s.get("name", "") for s in record.skills if s.get("name")])
    career_titles = " ".join([j.get("title", "") for j in record.career_history if j.get("title")])
    summary = p.summary or ""
    career_desc = " ".join([j.get("description", "") for j in record.career_history if j.get("description")])
    
    edu_parts = []
    for edu in record.education:
        if edu.get("field_of_study"):
            edu_parts.append(edu["field_of_study"])
        if edu.get("degree"):
            edu_parts.append(edu["degree"])
    edu_str = " ".join(edu_parts)
    
    other_parts = []
    for j in record.career_history:
        if j.get("company"):
            other_parts.append(j["company"])
    for edu in record.education:
        if edu.get("institution"):
            other_parts.append(edu["institution"])
    other_str = " ".join(other_parts)

    # Build recent roles snippet
    recent_roles = []
    for job in record.career_history[:3]:
        jt = job.get("title", "")
        comp = job.get("company", "")
        desc = job.get("description", "")
        r_parts = []
        if jt:
            if comp:
                r_parts.append(f"{jt} at {comp}")
            else:
                r_parts.append(jt)
        if desc:
            clean_d = re.sub(r'\s+', ' ', desc).strip()
            r_parts.append(f"({clean_d[:150]}...)")
        if r_parts:
            recent_roles.append(" ".join(r_parts))
    recent_str = ". ".join(recent_roles)

    full_text = f"Current Title: {title_headline}. Skills: {skills_str}. Summary: {summary}. Recent: {recent_str}."

    return ProcessedCandidateFields(
        candidate_id=record.candidate_id,
        title_headline=title_headline,
        skills=skills_str,
        career_titles=career_titles,
        summary=summary,
        career_descriptions=career_desc,
        education=edu_str,
        other=other_str,
        full_text=full_text
    )
