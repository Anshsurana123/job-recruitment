"""
Calibrated honeypot and anomaly detection filters.
Enforces physical, chronological, and platform validity constraints.
"""

import datetime
from typing import Dict, List, Optional, Tuple, Any

from ..dates import parse_date_string, parse_date


def detect_honeypots(
    career_history: List[Dict[str, Any]],
    skills: List[Dict[str, Any]],
    reference_date: datetime.date,
    foundation_dates: Dict[str, datetime.date],
    duration_tolerance_months: float = 3.0
) -> Tuple[bool, str]:
    """
    Evaluates candidate record against strict chronological and logical honeypot rules.
    Returns (is_honeypot, disqualification_reason).
    """
    # Rule 1: Job Claimed Duration Mismatch
    for job in career_history:
        claimed_months = job.get("duration_months", 0)
        s_date = parse_date_string(job.get("start_date"), reference_date=reference_date)
        e_date = parse_date_string(job.get("end_date"), reference_date=reference_date)
        if s_date:
            actual_end = e_date if e_date else reference_date
            actual_months = (actual_end - s_date).days / 30.44
            if claimed_months > actual_months + duration_tolerance_months:
                return True, f"Job claimed duration mismatch ({claimed_months}mo vs {actual_months:.1f}mo)"

    # Rule 2: Company Foundation Date Violation
    for job in career_history:
        comp_lower = str(job.get("company", "")).strip().lower()
        s_date = parse_date_string(job.get("start_date"), reference_date=reference_date)
        if s_date:
            for comp_key, f_date in foundation_dates.items():
                if comp_key in comp_lower and s_date < f_date:
                    return True, f"{comp_key.capitalize()} start date {s_date} before foundation {f_date.strftime('%B %Y')}"

    # Rule 3: Expert Proficiency with Zero Duration
    for s in skills:
        prof = str(s.get("proficiency", "")).strip().lower()
        dur = s.get("duration_months", 0)
        if prof == "expert" and dur == 0:
            return True, f"Expert skill '{s.get('name')}' claimed with zero duration"

    return False, ""
