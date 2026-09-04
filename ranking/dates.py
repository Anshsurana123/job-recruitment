"""
Date parsing, reference date resolution, and tenure calculations.
Explicitly distinguishes runtime wall-clock dates from reproducible benchmark evaluation dates.
"""

import re
import datetime
from typing import Optional, List, Dict, Tuple


MONTHS_MAP: Dict[str, int] = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    "january": 1, "february": 2, "march": 3, "april": 4, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
}

OPEN_ENDED_KEYWORDS = {"present", "current", "now", "ongoing"}
EMPTY_DATE_KEYWORDS = {"none", "null", "", "n/a", "unknown"}


def parse_date_string(date_str: Optional[str], reference_date: Optional[datetime.date] = None) -> Optional[datetime.date]:
    """
    Parses various date string formats into a datetime.date object.
    
    If the date represents an open-ended current period ('Present', 'Current', 'Now', 'Ongoing'),
    it resolves strictly to reference_date if provided, or datetime.date.today() if reference_date is None.
    """
    if not date_str:
        return None
        
    date_clean = str(date_str).strip().strip('"').strip("'").strip()
    date_lower = date_clean.lower()
    
    if date_lower in EMPTY_DATE_KEYWORDS:
        return None
        
    if date_lower in OPEN_ENDED_KEYWORDS:
        return reference_date if reference_date is not None else datetime.date.today()
        
    # 1. Standard ISO formats
    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", date_clean)
    if m:
        try:
            return datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
            
    m = re.match(r"^(\d{4})-(\d{1,2})$", date_clean)
    if m:
        try:
            return datetime.date(int(m.group(1)), int(m.group(2)), 1)
        except ValueError:
            pass

    m = re.match(r"^(\d{4})$", date_clean)
    if m:
        try:
            return datetime.date(int(m.group(1)), 1, 1)
        except ValueError:
            pass

    # 2. Slashing formats (MM/DD/YYYY, YYYY/MM/DD, MM/YYYY)
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", date_clean)
    if m:
        try:
            return datetime.date(int(m.group(3)), int(m.group(1)), int(m.group(2)))
        except ValueError:
            pass

    m = re.match(r"^(\d{4})/(\d{1,2})/(\d{1,2})$", date_clean)
    if m:
        try:
            return datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass

    m = re.match(r"^(\d{1,2})/(\d{4})$", date_clean)
    if m:
        try:
            return datetime.date(int(m.group(2)), int(m.group(1)), 1)
        except ValueError:
            pass

    # 3. English month formats (e.g. October 2021, Oct 2021, 15 October 2021)
    m = re.match(r"^([a-zA-Z]+)[,\s]+(\d{4})$", date_clean)
    if m:
        m_name = m.group(1).lower()[:3]
        if m_name in MONTHS_MAP:
            try:
                return datetime.date(int(m.group(2)), MONTHS_MAP[m_name], 1)
            except ValueError:
                pass

    m = re.match(r"^(\d{1,2})\s+([a-zA-Z]+)\s+(\d{4})$", date_clean)
    if m:
        m_name = m.group(2).lower()[:3]
        if m_name in MONTHS_MAP:
            try:
                return datetime.date(int(m.group(3)), MONTHS_MAP[m_name], int(m.group(1)))
            except ValueError:
                pass

    return None


def parse_date(d_str: Optional[str], reference_date: Optional[datetime.date] = None) -> Optional[datetime.date]:
    """Alias/wrapper supporting ISO parsing with fallback to full regex date parser."""
    if not d_str:
        return None
    d_clean = str(d_str).strip()
    if d_clean.lower() in OPEN_ENDED_KEYWORDS:
        return reference_date if reference_date is not None else datetime.date.today()
    try:
        return datetime.date.fromisoformat(d_clean)
    except Exception:
        return parse_date_string(d_str, reference_date=reference_date)


def calculate_years_span(
    start_date_str: Optional[str],
    end_date_str: Optional[str],
    reference_date: Optional[datetime.date] = None
) -> float:
    """
    Computes the tenure span in fractional years between start and end dates.
    If end_date is 'Present' (or empty/open-ended), it computes elapsed time up to reference_date.
    """
    try:
        ref_date = reference_date if reference_date is not None else datetime.date.today()
        start_date = parse_date_string(start_date_str, reference_date=ref_date)
        end_date = parse_date_string(end_date_str, reference_date=ref_date)
        
        if not start_date:
            return 1.0  # Fallback default if start date is missing
        if not end_date:
            end_date = ref_date
            
        span_days = (end_date - start_date).days
        return max(0.0, span_days / 365.25)
    except Exception:
        return 1.0


def calculate_days_active(
    last_active_str: Optional[str],
    reference_date: Optional[datetime.date] = None
) -> int:
    """
    Computes the number of days between reference_date and candidate last_active_date.
    Returns 999 if last_active is unknown or invalid.
    """
    ref_date = reference_date if reference_date is not None else datetime.date.today()
    active_date = parse_date(last_active_str, reference_date=ref_date)
    if not active_date:
        return 999
    return max(0, (ref_date - active_date).days)


def calculate_total_experience(
    career_history: List[Dict],
    reference_date: Optional[datetime.date] = None,
    merge_overlaps: bool = True
) -> float:
    """
    Calculates total years of experience across career history.
    If merge_overlaps is True, overlapping job intervals are merged to avoid
    inflating tenure for concurrent roles.
    """
    ref_date = reference_date if reference_date is not None else datetime.date.today()
    intervals: List[Tuple[datetime.date, datetime.date]] = []
    
    for job in career_history:
        s_date = parse_date_string(job.get("start_date"), reference_date=ref_date)
        e_date = parse_date_string(job.get("end_date"), reference_date=ref_date)
        if s_date:
            actual_end = e_date if e_date else ref_date
            if actual_end >= s_date:
                intervals.append((s_date, actual_end))
                
    if not intervals:
        return 0.0
        
    if not merge_overlaps:
        total_days = sum((e - s).days for s, e in intervals)
        return max(0.0, total_days / 365.25)
        
    # Sort by start date and merge overlapping ranges
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for current in intervals[1:]:
        prev_s, prev_e = merged[-1]
        if current[0] <= prev_e:
            merged[-1] = (prev_s, max(prev_e, current[1]))
        else:
            merged.append(current)
            
    total_days = sum((e - s).days for s, e in merged)
    return max(0.0, total_days / 365.25)
