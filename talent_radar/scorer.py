import datetime
import json
import re
from functools import cmp_to_key

try:
    from talent_radar.llm_parser import validate_name
except ImportError:
    from llm_parser import validate_name

# Today's reference date in the hackathon ecosystem
TODAY = datetime.date.today()

def detect_duplicate_content(resume_text):
    # Split into chunks of 200 chars, check how many are repeated
    chunks = [resume_text[i:i+200] for i in range(0, len(resume_text), 200)]
    unique = set(chunks)
    duplication_ratio = 1 - (len(unique) / max(len(chunks), 1))
    if duplication_ratio > 0.4:   # 40%+ content is duplicated
        return True, duplication_ratio
    return False, 0.0

SENIORITY_MAP = {
    "intern": 0,
    "junior": 1,
    "associate": 1,
    "mid": 2,
    "senior": 3,
    "lead": 4,
    "principal": 5,
    "staff": 5,
    "director": 6
}

def infer_seniority_level(title):
    title_lower = title.lower()
    if "director" in title_lower:
        return SENIORITY_MAP["director"]
    elif "principal" in title_lower:
        return SENIORITY_MAP["principal"]
    elif "staff" in title_lower:
        return SENIORITY_MAP["staff"]
    elif "lead" in title_lower or "head" in title_lower:
        return SENIORITY_MAP["lead"]
    elif "senior" in title_lower or "sr" in title_lower:
        return SENIORITY_MAP["senior"]
    elif "junior" in title_lower or "jr" in title_lower:
        return SENIORITY_MAP["junior"]
    elif "intern" in title_lower or "co-op" in title_lower:
        return SENIORITY_MAP["intern"]
    elif "associate" in title_lower:
        return SENIORITY_MAP["associate"]
    else:
        # Default for engineer/developer or standard roles
        return SENIORITY_MAP["mid"]

def parse_date_string(date_str):
    if not date_str:
        return None
    date_clean = str(date_str).strip().strip('"').strip("'").strip()
    if date_clean.lower() in ("present", "current", "now", "ongoing", "none", "null", ""):
        return TODAY
        
    # 1. Try standard ISO formats
    # Try YYYY-MM-DD
    match = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", date_clean)
    if match:
        try:
            return datetime.date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            pass
            
    # Try YYYY-MM
    match = re.match(r"^(\d{4})-(\d{1,2})$", date_clean)
    if match:
        try:
            return datetime.date(int(match.group(1)), int(match.group(2)), 1)
        except ValueError:
            pass

    # Try YYYY
    match = re.match(r"^(\d{4})$", date_clean)
    if match:
        try:
            return datetime.date(int(match.group(1)), 1, 1)
        except ValueError:
            pass

    # 2. Try slashing formats (MM/DD/YYYY, MM/YYYY, YYYY/MM/DD)
    # Try MM/DD/YYYY
    match = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", date_clean)
    if match:
        try:
            return datetime.date(int(match.group(3)), int(match.group(1)), int(match.group(2)))
        except ValueError:
            pass

    # Try YYYY/MM/DD
    match = re.match(r"^(\d{4})/(\d{1,2})/(\d{1,2})$", date_clean)
    if match:
        try:
            return datetime.date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            pass

    # Try MM/YYYY
    match = re.match(r"^(\d{1,2})/(\d{4})$", date_clean)
    if match:
        try:
            return datetime.date(int(match.group(2)), int(match.group(1)), 1)
        except ValueError:
            pass

    # 3. Try English month formats (e.g. October 2021, Oct 2021, 10 Oct 2021)
    months_map = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        "january": 1, "february": 2, "march": 3, "april": 4, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
    }
    
    # Format: "Month YYYY" or "Month, YYYY"
    match = re.match(r"^([a-zA-Z]+)[,\s]+(\d{4})$", date_clean)
    if match:
        m_name = match.group(1).lower()[:3]
        if m_name in months_map:
            try:
                return datetime.date(int(match.group(2)), months_map[m_name], 1)
            except ValueError:
                pass

    # Format: "DD Month YYYY"
    match = re.match(r"^(\d{1,2})\s+([a-zA-Z]+)\s+(\d{4})$", date_clean)
    if match:
        m_name = match.group(2).lower()[:3]
        if m_name in months_map:
            try:
                return datetime.date(int(match.group(3)), months_map[m_name], int(match.group(1)))
            except ValueError:
                pass

    return None

