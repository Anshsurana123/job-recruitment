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

def calculate_years_span(start_date_str, end_date_str):
    try:
        start_date = parse_date_string(start_date_str)
        end_date = parse_date_string(end_date_str)
        
        if not start_date:
            return 1.0
        if not end_date:
            end_date = TODAY
            
        span_days = (end_date - start_date).days
        return max(0.0, span_days / 365.25)
    except Exception:
        return 1.0 # Default fallback if date parsing fails
class CandidateScorer:
    def __init__(self, seniority_level="Senior", target_keywords=None, sector="TECH", semantic_weight=0.60, velocity_weight=0.25, freshness_weight=0.15):
        self.seniority_level = seniority_level.title()
        self.target_keywords = target_keywords or []
        self.sector = sector.upper().strip() if sector else "TECH"
        self.semantic_weight = semantic_weight
        self.velocity_weight = velocity_weight
        self.freshness_weight = freshness_weight

    def score_candidates(self, candidates):
        print("Executing Step 4: Career Momentum Calculator and Guardrails...")
        if not candidates:
            return []
            
        # Min-max normalization for semantic_depth_score across retrieved pool to utilize full [0, 1] range
        semantic_raws = [c.get("semantic_depth_score", 0.0) for c in candidates]
        min_s = min(semantic_raws) if semantic_raws else 0.0
        max_s = max(semantic_raws) if semantic_raws else 1.0
        s_range = max_s - min_s
        
        for cand in candidates:
            raw_s = cand.get("semantic_depth_score", 0.0)
            cand["raw_semantic_score"] = float(raw_s)
            if s_range > 0:
                normalized_s = (raw_s - min_s) / s_range
            else:
                normalized_s = 1.0
            cand["semantic_depth_score"] = float(normalized_s)
            
        # Debug print to verify semantic depth scores are properly normalized
        semantic_scores_sample = [c.get("semantic_depth_score", 0.0) for c in candidates[:5]]
        print(f"semantic scores sample: {semantic_scores_sample}")
            
        # 1. Calculate raw Career Velocity scores for all candidates
        velocity_raws = []
        for cand in candidates:
            # Retrospective name validation
            name = cand.get("name", "")
            title = cand.get("current_title", "")
            if not validate_name(name, title) or name == "Unknown Candidate":
                cand["name_not_extracted"] = True
                cand["name"] = "Unknown Candidate"
            else:
                cand["name_not_extracted"] = cand.get("name_not_extracted", False)

            history = cand["career_history"]
            
            # Guardrail: If career_history has fewer than 2 entries, set velocity_score = 0.3 as baseline
            if len(history) < 2:
                cand["velocity_raw"] = None
                continue
                
            max_level = 0
            earliest_start = None
            latest_end = None
            
            for pos in history:
                level = infer_seniority_level(pos["title"])
                max_level = max(max_level, level)
                
                # track start/end dates
                s_str = pos["start_date"]
                e_str = pos["end_date"]
                
                try:
                    s_date = parse_date_string(s_str)
                    e_date = parse_date_string(e_str)
                    
                    if s_date:
                        if earliest_start is None or s_date < earliest_start:
                            earliest_start = s_date
                            
                    if e_date:
                        if latest_end is None or e_date > latest_end:
                            latest_end = e_date
                except Exception:
                    pass
                    
            if earliest_start and latest_end:
                total_years = (latest_end - earliest_start).days / 365.25
            else:
                total_years = cand["years_experience"]
                
            total_years = max(0.5, total_years) # avoid div by zero
            velocity_raw = max_level / total_years
            cand["velocity_raw"] = velocity_raw
            velocity_raws.append(velocity_raw)
            
        # Min-max normalization values for velocity
        min_v = min(velocity_raws) if velocity_raws else 0.0
        max_v = max(velocity_raws) if velocity_raws else 1.0
        v_range = max_v - min_v
        
