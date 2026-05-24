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
        
        # 2. Score candidates individually
        for cand in candidates:
            # Score Career Velocity
            if len(cand["career_history"]) < 2:
                # Baseline baseline: normalized velocity_score is 0.3
                velocity_score = 0.3
            else:
                raw_v = cand["velocity_raw"]
                if v_range > 0:
                    velocity_score = (raw_v - min_v) / v_range
                else:
                    velocity_score = 1.0
                    
            cand["career_velocity_score"] = float(velocity_score)
            
            # Score Profile Freshness
            last_active = cand["last_active"]
            if last_active is None:
                # Guardrail: Null profile update sets freshness_score = 0.2
                freshness_score = 0.2
                days_since_update = 999
            else:
                try:
                    active_y, active_m, active_d = map(int, last_active.split("-"))
                    active_date = datetime.date(active_y, active_m, active_d)
                    days_since_update = (TODAY - active_date).days
                except Exception:
                    days_since_update = 999
                    
                freshness_raw = max(0.0, 1.0 - (days_since_update / 365.0))
                # Bonus for candidate updated in last 7 days
                if days_since_update <= 7:
                    freshness_raw = min(1.0, freshness_raw + 0.10)
                freshness_score = freshness_raw
                
            cand["freshness_score"] = float(freshness_score)
            
            # Freshness label
            if days_since_update <= 7:
                cand["freshness_label"] = "Active Now"
            elif days_since_update <= 60:
                cand["freshness_label"] = "Recent"
            else:
                cand["freshness_label"] = "Dormant"
                
            # Staleness Warning
            if days_since_update > 730:
                cand["staleness_warning"] = "Dormant (2+ years)"
            elif days_since_update > 365:
                cand["staleness_warning"] = "Inactive (1+ years)"
            else:
                cand["staleness_warning"] = None
                
            # Score Semantic Depth (with Keyword Stuffer Guardrail)
            semantic_score = cand["semantic_depth_score"] * 100
            
            # Guardrail: Keyword Overfitters
            # If 20+ skills but resume text < 200 words, penalize semantic_score by 0.85
            resume_txt_safe = cand.get("resume_text", "") or ""
            word_count = len(resume_txt_safe.split())
            is_stuffer = len(cand.get("skills_listed", [])) >= 20 and word_count < 200
            
            if is_stuffer:
                semantic_score *= 0.85
                cand["semantic_depth_score"] = semantic_score / 100.0
                cand["guardrail_keyword_penalty_applied"] = True
            else:
                cand["guardrail_keyword_penalty_applied"] = False
                
            # Guardrail: Duplicate Content Detection
            if "flags" not in cand:
                cand["flags"] = []
                
            is_duplicate, ratio = detect_duplicate_content(cand["resume_text"])
            if is_duplicate:
                semantic_score *= (1 - ratio * 0.5)
                cand["semantic_depth_score"] = semantic_score / 100.0
                cand["flags"].append(f"⚠ Duplicate content detected ({ratio*100:.0f}% repeated)")
                
            cand["semantic_score"] = round(semantic_score, 1)
            
            # Scale Velocity score for display (0-10 scale)
            cand["velocity_score"] = round(velocity_score * 10.0, 1)
            
            # Weighted Composite Formula:
            composite_score = (self.semantic_weight * cand["semantic_depth_score"]) + (self.velocity_weight * cand["career_velocity_score"]) + (self.freshness_weight * cand["freshness_score"])
            final_score = composite_score * 100
            
            # Score Education Bonus
            edu_bonus = 0.0
            edu_str = cand.get("education", "")
            if isinstance(edu_str, str) and edu_str:
                edu_lower = edu_str.lower()
                
                # Define sector-specific education keywords to ensure zero bias toward Tech in non-Tech sectors
                SECTOR_EDU_KEYWORDS = {
                    "TECH": ["computer science", "software engineering", "data science", "artificial intelligence", "machine learning", "information technology", "electrical engineering"],
                    "FIN": ["finance", "economics", "accounting", "mba", "business administration", "chartered financial analyst", "financial engineering"],
                    "HEALTH": ["medical", "medicine", "nursing", "biology", "pharmacy", "pharmacology", "biochemistry", "health studies", "clinical science"],
                    "LEGAL": ["law", "legal", "juris doctor", "legal studies", "criminology", "paralegal studies"],
                    "REAL": ["real estate", "property development", "urban planning", "construction management", "architecture"],
                    "MANU": ["mechanical engineering", "industrial engineering", "manufacturing", "chemical engineering", "materials science", "systems engineering"],
                    "COMM": ["marketing", "retail", "business administration", "commerce", "supply chain management", "mba"],
                    "LOGI": ["logistics", "supply chain", "operations research", "transportation management", "industrial engineering"],
                    "MEDIA": ["journalism", "media studies", "communications", "graphic design", "fine arts", "cinema", "broadcasting", "creative writing"],
                    "ENERGY": ["petroleum engineering", "renewable energy", "electrical engineering", "environmental science", "geology", "nuclear engineering"],
                    "EDU": ["education", "teaching", "curriculum design", "pedagogy", "educational leadership"],
                    "GOV": ["public policy", "political science", "public administration", "international relations", "government studies"]
                }
                
                relevant_keywords = SECTOR_EDU_KEYWORDS.get(self.sector, SECTOR_EDU_KEYWORDS["TECH"])
                
                has_special_match = False
                if self.sector == "TECH":
                    has_special_match = bool(re.search(r'\bcs\b', edu_lower))
                elif self.sector == "FIN":
                    has_special_match = bool(re.search(r'\bcfa\b|\bmba\b', edu_lower))
                elif self.sector == "HEALTH":
                    has_special_match = bool(re.search(r'\bmd\b|\bm\.d\.\b|\bbsn\b', edu_lower))
                elif self.sector == "LEGAL":
                    has_special_match = bool(re.search(r'\bjd\b|\bj\.d\.\b|\bllm\b|\bll\.m\.\b', edu_lower))
                
                if any(kw in edu_lower for kw in relevant_keywords) or has_special_match:
                    edu_bonus = 5.0
            
            cand["education_bonus"] = edu_bonus
            final_score = min(100.0, final_score + edu_bonus)
            
            # Guardrail: Seniority Mismatch
            # If JD seniority is Senior+ and candidate has < 2 years experience, cap final_score at 65/100
            is_seniority_mismatch = self.seniority_level in ["Senior", "Lead", "Principal"] and cand["years_experience"] < 2.0
            
            if is_seniority_mismatch:
                final_score = min(65.0, final_score)
                cand["guardrail_seniority_cap_applied"] = True
            else:
                cand["guardrail_seniority_cap_applied"] = False
                
            cand["final_score"] = round(final_score, 1)
            
            # Status Label Assignment
            if cand["final_score"] >= 72 and cand["velocity_score"] >= 3.0:
                cand["status_label"] = "Top Hidden Gem 🚀"
            elif cand["final_score"] >= 65:
                cand["status_label"] = "Solid Match 🏆"
            elif cand["final_score"] >= 55:
                cand["status_label"] = "Potential Fit ⭐"
            else:
                cand["status_label"] = "Longshot"
                
            # Perform Backend Skills Gap Matching
            matched_skills = []
            missing_skills = []
            if self.target_keywords:
                cand_skills_lower = [s.lower().strip() for s in cand.get("skills_listed", [])]
                for kw in self.target_keywords:
                    kw_clean = kw.strip()
                    kw_lower = kw_clean.lower()
                    matched = False
                    for cs in cand_skills_lower:
                        if kw_lower == cs or (len(kw_lower) > 3 and (kw_lower in cs or cs in kw_lower)):
                            matched = True
                            if cs not in matched_skills:
                                # Keep display case
                                matched_skills.append(kw_clean)
                            break
                    if not matched:
                        if kw_clean not in missing_skills:
                            missing_skills.append(kw_clean)
                            
            cand["matched_skills"] = matched_skills
            cand["missing_skills"] = missing_skills

            # Formulate human-readable reasoning
            cand["reasoning"] = generate_reasoning_sentence(cand, is_stuffer, is_seniority_mismatch, self.target_keywords, self.sector)
            
        # 3. Tie-breaking Sorting
        # Sort descending using cmp_to_key. Ranks higher profile_freshness first if final scores are within 0.5
        def compare_candidates(c1, c2):
            diff = c1["final_score"] - c2["final_score"]
            if abs(diff) <= 0.5:
                # Rank higher freshness_score first (descending)
                f_diff = c1["freshness_score"] - c2["freshness_score"]
                if f_diff > 0:
                    return -1 # c1 comes before c2
                elif f_diff < 0:
                    return 1  # c2 comes before c1
            # Standard final score sort (descending)
            if diff > 0:
                return -1
            elif diff < 0:
                return 1
            return 0
            
        sorted_candidates = sorted(candidates, key=cmp_to_key(compare_candidates))
        
        # Add a debug print after scoring 5 candidates
        print("\n--- DEBUG COMPOSITE SCORE SAMPLE (TOP 5) ---")
        for c in sorted_candidates[:5]:
            print(f"Name: {c['name']} | Semantic: {c['semantic_depth_score']:.4f} | Velocity: {c['career_velocity_score']:.4f} | Freshness: {c['freshness_score']:.4f} | Composite: {c['final_score']/100:.4f} | Final: {c['final_score']}")
            
        return sorted_candidates
 
def generate_reasoning_sentence(cand, is_stuffer, is_seniority_mismatch, target_keywords=None, sector="TECH"):
    name = cand["name"]
    title = cand["current_title"]
    skills_count = len(cand["skills_listed"])
    
    is_dormant_high_scorer = cand.get("semantic_score", 0.0) >= 85.0 and cand.get("freshness_label") == "Dormant"
    
    history = cand.get("career_history", [])
    if history:
        earliest_role = history[-1].get("title", "Junior Role")
        experience = cand.get("years_experience", 0.0)
        trajectory_str = f"Progressed from {earliest_role} to {title} over {experience} years."
    else:
        trajectory_str = f"Demonstrates {cand.get('years_experience', 0.0)} years of experience as {title}."

    matched_skills = cand.get("matched_skills", [])
    if matched_skills:
        skills_str = f" Aligned with core requirements in {', '.join(matched_skills[:3])}."
    else:
        skills_str = ""

    # Map dynamic sector terminology to ensure no tech bias
    sector = sector.upper().strip() if sector else "TECH"
    if sector == "TECH":
        sector_fit = "technical fit"
        leader_profile = "tech lead profile"
    elif sector == "FIN":
        sector_fit = "financial fit"
        leader_profile = "finance leader profile"
    elif sector == "HEALTH":
        sector_fit = "healthcare fit"
        leader_profile = "healthcare leader profile"
    elif sector == "LEGAL":
        sector_fit = "legal fit"
        leader_profile = "legal professional profile"
    else:
        sector_fit = "domain fit"
        leader_profile = "industry leader profile"

    if is_seniority_mismatch:
        return f"{name} presents {cand['years_experience']} years of experience for a senior role; capped at 65 due to lack of deep background."
    elif is_stuffer:
        return f"{name} lists a dense skill set ({skills_count} skills) with a very brief resume text; penalized for potential keyword stuffing."
    elif is_dormant_high_scorer:
        return f"Exceptional {sector_fit} ({cand.get('semantic_score', 0.0)}% semantic match) — but profile dormant. High skill match, low active intent signal."
    elif cand["final_score"] >= 72 and cand.get("velocity_score", 0.0) >= 3.0:
        return f"Elite candidate showing exceptional career velocity. {trajectory_str}{skills_str} High-caliber {leader_profile}."
    elif cand["final_score"] >= 65:
        return f"Strong matching profile. {trajectory_str}{skills_str} Steady career progression and solid competence."
    elif cand["final_score"] >= 55:
        return f"Satisfactory skills align with core stack, though profile freshness or experience velocity remains moderate.{skills_str}"
    else:
        return f"Insufficient contextual relevance or dormant profile activity makes this candidate a low-priority match."

