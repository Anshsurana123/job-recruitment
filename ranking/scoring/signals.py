"""
Redrob behavioral signals and availability multiplier calculation.
"""

import datetime
import math
from typing import Dict, List, Optional, Any

from ..dates import calculate_days_active


def calculate_availability_multiplier(
    candidate_profile: Any,
    signals: Dict[str, Any],
    target_cities: List[str],
    reference_date: datetime.date
) -> float:
    """
    Computes a composite multiplier [0.0, 1.5+] reflecting physical availability,
    logistical match, and behavioral platform engagement.
    """
    loc = (candidate_profile.location or "").lower()
    country = (candidate_profile.country or "India").lower()
    willing_reloc = signals.get("willing_to_relocate", False)

    # 1. Location Modifier
    is_target_city = any(city in loc for city in target_cities)
    is_tier1_indian = any(city in loc for city in ["bangalore", "bengaluru", "chennai", "hyderabad", "mumbai", "kolkata", "pune", "delhi", "noida", "gurgaon"])

    if is_target_city:
        loc_modifier = 1.00
    elif is_tier1_indian:
        loc_modifier = 0.90 if willing_reloc else 0.10
    elif "india" in loc or country == "india":
        loc_modifier = 0.85 if willing_reloc else 0.10
    else:
        loc_modifier = 0.50 if willing_reloc else 0.05

    # 2. Notice Period Modifier
    notice_days = signals.get("notice_period_days", 0)
    if notice_days <= 30:
        notice_modifier = 1.00
    elif notice_days <= 60:
        notice_modifier = 0.97
    elif notice_days <= 90:
        notice_modifier = 0.80
    else:
        notice_modifier = 0.50

    # 3. Recency & Activity Modifier
    last_active_str = signals.get("last_active_date", "")
    days_active = calculate_days_active(last_active_str, reference_date=reference_date)
    if days_active <= 30:
        act_modifier = 1.05
    elif days_active <= 90:
        act_modifier = 1.00
    elif days_active <= 365:
        act_modifier = 0.85
    else:
        act_modifier = 0.50

    if signals.get("open_to_work_flag", False):
        act_modifier += 0.05
    act_modifier = min(1.10, act_modifier)

    # 4. Behavioral Responsiveness & Engagement
    beh_modifier = 1.0
    resp_rate = signals.get("recruiter_response_rate", 1.0)
    if resp_rate < 0.15:
        beh_modifier *= 0.5
    elif resp_rate < 0.50:
        beh_modifier *= (0.5 + 0.5 * (resp_rate - 0.15) / 0.35)

    avg_resp_hours = signals.get("avg_response_time_hours", 0)
    if avg_resp_hours > 0:
        if avg_resp_hours <= 24:
            pass
        elif avg_resp_hours <= 72:
            beh_modifier *= 0.95
        elif avg_resp_hours <= 120:
            beh_modifier *= 0.88
        elif avg_resp_hours <= 168:
            beh_modifier *= 0.78
        elif avg_resp_hours <= 336:
            beh_modifier *= 0.65
        else:
            beh_modifier *= 0.50

    offer_rate = signals.get("offer_acceptance_rate", -1)
    if offer_rate == -1:
        pass
    elif offer_rate < 0.15:
        beh_modifier *= 0.65
    elif offer_rate < 0.35:
        beh_modifier *= 0.82
    elif offer_rate > 0.70:
        beh_modifier *= 1.05

    work_mode = signals.get("preferred_work_mode", "flexible")
    if work_mode == "remote":
        beh_modifier *= 0.90

    int_rate = signals.get("interview_completion_rate", 1.0)
    if int_rate < 0.30:
        beh_modifier *= 0.7

    github_score = signals.get("github_activity_score", -1)
    if github_score == -1:
        beh_modifier *= 0.95
    elif github_score >= 50:
        beh_modifier *= 1.05

    saved_count = signals.get("saved_by_recruiters_30d", 0)
    if saved_count >= 5:
        beh_modifier *= 1.10
    elif saved_count >= 2:
        beh_modifier *= 1.05

    search_appearances = signals.get("search_appearance_30d", 0)
    if search_appearances > 0:
        log_ratio = math.log(search_appearances) / math.log(500.0)
        search_boost = 1.0 + 0.05 * min(1.0, max(0.0, log_ratio))
        beh_modifier *= search_boost

    apps_30d = signals.get("applications_submitted_30d", 0)
    if apps_30d >= 3:
        beh_modifier *= 1.05
    elif apps_30d == 0 and not signals.get("open_to_work_flag", False):
        beh_modifier *= 0.95

    if not signals.get("verified_email", True):
        beh_modifier *= 0.95
    if not signals.get("verified_phone", True):
        beh_modifier *= 0.97
    if not signals.get("linkedin_connected", True):
        beh_modifier *= 0.98

    endorsements = signals.get("endorsements_received", 0)
    connections = signals.get("connection_count", 0)
    if endorsements >= 50:
        beh_modifier *= 1.03
    elif endorsements == 0 and connections >= 100:
        beh_modifier *= 0.97

    return loc_modifier * notice_modifier * act_modifier * beh_modifier
