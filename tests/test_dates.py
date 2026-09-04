"""
Tests for date parsing, tenure calculation, and regression verification of the reference-date fix.
"""

import datetime
import pytest
from ranking.dates import (
    parse_date_string,
    parse_date,
    calculate_years_span,
    calculate_days_active,
    calculate_total_experience,
)


def test_parse_date_formats():
    assert parse_date_string("2023-04-15") == datetime.date(2023, 4, 15)
    assert parse_date_string("2023-04") == datetime.date(2023, 4, 1)
    assert parse_date_string("2023") == datetime.date(2023, 1, 1)
    assert parse_date_string("04/15/2023") == datetime.date(2023, 4, 15)
    assert parse_date_string("2023/04/15") == datetime.date(2023, 4, 15)
    assert parse_date_string("04/2023") == datetime.date(2023, 4, 1)
    assert parse_date_string("October 2021") == datetime.date(2021, 10, 1)
    assert parse_date_string("Oct 2021") == datetime.date(2021, 10, 1)
    assert parse_date_string("15 Oct 2021") == datetime.date(2021, 10, 15)


def test_open_ended_date_resolves_to_reference_date():
    ref_1 = datetime.date(2026, 5, 20)
    ref_2 = datetime.date(2026, 9, 4)
    
    assert parse_date_string("Present", reference_date=ref_1) == ref_1
    assert parse_date_string("Present", reference_date=ref_2) == ref_2
    assert parse_date_string("Current", reference_date=ref_1) == ref_1
    assert parse_date_string("ongoing", reference_date=ref_2) == ref_2
    assert parse_date_string(None) is None
    assert parse_date_string("") is None
    assert parse_date_string("null") is None


def test_frozen_reference_date_bug_regression():
    """
    REGRESSION TEST:
    Given a candidate with:
      start = 2024-01-01
      end = Present
    The calculated experience must dynamically reflect the injected reference_date:
    tenure at 2026-09-04 must strictly exceed tenure at 2026-05-20.
    """
    start_date = "2024-01-01"
    end_date = "Present"
    
    date_may = datetime.date(2026, 5, 20)
    date_sept = datetime.date(2026, 9, 4)
    
    tenure_may = calculate_years_span(start_date, end_date, reference_date=date_may)
    tenure_sept = calculate_years_span(start_date, end_date, reference_date=date_sept)
    
    # Verify exact math:
    # 2024-01-01 to 2026-05-20 = 870 days -> 870 / 365.25 = ~2.382 years
    # 2024-01-01 to 2026-09-04 = 977 days -> 977 / 365.25 = ~2.675 years
    assert pytest.approx(tenure_may, 0.01) == 870 / 365.25
    assert pytest.approx(tenure_sept, 0.01) == 977 / 365.25
    
    # Must NOT return the same tenure if additional months elapsed
    assert tenure_sept > tenure_may
    diff_months = (tenure_sept - tenure_may) * 12
    assert pytest.approx(diff_months, 0.2) == 3.5  # ~3.5 months elapsed


def test_calculate_days_active():
    ref_date = datetime.date(2026, 5, 20)
    
    # Active 5 days before reference date
    active_date = "2026-05-15"
    assert calculate_days_active(active_date, reference_date=ref_date) == 5
    
    # Active 30 days before
    active_date_30 = "2026-04-20"
    assert calculate_days_active(active_date_30, reference_date=ref_date) == 30
    
    # Missing / None
    assert calculate_days_active(None, reference_date=ref_date) == 999
    assert calculate_days_active("invalid-date", reference_date=ref_date) == 999


def test_overlapping_concurrent_roles():
    history = [
        {"title": "Role A", "start_date": "2020-01-01", "end_date": "2022-01-01"},
        {"title": "Role B (Part time/Advisor)", "start_date": "2021-01-01", "end_date": "2023-01-01"},
        {"title": "Role C", "start_date": "2023-01-01", "end_date": "Present"}
    ]
    ref_date = datetime.date(2025, 1, 1)
    
    # Merged interval: 2020-01-01 to 2025-01-01 = 5 years exactly
    total_exp = calculate_total_experience(history, reference_date=ref_date, merge_overlaps=True)
    assert pytest.approx(total_exp, 0.05) == 5.0
    
    # Unmerged would falsely double-count 2021-2022
    total_unmerged = calculate_total_experience(history, reference_date=ref_date, merge_overlaps=False)
    assert total_unmerged > total_exp


def test_date_edge_cases():
    ref_date = datetime.date(2026, 5, 20)
    
    # Future start date should yield non-negative tenure
    assert pytest.approx(calculate_years_span("2028-01-01", "2028-06-01", reference_date=ref_date), 0.01) == 152 / 365.25
    assert calculate_years_span("2028-01-01", "Present", reference_date=ref_date) == 0.0
    
    # Invalid date strings should fall back gracefully
    assert calculate_years_span("invalid", "Present", reference_date=ref_date) == 1.0
    assert calculate_years_span("2020-01-01", "invalid", reference_date=ref_date) >= 0.0
    
    # Zero span (same start and end date)
    assert calculate_years_span("2024-01-01", "2024-01-01", reference_date=ref_date) == 0.0
