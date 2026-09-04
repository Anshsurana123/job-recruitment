"""
Scoring stages: CompositeScorer, Honeypots, and Signals.
"""

from .composite import CompositeScorer, infer_seniority_level
from .honeypots import detect_honeypots
from .signals import calculate_availability_multiplier

__all__ = [
    "CompositeScorer",
    "infer_seniority_level",
    "detect_honeypots",
    "calculate_availability_multiplier",
]
