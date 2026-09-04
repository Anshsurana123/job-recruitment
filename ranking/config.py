"""
Centralized configuration system for the Information Retrieval and Ranking pipeline.
Distinguishes reproducible benchmark configurations from runtime live execution.
"""

import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# Default reference date for reproducing the benchmark dataset snapshot
BENCHMARK_REFERENCE_DATE = datetime.date(2026, 5, 20)

# Default company foundation dates for temporal validation
DEFAULT_FOUNDATION_DATES: Dict[str, datetime.date] = {
    "krutrim": datetime.date(2023, 4, 1),
    "sarvam": datetime.date(2023, 7, 1),
    "mistral": datetime.date(2023, 4, 1),
    "xai": datetime.date(2023, 3, 1),
    "perplexity": datetime.date(2022, 8, 1),
    "cognition": datetime.date(2023, 11, 1),
}


@dataclass
class DateConfig:
    """
    Time-dependent configuration.
    Distinguishes:
    - runtime 'today' (wall-clock)
    - reproducible benchmark date (fixed evaluation_date)
    - dataset snapshot date
    """
    mode: str = "benchmark"  # "benchmark" | "live"
    evaluation_date: datetime.date = BENCHMARK_REFERENCE_DATE
    snapshot_date: datetime.date = BENCHMARK_REFERENCE_DATE
    foundation_dates: Dict[str, datetime.date] = field(default_factory=lambda: dict(DEFAULT_FOUNDATION_DATES))

    def resolve_reference_date(self, explicit_date: Optional[datetime.date] = None) -> datetime.date:
        """
        Resolves the reference date prioritizing:
        1. Explicit caller-injected date
        2. Live mode (datetime.date.today())
        3. Configured evaluation_date (benchmark reproducible default)
        """
        if explicit_date is not None:
            return explicit_date
        if self.mode == "live":
            return datetime.date.today()
        return self.evaluation_date


@dataclass
class RetrievalConfig:
    """Parameters for lexical, dense, and hybrid retrieval."""
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    field_weights: Dict[str, float] = field(default_factory=lambda: {
        "title_headline": 4.0,
        "skills": 3.0,
        "career_titles": 2.5,
        "summary": 1.5,
        "career_descriptions": 1.0,
        "education": 0.8,
        "other": 0.5
    })
    hybrid_weights: Tuple[float, float] = (0.60, 0.40)  # (bm25_weight, dense_weight)
    top_k_retrieval: int = 1000
    dense_model_path: str = "./model_cache/bge-small-en-v1.5"
    dense_batch_size: int = 64


@dataclass
class ScoringConfig:
    """Parameters for heuristic fit scoring and behavioral availability."""
    min_exp: float = 5.0
    max_exp: float = 9.0
    seniority_level: str = "Senior"
    target_cities: List[str] = field(default_factory=lambda: [
        "pune", "noida", "delhi", "new delhi", "gurugram", "gurgaon",
        "faridabad", "ghaziabad", "hyderabad", "mumbai"
    ])
    hybrid_score_weight: float = 0.30
    calibrated_honeypots: bool = True
    honeypot_duration_tolerance_months: float = 3.0


@dataclass
class RerankingConfig:
    """Parameters for cross-encoder reranking."""
    enabled: bool = True
    cross_encoder_path: str = "./model_cache/cross-encoder-ms-marco-MiniLM-L-6-v2"
    cross_encoder_top_k: int = 250
    alpha_fit: float = 0.85
    alpha_ce: float = 0.15
    batch_size: int = 32


@dataclass
class PipelineConfig:
    """Master pipeline configuration aggregating sub-configs."""
    dates: DateConfig = field(default_factory=DateConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    reranking: RerankingConfig = field(default_factory=RerankingConfig)
    top_k_output: int = 100
    random_seed: int = 42

    @classmethod
    def benchmark_default(cls, evaluation_date: Optional[datetime.date] = None) -> "PipelineConfig":
        config = cls()
        if evaluation_date is not None:
            config.dates.evaluation_date = evaluation_date
        return config

    @classmethod
    def live_default(cls) -> "PipelineConfig":
        config = cls()
        config.dates.mode = "live"
        return config
