#!/usr/bin/env python3
"""
Canonical evaluation benchmark runner.
Executes the reproducible IR and Ranking evaluation benchmark, measuring:
- Retrieval: Recall@K, Coverage
- Ranking: NDCG@K, MRR, Precision@K, MAP
- Stability: Kendall's tau rank correlation, score determinism
- Integrity: Honeypot disqualification rates
- Latency: Cold vs warm execution timings

Produces:
- benchmark_results.json (machine-readable)
- BENCHMARK_REPORT.md (human-readable)
"""

import argparse
import datetime
import sys
from pathlib import Path

from ranking.config import BENCHMARK_REFERENCE_DATE
from ranking.evaluation.benchmark import run_benchmark


def parse_args():
    parser = argparse.ArgumentParser(description="Candidate Ranking & Retrieval Benchmark Runner")
    parser.add_argument(
        "--candidates",
        type=str,
        default="sample_candidates.json",
        help="Path to evaluation candidates dataset (default: sample_candidates.json)"
    )
    parser.add_argument(
        "--evaluation-date",
        type=str,
        default=None,
        help="Evaluation reference date (YYYY-MM-DD), default: 2026-05-20"
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="benchmark_results.json",
        help="Path for machine-readable JSON artifact"
    )
    parser.add_argument(
        "--out-report",
        type=str,
        default="BENCHMARK_REPORT.md",
        help="Path for human-readable Markdown report"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    eval_date = None
    if args.evaluation_date:
        eval_date = datetime.date.fromisoformat(args.evaluation_date.strip())
    else:
        eval_date = BENCHMARK_REFERENCE_DATE

    print(f"=====================================================================")
    print(f"  IR & RANKING REPRODUCIBLE EVALUATION BENCHMARK")
    print(f"  Evaluation Date: {eval_date.isoformat()}")
    print(f"  Candidates File: {args.candidates}")
    print(f"=====================================================================\n")

    results = run_benchmark(
        candidates_file=args.candidates,
        output_json_path=args.out_json,
        output_report_path=args.out_report,
        evaluation_date=eval_date
    )

    print(f"\nBenchmark execution succeeded.")
    print(f"Ranking metrics: NDCG@10={results['metrics']['ranking']['ndcg_at_10']:.4f} | MRR={results['metrics']['ranking']['mrr']:.4f} | P@5={results['metrics']['ranking']['precision_at_5']:.4f}")
    print(f"Stability: Kendall tau={results['metrics']['stability']['kendall_tau']:.4f} | Deterministic={results['metrics']['stability']['score_determinism']}")
    print(f"Artifacts generated:")
    print(f"  - {args.out_json}")
    print(f"  - {args.out_report}")


if __name__ == "__main__":
    main()
