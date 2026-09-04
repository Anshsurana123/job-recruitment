#!/usr/bin/env python3
"""
Redrob Candidate Ranking Pipeline — CLI Entrypoint.
Backward-compatible wrapper around the modular `ranking` information retrieval package.
"""

import argparse
import datetime
import random
import sys
from pathlib import Path
import numpy as np

# Set deterministic random seeds
random.seed(42)
np.random.seed(42)
try:
    import torch
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
except ImportError:
    pass

from ranking.config import BENCHMARK_REFERENCE_DATE, PipelineConfig
from ranking.dates import parse_date_string, parse_date, calculate_years_span
from ranking.ingestion import load_candidates_from_file, normalize_candidate_record, extract_processed_fields
from ranking.pipeline import DEFAULT_SENIOR_AI_QUERY, RankingPipeline
from ranking.reasoning.generator import DeterministicReasoningGenerator
from ranking.retrieval.bm25f import BM25FIndex, tokenize
from ranking.scoring.composite import infer_seniority_level

# Default reference date for backward compatibility
REFERENCE_DATE = BENCHMARK_REFERENCE_DATE


def parse_jd_file(jd_path):
    """Parses text or docx job description file."""
    if not jd_path:
        return None
    p = Path(jd_path)
    if not p.exists():
        return None
    text = ""
    if p.suffix.lower() == ".docx":
        import zipfile
        import xml.etree.ElementTree as ET
        try:
            with zipfile.ZipFile(p) as docx:
                tree = ET.fromstring(docx.read('word/document.xml'))
                namespace = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
                paragraphs = []
                for paragraph in tree.iter(f"{{{namespace['w']}}}p"):
                    texts = [node.text for node in paragraph.iter(f"{{{namespace['w']}}}t") if node.text]
                    if texts:
                        paragraphs.append("".join(texts))
                text = "\n".join(paragraphs)
        except Exception:
            return None
    else:
        try:
            with open(p, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception:
            return None
    return text


def extract_experience_from_jd(jd_text):
    import re
    range_match = re.search(r'(\d+)\s*(?:-|–|to)\s*(\d+)\s*years', jd_text, re.IGNORECASE)
    if range_match:
        return float(range_match.group(1)), float(range_match.group(2))
    plus_match = re.search(r'(\d+)\s*\+\s*(?:years|yrs|year|yr)', jd_text, re.IGNORECASE)
    if plus_match:
        return float(plus_match.group(1)), 50.0
    return 5.0, 9.0


def extract_locations_from_jd(jd_text):
    cities = ["pune", "noida", "delhi", "new delhi", "gurugram", "gurgaon", "faridabad", "ghaziabad", "hyderabad", "mumbai", "bangalore", "bengaluru", "chennai", "kolkata", "ahmedabad", "kochi", "coimbatore"]
    jd_lower = jd_text.lower()
    found = [city for city in cities if city in jd_lower]
    if "delhi ncr" in jd_lower or "ncr" in jd_lower:
        for ncr_city in ["delhi", "new delhi", "gurugram", "gurgaon", "faridabad", "ghaziabad", "noida"]:
            if ncr_city not in found:
                found.append(ncr_city)
    return found if found else ["pune", "noida", "delhi", "new delhi", "gurugram", "gurgaon", "faridabad", "ghaziabad", "hyderabad", "mumbai"]


def build_candidate_fields(cand):
    record = normalize_candidate_record(cand)
    f = extract_processed_fields(record)
    return {
        "title_headline": f.title_headline,
        "skills": f.skills,
        "career_titles": f.career_titles,
        "summary": f.summary,
        "career_descriptions": f.career_descriptions,
        "education": f.education,
        "other": f.other
    }


def generate_candidate_reasoning(rank, item, reference_date):
    gen = DeterministicReasoningGenerator()
    # Support both dictionary and ScoredCandidate objects
    if hasattr(item, "candidate_id"):
        return gen.generate(rank, item, reference_date=reference_date)
    else:
        from ranking.models import ScoredCandidate
        cand_dict = item.get("cand", item)
        record = normalize_candidate_record(cand_dict)
        sc = ScoredCandidate(
            candidate_id=item.get("candidate_id", record.candidate_id),
            record=record,
            final_score=item.get("final_score", 0.0),
            fit_score=item.get("fit_score", 0.0),
            avail_multiplier=item.get("avail_multiplier", 1.0),
            is_cv_speech_primary=item.get("is_cv_speech_primary", False),
            has_nlp_ir_compensation=item.get("has_nlp_ir_compensation", False)
        )
        return gen.generate(rank, sc, reference_date=reference_date)


def parse_args():
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranking Pipeline")
    parser.add_argument("--candidates", type=str, default="./candidates.jsonl", help="Path to candidates jsonl or json file")
    parser.add_argument("--out", type=str, default="./team_TECHFLOW.csv", help="Path to output submission CSV file")
    parser.add_argument("--jd", type=str, default=None, help="Path to job description text/docx file")
    parser.add_argument("--evaluation-date", type=str, default=None, help="Evaluation reference date (YYYY-MM-DD) for reproducible scoring")
    parser.add_argument("--live", action="store_true", help="Run in live mode using wall-clock date as reference")
    return parser.parse_args()


def main():
    args = parse_args()
    candidates_path = Path(args.candidates)
    out_path = Path(args.out)

    if not candidates_path.exists():
        print(f"Error: Candidate file '{candidates_path}' does not exist.")
        sys.exit(1)

    # Initialize configuration
    config = PipelineConfig()
    
    # Resolve reference date
    eval_date = None
    if args.evaluation_date:
        eval_date = datetime.date.fromisoformat(args.evaluation_date.strip())
        config.dates.evaluation_date = eval_date
        config.dates.mode = "benchmark"
    elif args.live:
        config.dates.mode = "live"
        eval_date = datetime.date.today()
    else:
        eval_date = BENCHMARK_REFERENCE_DATE
        config.dates.evaluation_date = eval_date

    print(f"[Pipeline] Initialized. Mode: {config.dates.mode} | Evaluation Date: {eval_date}")

    # Process query
    query_text = DEFAULT_SENIOR_AI_QUERY
    if args.jd:
        jd_text = parse_jd_file(args.jd)
        if jd_text:
            print(f"[Pipeline] Loaded and parsed Job Description from {args.jd}")
            min_exp, max_exp = extract_experience_from_jd(jd_text)
            target_cities = extract_locations_from_jd(jd_text)
            config.scoring.min_exp = min_exp
            config.scoring.max_exp = max_exp
            config.scoring.target_cities = target_cities
            print(f"[Pipeline] Requirements: Experience {min_exp}-{max_exp} yrs | Cities: {target_cities[:4]}...")

            # Extract dynamic query from JD
            lines = jd_text.split("\n")
            relevant = []
            for line in lines:
                l_str = line.strip()
                if l_str and any(kw in l_str.lower() for kw in ["pytorch", "tensorflow", "ml", "ai", "embedding", "vector", "search", "retrieval", "rank", "eval", "python", "learning", "model", "ndcg", "mrr", "map", "rag"]):
                    relevant.append(l_str)
            if relevant:
                dynamic_query = " ".join(relevant[:20])
                if len(dynamic_query) > 50:
                    query_text = dynamic_query

    # Instantiate pipeline and run
    pipeline = RankingPipeline(config=config)
    ranked_candidates = pipeline.run(
        candidates_input=candidates_path,
        query_text=query_text,
        evaluation_date=eval_date,
        top_n=100
    )

    print(f"[Pipeline] Exporting {len(ranked_candidates)} ranked candidates to {out_path}...")
    pipeline.export_to_csv(ranked_candidates, out_path)
    print("[Pipeline] Execution complete. Submission file successfully written.")


if __name__ == "__main__":
    main()
