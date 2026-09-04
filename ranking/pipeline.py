"""
Master Information Retrieval & Ranking Pipeline.
Explicitly connects ingestion, indexing, hybrid retrieval, scoring, reranking, and reasoning.
"""

import csv
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from .config import PipelineConfig
from .dates import parse_date_string
from .ingestion import extract_processed_fields, load_candidates_from_file, normalize_candidate_record
from .leakage import detect_exact_duplicates
from .models import CandidateRecord, ProcessedCandidateFields, RankedCandidate, ScoredCandidate
from .reasoning.generator import DeterministicReasoningGenerator
from .reranking.cross_encoder import CrossEncoderReranker
from .retrieval.bm25f import BM25FIndex, tokenize
from .retrieval.dense import DenseRetriever
from .retrieval.hybrid import HybridFusion
from .scoring.composite import CompositeScorer


DEFAULT_SENIOR_AI_QUERY = (
    "Senior AI Engineer, Founding Team, machine learning, deep learning, PyTorch, embeddings, "
    "vector database, RAG, retrieval, ranking, search, Pinecone, Weaviate, Qdrant, Milvus, FAISS, "
    "OpenSearch, Elasticsearch, evaluation framework, NDCG, MRR, MAP, python, product company, "
    "information retrieval, recommendation system, semantic search, hybrid retrieval, dense retrieval, "
    "reranking, learning to rank, A/B testing, offline evaluation, "
    "collaborative filtering, matrix factorization, recommendation-style, recommender, search features, search pipeline, retrieval pipeline, "
    "data scientist, applied scientist, ml engineer, machine learning engineer, recommendation systems engineer, search engineer"
)


class RankingPipeline:
    """
    End-to-end multi-stage information retrieval and candidate ranking pipeline.
    """
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.bm25_index: Optional[BM25FIndex] = None
        self.dense_retriever = DenseRetriever(self.config.retrieval)
        self.hybrid_fusion = HybridFusion(self.config.retrieval)
        self.composite_scorer = CompositeScorer(self.config.scoring, self.config.dates)
        self.reranker = CrossEncoderReranker(self.config.reranking)
        self.reasoning_gen = DeterministicReasoningGenerator(self.config.scoring.target_cities)

    def run(
        self,
        candidates_input: Union[str, Path, List[Dict]],
        query_text: str = DEFAULT_SENIOR_AI_QUERY,
        evaluation_date: Optional[datetime.date] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        top_n: Optional[int] = None
    ) -> List[RankedCandidate]:
        """
        Executes the full ranking pipeline:
        Ingestion -> Preprocessing -> Lexical/Dense Retrieval -> Hybrid Fusion
        -> Composite Scoring & Honeypot Filtering -> Cross-Encoder Re-ranking
        -> Deterministic Tie-Breaking & Normalization -> Evidence Reasoning.
        """
        ref_date = self.config.dates.resolve_reference_date(evaluation_date)
        n_out = top_n or self.config.top_k_output
        cdir = Path(cache_dir) if cache_dir else Path(".")

        # Stage 1: Candidate Ingestion & Schema Normalization
        if isinstance(candidates_input, (str, Path)):
            raw_candidates = load_candidates_from_file(candidates_input)
            is_sample = "sample" in str(candidates_input).lower() or len(raw_candidates) < 1000
        else:
            raw_candidates = candidates_input
            is_sample = len(raw_candidates) < 1000

        if not raw_candidates:
            return []

        # Duplicate Audit
        dup_info = detect_exact_duplicates(raw_candidates)
        if dup_info["duplicate_ids"]:
            print(f"[Pipeline] Warning: Detected {len(dup_info['duplicate_ids'])} duplicate candidate IDs in raw input.")

        records: List[CandidateRecord] = [normalize_candidate_record(c) for c in raw_candidates]
        candidate_fields: List[ProcessedCandidateFields] = [extract_processed_fields(r) for r in records]

        # Stage 2: Lexical Index (BM25F)
        bm25_cache_file = cdir / ("bm25f_index_sample.pkl" if is_sample else "bm25f_index_full.pkl")
        loaded_bm25 = BM25FIndex.load(bm25_cache_file)
        if loaded_bm25 and loaded_bm25.N == len(records):
            self.bm25_index = loaded_bm25
        else:
            self.bm25_index = BM25FIndex(self.config.retrieval)
            for f in candidate_fields:
                self.bm25_index.add_candidate(f)
            self.bm25_index.finalize()
            if not is_sample or len(records) > 10:
                try:
                    self.bm25_index.save(bm25_cache_file)
                except Exception:
                    pass

        query_tokens = tokenize(query_text)
        bm25_scores = self.bm25_index.get_scores(query_tokens)

        # Stage 3: Dense Retrieval
        emb_cache_file = cdir / ("embeddings_sample.pkl" if is_sample else "embeddings_full.pkl")
        self.dense_retriever.load_cached_embeddings(emb_cache_file)
        dense_scores = self.dense_retriever.score_candidates(query_text, candidate_fields)

        # Stage 4: Hybrid Fusion
        retrieval_matches = self.hybrid_fusion.fuse(
            records=records,
            bm25_scores=bm25_scores,
            dense_scores=dense_scores,
            top_k=self.config.retrieval.top_k_retrieval
        )

        # Stage 5: Composite Scoring & Honeypots
        scored_candidates = self.composite_scorer.score(
            retrieval_matches=retrieval_matches,
            reference_date=ref_date
        )

        # Stage 6: Cross-Encoder Re-ranking
        reranked_candidates = self.reranker.rerank(
            scored_candidates=scored_candidates,
            query_text=query_text,
            top_k=self.config.reranking.cross_encoder_top_k
        )

        # Stage 7: Normalization, Tie-Breaking, and Reasoning Generation
        top_pool = reranked_candidates[:n_out]
        if not top_pool:
            return []

        max_final = max(c.final_score for c in top_pool)
        norm_factor = max_final if max_final > 0.0 else 1.0

        # Deterministic sort: non-increasing score, tie-break by candidate_id ascending
        sorted_top = sorted(top_pool, key=lambda c: (-round(c.final_score / norm_factor, 4), c.candidate_id))

        ranked_results: List[RankedCandidate] = []
        for idx, sc in enumerate(sorted_top):
            rank = idx + 1
            norm_score = round(sc.final_score / norm_factor, 4)
            reasoning = self.reasoning_gen.generate(rank, sc, reference_date=ref_date)
            
            ranked_results.append(RankedCandidate(
                candidate_id=sc.candidate_id,
                rank=rank,
                score=norm_score,
                reasoning=reasoning,
                metadata={
                    "fit_score": sc.fit_score,
                    "avail_multiplier": sc.avail_multiplier,
                    "is_honeypot": sc.is_honeypot,
                    "cross_encoder_score": sc.cross_encoder_score
                }
            ))

        return ranked_results

    @staticmethod
    def export_to_csv(ranked_candidates: List[RankedCandidate], output_path: Union[str, Path]):
        """Writes top ranked candidates to CSV adhering strictly to the submission spec."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["candidate_id", "rank", "score", "reasoning"])
            for c in ranked_candidates:
                writer.writerow([c.candidate_id, c.rank, f"{c.score:.4f}", c.reasoning])
