"""
Information Retrieval (IR) and Ranking evaluation metrics.
Implements Recall@K, Precision@K, MRR, NDCG@K, MAP, and Rank Stability.
"""

import math
from typing import Dict, List, Set, Tuple


def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Computes Recall@K: fraction of total relevant candidates present in top-K.
    """
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / float(len(relevant_ids))


def precision_at_k(ranked_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Computes Precision@K: fraction of top-K candidates that are relevant.
    """
    if k <= 0:
        return 0.0
    top_k = ranked_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for cid in top_k if cid in relevant_ids)
    return hits / float(k)


def mrr(ranked_ids: List[str], relevant_ids: Set[str]) -> float:
    """
    Computes Mean Reciprocal Rank (MRR): 1 / rank position of first relevant candidate.
    """
    for idx, cid in enumerate(ranked_ids):
        if cid in relevant_ids:
            return 1.0 / (idx + 1)
    return 0.0


def dcg_at_k(ranked_ids: List[str], relevance_map: Dict[str, float], k: int) -> float:
    """
    Computes Discounted Cumulative Gain at K using exponential gain formula:
    DCG@K = sum_{i=1}^K (2^{rel_i} - 1) / log2(i + 1)
    """
    dcg = 0.0
    for idx, cid in enumerate(ranked_ids[:k]):
        rel = relevance_map.get(cid, 0.0)
        gain = (2.0 ** rel) - 1.0
        discount = math.log2(idx + 2)  # idx + 2 because idx starts at 0 -> rank 1 is log2(2)
        dcg += gain / discount
    return dcg


def ndcg_at_k(ranked_ids: List[str], relevance_map: Dict[str, float], k: int) -> float:
    """
    Computes Normalized Discounted Cumulative Gain at K (NDCG@K).
    """
    if not relevance_map:
        return 0.0
    actual_dcg = dcg_at_k(ranked_ids, relevance_map, k)
    
    # Ideal DCG: sort all known relevant candidates descending by relevance score
    ideal_sorted = sorted(relevance_map.keys(), key=lambda cid: relevance_map[cid], reverse=True)
    ideal_dcg = dcg_at_k(ideal_sorted, relevance_map, k)
    
    if ideal_dcg == 0.0:
        return 0.0
    return actual_dcg / ideal_dcg


def map_score(ranked_ids: List[str], relevant_ids: Set[str]) -> float:
    """
    Computes Average Precision (AP) for a ranked list.
    """
    if not relevant_ids:
        return 0.0
    hits = 0
    sum_precisions = 0.0
    for idx, cid in enumerate(ranked_ids):
        if cid in relevant_ids:
            hits += 1
            precision_at_i = hits / (idx + 1)
            sum_precisions += precision_at_i
    return sum_precisions / float(len(relevant_ids))


def rank_stability_kendall_tau(ranked_ids_1: List[str], ranked_ids_2: List[str]) -> float:
    """
    Measures rank consistency between two runs using Kendall's tau correlation.
    Returns 1.0 for perfectly identical orderings.
    """
    if len(ranked_ids_1) != len(ranked_ids_2) or not ranked_ids_1:
        return 0.0
    n = len(ranked_ids_1)
    pos2 = {cid: idx for idx, cid in enumerate(ranked_ids_2)}
    
    concordant = 0
    discordant = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            cid_a = ranked_ids_1[i]
            cid_b = ranked_ids_1[j]
            if cid_a not in pos2 or cid_b not in pos2:
                continue
            pos_a2 = pos2[cid_a]
            pos_b2 = pos2[cid_b]
            
            if pos_a2 < pos_b2:
                concordant += 1
            elif pos_a2 > pos_b2:
                discordant += 1
                
    total_pairs = (n * (n - 1)) / 2.0
    return (concordant - discordant) / total_pairs if total_pairs > 0 else 1.0


def score_determinism_check(scores_run_1: List[float], scores_run_2: List[float], tolerance: float = 1e-6) -> Tuple[bool, float]:
    """
    Verifies exact numeric determinism across runs.
    Returns (is_deterministic, max_absolute_difference).
    """
    if len(scores_run_1) != len(scores_run_2):
        return False, float("inf")
    max_diff = 0.0
    for s1, s2 in zip(scores_run_1, scores_run_2):
        diff = abs(s1 - s2)
        if diff > max_diff:
            max_diff = diff
    return max_diff <= tolerance, max_diff
