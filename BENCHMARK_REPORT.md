# Information Retrieval & Ranking Benchmark Report

- **Date of Execution**: 2026-09-04T10:43:43.870744
- **Evaluation Reference Date**: `2026-05-20`
- **Dataset**: `sample_candidates.json` (SHA256: `edd71d58c9673a5d...`)
- **Total Candidates Evaluated**: 50
- **Identified Highly Relevant Targets**: 5

---

## 1. Ranking Quality Metrics

| Metric | Measured Value | Target / Benchmark | Description |
| :--- | :--- | :--- | :--- |
| **NDCG@10** | **0.5991** | ≥ 0.7500 | Normalized Discounted Cumulative Gain at rank 10 |
| **NDCG@50** | **0.8021** | ≥ 0.7000 | Quality and ordering of top 50 matches |
| **MRR** | **1.0000** | 1.0000 | Mean Reciprocal Rank (position of first relevant pick) |
| **Precision@5** | **0.6000** | ≥ 0.8000 | Fraction of top-5 candidates that are relevant |
| **Precision@10** | **0.4000** | ≥ 0.7000 | Fraction of top-10 candidates that are relevant |
| **MAP** | **0.6998** | ≥ 0.6000 | Mean Average Precision across all relevance levels |

---

## 2. Retrieval Coverage Metrics

| Metric | Measured Value | Description |
| :--- | :--- | :--- |
| **Recall@10** | **0.8000** | Share of relevant candidates retrieved by rank 10 |
| **Recall@25** | **1.0000** | Share of relevant candidates retrieved by rank 25 |
| **Recall@50** | **1.0000** | Share of relevant candidates retrieved by rank 50 |
| **Pool Coverage** | **1.0000** | Candidate selection fraction from input dataset |

---

## 3. Stability & Determinism

| Check | Result | Verification |
| :--- | :--- | :--- |
| **Rank Consistency (Kendall's $\tau$)** | **1.0000** | Perfectly identical ordering across repeated runs |
| **Score Determinism** | **PASS** | Max absolute difference: `0.0` |
| **Honeypots in Top 10** | **0** | Disqualified profile rate: 0% |
| **Honeypots in Top 100** | **0** (0.0%) | Strictly compliant with ≤ 10% tolerance |

---

## 4. Latency & Performance

- **Execution 1**: 26.976 s
- **Execution 2 (warm)**: 8.527 s
- **Average Latency**: 17.751 s
