"""
BM25F field-weighted lexical retrieval index.
"""

import math
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Union

from ..config import RetrievalConfig
from ..models import ProcessedCandidateFields


def tokenize(text: str) -> List[str]:
    """Tokenizes text and appends domain concept tokens."""
    text_lower = (text or "").lower()
    tokens = re.findall(r'\b[a-z0-9_]{2,}\b', text_lower)
    
    concepts = []
    # 1. Deep learning
    if any(k in text_lower for k in ["deep learning", "neural network", "neural model", "deepspeed", "fsdp"]):
        concepts.append('concept_deep_learning')
        
    # 2. LLM
    if any(k in text_lower for k in [
        "large language model", "llm", "transformer", "fine-tuning", "finetuning",
        "lora", "qlora", "peft", "bert", "gpt", "llama", "mistral", "huggingface"
    ]):
        concepts.append('concept_llm')
        
    # 3. Vector search
    if any(k in text_lower for k in [
        "vector search", "semantic search", "dense retrieval", "embedding", "pinecone",
        "weaviate", "qdrant", "milvus", "faiss", "nearest neighbor", "ann index",
        "similarity search", "hnsw", "annoy", "scann"
    ]):
        concepts.append('concept_vector_search')
        
    # 4. Information retrieval
    if any(k in text_lower for k in [
        "information retrieval", "search engine", "elasticsearch", "opensearch",
        "solr", "lucene", "splade", "bm25", "bm25f"
    ]) or re.search(r'\bir\b', text_lower):
        concepts.append('concept_information_retrieval')
        
    # 5. Recommendation
    if any(k in text_lower for k in [
        "recommendation", "recsys", "recommender", "collaborative filtering",
        "matrix factorization", "feed ranking", "ctr prediction", "recommending",
        "two-tower", "candidate generation"
    ]):
        concepts.append('concept_recommendation')
        
    # 6. Evaluation
    if any(k in text_lower for k in [
        "ndcg", "mrr", "a/b testing", "ab testing", "offline evaluation",
        "online evaluation", "evaluation framework", "evaluation metric",
        "precision", "recall", "f1-score", "roc-auc"
    ]) or re.search(r'\bmap\b', text_lower):
        concepts.append('concept_evaluation')
        
    # 7. ML framework
    if any(k in text_lower for k in ["pytorch", "tensorflow", "jax", "keras"]):
        concepts.append('concept_ml_framework')
        
    # 8. AI/ML
    if any(k in text_lower for k in ["machine learning", "data science"]) or re.search(r'\b(ml|ai)\b', text_lower):
        concepts.append('concept_artificial_intelligence')
        
    return tokens + concepts


class BM25FIndex:
    """
    Field-weighted BM25F index scoring candidate fields with length normalization.
    """
    def __init__(self, config: Optional[RetrievalConfig] = None):
        cfg = config or RetrievalConfig()
        self.k1 = cfg.bm25_k1
        self.b = cfg.bm25_b
        self.field_weights = cfg.field_weights
        
        # Per-field lengths & statistics
        self.field_bs = {f: self.b for f in self.field_weights}
        self.field_doc_lens: Dict[str, List[int]] = {f: [] for f in self.field_weights}
        self.field_avg_lens: Dict[str, float] = {}
        
        self.doc_ids: List[str] = []
        self.doc_term_freqs: List[Dict[str, Dict[str, int]]] = []  # doc_idx -> {field: Counter}
        self.doc_freq: Counter = Counter()
        self.N: int = 0
        self.is_finalized: bool = False

    def add_document(self, doc_id: str, fields: Dict[str, str]):
        self.doc_ids.append(doc_id)
        doc_field_tfs: Dict[str, Dict[str, int]] = {}
        unique_terms_in_doc = set()
        
        for field, weight in self.field_weights.items():
            text = fields.get(field, "")
            tokens = tokenize(text)
            self.field_doc_lens[field].append(len(tokens))
            tfs = Counter(tokens)
            doc_field_tfs[field] = dict(tfs)
            unique_terms_in_doc.update(tfs.keys())
            
        for term in unique_terms_in_doc:
            self.doc_freq[term] += 1
            
        self.doc_term_freqs.append(doc_field_tfs)
        self.N += 1

    def add_candidate(self, fields: ProcessedCandidateFields):
        field_dict = {
            "title_headline": fields.title_headline,
            "skills": fields.skills,
            "career_titles": fields.career_titles,
            "summary": fields.summary,
            "career_descriptions": fields.career_descriptions,
            "education": fields.education,
            "other": fields.other
        }
        self.add_document(fields.candidate_id, field_dict)

    def finalize(self):
        if self.N == 0:
            return
        for f in self.field_weights:
            lens = self.field_doc_lens[f]
            self.field_avg_lens[f] = sum(lens) / float(self.N) if self.N > 0 else 1.0
            if self.field_avg_lens[f] == 0.0:
                self.field_avg_lens[f] = 1.0
        self.is_finalized = True

    def get_scores(self, query_tokens: List[str]) -> Dict[str, float]:
        if not getattr(self, "is_finalized", False):
            self.finalize()
            
        if self.N == 0:
            return {}

        q_terms = [t for t in set(query_tokens) if t in self.doc_freq]
        scores: Dict[str, float] = {doc_id: 0.0 for doc_id in self.doc_ids}
        
        if not q_terms:
            return scores

        # Precompute IDF for terms: Robertson-Spärck Jones IDF
        idfs = {}
        for t in q_terms:
            df = self.doc_freq[t]
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
            idfs[t] = max(0.001, idf)

        # Precompute length normalization multipliers per doc and field
        B_factors: Dict[str, List[float]] = {}
        for f in self.field_weights:
            avg_len = self.field_avg_lens[f]
            b_val = self.field_bs[f]
            lens = self.field_doc_lens[f]
            B_factors[f] = [(1.0 - b_val) + b_val * (l / avg_len) for l in lens]

        weights = self.field_weights

        for idx in range(self.N):
            doc_tfs = self.doc_term_freqs[idx]
            doc_score = 0.0
            
            for t in q_terms:
                tf_eff = 0.0
                for f, w in weights.items():
                    c = doc_tfs[f].get(t, 0)
                    if c > 0:
                        B_f = B_factors[f][idx]
                        tf_eff += w * (c / B_f)
                        
                if tf_eff > 0.0:
                    term_score = idfs[t] * (tf_eff / (self.k1 + tf_eff))
                    doc_score += term_score
                    
            scores[self.doc_ids[idx]] = doc_score

        return scores

    def save(self, file_path: Union[str, Path]):
        with open(file_path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> Optional["BM25FIndex"]:
        path = Path(file_path)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            if hasattr(obj, "field_weights") and hasattr(obj, "doc_ids") and hasattr(obj, "field_doc_lens"):
                return obj
        except Exception:
            pass
        return None
