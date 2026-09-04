import sys
import json
import math
import pickle
import datetime
import re
import argparse
import random
from pathlib import Path
from collections import Counter
import numpy as np
import torch

# Set random seeds for determinism
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Reference date in the hackathon ecosystem
REFERENCE_DATE = datetime.date(2026, 5, 20)

def parse_date(d_str):
    if not d_str:
        return None
    try:
        return datetime.date.fromisoformat(str(d_str).strip())
    except:
        return None

def infer_seniority_level(title):
    title_lower = title.lower()
    if "director" in title_lower: return 6
    if "principal" in title_lower or "staff" in title_lower: return 5
    if "lead" in title_lower or "head" in title_lower: return 4
    if "senior" in title_lower or "sr" in title_lower: return 3
    if "junior" in title_lower or "jr" in title_lower: return 1
    if "intern" in title_lower or "co-op" in title_lower: return 0
    if "associate" in title_lower: return 1
    return 2

def tokenize(text):
    return re.findall(r'\b[a-z0-9_]{2,}\b', text.lower())

class BM25FIndex:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.field_weights = {
            "title_headline": 4.0,
            "skills": 3.0,
            "career_titles": 2.5,
            "summary": 1.5,
            "career_descriptions": 1.0,
            "education": 0.8,
            "other": 0.5
        }
        self.doc_len = []
        self.avg_doc_len = 0.0
        self.doc_tfs = []
        self.doc_ids = []
        self.dfs = {}
        self.N = 0

    def add_document(self, doc_id, fields):
        self.doc_ids.append(doc_id)
        weighted_tf = {}
        total_weighted_len = 0.0
        
        for field_name, text in fields.items():
            weight = self.field_weights.get(field_name, 1.0)
            tokens = tokenize(text)
            field_len = len(tokens)
            total_weighted_len += field_len * weight
            
            field_tf = Counter(tokens)
            for term, count in field_tf.items():
                weighted_tf[term] = weighted_tf.get(term, 0.0) + count * weight
                
        self.doc_len.append(total_weighted_len)
        self.doc_tfs.append(weighted_tf)
        
        for term in weighted_tf:
            self.dfs[term] = self.dfs.get(term, 0) + 1
        self.N += 1

    def finalize(self):
        if self.N > 0:
            self.avg_doc_len = sum(self.doc_len) / self.N
        else:
            self.avg_doc_len = 0.0

    def get_scores(self, query_tokens):
        unique_query = set(query_tokens)
        query_freqs = Counter(query_tokens)
        
        idfs = {}
        for term in unique_query:
            df = self.dfs.get(term, 0)
            idfs[term] = max(0.0001, math.log((self.N - df + 0.5) / (df + 0.5) + 1.0))
            
        len_factors = [self.k1 * (1.0 - self.b + self.b * (l / self.avg_doc_len)) for l in self.doc_len]
        
        scores = [0.0] * self.N
        for idx in range(self.N):
            tf_dict = self.doc_tfs[idx]
            lf = len_factors[idx]
            s = 0.0
            for term, qf in query_freqs.items():
                if term in tf_dict:
                    tf = tf_dict[term]
                    s += idfs[term] * (tf * (self.k1 + 1.0)) / (tf + lf) * qf
            scores[idx] = s
        return scores

def build_candidate_fields(cand):
    profile = cand.get("profile", {})
    career = cand.get("career_history", [])
    education = cand.get("education", [])
    skills = cand.get("skills", [])
    
    fields = {}
    parts_title = []
    if profile.get("current_title"): parts_title.append(profile["current_title"])
    if profile.get("headline"): parts_title.append(profile["headline"])
    fields["title_headline"] = " ".join(parts_title)
    
    fields["skills"] = " ".join([s.get("name", "") for s in skills if s.get("name")])
    fields["career_titles"] = " ".join([job.get("title", "") for job in career if job.get("title")])
    fields["summary"] = profile.get("summary", "")
    fields["career_descriptions"] = " ".join([job.get("description", "") for job in career if job.get("description")])
    
    edu_parts = []
    for edu in education:
        if edu.get("field_of_study"): edu_parts.append(edu["field_of_study"])
        if edu.get("degree"): edu_parts.append(edu["degree"])
    fields["education"] = " ".join(edu_parts)
    
    other_parts = []
    for job in career:
        if job.get("company"): other_parts.append(job["company"])
    for edu in education:
        if edu.get("institution"): other_parts.append(edu["institution"])
    fields["other"] = " ".join(other_parts)
    
    return fields

def parse_jd_file(jd_path):
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
                    if texts: paragraphs.append("".join(texts))
                text = "\n".join(paragraphs)
        except:
            return None
    else:
        try:
            with open(p, "r", encoding="utf-8") as f:
                text = f.read()
        except:
            return None
    return text

def extract_experience_from_jd(jd_text):
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
    return found if found else ["pune", "noida", "delhi", "new delhi", "gurugram", "gurgaon", "faridabad", "ghaziabad", "hyderabad", "mumbai"]

def main():
    parser = argparse.ArgumentParser(description="Train Student Tabular Ranker via Knowledge Distillation")
    parser.add_argument("--candidates", type=str, default="./candidates.jsonl", help="Path to candidates jsonl file")
    parser.add_argument("--jd", type=str, default="./job_description_full_extracted.txt", help="Path to job description")
    parser.add_argument("--embeddings", type=str, default="./embeddings_full.pkl", help="Path to precomputed embeddings")
    parser.add_argument("--output", type=str, default="./student_ranker.pkl", help="Path to save student model weights")
    parser.add_argument("--size", type=int, default=10000, help="Number of retrieved candidates to label for training")
    parser.add_argument("--evaluation-date", type=str, default=None, help="Evaluation reference date (YYYY-MM-DD), default: 2026-05-20")
    args = parser.parse_args()

    ref_date = datetime.date.fromisoformat(args.evaluation_date.strip()) if args.evaluation_date else REFERENCE_DATE
    print(f"Using evaluation reference date: {ref_date}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} for Cross-Encoder labeling.")

    candidates_path = Path(args.candidates)
    if not candidates_path.exists():
        print(f"Error: {candidates_path} not found.")
        sys.exit(1)

    print("Loading candidates...")
    candidates = []
    with open(candidates_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if content.startswith("[") and content.endswith("]"):
            try:
                candidates = json.loads(content)
            except Exception as e:
                print(f"Error loading as JSON array: {e}")
        
        if not candidates:
            f.seek(0)
            for line in f:
                if line.strip():
                    try:
                        candidates.append(json.loads(line))
                    except:
                        pass
    print(f"Loaded {len(candidates)} candidates.")

    # Parse JD
    jd_text = parse_jd_file(args.jd)
    min_exp, max_exp = 5.0, 9.0
    target_cities = ["pune", "noida", "delhi", "new delhi", "gurugram", "gurgaon", "faridabad", "ghaziabad", "hyderabad", "mumbai"]
    
    query_text = (
        "Senior AI Engineer, Founding Team, machine learning, deep learning, PyTorch, embeddings, "
        "vector database, RAG, retrieval, ranking, search, Pinecone, Weaviate, Qdrant, Milvus, FAISS, "
        "OpenSearch, Elasticsearch, evaluation framework, NDCG, MRR, MAP, python, product company, "
        "information retrieval, recommendation system, semantic search, hybrid retrieval, dense retrieval, "
        "reranking, learning to rank, A/B testing, offline evaluation, "
        "collaborative filtering, matrix factorization, recommendation-style, recommender, search features, search pipeline, retrieval pipeline, "
        "data scientist, applied scientist, ml engineer, machine learning engineer, recommendation systems engineer, search engineer"
    )

    if jd_text:
        print("Parsing Job Description...")
        min_exp, max_exp = extract_experience_from_jd(jd_text)
        target_cities = extract_locations_from_jd(jd_text)
        print(f"Extracted requirements: Exp={min_exp}-{max_exp} yrs, Cities={target_cities}")
        
        lines = jd_text.split("\n")
        relevant_parts = []
        for line in lines:
            line_clean = line.strip()
            if not line_clean: continue
            line_lower = line_clean.lower()
            if any(kw in line_lower for kw in ["pytorch", "tensorflow", "ml", "ai", "embedding", "vector", "search", "retrieval", "rank", "eval", "python", "learning", "model", "ndcg", "mrr", "map", "rag"]):
                relevant_parts.append(line_clean)
        dynamic_query = " ".join(relevant_parts[:20])
        if len(dynamic_query) > 50:
            query_text = dynamic_query

    query_tokens = tokenize(query_text)

    # 1. BM25F index loading/building
    is_sample = len(candidates) < 1000
    index_cache_name = "bm25f_index_sample.pkl" if is_sample else "bm25f_index_full.pkl"
    index_cache_path = Path(index_cache_name)

    def build_index(candidates):
        idx = BM25FIndex()
        print(f"Building BM25F index from {len(candidates)} candidates...")
        for i, cand in enumerate(candidates):
            cid = cand["candidate_id"]
            fields = build_candidate_fields(cand)
            idx.add_document(cid, fields)
            if (i + 1) % 20000 == 0:
                print(f"  Indexed {i + 1} candidates...")
        idx.finalize()
        with open(index_cache_path, "wb") as f:
            pickle.dump(idx, f)
        print("BM25F index built and saved.")
        return idx

    index = BM25FIndex()
    if index_cache_path.exists():
        print(f"Loading BM25F index from {index_cache_path}...")
        with open(index_cache_path, "rb") as f:
            index = pickle.load(f)
        # Guard: stale index (different N) causes IndexError downstream
        if index.N != len(candidates):
            print(f"WARNING: Cached index has N={index.N} but loaded {len(candidates)} candidates. Rebuilding...")
            index = build_index(candidates)
    else:
        index = build_index(candidates)

    print("Computing lexical scores...")
    bm25_scores = index.get_scores(query_tokens)

    # 2. Semantic scores
    emb_cache_path = Path(args.embeddings)
    id_to_embedding = {}
    if emb_cache_path.exists():
        print(f"Loading precomputed embeddings from {emb_cache_path}...")
        with open(emb_cache_path, "rb") as f:
            id_to_embedding = pickle.load(f)
    else:
        print(f"Warning: Precomputed embeddings file {emb_cache_path} not found. We will generate embeddings on the fly.")

    # Build the candidate embeddings list
    cand_embs = []
    uncached_cands = []
    uncached_indices = []
    
    for idx, cand in enumerate(candidates):
        cid = cand["candidate_id"]
        if cid in id_to_embedding:
            cand_embs.append(id_to_embedding[cid])
        else:
            cand_embs.append(None)
            uncached_cands.append(cand)
            uncached_indices.append(idx)

    # Build model for query and uncached candidate encoding
    from sentence_transformers import SentenceTransformer
    model_cache_path = Path("./model_cache/bge-small-en-v1.5")
    if not model_cache_path.exists():
        print("Model cache not found. Downloading BAAI/bge-small-en-v1.5...")
        model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        model.save(str(model_cache_path))
    else:
        model = SentenceTransformer(str(model_cache_path))

    if uncached_cands:
        print(f"Encoding {len(uncached_cands)} uncached candidates dynamically on CPU...")
        uncached_texts = []
        for cand in uncached_cands:
            fields = build_candidate_fields(cand)
            career = cand.get("career_history", [])
            recent_roles = []
            for job in career[:3]:
                jt = job.get("title", "")
                comp = job.get("company", "")
                desc = job.get("description", "")
                role_p = []
                if jt:
                    if comp: role_p.append(f"{jt} at {comp}")
                    else: role_p.append(jt)
                if desc:
                    clean_desc = re.sub(r'\s+', ' ', desc).strip()
                    role_p.append(f"({clean_desc[:150]}...)")
                if role_p:
                    recent_roles.append(" ".join(role_p))
            career_str = ". ".join(recent_roles)
            text = f"Current Title: {fields['title_headline']}. Skills: {fields['skills']}. Summary: {fields['summary']}. Recent: {career_str}."
            uncached_texts.append(text)
            
        uncached_embeddings = model.encode(
            uncached_texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        for i, idx in enumerate(uncached_indices):
            cand_embs[idx] = uncached_embeddings[i]

    print("Encoding query...")
    bge_query = "Represent this sentence for searching relevant passages: " + query_text
    query_embedding = model.encode(bge_query, convert_to_numpy=True, normalize_embeddings=True)

    print("Computing semantic scores...")
    cand_embs_matrix = np.array(cand_embs, dtype=np.float32)
    semantic_scores = np.dot(cand_embs_matrix, query_embedding)

    # 3. Hybrid fusion and top-N retrieval
    min_bm25, max_bm25 = min(bm25_scores), max(bm25_scores)
    bm25_range = max_bm25 - min_bm25 if max_bm25 > min_bm25 else 1.0
    
    min_sem, max_sem = min(semantic_scores), max(semantic_scores)
    sem_range = max_sem - min_sem if max_sem > min_sem else 1.0

    global_hybrid_scores = []
    for idx, cand in enumerate(candidates):
        norm_bm25 = 100.0 * (bm25_scores[idx] - min_bm25) / bm25_range
        norm_sem = 100.0 * (semantic_scores[idx] - min_sem) / sem_range
        hybrid_score = 0.60 * norm_bm25 + 0.40 * norm_sem
        global_hybrid_scores.append((hybrid_score, bm25_scores[idx], semantic_scores[idx], cand))

    global_hybrid_scores.sort(key=lambda x: x[0], reverse=True)
    top_n_pool = global_hybrid_scores[:args.size]
    print(f"Retrieved top {len(top_n_pool)} candidates for training.")

    # Get min/max hybrid for candidate-level calculations
    top_hybrid_vals = [item[0] for item in top_n_pool]
    max_hybrid = max(top_hybrid_vals) if top_hybrid_vals else 1.0
    min_hybrid = min(top_hybrid_vals) if top_hybrid_vals else 0.0
    hybrid_range = max_hybrid - min_hybrid

    # Build Skill IDF over full corpus
    skill_doc_freq = Counter()
    for cand_item in candidates:
        for s in cand_item.get("skills", []):
            if s.get("name"): skill_doc_freq[s["name"].lower()] += 1
            
    total_candidates = len(candidates)
    skill_idf = {sk: math.log(total_candidates / df) for sk, df in skill_doc_freq.items() if df > 0}
    max_idf = max(skill_idf.values()) if skill_idf else 1.0
    skill_idf_norm = {sk: v / max_idf for sk, v in skill_idf.items()}

    jd_relevant_skills = {
        "pytorch", "tensorflow", "keras", "jax", "cuda", "triton", "deep learning",
        "neural networks", "llms", "large language models", "transformers", "fine-tuning",
        "peft", "lora", "qlora", "bert", "gpt", "cnn", "rnn",
        "pinecone", "weaviate", "qdrant", "milvus", "faiss", "opensearch", "elasticsearch",
        "vector search", "semantic search", "hybrid search", "retrieval", "ranking",
        "reranking", "information retrieval", "recommendation", "recommendation systems",
        "recsys", "collaborative filtering", "rag",
        "ndcg", "mrr", "map", "a/b testing", "offline evaluation", "online evaluation",
        "python", "machine learning", "data science", "mlops", "feature engineering"
    }
    ml_domain_companies = {
        "google", "deepmind", "meta", "facebook", "openai", "anthropic", "microsoft",
        "amazon", "aws", "apple", "nvidia", "uber", "airbnb", "netflix", "spotify",
        "linkedin", "twitter", "x", "pinterest", "snap", "bytedance", "tiktok",
        "stripe", "shopify", "databricks", "snowflake", "palantir", "confluent",
        "hugging face", "huggingface", "cohere", "stability ai", "midjourney",
        "samsung research", "adobe", "salesforce", "oracle", "ibm research",
        "flipkart", "swiggy", "zomato", "meesho", "phonepe", "razorpay", "cred",
        "dream11", "juspay", "ola", "myntra", "paytm", "zerodha"
    }
    consulting_firms = ["tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant", "capgemini", "hcl", "mphasis"]

    print("Computing candidate tabular features...")
    candidate_features = []
    cross_encoder_texts = []
    
    for idx, (hybrid_score, bm25_score, semantic_score, cand) in enumerate(top_n_pool):
        profile = cand.get("profile", {})
        career = cand.get("career_history", [])
        education = cand.get("education", [])
        skills = cand.get("skills", [])
        signals = cand.get("redrob_signals", {})
        
        # Norm Hybrid
        norm_hybrid = 100.0 * (hybrid_score - min_hybrid) / hybrid_range if hybrid_range > 0 else 100.0
        
        # ML/DL score
        skills_lower = {s.get("name", "").lower() for s in skills if s.get("name")}
        career_text = " ".join([(j.get("title", "") + " " + j.get("description", "")).lower() for j in career])
        
        ml_dl_skills = {"pytorch", "tensorflow", "keras", "jax", "cuda", "triton", "deep learning", "neural networks", "llms", "large language models", "transformers", "fine-tuning", "peft", "lora", "qlora", "bert", "gpt", "cnn", "rnn"}
        ml_dl_matches = skills_lower.intersection(ml_dl_skills)
        has_ml_dl = len(ml_dl_matches) >= 2 or any(kw in career_text for kw in ["pytorch", "deep learning", "transformer", "lora", "llm", "embedding"])
        ml_dl_sub = 100.0 if has_ml_dl else (50.0 if ml_dl_matches else 20.0)
        
        # IR score
        ir_search_skills = {"pinecone", "weaviate", "qdrant", "milvus", "faiss", "opensearch", "elasticsearch", "vector search", "semantic search", "hybrid search", "retrieval", "ranking", "reranking", "information retrieval", "recommendation"}
        ir_matches = skills_lower.intersection(ir_search_skills)
        has_ir_search = len(ir_matches) >= 2 or any(kw in career_text for kw in ["vector search", "semantic search", "hybrid retrieval", "recommendation", "recommender", "pinecone", "weaviate", "milvus"])
        ir_search_sub = 100.0 if has_ir_search else (40.0 if ir_matches else 10.0)
        
        # Eval score
        eval_skills = {"ndcg", "mrr", "map", "a/b testing", "offline evaluation", "online evaluation", "evaluation framework"}
        eval_matches = skills_lower.intersection(eval_skills)
        has_eval = len(eval_matches) >= 1 or any(kw in career_text for kw in ["ndcg", "mrr", "map", "a/b testing", "ab testing", "evaluation framework"])
        eval_sub = 100.0 if has_eval else (30.0 if eval_matches else 0.0)
        
        tech_score = 0.35 * ml_dl_sub + 0.40 * ir_search_sub + 0.25 * eval_sub
        
        # Exp Score
        years_exp = profile.get("years_of_experience", 0.0)
        exp_score = 60.0
        if min_exp <= years_exp <= max_exp: exp_score = 100.0
        elif (min_exp - 1.0) <= years_exp < min_exp and min_exp > 1.0: exp_score = 95.0
        elif years_exp < (min_exp - 1.0): exp_score = 50.0 + 35.0 * (years_exp / max(1.0, min_exp - 1.0))
        elif max_exp < years_exp <= (max_exp + 1.0): exp_score = 100.0 - 10.0 * (years_exp - max_exp)
        else: exp_score = max(50.0, 90.0 - 5.0 * (years_exp - (max_exp + 3.0)))
        
        # Velocity
        chron_career = list(reversed(career))
        upward_count = 0
        for i in range(len(chron_career) - 1):
            if infer_seniority_level(chron_career[i+1].get("title", "")) > infer_seniority_level(chron_career[i].get("title", "")):
                upward_count += 1
        velocity_score = min(100.0, 30.0 * upward_count)
        
        # Company
        entire_consulting = True
        company_val = 50.0
        has_startup = False
        for job in career:
            comp_n = job.get("company", "").lower()
            if not any(cf in comp_n for cf in consulting_firms):
                entire_consulting = False
            c_sz = job.get("company_size", "unknown")
            if c_sz in ["1-10", "11-50", "51-200", "201-500"]:
                has_startup = True
        if entire_consulting: company_val = 0.0
        elif has_startup: company_val = 90.0
        else: company_val = 80.0
        
        # Title
        current_title = profile.get("current_title", "")
        title_val = 60.0
        if any(kw in current_title.lower() for kw in ["ai", "machine learning", "mle", "data scientist", "applied scientist", "search", "retrieval"]):
            title_val = 100.0
        elif any(kw in current_title.lower() for kw in ["software", "backend", "developer"]):
            title_val = 80.0
            
        # Edu
        edu_base = 30.0
        for edu in education:
            t = edu.get("tier", "unknown")
            if t == "tier_1": edu_base = max(edu_base, 100.0)
            elif t == "tier_2": edu_base = max(edu_base, 85.0)
            elif t == "tier_3": edu_base = max(edu_base, 70.0)
            
        # Availability modifier
        loc_lower = profile.get("location", "").lower()
        is_jd_city = any(c in loc_lower for c in target_cities)
        willing_reloc = signals.get("willing_to_relocate", False)
        loc_modifier = 1.0 if is_jd_city else (0.90 if willing_reloc else 0.10)
        
        notice_days = signals.get("notice_period_days", 0)
        notice_modifier = 1.0 if notice_days <= 30 else (0.80 if notice_days <= 90 else 0.50)
        
        last_active_str = signals.get("last_active_date", "")
        active_date = parse_date(last_active_str)
        days_active = (ref_date - active_date).days if active_date else 999
        act_modifier = 1.05 if days_active <= 30 else (1.00 if days_active <= 90 else 0.50)
        
        avail_multiplier = loc_modifier * notice_modifier * act_modifier
        
        # Add to features
        feats = [
            norm_hybrid,
            bm25_score,
            semantic_score,
            tech_score,
            exp_score,
            velocity_score,
            company_val,
            title_val,
            edu_base,
            avail_multiplier,
            years_exp,
            1.0 if signals.get("open_to_work_flag") else 0.0,
            float(signals.get("recruiter_response_rate", 1.0)),
            float(signals.get("github_activity_score", -1))
        ]
        
        candidate_features.append(feats)
        
        # Build Cross-Encoder text
        title = profile.get("current_title", "")
        headline = profile.get("headline", "")
        summary = profile.get("summary", "")
        skills_str = ", ".join(s.get("name", "") for s in skills[:15])
        
        recent_roles = []
        for job in career[:3]:
            jt = job.get("title", "")
            comp = job.get("company", "")
            desc = job.get("description", "")
            role_p = []
            if jt:
                if comp: role_p.append(f"{jt} at {comp}")
                else: role_p.append(jt)
            if desc:
                clean_desc = re.sub(r'\s+', ' ', desc).strip()
                role_p.append(f"({clean_desc[:150]}...)")
            if role_p:
                recent_roles.append(" ".join(role_p))
        career_str = ". ".join(recent_roles)
        
        candidate_text = f"{title}. {headline}. {summary}. Skills: {skills_str}. Recent: {career_str}."
        cross_encoder_texts.append(candidate_text)
        
        if (idx + 1) % 2000 == 0:
            print(f"  Processed {idx + 1} candidates...")

    # Load Cross-Encoder and generate labels
    from sentence_transformers.cross_encoder import CrossEncoder
    cross_encoder_path = Path("./model_cache/cross-encoder-ms-marco-MiniLM-L-6-v2")
    
    if not cross_encoder_path.exists():
        print("Cross-encoder cache not found. Downloading...")
        cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)
        cross_encoder.save(str(cross_encoder_path))
    else:
        cross_encoder = CrossEncoder(str(cross_encoder_path), device=device)

    print("Generating labels using local Cross-Encoder...")
    pairs = [(query_text, text) for text in cross_encoder_texts]
    
    # Predict in batches
    t0 = datetime.datetime.now()
    ce_scores = cross_encoder.predict(pairs, batch_size=64, show_progress_bar=True)
    elapsed = (datetime.datetime.now() - t0).total_seconds()
    print(f"Finished labeling in {elapsed:.1f} seconds.")

    # Build X and y
    X = np.array(candidate_features, dtype=np.float32)
    y = np.array(ce_scores, dtype=np.float32)

    # Train student model using scikit-learn
    print("Training student Gradient Boosting model...")
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import mean_squared_error, r2_score
    from ranking.leakage import compute_content_hash, validate_splits_leakage

    # Audit and prevent data leakage via content-hash group splitting
    groups = [compute_content_hash(text) for text in cross_encoder_texts]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups=groups))
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    train_cands = [top_n_pool[i][3] for i in train_idx]
    val_cands = [top_n_pool[i][3] for i in val_idx]
    leakage_issues = validate_splits_leakage(train_cands, val_cands)
    if leakage_issues:
        print(f"[Leakage Warning] {len(leakage_issues)} issues found between splits: {leakage_issues}")
    else:
        print("[Leakage Audit] PASS: Verified zero candidate ID or resume content leakage between train and val splits.")

    model_student = GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=4,
        min_samples_split=4,
        random_state=42
    )
    
    model_student.fit(X_train, y_train)
    
    # Evaluate
    val_preds = model_student.predict(X_val)
    val_mse = mean_squared_error(y_val, val_preds)
    val_r2 = r2_score(y_val, val_preds)
    print(f"Validation Performance: MSE={val_mse:.4f} | R2 Score={val_r2:.4f}")

    # Train on full dataset
    print("Fitting model on full dataset...")
    model_student.fit(X, y)

    # Save student model weights
    out_pkl = Path(args.output)
    print(f"Saving student model weights to {out_pkl}...")
    with open(out_pkl, "wb") as f:
        pickle.dump(model_student, f)
        
    print("Student ranker model successfully trained and serialized.")

if __name__ == "__main__":
    main()
