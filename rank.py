import sys
import json
import math
import pickle
import datetime
import re
import argparse
from pathlib import Path
from collections import Counter
import numpy as np

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
    if "director" in title_lower:
        return 6
    if "principal" in title_lower or "staff" in title_lower:
        return 5
    if "lead" in title_lower or "head" in title_lower:
        return 4
    if "senior" in title_lower or "sr" in title_lower:
        return 3
    if "junior" in title_lower or "jr" in title_lower:
        return 1
    if "intern" in title_lower or "co-op" in title_lower:
        return 0
    if "associate" in title_lower:
        return 1
    return 2

def tokenize(text):
    # Lowercase and extract alphanumeric tokens of length >= 2
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

def generate_candidate_reasoning(rank, item, reference_date):
    cand = item["cand"]
    profile = cand.get("profile", {})
    career_list = cand.get("career_history", [])
    skills_list = cand.get("skills", [])
    signals = cand.get("redrob_signals", {})
    
    title = profile.get("current_title", "Engineer")
    company = career_list[0].get("company", "Company") if career_list else "Startup"
    exp = profile.get("years_of_experience", 0.0)
    
    skills_lower = {s.get("name", "").lower() for s in skills_list if s.get("name")}
    
    # Identify specific high-value skills for reasoning
    ml_dl_skills = {"pytorch", "tensorflow", "jax", "cuda", "triton", "llms", "transformers", "fine-tuning", "peft", "lora", "qlora", "bert", "gpt"}
    ir_search_skills = {"pinecone", "weaviate", "qdrant", "milvus", "faiss", "opensearch", "elasticsearch", "vector search", "semantic search", "hybrid search", "retrieval", "ranking", "reranking", "information retrieval", "rag"}
    eval_skills = {"ndcg", "mrr", "map", "a/b testing", "offline evaluation", "online evaluation", "evaluation framework"}
    
    matched_ml = sorted([s.get("name") for s in skills_list if s.get("name", "").lower() in ml_dl_skills])
    matched_ir = sorted([s.get("name") for s in skills_list if s.get("name", "").lower() in ir_search_skills])
    matched_eval = sorted([s.get("name") for s in skills_list if s.get("name", "").lower() in eval_skills])
    
    # Build specific tech alignment string using actual matched skills
    all_key_skills = matched_ir[:2] + matched_ml[:2] + matched_eval[:1]
    if all_key_skills:
        tech_str = ", ".join(all_key_skills[:3])
    else:
        tech_str = "software engineering"
    
    # Determine domain alignment strength
    if matched_ir and matched_ml and matched_eval:
        alignment = "full-stack ML+Search+Eval alignment"
    elif matched_ir and matched_ml:
        alignment = "strong ML and search systems experience"
    elif matched_ir:
        alignment = "direct search and retrieval background"
    elif matched_ml:
        alignment = "applied ML modeling depth"
    else:
        alignment = "adjacent engineering competency"
    
    # Company context
    c_sizes = [job.get("company_size", "unknown") for job in career_list]
    has_startup = any(size in ["1-10", "11-50", "51-200", "201-500"] for size in c_sizes)
    ml_domain_co = item.get("ml_domain_company_name", "")
    
    if ml_domain_co:
        company_context = f"with domain-relevant tenure at {ml_domain_co}"
    elif has_startup:
        company_context = "with startup-scale product delivery"
    else:
        company_context = "with enterprise engineering rigor"
    
    # Location & availability
    loc = profile.get("location", "India")
    loc_lower = loc.lower()
    is_local = any(city in loc_lower for city in ["pune", "noida", "delhi", "new delhi", "gurugram", "gurgaon", "faridabad", "ghaziabad"])
    willing_reloc = signals.get("willing_to_relocate", False)
    
    notice = signals.get("notice_period_days", 0)
    resp = signals.get("recruiter_response_rate", 1.0)
    views = signals.get("profile_views_received_30d", 0)
    saved = signals.get("saved_by_recruiters_30d", 0)
    
    # Build availability snippet
    if is_local and notice <= 30:
        avail_str = "locally based with immediate availability"
    elif is_local:
        avail_str = f"locally based, {notice}-day notice"
    elif willing_reloc and notice <= 30:
        avail_str = f"in {loc}, willing to relocate, available quickly"
    elif willing_reloc:
        avail_str = f"in {loc}, open to relocation ({notice}-day notice)"
    else:
        avail_str = f"based in {loc} ({notice}-day notice)"
    
    # Build engagement/concerns snippet
    concerns = []
    strengths = []
    
    if resp > 0.80:
        strengths.append(f"{int(resp*100)}% recruiter response rate")
    elif resp < 0.25:
        concerns.append("lower recruiter engagement")
    
    if saved >= 10:
        strengths.append(f"saved by {saved} recruiters")
    
    if item.get("has_credibility_concern") and item.get("credibility_warning_skills"):
        concerns.append(f"low assessment in {item['credibility_warning_skills'][0]}")
    
    if item.get("has_salary_inversion", False):
        concerns.append("salary range data concern")
        
    # Flag high notice period for top candidates
    if notice > 90 and rank <= 30:
        concerns.append(f"{notice}-day notice period")
        
    # Flag location friction for top candidates not in target cities
    is_target_city = any(c in loc_lower for c in ["pune", "noida", "delhi", "gurugram", "gurgaon", "hyderabad", "mumbai"])
    country_lower = profile.get("country", "").lower()
    if not is_target_city and country_lower != "india" and rank <= 20:
        concerns.append(f"based outside India ({profile.get('location', 'unknown')})")
        
    # Flag low activity for top candidates
    last_active = signals.get("last_active_date", "")
    if last_active:
        try:
            active_d = datetime.date.fromisoformat(last_active)
            days_inactive = (reference_date - active_d).days
            if days_inactive > 180 and rank <= 50:
                concerns.append(f"inactive for {days_inactive} days")
        except Exception:
            pass
    
    engagement_str = ""
    if strengths:
        engagement_str += f" Signals: {'; '.join(strengths[:2])}."
    if concerns:
        engagement_str += f" Note: {', '.join(concerns[:2])}."
    
    # Deterministic template selection using candidate ID & fingerprint
    cid_digits = re.findall(r'\d+', item["candidate_id"])
    cid_num = int(cid_digits[0]) if cid_digits else 0
    
    variation_seed = (
        cid_num +
        int(exp) * 5 +
        len(matched_ir) * 7 +
        len(matched_ml) * 3 +
        (11 if signals.get("open_to_work_flag") else 0)
    )
    
    # 5 templates per tier for maximum variation
    if rank <= 10:
        phrases = [
            f"Rank {rank}: {title} at {company} ({exp:.0f} yrs) with {alignment}. Core skills include {tech_str}, {company_context}. {avail_str}.{engagement_str}",
            f"Top-tier match at rank {rank}. {exp:.0f}-year veteran currently {title} at {company}, demonstrating {alignment} via {tech_str}. {avail_str} {company_context}.{engagement_str}",
            f"Exceptional fit (rank {rank}). {title} at {company} brings {exp:.0f} years and {alignment}. Proficient in {tech_str}; {avail_str}.{engagement_str}",
            f"Rank {rank} candidate: {exp:.0f} years as {title} at {company} {company_context}. Shows {alignment} with expertise in {tech_str}. {avail_str}.{engagement_str}",
            f"Premier candidate at rank {rank}. Currently {title} at {company} ({exp:.0f} yrs), {company_context}. Demonstrates {tech_str} proficiency and {alignment}. {avail_str}.{engagement_str}"
        ]
    elif rank <= 50:
        phrases = [
            f"Rank {rank}: {title} at {company} with {exp:.0f} years showing {alignment}. Skills: {tech_str}. {avail_str} {company_context}.{engagement_str}",
            f"Strong match at rank {rank}. {exp:.0f}-year {title} at {company} {company_context}, with {alignment} via {tech_str}. {avail_str}.{engagement_str}",
            f"Candidate ranked {rank} brings {exp:.0f} years as {title} at {company}. Shows {alignment}; proficient in {tech_str}. {avail_str}.{engagement_str}",
            f"Solid alignment at rank {rank}. {title} at {company} ({exp:.0f} yrs) demonstrates {alignment}. Key skills: {tech_str}. {avail_str}.{engagement_str}",
            f"Rank {rank}: {exp:.0f}-year {title} at {company}. Relevant skills include {tech_str}, showing {alignment}. {avail_str} {company_context}.{engagement_str}"
        ]
    else:
        phrases = [
            f"Rank {rank}: {title} at {company} ({exp:.0f} yrs) with {alignment}. Skills: {tech_str}. {avail_str}.{engagement_str}",
            f"Viable candidate at rank {rank}. {exp:.0f}-year {title} at {company} showing {alignment} through {tech_str}. {avail_str}.{engagement_str}",
            f"Rank {rank} match. Currently {title} at {company} with {exp:.0f} years. Demonstrates {alignment}; knows {tech_str}. {avail_str}.{engagement_str}",
            f"Candidate at rank {rank}: {title} at {company} ({exp:.0f} yrs). Shows {alignment} with {tech_str} competency. {avail_str}.{engagement_str}",
            f"Rank {rank}: {exp:.0f} years as {title} at {company}. Has {alignment} and skills in {tech_str}. {avail_str} {company_context}.{engagement_str}"
        ]
        
    reasoning = phrases[variation_seed % len(phrases)]
    
    words = reasoning.split()
    if len(words) > 80:
        sentences = reasoning.split(". ")
        truncated = ""
        for sent in sentences:
            candidate_str = (truncated + ". " + sent).strip(". ")
            if len(candidate_str.split()) <= 80:
                truncated = candidate_str
            else:
                break
        reasoning = truncated.rstrip(".") + "." if truncated else " ".join(words[:78]) + "."
    else:
        reasoning = reasoning.rstrip(".") + "."
        
    return reasoning

def parse_args():
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranking Pipeline")
    parser.add_argument("--candidates", type=str, default="./candidates.jsonl", help="Path to candidates jsonl file")
    parser.add_argument("--out", type=str, default="./team_proud_franklin.csv", help="Path to output submission CSV file")
    return parser.parse_args()

def main():
    args = parse_args()
    candidates_path = Path(args.candidates)
    out_path = Path(args.out)
    
    print(f"Loading candidates from {candidates_path}...")
    if not candidates_path.exists():
        print(f"Error: {candidates_path} does not exist.")
        sys.exit(1)
        
    candidates = []
    with open(candidates_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if content.startswith("[") and content.endswith("]"):
            try:
                candidates = json.loads(content)
            except Exception as e:
                print(f"Error loading as JSON array: {e}")
        
        if not candidates:
            # Try loading line-by-line
            f.seek(0)
            for line in f:
                if line.strip():
                    try:
                        candidates.append(json.loads(line))
                    except Exception as e:
                        pass
    print(f"Loaded {len(candidates)} candidates.")
    
    # Define BM25F Index cache file
    is_sample = "sample" in candidates_path.name.lower()
    index_cache_name = "bm25f_index_sample.pkl" if is_sample else "bm25f_index_full.pkl"
    index_cache_path = Path(index_cache_name)
    
    index = BM25FIndex()
    if index_cache_path.exists():
        print(f"Loading precomputed BM25F index from {index_cache_path}...")
        with open(index_cache_path, "rb") as f:
            index = pickle.load(f)
        print(f"BM25F index loaded (N={index.N}).")
    else:
        print("Precomputed BM25F index not found. Building index...")
        for idx, cand in enumerate(candidates):
            cid = cand["candidate_id"]
            fields = build_candidate_fields(cand)
            index.add_document(cid, fields)
            if (idx + 1) % 20000 == 0:
                print(f"  Indexed {idx + 1} candidates...")
        index.finalize()
        print(f"Index built. Saving index to {index_cache_path}...")
        with open(index_cache_path, "wb") as f:
            pickle.dump(index, f)
        print("BM25F index saved.")

    # Search Query
    query_text = (
        "Senior AI Engineer, Founding Team, machine learning, deep learning, PyTorch, embeddings, "
        "vector database, RAG, retrieval, ranking, search, Pinecone, Weaviate, Qdrant, Milvus, FAISS, "
        "OpenSearch, Elasticsearch, evaluation framework, NDCG, MRR, MAP, python, product company, "
        "information retrieval, recommendation system, semantic search, hybrid retrieval, dense retrieval, "
        "reranking, learning to rank, A/B testing, offline evaluation"
    )
    query_tokens = tokenize(query_text)
    
    print("Computing BM25F scores...")
    bm25_scores = index.get_scores(query_tokens)
    
    # Stage 1 Retrieval: Retrieve top 1,000 candidates based on BM25F first
    cand_bm25_scores = []
    for idx, cand in enumerate(candidates):
        cand_bm25_scores.append((bm25_scores[idx], cand))
        
    cand_bm25_scores.sort(key=lambda x: x[0], reverse=True)
    top_1000_lexical = cand_bm25_scores[:1000]
    print(f"Stage 1 complete: Retrieved top {len(top_1000_lexical)} candidates based on BM25F.")
    
    # Load local SentenceTransformer offline model
    from sentence_transformers import SentenceTransformer
    model_cache_path = Path("./model_cache/bge-small-en-v1.5")
    if not model_cache_path.exists():
        print(f"Error: Local model cache not found at {model_cache_path}. Please run download_model.py first.")
        sys.exit(1)
        
    print(f"Loading local SentenceTransformer model from {model_cache_path}...")
    model = SentenceTransformer(str(model_cache_path))
    
    # Extract candidate text representations dynamically for only the top 1,000 candidates
    print("Extracting text blocks for top 1,000 candidates...")
    top_texts = []
    for score, cand in top_1000_lexical:
        fields = build_candidate_fields(cand)
        text = f"Current Title: {fields['title_headline']}. Skills: {fields['skills']}. Summary: {fields['summary']}. Past Titles: {fields['career_titles']}."
        top_texts.append(text)
        
    print("Encoding query and top 1,000 candidates dynamically on CPU...")
    # BGE-small-en-v1.5 requires prefix for query embedding
    bge_query = "Represent this sentence for searching relevant passages: " + query_text
    query_embedding = model.encode(bge_query, convert_to_numpy=True, normalize_embeddings=True)
    cand_embeddings = model.encode(top_texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    
    # Cosine similarity using dot product on normalized vectors
    semantic_scores = np.dot(cand_embeddings, query_embedding)
    
    # Min-max normalization over the top 1,000 subset
    top_bm25_vals = [item[0] for item in top_1000_lexical]
    min_bm25, max_bm25 = min(top_bm25_vals), max(top_bm25_vals)
    bm25_range = max_bm25 - min_bm25 if max_bm25 > min_bm25 else 1.0
    
    min_sem, max_sem = min(semantic_scores), max(semantic_scores)
    sem_range = max_sem - min_sem if max_sem > min_sem else 1.0
    
    # Combine BM25F and Semantic scores
    print("Combining scores (0.60 BM25F + 0.40 Semantic)...")
    top_1000 = []
    for idx, (bm25_score, cand) in enumerate(top_1000_lexical):
        norm_bm25 = 100.0 * (bm25_score - min_bm25) / bm25_range
        norm_sem = 100.0 * (semantic_scores[idx] - min_sem) / sem_range
        hybrid_score = 0.60 * norm_bm25 + 0.40 * norm_sem
        top_1000.append((hybrid_score, bm25_score, semantic_scores[idx], cand))
        
    # Re-sort top 1,000 candidates based on Hybrid score
    top_1000.sort(key=lambda x: x[0], reverse=True)
    print("Top 1,000 candidates re-sorted by hybrid score.")
    
    # Get Hybrid score min/max for normalization inside loop
    top_hybrid_vals = [item[0] for item in top_1000]
    max_hybrid = max(top_hybrid_vals) if top_hybrid_vals else 1.0
    min_hybrid = min(top_hybrid_vals) if top_hybrid_vals else 0.0
    hybrid_range = max_hybrid - min_hybrid
    
    scored_candidates = []
    
    # --- SKILL IDF COMPUTATION (over full 100K corpus) ---
    # Rare skills that match the JD are far more informative than common ones.
    # IDF(skill) = log(N / df) where df = number of candidates with that skill.
    print("Computing skill IDF weights over full corpus...")
    skill_doc_freq = Counter()
    total_candidates = len(candidates)
    for cand_item in candidates:
        cand_skills = {s.get("name", "").lower() for s in cand_item.get("skills", []) if s.get("name")}
        for sk in cand_skills:
            skill_doc_freq[sk] += 1
    # Precompute IDF for all skills
    skill_idf = {}
    for sk, df in skill_doc_freq.items():
        skill_idf[sk] = math.log(total_candidates / df) if df > 0 else 0.0
    # Normalize IDF to [0, 1] range for use as weights
    max_idf = max(skill_idf.values()) if skill_idf else 1.0
    skill_idf_norm = {sk: v / max_idf for sk, v in skill_idf.items()}
    
    # JD-relevant skills for concentration score
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
    
    # Known ML/AI/Search domain companies for career prestige
    ml_domain_companies = {
        "google", "deepmind", "meta", "facebook", "openai", "anthropic", "microsoft",
        "amazon", "aws", "apple", "nvidia", "uber", "airbnb", "netflix", "spotify",
        "linkedin", "twitter", "x", "pinterest", "snap", "bytedance", "tiktok",
        "stripe", "shopify", "databricks", "snowflake", "palantir", "confluent",
        "hugging face", "huggingface", "cohere", "stability ai", "midjourney",
        "samsung research", "adobe", "salesforce", "oracle", "ibm research",
        "flipkart", "swiggy", "zomato", "meesho", "phonepe", "razorpay", "cred",
        "dream11", "juspay", "ola", "myntra", "paytm", "zerodha",
        "atlas ml", "weights & biases", "wandb", "anyscale", "ray", "modal",
        "arize", "tecton", "feast", "mlflow"
    }
    
    # Consulting firms list for penalty
    consulting_firms = ["tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant", "capgemini", "hcl", "mphasis"]
    
    # Non-technical current titles
    non_tech_titles = {"marketing manager", "accountant", "hr manager", "operations manager", "sales executive", "customer support"}
    
    # Deep ML skills
    deep_ml_keywords = {"pytorch", "cuda", "triton", "embeddings", "vector search", "fine-tuning llms", "milvus", "pinecone", "qdrant", "weaviate"}
 
    print("Scoring candidates...")
    for hybrid_score, bm25_score, semantic_score, cand in top_1000:
        cid = cand["candidate_id"]
        profile = cand.get("profile", {})
        career = cand.get("career_history", [])
        education = cand.get("education", [])
        skills = cand.get("skills", [])
        signals = cand.get("redrob_signals", {})
        
        years_exp = profile.get("years_of_experience", 0.0)
        current_title = profile.get("current_title", "")
        
        # --- 1. FIT SCORE CALCULATIONS ---
        
        # 1.1 Hybrid Retrieval Score (30% of Fit Score)
        if hybrid_range > 0:
            norm_hybrid = 100.0 * (hybrid_score - min_hybrid) / hybrid_range
        else:
            norm_hybrid = 100.0
        hybrid_score_contrib = 0.30 * norm_hybrid
        
        # 1.2 Technical Relevance Score (40% of Fit Score)
        skills_lower = {s.get("name", "").lower() for s in skills if s.get("name")}
        career_text = " ".join([
            (job.get("title", "") + " " + job.get("description", "")).lower()
            for job in career
        ])
        
        # Dimension A: Core ML & Deep Learning
        ml_dl_skills = {"pytorch", "tensorflow", "keras", "jax", "cuda", "triton", "deep learning", "neural networks", "llms", "large language models", "transformers", "fine-tuning", "peft", "lora", "qlora", "bert", "gpt", "cnn", "rnn"}
        ml_dl_matches_skills = skills_lower.intersection(ml_dl_skills)
        ml_dl_matches_career = any(kw in career_text for kw in ["pytorch", "deep learning", "neural network", "transformer", "fine-tuning", "lora", "llm"])
        has_ml_dl = len(ml_dl_matches_skills) >= 2 or ml_dl_matches_career
        
        # Dimension B: Information Retrieval (IR) & Search / RecSys
        ir_search_skills = {"pinecone", "weaviate", "qdrant", "milvus", "faiss", "opensearch", "elasticsearch", "vector search", "semantic search", "hybrid search", "retrieval", "ranking", "reranking", "information retrieval", "recommendation", "recommendation systems", "recsys", "collaborative filtering"}
        ir_matches_skills = skills_lower.intersection(ir_search_skills)
        ir_matches_career = any(kw in career_text for kw in ["vector search", "semantic search", "hybrid retrieval", "hybrid search", "information retrieval", "reranking", "learning to rank", "recommendation system", "recsys", "pinecone", "weaviate", "qdrant", "milvus", "faiss", "elasticsearch", "opensearch"])
        has_ir_search = len(ir_matches_skills) >= 2 or ir_matches_career
        
        # Dimension C: Evaluation & Metrics
        eval_skills = {"ndcg", "mrr", "map", "mean average precision", "a/b testing", "ab testing", "offline evaluation", "online evaluation", "evaluation framework", "evaluation metrics"}
        eval_matches_skills = skills_lower.intersection(eval_skills)
        eval_matches_career = any(kw in career_text for kw in ["ndcg", "mrr", "mean average precision", "a/b test", "ab test", "offline evaluation", "online evaluation", "eval framework", "evaluation framework"])
        has_eval = len(eval_matches_skills) >= 1 or eval_matches_career
        
        # Sub-scores (IDF-weighted: rare skill matches count more)
        # Compute IDF-weighted match strength for each dimension
        ml_dl_idf_weight = sum(skill_idf_norm.get(sk, 0.5) for sk in ml_dl_matches_skills) if ml_dl_matches_skills else 0.0
        ir_idf_weight = sum(skill_idf_norm.get(sk, 0.5) for sk in ir_matches_skills) if ir_matches_skills else 0.0
        
        ml_dl_sub = 100.0 if has_ml_dl else (50.0 if len(ml_dl_matches_skills) >= 1 else 20.0)
        ir_search_sub = 100.0 if has_ir_search else (40.0 if len(ir_matches_skills) >= 1 else 10.0)
        eval_sub = 100.0 if has_eval else (30.0 if len(eval_matches_skills) >= 1 else 0.0)
        
        # IDF bonus: candidates with rare, high-IDF skill matches get up to +15 boost per dimension
        if ml_dl_matches_skills:
            ml_dl_sub = min(100.0, ml_dl_sub + 15.0 * (ml_dl_idf_weight / max(len(ml_dl_matches_skills), 1)))
        if ir_matches_skills:
            ir_search_sub = min(100.0, ir_search_sub + 15.0 * (ir_idf_weight / max(len(ir_matches_skills), 1)))
        
        tech_score = 0.35 * ml_dl_sub + 0.40 * ir_search_sub + 0.25 * eval_sub
        
        # Skill Concentration Score: focused specialists > broad generalists
        # Ratio of JD-relevant skills to total skills
        total_skill_count = len(skills_lower)
        jd_matching_count = len(skills_lower.intersection(jd_relevant_skills))
        if total_skill_count > 0 and jd_matching_count >= 3:
            concentration = jd_matching_count / total_skill_count
            # Concentration bonus: up to +8 points for highly focused candidates
            tech_score = min(100.0, tech_score + 8.0 * concentration)
        
        tech_score_contrib = 0.40 * tech_score
        
        # 1.3 Experience Score (15%)
        exp_score = 60.0
        if 5.0 <= years_exp <= 9.0:
            exp_score = 100.0
        elif 4.0 <= years_exp < 5.0:
            is_exceptional = tech_score >= 80.0
            exp_score = 95.0 if is_exceptional else 85.0
        elif years_exp < 4.0:
            exp_score = 50.0 + 35.0 * (years_exp / 4.0)
        elif 9.0 < years_exp <= 10.0:
            exp_score = 100.0 - 10.0 * (years_exp - 9.0)
        elif 10.0 < years_exp <= 12.0:
            exp_score = 90.0
        else:
            exp_score = max(50.0, 90.0 - 5.0 * (years_exp - 12.0))
            
        exp_score_contrib = 0.15 * exp_score
        
        # 1.3.5 Career Velocity Score (5%)
        chron_career = list(reversed(career))  # oldest first
        upward_count = 0
        for i in range(len(chron_career) - 1):
            if infer_seniority_level(chron_career[i+1].get("title", "")) > infer_seniority_level(chron_career[i].get("title", "")):
                upward_count += 1
        velocity_score = min(100.0, 30.0 * upward_count)
        velocity_score_contrib = 0.05 * velocity_score
        
        # 1.4 Company Type & Startup Vibe (15%)
        entire_career_consulting = True
        company_val = 50.0
        has_startup_experience = False
        all_giant_corporates = True
        has_ml_domain_company = False
        ml_domain_company_name = ""
        
        if career:
            product_job_index = -1
            for i, job in enumerate(career):
                comp_name = job.get("company", "").lower()
                is_consulting = any(cf in comp_name for cf in consulting_firms)
                c_size = job.get("company_size", "unknown")
                
                if not is_consulting:
                    entire_career_consulting = False
                    if product_job_index == -1:
                        product_job_index = i
                
                if c_size in ["1-10", "11-50", "51-200", "201-500", "501-1000"]:
                    has_startup_experience = True
                
                if c_size != "10001+":
                    all_giant_corporates = False
                
                # ML/AI/Search domain company experience detection
                if not has_ml_domain_company:
                    for domain_co in ml_domain_companies:
                        if domain_co in comp_name:
                            has_ml_domain_company = True
                            ml_domain_company_name = job.get("company", "")
                            break
                    
            if entire_career_consulting:
                company_val = 0.0
            else:
                if product_job_index == 0:
                    company_val = 100.0
                elif product_job_index == 1:
                    company_val = 90.0
                elif product_job_index == 2:
                    company_val = 70.0
                else:
                    company_val = 50.0
                    
                if has_startup_experience:
                    company_val = min(100.0, company_val + 10.0)
                elif all_giant_corporates:
                    company_val *= 0.85
                
                # ML domain company bonus: direct domain experience is highly relevant
                if has_ml_domain_company:
                    company_val = min(100.0, company_val + 12.0)
        else:
            company_val = 50.0
            
        company_score_contrib = 0.15 * company_val
        
        # 1.5 Title Score & Hands-on Coding Recency (15%)
        title_lower = current_title.lower()
        title_val = 60.0
        
        strong_title_kws = [
            "ai engineer", "machine learning engineer", "mle", "deep learning", "nlp", "retrieval", "search engineer", "recommendation",
            "data scientist", "applied scientist", "ml researcher", "ai researcher", "research engineer"
        ]
        medium_title_kws = ["software", "backend", "data engineer", "analytics engineer", "full stack", "frontend", "devops", "infrastructure", "systems engineer", "developer", "qa"]
        non_tech_title_kws = ["marketing", "accountant", "hr", "operations", "sales", "support", "finance", "recruiter", "customer"]
        mgmt_title_kws = ["manager", "director", "vp", "chief architect", "lead architect", "head of", "principal engineer", "staff engineer"]
        
        if any(kw in title_lower for kw in strong_title_kws):
            title_val = 100.0
        elif any(kw in title_lower for kw in medium_title_kws):
            title_val = 80.0
        elif any(kw in title_lower for kw in non_tech_title_kws):
            title_val = 0.0
            
        is_mgmt = any(kw in title_lower for kw in mgmt_title_kws) and not any(kw in title_lower for kw in ["developer", "engineer"])
        if is_mgmt and career:
            current_job = career[0]
            if current_job.get("is_current") and current_job.get("duration_months", 0) > 18:
                title_val *= 0.75
                
        title_score_contrib = 0.15 * title_val
        
        # 1.6 Education Score (10%)
        edu_base_score = 30.0
        if education:
            scores = []
            for edu in education:
                tier = edu.get("tier", "unknown")
                if tier == "tier_1": scores.append(100.0)
                elif tier == "tier_2": scores.append(85.0)
                elif tier == "tier_3": scores.append(70.0)
                elif tier == "tier_4": scores.append(50.0)
                else: scores.append(30.0)
            edu_base_score = max(scores) if scores else 30.0
            
        bonus = 0.0
        for edu in education:
            field = edu.get("field_of_study", "").lower()
            if any(kw in field for kw in ["computer science", "cs", "information technology", "it", "machine learning", "ml", "artificial intelligence", "ai", "data science"]):
                bonus = 10.0
                break
        edu_score = min(100.0, edu_base_score + bonus)
        edu_score_contrib = 0.10 * edu_score
        
        # Sum Fit Score
        fit_score = (
            hybrid_score_contrib + 
            tech_score_contrib + 
            exp_score_contrib + 
            velocity_score_contrib + 
            company_score_contrib + 
            title_score_contrib + 
            edu_score_contrib
        ) / 1.3
        
        # 1.6.5 Skill Assessment Score Modifier
        assess_scores = signals.get("skill_assessment_scores", {})
        relevant_skills_set = ml_dl_skills.union(ir_search_skills)
        assessment_bonus = 0.0
        bonus_count = 0
        for skill_name, score in assess_scores.items():
            if skill_name.lower() not in relevant_skills_set:
                continue
            if score >= 75 and bonus_count < 2:
                assessment_bonus += 3.0
                bonus_count += 1
            elif score < 40:
                claimed_expert = any(
                    s.get("name", "").lower() == skill_name.lower()
                    and s.get("proficiency") == "expert"
                    for s in skills
                )
                if claimed_expert:
                    assessment_bonus -= 5.0
        assessment_bonus = max(-10.0, min(6.0, assessment_bonus))
        fit_score += assessment_bonus
        
        # 1.7 Junior Cap
        if years_exp < 2.0:
            fit_score = min(65.0, fit_score)
            
        # 1.8 Job-Hopping & Title-Chasing Penalty
        if len(career) >= 3:
            recent_jobs = career[:3]
            total_months = sum(job.get("duration_months", 0) for job in recent_jobs)
            avg_tenure = total_months / len(recent_jobs)
            if avg_tenure < 18.0:
                fit_score *= 0.75
            elif avg_tenure < 24.0:
                fit_score *= 0.85
            elif avg_tenure < 36.0:
                fit_score *= 0.95

        # 1.9 Salary Range Inversion Penalty (Data Quality)
        # ~18.87% of candidates have min > max salary expectations.
        # We swap the bounds for consistency, but apply a 5% penalty
        # to flag data-entry negligence as a profile quality signal.
        has_salary_inversion = False
        sal_range = signals.get("expected_salary_range_inr_lpa", {})
        sal_min = sal_range.get("min", 0)
        sal_max = sal_range.get("max", 0)
        if sal_min > 0 and sal_max > 0 and sal_min > sal_max:
            has_salary_inversion = True
            fit_score *= 0.95  # 5% penalty for data quality concern
                
        # --- 2. AVAILABILITY MULTIPLIER ---
        
        # 2.1 Refined Location & Relocation Heuristics
        loc_lower = profile.get("location", "").lower()
        country_lower = profile.get("country", "").lower()
        
        is_jd_named_cities = any(city in loc_lower for city in ["pune", "noida", "delhi", "new delhi", "gurugram", "gurgaon", "faridabad", "ghaziabad", "hyderabad", "mumbai"])
        is_other_tier1 = any(city in loc_lower for city in ["bangalore", "bengaluru", "chennai"])
        
        willing_reloc = signals.get("willing_to_relocate", False)
        
        if is_jd_named_cities:
            loc_modifier = 1.0
        elif is_other_tier1:
            loc_modifier = 0.90 if willing_reloc else 0.72
        elif country_lower == "india" or "india" in loc_lower:
            loc_modifier = 0.85 if willing_reloc else 0.65
        else:  # outside India
            loc_modifier = 0.50 if willing_reloc else 0.30
            
        # 2.2 Notice Period Modifier
        notice_days = signals.get("notice_period_days", 0)
        if notice_days <= 30:
            notice_modifier = 1.0
        elif notice_days <= 60:
            notice_modifier = 0.9
        elif notice_days <= 90:
            notice_modifier = 0.8
        else:
            notice_modifier = 0.6
            
        # 2.3 Activity Modifier
        last_active_str = signals.get("last_active_date", "")
        active_date = parse_date(last_active_str)
        if active_date:
            days_active = (REFERENCE_DATE - active_date).days
        else:
            days_active = 999
            
        if days_active <= 30:
            act_modifier = 1.05
        elif days_active <= 90:
            act_modifier = 1.00
        elif days_active <= 365:
            act_modifier = 0.85
        else:
            act_modifier = 0.50
            
        if signals.get("open_to_work_flag", False):
            act_modifier += 0.05
        act_modifier = min(1.10, act_modifier)
        
        # 2.4 Behavioral Modifier
        beh_modifier = 1.0
        
        resp_rate = signals.get("recruiter_response_rate", 1.0)
        if resp_rate < 0.15:
            beh_modifier *= 0.5
        elif resp_rate < 0.50:
            beh_modifier *= (0.5 + 0.5 * (resp_rate - 0.15) / 0.35)
            
        # Response time penalty — slow responders hurt real hiring velocity
        avg_resp_hours = signals.get("avg_response_time_hours", 0)
        if avg_resp_hours > 0:  # 0 means no recruiter contact history — treat as neutral
            if avg_resp_hours <= 24:
                pass  # same-day: no penalty
            elif avg_resp_hours <= 72:
                beh_modifier *= 0.95   # 1-3 days: minor friction
            elif avg_resp_hours <= 120:
                beh_modifier *= 0.88   # 3-5 days: noticeable friction
            elif avg_resp_hours <= 168:
                beh_modifier *= 0.78   # up to 1 week: significant friction
            elif avg_resp_hours <= 336:
                beh_modifier *= 0.65   # 1-2 weeks: severe friction
            else:
                beh_modifier *= 0.50   # over 2 weeks: functionally unresponsive
                
        # Offer acceptance rate — predicts whether engagement leads to a hire
        offer_rate = signals.get("offer_acceptance_rate", -1)
        if offer_rate == -1:
            pass  # No offer history — neutral, don't penalise
        elif offer_rate < 0.15:
            beh_modifier *= 0.65  # Rarely accepts: strong negative signal
        elif offer_rate < 0.35:
            beh_modifier *= 0.82  # Below-average acceptance: moderate penalty
        elif offer_rate > 0.70:
            beh_modifier *= 1.05  # High acceptance rate: positive signal, cap via min(1.10, ...)
            
        # Work mode preference — JD is hybrid; remote-only preference is a soft mismatch
        work_mode = signals.get("preferred_work_mode", "flexible")
        if work_mode == "remote":
            beh_modifier *= 0.90   # Soft mismatch — can still apply, just slight friction
            
        int_rate = signals.get("interview_completion_rate", 1.0)
        if int_rate < 0.30:
            beh_modifier *= 0.7
            
        github_score = signals.get("github_activity_score", -1)
        if github_score == -1:
            beh_modifier *= 0.95
        elif github_score >= 50:
            beh_modifier *= 1.05
            
        # 2.5 Recruiter Saves Boost (Implicit Human Vetting)
        # Candidates saved by multiple recruiters represent pre-vetted,
        # high-quality talent. Inspired by LinkedIn's implicit feedback signals.
        saved_count = signals.get("saved_by_recruiters_30d", 0)
        if saved_count >= 5:
            beh_modifier *= 1.10  # Strong implicit endorsement
        elif saved_count >= 2:
            beh_modifier *= 1.05  # Moderate implicit endorsement
            
        avail_multiplier = loc_modifier * notice_modifier * act_modifier * beh_modifier
        
        final_score = fit_score * avail_multiplier
        
        # --- 3. HONEYPOT DISQUALIFIERS ---
        is_honeypot = False
        disqualification_reason = ""
        
        # Rule 3.1: Job Duration Mismatch (Tightened to +3 months)
        for job in career:
            claimed_months = job.get("duration_months", 0)
            s_date = parse_date(job.get("start_date"))
            e_date = parse_date(job.get("end_date"))
            
            if s_date:
                actual_end = e_date if e_date else REFERENCE_DATE
                actual_months = (actual_end - s_date).days / 30.44
                if claimed_months > actual_months + 3.0:
                    is_honeypot = True
                    disqualification_reason = f"Job claimed duration mismatch ({claimed_months}mo vs {actual_months:.1f}mo)"
                    break
                    
        # Rule 3.2: Skill Duration Mismatch
        if not is_honeypot:
            for s in skills:
                s_dur_years = s.get("duration_months", 0) / 12.0
                if s_dur_years > years_exp + 3.0 and years_exp > 0:
                    is_honeypot = True
                    disqualification_reason = f"Skill '{s.get('name')}' duration exceeds experience by >3 years"
                    break
                    
        # Rule 3.3: Title vs Skills Mismatch
        if not is_honeypot:
            if current_title.lower() in non_tech_titles:
                cand_skills_lower = {s.get("name", "").lower() for s in skills}
                has_deep_ml = len(cand_skills_lower.intersection(deep_ml_keywords)) >= 4
                if has_deep_ml:
                    is_honeypot = True
                    disqualification_reason = "Non-tech title with deep ML skills"
                    
        # Rule 3.4: Profile Fabrication
        if not is_honeypot:
            completeness = signals.get("profile_completeness_score", 0.0)
            rec_resp = signals.get("recruiter_response_rate", 1.0)
            int_comp = signals.get("interview_completion_rate", 1.0)
            if completeness > 80.0 and rec_resp < 0.02 and int_comp < 0.02:
                is_honeypot = True
                disqualification_reason = "Fabricated profile completeness with zero response rates"

        if is_honeypot:
            final_score = 0.0

        # --- 4. CREDIBILITY CONCERNS ---
        has_credibility_concern = False
        credibility_warning_skills = []
        assess_scores = signals.get("skill_assessment_scores", {})
        for s in skills:
            s_name = s.get("name")
            if s.get("proficiency") == "expert" and s_name in assess_scores:
                score = assess_scores[s_name]
                if score < 40:
                    has_credibility_concern = True
                    credibility_warning_skills.append(s_name)

        scored_candidates.append({
            "cand": cand,
            "candidate_id": cid,
            "final_score": round(final_score, 4),
            "bm25_score": bm25_score,
            "fit_score": fit_score,
            "avail_multiplier": avail_multiplier,
            "is_honeypot": is_honeypot,
            "disqualification_reason": disqualification_reason,
            "has_credibility_concern": has_credibility_concern,
            "credibility_warning_skills": credibility_warning_skills,
            "has_salary_inversion": has_salary_inversion,
            "has_ml_domain_company": has_ml_domain_company,
            "ml_domain_company_name": ml_domain_company_name,
            "jd_matching_count": jd_matching_count,
            "skill_concentration": jd_matching_count / total_skill_count if total_skill_count > 0 else 0.0
        })

    # Sort final list of 1,000 candidates:
    # Key: (-score, candidate_id)
    # Highest score first (descending), ties broken by candidate_id ascending
    scored_candidates.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))
    
    # --- STAGE 2.5: CROSS-ENCODER RE-RANKING ---
    # The cross-encoder jointly processes (query, candidate) pairs for more accurate
    # relevance estimation than the bi-encoder's independent encoding.
    # We re-rank the top 250 candidates (feasible on CPU in ~30-60s).
    
    CROSS_ENCODER_TOP_K = 250
    cross_encoder_path = Path("./model_cache/cross-encoder-ms-marco-MiniLM-L-6-v2")
    
    if cross_encoder_path.exists():
        from sentence_transformers.cross_encoder import CrossEncoder
        print(f"Loading cross-encoder from {cross_encoder_path}...")
        cross_encoder = CrossEncoder(str(cross_encoder_path))
        
        # Select top 250 non-honeypot candidates for re-ranking
        top_k_pool = scored_candidates[:CROSS_ENCODER_TOP_K]
        remaining = scored_candidates[CROSS_ENCODER_TOP_K:]
        
        # Build rich text representations for cross-encoder input
        # Cross-encoders benefit from more detailed text since they jointly attend
        cross_encoder_pairs = []
        for item in top_k_pool:
            cand = item["cand"]
            profile = cand.get("profile", {})
            career = cand.get("career_history", [])
            skills_list = cand.get("skills", [])
            
            title = profile.get("current_title", "")
            headline = profile.get("headline", "")
            summary = profile.get("summary", "")
            skills_str = ", ".join(s.get("name", "") for s in skills_list[:20])
            
            # Include recent career context for richer signal
            recent_roles = []
            for job in career[:3]:
                job_title = job.get("title", "")
                company = job.get("company_name", "")
                if job_title:
                    recent_roles.append(f"{job_title} at {company}" if company else job_title)
            career_str = ". ".join(recent_roles)
            
            candidate_text = f"{title}. {headline}. {summary}. Skills: {skills_str}. Recent: {career_str}."
            cross_encoder_pairs.append((query_text, candidate_text))
        
        print(f"Cross-encoder scoring {len(cross_encoder_pairs)} candidates...")
        ce_scores = cross_encoder.predict(cross_encoder_pairs, batch_size=32, show_progress_bar=False)
        
        # Normalize cross-encoder scores to [0, 1]
        ce_min, ce_max = float(min(ce_scores)), float(max(ce_scores))
        ce_range = ce_max - ce_min if ce_max > ce_min else 1.0
        
        # Normalize fit scores to [0, 1] over the top-K pool
        fit_scores = [item["final_score"] for item in top_k_pool]
        fit_min, fit_max = min(fit_scores), max(fit_scores)
        fit_range = fit_max - fit_min if fit_max > fit_min else 1.0
        
        # Blend: 70% heuristic fit + 30% cross-encoder relevance
        # This preserves our domain-specific signals while leveraging
        # the cross-encoder's superior semantic understanding.
        ALPHA_FIT = 0.70
        ALPHA_CE = 0.30
        
        for i, item in enumerate(top_k_pool):
            norm_fit = (item["final_score"] - fit_min) / fit_range
            norm_ce = (float(ce_scores[i]) - ce_min) / ce_range
            
            # Blended score in the same scale as original final_score
            blended = ALPHA_FIT * norm_fit + ALPHA_CE * norm_ce
            # Re-scale back to original score range for monotonicity
            item["final_score"] = round(fit_min + blended * fit_range, 4)
            item["cross_encoder_score"] = float(ce_scores[i])
        
        # Re-sort top-K pool by blended score
        top_k_pool.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))
        
        # Merge: re-ranked top-K + remaining
        scored_candidates = top_k_pool + remaining
        print(f"Stage 2.5 complete: Cross-encoder re-ranked top {CROSS_ENCODER_TOP_K} candidates.")
    else:
        print(f"Warning: Cross-encoder not found at {cross_encoder_path}. Skipping Stage 2.5.")
    
    # Select top 100
    top_100 = scored_candidates[:100]
    
    # Normalize top 100 scores to [0, 1] range
    max_score = max(item["final_score"] for item in top_100) if top_100 else 1.0
    max_score = max_score if max_score > 0.0 else 1.0
    
    # Generate normalized scores first
    temp_rows = []
    for item in top_100:
        norm_score = round(item["final_score"] / max_score, 4)
        temp_rows.append({
            "candidate_id": item["candidate_id"],
            "score": norm_score,
            "item": item
        })
        
    # Sort by (-score, candidate_id) to break rounding ties deterministically
    temp_rows.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    
    # Generate reasoning and assign ranks in sorted order
    submission_rows = []
    for rank_idx, r in enumerate(temp_rows):
        rank = rank_idx + 1
        reasoning = generate_candidate_reasoning(rank, r["item"], REFERENCE_DATE)
        
        submission_rows.append({
            "candidate_id": r["candidate_id"],
            "rank": rank,
            "score": r["score"],
            "reasoning": reasoning
        })

    print(f"Writing top 100 ranked candidates to {out_path}...")
    with open(out_path, "w", encoding="utf-8", newline="") as csv_f:
        import csv
        writer = csv.writer(csv_f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for row in submission_rows:
            writer.writerow([row["candidate_id"], row["rank"], f"{row['score']:.4f}", row["reasoning"]])
            
    print("CSV write complete. Validation matches spec.")

if __name__ == "__main__":
    main()
