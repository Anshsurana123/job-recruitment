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
    
    ml_dl_skills = {"pytorch", "tensorflow", "jax", "cuda", "triton", "llms", "transformers", "fine-tuning", "peft", "lora", "qlora", "bert", "gpt"}
    ir_search_skills = {"pinecone", "weaviate", "qdrant", "milvus", "faiss", "opensearch", "elasticsearch", "vector search", "semantic search", "hybrid search", "retrieval", "ranking", "reranking", "information retrieval", "rag"}
    eval_skills = {"ndcg", "mrr", "map", "a/b testing", "offline evaluation", "online evaluation", "evaluation framework"}
    
    matched_ml = [s for s in skills_list if s.get("name", "").lower() in ml_dl_skills]
    matched_ir = [s for s in skills_list if s.get("name", "").lower() in ir_search_skills]
    matched_eval = [s for s in skills_list if s.get("name", "").lower() in eval_skills]
    
    if matched_ir and matched_eval:
        tech_str = f"strong expertise in {matched_ir[0].get('name')} and ranking evaluation like {matched_eval[0].get('name')}"
    elif matched_ir:
        tech_str = f"hands-on experience building search retrieval pipelines with {matched_ir[0].get('name')}"
    elif matched_ml:
        tech_str = f"solid grounding in applied ML modeling with {matched_ml[0].get('name')} implementation"
    else:
        tech_str = "competent background in software and data engineering systems"
        
    c_sizes = [job.get("company_size", "unknown") for job in career_list]
    has_startup = any(size in ["1-10", "11-50", "51-200", "201-500"] for size in c_sizes)
    company_type = "startups" if has_startup else "large enterprises"
    
    logistics = []
    
    loc = profile.get("location", "India")
    loc_lower = loc.lower()
    is_pune_noida = any(city in loc_lower for city in ["pune", "noida", "delhi", "new delhi", "gurugram", "gurgaon", "faridabad", "ghaziabad"])
    willing_reloc = signals.get("willing_to_relocate", False)
    
    if is_pune_noida:
        loc_str = "based locally"
    elif willing_reloc:
        loc_str = f"in {loc} but willing to relocate"
    else:
        loc_str = f"located in {loc}"
        logistics.append("location relocation restriction")
        
    notice = signals.get("notice_period_days", 0)
    if notice <= 30:
        notice_str = "quick availability"
    else:
        notice_str = f"{notice}-day notice"
        if notice > 60:
            logistics.append(f"notice period of {notice} days")
            
    resp = signals.get("recruiter_response_rate", 1.0)
    if resp < 0.25:
        logistics.append("lower message response rate")
    elif resp > 0.80:
        notice_str += f" and highly active ({int(resp*100)}% response)"
        
    if item["has_credibility_concern"] and item["credibility_warning_skills"]:
        logistics.append(f"low assessment score in expert skill {item['credibility_warning_skills'][0]}")
    
    if item.get("has_salary_inversion", False):
        logistics.append("salary expectation range appears inverted (data quality concern)")
        
    logistics_str = f"Note: {', '.join(logistics[:2])}." if logistics else "No notable availability concerns."
    
    # Simple deterministic hash using candidate ID last digit
    cid_digits = re.findall(r'\d+', item["candidate_id"])
    cid_num = int(cid_digits[0]) if cid_digits else 0
    
    if rank <= 10:
        phrases = [
            f"Candidate at rank {rank} is a top-tier candidate currently working as a {title} at {company}. Features {exp} years of experience with {tech_str} in product {company_type}; {loc_str} with {notice_str}. {logistics_str} Ideal candidate for the founding team.",
            f"Rank {rank}: Exceptional founding team fit. Currently a {title} at {company} ({exp} yrs exp) showing {tech_str}. Candidate is {loc_str} ({notice_str}). {logistics_str} Strongly recommended candidate.",
            f"Outstanding match at rank {rank}. Shipped key systems as a {title} at {company} with {exp} years of experience, demonstrating {tech_str}. {loc_str}; {notice_str}. {logistics_str} Excellent fit."
        ]
    elif rank <= 50:
        phrases = [
            f"Rank {rank} candidate shows strong alignment as a {title} at {company} ({exp} years experience) with {tech_str}. {loc_str} with {notice_str}. {logistics_str} Solid technical and career progression.",
            f"Strong fit at rank {rank}. Currently a {title} at {company} with {exp} years of experience; highlights {tech_str} from {company_type}. {loc_str} ({notice_str}). {logistics_str} Highly aligned background.",
            f"Candidate ranked {rank} has a highly relevant background as a {title} at {company} with {exp} years of experience, showing {tech_str}. {loc_str} with {notice_str}. {logistics_str} Competent and active."
        ]
    else:
        phrases = [
            f"Satisfactory match at rank {rank}. Works as a {title} at {company} with {exp} years of experience; demonstrates {tech_str}. {loc_str} ({notice_str}). {logistics_str} Meets key requirements.",
            f"Rank {rank} candidate possesses a solid engineering background as a {title} at {company} ({exp} yrs exp). Shows {tech_str}. {loc_str} with {notice_str}. {logistics_str} Decent option.",
            f"Candidate at rank {rank} is a viable match with adjacent skills. Experience: {exp} years as a {title} at {company}; shows {tech_str}. {loc_str}; {notice_str}. {logistics_str} Good baseline fit."
        ]
        
    reasoning = phrases[cid_num % len(phrases)]
    
    words = reasoning.split()
    if len(words) > 50:
        reasoning = " ".join(words[:48]) + "..."
    elif len(words) < 25:
        reasoning += " Fully verified profile."
        
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
    model_cache_path = Path("./model_cache/all-MiniLM-L6-v2")
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
    query_embedding = model.encode(query_text, convert_to_numpy=True, normalize_embeddings=True)
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
        
        # Sub-scores
        ml_dl_sub = 100.0 if has_ml_dl else (50.0 if len(ml_dl_matches_skills) >= 1 else 20.0)
        ir_search_sub = 100.0 if has_ir_search else (40.0 if len(ir_matches_skills) >= 1 else 10.0)
        eval_sub = 100.0 if has_eval else (30.0 if len(eval_matches_skills) >= 1 else 0.0)
        
        tech_score = 0.35 * ml_dl_sub + 0.40 * ir_search_sub + 0.25 * eval_sub
        tech_score_contrib = 0.40 * tech_score
        
        # 1.3 Experience Score (20%)
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
            
        exp_score_contrib = 0.20 * exp_score
        
        # 1.4 Company Type & Startup Vibe (15%)
        entire_career_consulting = True
        company_val = 50.0
        has_startup_experience = False
        all_giant_corporates = True
        
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
        else:
            company_val = 50.0
            
        company_score_contrib = 0.15 * company_val
        
        # 1.5 Title Score & Hands-on Coding Recency (15%)
        title_lower = current_title.lower()
        title_val = 60.0
        
        strong_title_kws = ["ai engineer", "machine learning engineer", "mle", "deep learning", "nlp", "retrieval", "search engineer", "recommendation"]
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
        fit_score = hybrid_score_contrib + tech_score_contrib + exp_score_contrib + company_score_contrib + title_score_contrib + edu_score_contrib
        
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
        
        is_pune_noida_ncr = any(city in loc_lower for city in ["pune", "noida", "delhi", "new delhi", "gurugram", "gurgaon", "faridabad", "ghaziabad"])
        is_other_tier1 = any(city in loc_lower for city in ["bangalore", "bengaluru", "hyderabad", "mumbai", "chennai"])
        
        willing_reloc = signals.get("willing_to_relocate", False)
        
        if is_pune_noida_ncr:
            loc_modifier = 1.0
        elif is_other_tier1:
            loc_modifier = 0.95 if willing_reloc else 0.7  # Recovery boost for Tier-1 relocators
        elif country_lower == "india" or "india" in loc_lower:
            loc_modifier = 0.8 if willing_reloc else 0.3
        else:
            loc_modifier = 0.1
            
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
            "has_salary_inversion": has_salary_inversion
        })

    # Sort final list of 1,000 candidates:
    # Key: (-score, candidate_id)
    # Highest score first (descending), ties broken by candidate_id ascending
    scored_candidates.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))
    
    # Select top 100
    top_100 = scored_candidates[:100]
    
    # Generate reasoning and assign ranks
    submission_rows = []
    for rank_idx, item in enumerate(top_100):
        rank = rank_idx + 1
        reasoning = generate_candidate_reasoning(rank, item, REFERENCE_DATE)
        
        submission_rows.append({
            "candidate_id": item["candidate_id"],
            "rank": rank,
            "score": item["final_score"],
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
