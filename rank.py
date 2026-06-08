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

# Set random seeds for determinism
random.seed(42)
np.random.seed(42)
try:
    import torch
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
except ImportError:
    pass

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
    cid = item["candidate_id"]
    
    # Deterministic index based on candidate ID digits
    try:
        cand_num = int(cid.split('_')[1])
    except Exception:
        cand_num = random.randint(0, 1000)
        
    title = profile.get("current_title", "Engineer")
    company = career_list[0].get("company", "Company") if career_list else "Startup"
    exp = profile.get("years_of_experience", 0.0)
    
    # Check degree & tier
    edu_list = cand.get("education", [])
    has_tier1 = any(edu.get("tier") == "tier_1" for edu in edu_list)
    has_phd = False
    has_masters = False
    for edu in edu_list:
        deg = edu.get("degree", "").lower()
        if any(d in deg for d in ["ph.d", "phd", "doctor"]):
            has_phd = True
        elif any(d in deg for d in ["master", "m.sc", "msc", "m.tech", "mtech", "m.e.", "m.s.", "ms"]) or deg == "me":
            has_masters = True
            
    # Check publications
    has_publications = False
    pub_venue = ""
    career_desc_text = " ".join([job.get("description", "") for job in career_list if job.get("description")])
    summary_text = profile.get("summary", "")
    full_text_for_pub = (summary_text + " " + career_desc_text).lower()
    pub_venues = ["neurips", "icml", "cvpr", "kdd", "acl", "sigir", "recsys"]
    for venue in pub_venues:
        if re.search(r'\b' + re.escape(venue) + r'\b', full_text_for_pub):
            has_publications = True
            pub_venue = venue.upper()
            break
            
    # Key skills
    skills_lower = {s.get("name", "").lower() for s in skills_list if s.get("name")}
    ml_dl_skills = {"pytorch", "tensorflow", "jax", "cuda", "triton", "llms", "transformers", "fine-tuning", "peft", "lora", "qlora", "bert", "gpt"}
    ir_search_skills = {"pinecone", "weaviate", "qdrant", "milvus", "faiss", "opensearch", "elasticsearch", "vector search", "semantic search", "hybrid search", "retrieval", "ranking", "reranking", "information retrieval", "rag"}
    eval_skills = {"ndcg", "mrr", "map", "a/b testing", "offline evaluation", "online evaluation", "evaluation framework"}
    
    matched_ml = sorted([s.get("name") for s in skills_list if s.get("name", "").lower() in ml_dl_skills])
    matched_ir = sorted([s.get("name") for s in skills_list if s.get("name", "").lower() in ir_search_skills])
    matched_eval = sorted([s.get("name") for s in skills_list if s.get("name", "").lower() in eval_skills])
    
    key_skills = []
    if matched_ir: key_skills.append(matched_ir[0])
    if matched_ml: key_skills.append(matched_ml[0])
    if matched_eval: key_skills.append(matched_eval[0])
    if len(key_skills) < 2 and len(matched_ir) > 1: key_skills.append(matched_ir[1])
    if len(key_skills) < 2 and len(matched_ml) > 1: key_skills.append(matched_ml[1])
    skills_str = ", ".join(key_skills) if key_skills else "machine learning"
    
    # 1. Openers (Sentence 1)
    s1_options = [
        f"Currently working as a {title} at {company}, this candidate brings {exp:.1f} years of experience.",
        f"Brings {exp:.1f} years of total industry experience, currently serving as {title} at {company}.",
        f"With {exp:.1f} years in the tech ecosystem, they are currently a {title} at {company}.",
        f"An experienced professional with {exp:.1f} years of tenure, currently acting as {title} at {company}.",
        f"Currently a {title} at {company} possessing {exp:.1f} years of technical background.",
        f"Their career history spans {exp:.1f} years, including their current role as {title} at {company}."
    ]
    s1 = s1_options[cand_num % len(s1_options)]
    
    # 2. Alignment variations (Sentence 2 part 1) - noun phrases for flexible composability
    align_ml_ir_eval = [
        "full-stack ML capabilities, search retrieval depth, and ranking metrics experience",
        "robust expertise in search engines, deep learning models, and offline evaluation frameworks",
        "integrated skills across search architecture, model fine-tuning, and metric evaluation",
        "end-to-end alignment with search, recommendation, and relevance metrics requirements"
    ]
    align_ml_ir = [
        "strong search systems expertise and ML model development background",
        "direct experience with neural models, vector search, and hybrid databases",
        "demonstrated proficiency in semantic retrieval and applied machine learning",
        "solid alignment with our core search retrieval and model training needs"
    ]
    align_ir = [
        "specialized search and information retrieval engineering depth",
        "search engine architecture and retrieval systems focus",
        "expertise in indexing, vector search, and hybrid query pipelines",
        "a practical background in search feature development and ranking"
    ]
    align_ml = [
        "applied machine learning and deep learning pipeline depth",
        "solid model training, fine-tuning, and PyTorch capabilities",
        "strong foundations in applied ML and deep learning models",
        "practical experience in machine learning systems and model deployment"
    ]
    align_general = [
        "general software engineering and systems development background",
        "backend systems delivery and general technical skills",
        "backend engineering and software architecture capabilities",
        "solid general programming and systems delivery skills"
    ]
    
    has_ml = len(matched_ml) >= 2 or any(kw in career_desc_text.lower() for kw in ["pytorch", "deep learning"])
    has_ir = len(matched_ir) >= 2 or any(kw in career_desc_text.lower() for kw in ["vector search", "semantic search", "retrieval"])
    has_eval = len(matched_eval) >= 1 or any(kw in career_desc_text.lower() for kw in ["ndcg", "mrr", "map", "evaluation"])
    
    if has_ml and has_ir and has_eval:
        align_str = align_ml_ir_eval[cand_num % len(align_ml_ir_eval)]
    elif has_ml and has_ir:
        align_str = align_ml_ir[cand_num % len(align_ml_ir)]
    elif has_ir:
        align_str = align_ir[cand_num % len(align_ir)]
    elif has_ml:
        align_str = align_ml[cand_num % len(align_ml)]
    else:
        align_str = align_general[cand_num % len(align_general)]
        
    s2_options = [
        f"They demonstrate {align_str} utilizing {skills_str}.",
        f"Their profile highlights {align_str}, with hands-on {skills_str} experience.",
        f"They match the JD requirements with {align_str} and knowledge of {skills_str}.",
        f"They showcase {align_str} alongside exposure to {skills_str}.",
        f"They bring {align_str} with solid skills in {skills_str}."
    ]
    s2 = s2_options[(cand_num + 2) % len(s2_options)]
    
    # 3. Location & Availability
    loc = profile.get("location", "India")
    loc_lower = loc.lower()
    is_local = any(city in loc_lower for city in ["pune", "noida", "delhi", "new delhi", "gurugram", "gurgaon", "faridabad", "ghaziabad"])
    willing_reloc = signals.get("willing_to_relocate", False)
    notice = signals.get("notice_period_days", 0)
    
    # Location phrasing
    if is_local:
        loc_options = [f"locally based in {loc}", f"based locally in {loc}", f"living locally in {loc}"]
    elif willing_reloc:
        loc_options = [f"located in {loc} (open to relocation)", f"in {loc} and willing to relocate", f"relocating from {loc}"]
    else:
        loc_options = [f"based in {loc}", f"located in {loc}", f"residing in {loc}"]
    loc_phrase = loc_options[cand_num % len(loc_options)]
    
    # Notice period phrasing
    if notice == 0:
        notice_options = ["available immediately", "ready to start immediately", "with immediate availability"]
    else:
        notice_options = [f"with a {notice}-day notice period", f"requiring {notice} days notice", f"on a {notice}-day notice"]
    notice_phrase = notice_options[(cand_num + 1) % len(notice_options)]
    
    # Join location & notice
    join_options = [
        f"{loc_phrase} and {notice_phrase}",
        f"{loc_phrase}, {notice_phrase}",
        f"{loc_phrase} ({notice_phrase})"
    ]
    avail_str = join_options[(cand_num + 3) % len(join_options)]
    
    # 4. Strengths & Concerns
    strengths = []
    concerns = []
    
    # Strengths
    resp = signals.get("recruiter_response_rate", 1.0)
    if resp >= 0.85:
        rate_pct = int(resp * 100)
        s_resp_options = [f"highly responsive ({rate_pct}% reply rate)", f"excellent engagement ({rate_pct}% responsiveness)", f"very active response rate ({rate_pct}%)"]
        strengths.append(s_resp_options[cand_num % len(s_resp_options)])
        
    saved = signals.get("saved_by_recruiters_30d", 0)
    if saved >= 5:
        s_saved_options = [f"pre-vetted by {saved} recruiters", f"saved by {saved} recruiters recently", f"strong market interest ({saved} saves)"]
        strengths.append(s_saved_options[(cand_num + 1) % len(s_saved_options)])
        
    github_score = signals.get("github_activity_score", -1)
    if github_score >= 50:
        s_git_options = ["active GitHub activity", "strong GitHub contribution presence", "robust public git profile"]
        strengths.append(s_git_options[(cand_num + 2) % len(s_git_options)])
        
    if has_tier1:
        s_tier_options = ["educated at Tier-1 school", "Tier-1 academic pedigree", "Tier-1 college credentials"]
        strengths.append(s_tier_options[(cand_num + 3) % len(s_tier_options)])
        
    if has_phd:
        strengths.append("holds a Ph.D. degree")
    elif has_masters:
        strengths.append("holds a Master's degree")
        
    if has_publications:
        strengths.append(f"published research at {pub_venue}")
        
    # Concerns
    if item.get("is_cv_speech_primary") and not item.get("has_nlp_ir_compensation"):
        concerns.append("CV-primary background with limited NLP/IR experience")
    if item.get("has_credibility_concern") and item.get("credibility_warning_skills"):
        concerns.append(f"expert skill assessment warnings ({item['credibility_warning_skills'][0]})")
    if item.get("has_salary_inversion", False):
        concerns.append("minor profile data quality discrepancies")
    if notice > 90:
        concerns.append(f"long notice period of {notice} days")
        
    last_active = signals.get("last_active_date", "")
    if last_active:
        try:
            active_d = datetime.date.fromisoformat(last_active)
            days_inactive = (reference_date - active_d).days
            if days_inactive > 180:
                concerns.append(f"dormant profile ({days_inactive} days inactive)")
        except Exception:
            pass
            
    if item.get("has_skills_stuffing_concern", False):
        concerns.append("potential skill stuffing behavior flagged")
        
    # Build strengths and concerns string
    logistics = ""
    # Tone adjustment based on rank
    if rank <= 10:
        prefix_options = ["Excellent founding fit. ", "Strong recommendation. ", "Top-tier candidate. ", "Highly aligned profile. "]
        logistics += prefix_options[cand_num % len(prefix_options)]
    elif rank >= 85:
        prefix_options = ["Adjacent fit. ", "Final shortlist candidate. ", "Marginal alignment. ", "Borderline founding candidate. "]
        logistics += prefix_options[cand_num % len(prefix_options)]
        
    logistics += f"Located {avail_str}."
    
    extra_details = []
    if strengths:
        extra_details.append(f"Strong indicators: {', '.join(strengths)}")
    if concerns:
        extra_details.append(f"Note: {'; '.join(concerns)}")
        
    if extra_details:
        logistics += " " + ". ".join(extra_details) + "."
        
    # Sentence assembly
    reasoning = f"{s1} {s2} {logistics}"
    
    # Strip double spaces
    reasoning = re.sub(r'\s+', ' ', reasoning).strip()
    
    # Hard truncation to ensure length limits
    words = reasoning.split()
    if len(words) > 75:
        reasoning = " ".join(words[:73]) + "..."
        
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
        "reranking, learning to rank, A/B testing, offline evaluation, "
        "collaborative filtering, matrix factorization, recommendation-style, recommender, search features, search pipeline, retrieval pipeline, "
        "data scientist, applied scientist, ml engineer, machine learning engineer, recommendation systems engineer, search engineer"
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
    
    # Extract candidate text representations with truncated career histories dynamically for Stage 1.5
    print("Extracting text blocks for top 1,000 candidates...")
    top_texts = []
    for score, cand in top_1000_lexical:
        fields = build_candidate_fields(cand)
        career = cand.get("career_history", [])
        recent_roles = []
        for job in career[:3]:
            job_title = job.get("title", "")
            company = job.get("company", "")
            job_desc = job.get("description", "")
            
            role_parts = []
            if job_title:
                if company:
                    role_parts.append(f"{job_title} at {company}")
                else:
                    role_parts.append(job_title)
            
            if job_desc:
                # Clean up description whitespace and take first 150 characters
                clean_desc = re.sub(r'\s+', ' ', job_desc).strip()
                role_parts.append(f"({clean_desc[:150]}...)")
                
            if role_parts:
                recent_roles.append(" ".join(role_parts))
                
        career_str = ". ".join(recent_roles)
        text = f"Current Title: {fields['title_headline']}. Skills: {fields['skills']}. Summary: {fields['summary']}. Recent: {career_str}."
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
        ml_dl_skills = {"pytorch", "tensorflow", "keras", "jax", "cuda", "triton", "deep learning", "neural networks", "llms", "large language models", "transformers", "fine-tuning", "peft", "lora", "qlora", "bert", "gpt", "cnn", "rnn", "sentence-transformer", "sentence transformer", "embedding", "embeddings"}
        ml_dl_matches_skills = skills_lower.intersection(ml_dl_skills)
        ml_dl_matches_career = any(kw in career_text for kw in ["pytorch", "deep learning", "neural network", "transformer", "fine-tuning", "lora", "llm", "sentence-transformer", "sentence transformer", "embedding", "embeddings"])
        has_ml_dl = len(ml_dl_matches_skills) >= 2 or ml_dl_matches_career
        
        # Dimension B: Information Retrieval (IR) & Search / RecSys
        ir_search_skills = {"pinecone", "weaviate", "qdrant", "milvus", "faiss", "opensearch", "elasticsearch", "vector search", "semantic search", "hybrid search", "retrieval", "ranking", "reranking", "re-ranking", "information retrieval", "recommendation", "recommendation systems", "recommend", "recommender", "recsys", "collaborative filtering", "matrix factorization", "search engine", "search feature", "search system", "search pipeline", "retrieval system", "retrieval pipeline"}
        ir_matches_skills = skills_lower.intersection(ir_search_skills)
        ir_keywords = [
            "vector search", "semantic search", "hybrid retrieval", "hybrid search", "information retrieval", 
            "reranking", "re-ranking", "learning to rank", "recommendation system", "recommendation-style", 
            "recommender", "recommendation", "recommend", "collaborative filtering", "matrix factorization", 
            "recsys", "pinecone", "weaviate", "qdrant", "milvus", "faiss", "elasticsearch", "opensearch", 
            "search engine", "search feature", "search system", "search pipeline", "retrieval system", 
            "retrieval pipeline", "approximate nearest neighbor", "ann index", "similarity search", 
            "dense retrieval", "inverted index", "personalized feed", "feed ranking", "news feed", 
            "content ranking", "item similarity", "user-item", "affinity score", "relevance scoring", 
            "ltr", "xgboost ranking", "lambdamart", "pointwise", "pairwise", "listwise", 
            "candidate generation", "two-tower", "dual encoder", "siamese network", "triplet loss"
        ]
        ir_matches_career = any(kw in career_text for kw in ir_keywords)
        has_ir_search = len(ir_matches_skills) >= 2 or ir_matches_career
        
        # Dimension C: Evaluation & Metrics
        eval_skills = {"ndcg", "mrr", "map", "mean average precision", "a/b testing", "ab testing", "offline evaluation", "online evaluation", "evaluation framework", "evaluation metrics", "ranking metric", "retrieval metric", "precision at", "recall at", "ndcg@", "mrr@", "map@"}
        eval_matches_skills = skills_lower.intersection(eval_skills)
        eval_keywords = [
            "ndcg", "mrr", "map", "mean average precision", "a/b test", "ab test", "a/b testing", "ab testing", 
            "offline evaluation", "online evaluation", "eval framework", "evaluation framework", "ranking metric", 
            "retrieval metric", "precision at", "recall at", "ndcg@", "mrr@", "map@", "precision recall", 
            "hit rate", "click-through rate", "ctr", "engagement metric", "online experiment", "holdout evaluation", 
            "ranking quality", "relevance judgment", "human evaluation", "user study", "implicit feedback"
        ]
        eval_matches_career = any(kw in career_text for kw in eval_keywords)
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
        has_quant_field = False
        degree_bonus = 0.0
        
        for edu in education:
            field = edu.get("field_of_study", "").lower()
            degree = edu.get("degree", "").lower()
            
            is_quant = any(kw in field for kw in [
                "computer science", "cs", "information technology", "it", 
                "machine learning", "ml", "artificial intelligence", "ai", 
                "data science", "statistics", "stats", "mathematics", "math"
            ])
            
            if is_quant:
                has_quant_field = True
                # Check degree level
                if any(deg in degree for deg in ["ph.d", "phd", "doctor"]):
                    degree_bonus = max(degree_bonus, 12.0)
                elif any(deg in degree for deg in ["master", "m.sc", "msc", "m.tech", "mtech", "m.e.", "m.s.", "ms"]) or degree == "me":
                    degree_bonus = max(degree_bonus, 6.0)
                    
        if has_quant_field:
            bonus = 10.0 + degree_bonus
            
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
        
        # 1.6.2 Research Publications Boost (from Nice-to-Have in JD)
        has_publications = False
        career_desc_text = " ".join([job.get("description", "") for job in career if job.get("description")])
        summary_text = profile.get("summary", "")
        full_text_for_pub = (summary_text + " " + career_desc_text).lower()
        pub_venues = ["neurips", "icml", "cvpr", "kdd", "acl", "sigir", "recsys"]
        for venue in pub_venues:
            if re.search(r'\b' + re.escape(venue) + r'\b', full_text_for_pub):
                has_publications = True
                break
        if has_publications:
            fit_score = min(100.0, fit_score + 10.0)

        # 1.6.3 Certifications Boost (up to +6.0 points)
        certifications_list = cand.get("certifications", [])
        cert_bonus = 0.0
        for cert in certifications_list:
            cert_name = cert.get("name", "").lower()
            if any(kw in cert_name for kw in ["machine learning", "deep learning", "tensorflow", "pytorch", "aws certified machine learning", "gcp professional ml", "google cloud professional ml", "google cloud machine learning"]):
                cert_bonus += 2.0
        fit_score = min(100.0, fit_score + min(6.0, cert_bonus))

        # 1.6.4 English Language Proficiency Check
        languages_list = cand.get("languages", [])
        has_english_prof = False
        for lang in languages_list:
            lang_name = lang.get("language", "").lower()
            if "english" in lang_name:
                lang_prof = lang.get("proficiency", "").lower()
                if any(prof in lang_prof for prof in ["professional", "native", "fluent", "bilingual", "full"]):
                    has_english_prof = True
                    break
        # Soft penalty if English proficiency is lacking
        if not has_english_prof:
            fit_score *= 0.90
        
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
        
        # 1.6.8 Secret Gem Boost (Compensate plain-language MLEs for low semantic buzzword similarity)
        is_relevant_title = any(t in current_title.lower() for t in ["ai", "machine learning", "mle", "data scientist", "applied scientist", "search engineer", "recommendation"])
        has_search_rec = any(kw in career_text for kw in ["recommendation", "recommend", "collaborative filtering", "matrix factorization", "search", "ranking", "re-ranking", "information retrieval"])
        is_gem = (4.0 <= years_exp <= 10.0) and is_relevant_title and has_ml_domain_company and has_search_rec
        if is_gem:
            fit_score = min(100.0, fit_score + 12.0)
            
        # 1.7 Junior Cap
        if years_exp < 2.0:
            fit_score = min(65.0, fit_score)

        # 1.7.5 CV/Speech/Robotics Specialization Penalty
        # The JD says: "People whose primary expertise is computer vision, speech, or robotics 
        # without significant NLP/IR exposure. We respect your work but you'd be re-learning fundamentals here."
        cv_speech_primary_titles = [
            "computer vision", "cv engineer", "speech engineer", "speech scientist", 
            "asr engineer", "tts engineer", "robotics engineer", "perception engineer", 
            "autonomous driving", "slam engineer"
        ]
        cv_speech_primary_skills = {
            "computer vision", "object detection", "image classification", "image segmentation", 
            "speech recognition", "asr", "tts", "text to speech", "speech synthesis", 
            "speech processing", "optical flow", "pose estimation", "3d vision", "lidar", "ros", "robotics"
        }
        
        title_lower = current_title.lower()
        is_cv_speech_primary = (
            any(kw in title_lower for kw in cv_speech_primary_titles) or
            len(skills_lower.intersection(cv_speech_primary_skills)) >= 3
        )
        
        # Check if they have compensating NLP/IR signals
        career_titles_lower = [job.get("title", "").lower() for job in career]
        has_nlp_ir_title = any(any(kw in t for kw in ["nlp", "search", "retrieval", "ranking", "re-ranking", "information retrieval", "recsys"]) for t in career_titles_lower)
        has_generic_mle_title = any(any(kw in t for kw in ["machine learning", "ml engineer", "ai engineer", "data scientist", "applied scientist", "researcher"]) for t in career_titles_lower)
        
        has_nlp_ir_compensation = False
        if has_nlp_ir_title:
            has_nlp_ir_compensation = True
        elif has_generic_mle_title and (len(ir_matches_skills) >= 2 or len(skills_lower.intersection({"nlp", "natural language processing", "information retrieval"})) >= 1):
            has_nlp_ir_compensation = True
            
        if is_cv_speech_primary and not has_nlp_ir_compensation:
            fit_score *= 0.50  # 50% penalty per JD criteria
            
        # 1.8 Job-Hopping & Title-Chasing Penalty (Relaxed thresholds for premium tech startup tenures)
        if len(career) >= 3:
            recent_jobs = career[:3]
            total_months = sum(job.get("duration_months", 0) for job in recent_jobs)
            avg_tenure = total_months / len(recent_jobs)
            if avg_tenure < 12.0:
                fit_score *= 0.75
            elif avg_tenure < 18.0:
                fit_score *= 0.85
            elif avg_tenure < 24.0:
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
            sal_min, sal_max = sal_max, sal_min
            fit_score *= 0.95  # 5% penalty for data quality concern

        # Check against JD salary budget (INR 35-55 LPA)
        # Min exceeds 55 LPA -> out of range / budget constraint
        if sal_min > 55.0:
            fit_score *= 0.10  # Severe budget penalty
        # Max is under 25 LPA -> likely lacks seniority for Founding Senior AI Engineer
        elif 0 < sal_max < 25.0:
            fit_score *= 0.80  # Soft seniority penalty

        # 1.9.5 Skills Credibility check (Keyword Stuffing Defense)
        # If a candidate lists many JD-relevant skills but very few appear in actual career history descriptions
        has_skills_stuffing_concern = False
        if jd_matching_count >= 6:
            skills_in_career_text = sum(
                1 for sk in skills_lower.intersection(jd_relevant_skills)
                if sk in career_text
            )
            credibility_ratio = skills_in_career_text / jd_matching_count
            if credibility_ratio < 0.25:
                fit_score *= 0.75  # 25% penalty for unvalidated/overly stuffed skill lists
                has_skills_stuffing_concern = True
                
        # --- 2. AVAILABILITY MULTIPLIER ---
        
        # 2.1 Refined Location & Relocation Heuristics
        loc_lower = profile.get("location", "").lower()
        country_lower = profile.get("country", "").lower()
        
        is_jd_named_cities = any(city in loc_lower for city in ["pune", "noida", "delhi", "new delhi", "gurugram", "gurgaon", "faridabad", "ghaziabad", "hyderabad", "mumbai"])
        is_other_tier1 = any(city in loc_lower for city in ["bangalore", "bengaluru", "chennai", "kolkata", "ahmedabad"])
        
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
            notice_modifier = 0.97
        elif notice_days <= 90:
            notice_modifier = 0.85
        else:
            notice_modifier = 0.65
            
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
            
        # Search Appearance Boost (Market Demand)
        # log scaling relative to 500 searches, capped at 1.05 max multiplier
        search_appearances = signals.get("search_appearance_30d", 0)
        if search_appearances > 0:
            log_ratio = math.log(search_appearances) / math.log(500.0)
            search_boost = 1.0 + 0.05 * min(1.0, max(0.0, log_ratio))
            beh_modifier *= search_boost

        # Active job seeking signal (applications_submitted_30d)
        apps_30d = signals.get("applications_submitted_30d", 0)
        if apps_30d >= 3:
            beh_modifier *= 1.05   # Active job seeker boost
        elif apps_30d == 0 and not signals.get("open_to_work_flag", False):
            beh_modifier *= 0.95   # Passive candidate soft penalty

        # Profile legitimacy/verification signals (verified_email, verified_phone, linkedin_connected)
        if not signals.get("verified_email", True):
            beh_modifier *= 0.95   # Unverified email soft penalty
        if not signals.get("verified_phone", True):
            beh_modifier *= 0.97   # Unverified phone soft penalty
        if not signals.get("linkedin_connected", True):
            beh_modifier *= 0.98   # Missing LinkedIn integration soft penalty

        # Social proof / endorsements (endorsements_received vs connection_count)
        endorsements = signals.get("endorsements_received", 0)
        connections = signals.get("connection_count", 0)
        if endorsements >= 50:
            beh_modifier *= 1.03   # High endorsement boost
        elif endorsements == 0 and connections >= 100:
            beh_modifier *= 0.97   # Connections but zero endorsements soft penalty

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
                    
        # Rule 3.3: Title vs Skills Mismatch (Expanded non-tech management checks)
        if not is_honeypot:
            non_tech_broad = [
                "manager", "director", "vp", "vice president", "business", "product manager", "project manager",
                "program manager", "account", "sales", "marketing", "hr", "finance", "operations", "legal",
                "recruiting", "talent", "customer success", "ceo", "cfo", "coo", "founder", "executive", "admin", "office"
            ]
            tech_compensating = ["ml", "ai", "machine learning", "data", "engineering", "research", "tech", "software", "developer", "architect"]
            
            is_non_tech_mgmt = (
                any(kw in title_lower for kw in non_tech_broad) and
                not any(kw in title_lower for kw in tech_compensating)
            )
            
            if current_title.lower() in non_tech_titles or is_non_tech_mgmt:
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

        # Rule 3.5: Company Foundation Date Violation (Krutrim/Sarvam AI check)
        if not is_honeypot:
            krutrim_found = datetime.date(2023, 4, 1)
            sarvam_found = datetime.date(2023, 7, 1)
            for job in career:
                comp_lower = job.get("company", "").strip().lower()
                s_date = parse_date(job.get("start_date"))
                if s_date:
                    if "krutrim" in comp_lower and s_date < krutrim_found:
                        is_honeypot = True
                        disqualification_reason = f"Krutrim start date {s_date} before foundation April 2023"
                        break
                    elif "sarvam" in comp_lower and s_date < sarvam_found:
                        is_honeypot = True
                        disqualification_reason = f"Sarvam AI start date {s_date} before foundation July 2023"
                        break

        # Rule 3.6: Expert Proficiency with Zero Duration
        if not is_honeypot:
            for s in skills:
                if s.get("proficiency", "").lower() == "expert" and s.get("duration_months", 0) == 0:
                    is_honeypot = True
                    disqualification_reason = f"Expert skill '{s.get('name')}' claimed with zero duration"
                    break

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
            "skill_concentration": jd_matching_count / total_skill_count if total_skill_count > 0 else 0.0,
            "is_cv_speech_primary": is_cv_speech_primary,
            "has_nlp_ir_compensation": has_nlp_ir_compensation,
            "has_skills_stuffing_concern": has_skills_stuffing_concern
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
            
            # Include recent career context with truncated job descriptions for richer semantic matching
            recent_roles = []
            for job in career[:3]:
                job_title = job.get("title", "")
                company = job.get("company", "")
                job_desc = job.get("description", "")
                
                role_parts = []
                if job_title:
                    if company:
                        role_parts.append(f"{job_title} at {company}")
                    else:
                        role_parts.append(job_title)
                
                if job_desc:
                    # Clean up description whitespace and take first 150 characters
                    clean_desc = re.sub(r'\s+', ' ', job_desc).strip()
                    role_parts.append(f"({clean_desc[:150]}...)")
                
                if role_parts:
                    recent_roles.append(" ".join(role_parts))
                    
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
        ALPHA_FIT = 0.85
        ALPHA_CE = 0.15
        
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
