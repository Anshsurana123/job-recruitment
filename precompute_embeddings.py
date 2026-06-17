import json
import pickle
import re
import sys
import time
from pathlib import Path
import numpy as np

# Set random seeds for determinism
np.random.seed(42)

def build_candidate_fields(cand):
    profile = cand.get("profile", {})
    career = cand.get("career_history", [])
    education = cand.get("education", [])
    skills = cand.get("skills", [])
    
    fields = {}
    parts_title = []
    if profile.get("current_title"): 
        parts_title.append(profile["current_title"])
    if profile.get("headline"): 
        parts_title.append(profile["headline"])
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
    
    return fields

def extract_candidate_text(cand):
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
            clean_desc = re.sub(r'\s+', ' ', job_desc).strip()
            role_parts.append(f"({clean_desc[:150]}...)")
            
        if role_parts:
            recent_roles.append(" ".join(role_parts))
            
    career_str = ". ".join(recent_roles)
    return f"Current Title: {fields['title_headline']}. Skills: {fields['skills']}. Summary: {fields['summary']}. Recent: {career_str}."

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Precompute BGE Embeddings for Candidates")
    parser.add_argument("--candidates", type=str, default="./candidates.jsonl", help="Path to candidates jsonl or json file")
    args = parser.parse_args()
    
    candidates_path = Path(args.candidates)
    if not candidates_path.exists():
        print(f"Error: candidates path '{candidates_path}' does not exist.")
        sys.exit(1)
        
    print(f"Loading candidates from {candidates_path}...")
    candidates = []
    
    with open(candidates_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if content.startswith("[") and content.endswith("]"):
            try:
                candidates = json.loads(content)
            except Exception as e:
                print(f"Error loading as JSON array: {e}")
        
        if not candidates:
            # Load line-by-line
            f.seek(0)
            for line in f:
                if line.strip():
                    try:
                        candidates.append(json.loads(line))
                    except:
                        pass
                        
    num_candidates = len(candidates)
    print(f"Loaded {num_candidates} candidates.")
    
    model_cache_path = Path("./model_cache/bge-small-en-v1.5")
    if not model_cache_path.exists():
        print(f"Downloading model BAAI/bge-small-en-v1.5 and saving to {model_cache_path}...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        model.save(str(model_cache_path))
    else:
        from sentence_transformers import SentenceTransformer
        print(f"Loading model from {model_cache_path}...")
        model = SentenceTransformer(str(model_cache_path))
        
    print("Extracting candidate texts...")
    texts = [extract_candidate_text(cand) for cand in candidates]
    
    # Define output cache file name based on whether it is a sample or full candidates list
    is_sample = "sample" in candidates_path.name.lower() or num_candidates < 1000
    cache_name = "embeddings_sample.pkl" if is_sample else "embeddings_full.pkl"
    cache_path = Path(cache_name)
    
    print(f"Encoding {num_candidates} candidates on CPU. This will take some time...")
    t0 = time.time()
    
    # We will encode in batches of 128
    embeddings = model.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    elapsed = time.time() - t0
    print(f"Encoding finished in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes).")
    
    # Create the mapping mapping candidate_id -> embedding vector
    print("Building ID to embedding mapping dictionary...")
    id_to_emb = {}
    for idx, cand in enumerate(candidates):
        cid = cand["candidate_id"]
        id_to_emb[cid] = embeddings[idx].astype(np.float32) # save space by using float32
        
    print(f"Saving embeddings cache to {cache_path}...")
    with open(cache_path, "wb") as f:
        pickle.dump(id_to_emb, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print(f"Embeddings cache successfully saved to {cache_path}. Size: {cache_path.stat().st_size / (1024*1024):.1f} MB.")

if __name__ == "__main__":
    main()
