# DEPRECATED: This module (Cross-Encoder re-ranking) has been replaced by
# the SwarmEvaluator in matrix_pipeline.py. Not imported anywhere in the active pipeline.
# Kept for reference only.
import time
import math
from sentence_transformers import CrossEncoder

class CandidateReranker:
    def __init__(self):
        print("Initializing Candidate Reranker...")
        # Load the STS Cross Encoder model
        self.model = CrossEncoder("cross-encoder/stsb-roberta-base")
        print("Candidate Reranker ready.")

    def sigmoid(self, x):
        """STSb outputs similarity scores in [0, 1] directly. We clip to enforce bounds."""
        return max(0.0, min(1.0, float(x)))

    def rerank(self, job_description, expanded_query, candidates):
        print(f"Executing Step 3: Cross-Encoder Re-ranking on {len(candidates)} candidates...")
        if not candidates:
            return []
            
        start_time = time.time()
        
        # Extract title and required skills for a concise, high-signal query context
        title = ""
        skills = ""
        for line in job_description.strip().split("\n"):
            if line.lower().startswith("job title:"):
                title = line.split(":", 1)[1].strip()
            elif line.lower().startswith("required skills:"):
                skills = line.split(":", 1)[1].strip()
        
        if title or skills:
            query_context = f"Job Title: {title}. Required Skills: {skills}."
        else:
            # Fallback to the first 4 lines of the JD
            query_context = "\n".join(job_description.strip().split("\n")[:4])
        
        # Part A: Required skill hit rate (hard domain filter)
        REQUIRED_SKILLS = ["python", "pytorch", "transformers", "llm", "nlp", 
                           "langchain", "mlflow", "triton", "cuda"]
        DEEP_SIGNALS = ["lora", "fine-tun", "rlhf", "quantiz", "distill", 
                        "distributed training", "inference optimization", 
                        "tensorrt", "embedding", "drift detection"]

        to_predict_pairs = []
        to_predict_indices = []

        for idx, cand in enumerate(candidates):
            resume_text = cand["resume_text"]
            resume_lower = resume_text.lower()
            
            req_hits = sum(1 for s in REQUIRED_SKILLS if s in resume_lower)
            deep_hits = sum(1 for s in DEEP_SIGNALS if s in resume_lower)
            
            skill_score = req_hits / len(REQUIRED_SKILLS)
            depth_score = min(1.0, deep_hits / 5.0)
            
            # Stash scores for calculation
            cand["skill_score"] = skill_score
            cand["depth_score"] = depth_score
            
            # Part B: Cross-encoder only runs if skill_score >= 0.3 (domain gate)
            if skill_score < 0.3:
                semantic_final = skill_score * 0.5   # hard cap for wrong domain
                cand["semantic_depth_score"] = float(semantic_final)
                cand["raw_cross_encoder_logit"] = -99.0
            else:
                to_predict_pairs.append([query_context, resume_text])
                to_predict_indices.append(idx)

        # Execute Cross-Encoder only for gated candidates
        if to_predict_pairs:
            raw_scores = self.model.predict(to_predict_pairs)
            for idx_in_batch, raw_score in enumerate(raw_scores):
                cand_idx = to_predict_indices[idx_in_batch]
                cand = candidates[cand_idx]
                
                ce_score = float(1 / (1 + math.exp(-raw_score)))
                skill_score = cand["skill_score"]
                depth_score = cand["depth_score"]
                semantic_final = (skill_score * 0.4) + (depth_score * 0.3) + (ce_score * 0.3)
                
                cand["semantic_depth_score"] = float(semantic_final)
                cand["raw_cross_encoder_logit"] = float(raw_score)

        duration_ms = (time.time() - start_time) * 1000
        print(f"Hybrid Reranker completed in {duration_ms:.1f}ms. {len(to_predict_pairs)}/{len(candidates)} candidates passed the 0.3 skill gate to run through Cross-Encoder.")
        
        return candidates

if __name__ == "__main__":
    # Quick mock verification
    reranker = CandidateReranker()
    mock_candidates = [
        {"resume_text": "Highly proficient frontend engineer implementing module federation, LCP optimizations, and typescript generics.", "candidate_id": "c1"},
        {"resume_text": "Staff DevOps specialist building CI/CD pipelines, gitops deployment with ArgoCD, and Helm charts on AWS.", "candidate_id": "c2"}
    ]
    results = reranker.rerank("Frontend engineer with React, TS, micro-frontends", "module federation LCP typescript", mock_candidates)
    for c in results:
        print(f"ID: {c['candidate_id']}, Semantic Depth: {c['semantic_depth_score']:.4f} (logit: {c['raw_cross_encoder_logit']:.4f})")
