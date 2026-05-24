import time
import json
from pathlib import Path

from .query_expand import expand_query
from .matrix_pipeline import SwarmMatrixRanker
from .scorer import CandidateScorer

class TalentRadarPipeline:
    def __init__(self):
        print("=== Initializing Talent Radar End-to-End Pipeline ===")
        start_time = time.time()
        
        # Load candidates.json pool
        candidates_path = Path(__file__).parent / "candidates.json"
        print(f"Loading candidate records from {candidates_path.resolve()}...")
        try:
            with open(candidates_path, "r", encoding="utf-8") as f:
                self.candidates_pool = json.load(f)
        except Exception as e:
            print(f"Error loading candidates.json: {e}")
            self.candidates_pool = []
            
        print(f"Loaded {len(self.candidates_pool)} candidates.")
        
        duration = time.time() - start_time
        print(f"=== Pipeline initialized successfully in {duration:.2f}s ===")

    def run(self, job_description, seniority_level="Senior", top_k=50, sector="TECH", semantic_weight=0.60, velocity_weight=0.25, freshness_weight=0.15):
        print(f"\n==========================================")
        print(f"TALENT RADAR PIPELINE RUN: Seniority={seniority_level} | Sector={sector}")
        print(f"==========================================")
        
        timings = {
            "query_explosion_ms": 0.0,
            "vector_retrieval_ms": 0.0,
            "cross_encoder_rerank_ms": 0.0,
            "scorer_scoring_ms": 0.0,
            "candidate_load_ms": 0.0,
            "swarm_evaluation_ms": 0.0,
            "overall_ms": 0.0
        }
        overall_start = time.time()
        
        # Upstream Refinement (The Gemini Gateway)
        print("[Swarm Matrix] Initiating Gemini Gateway upstream refinement...")
        from .matrix_pipeline import GeminiGateway
        gateway = GeminiGateway()
        refined_matrix = gateway.route_and_polish(job_description, sector)
        self.resolved_sector = refined_matrix.sector_token
        print(f"[Swarm Matrix] Polished requirements: '{refined_matrix.polished_requirements}'")
        print(f"[Swarm Matrix] Target routing sector token: '{refined_matrix.sector_token}'")
        print(f"[Swarm Matrix] Gemini Extracted Keywords: {refined_matrix.top_keywords}")
        
        # STEP 1: Query Explosion
        step1_start = time.time()
        expanded_query = expand_query(job_description, seniority_level)
        timings["query_explosion_ms"] = (time.time() - step1_start) * 1000
        print(f"Step 1 Complete: Query Expanded. [{timings['query_explosion_ms']:.1f}ms]")
        
        # STEP 2: Candidate Load & Smart Pre-filtering
        step2_start = time.time()
        
        if not self.candidates_pool:
            timings["candidate_load_ms"] = (time.time() - step2_start) * 1000
            timings["vector_retrieval_ms"] = timings["candidate_load_ms"]
            timings["overall_ms"] = (time.time() - overall_start) * 1000
            return [], expanded_query, timings

        # Tokenize job description and expanded query to get unique keywords
        stop_words = {
            "the", "and", "for", "with", "a", "of", "to", "in", "is", "at", "by", "from", "on", "an", "this",
            "that", "these", "those", "are", "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "but", "or", "as", "if", "then", "else", "when", "where", "why", "how", "all", "any", "both", "each",
            "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
            "than", "too", "very", "s", "t", "can", "will", "just", "should", "now", "using", "experience",
            "work", "working", "knowledge", "skills", "ability", "years", "role", "team", "development", "engineer",
            "developer", "management", "systems", "solutions", "services", "infrastructure"
        }
        
        import re
        query_text = (job_description + " " + expanded_query).lower()
        query_words = set(re.findall(r'\b[a-z]{3,}\b', query_text)) - stop_words
        
        # Lowercase, split multi-word phrases, and clean Gemini extracted keywords
        gemini_keywords = set()
        for kw in refined_matrix.top_keywords:
            for word in re.findall(r'\b[a-z]{3,}\b', kw.lower()):
                gemini_keywords.add(word)
        gemini_keywords = gemini_keywords - stop_words
        
        # Score candidates by hybrid keyword matching
        pre_filtered_scored = []
        for cand in self.candidates_pool:
            cand_id = cand.get("candidate_id", "Unknown")
            resume_text = cand.get("resume_text", "").lower()
            
            # Extract job titles
            titles_text = " ".join([history.get("title", "") for history in cand.get("career_history", [])]).lower()
            
            resume_words = set(re.findall(r'\b[a-z]{3,}\b', resume_text))
            title_words = set(re.findall(r'\b[a-z]{3,}\b', titles_text))
            
            # 1. Base query overlap score: 1 point per unique matching word
            overlap_score = len(query_words.intersection(resume_words))
            
            # 2. Historical job title overlap bonus: 3 points per matching term
            title_bonus = len(query_words.intersection(title_words)) * 3.0
            
            # 3. Gemini top 30 keyword match bonus: 5 points per matching term
            gemini_bonus = len(gemini_keywords.intersection(resume_words)) * 5.0
            
            total_keyword_score = overlap_score + title_bonus + gemini_bonus
            
            pre_filtered_scored.append((total_keyword_score, cand))
            
        # Sort candidates by keyword score descending
        pre_filtered_scored.sort(key=lambda x: x[0], reverse=True)
        
        # Take at least 50 candidates (or the whole pool if smaller than 50) and up to 20% of the pool dynamically to ensure no hidden gems are missed
        rank_limit = min(len(self.candidates_pool), max(50, int(len(self.candidates_pool) * 0.20)))
        candidates_to_rank = [item[1] for item in pre_filtered_scored[:rank_limit]]
        
        timings["candidate_load_ms"] = (time.time() - step2_start) * 1000
        timings["vector_retrieval_ms"] = timings["candidate_load_ms"]
        print(f"Step 2 Complete: Loaded {len(self.candidates_pool)} candidates, smart pre-filtered to top {len(candidates_to_rank)} candidates. [{timings['candidate_load_ms']:.1f}ms]")
        
        # STEP 3: Swarm Evaluation (on pre-filtered subset)
        step3_start = time.time()
        ranker = SwarmMatrixRanker(raw_query=expanded_query, sector=refined_matrix.sector_token, refined_matrix=refined_matrix)
        results_df = ranker.rank_candidates(candidates_to_rank)
        timings["swarm_evaluation_ms"] = (time.time() - step3_start) * 1000
        timings["cross_encoder_rerank_ms"] = timings["swarm_evaluation_ms"]
        print(f"Step 3 Complete: Swarm evaluation finished. [{timings['swarm_evaluation_ms']:.1f}ms]")
        
        # STEP 4: Convert/Map and run Scorer with guardrails
        step4_start = time.time()
        
        # Build score lookup from DataFrame
        score_lookup = {}
        for _, row in results_df.iterrows():
            cand_id = row["Candidate ID"]
            score_lookup[cand_id] = row["Best Chunk Alignment Score"]
            
        # Re-attach scores to copies of original full candidate dicts to avoid cross-request contamination
        scored_pool = []
        for cand in self.candidates_pool:
            cand_copy = cand.copy()
            cand_copy["semantic_depth_score"] = score_lookup.get(cand["candidate_id"], 0.0)
            scored_pool.append(cand_copy)
            
        # Sort by score descending before passing to scorer
        sorted_candidates = sorted(
            scored_pool,
            key=lambda c: c["semantic_depth_score"],
            reverse=True
        )
        
        # Run CandidateScorer with custom weights
        scorer = CandidateScorer(
            seniority_level=seniority_level, 
            target_keywords=refined_matrix.top_keywords, 
            sector=refined_matrix.sector_token,
            semantic_weight=semantic_weight,
            velocity_weight=velocity_weight,
            freshness_weight=freshness_weight
        )
        final_ranked_candidates = scorer.score_candidates(sorted_candidates)
        timings["scorer_scoring_ms"] = (time.time() - step4_start) * 1000
        print(f"Step 4 Complete: Scored and ranked with guardrails. [{timings['scorer_scoring_ms']:.1f}ms]")
        
        timings["overall_ms"] = (time.time() - overall_start) * 1000
        print(f"Pipeline finished. Total: {timings['overall_ms']:.1f}ms")
        print(f"==========================================\n")
        
        return final_ranked_candidates, expanded_query, timings
