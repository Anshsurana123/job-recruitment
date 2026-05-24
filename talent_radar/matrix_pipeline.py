import os
import sys
import json
import time
from pathlib import Path
import pandas as pd
from pydantic import BaseModel, Field
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load env variables from the workspace root folder
base_dir = Path(__file__).parent
env_path = base_dir.parent / ".env"
if env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=env_path)

# ===================================================================== #
#  Phase 1: Input Refinement & Sector Routing (The Gemini Gateway)      #
# ===================================================================== #

class RequirementsMatrix(BaseModel):
    polished_requirements: str = Field(
        description="A dense, refined string of core domain concepts, operational/functional constraints, and role-specific requirements extracted from the job description."
    )
    sector_token: str = Field(
        description="A verified industry token mapping the search to one of the 12 macros: TECH, FIN, HEALTH, LEGAL, REAL, MANU, COMM, LOGI, MEDIA, ENERGY, EDU, GOV."
    )
    top_keywords: list[str] = Field(
        description="A list of the top 30 key domain terms, methodologies, skills, tools, credentials, and sector-specific keywords relevant to the job description."
    )

class GeminiGateway:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set in the environment or .env file.")
        
        # Instantiate official Google GenAI SDK client
        from google import genai
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = "gemini-2.5-flash"

    def route_and_polish(self, raw_query: str, raw_sector: str) -> RequirementsMatrix:
        """Uses gemini-2.5-flash with structured output to refine search query and identify industry sector."""
        from google.genai import types
        
        prompt = f"""
        Analyze the following recruiter search query and raw industry sector suggestion.
        1. Extract the core domain requirements, functional constraints, and operational dependencies specific to the target industry sector.
        2. Clean and polish them into a dense conceptual requirements list.
        3. Match the sector to one of these 12 strict sector tokens: TECH, FIN, HEALTH, LEGAL, REAL, MANU, COMM, LOGI, MEDIA, ENERGY, EDU, GOV.
        4. Extract the top 30 key domain terms, methodologies, professional skills, tools, credentials, and sector-specific keywords relevant to this job description.
        
        Query: "{raw_query}"
        Suggested Sector: "{raw_sector}"
        """
        
        system_instr = (
            "You are an expert recruiter and domain specialist specializing in search "
            "refinement and matching across multiple industry sectors."
        )
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instr,
                    response_mime_type="application/json",
                    response_schema=RequirementsMatrix,
                    temperature=0.1
                )
            )
            
            # Load parsed response JSON
            data = json.loads(response.text)
            return RequirementsMatrix(**data)
            
        except Exception as e:
            print(f"[Gateway Warning] Structured generation error: {e}. Utilizing local fallback mapping...")
            # Secure local fallback implementation
            valid_sectors = {"TECH", "FIN", "HEALTH", "LEGAL", "REAL", "MANU", "COMM", "LOGI", "MEDIA", "ENERGY", "EDU", "GOV"}
            token = raw_sector.upper().strip()
            if token not in valid_sectors:
                token = "TECH"
            
            # Simple fallback parser for local execution
            import re
            words = set(re.findall(r'\b[a-z]{3,}\b', raw_query.lower()))
            fallback_keywords = list(words)[:30]
            
            return RequirementsMatrix(
                polished_requirements=raw_query.strip(),
                sector_token=token,
                top_keywords=fallback_keywords
            )

# ===================================================================== #
#  Phase 2: Structural Fragmenting (The Chunking Engine)               #
# ===================================================================== #

class ResumeTextChunker:
    @staticmethod
    def chunk_text(text: str, max_words: int = 300) -> list[str]:
        """Splits candidate resume text into sentence-boundary aligned paragraphs under 300 words."""
        if not text or not text.strip():
            return []
            
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for para in paragraphs:
            para_words = para.split()
            if not para_words:
                continue
                
            if current_word_count + len(para_words) <= max_words:
                current_chunk.append(para)
                current_word_count += len(para_words)
            else:
                # Flush existing chunk
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                
                # If a single paragraph is larger than max_words, break it by sentences
                if len(para_words) > max_words:
                    sentences = para.replace('! ', '. ').replace('? ', '. ').split('. ')
                    temp_chunk = []
                    temp_count = 0
                    for sent in sentences:
                        sent = sent.strip()
                        if not sent:
                            continue
                        sent_words = sent.split()
                        if temp_count + len(sent_words) <= max_words:
                            temp_chunk.append(sent)
                            temp_count += len(sent_words)
                        else:
                            if temp_chunk:
                                chunks.append(". ".join(temp_chunk) + ".")
                            temp_chunk = [sent]
                            temp_count = len(sent_words)
                    if temp_chunk:
                        current_chunk = temp_chunk
                        current_word_count = temp_count
                    else:
                        current_chunk = []
                        current_word_count = 0
                else:
                    current_chunk = [para]
                    current_word_count = len(para_words)
                    
        if current_chunk:
            chunks.append("\n".join(current_chunk))
            
        return chunks

# ===================================================================== #
#  Phase 3 & 4: Model Matrix Loader & Swarm Evaluation Engine           #
# ===================================================================== #

# Sector macro model weight mapping
SECTOR_MODEL_MAP = {
    "TECH": "microsoft/codebert-base",
    "FIN": "ProsusAI/finbert",
    "HEALTH": "dmis-lab/biobert-base-cased-v1.2",
    "LEGAL": "nlpaueb/legal-bert-small-uncased",
    "REAL": "llmware/industry-bert-asset-management-v0.1",
    "MANU": "cea-list-ia/ManufactuBERT",
    "COMM": "nlptown/bert-base-multilingual-uncased-sentiment",
    "LOGI": "inovex/multi2convai-logistics-en-bert",
    "MEDIA": "cardiffnlp/twitter-roberta-base",
    "ENERGY": "Master-AI-Lab/EnergyBERT",
    "EDU": "vasugoel/K-12BERT",
    "GOV": "ESGBERT/GovRoBERTa-governance",
    # General fallback cross-encoder
    "DEFAULT": "cross-encoder/ms-marco-MiniLM-L-6-v2"
}

import threading

# Thread lock for safe model caching
cache_lock = threading.Lock()

# Module-level singleton cache: sector_token -> loaded SwarmEvaluator instance
_EVALUATOR_CACHE: dict[str, "SwarmEvaluator"] = {}

def get_cached_evaluator(sector_token: str) -> "SwarmEvaluator":
    """Returns a loaded SwarmEvaluator for the given sector, cached across requests."""
    key = sector_token.upper()
    with cache_lock:
        if key not in _EVALUATOR_CACHE:
            evaluator = SwarmEvaluator(key)
            evaluator.load_model()
            _EVALUATOR_CACHE[key] = evaluator
            print(f"[Swarm Cache] Cached evaluator for sector: {key}")
        else:
            print(f"[Swarm Cache] HIT — reusing loaded model for sector: {key}")
        return _EVALUATOR_CACHE[key]

class SwarmEvaluator:
    def __init__(self, sector_token: str):
        self.sector_token = sector_token.upper()
        self.model_name = SECTOR_MODEL_MAP.get(self.sector_token, SECTOR_MODEL_MAP["DEFAULT"])
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        """Lazy load HuggingFace models keeping runtime footprint minimal."""
        if self.model is not None:
            return
            
        print(f"[Swarm Matrix] Spawning dynamic SLM Swarm Node for sector: {self.sector_token} on {self.device.upper()}")
        print(f"[Swarm Matrix] Loading tokenizer and model: {self.model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Ensure pad token is set for batched processing
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # Direct probability distribution / sequence regression setup
        if self.model_name == SECTOR_MODEL_MAP["DEFAULT"]:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, 
                num_labels=1, 
                ignore_mismatched_sizes=True
            )
        else:
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(self.model_name)

        self.model.to(self.device)
        self.model.eval()
        
        # Compile model only when running on GPU (CUDA) to avoid long hangs on CPU-only/Windows environments
        if self.device == "cuda":
            print("[Swarm Matrix] GPU detected. Attempting torch.compile optimization...")
            try:
                self.model = torch.compile(self.model)
                print("[Swarm Matrix] torch.compile optimization succeeded.")
            except Exception as e:
                print(f"[Swarm Matrix] Skipping torch.compile: {e}")
        else:
            print("[Swarm Matrix] CPU detected. Skipping torch.compile to avoid startup hangs.")
            
        print(f"[Swarm Matrix] Swarm node successfully loaded and active.")

    def evaluate_fragments_batch(self, polished_requirements: str, fragments: list[str], batch_size: int = 32) -> list[float]:
        """Evaluates candidate resume fragments in optimized GPU/CPU batches."""
        if not fragments:
            return [0.0]
            
        self.load_model()
        scores = []
        
        if self.model_name != SECTOR_MODEL_MAP["DEFAULT"]:
            # Contextual Embedding Cosine Similarity path for base models
            def mean_pooling(model_output, attention_mask):
                token_embeddings = model_output[0]
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                return sum_embeddings / sum_mask

            # Encode query once
            query_inputs = self.tokenizer(
                [polished_requirements],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            query_inputs = {k: v.to(self.device) for k, v in query_inputs.items()}
            
            with torch.inference_mode():
                query_outputs = self.model(**query_inputs)
                query_emb = mean_pooling(query_outputs, query_inputs["attention_mask"]) # Shape: [1, hidden_dim]

            # Process resume fragments in batches
            all_frag_embs = []
            for i in range(0, len(fragments), batch_size):
                batch_frags = fragments[i:i+batch_size]
                
                inputs = self.tokenizer(
                    batch_frags,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.inference_mode():
                    outputs = self.model(**inputs)
                    frag_embs = mean_pooling(outputs, inputs["attention_mask"]) # Shape: [batch_size, hidden_dim]
                    all_frag_embs.append(frag_embs)
            
            # Stack all fragments to a single tensor
            all_frag_embs_tensor = torch.cat(all_frag_embs, dim=0) # Shape: [len(fragments), hidden_dim]
            
            # Subtraction centering to mitigate pre-trained BERT anisotropy (only if there is more than 1 fragment)
            if len(fragments) > 1:
                mu = torch.mean(all_frag_embs_tensor, dim=0, keepdim=True) # Shape: [1, hidden_dim]
                query_emb_centered = query_emb - mu
                frag_embs_centered = all_frag_embs_tensor - mu
                
                # Normalize centered embeddings
                q_emb_norm = torch.nn.functional.normalize(query_emb_centered, p=2, dim=1)
                frag_embs_norm = torch.nn.functional.normalize(frag_embs_centered, p=2, dim=1)
            else:
                # Fallback to raw embeddings if only 1 fragment is evaluated
                q_emb_norm = torch.nn.functional.normalize(query_emb, p=2, dim=1)
                frag_embs_norm = torch.nn.functional.normalize(all_frag_embs_tensor, p=2, dim=1)
                
            with torch.inference_mode():
                # Compute cosine similarities in one single matrix multiplication pass
                similarities = torch.mm(q_emb_norm, frag_embs_norm.transpose(0, 1)).squeeze(0)
                
                if len(fragments) == 1:
                    similarities = similarities.unsqueeze(0)
                    
                for sim in similarities:
                    val = sim.item()
                    # Cosine similarity is in [-1.0, 1.0]. Map to clean probability range [0.0, 1.0]
                    score = (val + 1.0) / 2.0
                    scores.append(score)
        else:
            # Traditional Cross-Encoder sequence classification path
            sequences = [f"{polished_requirements} [SEP] {frag}" for frag in fragments]
            
            for i in range(0, len(sequences), batch_size):
                batch_seqs = sequences[i:i+batch_size]
                
                inputs = self.tokenizer(
                    batch_seqs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.inference_mode():
                    outputs = self.model(**inputs)
                    logits = outputs.logits.squeeze(-1)
                    
                    if len(batch_seqs) == 1:
                        logits = logits.unsqueeze(0)
                        
                    for logit in logits:
                        val = logit.item()
                        # Sigmoid activation to yield clean float score in [0.0, 1.0]
                        score = 1.0 / (1.0 + float(torch.exp(torch.tensor(-val)).item()))
                        scores.append(score)
                        
        return scores

# ===================================================================== #
#  Phase 5: Scoring Integration & Unified Multi-Agent Engine           #
# ===================================================================== #

class SwarmMatrixRanker:
    def __init__(self, raw_query: str, sector: str, refined_matrix: RequirementsMatrix = None):
        self.raw_query = raw_query
        self.sector = sector
        self.chunker = ResumeTextChunker()
        self.refined_matrix = refined_matrix
        self._gateway = None  # Lazy — only created if refined_matrix is not provided

    @property
    def gateway(self):
        if self._gateway is None:
            self._gateway = GeminiGateway()
        return self._gateway

    def rank_candidates(self, candidates: list[dict]) -> pd.DataFrame:
        """E2E orchestrator: mega-batch inference across all candidates in one forward pass."""
        t0 = time.time()

        if self.refined_matrix is None:
            print(f"[Swarm Matrix] Initiating Gemini Gateway upstream refinement...")
            reqs = self.gateway.route_and_polish(self.raw_query, self.sector)
        else:
            reqs = self.refined_matrix

        print(f"[Swarm Matrix] Polished requirements: '{reqs.polished_requirements}'")
        print(f"[Swarm Matrix] Target routing sector token: '{reqs.sector_token}'")

        evaluator = get_cached_evaluator(reqs.sector_token)

        # PHASE 1: Chunk all resumes and build one flat fragment list
        # Track which fragments belong to which candidate via index ranges
        all_fragments = []
        candidate_index_ranges = []  # (start_idx, end_idx) into all_fragments per candidate

        for cand in candidates:
            resume_text = cand.get("resume_text", "")
            fragments = self.chunker.chunk_text(resume_text, max_words=300)
            if not fragments:
                fragments = [""]  # Ensure at least one entry so index math stays consistent
            start = len(all_fragments)
            all_fragments.extend(fragments)
            end = len(all_fragments)
            candidate_index_ranges.append((start, end))

        print(f"[Swarm Matrix] Mega-batch: {len(candidates)} candidates -> {len(all_fragments)} total fragments")

        # PHASE 2: ONE single batched forward pass over all fragments — no threading, no async
        t1 = time.time()
        all_scores = evaluator.evaluate_fragments_batch(
            reqs.polished_requirements,
            all_fragments,
            batch_size=32
        )
        t2 = time.time()
        print(f"[Swarm Matrix] Mega-batch inference complete: {len(all_fragments)} fragments in {(t2-t1)*1000:.0f}ms")

        # PHASE 3: Redistribute scores back to candidates and aggregate via MAX()
        scored_candidates = []
        for i, cand in enumerate(candidates):
            start, end = candidate_index_ranges[i]
            fragment_scores = all_scores[start:end]
            max_fragment_score = max(fragment_scores) if fragment_scores else 0.0

            role_transitions = len(cand.get("career_history", []))
            years_experience = float(cand.get("years_experience", 1.0))
            velocity = float(role_transitions) / (years_experience + 1.0)
            final_score = (0.75 * max_fragment_score) + (0.25 * min(1.0, velocity))

            scored_candidates.append({
                "Candidate ID": cand.get("candidate_id", "Unknown"),
                "Candidate Name": cand.get("name", "Unknown"),
                "Macro Sector Code": reqs.sector_token,
                "Best Chunk Alignment Score": max_fragment_score,
                "Career Velocity": velocity,
                "Final Unified Score": final_score * 100.0
            })

        print(f"[Swarm Matrix] Total rank_candidates(): {(time.time()-t0)*1000:.0f}ms")

        df = pd.DataFrame(scored_candidates)
        df = df.sort_values(by="Final Unified Score", ascending=False).reset_index(drop=True)
        df.insert(0, "Rank", df.index + 1)
        return df

# ===================================================================== #
#  Verification Script & Main Harness                                  #
# ===================================================================== #

if __name__ == "__main__":
    # Robust mock profile setups representing semantic edge cases
    candidates_pool = [
        {
            "candidate_id": "cand_001",
            "name": "Profile A (The Trap - IT Infrastructure)",
            "years_experience": 8.0,
            "career_history": [
                {"title": "Senior Systems Engineer", "company": "Global IT Solutions", "start_date": "2018-01-01", "end_date": "Present"}
            ],
            "resume_text": (
                "IT Infrastructure and Systems Engineer with 8 years of hardware racking and switch configuration. "
                "Familiar with setting up AI systems, ML architectures, and enterprise deployment data networks. "
                "Responsible for physical hardware deployments, routing cables, configuring router ports, "
                "handling client support tickets, network switches, cabling systems, server virtualization setups, "
                "Windows support systems, desktop provisioning, security groups, Active Directory groups, and router gateways."
            )
        },
        {
            "candidate_id": "cand_002",
            "name": "Profile B (The Hidden Gem - ML Compiler Dev)",
            "years_experience": 3.0,
            "career_history": [
                {"title": "Machine Learning Engineer", "company": "DeepTech Labs", "start_date": "2023-01-01", "end_date": "Present"},
                {"title": "Compiler Intern", "company": "Silicon Systems", "start_date": "2022-01-01", "end_date": "2023-01-01"}
            ],
            "resume_text": (
                "Software systems developer focusing on deep compiler structures. "
                "Designed and optimized flash attention matrix loops inside GPU kernels. "
                "Quantized base model parameters to 4-bit configurations, tracking hardware-level latency drift. "
                "Configured sparse recovery pipelines, Triton attention kernels, and low-level memory block caching "
                "to accelerate neural net inference engines under constrained budgets."
            )
        },
        {
            "candidate_id": "cand_003",
            "name": "Profile C (General Backend Developer)",
            "years_experience": 5.0,
            "career_history": [
                {"title": "Software Engineer II", "company": "E-Commerce Corp", "start_date": "2021-01-01", "end_date": "Present"}
            ],
            "resume_text": (
                "Full stack backend developer writing Python and Node.js APIs. "
                "Maintained Django applications, PostgreSQL databases, redis caches, and REST endpoints. "
                "Integrated payment webhooks, custom logging filters, unit tests, and CI/CD pipelines."
            )
        }
    ]
    
    # Recruiter Input Setup: Target query & Suggested Sector
    raw_query = "AI/ML Engineer to build and optimize local vector chunking, quantization parameters, memory footprint tracking, and flash attention mechanism optimization."
    suggested_sector = "TECH"
    
    print("=" * 80)
    print(" SWARMMATRIX AI: MULTI-AGENT SLM SWARM PIPELINE VERIFICATION ")
    print("=" * 80)
    
    # Executing synchronous SwarmMatrix ranking workflow
    try:
        ranker = SwarmMatrixRanker(raw_query, suggested_sector)
        results = ranker.rank_candidates(candidates_pool)

        print("\n" + "=" * 80)
        print("                        SWARMMATRIX RANKING RESULTS                      ")
        print("=" * 80)
        print(results.to_string(index=False))
        print("=" * 80)

        # Verify Profile B outranks Profile A (Correct contextual mapping over superficial keywords)
        assert results.iloc[0]["Candidate ID"] == "cand_002", (
            f"Assertion Failed! The Trap (Profile A) erroneously outranked the Hidden Gem (Profile B). "
            f"Rank 1 was: {results.iloc[0]['Candidate Name']}"
        )
        print("\n[Assertion Check] Success! Profile B (The Hidden Gem) successfully outranked Profile A (The Trap).")
        print("[Assertion Check] SwarmMatrix successfully bypassed concept dilution & aesthetic bias!")
    except Exception as err:
        print(f"\nVerification execution failed: {err}")
        sys.exit(1)
