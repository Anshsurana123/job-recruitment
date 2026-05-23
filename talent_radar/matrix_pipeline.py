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

