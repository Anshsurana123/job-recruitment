import os
import sys
import io
import json
import threading
from pathlib import Path

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

# Add workspace folder to path to enable importing
base_dir = Path(__file__).parent
sys.path.insert(0, str(base_dir.parent))

from talent_radar.pipeline import TalentRadarPipeline
from talent_radar.smart_ingest import SmartIngestJob, run_smart_ingest

# Thread locks for safe concurrency
pipeline_lock = threading.Lock()
jobs_lock = threading.Lock()

# Global pipeline instance (lazy loaded)
_pipeline = None

# Active background ingest jobs (job_id -> SmartIngestJob)
_active_jobs: dict[str, SmartIngestJob] = {}

def get_pipeline():
    global _pipeline
    with pipeline_lock:
        if _pipeline is None:
            _pipeline = TalentRadarPipeline()
        return _pipeline

app = FastAPI(title="Talent Radar API", version="1.0.0")

@app.on_event("startup")
def startup_event():
    print("Pre-loading pipeline and candidate pool...")
    get_pipeline()

    # Warm the TECH sector model — most common sector, loads it into cache now
    print("[Startup] Warming SLM model cache for TECH sector...")
    try:
        from talent_radar.matrix_pipeline import get_cached_evaluator
        evaluator = get_cached_evaluator("TECH")
        # Run one dummy forward pass to trigger JIT and allocate memory
        dummy_score = evaluator.evaluate_fragments_batch(
            "software engineer python pytorch",
            ["Experienced ML engineer with PyTorch and transformer fine-tuning."],
            batch_size=1
        )
        print(f"[Startup] TECH model warm-up complete. Dummy score: {dummy_score[0]:.4f}")
    except Exception as e:
        print(f"[Startup] Model warm-up failed (non-fatal): {e}")

    print("Pipeline fully ready. First request will be fast.")

# Enable CORS for frontend flexibility
# SECURITY WARNING: In production environments, allow_origins should be locked down to the actual deployment domain instead of '*' to prevent cross-origin exploits.
allowed_origins_str = os.getenv("ALLOWED_ORIGINS")
if allowed_origins_str:
    origins = [o.strip() for o in allowed_origins_str.split(",") if o.strip()]
else:
    # Allow common local development origins by default
    origins = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost:5173",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:5173",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RankRequest(BaseModel):
    job_description: str
    seniority_level: str = "Senior"
    top_k: int = 50
    sector: str = "TECH"
    semantic_weight: float = 0.60
    velocity_weight: float = 0.25
    freshness_weight: float = 0.15

@app.get("/")
def read_root():
    """Serves the front-end recruiter dashboard."""
    html_path = base_dir / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    raise HTTPException(status_code=404, detail="Recruiter Dashboard index.html not found.")

@app.post("/api/rank")
def api_rank(request: RankRequest):
    """Executes the 4-step end-to-end Talent Radar pipeline."""
    if not request.job_description.strip():
        raise HTTPException(status_code=400, detail="Job description cannot be empty.")
        
    try:
        pipeline = get_pipeline()
        candidates, expanded_query, timings = pipeline.run(
            job_description=request.job_description,
            seniority_level=request.seniority_level,
            top_k=request.top_k,
            sector=request.sector,
            semantic_weight=request.semantic_weight,
            velocity_weight=request.velocity_weight,
            freshness_weight=request.freshness_weight
        )
        
        # Format the top_k candidates response matching user input
        top_candidates = candidates[:request.top_k]
        for c in top_candidates:
            if "candidate_id" in c:
                c["id"] = c["candidate_id"]
        resolved_sector = getattr(pipeline, "resolved_sector", request.sector)
        
        return {
            "status": "success",
            "expanded_query": expanded_query,
            "timings": timings,
            "count_retrieved": len(candidates),
            "candidates": top_candidates,
            "sector": resolved_sector
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

@app.get("/api/sectors")
def api_sectors():
    """Returns the list of valid sector tokens and their human-readable labels."""
    SECTOR_LABELS = {
        "TECH": "Technology & Software",
        "FIN": "Finance & Banking",
        "HEALTH": "Healthcare & Life Sciences",
        "LEGAL": "Legal & Compliance",
        "REAL": "Real Estate",
        "MANU": "Manufacturing & Engineering",
        "COMM": "Commerce & Retail",
        "LOGI": "Logistics & Supply Chain",
        "MEDIA": "Media & Creative",
        "ENERGY": "Energy & Utilities",
        "EDU": "Education",
        "GOV": "Government & Public Sector"
    }
    return {"sectors": [{"token": k, "label": v} for k, v in SECTOR_LABELS.items()]}

@app.get("/api/templates")
def api_templates():
    """Returns sample Job Description templates for recruiters."""
    jds_dir = base_dir / "sample_jds"
    templates = []
    
    if jds_dir.exists():
        for file in jds_dir.glob("*.txt"):
            role_name = file.stem.replace("_", " ").title()
            with open(file, "r", encoding="utf-8") as f:
                content = f.read()
            templates.append({
                "id": file.stem,
                "name": role_name,
                "content": content,
                "seniority": "Senior" if "react" in file.stem or "backend" in file.stem else ("Lead" if "ml" in file.stem else "Senior")
            })
            
    # Default fallback templates if files are missing
    if not templates:
        templates = [
            {
                "id": "react_frontend",
                "name": "React Frontend Engineer",
                "content": "Looking for a Senior React Engineer with TypeScript experience. Optimizes Web Vitals and designs clean architectures.",
                "seniority": "Senior"
            }
        ]
        
    return templates

@app.get("/api/statistics")
def api_statistics():
    """Returns insights and demographic metrics about the 520 candidates in candidates.json."""
    candidates_path = base_dir / "candidates.json"
    if not candidates_path.exists():
        return {"status": "error", "message": "Candidates pool not indexed."}
        
    try:
        with open(candidates_path, "r", encoding="utf-8") as f:
            candidates = json.load(f)
            
        total = len(candidates)
        
        # Aggregations
        roles = {}
        experience_bands = {"Junior (<2 yrs)": 0, "Mid (2-5 yrs)": 0, "Senior (5-8 yrs)": 0, "Lead/Principal (>8 yrs)": 0}
        active_states = {"Active Now": 0, "Recent": 0, "Dormant": 0}
        
        import datetime
        today = datetime.date.today()
        
        for cand in candidates:
            # 1. Title roles
            title = cand["current_title"].lower()
            role = "Other"
            if "frontend" in title or "react" in title:
                role = "Frontend"
            elif "backend" in title or "api" in title or "distributed" in title:
                role = "Backend"
            elif "ml" in title or "data scientist" in title or "ai" in title or "nlp" in title:
                role = "Machine Learning"
            elif "devops" in title or "sre" in title or "cloud" in title:
                role = "DevOps"
                
            roles[role] = roles.get(role, 0) + 1
            
            # 2. Experience bands
            exp = cand["years_experience"]
            if exp < 2.0:
                experience_bands["Junior (<2 yrs)"] += 1
            elif exp < 5.0:
                experience_bands["Mid (2-5 yrs)"] += 1
            elif exp < 8.0:
                experience_bands["Senior (5-8 yrs)"] += 1
            else:
                experience_bands["Lead/Principal (>8 yrs)"] += 1
                
            # 3. Active status
            last_active = cand["last_active"]
            if last_active is None:
                active_states["Dormant"] += 1
            else:
                try:
                    act_date = datetime.date.fromisoformat(last_active)
                    days = (today - act_date).days
                    if days <= 7:
                        active_states["Active Now"] += 1
                    elif days <= 60:
                        active_states["Recent"] += 1
                    else:
                        active_states["Dormant"] += 1
                except Exception:
                    active_states["Dormant"] += 1
                    
        return {
            "total_indexed": total,
            "roles": roles,
            "experience_distribution": experience_bands,
            "freshness_distribution": active_states
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    """Receives a candidate JSON dataset file (overwrite), a raw PDF resume (AI-parsed, append), or a ZIP file of PDF resumes (extracted, parsed, and appended)."""
    global _pipeline
    
    filename = file.filename.lower()
    
    # Check if the uploaded file is a PDF
    if filename.endswith(".pdf") or file.content_type == "application/pdf":
        try:
            import io
            from pypdf import PdfReader
            
            # Read PDF bytes
            contents = await file.read()
            pdf_file = io.BytesIO(contents)
            
            # 1. Try pypdf text extraction
            extracted_text = ""
            try:
                reader = PdfReader(pdf_file)
                for page in reader.pages:
                    extracted_text += page.extract_text() or ""
                extracted_text = extracted_text.strip()
            except Exception as e:
                print(f"[Single Ingest] pypdf failed: {e}")
                extracted_text = ""
                
            # 2. Try pdfplumber fallback if text is empty
            if not extracted_text:
                try:
