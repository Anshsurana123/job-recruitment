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
                    import pdfplumber
                    pdf_file.seek(0)
                    with pdfplumber.open(pdf_file) as pdf:
                        for page in pdf.pages:
                            extracted_text += page.extract_text() or ""
                    extracted_text = extracted_text.strip()
                    if extracted_text:
                        print(f"[Single Ingest] pdfplumber successfully recovered text.")
                except Exception as e:
                    print(f"[Single Ingest] pdfplumber fallback failed: {e}")
                    
            # Parse candidate profile using Gemini Resume Parser
            from talent_radar.llm_parser import GeminiResumeParser
            parser = GeminiResumeParser()

            if not extracted_text:
                print(f"[Single Ingest] Scanned/Non-Text PDF detected. Attempting native visual PDF parsing...")
                try:
                    parsed_candidate = parser.parse_resume_pdf(contents, file.filename)
                    extracted_text = parsed_candidate.get("resume_text", "")
                except Exception as vis_err:
                    print(f"[Single Ingest] Visual PDF parsing failed: {vis_err}")
                    raise HTTPException(status_code=400, detail=f"The uploaded PDF has no extractable text, and visual parsing failed: {str(vis_err)}")
            else:
                print(f"Extracted {len(extracted_text)} characters from PDF. Sending to Gemini model {parser.model} for structured parsing...")
                parsed_candidate = parser.parse_resume(extracted_text, file.filename)
            
            # Load existing candidates from pool
            candidates_path = base_dir / "candidates.json"
            existing_candidates = []
            if candidates_path.exists():
                try:
                    with open(candidates_path, "r", encoding="utf-8") as f:
                        existing_candidates = json.load(f)
                except Exception:
                    existing_candidates = []
            
            # Formulate new candidate and assign ID standardizing on 4-digit width
            candidate_name = str(parsed_candidate.get("name", "Unknown Candidate")).strip()
            name_not_extracted = bool(parsed_candidate.get("name_not_extracted", False))
            
            # Clean and validate career history
            cleaned_history = []
            for job in parsed_candidate.get("career_history", []):
                if isinstance(job, dict):
                    cleaned_history.append({
                        "title": str(job.get("title", "Developer")),
                        "company": str(job.get("company", "Company")),
                        "start_date": str(job.get("start_date", "2020-01-01")),
                        "end_date": job.get("end_date")
                    })
            
            # Duplicate check
            existing_cand = None
            if candidate_name and candidate_name.lower() != "unknown candidate" and not name_not_extracted:
                existing_cand = next((c for c in existing_candidates if c.get("name", "").strip().lower() == candidate_name.lower()), None)
            
            if existing_cand:
                # Merge in-place
                existing_cand["name_not_extracted"] = False
                if parsed_candidate.get("current_title"):
                    existing_cand["current_title"] = str(parsed_candidate.get("current_title"))
                existing_cand["years_experience"] = max(
                    float(parsed_candidate.get("years_experience", 1.0)),
                    float(existing_cand.get("years_experience", 1.0))
                )
                if parsed_candidate.get("education"):
                    existing_cand["education"] = str(parsed_candidate.get("education"))
                if parsed_candidate.get("location"):
                    existing_cand["location"] = str(parsed_candidate.get("location"))
                if parsed_candidate.get("last_active"):
                    existing_cand["last_active"] = str(parsed_candidate.get("last_active"))
                existing_cand["resume_text"] = str(parsed_candidate.get("resume_text", extracted_text))
                existing_cand["source_file"] = file.filename
                
                # Merge skills
                skills_union = list(dict.fromkeys(
                    [s.strip() for s in existing_cand.get("skills_listed", [])] +
                    [s.strip() for s in parsed_candidate.get("skills_listed", [])]
                ))
                existing_cand["skills_listed"] = skills_union
                
                # Merge career history
                history_by_key = {}
                for job_entry in existing_cand.get("career_history", []) + cleaned_history:
                    key = (
                        job_entry.get("company", "").strip().lower(),
                        job_entry.get("title", "").strip().lower(),
                        job_entry.get("start_date", "").strip()
                    )
                    history_by_key[key] = job_entry
                existing_cand["career_history"] = list(history_by_key.values())
                
                candidate_record = existing_cand
                print(f"[Deduplication] Merged duplicate candidate '{candidate_name}' in-place in app.py upload.")
            else:
                existing_ids = {c.get("candidate_id") for c in existing_candidates}
                max_idx = 0
                for cid in existing_ids:
                    if cid and cid.startswith("cand_"):
                        try:
                            idx_part = int(cid.split("_")[1])
                            if idx_part > max_idx:
                                max_idx = idx_part
                        except (IndexError, ValueError):
                            pass
                next_idx = max_idx + 1
                cand_id = f"cand_{next_idx:04d}"
                    
                candidate_record = {
                    "candidate_id": cand_id,
                    "name": candidate_name,
                    "name_not_extracted": name_not_extracted,
                    "current_title": str(parsed_candidate.get("current_title", "Software Engineer")),
                    "years_experience": float(parsed_candidate.get("years_experience", 1.0)),
                    "career_history": cleaned_history,
                    "skills_listed": parsed_candidate.get("skills_listed", []),
                    "last_active": parsed_candidate.get("last_active") if parsed_candidate.get("last_active") else None,
                    "education": str(parsed_candidate.get("education", "")),
                    "location": str(parsed_candidate.get("location", "Remote")),
                    "resume_text": str(parsed_candidate.get("resume_text", extracted_text)),
                    "source_file": file.filename
                }
                # Append new candidate to existing pool
                existing_candidates.append(candidate_record)
            
            # Overwrite candidates.json with the expanded candidate pool
            with open(candidates_path, "w", encoding="utf-8") as f:
                json.dump(existing_candidates, f, indent=2, ensure_ascii=False)
                
            # Clear global pipeline cache to force reload candidate pool on next search query
            with pipeline_lock:
                _pipeline = None
            
            return {
                "status": "success",
                "message": f"Successfully parsed and indexed candidate '{candidate_record['name']}'.",
                "count": len(existing_candidates),
                "candidate": {
                    "candidate_id": candidate_record["candidate_id"],
                    "name": candidate_record["name"],
                    "current_title": candidate_record["current_title"],
                    "years_experience": candidate_record["years_experience"],
                    "skills_listed": candidate_record["skills_listed"]
                }
            }
            
        except HTTPException as he:
            raise he
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"PDF Ingestion Failed: {str(e)}")
            
    # Check if the uploaded file is a ZIP archive
    elif filename.endswith(".zip") or file.content_type in ["application/zip", "application/x-zip-compressed"]:
        try:
            # Read ZIP bytes into memory
            contents = await file.read()

            if len(contents) == 0:
                raise HTTPException(status_code=400, detail="The uploaded ZIP file is empty.")

            # Create a Smart Ingest job and register it
            job = SmartIngestJob()
            with jobs_lock:
                _active_jobs[job.job_id] = job

            # Define the background worker function
            def _run_ingest_background(zip_bytes: bytes, ingest_job: SmartIngestJob):
                global _pipeline
                try:
                    run_smart_ingest(zip_bytes, ingest_job)
                finally:
                    # Clear pipeline cache thread-safely
                    with pipeline_lock:
                        _pipeline = None

            # Launch in a background thread (not blocking the HTTP response)
            ingest_thread = threading.Thread(
                target=_run_ingest_background,
                args=(contents, job),
                daemon=True,
            )
            ingest_thread.start()

            # Return immediately with HTTP 202 Accepted
            return {
                "status": "accepted",
                "job_id": job.job_id,
                "message": f"Smart ingest started for '{file.filename}'. "
                           f"Stream progress at /api/upload/progress/{job.job_id}",
            }

        except HTTPException as he:
            raise he
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"ZIP Ingestion Failed: {str(e)}")
            
    # Process standard JSON dataset upload
    else:
        try:
            contents = await file.read()
            data = json.loads(contents.decode("utf-8"))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid file format: {str(e)}. Must be a valid JSON file.")
            
        if not isinstance(data, list):
            raise HTTPException(status_code=400, detail="Invalid data format. Dataset must be a JSON array (list of candidates).")
            
        if len(data) == 0:
            raise HTTPException(status_code=400, detail="Uploaded dataset is empty.")
            
        validated_candidates = []
        required_keys = ["candidate_id", "name", "current_title", "resume_text"]
        
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                raise HTTPException(status_code=400, detail=f"Candidate at index {idx} must be an object.")
                
            for key in required_keys:
                if key not in item:
                    raise HTTPException(status_code=400, detail=f"Candidate at index {idx} is missing required field '{key}'.")
                    
            candidate_id = str(item["candidate_id"])
            name = str(item["name"])
            current_title = str(item["current_title"])
            resume_text = str(item["resume_text"])
            
            years_experience = float(item.get("years_experience", 1.0))
            skills_listed = item.get("skills_listed", [])
            if not isinstance(skills_listed, list):
                skills_listed = [str(skills_listed)]
            skills_listed = [str(s) for s in skills_listed]
            
            career_history = item.get("career_history", [])
            if not isinstance(career_history, list):
                career_history = []
            cleaned_history = []
            for job in career_history:
                if isinstance(job, dict):
                    cleaned_history.append({
                        "title": str(job.get("title", "Developer")),
                        "company": str(job.get("company", "Company")),
                        "start_date": str(job.get("start_date", "2020-01-01")),
                        "end_date": job.get("end_date")
                    })
            
            last_active = item.get("last_active")
            if last_active is not None:
                last_active = str(last_active)
                
            validated_candidates.append({
                "candidate_id": candidate_id,
                "name": name,
                "current_title": current_title,
                "resume_text": resume_text,
                "years_experience": years_experience,
                "career_history": cleaned_history,
                "skills_listed": skills_listed,
                "last_active": last_active
            })
            
        candidates_path = base_dir / "candidates.json"
        try:
            with open(candidates_path, "w", encoding="utf-8") as f:
                json.dump(validated_candidates, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save dataset: {str(e)}")
            
        with pipeline_lock:
            _pipeline = None
            
        return {
            "status": "success",
            "message": f"Successfully uploaded and indexed {len(validated_candidates)} candidates.",
            "count": len(validated_candidates)
        }

@app.delete("/api/upload")
def api_delete_dataset():
    """Deletes the active custom candidate dataset and restores the pool to an empty clean slate (0 candidates)."""
    global _pipeline
    try:
        print("Restoring candidate pool to a clean slate (0 candidates)...")
        candidates_path = base_dir / "candidates.json"
        with open(candidates_path, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2, ensure_ascii=False)
            
        with pipeline_lock:
            _pipeline = None
        
        return {
            "status": "success",
            "message": "Successfully cleared custom dataset and restored database back to 0 candidates (clean slate).",
            "count": 0
        }
    except Exception as e:
