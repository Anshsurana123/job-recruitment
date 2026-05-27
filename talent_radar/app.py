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
            is_generic_name = candidate_name.lower().strip() in {
                "not specified", "unknown", "n/a", "none", "unknown candidate", "null", 
                "not specified.", "not-specified", "unspecified", "name", "candidate name", "n.a."
            }
            if candidate_name and not is_generic_name and not name_not_extracted:
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
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to reset dataset: {str(e)}")

@app.get("/api/upload/progress/{job_id}")
async def api_upload_progress(job_id: str):
    """
    Server-Sent Events (SSE) endpoint that streams real-time progress
    for a background Smart Ingest job. Emits a JSON event every 2 seconds
    with {parsed, total, skipped, failed, percent, status}.
    """
    with jobs_lock:
        if job_id not in _active_jobs:
            raise HTTPException(status_code=404, detail=f"No active ingest job found with id '{job_id}'.")
        job = _active_jobs[job_id]

    import asyncio

    async def event_generator():
        while True:
            data = job.to_dict()

            # If job is done or errored, send final event and close
            if job.status in ("done", "error"):
                # Attach total candidate count from the pool
                candidates_path = base_dir / "candidates.json"
                total_count = 0
                if candidates_path.exists():
                    try:
                        with open(candidates_path, "r", encoding="utf-8") as f:
                            total_count = len(json.load(f))
                    except Exception:
                        pass
                data["total_candidate_count"] = total_count

                yield f"data: {json.dumps(data)}\n\n"

                # Clean up: remove finished job from active tracking to prevent memory leak
                with jobs_lock:
                    _active_jobs.pop(job_id, None)
                break

            yield f"data: {json.dumps(data)}\n\n"
            await asyncio.sleep(2)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

class SettingsRequest(BaseModel):
    gemini_api_key: str | None = None
    hf_token: str | None = None
    local_ocr_backend: str | None = None

@app.get("/api/settings")
def api_get_settings():
    """Returns currently configured keys and model selections (masked for security)."""
    gemini_key = os.getenv("GEMINI_API_KEY") or ""
    hf_tok = os.getenv("HF_TOKEN") or ""
    ocr_backend = os.getenv("LOCAL_OCR_BACKEND") or "qwen2.5-vl"
    
    def mask_key(k):
        if not k:
            return ""
        if len(k) <= 8:
            return "*" * len(k)
        return k[:4] + "*" * (len(k) - 8) + k[-4:]

    return {
        "gemini_api_key_masked": mask_key(gemini_key),
        "hf_token_masked": mask_key(hf_tok),
        "local_ocr_backend": ocr_backend
    }

@app.post("/api/settings")
def api_post_settings(request: SettingsRequest):
    """Dynamically writes key configurations back to .env and updates the active process environment variables."""
    global _pipeline
    try:
        env_path = base_dir.parent / ".env"
        lines = []
        if env_path.exists():
            with open(env_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                
        # Parse existing keys
        env_keys = {}
        for i, line in enumerate(lines):
            clean = line.strip()
            if clean and not clean.startswith("#") and "=" in clean:
                parts = clean.split("=", 1)
                env_keys[parts[0].strip()] = (i, parts[1].strip())
                
        def update_key_in_lines(key, value):
            if value is None:
                return
            # If value contains asterisks, it means the user kept the masked placeholder - do not overwrite
            if "*" in value:
                return
            
            # Input sanitization against newline injection attacks
            if "\n" in value or "\r" in value:
                raise HTTPException(status_code=400, detail="Invalid character detected in configuration settings.")
                
            clean_value = value.strip().strip('"').strip("'")
            os.environ[key] = clean_value
            
            if key == "HF_TOKEN":
                os.environ["HF_TOKEN"] = clean_value
                os.environ["HUGGING_FACE_HUB_TOKEN"] = clean_value
                
            value_str = f'{key}="{clean_value}"\n' if key != "LOCAL_OCR_BACKEND" else f'{key}={clean_value}\n'
            if key in env_keys:
                idx, _ = env_keys[key]
                lines[idx] = value_str
            else:
                lines.append(value_str)
                
        update_key_in_lines("GEMINI_API_KEY", request.gemini_api_key)
        update_key_in_lines("HF_TOKEN", request.hf_token)
        update_key_in_lines("LOCAL_OCR_BACKEND", request.local_ocr_backend)
        
        # Write updated lines back to .env
        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
            
        # Reset the pipeline and OCR singleton to pick up the new keys/tokens instantly
        with pipeline_lock:
            _pipeline = None
        from talent_radar.ocr_engine import LocalOCREngine
        LocalOCREngine._instance = None
        
        return {
            "status": "success",
            "message": "System settings and API keys updated successfully."
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to save settings: {str(e)}")

# ===================================================================== #
#  New Recruiter-Centric Endpoints                                      #
# ===================================================================== #

class QuestionRequest(BaseModel):
    job_description: str
    sector: str = "TECH"

@app.post("/api/candidates/{candidate_id}/questions")
def api_generate_questions(candidate_id: str, request: QuestionRequest):
    """Generates 5 personalized phone screen questions using Gemini 2.5 Flash."""
    jd = request.job_description
    sector = request.sector
    if not jd.strip():
        raise HTTPException(status_code=400, detail="Job description cannot be empty.")
    
    pipeline = get_pipeline()
    # Support both id (frontend usage) and candidate_id (backend json structure)
    cand = next((c for c in pipeline.candidates_pool if c.get("candidate_id") == candidate_id or c.get("id") == candidate_id), None)
    
    if not cand:
        raise HTTPException(status_code=404, detail=f"Candidate with ID '{candidate_id}' not found in current pool.")
        
    from talent_radar.question_generator import TechnicalQuestionGenerator
    generator = TechnicalQuestionGenerator()
    try:
        # Perform quick skills gaps detection if candidate lacks matches/gaps fields
        if "matched_skills" not in cand or "missing_skills" not in cand:
            import re
            from talent_radar.scorer import CandidateScorer
            from talent_radar.matrix_pipeline import GeminiGateway
            try:
                gateway = GeminiGateway()
                refined_matrix = gateway.route_and_polish(jd, sector)
                target_keywords = refined_matrix.top_keywords
            except Exception:
                target_keywords = list(set(re.findall(r'\b[a-zA-Z]{3,}\b', jd.lower())))[:30]
                
            scorer = CandidateScorer(target_keywords=target_keywords, sector=sector)
            cand_copy = cand.copy()
            if "semantic_depth_score" not in cand_copy:
                cand_copy["semantic_depth_score"] = 0.5
            scored = scorer.score_candidates([cand_copy])
            cand = scored[0]

        guide = generator.generate_questions(cand, jd, sector)
        return guide.model_dump()
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate questions: {str(e)}")

class CompareRequest(BaseModel):
    candidate_ids: list[str]
    job_description: str
    sector: str = "TECH"

@app.post("/api/compare")
def api_compare_candidates(request: CompareRequest):
    """Compares 2 or 3 candidates side-by-side using Gemini 2.5 Flash."""
    if not request.job_description.strip():
        raise HTTPException(status_code=400, detail="Job description cannot be empty.")
    if len(request.candidate_ids) < 2:
        raise HTTPException(status_code=400, detail="Must provide at least two candidate IDs to compare side-by-side.")
        
    pipeline = get_pipeline()
    selected_cands = []
    for cid in request.candidate_ids:
        # Check by candidate_id or id (supporting both frontend/backend naming schemas)
        cand = next((c for c in pipeline.candidates_pool if c.get("candidate_id") == cid or c.get("id") == cid), None)
        if cand:
            selected_cands.append(cand.copy())
            
    if not selected_cands:
        raise HTTPException(status_code=404, detail="None of the specified candidates were found.")
        
    # Match skills gaps and metrics
    from talent_radar.scorer import CandidateScorer
    from talent_radar.matrix_pipeline import GeminiGateway
    import re
    try:
        gateway = GeminiGateway()
        refined_matrix = gateway.route_and_polish(request.job_description, request.sector)
        target_keywords = refined_matrix.top_keywords
    except Exception:
        target_keywords = list(set(re.findall(r'\b[a-zA-Z]{3,}\b', request.job_description.lower())))[:30]
        
    scorer = CandidateScorer(target_keywords=target_keywords, sector=request.sector)
    for c in selected_cands:
        if "semantic_depth_score" not in c:
            c["semantic_depth_score"] = 0.5
    scored_cands = scorer.score_candidates(selected_cands)
    
    from talent_radar.comparison import CandidateComparator
    comparator = CandidateComparator()
    try:
        synthesis = comparator.compare_candidates(scored_cands, request.job_description)
        return synthesis.model_dump()
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

class ExportRequest(BaseModel):
    job_description: str
    seniority_level: str = "Senior"
    sector: str = "TECH"
    candidates: list[dict]
    format: str = "markdown"

@app.post("/api/export")
def api_export_brief(request: ExportRequest):
    """Generates and downloads a shareable candidate search brief (Markdown or CSV)."""
    from talent_radar.exporter import BriefExporter
    if request.format.lower() == "csv":
        csv_data = BriefExporter.generate_csv_brief(request.candidates)
        return StreamingResponse(
            io.BytesIO(csv_data.encode("utf-8")),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=talent_radar_brief.csv"}
        )
    else:
        md_data = BriefExporter.generate_markdown_brief(
            request.job_description,
            request.seniority_level,
            request.sector,
            request.candidates
        )
        return StreamingResponse(
            io.BytesIO(md_data.encode("utf-8")),
            media_type="text/markdown",
            headers={"Content-Disposition": "attachment; filename=talent_radar_brief.md"}
        )

class OutreachRequest(BaseModel):
    job_description: str
    tone: str = "professional"

@app.post("/api/candidates/{candidate_id}/outreach")
def api_candidate_outreach(candidate_id: str, request: OutreachRequest):
    """Generates a highly-personalized outreach email using Gemini 2.5 Flash."""
    pipeline = get_pipeline()
    cand = next((c for c in pipeline.candidates_pool if c.get("candidate_id") == candidate_id or c.get("id") == candidate_id), None)
    if not cand:
        raise HTTPException(status_code=404, detail="Candidate not found.")
        
    skills = ", ".join(cand.get("skills_listed", []))
    title = cand.get("current_title", "Software Engineer")
    name = cand.get("name", "Candidate")
    experience = cand.get("years_experience", 1.0)
    
    # Check if Gemini key is available
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        try:
            from google import genai
            from google.genai import types
            
            client = genai.Client(api_key=api_key)
            prompt = f"""
            Write a personalized recruitment outreach email to the candidate below.
            
            Candidate Details:
            - Name: {name}
            - Current Title: {title}
            - Experience: {experience} years
            - Skills: {skills}
            
            Target Job Description:
            "{request.job_description}"
            
            Email Tone: {request.tone}
            
            Requirements:
            1. Keep it professional, highly engaging, and not overly salesy.
            2. Proactively mention their specific skills ({skills}) and explain why they are a perfect fit for this role.
            3. Make it concise and end with a call to action (scheduling a brief call).
            4. Do not include placeholders like "[Your Name]". Write the email as a senior recruiter from "SwarmMatrix executive search".
            """
            
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction="You are a premium headhunter and tech recruiter known for writing high-conversion outreach emails.",
                    temperature=0.7
                )
            )
            email_content = response.text.strip()
            return {
                "status": "success",
                "email": email_content
            }
        except Exception as e:
            print(f"[Outreach Engine] Structured generation error: {e}. Cascading to local fallback...")
            
    # Fallback email generation
    tone_greeting = "Hi" if request.tone.lower() == "casual" else "Dear"
    salutation = "Best regards,\nThe SwarmMatrix Executive Search Team"
    
    fallback_email = f"""Subject: Exciting Career Opportunity: {title} Role at SwarmMatrix Client

{tone_greeting} {name},

I hope this email finds you well. 

I came across your impressive profile while sourcing talent for a highly selective {title} position. Your extensive background as a {title} with {experience:.1f} years of experience and your deep expertise in {skills or "cutting-edge engineering stacks"} immediately stood out to us.

We are currently representing a high-growth client seeking a key contributor to lead development in areas that align perfectly with your technical skillset. Given your experience, we believe you would bring immense value and unique insights to their engineering team.

I would love to schedule a brief 10-minute introductory call to share more about the role, the team culture, and see if this aligns with your career goals. 

Would you be open to a quick call sometime this week? Let me know a few times that work best for you.

{salutation}"""
    
    return {
        "status": "success",
        "email": fallback_email.strip()
    }

@app.post("/api/candidates/{candidate_id}/similar")
def api_candidate_similar(candidate_id: str):
    """Retrieves candidates matching a target candidate profile's skills and title."""
    pipeline = get_pipeline()
    cand = next((c for c in pipeline.candidates_pool if c.get("candidate_id") == candidate_id or c.get("id") == candidate_id), None)
    if not cand:
        raise HTTPException(status_code=404, detail="Candidate not found.")
        
    skills = ", ".join(cand.get("skills_listed", []))
    title = cand.get("current_title", "Software Engineer")
    
    # Mock job description to find candidates similar to this one
    job_description = f"Looking for a {title} skilled in {skills}."
    
    try:
        # Run ranking pipeline using the target candidate's profile
        candidates, expanded_query, timings = pipeline.run(
            job_description=job_description,
            seniority_level=cand.get("seniority_level", "Senior"),
            top_k=50,
            sector="TECH"
        )
        
        # Exclude the source candidate themselves from the similarity list
        similar_candidates = [c for c in candidates if c.get("candidate_id") != candidate_id][:10]
        
        # Format candidate IDs
        for c in similar_candidates:
            if "candidate_id" in c:
                c["id"] = c["candidate_id"]
                
        return {
            "status": "success",
            "candidates": similar_candidates,
            "query_used": job_description,
            "timings": timings
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Similarity search error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Start web server when executing directly
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
