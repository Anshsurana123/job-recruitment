---
title: Job Recruitment
emoji: 🌌
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# 🧠 Job Recruitment AI — Talent Radar
### *Multi-Agent SLM Swarm Architecture for Intelligent Candidate Ranking & Recruiter Copilot*

> **Job Recruitment AI** is a production-grade AI recruiting engine that replaces keyword-matching ATS tools with a multi-phase, multi-model pipeline. It combines a Gemini LLM gateway, a swarm of domain-specialized Small Language Models (SLMs), a composite momentum scoring engine, and an intelligent **Recruiter Copilot Toolkit** to surface, evaluate, compare, and engage the *right* candidates — not just the ones with the most buzzwords on their resume.

---

## 📋 Table of Contents

- [What It Does](#-what-it-does)
- [Why It Exists — The Problem with ATS](#-why-it-exists--the-problem-with-ats)
- [System Architecture](#-system-architecture)
- [How It Works — Step by Step](#-how-it-works--step-by-step)
  - [Phase 0: Gemini Gateway (Input Refinement)](#phase-0-gemini-gateway-input-refinement)
  - [Phase 1: Query Explosion](#phase-1-query-explosion)
  - [Phase 2: Smart Candidate Pre-Filtering](#phase-2-smart-candidate-pre-filtering)
  - [Phase 3: Swarm Matrix Evaluation (SLM Swarm)](#phase-3-swarm-matrix-evaluation-slm-swarm)
  - [Phase 4: Composite Momentum Scoring](#phase-4-composite-momentum-scoring)
- [The Swarm Matrix — Technical Deep Dive](#-the-swarm-matrix--technical-deep-dive)
- [Recruiter Copilot Toolkit](#-recruiter-copilot-toolkit)
  - [Tailored Phone Screen Guide Generator](#tailored-phone-screen-guide-generator)
  - [Side-by-Side Candidate Comparator](#side-by-side-candidate-comparator)
  - [Personalized Outreach Email Generator](#personalized-outreach-email-generator)
  - [Similar Talent Search](#similar-talent-search)
- [Resume Ingestion Pipeline](#-resume-ingestion-pipeline)
  - [Single PDF Ingestion Cascade](#single-pdf-ingestion-cascade)
  - [Bulk ZIP Upload & Threaded Smart Ingest](#bulk-zip-upload--threaded-smart-ingest)
  - [Local OCR Engine with Qwen2-VL & EasyOCR](#local-ocr-engine-with-qwen2-vl--easyocr)
  - [Real-Time SSE Progress Tracking](#real-time-sse-progress-tracking)
  - [Atomic Checkpointing & Integrity](#atomic-checkpointing--integrity)
- [Scoring Formula & Guardrails](#-scoring-formula--guardrails)
- [API Reference](#-api-reference)
- [Frontend Recruiter Dashboard](#-frontend-recruiter-dashboard)
- [Technical Choices & Rationale](#-technical-choices--rationale)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Docker Deployment (Hugging Face Spaces)](#-docker-deployment-hugging-face-spaces)
- [Environment Variables](#-environment-variables)
- [Sector Coverage](#-sector-coverage)

---

## 🎯 What It Does

Job Recruitment AI takes a raw job description and returns a ranked, scored, and explained list of matching candidates from your talent pool. It performs:

1. **Semantic understanding** of the JD — not just keyword extraction
2. **Domain-specialist model routing** — uses a TECH-specific model for tech roles, a FinBERT model for finance roles, etc.
3. **Resume chunking + fragment-level scoring** — avoids superficial keyword matching by evaluating *contextual depth* of experience
4. **Career momentum calculation** — rewards candidates who have grown quickly and consistently
5. **Guardrail enforcement** — catches keyword stuffers, seniority mismatches, and duplicate-content resumes
6. **Smart Ingest with Local OCR** — processes raw digital or scanned PDFs in bulk using visual language models
7. **Recruiter Copilot Toolkit** — generates customized screen guides, writes personalized outreach emails, discovery of similar profiles, and structures side-by-side comparative dossiers

---

## ❓ Why It Exists — The Problem with ATS

Traditional Applicant Tracking Systems (ATS) rank candidates using **TF-IDF keyword overlap**. A resume that says "AI systems" in a job about AI will rank higher than a specialist who writes about "flash attention kernel optimization" — because "AI" is a keyword match and "flash attention" is not in the JD verbatim.

**Job Recruitment breaks this pattern with three dimensions of innovation:**

| Problem | Job Recruitment Solution |
|---|---|
| Superficial keyword matching | Domain-specialized SLM swarm with semantic cosine similarity |
| Generic embeddings for all industries | 12-sector model routing (TECH → CodeBERT, FIN → FinBERT, HEALTH → BioBERT...) |
| No experience quality signal | Career Velocity score: seniority progression speed over time |
| Resume stuffers game the system | Keyword stuffer guardrail + duplicate content detection |
| Scanned PDFs not parseable | Local Qwen2-VL vision model ➔ EasyOCR fallback cascade |
| Static job description as-is | Query Explosion via LLM to add expert-level adjacent signals |
| No screen prep or candidate context | **Recruiter Copilot**: personalized questions, side-by-side trade-offs, and custom outreach |
| Data loss on upload crash | **Atomic Checkpointing** using temporary file replacement buffers |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     TALENT RADAR PIPELINE                       │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │ Job Desc     │    │ GEMINI-2.5-  │    │  Requirements    │  │
│  │ (raw text)   │───▶│ FLASH        │───▶│  Matrix          │  │
│  │              │    │ GATEWAY      │    │  (structured)    │  │
│  └──────────────┘    └──────────────┘    └─────────┬────────┘  │
│                                                     │           │
│  ┌──────────────────────────────────────────────────▼────────┐  │
│  │               PHASE 1: QUERY EXPLOSION                    │  │
│  │   LLM / Rule-based semantic domain expansion              │  │
│  │   Adds: architectures, failure modes, behavioral signals  │  │
│  └──────────────────────────────────────────────────┬────────┘  │
│                                                     │           │
│  ┌──────────────────────────────────────────────────▼────────┐  │
│  │          PHASE 2: SMART CANDIDATE PRE-FILTERING           │  │
│  │   Hybrid keyword scoring: query overlap + title bonus     │  │
│  │   + Gemini top-30 keywords bonus → top 20% candidates     │  │
│  └──────────────────────────────────────────────────┬────────┘  │
│                                                     │           │
│  ┌──────────────────────────────────────────────────▼────────┐  │
│  │         PHASE 3: SWARM MATRIX SLM EVALUATION             │  │
│  │                                                           │  │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │  │
│  │   │ SECTOR      │  │ RESUME      │  │ MEGA-BATCH      │  │  │
│  │   │ ROUTING     │  │ CHUNKER     │  │ INFERENCE       │  │  │
│  │   │ (12 models) │  │ (300 words) │  │ (all fragments) │  │  │
│  │   └──────┬──────┘  └──────┬──────┘  └───────┬─────────┘  │  │
│  │          │                │                  │            │  │
│  │          └────────────────┴──────────────────┘            │  │
│  │                     Cosine Similarity with                │  │
│  │                     Anisotropy Correction                 │  │
│  └──────────────────────────────────────────────────┬────────┘  │
│                                                     │           │
│  ┌──────────────────────────────────────────────────▼────────┐  │
│  │         PHASE 4: COMPOSITE MOMENTUM SCORER               │  │
│  │   60% Semantic Depth + 25% Career Velocity + 15% Freshness│  │
│  │   + Guardrails: keyword stuffer / seniority cap / dupes   │  │
│  └──────────────────────────────────────────────────┬────────┘  │
│                                                     │           │
│  ┌──────────────────────────────────────────────────▼────────┐  │
│  │            PHASE 5: RECRUITER COPILOT TOOLKIT             │  │
│  │                                                           │  │
│  │ ┌──────────────┐ ┌─────────────┐ ┌──────────┐ ┌─────────┐ │  │
│  │ │ 5 tailored   │ │ Side-by-Side│ │ Custom   │ │ Similar │ │  │
│  │ │ questions    │ │ comparison  │ │ outreach │ │ search  │ │  │
│  │ │ (with gaps)  │ │ synthesis   │ │ emails   │ │ queries │ │  │
│  │ └──────────────┘ └─────────────┘ └──────────┘ └─────────┘ │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔬 How It Works — Step by Step

### Phase 0: Gemini Gateway (Input Refinement)

**File:** [`talent_radar/matrix_pipeline.py`](talent_radar/matrix_pipeline.py) → `GeminiGateway`

Before the pipeline even starts, the raw job description is sent to **Gemini 2.5 Flash** with a structured output schema (`RequirementsMatrix`). This extracts three things:

| Output | Description | Example |
|---|---|---|
| `polished_requirements` | Dense, refined string of core domain concepts | *"ML inference optimization, GPU kernel engineering, quantization"* |
| `sector_token` | One of 12 industry macro codes | `"TECH"` |
| `top_keywords` | Top 30 domain-specific terms extracted from the JD | `["flash attention", "triton", "quantization", ...]` |

**Structured API Integration:**
```python
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
    config=types.GenerateContentConfig(
        system_instruction=system_instr,
        response_mime_type="application/json",
        response_schema=RequirementsMatrix,
        temperature=0.1
    )
)
```

**Fallback:** If the Gemini API is unavailable, a local regex-based fallback parser extracts words and maps sector tokens.

---

### Phase 1: Query Explosion

**File:** [`talent_radar/query_expand.py`](talent_radar/query_expand.py) → `expand_query()`

The raw job description is **semantically expanded** beyond what the recruiter wrote. Think of this as adding what an expert *would write on their resume* but the recruiter forgot to mention.

**Domain Detection:** Keywords in the JD trigger one or more domain clusters:
- `react`, `ui`, `next.js` → **Frontend** domain
- `kafka`, `redis`, `grpc`, `django` → **Backend** domain
- `pytorch`, `llm`, `nlp` → **Machine Learning** domain
- `kubernetes`, `terraform`, `sre` → **DevOps** domain

**Expansion adds:**
- Architectural patterns: *"micro-frontend", "module federation", "CQRS", "event-driven"*
- Failure modes: *"hydration mismatch", "N+1 query join", "gradient explosion", "pod crashloopbackoff"*
- Performance signals: *"bundle splitting", "LCP", "model quantization", "read replicas"*
- Behavioral seniority signals: Senior → *"designed, led, owned, migrated, reduced latency by"*

**LLM Priority:** Uses OpenAI (`gpt-4o-mini`) → Gemini (`gemini-1.5-flash`) → local rule-based dictionary, depending on which keys are available.

---

### Phase 2: Smart Candidate Pre-Filtering

**File:** [`talent_radar/pipeline.py`](talent_radar/pipeline.py) → `TalentRadarPipeline.run()`

Before the expensive SLM forward pass, candidates are scored by a **3-signal hybrid keyword filter**:

| Signal | Weight | Description |
|---|---|---|
| Query word overlap | `1× per unique word` | How many expanded query words appear in resume text |
| Job title overlap | `3× per word` | How many query words match historical job titles |
| Gemini keyword bonus | `5× per word` | How many Gemini-extracted top-30 keywords appear in resume |

Only the **top 20%** of candidates by this score (minimum 50 candidates or the whole pool if smaller) are passed to the expensive SLM swarm. This reduces the compute cost by ~5× while preserving high-signal candidates.

---

### Phase 3: Swarm Matrix Evaluation (SLM Swarm)

**File:** [`talent_radar/matrix_pipeline.py`](talent_radar/matrix_pipeline.py) → `JobRecruitmentRanker`

This is the core innovation. Instead of one generic embedding model for all industries, Job Recruitment routes to **12 sector-specialized models**:

| Sector Token | Model | Specialty |
|---|---|---|
| `TECH` | `microsoft/codebert-base` | Code, software engineering, APIs |
| `FIN` | `ProsusAI/finbert` | Finance, banking, trading |
| `HEALTH` | `dmis-lab/biobert-base-cased-v1.2` | Biomedical, clinical, pharma |
| `LEGAL` | `nlpaueb/legal-bert-small-uncased` | Legal contracts, compliance |
| `REAL` | `llmware/industry-bert-asset-management-v0.1` | Real estate, assets |
| `MANU` | `cea-list-ia/ManufactuBERT` | Manufacturing, industrial |
| `COMM` | `nlptown/bert-base-multilingual-uncased-sentiment` | Commerce, retail |
| `LOGI` | `inovex/multi2convai-logistics-en-bert` | Logistics, supply chain |
| `MEDIA` | `cardiffnlp/twitter-roberta-base` | Media, creative, social |
| `ENERGY` | `Master-AI-Lab/EnergyBERT` | Energy, utilities |
| `EDU` | `vasugoel/K-12BERT` | Education |
| `GOV` | `ESGBERT/GovRoBERTa-governance` | Government, public sector |

**The Swarm Evaluation Process:**
```
1. CHUNK: Each resume → 300-word paragraph chunks (sentence-boundary aligned)
2. FLATTEN: All chunks from all candidates → ONE flat fragment list
3. MEGA-BATCH: Single forward pass over all fragments (batch_size=32)
4. SCORE: Cosine similarity between query embedding and each fragment
5. AGGREGATE: MAX(fragment scores) per candidate = Best Chunk Alignment Score
```

**Anisotropy Correction:** Pre-trained BERT models suffer from *representation degeneration* — embeddings cluster in a narrow cone, reducing cosine similarity resolution. Job Recruitment applies **subtraction centering** before cosine computation:
```python
mu = torch.mean(all_frag_embs, dim=0, keepdim=True)
query_emb_centered = query_emb - mu
frag_embs_centered = all_frag_embs - mu
```
This redistributes the embedding space, making cosine similarities meaningful across the full `[-1, 1]` range and mapped to `[0, 1]`.

---

### Phase 4: Composite Momentum Scoring

**File:** [`talent_radar/scorer.py`](talent_radar/scorer.py) → `CandidateScorer`

The final score is a weighted composite of three normalized dimensions:
```
Final Score = (0.60 × Semantic Depth) + (0.25 × Career Velocity) + (0.15 × Profile Freshness)
```

#### Semantic Depth Score (60%)
- The SLM swarm's `Best Chunk Alignment Score` for the candidate
- **Min-max normalized** across the retrieved pool to use the full `[0, 100]` range
- **Guardrail: Keyword Stuffer** — if a candidate has 20+ skills but <200 words in their resume, semantic score is penalized by 15%
- **Guardrail: Duplicate Content** — if >40% of resume content is repeated chunks, penalized proportionally

#### Career Velocity Score (25%)
- Measures how fast the candidate climbed the seniority ladder
- Computed as: `max_seniority_level_reached / total_career_years`
- Seniority levels: Intern(0) ➔ Junior(1) ➔ Mid(2) ➔ Senior(3) ➔ Lead(4) ➔ Principal(5) ➔ Director(6)
- Candidates with <2 jobs in their history receive a baseline score of `0.3`

#### Profile Freshness Score (15%)
- Days since last profile update: `max(0, 1 - days_since_update / 365)`
- +10% bonus for profiles updated in the last 7 days
- Null `last_active` ➔ freshness score of `0.2`

#### Guardrail: Seniority Mismatch
If the JD targets Senior/Lead/Principal and the candidate has <2 years of experience, their final score is **capped at 65/100**. This prevents junior candidates from gaming semantic similarity to appear in senior shortlists.

---

## 🕸️ The Swarm Matrix — Technical Deep Dive

The name "Job Recruitment" comes from the core architectural concept: a *swarm* of specialized models, each a node in a routing matrix, evaluating candidates in their domain of expertise.

### Why Not One Big Model?
- General-purpose embedding models (e.g., `all-MiniLM-L6-v2`) are trained on diverse corpora — they don't know that "bioavailability" is a medical term or that "flash attention" means GPU kernel efficiency.
- Domain-specialized models (BioBERT, FinBERT, CodeBERT) were fine-tuned on millions of domain-specific documents — their token representations capture deep domain semantics.
- Using the *right* model for each sector means a nurse's resume scores correctly against a healthcare JD, not against a generic BERT.

### Why MAX() Instead of MEAN() for Fragment Aggregation?
Resumes are not uniformly relevant — a single paragraph about "optimizing flash attention CUDA kernels" is more signal-dense than three paragraphs about general Python experience.

`MAX()` identifies the **peak semantic alignment moment** in the candidate's document — the fragment where they were most relevant to the JD — rather than diluting it with weaker sections. This surfaces "hidden gems": candidates with one exceptional section buried in an otherwise average resume.

---

## 🛠️ Recruiter Copilot Toolkit

Job Recruitment provides an advanced **Recruiter Copilot** directly on the dashboard to help hiring managers deeply vet and engage top-tier talent.

### Tailored Phone Screen Guide Generator
**File:** [`talent_radar/question_generator.py`](talent_radar/question_generator.py) → `TechnicalQuestionGenerator`

Generates exactly 5 personalized, deep-dive interview questions tailored to the candidate's specific profile gaps (missing skills), seniority, and background challenges.
- **Verification-focused**: If they claim a specific technology, it generates deep probe questions to see if they've worked with it or are just keyword stuffing.
- **Recruiter Answer Sheets**: Supplies expected concepts, terminology, and answers for the recruiter to look out for during the screen.
- **Robust Offline Fallback**: Features a high-fidelity local generator to maintain uptime when rate-limited.

### Side-by-Side Candidate Comparator
**File:** [`talent_radar/comparison.py`](talent_radar/comparison.py) → `CandidateComparator`

Compares 2 or 3 selected candidates in an in-depth comparative matrix using **Structured JSON Output** via Gemini 2.5 Flash.
- **Strengths Synthesis**: A side-by-side analysis of key technical/experience competencies.
- **Trade-off Analysis**: Compares specialized technical depth versus broader engineering capacity, or trajectory velocity versus stability.
- **Actionable Recommendation**: High-conviction hire recommendation specifying who to prioritize and why.

### Personalized Outreach Email Generator
**File:** [`talent_radar/app.py`](talent_radar/app.py) → `/api/candidates/{candidate_id}/outreach`

Writes customized recruitment outreach emails leveraging candidate experience, matches, and targeted stack skills.
- **Multi-Tone**: Supports both **Professional** and **Casual** templates.
- **Placeholders Eliminated**: Auto-generates clean, recruiter-ready emails ready to send.

### Similar Talent Search
**File:** [`talent_radar/app.py`](talent_radar/app.py) → `/api/candidates/{candidate_id}/similar`

Uses the selected candidate's profile skills and role title to run a synthetic JD search over the pool, enabling recruiters to quickly discover adjacent talent.

---

## 📥 Resume Ingestion Pipeline

The Job Recruitment resume pipeline handles single PDF uploads, bulk ZIP parsing, and OCR processing of scanned images.

```
                  ┌──────────────────────────────┐
                  │      UPLOADED RESUMES        │
                  └──────────────┬───────────────┘
                                 │
                 Is it a digital-native PDF?
                 ┌───────────────┴───────────────┐
             YES │                            NO │
                 ▼                               ▼
       ┌──────────────────┐            ┌──────────────────┐
       │   pypdf Reader   │            │  Qwen2-VL Model  │
       │(fast extraction) │            │ (Visual OCR text)│
       └─────────┬────────┘            └─────────┬────────┘
                 │ Empty                         │ Fail
                 ▼                               ▼
       ┌──────────────────┐            ┌──────────────────┐
       │ pdfplumber Text  │            │     EasyOCR      │
       │ (layout recovery)│            │(column-aware text)│
       └─────────┬────────┘            └─────────┬────────┘
                 │                               │
                 └───────────────┬───────────────┘
                                 ▼
                    ┌────────────────────────┐
                    │  Gemini Resume Parser  │
                    │   (Structured JSON)    │
                    └────────────┬───────────┘
                                 ▼
                    ┌────────────────────────┐
                    │     Deduplication      │
                    │   (atomic merge/id)    │
                    └────────────┬───────────┘
                                 ▼
                    ┌────────────────────────┐
                    │    candidates.json     │
                    └────────────────────────┘
```

### Single PDF Ingestion Cascade
**File:** [`talent_radar/llm_parser.py`](talent_radar/llm_parser.py) → `GeminiResumeParser`

Processes PDF files through a multi-tiered fallback pipeline:
1. `pypdf` extraction (sub-second, digital-native).
2. `pdfplumber` fallback if empty (recovers column layout).
3. **Visual PDF Parsing** via Gemini or the local OCR engine if text is missing.
4. Schema extraction via **Gemini 2.5 Flash** to clean data, build unified career histories, and list skills.

### Bulk ZIP Upload & Threaded Smart Ingest
**File:** [`talent_radar/smart_ingest.py`](talent_radar/smart_ingest.py)

Ingests ZIP archives containing thousands of resume PDFs with high-performance concurrent scheduling:
- **Batched API Slicing**: Bundles **15 resumes in a single Gemini API call** using unique delimiters (`===== RESUME N =====`). This reduces network overhead and API call limits by 15×.
- **Concurrency**: Schedules parallel extraction using a `ThreadPoolExecutor` of **5 concurrent workers**.
- **Cascade Fallbacks**: If a batch fails, individual resumes are automatically isolated, extracted, and processed.

### Local OCR Engine with Qwen2-VL & EasyOCR
**File:** [`talent_radar/ocr_engine.py`](talent_radar/ocr_engine.py) → `LocalOCREngine`

A thread-safe, memory-cached singleton engine implementing two OCR modules for scanned PDFs:
1. **Qwen2-VL-2B-Instruct**: A state-of-the-art vision-language model.
   - Pages are rendered to PNGs at 150 DPI.
   - Outputs highly accurate Markdown maintaining layout structure.
   - Automatically utilizes `float16` and scaled SDPA attention on GPU, falling back to `float32` on CPU.
2. **EasyOCR Fallback**: 
   - A layout-aware text detection algorithm.
   - Calculates **crossing-box ratio heuristics** to separate 2-column templates and sorts text block streams vertically to avoid text scrambling.

### Real-Time SSE Progress Tracking
**File:** [`talent_radar/app.py`](talent_radar/app.py) → `/api/upload/progress/{job_id}`

Uses **Server-Sent Events (SSE)** to stream processing metrics directly to the dashboard every 2 seconds:
```json
{
  "parsed": 12,
  "total": 45,
  "skipped": 2,
  "failed": 1,
  "percent": 26.6,
  "status": "processing"
}
```

### Atomic Checkpointing & Integrity
To prevent data loss or file corruption in high-volume uploads, Smart Ingest uses **atomic file writes**. It writes new data to a temporary file (`tempfile.mkstemp()`) and completes the replacement in a single operation (`os.replace()`), ensuring the active database is never corrupted.

---

## 📊 Scoring Formula & Guardrails

```
Final Score (0–100) = 
    60% × Semantic Depth Score    (SLM cosine similarity, normalized)
  + 25% × Career Velocity Score   (seniority progression rate, normalized)
  + 15% × Profile Freshness Score (days since last active, decayed)

Guardrails Applied Before Final Score:
  ├── Keyword Stuffer Penalty: 20+ skills + <200 words → −15% semantic score
  ├── Duplicate Content Penalty: >40% repeated chunks → proportional semantic reduction
  └── Seniority Cap: Junior candidate for Senior+ JD → cap at 65/100

Tie-Breaking:
  When final scores are within ±0.5, higher profile freshness wins.
```

---

## 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Recruiter Dashboard (serves `index.html`) |
| `POST` | `/api/rank` | Run 4-phase ranking pipeline for a JD |
| `GET` | `/api/sectors` | List of 12 sector tokens and labels |
| `GET` | `/api/templates` | Sample JD templates from `sample_jds/` |
| `GET` | `/api/statistics` | Candidate pool demographics and freshness |
| `POST` | `/api/upload` | Upload PDF (single), ZIP (bulk), or JSON dataset |
| `DELETE` | `/api/upload` | Clear custom database (reset pool to empty slate) |
| `GET` | `/api/upload/progress/{job_id}`| SSE progress stream for smart bulk zip ingest |
| `GET` | `/api/settings` | Read current API key configuration (masked) |
| `POST` | `/api/settings` | Update system API keys and OCR backends |
| `POST` | `/api/candidates/{id}/questions`| Generate 5 customized interview questions |
| `POST` | `/api/candidates/{id}/outreach` | Generate personalized outreach emails |
| `POST` | `/api/candidates/{id}/similar`  | Retrieve candidates with similar profiles |
| `POST` | `/api/compare` | Compare 2 or 3 candidates side-by-side |
| `POST` | `/api/export` | Download search brief (Markdown or CSV format) |

### `/api/compare` Request Body
```json
{
  "candidate_ids": ["cand_001", "cand_002"],
  "job_description": "Senior ML Engineer with Triton/CUDA...",
  "sector": "TECH"
}
```

### `/api/candidates/{id}/questions` Response
```json
{
  "candidate_name": "Jane Doe",
  "target_role": "Senior ML Engineer",
  "questions": [
    {
      "question": "You optimized GPU flash attention loops. Can you detail how you addressed block-level memory bank conflicts on CUDA shared memory?",
      "expected_answer": "Recruiter should look for: Shared memory padding, coalescing, warp shuffle operations, and latency reduction metrics.",
      "rationale": "Verifies their low-level CUDA optimization claim directly while screening for superficial keyword matching."
    }
  ]
}
```

---

## 🖥️ Frontend Recruiter Dashboard

Served directly at `http://127.0.0.1:8000/`, the dashboard is a **responsive single-page recruiter console** designed with a premium, glowing glassmorphism dark aesthetic:

1. **Recruiter Search & Template Library**: Select and load pre-curated industry templates (React, Backend, SRE, ML, legal) from `sample_jds/`.
2. **Interactive Rankings Grid**: Displays sorted candidates with animated percentage meters, matched/missing skill chips, starred shortlist pinning, and guardrail alerts (stuffers, seniority caps).
3. **Advanced Recruiter Copilot Panel**: A dynamic drawer that slides open to show:
   - **Phone Screen Guide**: Tailored interview questions with expectation checklists.
   - **Outreach Email Drafts**: Custom professional and casual email copy.
   - **Similar Candidates**: Sourced adjacent profiles matching the current candidate.
4. **Side-by-Side Comparison Drawer**: Selecting candidates via checkboxes displays a clean comparative grid detailing trade-offs, syntheses, and hiring suggestions.
5. **Statistics & Demographics View**: Dynamic bar, pie, and doughnut charts tracking candidate experience distribution, roles, and freshness using **Chart.js**.
6. **SSE Upload Progress**: Real-time progress bars for bulk resume uploads.
7. **Systems Settings Modal**: In-dashboard key updates (Gemini, HF, OpenAI) and OCR backend swapping (Qwen2-VL vs EasyOCR).

---

## ⚙️ Technical Choices & Rationale

| Choice | Rationale |
|---|---|
| **FastAPI** | Async-capable, Pydantic-native, excellent performance for ML-serving |
| **Gemini 2.5 Flash** | Fast, high-intelligence gateway, supports strict JSON output schemas via `response_schema` |
| **Sector-specialized SLMs** | Domain-specific pre-training makes cosine similarity semantically meaningful |
| **MAX() chunk aggregation** | Surfaces peak relevance signal; prevents average-dilution of highly relevant candidates |
| **Anisotropy correction** | Mean-centering restores meaningful cosine distances across BERT embeddings |
| **Hybrid keyword pre-filter** | Reduces SLM compute cost by 5× while preserving high-signal candidates |
| **ThreadPoolExecutor & Delimiters** | Speeds up bulk ZIP ingestion by batching 15 resumes in concurrent workers |
| **Atomic checkpoint (os.replace)**| Prevents database file corruption in high-volume uploads |
| **SSE (Server-Sent Events)** | Light HTTP-native real-time streaming, avoiding heavy WebSocket overheads |
| **Qwen2-VL-2B-Instruct** | Outstanding local visual OCR capabilities for scanned/non-native PDFs |
| **EasyOCR layout heuristics** | Fallback sorting to maintain single/dual-column vertical reading orders |
| **BriefExporter & Comparator** | Automatically handles structured Markdown reports and side-by-side comparative matrices |
| **Loaded Model Singletons** | Prevents reloading heavy ML weights (3GB+) to minimize latency and memory spikes |
| **torch.compile on GPU only** | Prevents JIT triton hangs on Windows / CPU environments |

---

## 📁 Project Structure

```
keen-hawking/
├── .env                          # API keys and model config
├── Dockerfile                    # Hugging Face Spaces deployment
├── requirements.txt              # Python dependencies
├── talent_radar_output.csv       # Sample output from a CLI run
├── talent_radar_output.json      # Sample output (JSON format)
│
└── talent_radar/
    ├── app.py                    # FastAPI web server, all REST API endpoints
    ├── main.py                   # CLI entrypoint (python -m talent_radar.main)
    ├── pipeline.py               # E2E Orchestrator (TalentRadarPipeline)
    ├── matrix_pipeline.py        # Swarm Matrix: GeminiGateway, SwarmEvaluator, JobRecruitmentRanker
    ├── query_expand.py           # Query Explosion: LLM + rule-based semantic expansion
    ├── scorer.py                 # CandidateScorer: composite momentum formula + guardrails
    ├── smart_ingest.py           # Bulk ZIP ingestion: threading, batching, checkpointing
    ├── llm_parser.py             # GeminiResumeParser: single, batch, and visual PDF parsing
    ├── ocr_engine.py             # LocalOCREngine: Qwen2-VL + EasyOCR with layout heuristics
    ├── comparison.py             # Recruiter Copilot: Side-by-side candidate comparison
    ├── question_generator.py     # Recruiter Copilot: Personalized 5-question phone screens
    ├── exporter.py               # Recruiter Copilot: Premium Markdown brief and CSV exports
    ├── candidates.json           # Active candidate pool (JSON array)
    ├── candidates_dataset.py     # Synthetic dataset generator (for testing)
    ├── check_candidates.py       # Diagnostic utility to inspect pool
    ├── index.html                # Recruiter dashboard (Single-Page SPA)
    ├── sample_jds/               # Sample job description text files
    │
    ├── output.py                 # [LEGACY] CLI CSV/JSON formatter
    ├── ingest.py                 # [DEPRECATED] Legacy ChromaDB vector indexer
    ├── retriever.py              # [DEPRECATED] Legacy ChromaDB vector retriever
    ├── reranker.py               # [DEPRECATED] Legacy Cross-Encoder re-ranking
    └── chroma_db/                # [DEPRECATED] Legacy ChromaDB persistence directory
```

---

## 🚀 Setup & Installation

### Prerequisites

- Python 3.10+
- A **Gemini API key** (required for structured parsing and query refinement)
- A GPU with CUDA support (optional, but highly recommended for fast local OCR / SLM Swarms)

### 1. Clone & Install

```bash
git clone <repo-url>
cd keen-hawking

pip install torch  # Install PyTorch (CPU or GPU version as needed)
pip install -r requirements.txt
```

### 2. Configure API Keys

Create or edit `.env` in the project root:

```env
GEMINI_API_KEY="your-gemini-api-key-here"
GEMINI_MODEL=gemini-2.5-flash
LOCAL_OCR_BACKEND=qwen2.5-vl
```

### 3. Run the Server

```bash
uvicorn talent_radar.app:app --host 127.0.0.1 --port 8000 --reload
```

Open `http://127.0.0.1:8000` in your browser to access the premium recruiter dashboard.

---

## 🐳 Docker Deployment (Hugging Face Spaces)

The included `Dockerfile` is configured for **Hugging Face Spaces** (port 7860, UID 1000 non-root user):

```bash
docker build -t job-recruitment-ai .
docker run -p 7860:7860 -e GEMINI_API_KEY=your-key job-recruitment-ai
```

---

## 🔑 Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | *(required)* | Google Gemini API key for parsing, comparison, and gateway |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model for candidate resume parsing |
| `LOCAL_OCR_BACKEND` | `qwen2.5-vl` | OCR backend: `qwen2.5-vl` or `easyocr` |
| `LOCAL_OCR_QWEN_WORKERS` | `1` | Concurrent Qwen page workers (set >1 only on large GPUs) |
| `LOCAL_OCR_EASYOCR_WORKERS` | `4` | Concurrent EasyOCR page workers |
| `LOCAL_OCR_QWEN_MIN_PATCHES` | `256` | Qwen minimum pixel patches per page |
| `LOCAL_OCR_QWEN_MAX_PATCHES` | `512` (CPU) / `1280` (GPU) | Qwen maximum pixel patches per page |
| `TORCH_CPU_THREADS` | `4` | PyTorch thread pool size on CPU |
| `HF_TOKEN` | *(optional)* | Hugging Face token for private model access |
| `OPENAI_API_KEY` | *(optional)* | OpenAI key for query expansion (uses GPT-4o-mini if set) |

---

## 🌐 Sector Coverage

Job Recruitment routes to a specialized model for each of the 12 industry macro sectors:

| Token | Label | Specialized Model |
|---|---|---|
| `TECH` | Technology & Software | `microsoft/codebert-base` |
| `FIN` | Finance & Banking | `ProsusAI/finbert` |
| `HEALTH` | Healthcare & Life Sciences | `dmis-lab/biobert-base-cased-v1.2` |
| `LEGAL` | Legal & Compliance | `nlpaueb/legal-bert-small-uncased` |
| `REAL` | Real Estate | `llmware/industry-bert-asset-management-v0.1` |
| `MANU` | Manufacturing & Engineering | `cea-list-ia/ManufactuBERT` |
| `COMM` | Commerce & Retail | `nlptown/bert-base-multilingual-uncased-sentiment` |
| `LOGI` | Logistics & Supply Chain | `inovex/multi2convai-logistics-en-bert` |
| `MEDIA` | Media & Creative | `cardiffnlp/twitter-roberta-base` |
| `ENERGY` | Energy & Utilities | `Master-AI-Lab/EnergyBERT` |
| `EDU` | Education | `vasugoel/K-12BERT` |
| `GOV` | Government & Public Sector | `ESGBERT/GovRoBERTa-governance` |

---

## 📜 License

This project is open-source. Built for the Job Recruitment AI Talent Radar challenge.

---

*"Don't rank candidates by who matches the most keywords. Rank them by who understands the domain deepest."*
