import os
import sys
import io
import json
import time
import requests
from dotenv import load_dotenv
from pathlib import Path

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# Load env variables from the workspace root folder
base_dir = Path(__file__).parent
env_path = base_dir.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Status codes that are safe to retry (transient server-side or rate-limit errors)
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

def validate_name(name: str, title: str = "") -> bool:
    """
    Returns True if the name is a valid candidate name, 
    False if it is a placeholder or job title.
    """
    if not name or not isinstance(name, str):
        return False
        
    name_clean = name.strip().strip('"').strip("'").strip()
    name_lower = name_clean.lower()
    
    # Generic missing names
    generic_names = {
        "not specified", "unknown", "n/a", "none", "unknown candidate", "null", 
        "not specified.", "not-specified", "unspecified", "name", "candidate name", "n.a."
    }
    if name_lower in generic_names:
        return False
        
    if title and name_lower == title.lower().strip():
        return False
        
    job_indicators = {
        "director", "engineer", "developer", "manager", "officer", "vp", "vice president", "chief", "cto", 
        "ceo", "cfo", "lead", "architect", "analyst", "administrator", "specialist", "consultant", "head",
        "intern", "designer", "senior", "junior", "associate", "staff", "principal"
    }
    
    # Clean the name to see if it is mostly job titles or delimiters
    words = [w.strip() for w in name_lower.replace("/", " ").replace("|", " ").replace("-", " ").split() if w.strip()]
    if not words:
        return False
        
    if len(words) == 1 and words[0] in job_indicators:
        return False
        
    # Check if more than 50% of the words are job titles
    job_count = sum(1 for w in words if w in job_indicators or any(ind in w for ind in job_indicators))
    if job_count / len(words) >= 0.5:
        return False
        
    # Check if name contains symbols like / or | which usually indicates a combined job title field
    if "/" in name_clean or "|" in name_clean:
        return False
        
    return True

def extract_name_from_filename(filename: str) -> str:
    """
    Attempts to extract a clean candidate name from the filename.
    e.g., 'John_Doe_Resume.pdf' -> 'John Doe'
          'cv_jane_smith_2026.docx' -> 'Jane Smith'
    """
    if not filename:
        return ""
    # Get stem
    stem = Path(filename).stem
    # Replace separators with spaces
    stem_clean = stem.replace("_", " ").replace("-", " ").replace(".", " ")
    
    # Split into words
    words = [w for w in stem_clean.split() if w]
    if not words:
        return ""
        
    # Filter out common CV/Resume words and job titles
    ignore_words = {
        "resume", "cv", "pdf", "docx", "txt", "portfolio", "application", "job", "candidate", 
        "profile", "copy", "latest", "updated", "final", "2023", "2024", "2025", "2026", "2027",
        "hiring", "apply", "work", "doc", "eng", "dev", "tech", "manager", "director", "engineer",
        "developer", "officer", "vp", "vice president", "chief", "cto", "ceo", "cfo", "lead",
        "architect", "analyst", "administrator", "specialist", "consultant", "head", "intern",
        "designer", "senior", "junior", "associate", "staff", "principal", "hr", "recruitment", "talent"
    }
    
    filtered_words = []
    for w in words:
        w_lower = w.lower()
        if w_lower not in ignore_words and not w_lower.isdigit():
            # Keep words that are alphabetic (or have letters)
            if any(c.isalpha() for c in w):
                filtered_words.append(w.capitalize())
                
    # If we have 2 or more words, it's likely a first and last name (e.g., "John Doe")
    if 2 <= len(filtered_words) <= 4:
        return " ".join(filtered_words)
    # If 1 word, it could be a first name (e.g. "John"), but let's check
    if len(filtered_words) == 1 and len(filtered_words[0]) > 2:
        return filtered_words[0]
        
    return ""

class GeminiResumeParser:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set in the environment or .env file.")
        self._url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"

    # ------------------------------------------------------------------ #
    #  Shared candidate schema (single object)                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _candidate_schema():
        """Returns the Gemini REST JSON schema for a single candidate object."""
        return {
            "type": "OBJECT",
            "properties": {
                "name": {
                    "type": "STRING",
                    "description": "Candidate's full name"
                },
                "current_title": {
                    "type": "STRING",
                    "description": "Candidate's current professional title or role (e.g. Senior React Developer, Financial Analyst, Legal Counsel, Chief Medical Officer)"
                },
                "years_experience": {
                    "type": "NUMBER",
                    "description": "Total number of years of experience as a floating point number (e.g. 6.5)"
                },
                "career_history": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "title": { "type": "STRING", "description": "Job title" },
                            "company": { "type": "STRING", "description": "Company name" },
                            "start_date": { "type": "STRING", "description": "Start date in YYYY-MM-DD or YYYY-MM format" },
                            "end_date": { "type": "STRING", "description": "End date in YYYY-MM-DD, YYYY-MM format, or 'Present'" }
                        },
                        "required": ["title", "company", "start_date"]
                    },
                    "description": "Professional experience history"
                },
                "skills_listed": {
                    "type": "ARRAY",
                    "items": {
                        "type": "STRING"
                    },
                    "description": "List of professional and domain-specific skills mentioned or demonstrated in the resume (e.g. React, Financial Modeling, Litigation, Patient Care)"
                },
                "last_active": {
                    "type": "STRING",
                    "description": "Approximate date of last resume update or activity in YYYY-MM-DD format. If no date can be extracted from the PDF, set this field to null."
                },
                "education": {
                    "type": "STRING",
                    "description": "University education degree, major, and graduation details"
                },
                "location": {
                    "type": "STRING",
                    "description": "City and State / Country of candidate (e.g. San Francisco, CA or Remote)"
                }
            },
            "required": ["name", "current_title", "years_experience", "career_history", "skills_listed", "education", "location"]
        }

    # ------------------------------------------------------------------ #
    #  Resilient HTTP POST with exponential backoff                       #
    # ------------------------------------------------------------------ #
    def _call_gemini(self, payload: dict) -> dict:
        """
        POST to the Gemini generateContent endpoint with resilient retry logic.
        Retries on transient HTTP errors (429, 500, 502, 503, 504) and
        network-level failures (ConnectionError, Timeout) up to 6 attempts
        with exponential backoff (2s → 4s → 8s → 16s → 32s → 64s).
        """
        headers = {"Content-Type": "application/json"}
        max_retries = 6
        base_delay = 2.0

        last_error = None
        for attempt in range(max_retries):
            try:
                response = requests.post(self._url, headers=headers, json=payload, timeout=120)

                if response.status_code == 200:
                    return response.json()

                if response.status_code in RETRYABLE_STATUS_CODES:
                    import random
                    jitter = random.uniform(0.8, 1.2)
                    delay = (base_delay * (2 ** attempt)) * jitter
                    print(f"[Retry] Gemini API returned {response.status_code}. "
                          f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(delay)
                    last_error = RuntimeError(
                        f"Gemini API returned error {response.status_code}: {response.text[:300]}"
                    )
                    continue

                # Non-retryable error (e.g. 400 Bad Request, 401 Unauthorized)
                raise RuntimeError(
                    f"Gemini API returned non-retryable error {response.status_code}: {response.text[:500]}"
                )

            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as net_err:
                import random
                jitter = random.uniform(0.8, 1.2)
                delay = (base_delay * (2 ** attempt)) * jitter
                print(f"[Retry] Network error: {type(net_err).__name__}. "
                      f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(delay)
                last_error = net_err
                continue

        raise RuntimeError(
            f"Gemini API request failed after {max_retries} retries. Last error: {last_error}"
        )

    # ------------------------------------------------------------------ #
    #  Single resume parsing (original interface, preserved)              #
    # ------------------------------------------------------------------ #
    @staticmethod
    def strip_json_fences(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text

    def parse_resume(self, raw_text: str, filename: str = "") -> dict:
        """
        Sends the raw extracted text of a resume to the Gemini API and retrieves a structured JSON candidate profile.
        """
        system_instruction = (
            "You are an expert recruiter and resume parser. "
            "Analyze the provided raw resume text and extract candidate details into the specified JSON structure. "
            "Determine the total years_experience carefully as a float. "
            "Format dates in YYYY-MM-DD or YYYY-MM. If a job is current or ongoing, the end_date MUST be 'Present'. "
            "Extract professional and domain-specific skills accurately and list them in the skills_listed array. "
            "Crucial: Pay close attention to extracting the candidate's ACTUAL personal full name, typically located at "
            "the very top of the resume in a large font. Do NOT populate the name field with job titles, headers, 'Unknown', 'Not specified', or 'N/A'."
        )

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": f"Raw Resume Text:\n{raw_text}"}
                    ]
                }
            ],
            "systemInstruction": {
                "parts": [
                    {"text": system_instruction}
                ]
            },
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": self._candidate_schema(),
                "temperature": 0.1
            }
        }

