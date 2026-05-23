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

