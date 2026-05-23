"""
Smart Ingest Engine — High-Volume ZIP Resume Processing

Handles massive ZIP archives (2,484+ PDF resumes) with:
  1. Concurrent ThreadPoolExecutor workers (default 5)
  2. Smart batching (15 resumes per Gemini API call)
  3. Incremental checkpointing to candidates.json after every batch
  4. source_file deduplication to skip already-parsed files on re-runs
  5. Automatic batch → individual fallback on partial failures
"""

import io
import json
import os
import threading
import time
import uuid
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import zipfile
from pypdf import PdfReader

from talent_radar.llm_parser import GeminiResumeParser

# Defaults (could be driven from .env in the future)
DEFAULT_BATCH_SIZE = 15
DEFAULT_MAX_WORKERS = 5

base_dir = Path(__file__).parent


# -------------------------------------------------------------------- #
#  Job State                                                            #
# -------------------------------------------------------------------- #
@dataclass
class SmartIngestJob:
    """Thread-safe state tracker for a single bulk import job."""

    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    total_files: int = 0
    skipped_files: int = 0
    parsed_count: int = 0
    failed_count: int = 0
    failed_files: list = field(default_factory=list)
    status: str = "queued"          # queued | running | done | error
    error_message: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def percent(self) -> int:
        processable = self.total_files - self.skipped_files
        if processable <= 0:
            return 100
        return min(int((self.parsed_count + self.failed_count) / processable * 100), 100)

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "total_files": self.total_files,
            "skipped_files": self.skipped_files,
            "parsed_count": self.parsed_count,
            "failed_count": self.failed_count,
            "failed_files": self.failed_files[:20],   # cap list length in responses
            "percent": self.percent,
            "status": self.status,
            "error_message": self.error_message,
        }


# -------------------------------------------------------------------- #
#  PDF Extraction                                                       #
# -------------------------------------------------------------------- #
def _extract_pdfs_from_zip(zip_bytes: bytes) -> list[tuple[str, str, bytes]]:
    """
    Opens a ZIP archive from raw bytes and extracts text from every PDF
    inside it (in-memory only, no disk writes during extraction).

    Returns:
        List of (filename, raw_text, pdf_bytes) tuples. Skips macOS metadata files.
    """
    pdf_items = []
    zip_buffer = io.BytesIO(zip_bytes)

    with zipfile.ZipFile(zip_buffer, "r") as z:
        all_names = z.namelist()
        print(f"[Smart Ingest Debug] ZIP contains {len(all_names)} total files/folders.")
        print(f"[Smart Ingest Debug] Sample files in ZIP: {all_names[:20]}")
        
        pdf_names = []
        for name in all_names:
            # Normalize backslashes to forward slashes for robust path checking
            norm_name = name.replace('\\', '/')
            base_name = norm_name.split('/')[-1]
            
            if (
                norm_name.lower().endswith(".pdf")
                and not base_name.startswith("._")
                and not norm_name.startswith("__MACOSX")
            ):
                pdf_names.append(name)
                
        print(f"[Smart Ingest Debug] Found {len(pdf_names)} matching PDF file paths in ZIP archive.")

        for pdf_name in pdf_names:
            try:
                pdf_bytes = z.read(pdf_name)
                
                # 1. Try pypdf text extraction
                extracted_text = ""
                try:
                    reader = PdfReader(io.BytesIO(pdf_bytes))
                    for page in reader.pages:
                        extracted_text += page.extract_text() or ""
                    extracted_text = extracted_text.strip()
                except Exception as e:
                    print(f"[Smart Ingest Extractor] pypdf failed on '{pdf_name}': {e}")
                    extracted_text = ""
                    
                # 2. Try pdfplumber fallback if text is empty
                if not extracted_text:
                    try:
                        import pdfplumber
                        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                            for page in pdf.pages:
                                extracted_text += page.extract_text() or ""
                        extracted_text = extracted_text.strip()
                        if extracted_text:
                            print(f"[Smart Ingest Extractor] pdfplumber successfully recovered text for '{pdf_name}'")
                    except Exception as e:
                        print(f"[Smart Ingest Extractor] pdfplumber fallback failed on '{pdf_name}': {e}")

                if not extracted_text:
                    print(f"[Smart Ingest Extractor] Scanned/Non-Text PDF detected (Will parse visually via Gemini): '{pdf_name}'")

                pdf_items.append((pdf_name, extracted_text, pdf_bytes))
            except Exception as e:
                print(f"[Skip] Error reading PDF '{pdf_name}': {e}")
                continue

    return pdf_items


# -------------------------------------------------------------------- #
#  Batching                                                             #
# -------------------------------------------------------------------- #
def _build_batches(
    pdf_items: list[tuple[str, str, bytes]], batch_size: int = DEFAULT_BATCH_SIZE
