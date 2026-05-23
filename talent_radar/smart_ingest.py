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
) -> list[list[tuple[str, str, bytes]]]:
    """Splits the list of (filename, raw_text, pdf_bytes) into chunks of batch_size."""
    return [
        pdf_items[i : i + batch_size]
        for i in range(0, len(pdf_items), batch_size)
    ]


# -------------------------------------------------------------------- #
#  Skip Logic (Checkpoint Dedup)                                        #
# -------------------------------------------------------------------- #
def _skip_already_parsed(
    pdf_items: list[tuple[str, str, bytes]],
    existing_candidates: list[dict],
) -> list[tuple[str, str, bytes]]:
    """
    Filters out PDFs whose filenames are already recorded as source_file
    in the existing candidate pool. Returns only the un-parsed items.
    """
    parsed_sources = {
        c.get("source_file")
        for c in existing_candidates
        if c.get("source_file")
    }

    remaining = [
        (fname, text, pbytes)
        for fname, text, pbytes in pdf_items
        if fname not in parsed_sources
    ]

    skipped = len(pdf_items) - len(remaining)
    if skipped > 0:
        print(f"[Checkpoint] Skipping {skipped} already-parsed files.")

    return remaining


# -------------------------------------------------------------------- #
#  Atomic Checkpoint                                                    #
# -------------------------------------------------------------------- #
def _checkpoint(candidates: list[dict], path: Path, lock: threading.Lock):
    """
    Thread-safely writes the full candidate list to candidates.json using
    a temp-file + os.replace() for a single atomic operation.
    """
    with lock:
        # Write to a temp file in the same directory so rename is atomic
        fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent), suffix=".tmp", prefix=".candidates_"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(candidates, f, indent=2, ensure_ascii=False)
            # os.replace() is atomic on both Windows and POSIX —
            # single syscall, no gap where the file is missing.
            os.replace(tmp_path, str(path))
            print(f"[Checkpoint] Saved {len(candidates)} candidates to {path.name}")
        except Exception:
            # Clean up temp file on error
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise


# -------------------------------------------------------------------- #
#  Batch Processing (with fallback)                                     #
# -------------------------------------------------------------------- #
def _process_batch(
    batch: list[tuple[str, str, bytes]],
    parser: GeminiResumeParser,
    job: SmartIngestJob,
    existing_candidates: list[dict],
    existing_ids: set[str],
    candidates_path: Path,
) -> list[dict]:
    """
    Parses a batch of resumes via Gemini batch API. On batch failure or for
    scanned/non-text PDFs, automatically degrades to individual visual parsing.

    Returns:
        List of newly created candidate records from this batch.
    """
    new_records = []
    text_resumes = []
    scanned_resumes = []

    # Separate text-based resumes (batched) and scanned resumes (processed individually)
    for filename, raw_text, pdf_bytes in batch:
        if raw_text:
            text_resumes.append((filename, raw_text, pdf_bytes))
        else:
            scanned_resumes.append((filename, raw_text, pdf_bytes))

    # 1. Process text resumes in batch
    if text_resumes:
        batch_input = [(fname, txt) for fname, txt, _ in text_resumes]
        try:
            # Attempt batch parsing
            parsed_array = parser.parse_resume_batch(batch_input)
            matched_indices = set()

            for idx, parsed in enumerate(parsed_array):
                if idx >= len(text_resumes):
                    break
                filename, raw_text, pdf_bytes = text_resumes[idx]
                record = _build_candidate_record(
                    parsed, raw_text, filename, existing_candidates, existing_ids, job
                )
                if record is not None:
                    new_records.append(record)
                matched_indices.add(idx)
                with job._lock:
                    job.parsed_count += 1

            # If Gemini returned fewer items than the batch size, process individually
            if len(matched_indices) < len(text_resumes):
                missing = [
                    (i, text_resumes[i]) for i in range(len(text_resumes))
                    if i not in matched_indices
                ]
                print(f"[Partial] Batch returned {len(matched_indices)}/{len(text_resumes)}. "
                      f"Individually parsing {len(missing)} missing resumes...")
                for i, (filename, raw_text, pdf_bytes) in missing:
                    try:
                        parsed = parser.parse_resume(raw_text, filename)
                        record = _build_candidate_record(
                            parsed, raw_text, filename, existing_candidates, existing_ids, job
                        )
                        if record is not None:
                            new_records.append(record)
                        with job._lock:
                            job.parsed_count += 1
                    except Exception as ind_err:
                        try:
                            print(f"[Fallback] Text parsing failed for '{filename}'. Trying visual PDF parsing...")
                            parsed = parser.parse_resume_pdf(pdf_bytes, filename)
                            record = _build_candidate_record(
                                parsed, parsed.get("resume_text", ""), filename, existing_candidates, existing_ids, job
                            )
                            if record is not None:
                                new_records.append(record)
                            with job._lock:
                                job.parsed_count += 1
                        except Exception as vis_err:
                            print(f"[Error] Failed to parse '{filename}': {vis_err}")
                            with job._lock:
                                job.failed_count += 1
                                job.failed_files.append(filename)

        except Exception as batch_err:
            print(f"[Fallback] Batch of {len(text_resumes)} failed: {batch_err}. "
                  f"Degrading to individual text/visual parsing...")

            # Full fallback for text resumes: parse each resume individually
            for filename, raw_text, pdf_bytes in text_resumes:
                try:
                    parsed = parser.parse_resume(raw_text, filename)
                    record = _build_candidate_record(
                        parsed, raw_text, filename, existing_candidates, existing_ids, job
                    )
                    if record is not None:
                        new_records.append(record)
                    with job._lock:
                        job.parsed_count += 1
                except Exception as ind_err:
                    try:
                        print(f"[Fallback] Text parsing failed for '{filename}'. Trying visual PDF parsing...")
                        parsed = parser.parse_resume_pdf(pdf_bytes, filename)
                        record = _build_candidate_record(
                            parsed, parsed.get("resume_text", ""), filename, existing_candidates, existing_ids, job
                        )
                        if record is not None:
                            new_records.append(record)
                        with job._lock:
                            job.parsed_count += 1
                    except Exception as vis_err:
                        print(f"[Error] Failed to parse '{filename}': {vis_err}")
                        with job._lock:
                            job.failed_count += 1
                            job.failed_files.append(filename)

    # 2. Process scanned resumes individually using native visual fallback
    for filename, raw_text, pdf_bytes in scanned_resumes:
        try:
            print(f"[Multimodal] Visually parsing scanned/non-text PDF resume: '{filename}'...")
            parsed = parser.parse_resume_pdf(pdf_bytes, filename)
            record = _build_candidate_record(
                parsed, parsed.get("resume_text", ""), filename, existing_candidates, existing_ids, job
            )
            if record is not None:
                new_records.append(record)
            with job._lock:
                job.parsed_count += 1
        except Exception as vis_err:
            print(f"[Error] Failed to parse scanned PDF '{filename}' visually: {vis_err}")
            with job._lock:
                job.failed_count += 1
                job.failed_files.append(filename)

