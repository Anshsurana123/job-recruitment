import os
import sys
import json
import time
from pathlib import Path
from pypdf import PdfReader

# Add workspace directory to path
base_dir = Path(__file__).parent
sys.path.insert(0, str(base_dir.parent))

from talent_radar.llm_parser import GeminiResumeParser

def extract_text_from_pdf(pdf_path: Path) -> tuple[str, bytes]:
    """Reads PDF and extracts text using pypdf/pdfplumber."""
    try:
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
            
        extracted_text = ""
        # 1. Try pypdf text extraction
        try:
            reader = PdfReader(Path(pdf_path))
            for page in reader.pages:
                extracted_text += page.extract_text() or ""
            extracted_text = extracted_text.strip()
        except Exception as e:
            print(f"[Extractor] pypdf failed on '{pdf_path.name}': {e}")
            extracted_text = ""
            
        # 2. Try pdfplumber fallback
        if not extracted_text:
            try:
                import pdfplumber
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        extracted_text += page.extract_text() or ""
                extracted_text = extracted_text.strip()
                if extracted_text:
                    print(f"[Extractor] pdfplumber successfully recovered text for '{pdf_path.name}'")
            except Exception as e:
                print(f"[Extractor] pdfplumber fallback failed on '{pdf_path.name}': {e}")
                
        return extracted_text, pdf_bytes
    except Exception as e:
        print(f"[Error] Failed to read file '{pdf_path.name}': {e}")
        return "", b""

def build_candidate_record(parsed: dict, raw_text: str, filename: str, existing_candidates: list, existing_ids: set) -> dict:
    """Builds a validated candidate record from the parsed Gemini response."""
    cand_idx = len(existing_candidates) + 1
    cand_id = f"cand_{cand_idx:04d}"
    while cand_id in existing_ids:
        cand_idx += 1
        cand_id = f"cand_{cand_idx:04d}"
    existing_ids.add(cand_id)

    # Clean career history
    cleaned_history = []
    for entry in parsed.get("career_history", []):
        if isinstance(entry, dict):
            cleaned_history.append({
                "title": str(entry.get("title", "Developer")),
                "company": str(entry.get("company", "Company")),
                "start_date": str(entry.get("start_date", "2020-01-01")),
                "end_date": entry.get("end_date"),
            })

    return {
        "candidate_id": cand_id,
        "name": str(parsed.get("name", "Unknown Candidate")),
        "current_title": str(parsed.get("current_title", "Software Engineer")),
        "years_experience": float(parsed.get("years_experience", 1.0)),
        "career_history": cleaned_history,
        "skills_listed": parsed.get("skills_listed", []),
        "last_active": parsed.get("last_active") if parsed.get("last_active") else None,
        "education": str(parsed.get("education", "")),
        "location": str(parsed.get("location", "Remote")),
        "resume_text": str(parsed.get("resume_text", raw_text)),
        "source_file": filename,
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python ingest_directory.py <directory_path>")
        sys.exit(1)
        
    dir_path = Path(sys.argv[1])
    if not dir_path.exists() or not dir_path.is_dir():
        print(f"Directory not found: {dir_path}")
        sys.exit(1)
        
    print(f"\nScanning directory: {dir_path.resolve()}")
    pdf_files = sorted(list(dir_path.glob("*.pdf")))
    print(f"Found {len(pdf_files)} PDF files.")
    
    candidates_path = base_dir / "candidates.json"
    existing_candidates = []
    if candidates_path.exists():
        try:
            with open(candidates_path, "r", encoding="utf-8") as f:
                existing_candidates = json.load(f)
        except Exception:
            existing_candidates = []
            
    print(f"Loaded {len(existing_candidates)} existing candidates from candidates.json.")
    
    # Skip already-parsed files
    parsed_sources = {c.get("source_file") for c in existing_candidates if c.get("source_file")}
    unparsed_files = [f for f in pdf_files if f.name not in parsed_sources]
    
    print(f"Skipping {len(pdf_files) - len(unparsed_files)} already-parsed files.")
    print(f"{len(unparsed_files)} files remaining to be parsed.")
    
    if not unparsed_files:
        print("No new files to parse. Ingestion complete!")
        return

    parser = GeminiResumeParser()
    existing_ids = {c.get("candidate_id") for c in existing_candidates}
    
    # Process files in batches of 15
    batch_size = 15
    new_records = []
    
    for i in range(0, len(unparsed_files), batch_size):
        batch_files = unparsed_files[i:i+batch_size]
        print(f"\nProcessing batch {i // batch_size + 1} of {(len(unparsed_files) - 1) // batch_size + 1} ({len(batch_files)} files)...")
        
        batch_text_items = []
        batch_scanned_items = []
        
        for fpath in batch_files:
            txt, pbytes = extract_text_from_pdf(fpath)
            if txt:
                batch_text_items.append((fpath.name, txt, pbytes))
            else:
                batch_scanned_items.append((fpath.name, txt, pbytes))
                
        # 1. Process text items in batch
        if batch_text_items:
            batch_input = [(fname, txt) for fname, txt, _ in batch_text_items]
            try:
                print(f"Sending batch of {len(batch_text_items)} text resumes to Gemini...")
                parsed_array = parser.parse_resume_batch(batch_input)
                
                for idx, parsed in enumerate(parsed_array):
                    if idx >= len(batch_text_items):
                        break
                    fname, txt, pbytes = batch_text_items[idx]
                    record = build_candidate_record(parsed, txt, fname, existing_candidates + new_records, existing_ids)
                    new_records.append(record)
                    print(f"  Parsed: {record['name']} | {record['current_title']}")
                    
            except Exception as e:
                print(f"[Warning] Batch parsing failed: {e}. Falling back to individual parsing...")
                for fname, txt, pbytes in batch_text_items:
                    try:
                        parsed = parser.parse_resume(txt)
                        record = build_candidate_record(parsed, txt, fname, existing_candidates + new_records, existing_ids)
                        new_records.append(record)
                        print(f"  Parsed (Individual): {record['name']} | {record['current_title']}")
                    except Exception as ind_err:
                        print(f"  [Error] Failed to parse '{fname}': {ind_err}")
                        
        # 2. Process scanned items individually
        for fname, txt, pbytes in batch_scanned_items:
            try:
                print(f"Visually parsing scanned resume: '{fname}'...")
                parsed = parser.parse_resume_pdf(pbytes)
                record = build_candidate_record(parsed, parsed.get("resume_text", ""), fname, existing_candidates + new_records, existing_ids)
                new_records.append(record)
                print(f"  Parsed (Visual): {record['name']} | {record['current_title']}")
            except Exception as vis_err:
                print(f"  [Error] Failed to parse scanned '{fname}' visually: {vis_err}")
                
        # Checkpoint: save intermediate results after each batch
        if new_records:
            checkpoint_pool = existing_candidates + new_records
            try:
                with open(candidates_path, "w", encoding="utf-8") as f:
                    json.dump(checkpoint_pool, f, indent=2, ensure_ascii=False)
                print(f"Checkpoint saved: Total {len(checkpoint_pool)} candidates in candidates.json")
            except Exception as save_err:
                print(f"[Error] Failed to save candidates.json checkpoint: {save_err}")
                
    print(f"\nDirectory Ingestion complete! Successfully ingested {len(new_records)} new candidates.")

if __name__ == "__main__":
    main()
