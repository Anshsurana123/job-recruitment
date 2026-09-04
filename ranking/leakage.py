"""
Data leakage detection and duplicate content management.
Prevents train/evaluation leakage, identical/near-duplicate resumes, and future information contamination.
"""

import hashlib
import re
from typing import List, Dict, Set, Tuple, Any


def compute_content_hash(text: str) -> str:
    """Computes a SHA256 hash of normalized text (lowercase, stripped whitespace)."""
    norm = re.sub(r'\s+', ' ', (text or "").lower()).strip()
    return hashlib.sha256(norm.encode('utf-8')).hexdigest()


def detect_duplicate_content(resume_text: str, chunk_size: int = 200, threshold: float = 0.4) -> Tuple[bool, float]:
    """
    Detects repeated/synthetic boilerplate content within a single resume
    by chunking text into fixed slices and measuring unique ratio.
    """
    if not resume_text or len(resume_text) < chunk_size:
        return False, 0.0
    chunks = [resume_text[i:i + chunk_size] for i in range(0, len(resume_text), chunk_size)]
    unique = set(chunks)
    duplication_ratio = 1.0 - (len(unique) / max(len(chunks), 1))
    return duplication_ratio > threshold, duplication_ratio


def detect_exact_duplicates(candidates: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Identifies exact duplicate candidate IDs and exact identical resume texts.
    Returns a dictionary of duplicate groupings.
    """
    seen_ids: Dict[str, int] = {}
    id_duplicates: List[str] = []
    
    hash_to_ids: Dict[str, List[str]] = {}
    
    for cand in candidates:
        cid = cand.get("candidate_id", "UNKNOWN")
        seen_ids[cid] = seen_ids.get(cid, 0) + 1
        if seen_ids[cid] == 2:
            id_duplicates.append(cid)
            
        # Compute resume or profile text hash
        resume_text = cand.get("resume_text", "")
        if not resume_text and "profile" in cand:
            profile = cand.get("profile", {})
            resume_text = f"{profile.get('current_title', '')} {profile.get('summary', '')}"
            
        h = compute_content_hash(resume_text)
        if h:
            hash_to_ids.setdefault(h, []).append(cid)
            
    content_duplicates = {h: ids for h, ids in hash_to_ids.items() if len(ids) > 1}
    
    return {
        "duplicate_ids": id_duplicates,
        "duplicate_content_groups": content_duplicates
    }


def compute_shingle_jaccard(text1: str, text2: str, k: int = 4) -> float:
    """Computes Jaccard similarity over word k-shingles."""
    words1 = re.findall(r'\b[a-z0-9_]+\b', text1.lower())
    words2 = re.findall(r'\b[a-z0-9_]+\b', text2.lower())
    
    if len(words1) < k or len(words2) < k:
        s1 = set(words1)
        s2 = set(words2)
        union = len(s1 | s2)
        return len(s1 & s2) / union if union > 0 else 0.0
        
    shingles1 = set(tuple(words1[i:i + k]) for i in range(len(words1) - k + 1))
    shingles2 = set(tuple(words2[i:i + k]) for i in range(len(words2) - k + 1))
    
    union = len(shingles1 | shingles2)
    return len(shingles1 & shingles2) / union if union > 0 else 0.0


def validate_splits_leakage(
    train_candidates: List[Dict[str, Any]],
    val_candidates: List[Dict[str, Any]],
    similarity_threshold: float = 0.90
) -> List[str]:
    """
    Audits train and validation/evaluation splits for data leakage:
    1. Candidate ID overlap
    2. Exact resume/profile text match
    3. Near-duplicate text similarity above threshold
    """
    violations: List[str] = []
    
    train_ids = {c.get("candidate_id") for c in train_candidates if c.get("candidate_id")}
    val_ids = {c.get("candidate_id") for c in val_candidates if c.get("candidate_id")}
    
    id_overlap = train_ids & val_ids
    if id_overlap:
        violations.append(f"Leakage detected: {len(id_overlap)} candidate_id(s) exist in both train and validation splits: {list(id_overlap)[:5]}")
        
    train_hashes = {}
    for c in train_candidates:
        cid = c.get("candidate_id", "")
        text = c.get("resume_text", "")
        if not text and "profile" in c:
            text = f"{c['profile'].get('current_title', '')} {c['profile'].get('summary', '')}"
        h = compute_content_hash(text)
        train_hashes[h] = cid
        
    for c in val_candidates:
        cid = c.get("candidate_id", "")
        text = c.get("resume_text", "")
        if not text and "profile" in c:
            text = f"{c['profile'].get('current_title', '')} {c['profile'].get('summary', '')}"
        h = compute_content_hash(text)
        if h in train_hashes:
            violations.append(f"Content leakage: Candidate '{cid}' in validation has exact identical resume content to '{train_hashes[h]}' in train.")
            
    return violations
