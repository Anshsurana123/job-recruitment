import json
from pathlib import Path

candidates_path = Path(__file__).parent / "candidates.json"
if not candidates_path.exists():
    print("candidates.json does not exist!")
    exit()

with open(candidates_path, "r", encoding="utf-8") as f:
    candidates = json.load(f)

print(f"Total candidates: {len(candidates)}")

# Print first 5 candidate names and current_titles
print("\nFirst 5 candidates:")
for c in candidates[:5]:
    print(f"- {c.get('name')} | {c.get('current_title')} | {c.get('candidate_id')}")

# Print last 5 candidate names and current_titles
print("\nLast 5 candidates:")
for c in candidates[-5:]:
    print(f"- {c.get('name')} | {c.get('current_title')} | {c.get('candidate_id')}")

# Count keywords in resume texts
healthcare_keywords = ["surgical", "surgery", "surgeon", "hospital", "patient", "medical", "clinical", "nurse", "doctor"]
counts = {k: 0 for k in healthcare_keywords}
source_file_counts = 0
unique_source_paths = set()

for c in candidates:
    resume = c.get("resume_text", "").lower()
    for k in healthcare_keywords:
        if k in resume:
            counts[k] += 1
    if c.get("source_file"):
        source_file_counts += 1
        unique_source_paths.add(c.get("source_file"))

print(f"\nCandidates with source_file field: {source_file_counts}")

# Categorize source files by mapping to the actual directories in Downloads
downloads_data_dir = Path(r"C:\Users\ANSH\Downloads\archive (1)\data\data")
if downloads_data_dir.exists():
    category_map = {}
    # Scan all directories in downloads and index their files
    for sub in downloads_data_dir.iterdir():
        if sub.is_dir():
            category_map[sub.name] = {f.name for f in sub.glob("*.pdf")}
    
    source_categories = {}
    for c in candidates:
        sf = c.get("source_file")
        if sf:
            # Strip path if it's stored with folder names
            sf_name = Path(sf).name
            matched = False
            for cat, files in category_map.items():
                if sf_name in files:
                    source_categories[cat] = source_categories.get(cat, 0) + 1
                    matched = True
                    break
            if not matched:
                source_categories["Unknown"] = source_categories.get("Unknown", 0) + 1
                
    print("\nCandidate counts by source category:")
    for cat, count in sorted(source_categories.items(), key=lambda x: x[1], reverse=True):
        print(f"- {cat}: {count}")

print("\nKeyword counts in resume texts:")
for k, count in counts.items():
    print(f"- {k}: {count}")
