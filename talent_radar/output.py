import pandas as pd
import json
from pathlib import Path

def format_and_write_output(candidates, csv_path, json_path=None):
    """
    Slices the top 20 ranked candidates and formats them into
    the exact columns requested in the output specification.
    """
    print(f"Formatting top 20 ranked candidates into output schemas...")
    
    # Slice top 20
    top_20 = candidates[:20]
    
    output_rows = []
    for idx, cand in enumerate(top_20):
        output_rows.append({
            "rank": idx + 1,
            "candidate_id": cand["candidate_id"],
            "name": cand["name"],
            "current_title": cand["current_title"],
            "semantic_score": cand["semantic_score"],
            "velocity_score": cand["velocity_score"],
            "freshness_label": cand["freshness_label"],
            "final_score": cand["final_score"],
            "status_label": cand["status_label"],
            "reasoning": cand["reasoning"]
        })
        
    # Write to CSV
    df = pd.DataFrame(output_rows)
    csv_file = Path(csv_path)
    df.to_csv(csv_file, index=False, encoding="utf-8")
    print(f"Successfully exported ranked CSV to: {csv_file.resolve()}")
    
    # Write to JSON for the Web UI or other API clients
    if json_path:
        json_file = Path(json_path)
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(output_rows, f, indent=2, ensure_ascii=False)
        print(f"Successfully exported ranked JSON to: {json_file.resolve()}")
        
    return df
