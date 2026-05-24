import argparse
import sys
from pathlib import Path

# Add the parent directory of talent_radar to sys.path to support importing talent_radar
sys.path.insert(0, str(Path(__file__).parent.parent))

from talent_radar.pipeline import TalentRadarPipeline
from talent_radar.output import format_and_write_output

def main():
    parser = argparse.ArgumentParser(description="Talent Radar — AI-Powered Candidate Ranking Engine")
    parser.add_argument(
        "--jd", 
        type=str, 
        required=True, 
        help="Path to the Job Description (JD) text file."
    )
    parser.add_argument(
        "--seniority", 
        type=str, 
        default="Senior", 
        choices=["Junior", "Mid", "Senior", "Lead", "Principal"],
        help="Target seniority level for the position (default: Senior)"
    )
    parser.add_argument(
        "--out", 
        type=str, 
        default="talent_radar_output.csv", 
        help="Path to output the ranked candidates CSV file (default: talent_radar_output.csv)"
    )
    parser.add_argument(
        "--top_k", 
        type=int, 
        default=50, 
        help="Number of vector space candidates to retrieve initially (default: 50)"
    )
    
    args = parser.parse_args()
    
    jd_path = Path(args.jd)
    if not jd_path.exists():
        print(f"Error: Job Description file not found at: {jd_path.resolve()}")
        sys.exit(1)
        
    print(f"Reading Job Description from: {jd_path.resolve()}...")
    with open(jd_path, "r", encoding="utf-8") as f:
        job_description = f.read()
        
    print("\n--- JOB DESCRIPTION PREVIEW ---")
    lines = job_description.strip().split("\n")
    for line in lines[:5]:
        print(line)
    if len(lines) > 5:
        print("...")
    print("--------------------------------\n")
    
    try:
        # Initialize pipeline
        pipeline = TalentRadarPipeline()
