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
        
        # Run pipeline
        candidates, expanded_query, timings = pipeline.run(
            job_description=job_description,
            seniority_level=args.seniority,
            top_k=args.top_k
        )
        
        if not candidates:
            print("No matching candidates found.")
            sys.exit(0)
            
        # Export outputs
        csv_out_path = Path(args.out)
        json_out_path = csv_out_path.with_suffix(".json")
        
        df = format_and_write_output(
            candidates=candidates,
            csv_path=csv_out_path,
            json_path=json_out_path
        )
        
        def clean(val):
            if not isinstance(val, str):
                return val
            return val.encode('ascii', 'ignore').decode('ascii').strip()
            
        # Output gorgeous visual summary to console
        print("\n" + "="*80)
        print(f"  TOP 5 CANDIDATES MATCHED FOR {args.seniority.upper()} ROLE  ")
        print("="*80)
        for idx, row in df.head(5).iterrows():
            name_clean = clean(row['name'])
            title_clean = clean(row['current_title'])
            status_clean = clean(row['status_label'])
            reasoning_clean = clean(row['reasoning'])
            
            print(f"Rank {row['rank']}: {name_clean} ({title_clean})")
            print(f"  - Composite Score: {row['final_score']:.1f}/100 | Semantic Score: {row['semantic_score']:.1f}/100 | Velocity: {row['velocity_score']:.1f}/10")
            print(f"  - Freshness: {row['freshness_label']} | Status: {status_clean}")
            print(f"  - Why: {reasoning_clean}")
            print("-" * 80)
            
        print("\nPipeline execution metrics:")
        print(f"- Step 1 Query Explosion:    {timings['query_explosion_ms']:.1f}ms")
        print(f"- Step 2 Vector Retrieval:   {timings['vector_retrieval_ms']:.1f}ms")
        print(f"- Step 3 Semantic Reranking: {timings['cross_encoder_rerank_ms']:.1f}ms")
        print(f"- Step 4 Momentum Scorer:    {timings['scorer_scoring_ms']:.1f}ms")
        print(f"- Total Pipeline E2E Time:   {timings['overall_ms']:.1f}ms")
        print("="*80)
        
    except Exception as e:
        import traceback
        print(f"\nPipeline failure during execution: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
