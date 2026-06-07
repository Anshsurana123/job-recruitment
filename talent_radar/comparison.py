import os
import json
from pathlib import Path
from pydantic import BaseModel, Field
from google.genai import types

# Load env
base_dir = Path(__file__).parent
env_path = base_dir.parent / ".env"
if env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=env_path)

class CandidateMetricComparison(BaseModel):
    candidate_name: str
    final_score: float
    years_experience: float
    career_velocity: float
    freshness: str
    key_strengths: list[str] = Field(description="Top 3 technical or experience strengths of this candidate.")
    critical_weaknesses: list[str] = Field(description="Top 2 gaps or development areas for this candidate relative to the JD.")

class ComparisonSynthesis(BaseModel):
    metrics: list[CandidateMetricComparison]
    strengths_synthesis: str = Field(description="A detailed side-by-side analysis comparing candidates' key strengths.")
    tradeoffs_synthesis: str = Field(description="A clear summary of technical, experience, and seniority trade-offs between the candidates.")
    recruiter_recommendation: str = Field(description="Direct, action-oriented recommendation on who to prioritize for this specific job description, and why.")

class CandidateComparator:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = "gemini-3.1-flash-lite"
        self._client = None

    @property
    def client(self):
        if self._client is None and self.api_key:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def compare_candidates(self, candidates: list[dict], job_description: str) -> ComparisonSynthesis:
        """Runs side-by-side comparison over 2 or 3 candidates using Gemini 2.5 Flash."""
        if not candidates:
            raise ValueError("No candidates provided for comparison.")

        candidates_data_str = []
        for cand in candidates:
            c_desc = f"""
            Name: {cand.get('name')}
            Title: {cand.get('current_title')}
            Experience: {cand.get('years_experience')} years
            Final Score: {cand.get('final_score')}
            Semantic Depth: {cand.get('semantic_score')}%
            Velocity Score: {cand.get('velocity_score')}/10
            Freshness: {cand.get('freshness_label')}
            Skills: {', '.join(cand.get('skills_listed', []))}
            Matches: {', '.join(cand.get('matched_skills', []))}
            Gaps: {', '.join(cand.get('missing_skills', []))}
            Pitch: {cand.get('reasoning')}
            """
            candidates_data_str.append(c_desc)

        prompt = f"""
        Analyze and perform a professional, side-by-side recruitment evaluation comparing the following candidates:
        
        {chr(10).join(candidates_data_str)}
        
        Target Job Description:
        "{job_description}"
        
        Requirements:
        1. Compare their respective metrics and skills.
        2. Identify core technical strengths and critical weaknesses/gaps for EACH candidate.
        3. Formulate a comprehensive synthesis comparing their strengths and trade-offs.
        4. Deliver a final, objective recruiter recommendation indicating who is the primary candidate to hire, and why.
        """

        system_instruction = (
            "You are an expert Chief Technology Officer and Executive Search Consultant. "
            "Your job is to compare top-tier candidates side-by-side, highlight critical engineering "
            "trade-offs (e.g. deep niche skill vs generalist capacity, rapid velocity vs long-term stability), "
            "and supply an actionable, high-conviction hire recommendation."
        )

        if self.client:
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        response_schema=ComparisonSynthesis,
                        temperature=0.2
                    )
                )
                data = json.loads(response.text)
                return ComparisonSynthesis(**data)
            except Exception as e:
                print(f"[Comparison Engine] Structured generation error: {e}. Cascading to local fallback generator...")

        return self._generate_local_fallback(candidates)

    def _generate_local_fallback(self, candidates: list[dict]) -> ComparisonSynthesis:
        """High-fidelity local fallback generator for candidate comparisons."""
        metrics_list = []
        names = []
        for cand in candidates:
            name = cand.get("name", "Unknown Candidate")
            names.append(name)
            
            matched = cand.get("matched_skills", [])
            missing = cand.get("missing_skills", [])
            
            strengths = matched[:3] if matched else ["Experience profile"]
            if len(strengths) < 3:
                strengths.append("Career history progression")
            if len(strengths) < 3:
                strengths.append("Self-declared skill alignment")
                
            weaknesses = missing[:2] if missing else ["Profile details verification"]
            if len(weaknesses) < 2:
                weaknesses.append("Missing adjacent stack elements")

            metrics_list.append(CandidateMetricComparison(
                candidate_name=name,
                final_score=float(cand.get("final_score", 50.0)),
                years_experience=float(cand.get("years_experience", 1.0)),
                career_velocity=float(cand.get("velocity_score", 1.0)),
                freshness=str(cand.get("freshness_label", "Dormant")),
                key_strengths=strengths,
                critical_weaknesses=weaknesses
            ))

        # Generate comparative sentences based on candidate stats
        sorted_cand = sorted(candidates, key=lambda c: c.get("final_score", 0.0), reverse=True)
        top_cand = sorted_cand[0]
        top_name = top_cand.get("name", "Top Candidate")

        names_str = " vs ".join(names)
        
        strengths_syn = (
            f"The comparative review of {names_str} highlights distinct architectural alignments. "
            f"{top_name} leads the pool with a composite score of {top_cand.get('final_score')}, demonstrating "
            f"strong matches in {', '.join(top_cand.get('matched_skills', [])[:3])}. "
        )
        if len(sorted_cand) > 1:
            sec_cand = sorted_cand[1]
            sec_name = sec_cand.get("name")
            strengths_syn += (
                f"On the other hand, {sec_name} exhibits strong competency in "
                f"{', '.join(sec_cand.get('matched_skills', [])[:2])} with {sec_cand.get('years_experience')} years of experience."
            )

        tradeoffs_syn = (
            f"Key trade-offs lie between raw experience depth and career trajectory velocity. "
            f"While some candidates bring longer stability, others demonstrate rapid promotion slopes and active profile intent. "
            f"Recruiters must balance quick learning potential (high career velocity) against long-term stable experience."
        )

        rec_rec = (
            f"Based on the composite evaluation, we highly recommend prioritizing {top_name} for the target position. "
            f"With a {top_cand.get('semantic_score')}% semantic match and a career velocity of {top_cand.get('velocity_score')}/10, "
            f"they represent the most balanced fit to immediately contribute to key development milestones while raising team standards."
        )

        return ComparisonSynthesis(
            metrics=metrics_list,
            strengths_synthesis=strengths_syn,
            tradeoffs_synthesis=tradeoffs_syn,
            recruiter_recommendation=rec_rec
        )
