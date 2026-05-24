import os
import json
import re
from pathlib import Path
from pydantic import BaseModel, Field
from google.genai import types

# Load env
base_dir = Path(__file__).parent
env_path = base_dir.parent / ".env"
if env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=env_path)

class InterviewQuestion(BaseModel):
    question: str = Field(description="The interview question to ask the candidate.")
    expected_answer: str = Field(description="What key concepts, technologies, or engineering answers the recruiter should listen for.")
    rationale: str = Field(description="Rationale explaining why this question is targeted specifically to this candidate based on their skills gaps, seniority, or background.")

class PhoneScreenGuide(BaseModel):
    candidate_name: str
    target_role: str
    questions: list[InterviewQuestion] = Field(description="A list of exactly 5 tailored interview questions.")

class TechnicalQuestionGenerator:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = "gemini-2.5-flash"
        self._client = None

    @property
    def client(self):
        if self._client is None and self.api_key:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def generate_questions(self, candidate: dict, job_description: str, sector: str = "TECH") -> PhoneScreenGuide:
        """Generates 5 personalized phone screen questions using Gemini 2.5 Flash structured output."""
        name = candidate.get("name", "Unknown Candidate")
        current_title = candidate.get("current_title", "Software Engineer")
        experience = candidate.get("years_experience", 1.0)
        skills_listed = candidate.get("skills_listed", [])
        education = candidate.get("education", "")
        career_history = candidate.get("career_history", [])
        matched_skills = candidate.get("matched_skills", [])
        missing_skills = candidate.get("missing_skills", [])

        # Format candidate history
        history_desc = []
        for job in career_history:
            history_desc.append(f"- {job.get('title')} at {job.get('company')} ({job.get('start_date')} to {job.get('end_date')})")
        history_str = "\n".join(history_desc)

        prompt = f"""
        You are an elite technical recruiting manager. Generate exactly 5 highly customized, high-signal interview questions for:
        
        Candidate: {name}
        Current Role: {current_title}
        Total Experience: {experience} years
        Education: {education}
        Key Skills Listed: {', '.join(skills_listed)}
        Matched Job Skills: {', '.join(matched_skills)}
        Missing / Gapped Skills: {', '.join(missing_skills)}
        
        Career Trajectory:
        {history_str}
        
        Target Job Description:
        "{job_description}"
        Target Industry Sector: {sector}
        
        Requirements:
        1. Create exactly 5 distinct questions.
        2. Tailor them specifically to this candidate's profile.
        3. Do NOT make them generic. Focus on the GAPS (missing skills), their fast or slow career velocity, or specific challenges mentioned in their background.
        4. Focus on deep understanding: if the candidate claims "flash attention" or "micro-frontend", ask a question verifying *how* they actually scaled or resolved real-world issues.
        5. For each question, supply a concrete answer check-sheet of key concepts or terms the recruiter should listen for.
        """

        system_instruction = (
            "You are an expert technical interviewer and recruiting director. "
            "Your goal is to screen candidates effectively by asking deep, tailored questions "
            "that immediately expose the difference between a high-caliber hidden gem and a resume stuffer."
        )

        if self.client:
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
