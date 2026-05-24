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
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        response_schema=PhoneScreenGuide,
                        temperature=0.2
                    )
                )
                data = json.loads(response.text)
                return PhoneScreenGuide(**data)
            except Exception as e:
                print(f"[Question Generator] Structured generation error: {e}. Cascading to high-fidelity local generator...")

        # Robust, high-fidelity local fallback generator if offline or rate-limited
        return self._generate_local_fallback(name, current_title, experience, missing_skills, matched_skills, sector)

    def _generate_local_fallback(self, name: str, current_title: str, experience: float, missing_skills: list, matched_skills: list, sector: str) -> PhoneScreenGuide:
        """Fallback local question generator based on candidate metrics."""
        questions = []
        
        # Q1: Dynamic sector competency check
        questions.append(InterviewQuestion(
            question=f"In your recent roles in the {sector} space, what has been the most complex scaling challenge you owned, and how did you diagnose and solve it?",
            expected_answer="Recruiter should look for: Specific engineering tradeoffs, concrete metrics (e.g. reduced latency, hosting costs), and deep architectural understanding.",
            rationale="Verifies claimed high-level experience and career velocity directly by probing their peak engineering moment."
        ))

        # Q2: Missing skill deep dive 1
        g1 = missing_skills[0] if len(missing_skills) > 0 else "System Optimization"
        questions.append(InterviewQuestion(
            question=f"The job requires hands-on experience with {g1}, which isn't heavily emphasized in your resume. How would you approach adopting {g1} in your work, or can you share adjacent experience?",
            expected_answer=f"Recruiter should look for: Conceptual familiarity with {g1}, concrete analogies to tools they *do* know, and quick learning aptitude.",
            rationale=f"Directly screens for the primary skill gap ({g1}) to assess if they can quickly ramp up."
        ))

        # Q3: Missing skill deep dive 2
        g2 = missing_skills[1] if len(missing_skills) > 1 else "Architectural Patterns"
        questions.append(InterviewQuestion(
            question=f"Another key requirement is {g2}. How have you handled similar architectural design decisions in your past projects, and what are the failure modes you watch out for?",
            expected_answer=f"Recruiter should look for: Critical design thinking, concrete patterns (e.g., microservices, micro-frontends, event-sourcing), and testing strategies.",
            rationale=f"Probes their design competence around the secondary gap ({g2}) to assess overall senior capabilities."
        ))

        # Q4: Career trajectory & growth
        questions.append(InterviewQuestion(
            question="Your profile shows a dynamic growth path over your career. What standard processes or engineering standards did you actively define or drive at your previous companies to scale team execution?",
            expected_answer="Recruiter should look for: Mentorship signals, establishing code-reviews, driving RFC/design docs, and elevating overall engineering standards.",
            rationale="Determines senior leadership/behavioral capability relative to their career progression velocity."
        ))

        # Q5: Technical choice & tradeoff
        m1 = matched_skills[0] if len(matched_skills) > 0 else (matched_skills[0] if len(matched_skills) > 0 else "Python")
        questions.append(InterviewQuestion(
            question=f"You listed {m1} as a core strength. What is a specific, non-obvious performance optimization or failure mode you encountered when using {m1} in production, and how did you resolve it?",
            expected_answer="Recruiter should look for: High-fidelity details, deep language or framework understanding, and genuine hands-on debugging competency.",
            rationale="Double-checks matched skills to verify they have deep contextual relevance rather than superficial keyword overlap."
        ))

        return PhoneScreenGuide(
            candidate_name=name,
            target_role=current_title,
            questions=questions
        )
