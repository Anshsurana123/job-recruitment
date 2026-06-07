import csv
import io
import datetime

class BriefExporter:
    @staticmethod
    def generate_markdown_brief(
        job_description: str,
        seniority_level: str,
        sector: str,
        candidates: list[dict],
        top_k: int = 5
    ) -> str:
        """Generates a premium, recruiter-ready search brief in beautiful Markdown."""
        today_str = datetime.date.today().strftime("%B %d, %Y")
        
        # Sector naming
        SECTOR_NAMES = {
            "TECH": "Technology & Software",
            "FIN": "Finance & Banking",
            "HEALTH": "Healthcare & Life Sciences",
            "LEGAL": "Legal & Compliance",
            "REAL": "Real Estate",
            "MANU": "Manufacturing & Engineering",
            "COMM": "Commerce & Retail",
            "LOGI": "Logistics & Supply Chain",
            "MEDIA": "Media & Creative",
            "ENERGY": "Energy & Utilities",
            "EDU": "Education",
            "GOV": "Government & Public Sector"
        }
        sector_name = SECTOR_NAMES.get(sector, "Specialist Domain")

        # Compile statistics
        total_pool = len(candidates)
        gems = sum(1 for c in candidates if "Hidden Gem" in c.get("status_label", ""))
        solids = sum(1 for c in candidates if "Solid Match" in c.get("status_label", ""))
        potentials = sum(1 for c in candidates if "Potential Fit" in c.get("status_label", ""))
        
        md = []
        md.append(f"# 🧠 SwarmMatrix AI — Talent Radar Search Brief")
        md.append(f"### *Executive Candidate Assessment & Pipeline Overview*")
        md.append(f"**Date**: {today_str} | **Sector**: {sector} ({sector_name}) | **Target Seniority**: {seniority_level}")
        md.append(f"\n---")
        
        # 1. Target Role & Search parameters
        md.append(f"## 📋 Search Parameters & Objectives")
        md.append(f"### Raw Requirement")
        md.append(f"> {job_description.strip()}")
        md.append(f"\n### Candidate Pool Statistics")
        md.append(f"- **Total Candidates Evaluated**: {total_pool}")
        md.append(f"- **Top Hidden Gems Surface (🚀)**: {gems}")
        md.append(f"- **Solid Matches Identified (🏆)**: {solids}")
        md.append(f"- **Potential Fits Shortlisted (⭐)**: {potentials}")
        md.append(f"\n---")
        
        # 2. Pipeline Summary Table
        md.append(f"## 📊 Candidate Screening Pipeline")
        md.append(f"Below are the top matching profiles listed in rank order based on the SwarmMatrix Composite scoring formula (60% Semantic Depth + 25% Career Velocity + 15% Profile Freshness).")
        md.append(f"\n| Rank | Name | Current Title | Experience | Velocity | Freshness | Match Label | Final Score |")
        md.append(f"| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        
        limit_candidates = candidates[:top_k]
        for idx, cand in enumerate(candidates[:20], 1):
            name = cand.get("name", "Unknown Candidate")
            title = cand.get("current_title", "Developer")
            exp = cand.get("years_experience", 1.0)
            velocity = cand.get("velocity_score", 0.0)
            freshness = cand.get("freshness_label", "Dormant")
            status = cand.get("status_label", "Longshot")
            score = cand.get("final_score", 0.0)
            md.append(f"| {idx:02d} | **{name}** | {title} | {exp:.1f}y | {velocity}/10 | {freshness} | {status} | **{score:.1f}** |")
            
        md.append(f"\n---")
        
        # 3. Top Candidates Profiles Dossier
        md.append(f"## 👤 Top Match In-Depth Profiles")
        md.append(f"Detailed contextual summaries for the top 5 candidates showing peak semantic alignment.")
        
        for idx, cand in enumerate(limit_candidates, 1):
            name = cand.get("name", "Unknown Candidate")
            title = cand.get("current_title", "Developer")
            exp = cand.get("years_experience", 1.0)
            velocity = cand.get("velocity_score", 0.0)
            fresh = cand.get("freshness_label", "Dormant")
            status = cand.get("status_label", "Longshot")
            score = cand.get("final_score", 0.0)
            reason = cand.get("reasoning", "")
            location = cand.get("location", "Remote")
            education = cand.get("education", "Not specified")
            
            matched = cand.get("matched_skills", [])
            missing = cand.get("missing_skills", [])
            
            md.append(f"\n### {idx}. {name} — {status}")
            md.append(f"- **Current Title**: {title}")
            md.append(f"- **Final Unified Score**: **{score:.1f} / 100**")
            md.append(f"- **Location**: {location} | **Education**: {education}")
            md.append(f"- **Metrics**: {exp:.1f} years experience | Career Velocity: {velocity}/10 | Activity Index: {fresh}")
            
            # Matched & Missing Skills
            md.append(f"\n#### Skill Analysis")
            md.append(f"- **✓ Core Matches**: {', '.join(matched) if matched else 'None'}")
            md.append(f"- **⚠️ Key Gaps**: {', '.join(missing) if missing else 'None'}")
            
            # Recruiter Pitch
            md.append(f"\n#### Recruiter Pitch & Alignment Rationale")
            md.append(f"> {reason}")
            
            # Experience history
            md.append(f"\n#### Career History Timeline")
            for job in cand.get("career_history", []):
                md.append(f"- *{job.get('title')}* at **{job.get('company')}** ({job.get('start_date')} to {job.get('end_date')})")
            md.append(f"\n---")
            
        md.append(f"\n*Report generated securely by SwarmMatrix AI Talent Radar Recruiter Console. Confidential - For Internal Use Only.*")
        
        return "\n".join(md)

    @staticmethod
    def generate_csv_brief(candidates: list[dict]) -> str:
        """Generates a downloadable CSV of the entire ranked candidate pool."""
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Headers
        writer.writerow([
            "Rank", "Candidate ID", "Name", "Current Title", "Years Experience", 
            "Career Velocity (0-10)", "Freshness Label", "Status Label", 
            "Semantic Score (raw)", "Education Bonus", "Final Composite Score", "Confidential Reasoning"
        ])
        
        for idx, cand in enumerate(candidates, 1):
            writer.writerow([
                idx,
                cand.get("candidate_id"),
                cand.get("name"),
                cand.get("current_title"),
                cand.get("years_experience"),
                cand.get("velocity_score"),
                cand.get("freshness_label"),
                cand.get("status_label"),
                cand.get("semantic_score"),
                cand.get("education_bonus"),
                cand.get("final_score"),
                cand.get("reasoning")
            ])
            
        return output.getvalue()
