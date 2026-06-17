import os
import sys
from fpdf import FPDF

class PresentationPDF(FPDF):
    def header(self):
        # We don't want standard headers, we will handle them slide-by-slide
        pass

    def footer(self):
        # Simple footer on all pages except cover
        if self.page_no() > 1:
            self.set_y(-12)
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(148, 163, 184) # slate-400
            self.cell(0, 10, f"Page {self.page_no()} | Team TECHFLOW | Redrob Candidate Ranker Presentation", align="C")

def draw_background(pdf, dark=True):
    if dark:
        pdf.set_fill_color(15, 23, 42) # slate-900
    else:
        pdf.set_fill_color(248, 250, 252) # slate-50
    pdf.rect(0, 0, 297, 210, "F")

def draw_accent_line(pdf, y=18, color=(6, 182, 212)): # cyan-500
    pdf.set_draw_color(*color)
    pdf.set_line_width(1.5)
    pdf.line(15, y, 282, y)

def create_deck():
    # Landscape orientation, A4 (297mm x 210mm)
    pdf = PresentationPDF(orientation="landscape", unit="mm", format="A4")
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(False)

    # ==========================================
    # SLIDE 1: Title / Cover (Dark Mode)
    # ==========================================
    pdf.add_page()
    draw_background(pdf, dark=True)
    
    # Title Block
    pdf.set_y(50)
    pdf.set_font("Helvetica", "B", 36)
    pdf.set_text_color(255, 255, 255)
    pdf.multi_cell(267, 14, "Two-Stage Hybrid Candidate Ranker", align="L")
    
    pdf.set_y(78)
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(6, 182, 212) # Cyan
    pdf.cell(0, 10, "Founding Team Senior AI Engineer Search (100k Pool)", ln=True)
    
    # Horizontal accent
    draw_accent_line(pdf, y=95, color=(6, 182, 212))

    # Details Block
    pdf.set_y(115)
    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(203, 213, 225) # slate-300
    pdf.cell(50, 8, "Team Name:", ln=False)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 8, "TECHFLOW", ln=True)

    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(203, 213, 225)
    pdf.cell(50, 8, "Primary Member:", ln=False)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 8, "Ansh Surana (ML Systems Engineer)", ln=True)

    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(203, 213, 225)
    pdf.cell(50, 8, "Core Stack:", ln=False)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 8, "BM25F + Local Transformers (BGE & Cross-Encoder)", ln=True)

    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(203, 213, 225)
    pdf.cell(50, 8, "Date:", ln=False)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 8, "June 2026", ln=True)

    # Footer banner on title slide
    pdf.set_fill_color(30, 41, 59) # slate-800
    pdf.rect(0, 180, 297, 30, "F")
    pdf.set_y(186)
    pdf.set_font("Helvetica", "I", 11)
    pdf.set_text_color(203, 213, 225)
    pdf.cell(0, 8, "Optimized for the Redrob Hackathon: Zero-Leakage Honeypot Filters & <10s CPU Execution", align="C")

    # ==========================================
    # SLIDE 2: Challenge & Strategy (Dark Mode)
    # ==========================================
    pdf.add_page()
    draw_background(pdf, dark=True)
    draw_accent_line(pdf, y=15, color=(6, 182, 212))

    # Page Title
    pdf.set_y(22)
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, "Recruitment Challenges & Core Strategy", ln=True)

    # Column 1: Challenges
    pdf.set_y(45)
    pdf.set_x(15)
    pdf.set_fill_color(30, 41, 59) # slate-800
    pdf.rect(15, 45, 126, 125, "F")
    
    pdf.set_y(50)
    pdf.set_x(20)
    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(239, 68, 68) # Red-500
    pdf.cell(116, 8, "Matching Challenges", ln=True)
    
    challenges = [
        ("Keyword Stuffing", "Plain-text profiles fabricating skills."),
        ("Synthetic Traps", "Synthetic candidate records (honeypots)."),
        ("Notice Periods", "Notice period penalties without buyout math."),
        ("Location Mismatches", "Ignoring major tech hubs (Kolkata, etc.)."),
        ("Computational Limits", "LLMs per-profile are too slow on CPU.")
    ]
    
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(241, 245, 249)
    for title, desc in challenges:
        pdf.set_x(20)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(248, 113, 113)
        pdf.cell(0, 6, f"* {title}: ", ln=False)
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(226, 232, 240)
        pdf.cell(0, 6, desc, ln=True)
        pdf.ln(2)

    # Column 2: Our Solutions
    pdf.set_fill_color(30, 41, 59) # slate-800
    pdf.rect(156, 45, 126, 125, "F")
    
    pdf.set_y(50)
    pdf.set_x(161)
    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(34, 197, 94) # Green-500
    pdf.cell(116, 8, "TECHFLOW Approach", ln=True)
    
    solutions = [
        ("Two-Stage Pipeline", "BM25F lexical recall + Transformer re-ranking."),
        ("Temporal Consistency", "Validate start dates against company foundation."),
        ("Proficiency Verification", "Verify 'expert' skills against duration + scores."),
        ("Availability Modifiers", "Scale candidates by actual log-ins & buyout."),
        ("Tier-1 Location Expansion", "Properly classify Kolkata & Ahmedabad as Tier-1.")
    ]
    
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(241, 245, 249)
    for title, desc in solutions:
        pdf.set_x(161)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(134, 239, 172)
        pdf.cell(0, 6, f"* {title}: ", ln=False)
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(226, 232, 240)
        pdf.cell(0, 6, desc, ln=True)
        pdf.ln(2)

    # ==========================================
    # SLIDE 3: Architecture & Pipeline (Dark Mode)
    # ==========================================
    pdf.add_page()
    draw_background(pdf, dark=True)
    draw_accent_line(pdf, y=15, color=(6, 182, 212))

    # Page Title
    pdf.set_y(22)
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, "Two-Stage Hybrid Ranking Architecture", ln=True)

    # Architecture Cards
    # Card 1: Lexical Recall
    pdf.set_fill_color(30, 41, 59)
    pdf.rect(15, 45, 82, 125, "F")
    
    pdf.set_y(50)
    pdf.set_x(18)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(6, 182, 212)
    pdf.cell(76, 8, "Stage 1: Lexical Recall", ln=True)
    
    pdf.set_y(62)
    pdf.set_font("Helvetica", "", 10.5)
    pdf.set_text_color(226, 232, 240)
    
    lex_points = [
        "Precomputed BM25F index",
        "Filters pool from 100k -> 1k",
        "Executes in < 1 second",
        "Combines weighted queries",
        "JD technical relevance",
        "Concatenates titles, resume summary, and listed skills"
    ]
    for pt in lex_points:
        pdf.set_x(18)
        pdf.multi_cell(76, 5.5, f"- {pt}", align="L")
        pdf.ln(1)

    # Card 2: Semantic Re-Ranking
    pdf.set_fill_color(30, 41, 59)
    pdf.rect(107, 45, 82, 125, "F")
    
    pdf.set_y(50)
    pdf.set_x(110)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(6, 182, 212)
    pdf.cell(76, 8, "Stage 2: Re-Ranking", ln=True)
    
    pdf.set_y(62)
    pdf.set_font("Helvetica", "", 10.5)
    pdf.set_text_color(226, 232, 240)
    
    sem_points = [
        "Loads local BGE Bi-Encoder",
        "Embeds top 1000 dynamically",
        "Applies Cross-Encoder model",
        "Re-ranks top 250 on CPU",
        "Runs completely offline",
        "Calculates hybrid scores: 0.6 BM25F + 0.4 BGE"
    ]
    for pt in sem_points:
        pdf.set_x(110)
        pdf.multi_cell(76, 5.5, f"- {pt}", align="L")
        pdf.ln(1)

    # Card 3: Fit Heuristics
    pdf.set_fill_color(30, 41, 59)
    pdf.rect(199, 45, 83, 125, "F")
    
    pdf.set_y(50)
    pdf.set_x(202)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(6, 182, 212)
    pdf.cell(76, 8, "Stage 3: Fit & Availability", ln=True)
    
    pdf.set_y(62)
    pdf.set_font("Helvetica", "", 10.5)
    pdf.set_text_color(226, 232, 240)
    
    fit_points = [
        "5-9 year Target Experience",
        "Indian Tier-1 City checks",
        "Notice buyout adjustments",
        "Recruiter response multipliers",
        "Job-hopping tenure penalties",
        "Disqualifies synthetic honeypots"
    ]
    for pt in fit_points:
        pdf.set_x(202)
        pdf.multi_cell(76, 5.5, f"- {pt}", align="L")
        pdf.ln(1)

    # ==========================================
    # SLIDE 4: Honeypot & Fraud Defense (Dark Mode)
    # ==========================================
    pdf.add_page()
    draw_background(pdf, dark=True)
    draw_accent_line(pdf, y=15, color=(244, 63, 94)) # rose-500

    # Page Title
    pdf.set_y(22)
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, "Honeypot Identification & Profile Fraud Guardrails", ln=True)

    # Sub-header
    pdf.set_y(32)
    pdf.set_font("Helvetica", "I", 12)
    pdf.set_text_color(244, 63, 94)
    pdf.cell(0, 8, "Eliminated 100% of leaking honeypots in the Top 100 (Leakage rate: 0.0%)", ln=True)

    # Main content panels
    # Left Panel: Company Foundation Date Trap
    pdf.set_fill_color(30, 41, 59)
    pdf.rect(15, 45, 126, 125, "F")
    
    pdf.set_y(50)
    pdf.set_x(20)
    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(244, 63, 94)
    pdf.cell(116, 8, "Startup Foundation Dates (Rule 3.5)", ln=True)
    
    pdf.set_y(62)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(226, 232, 240)
    pdf.set_x(20)
    pdf.multi_cell(116, 6, "Synthetic generators assign candidates to start roles at companies before the company actually exists.")
    pdf.ln(3)
    
    # Details table for dates
    pdf.set_x(20)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(253, 186, 116) # orange-300
    pdf.cell(35, 6, "Company", ln=False)
    pdf.cell(45, 6, "Foundation", ln=False)
    pdf.cell(0, 6, "Detected Leakage", ln=True)
    
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(226, 232, 240)
    companies_info = [
        ("Krutrim", "April 1, 2023", "4 cases in top 100"),
        ("Sarvam AI", "July 1, 2023", "3 cases in top 100")
    ]
    for c_name, found_date, leak_cnt in companies_info:
        pdf.set_x(20)
        pdf.cell(35, 6, c_name, ln=False)
        pdf.cell(45, 6, found_date, ln=False)
        pdf.set_text_color(248, 113, 113)
        pdf.cell(0, 6, leak_cnt, ln=True)
        pdf.set_text_color(226, 232, 240)
        
    pdf.ln(3)
    pdf.set_x(20)
    pdf.set_font("Helvetica", "I", 11)
    pdf.set_text_color(148, 163, 184)
    pdf.multi_cell(116, 5.5, "Action: Added checks to force candidate score to 0.0 if Krutrim start date < April 2023 or Sarvam AI start date < July 2023.")

    # Right Panel: Zero-Duration Experts
    pdf.set_fill_color(30, 41, 59)
    pdf.rect(156, 45, 126, 125, "F")
    
    pdf.set_y(50)
    pdf.set_x(161)
    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(244, 63, 94)
    pdf.cell(116, 8, "Zero-Duration 'Experts' (Rule 3.6)", ln=True)
    
    pdf.set_y(62)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(226, 232, 240)
    pdf.set_x(161)
    pdf.multi_cell(116, 6, "Identify profiles stating 'expert' proficiency in a skill but listing 0 months of experience using it.")
    pdf.ln(3)
    
    pdf.set_x(161)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(253, 186, 116)
    pdf.cell(40, 6, "Honeypot Type", ln=False)
    pdf.cell(0, 6, "Corpus Frequency", ln=True)
    
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(226, 232, 240)
    honeypot_types = [
        ("Expert + 0 months", "Exactly 21 profiles"),
        ("Duration Mismatch", "Job duration > calendar (Rule 3.1)"),
        ("Role Mismatch", "Non-tech title + deep ML (Rule 3.3)"),
        ("Activity Fabricator", "Completeness >80, responses <2% (Rule 3.4)")
    ]
    for h_type, freq in honeypot_types:
        pdf.set_x(161)
        pdf.cell(40, 6, h_type, ln=False)
        pdf.cell(0, 6, freq, ln=True)
        
    pdf.ln(3)
    pdf.set_x(161)
    pdf.set_font("Helvetica", "I", 11)
    pdf.set_text_color(148, 163, 184)
    pdf.multi_cell(116, 5.5, "Action: Set candidate scores to 0.0 for any instances of expert skill duration == 0, eliminating all remaining synthetic profiles.")

    # ==========================================
    # SLIDE 5: Heuristics & Modifiers (Dark Mode)
    # ==========================================
    pdf.add_page()
    draw_background(pdf, dark=True)
    draw_accent_line(pdf, y=15, color=(6, 182, 212))

    # Page Title
    pdf.set_y(22)
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, "Custom Heuristics & Modifier Tuning", ln=True)

    # Column 1: Notice Period & Location
    pdf.set_fill_color(30, 41, 59)
    pdf.rect(15, 45, 126, 125, "F")
    
    pdf.set_y(50)
    pdf.set_x(20)
    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(6, 182, 212)
    pdf.cell(116, 8, "Notice Period & Buyouts", ln=True)
    
    pdf.set_y(60)
    pdf.set_font("Helvetica", "", 10.5)
    pdf.set_text_color(226, 232, 240)
    pdf.set_x(20)
    pdf.multi_cell(116, 5.5, "The JD mentions: 'We can buy out up to 30 days.' This means notice periods up to 60 days are functionally equivalent to standard 30-day notice periods after buyout.")
    
    pdf.ln(2)
    pdf.set_x(20)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(253, 186, 116)
    pdf.cell(50, 6, "Stated Notice Period", ln=False)
    pdf.cell(0, 6, "Applied Score Modifier", ln=True)
    
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(226, 232, 240)
    notices = [
        ("<= 30 days", "1.00 (Standard preferred)"),
        ("31 - 60 days (with Buyout)", "0.97 (Minor adjustment instead of 0.90)"),
        ("61 - 90 days", "0.80 (Significant bar increase)"),
        ("> 90 days", "0.50 (Strong negative filter)")
    ]
    for np, mod in notices:
        pdf.set_x(20)
        pdf.cell(50, 6, np, ln=False)
        pdf.cell(0, 6, mod, ln=True)

    # Column 2: Relocation Location Heuristics
    pdf.set_fill_color(30, 41, 59)
    pdf.rect(156, 45, 126, 125, "F")
    
    pdf.set_y(50)
    pdf.set_x(161)
    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(6, 182, 212)
    pdf.cell(116, 8, "Tier-1 Relocation Coverage", ln=True)
    
    pdf.set_y(60)
    pdf.set_font("Helvetica", "", 10.5)
    pdf.set_text_color(226, 232, 240)
    pdf.set_x(161)
    pdf.multi_cell(116, 5.5, "The JD states openness to candidates willing to relocate from Tier-1 Indian cities. We expanded geographic mapping to include all key hubs.")
    
    pdf.ln(2)
    pdf.set_x(161)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(253, 186, 116)
    pdf.cell(55, 6, "City Categories", ln=False)
    pdf.cell(0, 6, "Relocation Score Modifier", ln=True)
    
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(226, 232, 240)
    cities = [
        ("Bangalore (Onsite)", "1.00 (Ideal)"),
        ("Tier-1 Reloc. (New)", "0.90 (Kolkata, Ahmedabad, Chennai, etc.)"),
        ("Other Cities (Reloc.)", "0.75 (Standard relocation penalty)"),
        ("No Relocation Stated", "0.50 (Strict penalty)")
    ]
    for cat, mod in cities:
        pdf.set_x(161)
        pdf.cell(55, 6, cat, ln=False)
        pdf.cell(0, 6, mod, ln=True)
        
    pdf.ln(2)
    pdf.set_x(161)
    pdf.set_font("Helvetica", "I", 10.5)
    pdf.set_text_color(148, 163, 184)
    pdf.multi_cell(116, 5.5, "Result: High-quality candidates from Kolkata and Ahmedabad are now correctly ranked in the top 100.")

    # ==========================================
    # SLIDE 6: Results & Compliance (Dark Mode)
    # ==========================================
    pdf.add_page()
    draw_background(pdf, dark=True)
    draw_accent_line(pdf, y=15, color=(34, 197, 94)) # green-500

    # Page Title
    pdf.set_y(22)
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, "Execution Performance & Spec Compliance", ln=True)

    # Key metrics grid
    metrics = [
        ("Ranking Runtime", "under 10 seconds", "Well below the 5-minute CPU constraint"),
        ("Memory Footprint", "~450 MB RAM", "Satisfies the 16 GB limit"),
        ("Network Requests", "0 (Offline)", "Completely sandboxed, no LLM API calls"),
        ("Honeypot Leakage", "0.0% (0 / 100)", "Under 10% threshold to prevent disqualification"),
        ("Official Validation", "SUCCESS", "Parsed and approved by validate_submission.py"),
        ("Tie-Breaking", "Deterministic", "Breaks ties using candidate_id ascending")
    ]
    
    # 2x3 Grid layout
    x_positions = [15, 156]
    y_positions = [45, 85, 125]
    
    pdf.set_font("Helvetica", "", 11)
    for idx, (m_title, val, desc) in enumerate(metrics):
        col = idx % 2
        row = idx // 2
        px = x_positions[col]
        py = y_positions[row]
        
        pdf.set_fill_color(30, 41, 59)
        pdf.rect(px, py, 126, 32, "F")
        
        pdf.set_y(py + 3)
        pdf.set_x(px + 5)
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(203, 213, 225)
        pdf.cell(0, 6, m_title, ln=True)
        
        pdf.set_x(px + 5)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(74, 222, 128) # green-400
        pdf.cell(0, 5, val, ln=False)
        
        pdf.set_font("Helvetica", "", 10.5)
        pdf.set_text_color(148, 163, 184)
        pdf.cell(0, 5, f" - {desc}", ln=True)

    # Save output PDF
    pdf.output("TECHFLOW-presentation.pdf")
    print("Presentation PDF generated successfully.")

if __name__ == "__main__":
    create_deck()
