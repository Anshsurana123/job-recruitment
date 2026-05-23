import json
import random
import datetime
from pathlib import Path

# Set random seed for deterministic generation
random.seed(42)

# Names pool
FIRST_NAMES = [
    "Aarav", "Aditi", "Amit", "Ananya", "Arjun", "Neha", "Rahul", "Priya", "Rohan", "Siddharth",
    "John", "Emily", "Michael", "Sarah", "David", "Jessica", "James", "Emma", "Daniel", "Olivia",
    "Alex", "Sophia", "Ryan", "Isabella", "Matthew", "Mia", "Tyler", "Charlotte", "Andrew", "Amelia",
    "Kenji", "Yuki", "Hiroshi", "Sakura", "Min-jun", "Ji-woo", "Wei", "Li", "Lei", "Fang",
    "Carlos", "Sofia", "Alejandro", "Isabella", "Mateo", "Camila", "Diego", "Valentina", "Lucas", "Lucia"
]

LAST_NAMES = [
    "Sharma", "Verma", "Gupta", "Patel", "Mehta", "Singh", "Reddy", "Nair", "Rao", "Joshi",
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Garcia", "Rodriguez", "Wilson",
    "Sato", "Suzuki", "Takahashi", "Kim", "Lee", "Park", "Chen", "Wang", "Zhang", "Liu",
    "Silva", "Santos", "Oliveira", "Souza", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Perez"
]

COMPANIES = [
    "TechGiant Corp", "CloudScale Systems", "WebFlow Technologies", "NeuralNet Labs", "DataPrism Inc",
    "ByteDance Solutions", "Apex Fintech", "DevOpsFlow", "Streamline Media", "LogiTech Solutions",
    "CyberGuard LLC", "GreenEnergy Corp", "Innova Health", "EduLearn Tech", "RetailSync Systems"
]

UNIVERSITIES = [
    "Stanford University", "Massachusetts Institute of Technology", "UC Berkeley", "Carnegie Mellon University",
    "Indian Institute of Technology, Bombay", "Indian Institute of Technology, Delhi", "University of Waterloo",
    "University of Oxford", "Tsinghua University", "National University of Singapore",
    "University of Toronto", "Georgia Institute of Technology", "UT Austin", "University of Illinois Urbana-Champaign"
]

SKILLS_POOL = {
    "Frontend": ["React", "TypeScript", "JavaScript", "HTML5", "CSS3", "Next.js", "Redux", "GraphQL", "TailwindCSS", "Webpack", "Vite", "WebAssembly", "Svelte", "Vue.js", "Jest", "Cypress"],
    "Backend": ["Python", "Node.js", "Go", "Java", "Express.js", "FastAPI", "Django", "Spring Boot", "PostgreSQL", "MongoDB", "Redis", "Elasticsearch", "gRPC", "Docker", "REST APIs", "SQL", "RabbitMQ"],
    "Machine Learning": ["Python", "PyTorch", "TensorFlow", "Scikit-Learn", "pandas", "NumPy", "MLflow", "Kubeflow", "transformers", "Hugging Face", "LLMs", "NLP", "Computer Vision", "LangChain", "CUDA"],
    "DevOps": ["AWS", "Terraform", "Kubernetes", "Docker", "CI/CD", "GitHub Actions", "Prometheus", "Grafana", "Linux", "Bash", "Ansible", "Helm", "CloudFormation", "GCP", "Azure", "Nginx"]
}

TITLES_POOL = {
    "Frontend": ["Frontend Engineer", "React Developer", "UI/UX Engineer", "Web Developer"],
    "Backend": ["Backend Engineer", "Software Engineer - Backend", "Distributed Systems Engineer", "API Developer"],
    "Machine Learning": ["ML Engineer", "Data Scientist", "AI Researcher", "NLP Specialist", "Deep Learning Engineer"],
    "DevOps": ["DevOps Engineer", "Site Reliability Engineer", "Cloud Architect", "Infrastructure Engineer"]
}

LOCATIONS = [
    "San Francisco, CA", "New York, NY", "Seattle, WA", "Austin, TX", "Boston, MA",
    "Bangalore, India", "London, UK", "Toronto, Canada", "Berlin, Germany", "Tokyo, Japan",
    "Remote, USA", "Remote, India", "Chicago, IL", "Denver, CO", "London, UK"
]

# Generate random past dates
def get_random_date_range(years_back_start, years_back_end):
    today = datetime.date(2026, 5, 20)
    start_days_ago = random.randint(int(years_back_end * 365), int(years_back_start * 365))
    duration_days = random.randint(300, 1200)
    
    start_date = today - datetime.timedelta(days=start_days_ago)
    end_date = start_date + datetime.timedelta(days=duration_days)
    
    if end_date >= today:
        end_date_str = "Present"
    else:
        end_date_str = end_date.isoformat()
        
    return start_date.isoformat(), end_date_str

def get_last_active_date(cohort):
    today = datetime.date(2026, 5, 20)
    if cohort == "Fresh":
        # Updated in last 7 days
        days_ago = random.randint(0, 7)
    elif cohort == "Recent":
        # Updated in last 60 days
        days_ago = random.randint(8, 60)
    elif cohort == "Dormant":
        # Updated more than 1 year ago (up to 3 years)
        days_ago = random.randint(366, 1000)
    else: # Normal distribution
        choices = [
            (random.randint(0, 7), 0.15),      # 15% Active Now
            (random.randint(8, 60), 0.35),     # 35% Recent
            (random.randint(61, 365), 0.30),    # 30% Older
            (random.randint(366, 1000), 0.20)  # 20% Dormant
        ]
        # Weighted selection
        r = random.random()
        cumulative = 0
        days_ago = 100
        for val, weight in choices:
            cumulative += weight
            if r <= cumulative:
                days_ago = val
                break
                
    active_date = today - datetime.timedelta(days=days_ago)
    return active_date.isoformat()

def generate_education(years_experience):
    today_year = 2026
    start_year = today_year - int(years_experience) - 4
    degrees = ["B.S.", "M.S.", "Ph.D."]
    weights = [0.70, 0.25, 0.05]
    degree = random.choices(degrees, weights=weights)[0]
    
    fields = ["Computer Science", "Software Engineering", "Electrical Engineering", "Data Science", "Information Technology"]
    field = random.choice(fields)
    institution = random.choice(UNIVERSITIES)
    
    edu = [{
        "degree": degree,
        "field": field,
        "institution": institution,
        "year": start_year + 4
    }]
    
    # Add a second degree (e.g. Master's after BS) for some candidates
    if degree == "M.S." and random.random() > 0.5:
        edu.insert(0, {
            "degree": "B.S.",
            "field": "Computer Science",
            "institution": random.choice(UNIVERSITIES),
            "year": start_year
        })
        
    return edu

# Resume text narratives generator based on cohort
def generate_resume_narrative(role, cohort, name, current_title, skills, experience):
    if cohort == "Keyword Stuffer":
        # Under 200 words, dense skill listings, generic buzzwords
        skills_str = ", ".join(skills + random.sample(SKILLS_POOL["Frontend"] + SKILLS_POOL["Backend"] + SKILLS_POOL["DevOps"] + SKILLS_POOL["Machine Learning"], 10))
        return (
            f"PROFESSIONAL RESUME OF {name.upper()}\n"
            f"Current Position: {current_title}\n"
            f"Objective: Seeking a challenging role as a senior technology leader to leverage my extensive technical skill set.\n\n"
            f"CORE SKILLS & TECHNOLOGIES:\n{skills_str}\n\n"
            f"EXPERIENCE SUMMARY:\n"
            f"- Highly skilled professional developer with over {experience:.1f} years of experience in the software industry.\n"
            f"- Proficient in agile methodologies, team collaboration, systems analysis, and software development lifecycle (SDLC).\n"
            f"- Experienced in building scalable web applications, RESTful APIs, databases, cloud architecture, and serverless computing.\n"
            f"- Dedicated to clean code, test-driven development, continuous integration, and delivering high quality projects on time.\n"
            f"Contact for details. References available upon request."
        )
        
    # High-quality narratives detailing HOW and WHY with specific metrics
    metrics = [
        "reducing API latency by 45% using Redis caching and query optimization",
        "improving Web Vitals (LCP) by 60% via code splitting, dynamic imports, and image lazy loading",
        "slashing CI/CD deployment pipeline execution time from 22 minutes to 5.5 minutes using GitHub Actions runner caching",
        "scaling distributed search services from 10k to 150k concurrent requests using Elasticsearch cluster sharding",
        "migrating a legacy monolithic codebase to Dockerized microservices, reducing infrastructure costs by 35%",
        "implementing robust telemetry (Prometheus/Grafana) that reduced Mean Time to Resolution (MTTR) by 50%",
        "training a custom BERT model that increased customer intent classification accuracy from 81% to 94.3%",
        "architecting a module-federated micro-frontend framework, allowing 5 independent teams to deploy concurrently"
    ]
    
    architectures = [
        "module federation and micro-frontends to isolate domain-specific UI deployments",
        "event-driven architecture using Apache Kafka to orchestrate asynchronous microservices",
        "domain-driven design principles combined with CQRS to scale write and read database workloads independently",
        "secure, zero-trust cloud network architecture on AWS using private subnets, VPC peering, and IAM service roles",
        "React Server Components (RSC) and streaming SSR to bypass client-side JS hydrations costs"
    ]
    
    challenges = [
        "We faced a severe database connection pool exhaustion failure during traffic spikes. I resolved this by rewriting our queries to avoid expensive N+1 joins, introducing PgBouncer connection pooling, and establishing read replicas.",
        "A critical memory leak and hydration mismatch caused performance degradation in production. I analyzed the memory profiles using Chrome DevTools, traced the leak to an uncleaned window event listener in a custom React hook, and refactored it safely.",
        "We hit a wall with CPU bottlenecks during real-time ML inference. I resolved this by compiling the model to TensorRT, batching requests dynamically using Triton Inference Server, and leveraging CUDA stream concurrency.",
        "Our deployments were highly brittle and frequently failed due to configuration drift across environments. I established a declarative GitOps pipeline using ArgoCD and Terraform, unifying dev/prod parity."
    ]
    
    role_desc = ""
    if role == "Frontend":
        role_desc = (
            f"Led frontend architecture migrations, focusing heavily on modern rendering strategies. "
            f"Authored reusable design systems in TypeScript using strict generics for maximum type safety. "
            f"Designed and built our primary application using {random.choice(architectures)}, which succeeded in {random.choice(metrics)}. "
            f"Deeply knowledgeable in DOM reconciliation, avoiding stale closures in custom React hooks, and eliminating render-blocking resources. "
            f"\n\nTechnical Problem Solved:\n{random.choice(challenges)}"
        )
    elif role == "Backend":
        role_desc = (
            f"Architected and maintained highly concurrent distributed backend services processing billions of events. "
            f"Designed our backend services using {random.choice(architectures)}, successfully {random.choice(metrics)}. "
            f"Expert in database schema migrations, distributed locking using Redis, and profiling application CPU/memory bottlenecks. "
            f"Championed clean, test-driven development practices (Jest, PyTest, Go Test) maintaining over 92% code coverage. "
            f"\n\nTechnical Problem Solved:\n{random.choice(challenges)}"
        )
    elif role == "Machine Learning":
        role_desc = (
            f"Developed, trained, and productionized robust machine learning pipelines. "
            f"Led model architecture design utilizing {random.choice(architectures)}, yielding substantial improvement: {random.choice(metrics)}. "
            f"Deeply familiar with PyTorch internals, custom CUDA kernel optimization, and setting up automated drift monitoring pipelines (MLflow). "
            f"Successfully designed, fine-tuned, and deployed LLM agents to production while managing context window budgets and retrieval latency. "
            f"\n\nTechnical Problem Solved:\n{random.choice(challenges)}"
        )
    elif role == "DevOps":
        role_desc = (
            f"Architected resilient, self-healing, multi-region cloud infrastructures. "
            f"Designed our infrastructure state using {random.choice(architectures)}, resulting in {random.choice(metrics)}. "
            f"Automated all infrastructure provisioning via Terraform and Helm charts. "
            f"Set up comprehensive service meshes (Istio) and implemented rigorous canary deployments with automated rollbacks. "
            f"\n\nTechnical Problem Solved:\n{random.choice(challenges)}"
        )
        
    narrative = (
        f"{name}\n"
        f"Email: {name.lower().replace(' ', '.')}@example.com | Current Role: {current_title}\n"
        f"Location: {random.choice(LOCATIONS)}\n\n"
        f"PROFESSIONAL PROFILE:\n"
        f"Detail-oriented and high-velocity Software Professional with {experience:.1f} years of hands-on experience in modern technology ecosystems. "
        f"Possesses a proven track record of diagnosing deep technical failures, scaling system limits, and driving engineering excellence across teams. "
        f"Believer in documentation, robust architecture, and mentoring junior engineers to elevate engineering standards.\n\n"
        f"KEY ENGINEERING WORK & METRICS:\n"
        f"{role_desc}\n\n"
        f"TECHNICAL SKILLS:\n"
        f"Languages & Frameworks: {', '.join(skills)}\n"
        f"Architectures & Practices: Test-Driven Development (TDD), Microservices, CI/CD pipelines, System Telemetry, Cloud Security, Agile Scrum, API Design."
    )
    
    if cohort == "Hidden Gem":
        # Add extra stellar, lead-level leadership signals and extreme impact metrics
        narrative += (
            f"\n\nLEADERSHIP & ENGINEERING IMPACT:\n"
            f"- Established engineering standards, coding style guidelines, and RFC processes, boosting developer onboarding speed by 40%.\n"
            f"- Mentored and guided 6 junior and mid-level developers, with 3 achieving promotions under my guidance.\n"
            f"- Led cross-functional initiatives to review and deprecate unused cloud assets, saving $120,000 annually in hosting costs."
        )
        
    return narrative

# Generate individual candidate record
def generate_candidate(candidate_id, role, cohort):
    name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    
    # Seniority profile based on cohort
    if cohort == "Seniority Mismatch":
        years_experience = round(random.uniform(0.5, 1.8), 1)
        # Give them a high title despite lack of experience
        current_title = f"Senior {random.choice(TITLES_POOL[role])}"
    elif cohort == "Hidden Gem":
        # Exceptional engineers: fast growth (young, high velocity)
        years_experience = round(random.uniform(2.5, 6.0), 1)
        current_title = f"Lead {random.choice(TITLES_POOL[role])}"
    else:
        years_experience = round(random.uniform(1.0, 15.0), 1)
        if years_experience < 2:
            current_title = f"Junior {random.choice(TITLES_POOL[role])}"
        elif years_experience < 5:
            current_title = random.choice(TITLES_POOL[role])
        elif years_experience < 8:
            current_title = f"Senior {random.choice(TITLES_POOL[role])}"
        else:
            current_title = f"Lead {random.choice(TITLES_POOL[role])}"
            
    # Add staff/principal options randomly for very experienced
    if years_experience > 10 and cohort != "Seniority Mismatch" and random.random() > 0.6:
        current_title = f"Principal {random.choice(TITLES_POOL[role])}"
        
    # Skills listed
    base_skills = random.sample(SKILLS_POOL[role], min(len(SKILLS_POOL[role]), random.randint(5, 10)))
    if cohort == "Keyword Stuffer":
        # Overflow with skills
        skills_listed = list(set(base_skills + random.sample(SKILLS_POOL["Frontend"] + SKILLS_POOL["Backend"] + SKILLS_POOL["DevOps"] + SKILLS_POOL["Machine Learning"], 22)))
    else:
        skills_listed = base_skills
        
    # Last active updated timestamp
    last_active = get_last_active_date(cohort)
    # Edge case: 5% of candidates have null last_active
    if random.random() < 0.05:
        last_active = None
        
    # Education
    education = generate_education(years_experience)
    
    # Career History
    career_history = []
    # Calculate how many positions based on years_experience
    if years_experience < 2:
        num_positions = 1
    elif years_experience < 5:
        num_positions = 2
    else:
        num_positions = random.randint(3, 5)
        
    # Cohort Edge Case: Missing career history (fewer than 2 entries)
    if cohort == "Missing Career History":
        num_positions = 1
        
    remaining_exp = years_experience
    current_year = 2026.4  # May 2026
    
    titles_list = ["Junior " + random.choice(TITLES_POOL[role]), random.choice(TITLES_POOL[role]), "Senior " + random.choice(TITLES_POOL[role])]
    if cohort == "Hidden Gem":
        # Rapid progression: Intern -> Mid -> Senior -> Lead
        titles_list = ["Intern", "Junior " + random.choice(TITLES_POOL[role]), random.choice(TITLES_POOL[role]), "Lead " + random.choice(TITLES_POOL[role])]
        num_positions = min(len(titles_list), num_positions)
        
    for i in range(num_positions):
        # Determine position seniority title
        if cohort == "Hidden Gem" and i < len(titles_list):
            title = titles_list[i]
        elif i == 0:
            # Earliest position
            title = "Junior " + random.choice(TITLES_POOL[role]) if "Junior" not in current_title else "Intern"
        elif i == num_positions - 1:
            # Latest position
            title = current_title
        else:
            # Middle positions
            title = random.choice(TITLES_POOL[role])
            
        pos_duration = round(remaining_exp / (num_positions - i), 1)
        remaining_exp -= pos_duration
        
        start_year = current_year - remaining_exp - pos_duration
        end_year = start_year + pos_duration
        
        # Format dates as ISO string (Year-Month-Day approximation)
        start_month = random.randint(1, 12)
        start_date = f"{int(start_year)}-{start_month:02d}-01"
        
        if i == num_positions - 1:
            end_date = "Present"
        else:
            end_month = random.randint(1, 12)
            end_date = f"{int(end_year)}-{end_month:02d}-01"
            
        career_history.insert(0, {
            "title": title,
            "company": random.choice(COMPANIES),
            "start_date": start_date,
            "end_date": end_date
        })
        
    resume_text = generate_resume_narrative(role, cohort, name, current_title, skills_listed, years_experience)
    
    return {
        "candidate_id": candidate_id,
        "name": name,
        "resume_text": resume_text,
        "current_title": current_title,
        "years_experience": float(years_experience),
        "career_history": career_history,
        "skills_listed": skills_listed,
        "last_active": last_active,
        "education": education,
        "location": random.choice(LOCATIONS)
    }

def generate_full_dataset(count=520):
    candidates = []
    
    # Cohorts breakdown:
    # 1. Hidden Gems: ~10% (high velocity, premium metrics)
    # 2. Keyword Stuffers: ~15% (20+ skills, short resume text)
    # 3. Dormant Profiles: ~15% (updated >1 year ago)
    # 4. Fresh Profiles: ~15% (updated last 7 days)
    # 5. Seniority Mismatch: ~10% (under 2 years experience, high titles)
    # 6. Missing Career History: ~5% (1 history entry)
    # 7. Standard Profiles: ~30% (normal mix)
    
    roles = ["Frontend", "Backend", "Machine Learning", "DevOps"]
    
    for i in range(1, count + 1):
        candidate_id = f"cand_{i:03d}"
        role = roles[(i - 1) % len(roles)]
        
        # Decide cohort
        r = random.random()
        if r < 0.10:
            cohort = "Hidden Gem"
        elif r < 0.25:
            cohort = "Keyword Stuffer"
        elif r < 0.40:
            cohort = "Dormant"
        elif r < 0.55:
            cohort = "Fresh"
        elif r < 0.65:
            cohort = "Seniority Mismatch"
        elif r < 0.70:
            cohort = "Missing Career History"
        else:
            cohort = "Standard"
            
        cand = generate_candidate(candidate_id, role, cohort)
        candidates.append(cand)
        
    return candidates

if __name__ == "__main__":
    print("Generating synthetic candidate pool of 520 candidates with high realistic variance...")
    dataset = generate_full_dataset(520)
    
    output_dir = Path(__file__).parent
    output_path = output_dir / "candidates.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
        
    print(f"Dataset successfully written to: {output_path.resolve()}")
    print("Cohorts Breakdown Summary:")
    print("- Created candidates with deep resumes (metrics, how, why, scaling challenges)")
    print("- Created keyword overfitters with 20+ skills but sparse resumes")
    # Count stats
    stuffer_count = sum(1 for c in dataset if len(c["skills_listed"]) >= 20 and len(c["resume_text"].split()) < 200)
    mismatch_count = sum(1 for c in dataset if c["years_experience"] < 2.0 and "Senior" in c["current_title"])
    fresh_count = sum(1 for c in dataset if c["last_active"] and (datetime.date(2026, 5, 20) - datetime.date.fromisoformat(c["last_active"])).days <= 7)
    dormant_count = sum(1 for c in dataset if c["last_active"] and (datetime.date(2026, 5, 20) - datetime.date.fromisoformat(c["last_active"])).days > 365)
    null_active = sum(1 for c in dataset if c["last_active"] is None)
    single_exp = sum(1 for c in dataset if len(c["career_history"]) < 2)
    
    print(f"Total candidates: {len(dataset)}")
    print(f"- Keyword Stuffers (20+ skills, <200 words): {stuffer_count}")
    print(f"- Seniority Mismatch (Senior title, <2 years exp): {mismatch_count}")
    print(f"- Fresh Profiles (Updated <= 7 days ago): {fresh_count}")
    print(f"- Dormant Profiles (Updated > 365 days ago): {dormant_count}")
    print(f"- Null Profile Updates: {null_active}")
    print(f"- Single History Entry: {single_exp}")
