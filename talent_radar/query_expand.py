import os
import re

# Premium dictionary mapping for high-fidelity fallback query expansion
SEMANTIC_DOMAINS = {
    "Frontend": {
        "architectures": ["micro-frontend", "module federation", "island architecture", "component-driven architecture"],
        "debugging": ["hydration mismatch", "reconciliation", "stale closure", "memory leak", "Chrome DevTools", "layout shift"],
        "performance": ["bundle splitting", "LCP", "tree-shaking", "lazy loading", "code splitting", "critical rendering path"],
        "adjacent": ["TypeScript generics", "React Server Components", "streaming SSR", "state management", "Next.js routing"]
    },
    "Backend": {
        "architectures": ["event-driven architecture", "domain-driven design", "CQRS", "microservices", "serverless", "distributed systems"],
        "debugging": ["connection pool exhaustion", "N+1 query join", "deadlocks", "race conditions", "memory leak", "CPU bottleneck"],
        "performance": ["connection pooling", "indexing", "caching", "query optimization", "read replicas", "load balancing"],
        "adjacent": ["gRPC", "asyncio", "concurrency", "distributed locking", "message queues", "RabbitMQ", "Kafka", "PostgreSQL", "Redis"]
    },
    "Machine Learning": {
        "architectures": ["ML pipelines", "feature store", "model registry", "vector databases", "RAG systems", "deep learning architecture"],
        "debugging": ["gradient explosion", "overfitting", "concept drift", "CUDA out of memory", "tensor dimension mismatch"],
        "performance": ["model quantization", "TensorRT", "batch inference", "GPU memory budget", "distributed training"],
        "adjacent": ["LLM fine-tuning", "LangChain", "embedding vectors", "reinforcement learning", "neural networks", "PyTorch", "transformers"]
    },
    "DevOps": {
        "architectures": ["infrastructure as code (IaC)", "declarative gitops", "service mesh", "high availability", "multi-region failover"],
        "debugging": ["configuration drift", "pod crashloopbackoff", "network partition", "resource throttling", "liveness probe failure"],
        "performance": ["auto-scaling", "resource budget limits", "caching proxy", "load balancer latency"],
        "adjacent": ["Terraform modules", "Istio", "Helm charts", "Prometheus metrics", "Grafana dashboards", "ArgoCD", "Kubernetes", "AWS"]
    }
}

BEHAVIORAL_SIGNALS = {
    "Junior": ["learning", "exposure to", "worked under", "assisted in", "monitored"],
    "Senior": ["designed", "led", "owned", "migrated", "reduced latency by", "scaled system limits", "diagnosed production issues"],
    "Lead": ["mentored", "architected", "defined standards", "drove adoption", "RFC processes", "established engineering standards", "guided junior developers"]
}

def rule_based_expansion(job_description, seniority_level="Senior"):
    jd_lower = job_description.lower()
    
    # 1. Identify domain
    matched_domains = []
    
    frontend_keywords = ["react", "frontend", "ui", "ux", "web", "html", "css", "client", "next.js", "svelte", "vue"]
    backend_keywords = ["backend", "node", "django", "fastapi", "go", "java", "api", "database", "postgres", "redis", "mongodb", "server"]
    ml_keywords = ["ml", "machine learning", "ai", "data science", "pytorch", "tensorflow", "llm", "deep learning", "nlp", "computer vision"]
    devops_keywords = ["devops", "sre", "cloud", "aws", "kubernetes", "terraform", "infrastructure", "docker", "pipeline"]
    
    def match_word(keywords, text):
        for k in keywords:
            pattern = rf"(?<!\w){re.escape(k)}(?!\w)"
            if re.search(pattern, text):
                return True
        return False
    if match_word(frontend_keywords, jd_lower):
        matched_domains.append("Frontend")
    if match_word(backend_keywords, jd_lower):
        matched_domains.append("Backend")
    if match_word(ml_keywords, jd_lower):
        matched_domains.append("Machine Learning")
    if match_word(devops_keywords, jd_lower):
        matched_domains.append("DevOps")
        
    # Default to Backend if no domain detected
    if not matched_domains:
        matched_domains.append("Backend")
        
    # 2. Extract core keywords from JD for reinforcement
    # Pull out words that look like technologies (capitalized or common terms)
    tech_patterns = r"\b(React|TypeScript|Python|Node\.js|Go|Kubernetes|AWS|Terraform|Docker|PostgreSQL|Redis|FastAPI|PyTorch|Git|Django|Java|Svelte|Vue|GraphQL)\b"
    tech_matches = list(set(re.findall(tech_patterns, job_description, re.IGNORECASE)))
    
    # 3. Compile Expansion Parts
    expansion_parts = []
    
    # Add domain details
    for domain in matched_domains:
        domain_dict = SEMANTIC_DOMAINS[domain]
        expansion_parts.append(f"Domain: {domain}")
        expansion_parts.append(f"Architectures: {', '.join(domain_dict['architectures'])}")
        expansion_parts.append(f"Debugging & Failure Modes: {', '.join(domain_dict['debugging'])}")
        expansion_parts.append(f"Performance Optimization: {', '.join(domain_dict['performance'])}")
        expansion_parts.append(f"Adjacent Expertise: {', '.join(domain_dict['adjacent'])}")
        
    # Add behavioral signals based on seniority
    seniority = seniority_level.title()
    if seniority in ["Lead", "Principal"]:
        signals = BEHAVIORAL_SIGNALS["Lead"] + BEHAVIORAL_SIGNALS["Senior"]
    elif seniority == "Senior":
        signals = BEHAVIORAL_SIGNALS["Senior"]
    else: # Junior / Mid
        signals = BEHAVIORAL_SIGNALS["Junior"]
        
    expansion_parts.append(f"Behavioral & Seniority Signals ({seniority}): {', '.join(signals)}")
    
    if tech_matches:
        expansion_parts.append(f"Core Tech Stack: {', '.join(tech_matches)}")
        
    expanded_query = "\n".join(expansion_parts)
    return expanded_query

def expand_query(job_description, seniority_level="Senior"):
    print("Executing Step 1: Query Explosion...")
    
    # Check for API keys to see if we can use an LLM
    openai_key = os.environ.get("OPENAI_API_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            print("Using OpenAI for query expansion...")
            prompt = get_llm_prompt(job_description, seniority_level)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI expansion failed: {e}. Falling back to rule-based parser.")
            
    elif gemini_key:
        try:
            # Try official Google GenAI SDK
            from google import genai
            print("Using official Google GenAI SDK for query expansion...")
            client = genai.Client(api_key=gemini_key)
            prompt = get_llm_prompt(job_description, seniority_level)
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            print(f"Gemini expansion failed: {e}. Falling back to rule-based parser.")
            
    # Fallback to premium local rule-based system
    print("Using elite local semantic expansion dictionary.")
    return rule_based_expansion(job_description, seniority_level)

def get_llm_prompt(job_description, seniority_level):
    return f"""
Given the following Job Description (JD) and Seniority Level, generate an EXPANDED SEMANTIC QUERY.
This query is used in a vector search system to find elite technical candidates.
Do NOT just list the same skills from the JD verbatim. Think like a senior engineering director who has interviewed 500+ candidates.

JD:
{job_description}

Seniority Level: {seniority_level}

Requirements for the Expanded Query:
1. Identify the core technology stack and engineering domain (e.g., "React frontend engineer").
2. Add advanced/expert-level terminology that a TRUE expert would use in their resume:
   - Architectural patterns (e.g., "micro-frontend", "module federation", "island architecture", "CQRS", "DDD")
   - Failure modes & debugging knowledge (e.g., "hydration mismatch", "reconciliation", "stale closure", "connection pool exhaustion", "N+1 joins")
   - Performance concerns (e.g., "bundle splitting", "LCP", "tree-shaking", "indexing", "caching")
   - Adjacent deep expertise (e.g., "TypeScript generics", "React Server Components", "streaming SSR", "gRPC", "asyncio")
3. Include role-level behavioral signals:
   - Junior signals: "learning", "exposure to", "worked under"
   - Senior signals: "designed", "led", "owned", "migrated", "reduced latency by"
   - Lead/Principal signals: "mentored", "architected", "defined standards", "drove adoption"

Output ONLY the final expanded query string. Keep it concise, structured, and signal-rich.
"""

if __name__ == "__main__":
    test_jd = "Looking for a React developer who knows TypeScript and can optimize front-end performance."
    print("Testing query expansion:")
    expanded = expand_query(test_jd, "Senior")
    print(expanded)
