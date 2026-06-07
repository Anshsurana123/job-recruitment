# DEPRECATED: ChromaDB vector indexing is no longer used. The new pipeline reads
# candidates.json directly. Use smart_ingest.py for bulk PDF resume ingestion instead.
# Kept for reference only.
import json
import os
from pathlib import Path

def main():
    print("=== Talent Radar: Ingestion Pipeline (DEPRECATED) ===")
    try:
        from sentence_transformers import SentenceTransformer
        import chromadb
    except ImportError:
        print("[Deprecation Warning] sentence-transformers or chromadb not installed. Ingest is a no-op.")
        return
    
    # Path setup
    base_dir = Path(__file__).parent
    candidates_path = base_dir / "candidates.json"
    chroma_db_path = base_dir / "chroma_db"
    
    # 1. Load candidate data
    if not candidates_path.exists():
        print(f"Error: Candidate dataset not found at {candidates_path.resolve()}. Please run candidates_dataset.py first.")
        return
        
    print(f"Loading candidate records from {candidates_path.resolve()}...")
    with open(candidates_path, "r", encoding="utf-8") as f:
        candidates = json.load(f)
    print(f"Loaded {len(candidates)} candidates.")
    
    # 2. Load the Embedding Model
    print("Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
    # This downloads and initializes the bi-encoder
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Embedding model loaded successfully.")
    
    # 3. Prepare texts for embedding and metadata
    ids = []
    documents = []
    embeddings = []
    metadatas = []
    
    print("Preparing candidate profiles and generating vector embeddings...")
    raw_texts = []
    for cand in candidates:
        cand_id = cand["candidate_id"]
        name = cand["name"]
        title = cand["current_title"]
        resume = cand["resume_text"]
        skills = ", ".join(cand["skills_listed"])
        
        # Pipeline architecture requirement:
        # Concatenate: name + current_title + resume_text + skills_listed.join(", ")
        text_to_embed = f"{name} {title} {resume} {skills}"
        
        ids.append(cand_id)
        documents.append(resume) # The actual resume content goes as document
        raw_texts.append(text_to_embed)
        
        # ChromaDB metadata must be flat: strings, ints, floats, bools.
        # So we serialize complex fields (like career_history) as JSON string.
        last_active = cand["last_active"] if cand["last_active"] is not None else "null"
        
        metadatas.append({
            "candidate_id": cand_id,
            "name": name,
            "current_title": title,
            "years_experience": float(cand["years_experience"]),
            "last_active": last_active,
            "career_history": json.dumps(cand["career_history"]),
            "skills_listed": json.dumps(cand["skills_listed"])
        })
        
    # Generate embeddings in batches for efficiency
    print("Generating dense vector representations (384-dimensions)...")
    encoded_vectors = model.encode(raw_texts, show_progress_bar=True)
    embeddings = [vector.tolist() for vector in encoded_vectors]
    print(f"Generated {len(embeddings)} embeddings successfully.")
    
    # 4. Populate ChromaDB
    print(f"Initializing persistent ChromaDB client at {chroma_db_path.resolve()}...")
    db_client = chromadb.PersistentClient(path=str(chroma_db_path))
    
    # Clean existing collection to support running ingest multiple times clean
    collection_name = "candidates_pool"
    try:
        db_client.delete_collection(name=collection_name)
        print(f"Cleared existing collection '{collection_name}'.")
    except Exception:
        pass # Collection didn't exist
        
    print(f"Creating collection '{collection_name}' with Cosine distance metric...")
    collection = db_client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Add to ChromaDB
    print(f"Indexing {len(candidates)} candidates into vector store...")
    # Chroma has limits on batch size, but 520 candidates comfortably fits in a single write.
    if ids:
        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
    else:
        print("No candidates to index in ChromaDB. Vector store created empty.")
    
    print("\nIngestion pipeline completed successfully!")
    print(f"ChromaDB persistent files saved. Total index count: {collection.count()}")

if __name__ == "__main__":
    main()
