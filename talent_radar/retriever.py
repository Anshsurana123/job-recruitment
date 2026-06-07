# DEPRECATED: This module (ChromaDB vector retrieval) has been replaced by
# the SwarmEvaluator in matrix_pipeline.py. Not imported anywhere in the active pipeline.
# Kept for reference only.
import time
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

class CandidateRetriever:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.chroma_db_path = self.base_dir / "chroma_db"
        
        print("Initializing Candidate Retriever...")
        # Load the persistent db client
        self.db_client = chromadb.PersistentClient(path=str(self.chroma_db_path))
        self.collection_name = "candidates_pool"
        try:
            self.collection = self.db_client.get_collection(name=self.collection_name)
        except Exception:
            # Create if it does not exist to support clean start
            self.collection = self.db_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        
        # Load embedding model
        print("Loading retriever embedding model (all-MiniLM-L6-v2)...")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Candidate Retriever ready.")

    def retrieve(self, expanded_query, top_k=50):
        print(f"Executing Step 2: Dense Vector Retrieval (top_k={top_k})...")
        start_time = time.time()
        
        # 1. Embed the expanded query
        query_vector = self.embedding_model.encode(expanded_query).tolist()
        
        # 2. Query ChromaDB
        try:
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=top_k
            )
        except Exception as e:
            print(f"ChromaDB query failed: {e}. Attempting to re-fetch collection handle...")
            try:
                self.collection = self.db_client.get_collection(name=self.collection_name)
                results = self.collection.query(
                    query_embeddings=[query_vector],
                    n_results=top_k
                )
                print("Successfully recovered collection handle and completed query.")
            except Exception as err:
                print(f"Recovery failed: {err}")
                raise err
        
        duration_ms = (time.time() - start_time) * 1000
        print(f"Vector search completed in {duration_ms:.1f}ms (Target: <100ms).")
        
        # Parse and return matches
        candidates_retrieved = []
        
        if not results or not results["ids"] or len(results["ids"][0]) == 0:
            print("No candidates retrieved from vector store.")
            return candidates_retrieved
            
        ids = results["ids"][0]
        metadatas = results["metadatas"][0]
        documents = results["documents"][0]
        distances = results["distances"][0]
        
        for idx in range(len(ids)):
            meta = metadatas[idx]
            doc = documents[idx]
            dist = distances[idx]
            
            # Reconstruct original structured representation from flat metadata
            skills_listed = json.loads(meta["skills_listed"]) if "skills_listed" in meta else []
            career_history = json.loads(meta["career_history"]) if "career_history" in meta else []
            
            # Handle null values stored as strings in ChromaDB
            last_active = meta["last_active"]
            if last_active == "null":
                last_active = None
                
            candidate = {
                "candidate_id": ids[idx],
                "name": meta["name"],
                "resume_text": doc, # The document field stores the resume_text
                "current_title": meta["current_title"],
                "years_experience": float(meta["years_experience"]),
                "career_history": career_history,
                "skills_listed": skills_listed,
                "last_active": last_active,
                "vector_distance": float(dist)
            }
            candidates_retrieved.append(candidate)
            
        return candidates_retrieved

if __name__ == "__main__":
    # Test execution
    retriever = CandidateRetriever()
    test_query = "React frontend developer, module federation, hydration, microservice"
    results = retriever.retrieve(test_query, top_k=5)
    print(f"Retrieved {len(results)} candidates.")
    for idx, c in enumerate(results):
        print(f"[{idx+1}] ID: {c['candidate_id']}, Name: {c['name']}, Exp: {c['years_experience']} yrs, Dist: {c['vector_distance']:.4f}")
