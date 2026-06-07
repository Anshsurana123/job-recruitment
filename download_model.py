import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder

def download_bi_encoder():
    model_name = "BAAI/bge-small-en-v1.5"
    cache_dir = Path("./model_cache/bge-small-en-v1.5")
    
    print(f"Checking for bi-encoder cache at '{cache_dir}'...")
    if cache_dir.exists() and (cache_dir / "model.safetensors").exists():
        print("Bi-encoder cache already exists and is complete.")
        return
        
    print(f"Downloading bi-encoder '{model_name}'...")
    try:
        model = SentenceTransformer(model_name)
        model.save(str(cache_dir))
        print(f"Bi-encoder saved at '{cache_dir}'.")
    except Exception as e:
        print(f"Error downloading bi-encoder: {e}")
        sys.exit(1)

def download_cross_encoder():
    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cache_dir = Path("./model_cache/cross-encoder-ms-marco-MiniLM-L-6-v2")
    
    print(f"Checking for cross-encoder cache at '{cache_dir}'...")
    if cache_dir.exists() and (cache_dir / "model.safetensors").exists():
        print("Cross-encoder cache already exists and is complete.")
        return
        
    print(f"Downloading cross-encoder '{model_name}'...")
    try:
        model = CrossEncoder(model_name)
        model.save(str(cache_dir))
        print(f"Cross-encoder saved at '{cache_dir}'.")
    except Exception as e:
        print(f"Error downloading cross-encoder: {e}")
        sys.exit(1)

def main():
    download_bi_encoder()
    download_cross_encoder()
    print("\nAll models cached locally. Ready for offline execution.")

if __name__ == "__main__":
    main()
