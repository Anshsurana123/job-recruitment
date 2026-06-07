import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer

def main():
    model_name = "all-MiniLM-L6-v2"
    cache_dir = Path("./model_cache") / model_name
    
    print(f"Checking for local model cache at '{cache_dir}'...")
    if cache_dir.exists() and (cache_dir / "model.safetensors").exists():
        print("Model cache already exists and is complete.")
        return
        
    print(f"Downloading model '{model_name}' and saving to local cache...")
    try:
        model = SentenceTransformer(model_name)
        model.save(str(cache_dir))
        print(f"Model successfully saved locally at '{cache_dir}'.")
    except Exception as e:
        print(f"Error downloading model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
