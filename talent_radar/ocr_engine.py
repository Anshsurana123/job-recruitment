import os
import sys
import io
import torch
import threading
from PIL import Image
import fitz  # PyMuPDF

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

class LocalOCREngine:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(LocalOCREngine, cls).__new__(cls, *args, **kwargs)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.backend = os.getenv("LOCAL_OCR_BACKEND", "qwen2.5-vl").lower()
        self._model = None
        self._processor = None
        self._model_lock = threading.Lock()
        
        # Configure worker counts
        self.easyocr_workers = int(os.getenv("LOCAL_OCR_EASYOCR_WORKERS", "4"))
        self.qwen_workers = int(os.getenv("LOCAL_OCR_QWEN_WORKERS", "1"))
        
        # Configure min/max patches to limit pixel count and speed up CPU inference
        device = "cuda" if torch.cuda.is_available() else "cpu"
        default_max_patches = "1280" if device == "cuda" else "512"
        self.qwen_min_patches = int(os.getenv("LOCAL_OCR_QWEN_MIN_PATCHES", "256"))
        self.qwen_max_patches = int(os.getenv("LOCAL_OCR_QWEN_MAX_PATCHES", default_max_patches))
        
        self._initialized = True

    def _init_qwen(self):
        with self._model_lock:
            if self._model is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"[Local OCR] Loading Qwen2-VL-2B-Instruct vision model and processor (Device: {device})...", flush=True)
                from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
                
                # Dynamic CPU optimization: set PyTorch thread pool size to prevent thrashing
                if device == "cpu":
                    torch_threads = int(os.getenv("TORCH_CPU_THREADS", "4"))
                    torch.set_num_threads(torch_threads)
                    print(f"[Local OCR] Optimizing PyTorch CPU threads: set to {torch_threads} to prevent core thrashing.", flush=True)
                
                dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                
                # Load the model with SDPA attention (highly vectorized and fast on both GPU and CPU)
                # Omit device_map="auto" on CPU to prevent accelerate hook overhead
                if device == "cuda":
                    self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                        "Qwen/Qwen2-VL-2B-Instruct",
                        torch_dtype=dtype,
                        device_map="auto",
                        attn_implementation="sdpa"
                    )
                else:
                    self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                        "Qwen/Qwen2-VL-2B-Instruct",
                        torch_dtype=dtype,
                        attn_implementation="sdpa"
                    ).to("cpu")
                    
                self._processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
                print(f"[Local OCR] Qwen2-VL-2B-Instruct vision model loaded successfully on device: {device}", flush=True)
                print(f"[Local OCR] Configured Qwen dynamic pixel range: min={self.qwen_min_patches * 28 * 28} ({self.qwen_min_patches} patches), max={self.qwen_max_patches * 28 * 28} ({self.qwen_max_patches} patches)", flush=True)
                print(f"[Local OCR] Configured Qwen ThreadPool workers: {self.qwen_workers}", flush=True)

    def _init_easyocr(self):
        with self._model_lock:
            if self._model is None:
                gpu_avail = torch.cuda.is_available()
                print(f"[Local OCR] Loading EasyOCR English reader (GPU Available: {gpu_avail})...", flush=True)
                import easyocr
                self._model = easyocr.Reader(['en'], gpu=gpu_avail)
