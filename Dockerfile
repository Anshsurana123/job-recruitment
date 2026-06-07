# ============================================================
# SwarmMatrix AI — Hugging Face Spaces Dockerfile
# Multi-Agent SLM Swarm Architecture for Candidate Ranking
# ============================================================

FROM python:3.10-slim

# Install system dependencies (build-essential, git, and graphics libraries for vision/OCR compatibility)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user (Hugging Face Spaces requirement: UID 1000)
RUN useradd -m -u 1000 user

# Set up working directory inside the user's home to guarantee write permissions
WORKDIR /home/user/app

# Set environment variables for clean python execution
ENV PATH="/home/user/.local/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Copy requirements.txt and pre-install dependencies as root to cache docker layers
COPY ./requirements.txt /home/user/app/requirements.txt

# Upgrade pip and install standard PyTorch (supports both CPU and GPU CUDA runtimes)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch && \
    pip install --no-cache-dir -r /home/user/app/requirements.txt

# Copy the entire project repository and set ownership to our non-root user
COPY --chown=user:user . /home/user/app

# Switch to the non-root user for security and compliance with Hugging Face Spaces
USER user

# Hugging Face Spaces requires port 7860
EXPOSE 7860

# Launch FastAPI via uvicorn (app is located in the talent_radar package)
CMD ["uvicorn", "talent_radar.app:app", "--host", "0.0.0.0", "--port", "7860"]
