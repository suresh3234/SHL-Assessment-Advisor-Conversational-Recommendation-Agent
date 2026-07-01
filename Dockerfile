FROM python:3.11-slim

WORKDIR /app

# Set thread count limits to minimize memory consumption during runtime
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MALLOC_ARENA_MAX=2

# Install system dependencies needed for building some packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .

# Install CPU-only version of PyTorch to avoid massive CUDA downloads and excessive memory overhead
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the sentence-transformers model to warm the cache during image build
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application code and prebuilt index/catalog files
COPY app/ ./app/

# Expose port
EXPOSE 8000

# Run uvicorn server
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
