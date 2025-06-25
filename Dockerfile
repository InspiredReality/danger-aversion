# ==========================================
# Stage 1: Model Download
# ==========================================
FROM python:3.9-slim as model-downloader

# Install dependencies for model download
RUN apt-get update && apt-get install -y git wget && rm -rf /var/lib/apt/lists/*

# Install ML packages
RUN pip install --no-cache-dir \
    torch==2.1.0+cpu \
    sentence-transformers==2.2.2 \
    transformers==4.35.0 \
    huggingface-hub==0.17.3 \
    --index-url https://download.pytorch.org/whl/cpu

# Create model directory
RUN mkdir -p /tmp/models

# Create the download script as a separate file
COPY download_model.py /tmp/download_model.py

# Run the download script
RUN python3 /tmp/download_model.py

# ==========================================
# Stage 2: Runtime Image
# ==========================================
FROM python:3.9-slim

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy requirements and install runtime packages
COPY requirements-railway-runtime.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# Copy pre-downloaded models from stage 1
COPY --from=model-downloader /tmp/models/ /app/models/

# Verify models were copied
RUN ls -la /app/models/ && \
    find /app/models -name "*.json" -o -name "*.bin" | head -5

# Copy application code
COPY main.py streamlit_app.py ./

# Set environment variables
ENV SENTENCE_TRANSFORMERS_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models
ENV HF_HOME=/app/models
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:$PORT/api/health || exit 1

# Expose port
EXPOSE $PORT

# Run the application
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
