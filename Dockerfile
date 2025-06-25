# ==========================================
# Stage 1: Model Download
# ==========================================
FROM python:3.9-slim as model-downloader

# Install git for downloading from huggingface
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install ML packages with working versions
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch==2.0.1+cpu --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    sentence-transformers==2.5.1 \
    transformers==4.40.2 \
    huggingface-hub==0.22.2

# Create model directory
RUN mkdir -p /tmp/models

# Download model directly with Python commands
RUN python3 -c "
import os
os.makedirs('/tmp/models', exist_ok=True)
print('=== Starting model download ===')

from sentence_transformers import SentenceTransformer
print('Downloading all-MiniLM-L6-v2...')

model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='/tmp/models')
print('Model downloaded successfully!')

# Test the model
test_embedding = model.encode(['test'])
print(f'Model test passed! Shape: {test_embedding.shape}')

# Verify files exist
for root, dirs, files in os.walk('/tmp/models'):
    for file in files[:3]:
        print(f'Found: {file}')
        break

print('=== Model download complete ===')
"

# ==========================================
# Stage 2: Runtime Image (Final)
# ==========================================
FROM python:3.9-slim

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements-railway-runtime.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy models from build stage
COPY --from=model-downloader /tmp/models/ /app/models/

# Verify models copied
RUN ls -la /app/models/ || echo "Models directory empty"

# Copy app files
COPY main.py streamlit_app.py ./

# Set environment variables
ENV SENTENCE_TRANSFORMERS_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models
ENV HF_HOME=/app/models
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:$PORT/api/health || exit 1

EXPOSE $PORT

CMD uvicorn main:app --host 0.0.0.0 --port $PORT