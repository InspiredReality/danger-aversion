# Multi-stage build for Railway deployment
# ==========================================
# Stage 1: Model Download (discarded after build)
# ==========================================
FROM python:3.9-slim as model-downloader

# Install system dependencies for model download
RUN apt-get update && apt-get install -y git wget && rm -rf /var/lib/apt/lists/*

# Install specific compatible versions for model download
RUN pip install --no-cache-dir \
    torch==2.1.0+cpu \
    sentence-transformers==2.2.2 \
    transformers==4.35.0 \
    huggingface-hub==0.17.3 \
    --index-url https://download.pytorch.org/whl/cpu

# Create model cache directory
RUN mkdir -p /tmp/models

# Download model with verbose logging
RUN python -c "
import os
print('=== Starting model download ===')
os.makedirs('/tmp/models', exist_ok=True)

from sentence_transformers import SentenceTransformer
print('Downloading all-MiniLM-L6-v2...')

try:
    model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='/tmp/models')
    print('✅ Model downloaded successfully!')
    
    # Test the model
    test_embedding = model.encode(['test sentence'])
    print(f'✅ Model test passed! Embedding shape: {test_embedding.shape}')
    
    # List downloaded files
    import os
    for root, dirs, files in os.walk('/tmp/models'):
        level = root.replace('/tmp/models', '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Show first 5 files
            print(f'{subindent}{file}')
    
except Exception as e:
    print(f'❌ Model download failed: {e}')
    raise e

print('=== Model download complete ===')
"

# ==========================================
# Stage 2: Production Image (final image)
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

# Copy requirements and install Python packages
COPY requirements-railway.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy pre-downloaded models from build stage
COPY --from=model-downloader /tmp/models/ /app/models/

# Verify models were copied
RUN echo "=== Verifying models in production image ===" && \
    ls -la /app/models/ && \
    find /app/models -name "*.json" -o -name "*.bin" | head -5

# Copy application code
COPY . .

# Set environment variables for model location
ENV SENTENCE_TRANSFORMERS_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models
ENV HF_HOME=/app/models
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:$PORT/api/health || exit 1

# Expose port (Railway sets $PORT)
EXPOSE $PORT

# Run the application
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]