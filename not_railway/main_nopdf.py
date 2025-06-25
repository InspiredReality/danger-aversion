from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes
import numpy as np
import io
from datetime import datetime
import os

app = FastAPI(title="OCR Document Search - Production")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage
documents: Dict[int, dict] = {}
doc_counter = 0
chunks_data = []
embedder = None
search_type = "loading"

def load_embedder():
    """Load sentence transformer model from local cache"""
    global embedder, search_type
    
    print("🔄 Loading sentence transformer model...")
    
    # Set environment variables to use local models
    model_cache_dir = "/app/models"
    os.environ['SENTENCE_TRANSFORMERS_HOME'] = model_cache_dir
    os.environ['TRANSFORMERS_CACHE'] = model_cache_dir
    os.environ['HF_HOME'] = model_cache_dir
    
    try:
        # Check if local model exists
        local_model_paths = [
            "/app/models/sentence-transformers_all-MiniLM-L6-v2",
            "/app/models/models--sentence-transformers--all-MiniLM-L6-v2",
            "/app/models/all-MiniLM-L6-v2"
        ]
        
        model_found = False
        for model_path in local_model_paths:
            if os.path.exists(model_path):
                print(f"✅ Found local model at: {model_path}")
                
                # List contents for debugging
                contents = os.listdir(model_path)
                print(f"📁 Model contents: {contents}")
                
                # Check for required files
                required_files = ['config.json']
                if any(f in contents for f in required_files):
                    try:
                        from sentence_transformers import SentenceTransformer
                        embedder = SentenceTransformer(model_path)
                        model_found = True
                        print("✅ Model loaded from local cache!")
                        break
                    except Exception as e:
                        print(f"❌ Failed to load from {model_path}: {e}")
                        continue
        
        if not model_found:
            print("❌ No valid local model found, attempting download...")
            # Try to download (fallback)
            from sentence_transformers import SentenceTransformer
            embedder = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=model_cache_dir)
            print("✅ Model downloaded and loaded!")
        
        # Test the model
        test_embedding = embedder.encode(["test sentence"])
        print(f"✅ Model test successful! Embedding shape: {test_embedding.shape}")
        search_type = "semantic"
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        print("🔄 Falling back to keyword search...")
        embedder = None
        search_type = "keyword"

# Load model on startup
load_embedder()

class SearchRequest(BaseModel):
    query: str
    limit: int = 5

@app.get("/")
def root():
    return {
        "message": "OCR Document Search - Production Ready",
        "search_type": search_type,
        "embedder_loaded": embedder is not None,
        "endpoints": {
            "upload": "/api/upload",
            "search": "/api/search",
            "documents": "/api/documents",
            "health": "/api/health"
        }
    }

@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "documents": len(documents),
        "chunks": len(chunks_data),
        "search_type": search_type,
        "embedder_loaded": embedder is not None,
        "model_path": "/app/models" if embedder else None
    }

def extract_text_from_image(image_bytes: bytes) -> tuple[str, float]:
    """Extract text from image using Tesseract OCR"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Optimize for OCR
        width, height = image.size
        if max(width, height) > 2000:
            scale = 2000 / max(width, height)
            new_size = (int(width * scale), int(height * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Perform OCR with optimized settings
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(image, config=custom_config)
        
        # Calculate confidence
        try:
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config=custom_config)
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        except:
            avg_confidence = 75.0
        
        return text.strip(), avg_confidence
        
    except Exception as e:
        print(f"OCR Error: {e}")
        return "", 0.0

def extract_text_from_pdf(pdf_bytes: bytes) -> List[tuple[str, float]]:
    """Extract text from PDF pages"""
    try:
        print(f"📄 Processing PDF of size: {len(pdf_bytes) / 1024 / 1024:.1f}MB")
        
        # Convert with reasonable settings
        images = convert_from_bytes(pdf_bytes, dpi=150, first_page=1, last_page=10)
        results = []
        
        for i, image in enumerate(images):
            print(f"🔍 Processing page {i+1}/{len(images)}...")
            
            # Convert to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG', optimize=True)
            text, confidence = extract_text_from_image(img_byte_arr.getvalue())
            results.append((text, confidence))
        
        return results
        
    except Exception as e:
        print(f"PDF Error: {e}")
        return [("", 0.0)]

def create_chunks(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks"""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size].strip()
        if len(chunk) > 30:
            chunks.append(chunk)
    return chunks

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    global doc_counter
    
    print(f"📤 Processing {file.filename}")
    
    contents = await file.read()
    file_size_mb = len(contents) / 1024 / 1024
    
    if file_size_mb > 50:
        raise HTTPException(400, f"File too large: {file_size_mb:.1f}MB")
    
    # Extract text
    if file.content_type == 'application/pdf':
        pages_results = extract_text_from_pdf(contents)
        full_text = "\n\n--- PAGE BREAK ---\n\n".join([text for text, _ in pages_results])
        avg_confidence = sum(conf for _, conf in pages_results) / len(pages_results) if pages_results else 0
    elif file.content_type and file.content_type.startswith('image/'):
        full_text, avg_confidence = extract_text_from_image(contents)
    else:
        raise HTTPException(400, f"Unsupported file type: {file.content_type}")
    
    if len(full_text.strip()) < 10:
        raise HTTPException(400, "Could not extract sufficient text")
    
    # Store document
    doc_counter += 1
    documents[doc_counter] = {
        "id": doc_counter,
        "filename": file.filename,
        "text": full_text,
        "confidence": round(avg_confidence, 2),
        "uploaded_at": datetime.now().isoformat(),
        "text_length": len(full_text)
    }
    
    # Create chunks
    chunks = create_chunks(full_text)
    for chunk in chunks:
        chunks_data.append({
            "doc_id": doc_counter,
            "filename": file.filename,
            "text": chunk,
            "confidence": avg_confidence
        })
    
    print(f"✅ Processed {file.filename}: {len(chunks)} chunks created")
    
    return {
        "success": True,
        "document_id": doc_counter,
        "filename": file.filename,
        "text_length": len(full_text),
        "confidence": round(avg_confidence, 2),
        "chunks_created": len(chunks),
        "search_type": search_type,
        "message": f"Document processed with {len(chunks)} searchable chunks"
    }

@app.post("/api/search")
def search_documents(request: SearchRequest):
    if not chunks_data:
        return {
            "query": request.query,
            "results": [],
            "message": "No documents uploaded yet"
        }
    
    results = []
    
    # Semantic search with local model
    if search_type == "semantic" and embedder:
        try:
            print(f"🤖 Semantic search for: '{request.query}'")
            
            # Encode query and chunks
            query_embedding = embedder.encode([request.query])
            chunk_texts = [chunk["text"] for chunk in chunks_data]
            chunk_embeddings = embedder.encode(chunk_texts)
            
            # Calculate similarities
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
            
            # Get top results
            top_indices = similarities.argsort()[-request.limit:][::-1]
            
            for idx in top_indices:
                if similarities[idx] > 0.1:
                    chunk = chunks_data[idx]
                    results.append({
                        "document_id": chunk["doc_id"],
                        "filename": chunk["filename"],
                        "text": chunk["text"],
                        "confidence": round(chunk["confidence"], 2),
                        "relevance_score": round(float(similarities[idx]), 3)
                    })
            
            print(f"✅ Semantic search found {len(results)} results")
            
        except Exception as e:
            print(f"❌ Semantic search error: {e}")
            # Fall back to keyword search
    
    # Keyword search fallback
    if not results:
        print(f"🔍 Keyword search for: '{request.query}'")
        query_lower = request.query.lower()
        
        for chunk in chunks_data:
            if query_lower in chunk['text'].lower():
                matches = chunk['text'].lower().count(query_lower)
                words = len(chunk['text'].split())
                score = matches / words if words > 0 else 0
                
                results.append({
                    "document_id": chunk['doc_id'],
                    "filename": chunk['filename'],
                    "text": chunk['text'],
                    "confidence": round(chunk['confidence'], 2),
                    "relevance_score": round(score, 3)
                })
        
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        results = results[:request.limit]
        
        print(f"✅ Keyword search found {len(results)} results")
    
    # Generate answer
    if results:
        search_method = "semantic AI" if search_type == "semantic" else "keyword"
        answer = f"Found {len(results)} relevant sections using {search_method} search for '{request.query}'"
    else:
        answer = f"No relevant information found for '{request.query}'"
    
    return {
        "query": request.query,
        "answer": answer,
        "results": results,
        "search_type": search_type
    }

@app.get("/api/documents")
def list_documents():
    return {
        "documents": [
            {
                "id": doc["id"],
                "filename": doc["filename"],
                "confidence": doc["confidence"],
                "text_length": doc["text_length"],
                "uploaded_at": doc["uploaded_at"]
            }
            for doc in documents.values()
        ],
        "total": len(documents),
        "total_chunks": len(chunks_data),
        "search_type": search_type
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting OCR API with {search_type} search on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)