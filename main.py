from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes
import io
from datetime import datetime
import os

app = FastAPI(title="OCR Document Search - Railway with Local Models")

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

# Load embedder from local models
embedder = None
search_type = "loading"

def load_local_embedder():
    """Load sentence transformer from pre-downloaded local models"""
    global embedder, search_type
    
    if embedder is None:
        print("🔄 Loading sentence transformer from local models...")
        
        try:
            from sentence_transformers import SentenceTransformer
            
            # Set environment to use local models
            model_cache_dir = "/app/models"
            os.environ['SENTENCE_TRANSFORMERS_HOME'] = model_cache_dir
            os.environ['TRANSFORMERS_CACHE'] = model_cache_dir
            os.environ['HF_HOME'] = model_cache_dir
            
            # Check if local model exists
            if os.path.exists(model_cache_dir):
                print(f"✅ Model cache directory found: {model_cache_dir}")
                
                # List contents for debugging
                contents = os.listdir(model_cache_dir)
                print(f"📁 Cache contents: {contents}")
                
                # Look for the model directory
                model_patterns = [
                    "sentence-transformers_all-MiniLM-L6-v2",
                    "models--sentence-transformers--all-MiniLM-L6-v2",
                    "all-MiniLM-L6-v2"
                ]
                
                model_path = None
                for pattern in model_patterns:
                    potential_path = os.path.join(model_cache_dir, pattern)
                    if os.path.exists(potential_path):
                        model_path = potential_path
                        print(f"🎯 Found model at: {model_path}")
                        break
                
                if model_path:
                    # Load from local path
                    embedder = SentenceTransformer(model_path)
                    search_type = "semantic"
                    print("✅ Model loaded from local cache!")
                else:
                    # Try loading by name (will use cache if available)
                    print("🔄 Loading by model name...")
                    embedder = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=model_cache_dir)
                    search_type = "semantic"
                    print("✅ Model loaded!")
            else:
                print(f"❌ Model cache directory not found: {model_cache_dir}")
                # Try downloading (shouldn't happen in Railway if build worked)
                embedder = SentenceTransformer('all-MiniLM-L6-v2')
                search_type = "semantic"
                print("⚠️ Model downloaded at runtime (build cache failed)")
            
            # Test the model
            test_embedding = embedder.encode(["test sentence"])
            print(f"✅ Model test successful! Shape: {test_embedding.shape}")
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            print("🔄 Falling back to keyword search...")
            embedder = "fallback"
            search_type = "keyword"
    
    return embedder

class SearchRequest(BaseModel):
    query: str
    limit: int = 5

@app.get("/")
def root():
    return {
        "message": "OCR Document Search - Railway Production",
        "search_type": search_type,
        "embedder_loaded": embedder is not None and embedder != "fallback",
        "status": "running"
    }

@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "documents": len(documents),
        "chunks": len(chunks_data),
        "search_type": search_type,
        "embedder_loaded": embedder is not None and embedder != "fallback"
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
        
        # Perform OCR
        text = pytesseract.image_to_string(image, config='--psm 6')
        
        # Calculate confidence
        try:
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
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
        print(f"📄 Processing PDF ({len(pdf_bytes) / 1024 / 1024:.1f}MB)")
        
        # Convert PDF to images
        images = convert_from_bytes(
            pdf_bytes,
            dpi=150,
            first_page=1,
            last_page=5  # Limit for Railway performance
        )
        
        print(f"✅ Converted to {len(images)} images")
        results = []
        
        for i, image in enumerate(images):
            print(f"🔍 Processing page {i+1}/{len(images)}...")
            
            # Convert PIL image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG', optimize=True)
            text, confidence = extract_text_from_image(img_byte_arr.getvalue())
            
            if text.strip():
                results.append((text, confidence))
                print(f"  ✅ Page {i+1}: {len(text)} chars, {confidence:.1f}% confidence")
            else:
                print(f"  ⚠️ Page {i+1}: No text extracted")
        
        print(f"✅ PDF processing complete: {len(results)} pages with text")
        return results
        
    except Exception as e:
        print(f"❌ PDF processing error: {e}")
        return [("", 0.0)]

def create_chunks(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks"""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size].strip()
        if len(chunk) > 50:
            chunks.append(chunk)
    return chunks

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    global doc_counter
    
    print(f"📤 Processing {file.filename}")
    
    contents = await file.read()
    file_size_mb = len(contents) / 1024 / 1024
    
    if file_size_mb > 20:  # Railway limit
        raise HTTPException(400, f"File too large: {file_size_mb:.1f}MB. Please use files under 20MB.")
    
    # Extract text based on file type
    if file.content_type == 'application/pdf':
        pages_results = extract_text_from_pdf(contents)
        if pages_results and any(text.strip() for text, _ in pages_results):
            valid_pages = [(text, conf) for text, conf in pages_results if text.strip()]
            full_text = "\n\n--- PAGE BREAK ---\n\n".join([text for text, _ in valid_pages])
            avg_confidence = sum(conf for _, conf in valid_pages) / len(valid_pages)
        else:
            raise HTTPException(400, "Could not extract text from PDF")
            
    elif file.content_type and file.content_type.startswith('image/'):
        full_text, avg_confidence = extract_text_from_image(contents)
        if len(full_text.strip()) < 10:
            raise HTTPException(400, "Could not extract sufficient text from image")
    else:
        raise HTTPException(400, f"Unsupported file type: {file.content_type}")
    
    if len(full_text.strip()) < 20:
        raise HTTPException(400, "Extracted text too short")
    
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
    
    print(f"✅ Document processed: {len(chunks)} chunks created")
    
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
    
    # Load embedder if not loaded yet
    embedder_model = load_local_embedder()
    
    results = []
    
    # Semantic search
    if search_type == "semantic" and embedder_model != "fallback":
        try:
            print(f"🤖 Semantic search: '{request.query}'")
            
            query_embedding = embedder_model.encode([request.query])
            chunk_texts = [chunk["text"] for chunk in chunks_data]
            chunk_embeddings = embedder_model.encode(chunk_texts)
            
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
            
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
            
            print(f"✅ Found {len(results)} semantic matches")
            
        except Exception as e:
            print(f"❌ Semantic search error: {e}")
    
    # Keyword search fallback
    if not results:
        print(f"🔍 Keyword search: '{request.query}'")
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
        
        print(f"✅ Found {len(results)} keyword matches")
    
    # Generate answer
    if results:
        search_method = "AI semantic" if search_type == "semantic" else "keyword"
        answer = f"Found {len(results)} relevant sections using {search_method} search"
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
        "total_chunks": len(chunks_data)
    }

# Load embedder on startup
load_local_embedder()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting OCR API on port {port}")
    print(f"🔧 Search type: {search_type}")
    uvicorn.run(app, host="0.0.0.0", port=port)