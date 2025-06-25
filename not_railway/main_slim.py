from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import io
from datetime import datetime
import os

app = FastAPI(title="OCR RAG Demo - Slim Version")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
documents: Dict[int, dict] = {}
doc_counter = 0
chunks_data = []
vectorizer = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2),  # Capture some phrases
    stop_words='english'
)
tfidf_matrix = None

class SearchRequest(BaseModel):
    query: str
    limit: int = 5

@app.get("/")
def root():
    return {
        "message": "OCR RAG API - Slim Version",
        "version": "TF-IDF",
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
        "search_engine": "TF-IDF",
        "version": "slim"
    }

def extract_text_from_image(image_bytes: bytes) -> tuple[str, float]:
    """Extract text from image using Tesseract OCR"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        text = pytesseract.image_to_string(image)
        
        # Get confidence
        try:
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        except:
            avg_confidence = 80.0  # Default
        
        return text.strip(), avg_confidence
    except Exception as e:
        print(f"OCR Error: {e}")
        return "", 0.0

def extract_text_from_pdf(pdf_bytes: bytes) -> List[tuple[str, float]]:
    """Extract text from PDF pages"""
    try:
        images = convert_from_bytes(pdf_bytes, dpi=150)  # Lower DPI for speed
        results = []
        
        for i, image in enumerate(images):
            print(f"Processing page {i+1}/{len(images)}...")
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            text, confidence = extract_text_from_image(img_byte_arr)
            results.append((text, confidence))
        
        return results
    except Exception as e:
        print(f"PDF Error: {e}")
        return [("", 0.0)]

def create_chunks(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks"""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size].strip()
        if len(chunk) > 50:
            chunks.append(chunk)
    return chunks

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    global doc_counter, tfidf_matrix, chunks_data
    
    contents = await file.read()
    
    # Extract text
    if file.content_type == 'application/pdf':
        pages_results = extract_text_from_pdf(contents)
        full_text = "\n\n".join([text for text, _ in pages_results])
        avg_confidence = sum(conf for _, conf in pages_results) / len(pages_results) if pages_results else 0
    elif file.content_type and file.content_type.startswith('image/'):
        full_text, avg_confidence = extract_text_from_image(contents)
    else:
        raise HTTPException(400, f"Unsupported file type: {file.content_type}")
    
    if not full_text or len(full_text.strip()) < 10:
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
    
    # Add to chunks data
    for chunk in chunks:
        chunks_data.append({
            "doc_id": doc_counter,
            "filename": file.filename,
            "text": chunk,
            "confidence": avg_confidence
        })
    
    # Rebuild TF-IDF matrix
    if chunks_data:
        all_texts = [chunk["text"] for chunk in chunks_data]
        tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    return {
        "success": True,
        "document_id": doc_counter,
        "filename": file.filename,
        "text_length": len(full_text),
        "confidence": round(avg_confidence, 2),
        "chunks_created": len(chunks),
        "message": f"Document processed with {len(chunks)} searchable chunks"
    }

@app.post("/api/search")
def search_documents(request: SearchRequest):
    global tfidf_matrix
    
    if not chunks_data or tfidf_matrix is None:
        return {
            "query": request.query,
            "results": [],
            "message": "No documents uploaded yet"
        }
    
    # Transform query
    query_vec = vectorizer.transform([request.query])
    
    # Calculate similarities
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Get top results
    top_indices = similarities.argsort()[-request.limit:][::-1]
    
    results = []
    for idx in top_indices:
        if similarities[idx] > 0.1:  # Threshold
            chunk = chunks_data[idx]
            results.append({
                "document_id": chunk["doc_id"],
                "filename": chunk["filename"],
                "text": chunk["text"],
                "confidence": round(chunk["confidence"], 2),
                "relevance_score": round(float(similarities[idx]), 3)
            })
    
    # Simple answer
    if results:
        answer = f"Found {len(results)} relevant sections for '{request.query}' in your documents."
    else:
        answer = f"No relevant information found for '{request.query}'."
    
    return {
        "query": request.query,
        "answer": answer,
        "results": results,
        "search_type": "TF-IDF"
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

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)