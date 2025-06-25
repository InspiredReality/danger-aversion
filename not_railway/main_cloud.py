from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import io
import easyocr
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from datetime import datetime
import sqlite3
import json

app = FastAPI(title="OCR RAG Demo API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models (happens once on startup)
print("Initializing models...")
reader = easyocr.Reader(['en'], gpu=False)
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("Models ready!")

# In-memory storage for demo (resets on restart)
documents_db = {}
document_counter = 0
faiss_index = faiss.IndexFlatL2(384)
indexed_chunks = []

class SearchRequest(BaseModel):
    query: str
    limit: int = 5

@app.get("/")
async def root():
    return {
        "message": "OCR RAG Demo API",
        "endpoints": {
            "upload": "/api/upload",
            "search": "/api/search",
            "documents": "/api/documents",
            "health": "/api/health"
        }
    }

@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "models": "loaded",
        "documents_count": len(documents_db)
    }

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    global document_counter
    
    # Validate file type
    allowed_types = ['image/jpeg', 'image/png', 'image/jpg']
    if file.content_type not in allowed_types:
        raise HTTPException(400, f"File type {file.content_type} not supported")
    
    # Read file
    contents = await file.read()
    
    # OCR with EasyOCR
    image_bytes = io.BytesIO(contents)
    results = reader.readtext(image_bytes)
    
    if not results:
        raise HTTPException(400, "No text found in image")
    
    # Extract text and confidence
    texts = []
    confidences = []
    for (bbox, text, conf) in results:
        texts.append(text)
        confidences.append(conf * 100)
    
    full_text = " ".join(texts)
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    # Store document
    document_counter += 1
    doc_id = document_counter
    
    documents_db[doc_id] = {
        'id': doc_id,
        'filename': file.filename,
        'text': full_text,
        'confidence': avg_confidence,
        'uploaded_at': datetime.now().isoformat()
    }
    
    # Index for search
    chunks = create_chunks(full_text, chunk_size=200, overlap=50)
    for chunk in chunks:
        embedding = embedder.encode([chunk])[0]
        faiss_index.add(np.array([embedding]))
        indexed_chunks.append({
            'doc_id': doc_id,
            'filename': file.filename,
            'text': chunk,
            'confidence': avg_confidence
        })
    
    return {
        "success": True,
        "document_id": doc_id,
        "filename": file.filename,
        "text_length": len(full_text),
        "confidence": round(avg_confidence, 2),
        "message": f"Document processed successfully with {len(chunks)} searchable chunks"
    }

@app.post("/api/search")
async def search_documents(request: SearchRequest):
    if not indexed_chunks:
        return {
            "query": request.query,
            "message": "No documents uploaded yet",
            "results": []
        }
    
    # Encode query
    query_embedding = embedder.encode([request.query])[0]
    
    # Search
    k = min(request.limit, len(indexed_chunks))
    distances, indices = faiss_index.search(np.array([query_embedding]), k)
    
    # Format results
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        if idx < len(indexed_chunks):
            chunk = indexed_chunks[idx]
            results.append({
                "text": chunk['text'],
                "filename": chunk['filename'],
                "confidence": round(chunk['confidence'], 2),
                "relevance_score": round(1 / (1 + float(distance)), 3)
            })
    
    # Simple answer generation (without LLM for cloud deployment)
    answer = f"Found {len(results)} relevant sections for '{request.query}'."
    if results:
        answer += f" The most relevant section is from {results[0]['filename']}."
    
    return {
        "query": request.query,
        "answer": answer,
        "results": results
    }

@app.get("/api/documents")
async def list_documents():
    return list(documents_db.values())

def create_chunks(text: str, chunk_size: int = 200, overlap: int = 50):
    """Split text into overlapping chunks"""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if len(chunk) > 50:
            chunks.append(chunk)
    return chunks