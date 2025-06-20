from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import io
from datetime import datetime
import os

app = FastAPI(title="OCR RAG Demo with Real OCR")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
print("Loading sentence transformer model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Small but effective model
print("Model loaded!")

# Storage
documents: Dict[int, dict] = {}
doc_counter = 0
faiss_index = None
indexed_chunks = []

# Initialize FAISS index
embedding_dimension = 384  # all-MiniLM-L6-v2 dimension
faiss_index = faiss.IndexFlatL2(embedding_dimension)

class SearchRequest(BaseModel):
    query: str
    limit: int = 5

@app.get("/")
def root():
    return {
        "message": "OCR RAG API with Real OCR and Semantic Search",
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
        "indexed_chunks": len(indexed_chunks),
        "ocr": "pytesseract",
        "search": "sentence-transformers"
    }

def extract_text_from_image(image_bytes: bytes) -> tuple[str, float]:
    """Extract text from image using Tesseract OCR"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Perform OCR
        text = pytesseract.image_to_string(image)
        
        # Get confidence scores
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return text.strip(), avg_confidence
    except Exception as e:
        print(f"OCR Error: {e}")
        # Fallback to mock data if OCR fails
        return "Error extracting text from image. This is fallback text for demo.", 0.0

def extract_text_from_pdf(pdf_bytes: bytes) -> List[tuple[str, float]]:
    """Extract text from PDF pages"""
    try:
        # Convert PDF to images
        images = convert_from_bytes(pdf_bytes, dpi=200)
        results = []
        
        for i, image in enumerate(images):
            print(f"Processing page {i+1}/{len(images)}...")
            text, confidence = extract_text_from_image(image.tobytes())
            results.append((text, confidence))
        
        return results
    except Exception as e:
        print(f"PDF Error: {e}")
        return [("Error processing PDF. This is fallback text.", 0.0)]

def create_chunks(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks for better search"""
    if not text:
        return []
    
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size].strip()
        if len(chunk) > 50:  # Minimum chunk size
            chunks.append(chunk)
    
    return chunks

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    global doc_counter, faiss_index, indexed_chunks
    
    # Read file
    contents = await file.read()
    
    # Extract text based on file type
    if file.content_type == 'application/pdf':
        # Process PDF
        pages_results = extract_text_from_pdf(contents)
        full_text = "\n\n--- PAGE BREAK ---\n\n".join([text for text, _ in pages_results])
        avg_confidence = sum(conf for _, conf in pages_results) / len(pages_results) if pages_results else 0
    elif file.content_type.startswith('image/'):
        # Process image
        full_text, avg_confidence = extract_text_from_image(contents)
    else:
        raise HTTPException(400, f"Unsupported file type: {file.content_type}")
    
    if not full_text or len(full_text.strip()) < 10:
        raise HTTPException(400, "Could not extract sufficient text from the document")
    
    # Store document
    doc_counter += 1
    documents[doc_counter] = {
        "id": doc_counter,
        "filename": file.filename,
        "text": full_text,
        "confidence": round(avg_confidence, 2),
        "uploaded_at": datetime.now().isoformat(),
        "content_type": file.content_type,
        "text_length": len(full_text)
    }
    
    # Create chunks and index them
    chunks = create_chunks(full_text)
    
    for chunk in chunks:
        # Create embedding
        embedding = embedder.encode([chunk])[0]
        
        # Add to FAISS index
        faiss_index.add(np.array([embedding]).astype('float32'))
        
        # Store chunk metadata
        indexed_chunks.append({
            "doc_id": doc_counter,
            "filename": file.filename,
            "text": chunk,
            "confidence": avg_confidence
        })
    
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
    if not indexed_chunks:
        return {
            "query": request.query,
            "results": [],
            "message": "No documents uploaded yet"
        }
    
    # Create embedding for query
    query_embedding = embedder.encode([request.query])[0]
    
    # Search in FAISS
    k = min(request.limit, len(indexed_chunks))
    distances, indices = faiss_index.search(
        np.array([query_embedding]).astype('float32'), 
        k
    )
    
    # Format results
    results = []
    seen_chunks = set()  # Avoid duplicate content
    
    for idx, distance in zip(indices[0], distances[0]):
        if idx < len(indexed_chunks):
            chunk = indexed_chunks[idx]
            chunk_preview = chunk['text'][:200]
            
            # Skip if we've seen very similar content
            if chunk_preview not in seen_chunks:
                seen_chunks.add(chunk_preview)
                results.append({
                    "document_id": chunk['doc_id'],
                    "filename": chunk['filename'],
                    "text": chunk['text'],
                    "confidence": round(chunk['confidence'], 2),
                    "relevance_score": round(1 / (1 + float(distance)), 3),
                    "distance": float(distance)
                })
    
    # Create a simple answer
    if results:
        top_result = results[0]
        answer = f"Based on your search for '{request.query}', I found relevant information in {top_result['filename']}. The most relevant section mentions: {top_result['text'][:150]}..."
    else:
        answer = f"No relevant information found for '{request.query}'."
    
    return {
        "query": request.query,
        "answer": answer,
        "results": results,
        "total_chunks_searched": len(indexed_chunks)
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
        "total_chunks": len(indexed_chunks)
    }

@app.get("/api/documents/{doc_id}")
def get_document(doc_id: int):
    if doc_id not in documents:
        raise HTTPException(404, "Document not found")
    
    doc = documents[doc_id]
    return {
        **doc,
        "preview": doc["text"][:500] + "..." if len(doc["text"]) > 500 else doc["text"]
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)