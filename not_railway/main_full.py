from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes
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

# Global variables
embedder = None
documents: Dict[int, dict] = {}
doc_counter = 0
faiss_index = None
indexed_chunks = []

# Initialize models on first request
def get_embedder():
    global embedder
    if embedder is None:
        print("Loading sentence transformer model (this may take a moment on first request)...")
        try:
            from sentence_transformers import SentenceTransformer
            embedder = SentenceTransformer('all-MiniLM-L6-v2')
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to simple search if model fails
            embedder = "fallback"
    return embedder

# Initialize FAISS index
def init_faiss():
    global faiss_index
    if faiss_index is None:
        embedding_dimension = 384  # all-MiniLM-L6-v2 dimension
        faiss_index = faiss.IndexFlatL2(embedding_dimension)
    return faiss_index

class SearchRequest(BaseModel):
    query: str
    limit: int = 5

@app.get("/")
def root():
    return {
        "message": "OCR RAG API with Real OCR",
        "endpoints": {
            "upload": "/api/upload",
            "search": "/api/search", 
            "documents": "/api/documents",
            "health": "/api/health"
        },
        "note": "First request may be slow while models load"
    }

@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "documents": len(documents),
        "indexed_chunks": len(indexed_chunks),
        "ocr": "pytesseract",
        "embedder_loaded": embedder is not None
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
        try:
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        except:
            avg_confidence = 75.0  # Default confidence if calculation fails
        
        return text.strip(), avg_confidence
    except Exception as e:
        print(f"OCR Error: {e}")
        return "Error extracting text from image.", 0.0

def extract_text_from_pdf(pdf_bytes: bytes) -> List[tuple[str, float]]:
    """Extract text from PDF pages"""
    try:
        images = convert_from_bytes(pdf_bytes, dpi=200)
        results = []
        
        for i, image in enumerate(images):
            print(f"Processing page {i+1}/{len(images)}...")
            # Convert PIL Image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            text, confidence = extract_text_from_image(img_byte_arr)
            results.append((text, confidence))
        
        return results
    except Exception as e:
        print(f"PDF Error: {e}")
        return [("Error processing PDF.", 0.0)]

def create_chunks(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks"""
    if not text:
        return []
    
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size].strip()
        if len(chunk) > 50:
            chunks.append(chunk)
    
    return chunks

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    global doc_counter, indexed_chunks
    
    # Initialize models if needed
    embedder_model = get_embedder()
    index = init_faiss()
    
    # Read file
    contents = await file.read()
    
    # Extract text based on file type
    if file.content_type == 'application/pdf':
        pages_results = extract_text_from_pdf(contents)
        full_text = "\n\n--- PAGE BREAK ---\n\n".join([text for text, _ in pages_results])
        avg_confidence = sum(conf for _, conf in pages_results) / len(pages_results) if pages_results else 0
    elif file.content_type and file.content_type.startswith('image/'):
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
        "text_length": len(full_text)
    }
    
    # Create chunks and index them
    chunks = create_chunks(full_text)
    
    # Only create embeddings if model loaded successfully
    if embedder_model != "fallback":
        for chunk in chunks:
            try:
                # Create embedding
                embedding = embedder_model.encode([chunk])[0]
                
                # Add to FAISS index
                index.add(np.array([embedding]).astype('float32'))
                
                # Store chunk metadata
                indexed_chunks.append({
                    "doc_id": doc_counter,
                    "filename": file.filename,
                    "text": chunk,
                    "confidence": avg_confidence
                })
            except Exception as e:
                print(f"Embedding error: {e}")
    else:
        # Fallback: store chunks without embeddings
        for chunk in chunks:
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
    
    embedder_model = get_embedder()
    
    # If embedder failed, fallback to keyword search
    if embedder_model == "fallback":
        # Simple keyword search
        results = []
        query_lower = request.query.lower()
        
        for chunk in indexed_chunks:
            if query_lower in chunk['text'].lower():
                results.append({
                    "document_id": chunk['doc_id'],
                    "filename": chunk['filename'],
                    "text": chunk['text'],
                    "confidence": round(chunk['confidence'], 2),
                    "relevance_score": chunk['text'].lower().count(query_lower) / len(chunk['text'].split())
                })
        
        # Sort by relevance
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        results = results[:request.limit]
    else:
        # Semantic search with embeddings
        try:
            query_embedding = embedder_model.encode([request.query])[0]
            
            # Search in FAISS
            k = min(request.limit, len(indexed_chunks))
            index = init_faiss()
            distances, indices = index.search(
                np.array([query_embedding]).astype('float32'), 
                k
            )
            
            # Format results
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(indexed_chunks):
                    chunk = indexed_chunks[idx]
                    results.append({
                        "document_id": chunk['doc_id'],
                        "filename": chunk['filename'],
                        "text": chunk['text'],
                        "confidence": round(chunk['confidence'], 2),
                        "relevance_score": round(1 / (1 + float(distance)), 3)
                    })
        except Exception as e:
            print(f"Search error: {e}")
            return {"query": request.query, "error": "Search failed", "results": []}
    
    # Create answer
    if results:
        answer = f"Found {len(results)} relevant sections for '{request.query}' in your documents."
    else:
        answer = f"No relevant information found for '{request.query}'."
    
    return {
        "query": request.query,
        "answer": answer,
        "results": results,
        "search_type": "semantic" if embedder_model != "fallback" else "keyword"
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

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)