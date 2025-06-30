# main_comparison.py - Backend with TF-IDF vs Semantic comparison
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import io
from datetime import datetime
import os
import asyncio
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OCR RAG with Search Comparison")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Semantic model imports (with fallback)
try:
    from sentence_transformers import SentenceTransformer
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    logger.warning("sentence-transformers not available")

# Global variables
documents: Dict[int, dict] = {}
doc_counter = 0
indexed_chunks = []

# TF-IDF components (always available)
tfidf_vectorizer = TfidfVectorizer(
    max_features=2000,
    ngram_range=(1, 2),
    stop_words='english'
)
tfidf_matrix = None

# Semantic search components
semantic_model = None
semantic_embeddings = []
semantic_status = "not_initialized"

class SearchRequest(BaseModel):
    query: str
    limit: int = 5

class ComparisonRequest(BaseModel):
    query: str
    limit: int = 5

# Model initialization
async def initialize_semantic_model():
    """Initialize semantic model in background"""
    global semantic_model, semantic_status
    
    if not SEMANTIC_AVAILABLE:
        semantic_status = "unavailable"
        return
    
    try:
        semantic_status = "loading"
        logger.info("🔄 Loading semantic model...")
        
        # Try to load model with timeout
        semantic_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        semantic_status = "ready"
        logger.info("✅ Semantic model loaded successfully")
        
        # Re-index existing documents with semantic embeddings
        if indexed_chunks:
            await reindex_with_semantic()
            
    except Exception as e:
        logger.error(f"❌ Failed to load semantic model: {e}")
        semantic_status = f"error: {str(e)}"

async def reindex_with_semantic():
    """Re-index existing chunks with semantic embeddings"""
    global semantic_embeddings
    
    if semantic_model is None or not indexed_chunks:
        return
    
    try:
        logger.info("🔄 Creating semantic embeddings for existing documents...")
        texts = [chunk["text"] for chunk in indexed_chunks]
        semantic_embeddings = semantic_model.encode(texts, batch_size=32, show_progress_bar=False)
        logger.info(f"✅ Created semantic embeddings for {len(texts)} chunks")
    except Exception as e:
        logger.error(f"❌ Failed to create semantic embeddings: {e}")
        semantic_embeddings = []

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    # Start semantic model loading in background
    asyncio.create_task(initialize_semantic_model())

@app.get("/")
def root():
    return {
        "message": "OCR RAG API with Search Comparison",
        "endpoints": {
            "upload": "/api/upload",
            "search": "/api/search",
            "search_tfidf": "/api/search/tfidf",
            "search_semantic": "/api/search/semantic", 
            "search_compare": "/api/search/compare",
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
        "tfidf_status": "ready" if tfidf_matrix is not None else "no_data",
        "semantic_status": semantic_status,
        "semantic_embeddings": len(semantic_embeddings),
        "model_status": {
            "status": semantic_status,
            "method": "semantic" if semantic_status == "ready" else "tfidf_only"
        }
    }

def extract_text_from_image(image_bytes: bytes) -> tuple[str, float]:
    """Extract text from image using Tesseract OCR"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        text = pytesseract.image_to_string(image)
        
        # Simple confidence estimation
        words = text.split()
        if len(words) > 10:
            confidence = 85.0
        elif len(words) > 3:
            confidence = 65.0
        else:
            confidence = 30.0
        
        return text.strip(), confidence
    except Exception as e:
        logger.error(f"OCR Error: {e}")
        return "", 0.0

def extract_text_from_pdf(pdf_bytes: bytes) -> List[tuple[str, float]]:
    """Extract text from PDF pages"""
    try:
        images = convert_from_bytes(pdf_bytes, dpi=150, first_page=1, last_page=10)
        results = []
        
        for i, image in enumerate(images):
            logger.info(f"Processing page {i+1}/{len(images)}...")
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            text, confidence = extract_text_from_image(img_byte_arr)
            results.append((text, confidence))
        
        return results
    except Exception as e:
        logger.error(f"PDF Error: {e}")
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
    global doc_counter, tfidf_matrix, indexed_chunks, semantic_embeddings
    
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
    
    # Add chunks to index
    chunk_start_idx = len(indexed_chunks)
    for chunk in chunks:
        indexed_chunks.append({
            "doc_id": doc_counter,
            "filename": file.filename,
            "text": chunk,
            "confidence": avg_confidence
        })
    
    # Rebuild TF-IDF matrix
    if indexed_chunks:
        all_texts = [chunk["text"] for chunk in indexed_chunks]
        tfidf_matrix = tfidf_vectorizer.fit_transform(all_texts)
    
    # Add semantic embeddings for new chunks if model is ready
    if semantic_model is not None and semantic_status == "ready":
        try:
            new_embeddings = semantic_model.encode(chunks, batch_size=32, show_progress_bar=False)
            
            # Extend existing embeddings array
            if len(semantic_embeddings) == 0:
                semantic_embeddings = new_embeddings
            else:
                semantic_embeddings = np.vstack([semantic_embeddings, new_embeddings])
            
            logger.info(f"✅ Added semantic embeddings for {len(chunks)} new chunks")
        except Exception as e:
            logger.error(f"❌ Failed to create semantic embeddings for new chunks: {e}")
    
    return {
        "success": True,
        "document_id": doc_counter,
        "filename": file.filename,
        "text_length": len(full_text),
        "confidence": round(avg_confidence, 2),
        "chunks_created": len(chunks),
        "message": f"Document processed with {len(chunks)} searchable chunks"
    }

def search_tfidf(query: str, limit: int = 5) -> tuple[List[dict], str]:
    """Perform TF-IDF search"""
    if not indexed_chunks or tfidf_matrix is None:
        return [], "no_data"
    
    try:
        start_time = time.time()
        
        query_vec = tfidf_vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        
        # Get top results
        top_indices = similarities.argsort()[-limit:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.05:  # Lower threshold for TF-IDF
                chunk = indexed_chunks[idx]
                results.append({
                    "document_id": chunk["doc_id"],
                    "filename": chunk["filename"],
                    "text": chunk["text"],
                    "confidence": round(chunk["confidence"], 2),
                    "relevance_score": round(float(similarities[idx]), 3),
                    "method": "tfidf"
                })
        
        search_time = time.time() - start_time
        return results, "success"
        
    except Exception as e:
        logger.error(f"TF-IDF search error: {e}")
        return [], f"error: {str(e)}"

def search_semantic(query: str, limit: int = 5) -> tuple[List[dict], str]:
    """Perform semantic search"""
    if semantic_model is None:
        return [], "model_not_ready"
    
    if len(semantic_embeddings) == 0:
        return [], "no_embeddings"
    
    try:
        start_time = time.time()
        
        # Get query embedding
        query_embedding = semantic_model.encode([query])[0]
        
        # Calculate similarities
        similarities = []
        for i, doc_embedding in enumerate(semantic_embeddings):
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            similarities.append((similarity, i))
        
        # Sort by similarity
        similarities.sort(reverse=True)
        
        results = []
        for similarity, idx in similarities[:limit]:
            if similarity > 0.3:  # Higher threshold for semantic
                if idx < len(indexed_chunks):
                    chunk = indexed_chunks[idx]
                    results.append({
                        "document_id": chunk["doc_id"],
                        "filename": chunk["filename"],
                        "text": chunk["text"],
                        "confidence": round(chunk["confidence"], 2),
                        "relevance_score": round(float(similarity), 3),
                        "method": "semantic"
                    })
        
        search_time = time.time() - start_time
        return results, "success"
        
    except Exception as e:
        logger.error(f"Semantic search error: {e}")
        return [], f"error: {str(e)}"

@app.post("/api/search/tfidf")
def search_tfidf_only(request: SearchRequest):
    """TF-IDF search endpoint"""
    results, status = search_tfidf(request.query, request.limit)
    
    return {
        "query": request.query,
        "results": results,
        "search_type": "tfidf",
        "status": status,
        "count": len(results)
    }

@app.post("/api/search/semantic")
def search_semantic_only(request: SearchRequest):
    """Semantic search endpoint"""
    results, status = search_semantic(request.query, request.limit)
    
    return {
        "query": request.query,
        "results": results,
        "search_type": "semantic",
        "status": status,
        "count": len(results)
    }

@app.post("/api/search/compare")
def search_compare(request: ComparisonRequest):
    """Compare TF-IDF vs Semantic search side by side"""
    start_time = time.time()
    
    # Perform both searches
    tfidf_start = time.time()
    tfidf_results, tfidf_status = search_tfidf(request.query, request.limit)
    tfidf_time = time.time() - tfidf_start
    
    semantic_start = time.time()
    semantic_results, semantic_status = search_semantic(request.query, request.limit)
    semantic_time = time.time() - semantic_start
    
    total_time = time.time() - start_time
    
    # Analysis metrics
    tfidf_texts = {r.get('text', '') for r in tfidf_results}
    semantic_texts = {r.get('text', '') for r in semantic_results}
    overlap_count = len(tfidf_texts.intersection(semantic_texts))
    
    return {
        "query": request.query,
        "tfidf_results": tfidf_results,
        "semantic_results": semantic_results,
        "tfidf_status": tfidf_status,
        "semantic_status": semantic_status,
        "tfidf_time": round(tfidf_time, 3),
        "semantic_time": round(semantic_time, 3),
        "total_time": round(total_time, 3),
        "analysis": {
            "tfidf_count": len(tfidf_results),
            "semantic_count": len(semantic_results),
            "overlap_count": overlap_count,
            "unique_results": len(tfidf_texts.union(semantic_texts))
        }
    }

@app.post("/api/search")
def search_documents(request: SearchRequest):
    """Default search endpoint (hybrid approach)"""
    # Try semantic first, fall back to TF-IDF
    semantic_results, semantic_status = search_semantic(request.query, request.limit)
    
    if semantic_status == "success" and len(semantic_results) > 0:
        return {
            "query": request.query,
            "results": semantic_results,
            "search_type": "semantic",
            "status": semantic_status,
            "message": f"Found {len(semantic_results)} semantic results"
        }
    else:
        tfidf_results, tfidf_status = search_tfidf(request.query, request.limit)
        return {
            "query": request.query,
            "results": tfidf_results,
            "search_type": "tfidf_fallback",
            "status": tfidf_status,
            "message": f"Found {len(tfidf_results)} TF-IDF results (semantic unavailable: {semantic_status})"
        }

@app.get("/api/search/status")
def search_status():
    """Get detailed status of search capabilities"""
    return {
        "tfidf": {
            "status": "ready" if tfidf_matrix is not None else "no_data",
            "documents_indexed": len(indexed_chunks),
            "vocabulary_size": tfidf_vectorizer.vocabulary_.__len__() if hasattr(tfidf_vectorizer, 'vocabulary_') else 0
        },
        "semantic": {
            "status": semantic_status,
            "embeddings_count": len(semantic_embeddings),
            "model_loaded": semantic_model is not None,
            "available": SEMANTIC_AVAILABLE
        },
        "documents": {
            "total_documents": len(documents),
            "total_chunks": len(indexed_chunks)
        }
    }

@app.get("/api/documents")
def list_documents():
    """List all uploaded documents"""
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
    """Get specific document details"""
    if doc_id not in documents:
        raise HTTPException(404, "Document not found")
    
    doc = documents[doc_id]
    doc_chunks = [chunk for chunk in indexed_chunks if chunk["doc_id"] == doc_id]
    
    return {
        "document": doc,
        "chunks": doc_chunks,
        "chunk_count": len(doc_chunks)
    }

@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: int):
    """Delete a document and its chunks"""
    global tfidf_matrix, semantic_embeddings, indexed_chunks
    
    if doc_id not in documents:
        raise HTTPException(404, "Document not found")
    
    # Remove document
    filename = documents[doc_id]["filename"]
    del documents[doc_id]
    
    # Remove chunks and embeddings
    old_chunks = indexed_chunks
    indexed_chunks = [chunk for chunk in indexed_chunks if chunk["doc_id"] != doc_id]
    
    chunks_removed = len(old_chunks) - len(indexed_chunks)
    
    # Rebuild indices
    if indexed_chunks:
        # Rebuild TF-IDF
        all_texts = [chunk["text"] for chunk in indexed_chunks]
        tfidf_matrix = tfidf_vectorizer.fit_transform(all_texts)
        
        # Rebuild semantic embeddings if available
        if semantic_model is not None and semantic_status == "ready":
            try:
                semantic_embeddings = semantic_model.encode(all_texts, batch_size=32, show_progress_bar=False)
                logger.info(f"✅ Rebuilt semantic embeddings after deletion")
            except Exception as e:
                logger.error(f"❌ Failed to rebuild semantic embeddings: {e}")
                semantic_embeddings = []
    else:
        # No documents left
        tfidf_matrix = None
        semantic_embeddings = []
    
    return {
        "success": True,
        "message": f"Deleted document '{filename}' and {chunks_removed} chunks",
        "remaining_documents": len(documents),
        "remaining_chunks": len(indexed_chunks)
    }

@app.post("/api/reindex")
async def reindex_documents():
    """Manually trigger reindexing of all documents"""
    global tfidf_matrix, semantic_embeddings
    
    if not indexed_chunks:
        return {"message": "No documents to reindex"}
    
    try:
        # Rebuild TF-IDF
        all_texts = [chunk["text"] for chunk in indexed_chunks]
        tfidf_matrix = tfidf_vectorizer.fit_transform(all_texts)
        
        # Rebuild semantic embeddings
        await reindex_with_semantic()
        
        return {
            "success": True,
            "message": f"Reindexed {len(indexed_chunks)} chunks",
            "tfidf_ready": tfidf_matrix is not None,
            "semantic_ready": len(semantic_embeddings) > 0
        }
    except Exception as e:
        logger.error(f"Reindexing failed: {e}")
        raise HTTPException(500, f"Reindexing failed: {str(e)}")

# Enhanced search with query expansion for medical terms
medical_abbreviations = {
    'MI': 'myocardial infarction heart attack',
    'HTN': 'hypertension high blood pressure', 
    'DM': 'diabetes mellitus blood sugar',
    'COPD': 'chronic obstructive pulmonary disease',
    'CHF': 'congestive heart failure',
    'CVA': 'cerebrovascular accident stroke',
    'SOB': 'shortness of breath dyspnea',
    'CP': 'chest pain thoracic pain',
    'URI': 'upper respiratory infection',
    'UTI': 'urinary tract infection'
}

def expand_medical_query(query: str) -> str:
    """Expand medical abbreviations in query"""
    expanded = query
    for abbrev, expansion in medical_abbreviations.items():
        if abbrev.upper() in query.upper():
            expanded += f" {expansion}"
    return expanded

@app.post("/api/search/medical")
def search_medical(request: SearchRequest):
    """Enhanced search for medical documents with abbreviation expansion"""
    # Expand medical terms
    expanded_query = expand_medical_query(request.query)
    
    # Perform comparison with expanded query
    tfidf_results, tfidf_status = search_tfidf(expanded_query, request.limit)
    semantic_results, semantic_status = search_semantic(expanded_query, request.limit)
    
    return {
        "original_query": request.query,
        "expanded_query": expanded_query,
        "tfidf_results": tfidf_results,
        "semantic_results": semantic_results,
        "tfidf_status": tfidf_status,
        "semantic_status": semantic_status,
        "search_type": "medical_enhanced"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)