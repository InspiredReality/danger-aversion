from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import io
from PIL import Image
import pytesseract
import base64

app = FastAPI(title="OCR Demo API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage
documents = {}
doc_counter = 0

class SearchRequest(BaseModel):
    query: str
    limit: int = 5

@app.get("/")
async def root():
    return {
        "message": "OCR Demo API - Simplified Version",
        "endpoints": {
            "upload": "/api/upload",
            "search": "/api/search",
            "documents": "/api/documents"
        }
    }

@app.get("/api/health")
async def health():
    return {"status": "healthy", "documents_count": len(documents)}

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    global doc_counter
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Simple OCR (or mock for demo)
        try:
            text = pytesseract.image_to_string(image)
        except:
            # Fallback to mock data if OCR fails
            text = "This is mock text for demo purposes. The actual OCR would extract text from your uploaded image."
        
        # Store document
        doc_counter += 1
        documents[doc_counter] = {
            "id": doc_counter,
            "filename": file.filename,
            "text": text,
            "text_preview": text[:200] + "..." if len(text) > 200 else text
        }
        
        return {
            "success": True,
            "document_id": doc_counter,
            "filename": file.filename,
            "text_length": len(text),
            "message": "Document uploaded successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search")
async def search_documents(request: SearchRequest):
    if not documents:
        return {"query": request.query, "results": [], "message": "No documents uploaded yet"}
    
    # Simple keyword search (no embeddings for now)
    results = []
    query_lower = request.query.lower()
    
    for doc_id, doc in documents.items():
        if query_lower in doc["text"].lower():
            # Find the sentence containing the query
            sentences = doc["text"].split(".")
            matching_sentences = [s for s in sentences if query_lower in s.lower()]
            
            results.append({
                "document_id": doc_id,
                "filename": doc["filename"],
                "matches": matching_sentences[:2],  # First 2 matching sentences
                "relevance": len(matching_sentences)
            })
    
    # Sort by relevance
    results.sort(key=lambda x: x["relevance"], reverse=True)
    
    return {
        "query": request.query,
        "results": results[:request.limit],
        "total_matches": len(results)
    }

@app.get("/api/documents")
async def list_documents():
    return list(documents.values())

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)