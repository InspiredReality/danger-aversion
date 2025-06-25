from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from datetime import datetime
import io

app = FastAPI(title="OCR Document Search - Fixed PDF Processing")

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

# Test dependencies
def check_dependencies():
    status = {"tesseract": False, "poppler": False, "sentence_transformers": False}
    
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract {version} available")
        status["tesseract"] = True
    except Exception as e:
        print(f"❌ Tesseract error: {e}")
    
    try:
        from pdf2image import convert_from_bytes
        print("✅ Poppler available")
        status["poppler"] = True
    except Exception as e:
        print(f"❌ Poppler error: {e}")
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ Sentence Transformers available")
        status["sentence_transformers"] = True
    except Exception as e:
        print(f"❌ Sentence Transformers error: {e}")
    
    return status

DEPS = check_dependencies()

# Load AI model
embedder = None
search_type = "keyword"

if DEPS["sentence_transformers"]:
    try:
        print("🔄 Loading AI model...")
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        search_type = "semantic"
        print("✅ AI model loaded!")
    except Exception as e:
        print(f"❌ AI model failed: {e}")

class SearchRequest(BaseModel):
    query: str
    limit: int = 5

@app.get("/")
def root():
    return {
        "message": "OCR Document Search - Fixed PDF",
        "search_type": search_type,
        "dependencies": DEPS
    }

@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "documents": len(documents),
        "chunks": len(chunks_data),
        "search_type": search_type,
        "dependencies": DEPS
    }

def extract_text_from_image(image_bytes: bytes) -> tuple[str, float]:
    """Extract text from image using Tesseract"""
    try:
        import pytesseract
        from PIL import Image
        
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Perform OCR
        text = pytesseract.image_to_string(image)
        
        # Get confidence
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
    """Extract text from PDF - FIXED VERSION"""
    
    # Check if we have the required dependencies
    if not DEPS["poppler"]:
        print("❌ Cannot process PDF: Poppler not available")
        return [("", 0.0)]
    
    if not DEPS["tesseract"]:
        print("❌ Cannot process PDF: Tesseract not available")
        return [("", 0.0)]
    
    try:
        from pdf2image import convert_from_bytes
        
        print(f"📄 Converting PDF to images...")
        
        # Convert PDF pages to images
        images = convert_from_bytes(
            pdf_bytes,
            dpi=150,
            first_page=1,
            last_page=5  # Process first 5 pages
        )
        
        print(f"✅ Converted to {len(images)} images")
        
        results = []
        total_chars = 0
        
        for i, image in enumerate(images):
            print(f"🔍 OCR on page {i+1}/{len(images)}...")
            
            # Convert PIL image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()
            
            # Extract text from this page
            page_text, page_confidence = extract_text_from_image(img_bytes)
            
            print(f"  Page {i+1}: {len(page_text)} chars, {page_confidence:.1f}% confidence")
            
            if page_text.strip():  # Only add pages with actual text
                results.append((page_text, page_confidence))
                total_chars += len(page_text)
            else:
                print(f"  Page {i+1}: No text found")
        
        print(f"✅ PDF OCR complete: {total_chars} total characters from {len(results)} pages")
        return results
        
    except Exception as e:
        print(f"❌ PDF processing failed: {e}")
        return [("", 0.0)]

def create_chunks(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """Split text into chunks"""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size].strip()
        if len(chunk) > 50:
            chunks.append(chunk)
    return chunks

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    global doc_counter
    
    print(f"\n📤 Processing {file.filename}")
    print(f"📋 Content type: {file.content_type}")
    
    contents = await file.read()
    file_size_mb = len(contents) / 1024 / 1024
    print(f"📊 Size: {file_size_mb:.1f}MB")
    
    if file_size_mb > 50:
        raise HTTPException(400, f"File too large: {file_size_mb:.1f}MB")
    
    # Process the file
    full_text = ""
    avg_confidence = 0.0
    ocr_type = "unknown"
    
    if file.content_type == 'application/pdf':
        print("📄 PDF file detected - processing with PDF pipeline")
        
        # Try real PDF processing first
        pages_results = extract_text_from_pdf(contents)
        
        if pages_results and any(text.strip() for text, _ in pages_results):
            # We got real text from PDF OCR
            valid_pages = [(text, conf) for text, conf in pages_results if text.strip()]
            all_texts = [text for text, _ in valid_pages]
            all_confidences = [conf for _, conf in valid_pages]
            
            full_text = "\n\n--- PAGE BREAK ---\n\n".join(all_texts)
            avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
            ocr_type = "real_pdf"
            
            print(f"✅ PDF OCR successful: {len(full_text)} chars, {avg_confidence:.1f}% confidence")
        else:
            # PDF OCR failed
            print("❌ PDF OCR failed - dependencies missing or error occurred")
            raise HTTPException(400, "PDF processing failed. Check that Tesseract and Poppler are installed.")
            
    elif file.content_type and file.content_type.startswith('image/'):
        print("🖼️ Image file detected - processing with image OCR")
        
        if DEPS["tesseract"]:
            full_text, avg_confidence = extract_text_from_image(contents)
            ocr_type = "real_image"
            print(f"✅ Image OCR: {len(full_text)} chars, {avg_confidence:.1f}% confidence")
        else:
            print("❌ Image OCR failed - Tesseract not available")
            raise HTTPException(400, "Image OCR failed. Check that Tesseract is installed.")
    else:
        raise HTTPException(400, f"Unsupported file type: {file.content_type}")
    
    # Validate we got useful text
    if len(full_text.strip()) < 20:
        print(f"⚠️ Extracted text too short: '{full_text[:100]}...'")
        raise HTTPException(400, f"Could not extract sufficient text. Got only {len(full_text)} characters.")
    
    # Store document
    doc_counter += 1
    documents[doc_counter] = {
        "id": doc_counter,
        "filename": file.filename,
        "text": full_text,
        "confidence": round(avg_confidence, 2),
        "uploaded_at": datetime.now().isoformat(),
        "text_length": len(full_text),
        "ocr_type": ocr_type
    }
    
    # Create searchable chunks
    chunks = create_chunks(full_text)
    for chunk in chunks:
        chunks_data.append({
            "doc_id": doc_counter,
            "filename": file.filename,
            "text": chunk,
            "confidence": avg_confidence
        })
    
    print(f"✅ Document stored successfully:")
    print(f"   ID: {doc_counter}")
    print(f"   Text length: {len(full_text):,} characters")
    print(f"   Chunks: {len(chunks)}")
    print(f"   Confidence: {avg_confidence:.1f}%")
    print(f"   OCR type: {ocr_type}")
    
    return {
        "success": True,
        "document_id": doc_counter,
        "filename": file.filename,
        "text_length": len(full_text),
        "confidence": round(avg_confidence, 2),
        "chunks_created": len(chunks),
        "search_type": search_type,
        "ocr_type": ocr_type,
        "message": f"Successfully processed {file.filename} with real OCR"
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
    
    # Semantic search if available
    if search_type == "semantic" and embedder:
        try:
            print(f"🤖 Semantic search: '{request.query}'")
            
            query_embedding = embedder.encode([request.query])
            chunk_texts = [chunk["text"] for chunk in chunks_data]
            chunk_embeddings = embedder.encode(chunk_texts)
            
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
                "uploaded_at": doc["uploaded_at"],
                "ocr_type": doc.get("ocr_type", "unknown")
            }
            for doc in documents.values()
        ],
        "total": len(documents),
        "total_chunks": len(chunks_data),
        "dependencies": DEPS
    }

if __name__ == "__main__":
    import uvicorn
    print(f"\n🚀 Starting OCR Document Search")
    print(f"📋 Available dependencies:")
    for dep, available in DEPS.items():
        status = "✅" if available else "❌"
        print(f"   {status} {dep}")
    
    if DEPS["poppler"] and DEPS["tesseract"]:
        print("🎉 PDF + Image OCR ready!")
    elif DEPS["tesseract"]:
        print("📷 Image OCR only")
    else:
        print("❌ No OCR available")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)