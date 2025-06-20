import streamlit as st
import requests
import time
from PIL import Image
import io

# API URL - change this after deployment
API_URL = st.secrets.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="OCR Document Search", page_icon="📄", layout="wide")

st.title("📄 OCR Document Search Demo")
st.markdown("Upload documents and search through them using AI")

# Initialize session state
if 'documents' not in st.session_state:
    st.session_state.documents = []

# Sidebar
with st.sidebar:
    st.header("📤 Upload Documents")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a document image to extract text"
    )
    
    if uploaded_file is not None:
        if st.button("Process Document", type="primary"):
            with st.spinner("Processing with OCR..."):
                # Upload to API
                files = {"file": uploaded_file}
                response = requests.post(f"{API_URL}/api/upload", files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"✅ Processed successfully!")
                    st.json(result)
                    
                    # Refresh document list
                    docs_response = requests.get(f"{API_URL}/api/documents")
                    if docs_response.status_code == 200:
                        st.session_state.documents = docs_response.json()
                else:
                    st.error(f"Error: {response.text}")
    
    # Show uploaded documents
    st.header("📚 Uploaded Documents")
    if st.button("Refresh"):
        docs_response = requests.get(f"{API_URL}/api/documents")
        if docs_response.status_code == 200:
            st.session_state.documents = docs_response.json()
    
    for doc in st.session_state.documents:
        st.text(f"📄 {doc['filename']}")
        st.caption(f"Confidence: {doc['confidence']:.1f}%")

# Main area - Search
st.header("🔍 Search Documents")

query = st.text_input("Enter your search query:", placeholder="e.g., What is the budget?")

if query:
    with st.spinner("Searching..."):
        response = requests.post(
            f"{API_URL}/api/search",
            json={"query": query, "limit": 5}
        )
        
        if response.status_code == 200:
            results = response.json()
            
            st.markdown("### 💡 Answer")
            st.info(results['answer'])
            
            st.markdown("### 📑 Relevant Sections")
            for i, result in enumerate(results['results'], 1):
                with st.expander(f"Result {i} - {result['filename']} (Relevance: {result['relevance_score']})"):
                    st.text(result['text'])
                    st.caption(f"OCR Confidence: {result['confidence']}%")
        else:
            st.error("Search failed")

# Footer
st.markdown("---")
st.markdown("Built with FastAPI + EasyOCR + Sentence Transformers")