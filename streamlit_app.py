# streamlit_app.py
import streamlit as st
import requests
import json
from PIL import Image
import io

# Configure page
st.set_page_config(
    page_title="OCR Document Search",
    page_icon="📄",
    layout="wide"
)

# Get API URL from secrets or use default
API_URL = st.secrets.get("API_URL", "http://localhost:8000")

st.title("📄 OCR Document Search Demo")
st.markdown("Upload documents and search through them using AI-powered OCR and semantic search")

# Initialize session state
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'search_results' not in st.session_state:
    st.session_state.search_results = None

# Sidebar for upload
with st.sidebar:
    st.header("📤 Upload Documents")
    
    # Check API health
    try:
        health_response = requests.get(f"{API_URL}/api/health", timeout=5)
        if health_response.status_code == 200:
            health_data = health_response.json()
            st.success(f"✅ API Connected")
            st.caption(f"Documents: {health_data.get('documents', 0)}")
        else:
            st.error("❌ API Connection Failed")
    except:
        st.warning("⚠️ Cannot connect to API")
        st.info(f"Make sure the API is running at: {API_URL}")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['png', 'jpg', 'jpeg', 'pdf'],
        help="Upload an image or PDF to extract text"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📤 Upload", type="primary", use_container_width=True):
                with st.spinner("Processing with OCR..."):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        response = requests.post(f"{API_URL}/api/upload", files=files, timeout=30)
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success("✅ Upload successful!")
                            st.json(result)
                            
                            # Refresh documents list
                            load_documents()
                        else:
                            st.error(f"Upload failed: {response.text}")
                    except requests.exceptions.ConnectionError:
                        st.error("Cannot connect to API. Is it running?")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        with col2:
            if uploaded_file.type.startswith('image/'):
                st.image(uploaded_file, caption="Preview", use_column_width=True)
    
    # Show uploaded documents
    st.divider()
    st.header("📚 Documents")
    
    if st.button("🔄 Refresh", use_container_width=True):
        load_documents()
    
    if st.session_state.documents:
        for doc in st.session_state.documents:
            with st.expander(f"📄 {doc['filename']}"):
                st.text(f"Confidence: {doc.get('confidence', 'N/A')}%")
                st.text(f"Length: {doc.get('text_length', 'N/A')} chars")
                st.caption(f"Uploaded: {doc.get('uploaded_at', 'Unknown')}")
    else:
        st.info("No documents uploaded yet")

# Main area - Search
st.header("🔍 Search Documents")

col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_input(
        "Enter your search query",
        placeholder="e.g., What is the budget? Who attended the meeting?",
        help="Search across all uploaded documents"
    )

with col2:
    search_button = st.button("🔍 Search", type="primary", use_container_width=True)

if search_button and query:
    with st.spinner("Searching..."):
        try:
            response = requests.post(
                f"{API_URL}/api/search",
                json={"query": query, "limit": 5},
                timeout=10
            )
            
            if response.status_code == 200:
                st.session_state.search_results = response.json()
            else:
                st.error(f"Search failed: {response.text}")
        except Exception as e:
            st.error(f"Search error: {str(e)}")

# Display results
if st.session_state.search_results:
    results = st.session_state.search_results
    
    st.divider()
    
    # Answer section
    st.markdown("### 💡 Answer")
    st.info(results.get('answer', 'No answer available'))
    
    # Search metadata
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Search Type", results.get('search_type', 'Unknown'))
    with col2:
        st.metric("Results Found", len(results.get('results', [])))
    
    # Results section
    if results.get('results'):
        st.markdown("### 📑 Relevant Sections")
        
        for i, result in enumerate(results['results'], 1):
            with st.expander(
                f"Result {i} - {result['filename']} "
                f"(Relevance: {result.get('relevance_score', 'N/A')})"
            ):
                st.text_area(
                    "Extracted Text",
                    result['text'],
                    height=150,
                    disabled=True,
                    key=f"result_{i}"
                )
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.caption(f"📄 Document ID: {result['document_id']}")
                with col2:
                    st.caption(f"🎯 OCR Confidence: {result['confidence']}%")
                with col3:
                    st.caption(f"📊 Relevance: {result.get('relevance_score', 'N/A')}")

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #888;'>
        Built with FastAPI + Tesseract OCR + Sentence Transformers<br>
        <a href='/docs' target='_blank'>API Documentation</a>
    </div>
    """,
    unsafe_allow_html=True
)

# Helper function
def load_documents():
    try:
        response = requests.get(f"{API_URL}/api/documents", timeout=5)
        if response.status_code == 200:
            data = response.json()
            st.session_state.documents = data.get('documents', [])
    except:
        st.session_state.documents = []

# Load documents on startup
if not st.session_state.documents:
    load_documents()
    