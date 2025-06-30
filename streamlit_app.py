# streamlit_app.py - Railway Deployment Version
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

# Get API URL - Railway specific
def get_api_url():
    """Get the correct API URL for Railway deployment"""
    # For Railway, the API runs on the same domain but different port
    # When both are deployed together, they share the same base URL
    
    # Try to get from Streamlit secrets first
    try:
        api_url = st.secrets.get("API_URL")
        if api_url:
            return api_url
    except:
        pass
    
    # For Railway deployment, the API will be on the same domain
    # Railway typically uses https://your-app-name.up.railway.app
    current_url = st.get_option("browser.serverAddress") or "localhost"
    
    if "railway.app" in str(current_url) or "up.railway.app" in str(current_url):
        # Production Railway URL - API on same domain
        return f"https://{current_url.replace('https://', '').replace('http://', '')}"
    elif current_url != "localhost":
        # Other production environments
        return f"https://{current_url}"
    else:
        # Local development
        return "http://127.0.0.1:8000"

API_URL = get_api_url()

# Helper function - DEFINE EARLY!
def load_documents():
    """Load documents from API and update session state"""
    try:
        response = requests.get(f"{API_URL}/api/documents", timeout=10)
        if response.status_code == 200:
            data = response.json()
            st.session_state.documents = data.get('documents', [])
        else:
            st.session_state.documents = []
    except Exception as e:
        print(f"Error loading documents: {e}")
        st.session_state.documents = []

# Initialize session state
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'search_results' not in st.session_state:
    st.session_state.search_results = None

st.title("📄 OCR Document Search Demo")
st.markdown("Upload documents and search through them using AI-powered OCR and semantic search")

# Sidebar for upload
with st.sidebar:
    st.header("📤 Upload Documents")
    
    # Check API health and show status
    try:
        health_response = requests.get(f"{API_URL}/api/health", timeout=10)
        if health_response.status_code == 200:
            health_data = health_response.json()
            st.success(f"✅ API Connected")
            
            # Show API status details
            search_type = health_data.get('search_type', 'unknown')
            if search_type == 'semantic':
                st.success("🤖 AI Semantic Search Active")
            elif search_type == 'keyword':
                st.warning("🔤 Keyword Search Mode")
            else:
                st.info("⏳ Search Engine Loading...")
            
            st.caption(f"Documents: {health_data.get('documents', 0)}")
            st.caption(f"Search chunks: {health_data.get('chunks', 0)}")
            
            # Show dependencies if available
            if 'dependencies' in health_data:
                deps = health_data['dependencies']
                if deps.get('tesseract'):
                    st.caption("✅ Real OCR Available")
                else:
                    st.caption("🎭 Simulated OCR")
        else:
            st.error("❌ API Connection Failed")
            st.caption(f"Status: {health_response.status_code}")
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API")
        st.caption("Check if API is running")
    except requests.exceptions.Timeout:
        st.warning("⏳ API Response Timeout")
    except Exception as e:
        st.error(f"❌ API Error: {str(e)}")
    
    # Show current API URL for debugging
    with st.expander("🔧 Debug Info"):
        st.code(f"API URL: {API_URL}")
        if st.button("Test API Connection"):
            try:
                test_response = requests.get(f"{API_URL}/", timeout=5)
                if test_response.status_code == 200:
                    st.success("✅ API responding")
                    st.json(test_response.json())
                else:
                    st.error(f"❌ API error: {test_response.status_code}")
            except Exception as e:
                st.error(f"❌ Connection failed: {e}")
    
    st.divider()
    
    # File upload section
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['png', 'jpg', 'jpeg', 'pdf'],
        help="Upload an image or PDF to extract text"
    )
    
    if uploaded_file is not None:
        # Show file preview
        file_size_mb = len(uploaded_file.getvalue()) / 1024 / 1024
        st.info(f"📁 {uploaded_file.name}")
        st.info(f"📊 Size: {file_size_mb:.1f}MB")
        
        if uploaded_file.type.startswith('image/'):
            st.image(uploaded_file, caption="Preview", use_column_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📤 Upload", type="primary", use_container_width=True):
                if file_size_mb > 20:
                    st.error("File too large! Please use files under 20MB.")
                else:
                    with st.spinner("Processing with OCR..."):
                        try:
                            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                            response = requests.post(f"{API_URL}/api/upload", files=files, timeout=120)
                            
                            if response.status_code == 200:
                                result = response.json()
                                st.success("✅ Upload successful!")
                                
                                # Show upload results
                                st.metric("Text Length", f"{result.get('text_length', 0):,} chars")
                                st.metric("OCR Confidence", f"{result.get('confidence', 0):.1f}%")
                                st.metric("Chunks Created", result.get('chunks_created', 0))
                                
                                # Show OCR type
                                ocr_type = result.get('ocr_type', 'unknown')
                                if ocr_type == 'real':
                                    st.success("🔍 Real OCR used")
                                else:
                                    st.info("🎭 Simulated OCR used")
                                
                                # Refresh documents list
                                load_documents()
                                st.rerun()
                                
                            else:
                                error_text = response.text
                                st.error(f"Upload failed: {error_text}")
                                
                        except requests.exceptions.ConnectionError:
                            st.error("Cannot connect to API. Is it running?")
                        except requests.exceptions.Timeout:
                            st.error("Upload timed out. File might be too large or complex.")
                        except Exception as e:
                            st.error(f"Upload error: {str(e)}")
        
        with col2:
            if st.button("🔄 Clear", use_container_width=True):
                st.rerun()
    
    # Show uploaded documents
    st.divider()
    st.header("📚 Documents")
    
    if st.button("🔄 Refresh", use_container_width=True):
        load_documents()
        st.rerun()
    
    if st.session_state.documents:
        for doc in st.session_state.documents:
            with st.expander(f"📄 {doc['filename']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Confidence", f"{doc.get('confidence', 0)}%")
                with col2:
                    st.metric("Length", f"{doc.get('text_length', 0):,} chars")
                
                st.caption(f"Uploaded: {doc.get('uploaded_at', 'Unknown')[:19]}")
                
                # Show OCR type if available
                if 'ocr_type' in doc:
                    if doc['ocr_type'] == 'real':
                        st.caption("🔍 Real OCR")
                    else:
                        st.caption("🎭 Simulated")
    else:
        st.info("No documents uploaded yet")

# Main area - Search
st.header("🔍 Search Documents")

if not st.session_state.documents:
    st.info("👆 Upload a document in the sidebar to start searching!")
    
    # Show some example searches
    st.markdown("### Example searches to try after uploading:")
    st.code("monthly rent")
    st.code("lease terms") 
    st.code("contact information")
    st.code("due date")
    
else:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Enter your search query",
            placeholder="e.g., monthly rent, lease terms, total amount, contact info",
            help="Search across all uploaded documents"
        )
    
    with col2:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    # Example searches
    st.markdown("**Quick searches:**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("💰 Monthly Rent", use_container_width=True):
            query = "monthly rent"
            search_button = True
    with col2:
        if st.button("📅 Lease Terms", use_container_width=True):
            query = "lease terms"
            search_button = True
    with col3:
        if st.button("📞 Contact", use_container_width=True):
            query = "contact information"
            search_button = True
    with col4:
        if st.button("🏠 Property", use_container_width=True):
            query = "property address"
            search_button = True
    
    if search_button and query:
        with st.spinner("Searching..."):
            try:
                response = requests.post(
                    f"{API_URL}/api/search",
                    json={"query": query, "limit": 5},
                    timeout=30
                )
                
                if response.status_code == 200:
                    st.session_state.search_results = response.json()
                else:
                    st.error(f"Search failed: {response.text}")
                    st.session_state.search_results = None
                    
            except requests.exceptions.Timeout:
                st.error("Search timed out. Please try a simpler query.")
            except Exception as e:
                st.error(f"Search error: {str(e)}")
                st.session_state.search_results = None

# Display results
if st.session_state.search_results:
    results = st.session_state.search_results
    
    st.divider()
    
    # Answer section
    st.markdown("### 💡 Answer")
    answer = results.get('answer', 'No answer available')
    if "Found" in answer and "relevant sections" in answer:
        st.success(answer)
    elif "No relevant information" in answer:
        st.warning(answer)
    else:
        st.info(answer)
    
    # Search metadata
    col1, col2, col3 = st.columns(3)
    with col1:
        search_type = results.get('search_type', 'unknown')
        if search_type == 'semantic':
            st.metric("Search Type", "🤖 AI Semantic")
        elif search_type == 'keyword':
            st.metric("Search Type", "🔤 Keyword")
        else:
            st.metric("Search Type", search_type.title())
            
    with col2:
        st.metric("Results Found", len(results.get('results', [])))
    with col3:
        st.metric("Documents Searched", len(st.session_state.documents))
    
    # Results section
    if results.get('results'):
        st.markdown("### 📑 Relevant Sections")
        
        for i, result in enumerate(results['results'], 1):
            relevance = result.get('relevance_score', 0)
            
            # Color code relevance
            if relevance > 0.7:
                relevance_color = "🟢"
            elif relevance > 0.4:
                relevance_color = "🟡"
            else:
                relevance_color = "🔴"
            
            with st.expander(
                f"Result {i} - {result['filename']} "
                f"{relevance_color} Relevance: {relevance:.3f}"
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
                    st.caption(f"📄 Document: {result['filename']}")
                with col2:
                    st.caption(f"🎯 OCR Confidence: {result['confidence']}%")
                with col3:
                    st.caption(f"📊 Relevance: {relevance:.3f}")
    else:
        st.info("No relevant sections found. Try different search terms or upload more documents.")

# Footer
st.divider()
st.markdown(
    f"""
    <div style='text-align: center; color: #666;'>
        🚀 Built with FastAPI + Tesseract OCR + Sentence Transformers<br>
        🚂 Deployed on Railway | API: <code>{API_URL}</code>
    </div>
    """,
    unsafe_allow_html=True
)

# Load documents on startup
if not st.session_state.documents:
    load_documents()