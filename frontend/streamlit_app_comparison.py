# streamlit_app_comparison.py
import streamlit as st
import requests
import json
from PIL import Image
import io
import os
import pandas as pd

# Configure page
st.set_page_config(
    page_title="OCR Search Comparison",
    page_icon="🔍",
    layout="wide"
)

# Get API URL
if "RENDER" in os.environ:
    API_URL = os.getenv("API_URL", "https://your-backend.onrender.com")
else:
    API_URL = st.secrets.get("API_URL", "http://localhost:8000")

st.title("🔍 OCR Search Method Comparison")
st.markdown("Upload documents and compare **TF-IDF (Keyword)** vs **Semantic Search** results side by side")

# Initialize session state
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = None

# Sidebar for upload and settings
with st.sidebar:
    st.header("📤 Upload Documents")
    
    # API Health Check
    try:
        health_response = requests.get(f"{API_URL}/health", timeout=10)
        if health_response.status_code == 200:
            health_data = health_response.json()
            st.success(f"✅ API Connected")
            
            # Show model status if available
            if 'model_status' in health_data:
                model_info = health_data['model_status']
                if model_info.get('status') == 'ready':
                    st.info(f"🧠 Model: {model_info.get('method', 'Unknown')}")
                elif model_info.get('status') == 'loading':
                    st.warning("⏳ Model loading...")
                else:
                    st.error("❌ Model error")
            
            st.caption(f"Documents: {health_data.get('documents', 0)}")
        else:
            st.error("❌ API Connection Failed")
    except:
        st.warning("⚠️ Cannot connect to API")
        st.info(f"API URL: {API_URL}")
    
    # File upload
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
                        response = requests.post(f"{API_URL}/api/upload", files=files, timeout=60)
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success("✅ Upload successful!")
                            
                            # Show upload details
                            st.json({
                                "filename": result.get("filename"),
                                "chunks_created": result.get("chunks_created"),
                                "confidence": result.get("confidence"),
                                "text_length": result.get("text_length")
                            })
                            
                            # Refresh documents list
                            load_documents()
                        else:
                            st.error(f"Upload failed: {response.text}")
                    except requests.exceptions.ConnectionError:
                        st.error("Cannot connect to API")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        with col2:
            if uploaded_file.type.startswith('image/'):
                st.image(uploaded_file, caption="Preview", use_column_width=True)
    
    # Search Settings
    st.divider()
    st.header("🔧 Search Settings")
    
    num_results = st.slider("Results per method", min_value=3, max_value=10, value=5)
    show_scores = st.checkbox("Show relevance scores", value=True)
    show_methods = st.checkbox("Show search method details", value=True)
    
    # Documents list
    st.divider()
    st.header("📚 Documents")
    
    if st.button("🔄 Refresh", use_container_width=True):
        load_documents()
    
    if st.session_state.documents:
        for doc in st.session_state.documents:
            with st.expander(f"📄 {doc['filename']}"):
                st.metric("OCR Confidence", f"{doc.get('confidence', 'N/A')}%")
                st.metric("Text Length", f"{doc.get('text_length', 'N/A')} chars")
                st.caption(f"Uploaded: {doc.get('uploaded_at', 'Unknown')}")
    else:
        st.info("No documents uploaded yet")

# Main search interface
st.header("🔍 Search & Compare")

# Search input
search_col1, search_col2 = st.columns([4, 1])

with search_col1:
    query = st.text_input(
        "Enter your search query",
        placeholder="e.g., What is the budget? Who attended the meeting?",
        help="Search across all uploaded documents with both methods"
    )

with search_col2:
    search_button = st.button("🔍 Compare", type="primary", use_container_width=True)

# Perform comparison search
if search_button and query:
    if not st.session_state.documents:
        st.warning("⚠️ Please upload some documents first")
    else:
        with st.spinner("Searching with both methods..."):
            try:
                # Call the comparison endpoint
                response = requests.post(
                    f"{API_URL}/api/search/compare",
                    json={"query": query, "limit": num_results},
                    timeout=30
                )
                
                if response.status_code == 200:
                    st.session_state.comparison_results = response.json()
                    st.success("✅ Search completed")
                else:
                    st.error(f"Search failed: {response.text}")
            except Exception as e:
                st.error(f"Search error: {str(e)}")

# Display comparison results
if st.session_state.comparison_results:
    results = st.session_state.comparison_results
    
    # Search summary
    st.divider()
    
    # Query and metadata
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Query", f'"{results.get("query", "")}"')
    with col2:
        st.metric("TF-IDF Results", len(results.get("tfidf_results", [])))
    with col3:
        st.metric("Semantic Results", len(results.get("semantic_results", [])))
    
    # Method status indicators
    if show_methods:
        method_col1, method_col2 = st.columns(2)
        
        with method_col1:
            tfidf_status = results.get("tfidf_status", "unknown")
            if tfidf_status == "success":
                st.success("🎯 TF-IDF: Working")
            else:
                st.error(f"❌ TF-IDF: {tfidf_status}")
        
        with method_col2:
            semantic_status = results.get("semantic_status", "unknown")
            if semantic_status == "success":
                st.success("🧠 Semantic: Working")
            elif semantic_status == "api_fallback":
                st.info("🌐 Semantic: Using API")
            else:
                st.warning(f"⚠️ Semantic: {semantic_status}")
    
    # Side-by-side comparison
    st.subheader("📊 Results Comparison")
    
    left_col, right_col = st.columns(2)
    
    # TF-IDF Results (Left Column)
    with left_col:
        st.markdown("### 🎯 TF-IDF (Keyword) Search")
        st.caption("Traditional keyword matching based on term frequency")
        
        tfidf_results = results.get("tfidf_results", [])
        
        if tfidf_results:
            for i, result in enumerate(tfidf_results, 1):
                with st.expander(
                    f"Result {i} - {result.get('filename', 'Unknown')} "
                    f"{'(' + str(round(result.get('relevance_score', 0), 3)) + ')' if show_scores else ''}"
                ):
                    st.text_area(
                        "Text Content",
                        result.get('text', ''),
                        height=150,
                        disabled=True,
                        key=f"tfidf_result_{i}"
                    )
                    
                    if show_scores:
                        score_col1, score_col2, score_col3 = st.columns(3)
                        with score_col1:
                            st.caption(f"📊 Relevance: {result.get('relevance_score', 'N/A')}")
                        with score_col2:
                            st.caption(f"🎯 OCR Confidence: {result.get('confidence', 'N/A')}%")
                        with score_col3:
                            st.caption(f"📄 Doc ID: {result.get('document_id', 'N/A')}")
        else:
            st.info("No TF-IDF results found")
            st.caption("Try different keywords or check spelling")
    
    # Semantic Results (Right Column)
    with right_col:
        st.markdown("### 🧠 Semantic Search")
        st.caption("AI-powered understanding of meaning and context")
        
        semantic_results = results.get("semantic_results", [])
        
        if semantic_results:
            for i, result in enumerate(semantic_results, 1):
                with st.expander(
                    f"Result {i} - {result.get('filename', 'Unknown')} "
                    f"{'(' + str(round(result.get('relevance_score', 0), 3)) + ')' if show_scores else ''}"
                ):
                    st.text_area(
                        "Text Content",
                        result.get('text', ''),
                        height=150,
                        disabled=True,
                        key=f"semantic_result_{i}"
                    )
                    
                    if show_scores:
                        score_col1, score_col2, score_col3 = st.columns(3)
                        with score_col1:
                            st.caption(f"📊 Relevance: {result.get('relevance_score', 'N/A')}")
                        with score_col2:
                            st.caption(f"🎯 OCR Confidence: {result.get('confidence', 'N/A')}%")
                        with score_col3:
                            st.caption(f"📄 Doc ID: {result.get('document_id', 'N/A')}")
        else:
            st.info("No semantic results found")
            st.caption("Model may still be loading or query needs refinement")
    
    # Results analysis
    st.divider()
    st.subheader("📈 Analysis")
    
    # Create comparison metrics
    analysis_col1, analysis_col2, analysis_col3 = st.columns(3)
    
    with analysis_col1:
        # Overlap analysis
        tfidf_texts = {r.get('text', '') for r in tfidf_results}
        semantic_texts = {r.get('text', '') for r in semantic_results}
        
        overlap = len(tfidf_texts.intersection(semantic_texts))
        total_unique = len(tfidf_texts.union(semantic_texts))
        
        if total_unique > 0:
            overlap_percentage = (overlap / total_unique) * 100
            st.metric("Result Overlap", f"{overlap}/{total_unique}", f"{overlap_percentage:.1f}%")
        else:
            st.metric("Result Overlap", "0/0", "N/A")
    
    with analysis_col2:
        # Average scores
        tfidf_avg = sum(r.get('relevance_score', 0) for r in tfidf_results) / max(len(tfidf_results), 1)
        semantic_avg = sum(r.get('relevance_score', 0) for r in semantic_results) / max(len(semantic_results), 1)
        
        st.metric("Avg TF-IDF Score", f"{tfidf_avg:.3f}")
        st.metric("Avg Semantic Score", f"{semantic_avg:.3f}")
    
    with analysis_col3:
        # Response times (if available)
        tfidf_time = results.get("tfidf_time", 0)
        semantic_time = results.get("semantic_time", 0)
        
        if tfidf_time > 0:
            st.metric("TF-IDF Time", f"{tfidf_time:.2f}s")
        if semantic_time > 0:
            st.metric("Semantic Time", f"{semantic_time:.2f}s")
    
    # Detailed comparison table
    if st.checkbox("Show detailed comparison table"):
        st.subheader("📋 Detailed Results Table")
        
        # Combine results for table
        table_data = []
        
        # Add TF-IDF results
        for i, result in enumerate(tfidf_results):
            table_data.append({
                "Rank": i + 1,
                "Method": "TF-IDF",
                "Score": round(result.get('relevance_score', 0), 3),
                "Filename": result.get('filename', 'Unknown'),
                "OCR Confidence": f"{result.get('confidence', 'N/A')}%",
                "Text Preview": result.get('text', '')[:100] + "..." if len(result.get('text', '')) > 100 else result.get('text', '')
            })
        
        # Add semantic results
        for i, result in enumerate(semantic_results):
            table_data.append({
                "Rank": i + 1,
                "Method": "Semantic",
                "Score": round(result.get('relevance_score', 0), 3),
                "Filename": result.get('filename', 'Unknown'),
                "OCR Confidence": f"{result.get('confidence', 'N/A')}%",
                "Text Preview": result.get('text', '')[:100] + "..." if len(result.get('text', '')) > 100 else result.get('text', '')
            })
        
        if table_data:
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No results to display in table")

# Quick search examples
st.divider()
st.subheader("💡 Try These Example Queries")

example_queries = [
    "budget and expenses",
    "meeting attendees", 
    "project deadline",
    "contact information",
    "financial summary",
    "action items"
]

cols = st.columns(3)
for i, example in enumerate(example_queries):
    with cols[i % 3]:
        if st.button(f"📝 {example}", key=f"example_{i}", use_container_width=True):
            # Set the query and trigger search
            st.session_state.example_query = example
            st.rerun()

# Handle example query selection
if 'example_query' in st.session_state:
    query = st.session_state.example_query
    del st.session_state.example_query
    st.rerun()

# Footer with tips
st.divider()
with st.expander("🔍 Search Tips & Method Differences"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **TF-IDF (Keyword) Search:**
        - Matches exact words and phrases
        - Good for specific terms and names
        - Fast and consistent
        - Works well with acronyms
        - Example: "budget 2024" finds documents with those exact words
        """)
    
    with col2:
        st.markdown("""
        **Semantic Search:**
        - Understands meaning and context
        - Finds related concepts even with different words
        - Better for natural language queries
        - Handles synonyms and paraphrasing
        - Example: "financial plan" might find "budget" documents
        """)

# Helper function to load documents
def load_documents():
    """Load documents from API"""
    try:
        response = requests.get(f"{API_URL}/api/documents", timeout=10)
        if response.status_code == 200:
            data = response.json()
            st.session_state.documents = data.get('documents', [])
    except:
        st.session_state.documents = []

# Load documents on startup
if not st.session_state.documents:
    load_documents()