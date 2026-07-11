import streamlit as st
import os
import shutil
import sys

# Ensure workspace packages can be imported
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

# Set page configuration first
st.set_page_config(
    page_title="AskMyBook - Academic RAG Assistant",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Imports from src
try:
    from src.ingest import run_pipeline, get_local_pdfs, COLLECTION_NAME, DB_DIR
    from src.retrieve import retrieve_hybrid
    from src.generate import generate_answer
    from qdrant_client import QdrantClient
except ImportError as e:
    st.error(f"Failed to import backend modules: {e}. Please ensure src/ directory and dependencies are correct.")

# Inject Custom CSS for premium aesthetics
st.markdown("""
<style>
    /* Google Font Import */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    /* Main body styling */
    .stApp {
        background-color: #0b0c10;
        color: #c5c6c7;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Headers styling */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        color: #ffffff;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    /* Gaining control of sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #1f2833;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #66fcf1;
    }
    
    /* Custom buttons with gradient and animations */
    .stButton > button {
        background: linear-gradient(135deg, #1f4068 0%, #162447 100%);
        color: #66fcf1;
        border: 1px solid #66fcf1;
        border-radius: 8px;
        padding: 10px 24px;
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        font-size: 16px;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #66fcf1 0%, #45a29e 100%);
        color: #0b0c10;
        border-color: #66fcf1;
        transform: translateY(-2px);
        box-shadow: 0 0 15px rgba(102, 252, 241, 0.4);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }

    /* Glassmorphic cards for context chunks */
    .chunk-card {
        background: rgba(31, 40, 51, 0.4);
        border: 1px solid rgba(102, 252, 241, 0.15);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 16px;
        backdrop-filter: blur(8px);
        transition: all 0.3s ease;
    }
    
    .chunk-card:hover {
        border-color: #66fcf1;
        background: rgba(31, 40, 51, 0.6);
        box-shadow: 0 4px 12px rgba(102, 252, 241, 0.1);
    }
    
    .chunk-meta {
        font-size: 0.85em;
        color: #45a29e;
        margin-bottom: 10px;
        display: flex;
        justify-content: space-between;
        font-weight: 500;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        padding-bottom: 6px;
    }
    
    .chunk-text {
        font-size: 0.95em;
        line-height: 1.6;
        color: #e5e7eb;
    }
    
    .doc-badge {
        background-color: #1f4068;
        color: #66fcf1;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8em;
    }
    
    .page-badge {
        background-color: rgba(255, 255, 255, 0.1);
        color: #ffffff;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8em;
    }
    
    .score-badge {
        background-color: rgba(102, 252, 241, 0.1);
        color: #66fcf1;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8em;
    }

    /* Styled main title section */
    .title-container {
        background: linear-gradient(135deg, rgba(31, 40, 51, 0.8) 0%, rgba(11, 12, 16, 0.9) 100%);
        border: 1px solid rgba(102, 252, 241, 0.2);
        padding: 30px;
        border-radius: 16px;
        margin-bottom: 30px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    
    .title-gradient {
        background: linear-gradient(45deg, #ffffff, #66fcf1, #45a29e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.2em;
        font-weight: 800;
        margin-bottom: 10px;
    }
    
    /* Styled container for the model responses */
    .answer-container {
        background: rgba(31, 40, 51, 0.5);
        border-left: 4px solid #66fcf1;
        padding: 20px;
        border-radius: 0 12px 12px 0;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        font-size: 1.05em;
        line-height: 1.7;
        color: #f3f4f6;
    }
    
    /* Input field styling adjustments */
    .stTextInput > div > div > input {
        background-color: #1f2833;
        color: #ffffff;
        border: 1px solid rgba(102, 252, 241, 0.2);
        border-radius: 8px;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #66fcf1;
        box-shadow: 0 0 10px rgba(102, 252, 241, 0.25);
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE SETUP -----------------
if "ingestion_progress" not in st.session_state:
    st.session_state.ingestion_progress = None

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# API Key handling
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    # Allow user to set key in sidebar if not in environment
    st.sidebar.warning("⚠️ GEMINI_API_KEY is not set in environmental variables.")
    user_api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")
    if user_api_key:
        os.environ["GEMINI_API_KEY"] = user_api_key
        st.sidebar.success("API Key applied for this session!")
        st.rerun()

# ----------------- SIDEBAR: DOCUMENT MANAGER -----------------
st.sidebar.title("📚 Document Manager")

# View ingested files
local_pdfs = get_local_pdfs()
st.sidebar.subheader("Active Documents")
if local_pdfs:
    for doc_name in local_pdfs.keys():
        st.sidebar.markdown(f"📄 `{doc_name}`")
else:
    st.sidebar.info("No documents found. Ingestion defaults will download automatically.")

# File Uploader
st.sidebar.subheader("Upload PDF")
uploaded_files = st.sidebar.file_uploader(
    "Choose academic papers or textbook chapters", 
    type=["pdf"], 
    accept_multiple_files=True
)

# Ingest action
if st.sidebar.button("⚡ Ingest & Index Documents"):
    if uploaded_files:
        with st.spinner("Saving uploaded files to disk..."):
            for uploaded_file in uploaded_files:
                file_path = os.path.join("data", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
        st.sidebar.success(f"Saved {len(uploaded_files)} files!")
    
    # Run the ingestion pipeline
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    
    try:
        status_text.text("Parsing, chunking, and embedding PDFs...")
        progress_bar.progress(30)
        
        # Run pipeline
        run_pipeline()
        
        progress_bar.progress(100)
        status_text.text("Ingestion completed successfully!")
        st.sidebar.success("Database updated! All systems ready.")
        st.rerun()
    except Exception as e:
        status_text.text("Ingestion failed!")
        st.sidebar.error(f"Error: {e}")

# Reset Database Action
st.sidebar.subheader("Danger Zone")
if st.sidebar.button("🗑️ Reset Database & Files", help="Deletes all PDF documents and wipes the databases"):
    # Delete custom PDFs
    for filename in os.listdir("data"):
        file_path = os.path.join("data", filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            st.sidebar.error(f"Failed to delete {filename}: {e}")
            
    # Delete DB folder
    if os.path.exists(DB_DIR):
        try:
            shutil.rmtree(DB_DIR)
            st.sidebar.success("Wiped vector store & indices.")
        except Exception as e:
            st.sidebar.error(f"Failed to wipe vector store: {e}")
            
    st.rerun()

# Settings Slider
st.sidebar.subheader("Retrieval Settings")
top_k = st.sidebar.slider("Number of retrieved chunks (top-k)", min_value=1, max_value=10, value=4)


# ----------------- MAIN PANEL -----------------
# Elegant Header
st.markdown("""
<div class="title-container">
    <div class="title-gradient">AskMyBook 📖</div>
    <div style="font-size: 1.25em; color: #45a29e; font-weight: 500;">
        Academic RAG Assistant — Ask questions and compare notes across your PDF library.
    </div>
</div>
""", unsafe_allow_html=True)

# Main UI Tabs
tab_qa, = st.tabs(["🕵️ Q&A Workspace"])

# ----------------- TAB 1: STANDARD Q&A -----------------
with tab_qa:
    st.subheader("Query Your Ingested Library")
    
    # Document Selection Filter
    doc_options = ["All Documents"] + list(local_pdfs.keys())
    selected_doc = st.selectbox("Select document to search within", doc_options)
    
    query = st.text_input("Ask a question about the study materials:", placeholder="e.g. What are the key parameters of the transformer?")
    
    if st.button("🔍 Get Answer", key="qa_btn"):
        if not query.strip():
            st.warning("Please enter a valid query.")
        else:
            with st.spinner("Searching corpus and synthesizing answer..."):
                doc_filter = None if selected_doc == "All Documents" else selected_doc
                
                try:
                    # Run hybrid retrieval
                    contexts = retrieve_hybrid(query, top_k=top_k, doc_filter=doc_filter)
                    
                    if not contexts:
                        st.info("No matching contexts found in the database. Try ingesting documents first.")
                    else:
                        # Generate answer
                        answer = generate_answer(query, contexts)
                        
                        # Display Answer
                        st.markdown("### Answer")
                        st.markdown(f'<div class="answer-container">{answer}</div>', unsafe_allow_html=True)
                        
                        # Display Source Passages
                        with st.expander("📚 Retrieved Source Passages (Evidence)", expanded=True):
                            for idx, ctx in enumerate(contexts):
                                rrf_score_str = f"{ctx.get('rrf_score', 0):.4f}" if 'rrf_score' in ctx else 'N/A'
                                st.markdown(f"""
                                <div class="chunk-card">
                                    <div class="chunk-meta">
                                        <span><span class="doc-badge">📄 {ctx['document_name']}</span> <span class="page-badge">Page {ctx['page_number']}</span></span>
                                        <span class="score-badge">Fusion Rank Score: {rrf_score_str}</span>
                                    </div>
                                    <div class="chunk-text">"{ctx['text']}"</div>
                                </div>
                                """, unsafe_allow_html=True)
                except FileNotFoundError:
                    st.error("No database index found. Please click '⚡ Ingest & Index Documents' in the sidebar first to build the index.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
