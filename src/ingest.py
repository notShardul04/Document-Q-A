import os
import pickle
import requests
import fitz  # PyMuPDF
from dotenv import load_dotenv
from google import genai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rank_bm25 import BM25Okapi

load_dotenv()

# Configuration
DATA_DIR = "data"
DB_DIR = "qdrant_db"
COLLECTION_NAME = "askmybook"
BM25_PATH = os.path.join(DB_DIR, "bm25_index.pkl")

# Source PDFs to download
DOCUMENTS = {
    "Attention_Is_All_You_Need": "https://arxiv.org/pdf/1706.03762",
    "BERT_Language_Understanding": "https://arxiv.org/pdf/1810.04805"
}

def download_pdfs():
    """Downloads the target PDF papers if not already present."""
    os.makedirs(DATA_DIR, exist_ok=True)
    downloaded_paths = {}
    for doc_name, url in DOCUMENTS.items():
        pdf_path = os.path.join(DATA_DIR, f"{doc_name}.pdf")
        if not os.path.exists(pdf_path):
            print(f"Downloading {doc_name} from {url}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(pdf_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Saved to {pdf_path}")
        else:
            print(f"{doc_name}.pdf already exists in {DATA_DIR}.")
        downloaded_paths[doc_name] = pdf_path
    return downloaded_paths

def parse_pdf(pdf_path, doc_name):
    """Extracts pages and text content from a PDF file."""
    print(f"Parsing {pdf_path}...")
    doc = fitz.open(pdf_path)
    pages_data = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            pages_data.append({
                "document_name": doc_name,
                "page_number": page_num + 1,  # 1-indexed for citation
                "text": text
            })
    print(f"Parsed {len(pages_data)} pages from {doc_name}.")
    return pages_data

def chunk_text(pages_data, chunk_size=800, overlap=200):
    """Chunks page text into smaller sliding window segments, maintaining page metadata."""
    chunks = []
    for page in pages_data:
        text = page["text"]
        doc_name = page["document_name"]
        page_num = page["page_number"]
        
        # Simple sliding window chunking
        i = 0
        while i < len(text):
            chunk = text[i:i + chunk_size].strip()
            # If the chunk is too short and not the first chunk, skip or append to previous if possible
            if len(chunk) < 100 and i > 0:
                break
                
            chunks.append({
                "document_name": doc_name,
                "page_number": page_num,
                "text": chunk
            })
            i += (chunk_size - overlap)
            
    print(f"Created {len(chunks)} chunks total.")
    return chunks

def get_embeddings(texts):
    """Fetches dense embeddings for a list of texts using the Gemini API."""
    import time
    print("Generating dense embeddings using gemini-embedding-2...")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
        
    client = genai.Client(api_key=api_key)
    
    # Let's batch the embedding generation
    batch_size = 50
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}...")
        
        retries = 5
        delay = 10  # Seconds
        contents_list = [genai.types.Content(parts=[genai.types.Part(text=t)]) for t in batch]
        for attempt in range(retries):
            try:
                response = client.models.embed_content(
                    model="gemini-embedding-2",
                    contents=contents_list
                )
                break
            except Exception as e:
                if "429" in str(e) and attempt < retries - 1:
                    print(f"Rate limit hit (429). Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise e
                    
        # response.embeddings is a list of ContentEmbedding objects
        for embedding in response.embeddings:
            all_embeddings.append(embedding.values)
            
    return all_embeddings


def build_vector_db(chunks, embeddings):
    """Stores text chunks and their embeddings in Qdrant (local file mode)."""
    os.makedirs(DB_DIR, exist_ok=True)
    print(f"Connecting to Qdrant local storage at {DB_DIR}...")
    qdrant_client = QdrantClient(path=DB_DIR)
    
    # Recreate the collection if it exists to ensure freshness
    if qdrant_client.collection_exists(COLLECTION_NAME):
        print(f"Recreating existing collection: {COLLECTION_NAME}")
        qdrant_client.delete_collection(COLLECTION_NAME)
        
    # gemini-embedding-2 has 3072 dimensions
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=3072, distance=Distance.COSINE)
    )
    
    points = []
    for idx, (chunk, vector) in enumerate(zip(chunks, embeddings)):
        points.append(PointStruct(
            id=idx,
            vector=vector,
            payload={
                "text": chunk["text"],
                "document_name": chunk["document_name"],
                "page_number": chunk["page_number"]
            }
        ))
        
    print(f"Upserting {len(points)} points into Qdrant collection '{COLLECTION_NAME}'...")
    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
    print("Vector database built successfully.")

def build_bm25_index(chunks):
    """Builds and serializes a BM25 index on the text chunks for sparse retrieval."""
    print("Building BM25 index...")
    # Tokenize by simple splitting for BM25
    tokenized_corpus = [chunk["text"].lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Save the index and the raw chunks together
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)
    print(f"BM25 index saved to {BM25_PATH}.")

def get_local_pdfs():
    """Finds all PDF files in the DATA_DIR and returns a dictionary of doc_name -> path."""
    pdf_paths = {}
    if os.path.exists(DATA_DIR):
        for filename in os.listdir(DATA_DIR):
            if filename.lower().endswith(".pdf"):
                doc_name = os.path.splitext(filename)[0]
                pdf_paths[doc_name] = os.path.join(DATA_DIR, filename)
    return pdf_paths

def run_pipeline():
    """Runs the entire ingestion and indexing pipeline."""
    print("=== AskMyBook Ingestion Pipeline ===")
    
    # 1. Download documents
    download_pdfs()
    
    # 2. Find all local PDFs in DATA_DIR
    local_pdfs = get_local_pdfs()
    print(f"Found {len(local_pdfs)} PDFs to ingest: {list(local_pdfs.keys())}")
    
    # 3. Parse PDFs
    all_pages = []
    for doc_name, pdf_path in local_pdfs.items():
        all_pages.extend(parse_pdf(pdf_path, doc_name))
        
    # 4. Chunk text
    chunks = chunk_text(all_pages)
    
    # 5. Generate embeddings
    texts = [c["text"] for c in chunks]
    embeddings = get_embeddings(texts)
    
    # 6. Build DBs
    build_vector_db(chunks, embeddings)
    build_bm25_index(chunks)
    
    print("=== Pipeline Complete! Data is ready for retrieval. ===")

if __name__ == "__main__":
    run_pipeline()
