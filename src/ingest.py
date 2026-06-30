import os
import requests
import fitz  # PyMuPDF
from dotenv import load_dotenv

load_dotenv()

# Configuration
DATA_DIR = "data"

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

def get_local_pdfs():
    """Finds all PDF files in the DATA_DIR and returns a dictionary of doc_name -> path."""
    pdf_paths = {}
    if os.path.exists(DATA_DIR):
        for filename in os.listdir(DATA_DIR):
            if filename.lower().endswith(".pdf"):
                doc_name = os.path.splitext(filename)[0]
                pdf_paths[doc_name] = os.path.join(DATA_DIR, filename)
    return pdf_paths

if __name__ == "__main__":
    print("=== AskMyBook Ingestion Pipeline (Step 2: Parsing) ===")
    download_pdfs()
    local_pdfs = get_local_pdfs()
    for name, path in local_pdfs.items():
        parse_pdf(path, name)
