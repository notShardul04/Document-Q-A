import os
import requests
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

if __name__ == "__main__":
    print("=== AskMyBook Ingestion Pipeline (Step 1: Download) ===")
    download_pdfs()
