import os
from google import genai
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

load_dotenv()

DB_DIR = "qdrant_db"
COLLECTION_NAME = "askmybook"

_qdrant_client = None

def get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(path=DB_DIR)
    return _qdrant_client

def get_query_embedding(query):
    """Embeds the search query using gemini-embedding-2."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")
    client = genai.Client(api_key=api_key)
    response = client.models.embed_content(
        model="gemini-embedding-2",
        contents=query
    )
    return response.embeddings[0].values

def dense_retrieve(query_vector, limit=20, doc_filter=None):
    """Retrieves top matches from Qdrant using vector similarity."""
    client = get_qdrant_client()
    
    qdrant_filter = None
    if doc_filter:
        qdrant_filter = Filter(
            must=[
                FieldCondition(
                    key="document_name",
                    match=MatchValue(value=doc_filter)
                )
            ]
        )
        
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=qdrant_filter,
        limit=limit
    ).points
    
    # Format results
    retrieved = []
    for hit in results:
        retrieved.append({
            "text": hit.payload["text"],
            "document_name": hit.payload["document_name"],
            "page_number": hit.payload["page_number"],
            "score": hit.score
        })
    return retrieved

if __name__ == "__main__":
    try:
        q_vec = get_query_embedding("What are the key parameters of the transformer?")
        results = dense_retrieve(q_vec, limit=3)
        print("Dense retrieval successful:")
        for r in results:
            print(f"- [{r['document_name']}, p.{r['page_number']}] (Score: {r['score']:.4f})")
    except Exception as e:
        print(f"Dense retrieve failed: {e}")
