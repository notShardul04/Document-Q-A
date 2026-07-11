import os
import pickle
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from sentence_transformers import SentenceTransformer

load_dotenv()

DB_DIR = "qdrant_db"
COLLECTION_NAME = "askmybook"
BM25_PATH = os.path.join(DB_DIR, "bm25_index.pkl")
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"

# Cache clients, model and index
_qdrant_client = None
_bm25_index = None
_embedding_model = None

def get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(path=DB_DIR)
    return _qdrant_client

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model

def get_bm25_index():
    global _bm25_index
    if _bm25_index is None:
        if not os.path.exists(BM25_PATH):
            raise FileNotFoundError(f"BM25 index not found at {BM25_PATH}. Please run ingest.py first.")
        with open(BM25_PATH, "rb") as f:
            _bm25_index = pickle.load(f)
    return _bm25_index

def get_query_embedding(query):
    """Embeds the search query using local BAAI/bge-base-en-v1.5 model."""
    model = get_embedding_model()
    embedding = model.encode(query, normalize_embeddings=True)
    return embedding.tolist()

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

def sparse_retrieve(query_text, limit=20, doc_filter=None):
    """Retrieves top matches using BM25 BM25Okapi scoring."""
    index_data = get_bm25_index()
    bm25 = index_data["bm25"]
    chunks = index_data["chunks"]
    
    tokenized_query = query_text.lower().split()
    scores = bm25.get_scores(tokenized_query)
    
    # Associate scores with chunks and index position
    scored_chunks = []
    for idx, (chunk, score) in enumerate(zip(chunks, scores)):
        if doc_filter and chunk["document_name"] != doc_filter:
            continue
        scored_chunks.append({
            "text": chunk["text"],
            "document_name": chunk["document_name"],
            "page_number": chunk["page_number"],
            "score": score
        })
        
    # Sort by score descending
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    return scored_chunks[:limit]

def reciprocal_rank_fusion(dense_results, sparse_results, k=60, top_n=5):
    """Merges two ranked lists using Reciprocal Rank Fusion (RRF)."""
    rrf_scores = {}
    
    # Helper key to uniquely identify a chunk
    def get_chunk_key(res):
        return (res["document_name"], res["page_number"], res["text"])
    
    # Store items by key for reconstruction
    items_by_key = {}
    
    # Process dense results
    for rank, res in enumerate(dense_results):
        key = get_chunk_key(res)
        items_by_key[key] = res
        rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (k + (rank + 1)))
        
    # Process sparse results
    for rank, res in enumerate(sparse_results):
        key = get_chunk_key(res)
        items_by_key[key] = res
        rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (k + (rank + 1)))
        
    # Sort keys by fusion score
    sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    
    fused_results = []
    for key in sorted_keys[:top_n]:
        res = items_by_key[key]
        res["rrf_score"] = rrf_scores[key]
        fused_results.append(res)
        
    return fused_results

def retrieve_hybrid(query, top_k=5, doc_filter=None):
    """Executes hybrid retrieval using Qdrant (dense) and BM25 (sparse), fused by RRF."""
    # 1. Embed query
    query_vector = get_query_embedding(query)
    
    # 2. Dense search (fetch more candidate docs for fusion)
    dense_matches = dense_retrieve(query_vector, limit=top_k * 3, doc_filter=doc_filter)
    
    # 3. Sparse search
    sparse_matches = sparse_retrieve(query, limit=top_k * 3, doc_filter=doc_filter)
    
    # 4. Merge using Reciprocal Rank Fusion
    fused_matches = reciprocal_rank_fusion(dense_matches, sparse_matches, top_n=top_k)
    
    return fused_matches

if __name__ == "__main__":
    # Test retrieval
    try:
        results = retrieve_hybrid("What are the key parameters of the transformer?", top_k=3)
        print("Test Retrieval Successful:")
        for r in results:
            print(f"- [{r['document_name']}, p.{r['page_number']}] (RRF: {r['rrf_score']:.4f}): {r['text'][:100]}...")
    except Exception as e:
        print(f"Retrieval test failed (expected if DB not initialized yet): {e}")
