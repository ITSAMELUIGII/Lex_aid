import re
import sys
import pickle
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

DB_DIR = "./chroma_db"
BM25_PATH = "./bm25_index.pkl"

def reciprocal_rank_fusion(vector_results, bm25_results, k=60):
    """
    Combines sorted results from Vector DB and BM25 using RRF.
    k is a smoothing constant, generally set to 60.
    """
    fused_scores = {}
    
    # Assuming vector_results and bm25_results are lists of (Document, score/rank)
    # Actually, we will just use their rank.
    for rank, doc in enumerate(vector_results):
        doc_id = doc.page_content  # We use content as a naive unique ID for the union
        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (k + rank + 1)
        doc.metadata['fused_score'] = fused_scores[doc_id]
        
    for rank, doc in enumerate(bm25_results):
        doc_id = doc.page_content
        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (k + rank + 1)
        doc.metadata['fused_score'] = fused_scores[doc_id]

    # Map back to documents
    unique_docs = {doc.page_content: doc for doc in vector_results + bm25_results}
    
    # Sort docs by their fused_score
    sorted_docs = sorted(unique_docs.values(), key=lambda d: fused_scores[d.page_content], reverse=True)
    return sorted_docs

def hybrid_search(query, top_k=5):
    # 1. Load BM25 Index
    try:
        with open(BM25_PATH, 'rb') as f:
            data = pickle.load(f)
            bm25 = data['bm25']
            documents = data['documents']
    except Exception as e:
        print(f"Error loading BM25 index: {e}")
        return []

    # 2. Get BM25 top docs
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    # Sort documents by BM25 score decreasing
    top_bm25_indices = bm25_scores.argsort()[::-1][:10] # Get top 10 for blending
    bm25_results = [documents[i] for i in top_bm25_indices]
    
    # 3. Load Vector DB
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    
    # 4. Get Vector top docs
    vector_results = vectorstore.similarity_search(query, k=10)
    
    # 5. RRF Fusion
    final_docs = reciprocal_rank_fusion(vector_results, bm25_results, k=60)
    
    return final_docs[:top_k]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python hybrid_search.py \"your legal query\"")
        sys.exit(1)
        
    query = sys.argv[1]
    print(f"\n--- Searching for: '{query}' ---\n")
    
    results = hybrid_search(query, top_k=3)
    
    for i, doc in enumerate(results):
        print(f"\n======= Rank {i+1} =======")
        print(f"Act: {doc.metadata.get('act', 'Unknown')}")
        print(f"RRF Score: {doc.metadata.get('fused_score', 0):.4f}")
        print("Content Preview:")
        print(doc.page_content.strip()[:600] + "...")
