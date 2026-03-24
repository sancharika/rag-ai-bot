import faiss
import numpy as np
import db.faiss_index as faiss_db
from config import TOP_K
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def keyword_search(query, docs):
    results = []
    for text in docs:
        if query.lower() in text.lower():
            results.append(text)
    return results[:2]

def rerank(query, chunks):
    pairs = [[query, chunk] for chunk in chunks]
    scores = reranker.predict(pairs)

    scored_chunks = list(zip(chunks, scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    return [chunk for chunk, score in scored_chunks[:3]]

SCORE_THRESHOLD = 0.30

def retrieve(query_embedding, query):
    if faiss_db.index is None:
        print("FAISS index not loaded")
        return [], []

    query_embedding = np.array([query_embedding]).astype("float32")
    faiss.normalize_L2(query_embedding)

    scores, indices = faiss_db.index.search(query_embedding, TOP_K)

    chunk_source_map = {}

    # Vector search
    for i, idx in enumerate(indices[0]):
        if scores[0][i] > SCORE_THRESHOLD:
            chunk = faiss_db.texts[idx]
            source = faiss_db.metadatas[idx]["source"]
            chunk_source_map[chunk] = source

    # Keyword search
    keyword_chunks = keyword_search(query, faiss_db.texts)
    for chunk in keyword_chunks:
        if chunk in faiss_db.texts:
            idx = faiss_db.texts.index(chunk)
            source = faiss_db.metadatas[idx]["source"]
            chunk_source_map[chunk] = source

    # Combine
    all_chunks = list(chunk_source_map.keys())

    # Re-rank
    reranked_chunks = rerank(query, all_chunks)

    # Sources
    final_sources = [chunk_source_map[c] for c in reranked_chunks]

    return reranked_chunks, final_sources