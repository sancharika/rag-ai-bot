import faiss
import numpy as np
from db.vector_store import get_all_documents

index = None
texts = []
metadatas = []

def build_index():
    global index, texts, metadatas

    docs = get_all_documents()

    if not docs:
        print("No documents in DB. Index not created.")
        index = None
        return

    embeddings = []

    for text, emb, source in docs:
        texts.append(text)
        metadatas.append({"source": source})
        embeddings.append(emb)

    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    print("FAISS index built with", index.ntotal, "documents")