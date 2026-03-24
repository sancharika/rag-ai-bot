from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL

model = SentenceTransformer(EMBED_MODEL)

def embed_texts(texts):
    return model.encode(texts)

def embed_query(query):
    return model.encode([query])[0]