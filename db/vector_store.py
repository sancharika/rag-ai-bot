import sqlite3
import numpy as np
from config import DB_PATH

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT,
    embedding BLOB,
    source TEXT
)
""")
conn.commit()


def add_document(text, embedding, source):
    embedding_blob = embedding.tobytes()
    cursor.execute(
        "INSERT INTO documents (text, embedding, source) VALUES (?, ?, ?)",
        (text, embedding_blob, source)
    )
    conn.commit()


def get_all_documents():
    cursor.execute("SELECT text, embedding, source FROM documents")
    rows = cursor.fetchall()

    docs = []
    for text, emb_blob, source in rows:
        emb = np.frombuffer(emb_blob, dtype=np.float32)
        docs.append((text, emb, source))
    return docs

def get_document_count():
    cursor.execute("SELECT COUNT(*) FROM documents")
    return cursor.fetchone()[0]