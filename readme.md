# RAG AI Bot (Telegram + Discord + Vision)

A multi-platform AI assistant built using Retrieval-Augmented Generation (RAG) with FAISS, Hybrid Search, Re-ranking, Conversation Memory, and Image Captioning.

The bot can answer questions from your documents, remember conversation context, and describe uploaded images.

---
![Demo](demo.gif)
---

## Features

- Document Question Answering (RAG)
- Hybrid Search (Vector + Keyword)
- FAISS Vector Search
- Re-ranking using Cross-Encoder
- Conversation Memory
- Source Citation
- Image Captioning + Tags
- Telegram Bot Integration
- Discord Bot Integration
- SQLite Vector Storage
- Multi-document support

---

## Models and APIs Used

| Purpose | Model / Tool |
|--------|---------------|
| Embeddings | Qwen/Qwen3-Embedding-0.6B |
| LLM | mistral:7b via Ollama |
| Re-ranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Vector Database | FAISS |
| Database | SQLite |
| Image Captioning | BLIP Vision Model |
| Bots | Telegram API, Discord API |

---

## Project Structure

```
rag-bot/
│
├── app.py                  # Telegram bot
├── discord_bot.py          # Discord bot
├── config.py               # Config & model settings
│
├── rag/
│   ├── embedder.py
│   ├── retriever.py
│   ├── generator.py
│   ├── memory.py
│
├── db/
│   ├── vector_store.py     # SQLite storage
│   ├── faiss_index.py      # FAISS index
│
├── vision/
│   ├── caption.py          # Image captioning
│
├── data/docs/              # Documents for RAG
├── db/vectors.db           # SQLite DB
```

---

## Configuration

Update `config.py`:

```python
TELEGRAM_BOT_TOKEN = "YOUR_TOKEN"
OLLAMA = "YOUR_OLLAMA_KEY"

EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
LLM_MODEL = "mistral:7b"

DB_PATH = "db/vectors.db"
DOCS_PATH = "data/docs/"
TOP_K = 3
```

---

## How to Run Locally

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Download NLTK data:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

---

### 2. Start Ollama

```bash
ollama run mistral:7b
```

---

### 3. Add Documents

Place `.md` files inside:

```
data/docs/
```

Example:

```
data/docs/faq.md
data/docs/product.md
```

---

### 4. Run Telegram Bot

```bash
python app.py
```

---

### 5. Run Discord Bot

```bash
python discord_bot.py
```

---

## Bot Commands

| Command | Description |
|--------|-------------|
| /ask <question> | Ask question from documents |
| /image | Upload image for caption |
| /help | Show help |

---

## System Architecture

```
                +-------------------+
                |   User (TG/DS)    |
                +---------+---------+
                          |
                          v
                 +--------+--------+
                 |   Query Input   |
                 +--------+--------+
                          |
                          v
                 +--------+--------+
                 |   Embedding     |
                 | Qwen Embedding  |
                 +--------+--------+
                          |
                          v
        +-----------------+------------------+
        |                                    |
        v                                    v
+---------------+                  +----------------+
| FAISS Vector  |                  | Keyword Search |
|   Search      |                  |                |
+-------+-------+                  +--------+-------+
        |                                   |
        +---------------+-------------------+
                        |
                        v
                 +------+------+
                 |   Re-ranker |
                 | CrossEncoder|
                 +------+------+
                        |
                        v
                 +------+------+
                 |  Top Chunks |
                 +------+------+
                        |
                        v
                 +------+------+
                 | Conversation|
                 |   Memory    |
                 +------+------+
                        |
                        v
                 +------+------+
                 |    LLM      |
                 |  Mistral 7B |
                 +------+------+
                        |
                        v
                 +------+------+
                 | Answer +    |
                 |  Sources    |
                 +-------------+
```

---

## RAG Pipeline

1. User asks a question
2. Query converted to embedding
3. FAISS retrieves similar chunks
4. Keyword search retrieves matching text
5. Results combined (Hybrid search)
6. Re-ranked using Cross-Encoder
7. Memory added
8. Sent to LLM
9. LLM generates answer
10. Sources displayed

---

## Memory System

The bot stores conversation history per user, allowing contextual conversations across multiple messages.

---

## Vision System

Users can upload images and the bot returns:
- Caption
- Tags

---

## Example Response

- RAG System

```
💡  Based on the provided context, this system is designed for Retrieval-Augmented Generation (RAG) with Vision AI capabilities to create a smart assistant that can answer questions and describe visual content in real time. This means it can process natural language queries and also understand images to provide relevant responses. However, it does not have its own autonomous capabilities yet. Interaction is typically initiated by a user or another system.

📄 Sources:
product.md
```

- Caption
```
🖼 Image: Frame_1321316465.png
Caption: a black card with a qr - code on it
Tags: code, card, black
```

---

## Future Improvements

- Web search fallback
- Admin panel to upload documents
- PDF support
- Voice input
- Feedback system
- Analytics dashboard
- Deployment (AWS / Render)
- Docker support

---

## Summary

This project is a Multi-Platform AI Assistant that combines:

- Retrieval-Augmented Generation (RAG)
- Vector Search (FAISS)
- Hybrid Retrieval
- Re-ranking
- Conversation Memory
- Vision (Image Captioning)
- Telegram and Discord Integration

---
## Author

[Sancharika Debnath](https://sancharika.github.io/)

AI/ML Developer | RAG | LLM | Generative AI | Knowledge Graphs