import os
from dotenv import load_dotenv
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
OLLAMA = os.getenv("OLLAMA")
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
LLM_MODEL = "mistral:7b"  # via Ollama

DB_PATH = "db/vectors.db"
DOCS_PATH = "data/docs/"
TOP_K = 3