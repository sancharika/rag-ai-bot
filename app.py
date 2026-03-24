import os
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

from config import TELEGRAM_BOT_TOKEN, DOCS_PATH
from rag.embedder import embed_texts, embed_query
from rag.retriever import retrieve
from rag.generator import generate_answer
from db.vector_store import add_document
from vision.caption import generate_caption
from db.faiss_index import build_index
from rag.memory import get_memory, update_memory

# -------- Load + Index Docs -------- #

def chunk_text(text, size=300):
    return [text[i:i+size] for i in range(0, len(text), size)]


def load_documents():
    chunks = []

    for file in os.listdir(DOCS_PATH):
        with open(os.path.join(DOCS_PATH, file), "r", encoding="utf-8") as f:
            text = f.read()
            chunks.extend(chunk_text(text))

    return chunks


def index_documents():
    for file in os.listdir(DOCS_PATH):
        with open(os.path.join(DOCS_PATH, file), "r", encoding="utf-8") as f:
            text = f.read()
            chunks = chunk_text(text)

            embeddings = embed_texts(chunks)

            for chunk, emb in zip(chunks, embeddings):
                add_document(chunk, emb, file)  # store filename as source

    print("Indexing complete.")


# -------- Telegram Handlers -------- #
user_states = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 RAG Bot Ready!\n\nUse:\n/ask <your question>\n/image - Upload an image for captioning\n/help"
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "/ask <query> - Ask questions from docs\n/image - Upload an image for captioning (or you can directly send one for captioning)\n/help - Show this message"
    )


async def ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    query = " ".join(context.args)

    if not query:
        await update.message.reply_text("❗ Please provide a question.")
        return

    await update.message.reply_text("🔍 Searching...")

    # Hybrid Search (Vector + Keyword Search)
    query_emb = embed_query(query)
    chunks, chunk_sources = retrieve(query_emb, query)
    
    memory = get_memory(user_id)

    answer = generate_answer(query, chunks, memory)

    update_memory(user_id, query, answer)

    response = f"💡 {answer}\n\n📄 Sources:\n"

    unique_sources = list(set(chunk_sources))
    for i, src in enumerate(unique_sources):
        response += f"{i+1}. {src}\n"

    await update.message.reply_text(response)

async def image_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id

    user_states[user_id] = "WAITING_FOR_IMAGE"

    await update.message.reply_text("📸 Please upload an image.")

async def image_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message.photo:
        await update.message.reply_text("❗ Please send an image.")
        return

    await update.message.reply_text("🖼 Processing image...")

    photo = update.message.photo[-1]
    file = await photo.get_file()

    file_path = "temp.jpg"
    await file.download_to_drive(file_path)

    caption, tags = generate_caption(file_path)

    response = f"🖼 Caption: {caption}\n🏷 Tags: {tags}"
    await update.message.reply_text(response)


# -------- Main -------- #

def main():
    # Only index if DB is empty
    print("Indexing documents...")
    index_documents()   # Save to SQLite

    print("Building FAISS index...")
    build_index()       # Load from SQLite → FAISS


    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("ask", ask))
    app.add_handler(CommandHandler("image", image_command))
    app.add_handler(MessageHandler(filters.PHOTO, image_handler))

    print("Bot running...")
    app.run_polling()


if __name__ == "__main__":
    main()