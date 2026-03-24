import discord
import os

from rag.embedder import embed_query
from rag.retriever import retrieve
from rag.generator import generate_answer
from vision.caption import generate_caption
from rag.memory import get_memory, update_memory
from db.faiss_index import build_index
from app import index_documents 
from dotenv import load_dotenv
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")


intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)


@client.event
async def on_ready():
    print(f"Logged in as {client.user}")
    
    print("Indexing documents...")
    index_documents()

    print("Building FAISS index...")
    build_index()

    print("Bot ready.")


@client.event
async def on_message(message):
    if message.author == client.user:
        return
    
    # -------- Help -------- #
    if message.content.startswith("/help"):
        help_text = (
            "🤖 **RAG Bot Help**\n"
            "/ask <question> - Ask a question\n"
            "/image - Upload an image for captioning\n"
            "/help - Show this help message"
        )
        await message.channel.send(help_text)
        return

    # -------- RAG -------- #
    if message.content.startswith("/ask"):
        user_id = message.author.id
        query = message.content.replace("/ask", "").strip()

        if not query:
            await message.channel.send("❗ Provide a question.")
            return

        await message.channel.send("🔍 Searching...")

        # Hybrid Search (Vector + Keyword Search)
        query_emb = embed_query(query)
        chunks, chunk_sources = retrieve(query_emb, query)
        
        memory = get_memory(user_id)

        # If no relevant chunks found, generate answer from memory alone
        if not chunks:
            answer = generate_answer(query, [], memory)
            await message.channel.send(f"💡 {answer}")
            return

        answer = generate_answer(query, chunks, memory)

        update_memory(user_id, query, answer)

        response = f"💡 {answer}\n\n📄 Sources:\n"

        unique_sources = list(set(chunk_sources))
        for i, src in enumerate(unique_sources):
            response += f"{i+1}. {src}\n"

        await message.channel.send(response)

    # -------- IMAGE -------- #
    if message.content.startswith("/image") or message.attachments:

        if not message.attachments:
            await message.channel.send("❗ Please upload an image with /image command.")
            return

        for attachment in message.attachments:

            if not attachment.filename.lower().endswith((".jpg", ".jpeg", ".png")):
                await message.channel.send(f"❌ {attachment.filename} is not a supported image.")
                continue

            await message.channel.send(f"🖼 Processing {attachment.filename}...")

            try:
                file_path = f"temp_{attachment.id}.jpg"
                await attachment.save(file_path)

                caption, tags = generate_caption(file_path)

                response = (
                    f"**🖼 Image:** {attachment.filename}\n"
                    f"**Caption:** {caption}\n"
                    f"**Tags:** {', '.join(tags)}"
                )

                await message.channel.send(response)

                # Delete temp file after processing
                if os.path.exists(file_path):
                    os.remove(file_path)

            except Exception as e:
                await message.channel.send(f"❌ Error processing {attachment.filename}")
                print("Image error:", e)
                
client.run(DISCORD_TOKEN)