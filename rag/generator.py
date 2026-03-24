import ollama
from config import LLM_MODEL

def generate_answer(query, context_chunks, memory):
    context = "\n".join(context_chunks)
    history = ""
    for m in memory:
        history += f"User: {m['question']}\nAssistant: {m['answer']}\n"

    prompt = f"""
You are an AI assistant with memory that answers user questions.

Follow these rules strictly:

1. First, check the provided CONTEXT and answer using only the context if the answer is present there.
2. If the answer is NOT in the context, Then use Conversation History
3. If the answer is NOT in the context, Then use general knowledge but you know the answer from general knowledge, answer clearly and say: "This answer is based on general knowledge."
4. If you are NOT sure or do not know the answer, say:
   "I don't know yet. Please reach out to me at sancharikagmail.com and I will help you with this."

5. Do NOT hallucinate.
6. Do NOT make up information.
7. Keep answers clear, structured, and professional.

Conversation History:
{history}

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    print("LLM Response:", response)

    return response["message"]["content"]