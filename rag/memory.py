conversation_memory = {}

def get_memory(user_id):
    return conversation_memory.get(user_id, [])

def update_memory(user_id, question, answer):
    if user_id not in conversation_memory:
        conversation_memory[user_id] = []

    conversation_memory[user_id].append({
        "question": question,
        "answer": answer
    })

    # Keep only last 3 conversations
    conversation_memory[user_id] = conversation_memory[user_id][-3:]