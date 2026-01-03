import ollama
import chromadb
from chromadb.utils import embedding_functions
import re

# --- Configuration ---
JOURNAL_FILE = "journal.txt"
EMBEDDING_MODEL = "nomic-embed-text"
GENERATION_MODEL = "llama3.1:8b-instruct-q4_K_M" # Or 'llama3.2', 'mistral', etc.
CHROMA_PATH = "./chroma_db" # Persist data so you don't reload every time

# --- Load Data ---
try:
    with open(JOURNAL_FILE, "r", encoding="utf-8") as file:
        text_file = file.read()
except FileNotFoundError:
    print(f"Error: {JOURNAL_FILE} not found. Please create it first.")
    exit()

# --- Chunking Strategies ---

def semantic_chunker(text):
    """
    Best for structured data like journals with clear separators.
    Splits by double newlines (\n\n).
    """
    chunks = text.split('\n\n')
    clean_chunks = [c.strip() for c in chunks if c.strip()]
    return clean_chunks

def universal_chunker(text, target_chunk_size=500, overlap_size=100):
    """
    Best for unstructured text (PDFs, essays) where structure is unknown.
    Splits by sentences while maintaining strict size limits.
    """
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?]) +', text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_len = len(sentence)
        if current_length + sentence_len > target_chunk_size:
            chunks.append(" ".join(current_chunk))
            overlap_buffer = []
            overlap_count = 0
            for s in reversed(current_chunk):
                if overlap_count < overlap_size:
                    overlap_buffer.insert(0, s)
                    overlap_count += len(s)
                else:
                    break
            current_chunk = overlap_buffer
            current_length = overlap_count
        current_chunk.append(sentence)
        current_length += sentence_len
        
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# --- Main Pipeline ---

# 1. Select Chunker
print("--- 1. Chunking Data ---")
# Using semantic chunker because we know the file structure (Journal)
my_chunks = semantic_chunker(text_file) 
print(f"Created {len(my_chunks)} chunks.")

# 2. Setup Database
print("--- 2. Building Vector Store ---")
ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    model_name=EMBEDDING_MODEL, 
    url="http://localhost:11434/api/embeddings"
)

client = chromadb.Client() # In-memory (Change to PersistentClient to save to disk)
collection = client.get_or_create_collection(name="journal_rag", embedding_function=ollama_ef)

# Add data to Chroma
ids = [f"id_{i}" for i in range(len(my_chunks))]
collection.add(documents=my_chunks, ids=ids)

# 3. Initialize Memory
convo_history = [
    {'role':'system', 'content':'You are a helpful assistant. Be brief.'}
]

print("\n✅ System Ready! Chat with your journal. (Type 'exit' to stop)")

# 4. Chat Loop
while True:
    user_input = input("\nYou: ")
    if user_input.lower() == 'exit':
        break

    # Retrieval
    # We fetch 5 chunks to ensure we catch relevant info even if it's spread out
    result = collection.query(query_texts=[user_input], n_results=5)
    
    # Fix: Join all retrieved chunks into one context string
    retrieved_chunks = result['documents'][0]
    context_data = "\n\n".join(retrieved_chunks)

    # Debug print (Optional: Remove if you want a clean UI)
    print(f"\n[DEBUG] Retrieved {len(retrieved_chunks)} chunks for context.")

    # Strict Prompting
    prompt_with_context = f"""
    You are a strict assistant. You must ONLY use the provided Context info to answer.
    If the answer is not in the Context info, you must say "I don't know".
    Do not use your own external knowledge.

    Context info: {context_data}
    User Question: {user_input}
    """

    # Add to temporary history for this API call
    messages_to_send = convo_history + [{'role': 'user', 'content': prompt_with_context}]    

    # Generation
    response = ollama.chat(
        model=GENERATION_MODEL,
        messages=messages_to_send
    )
    ai_answer = response['message']['content']
    print(f"AI: {ai_answer}")

    # Update History (Clean version)
    convo_history.append({'role': 'user', 'content': user_input})
    convo_history.append({'role': 'assistant', 'content': ai_answer})