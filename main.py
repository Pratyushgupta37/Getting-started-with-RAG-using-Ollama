import ollama
import chromadb
from chromadb.utils import embedding_functions
import re

# --- Configuration ---
JOURNAL_FILE = "journal.txt"
EMBEDDING_MODEL = "nomic-embed-text"
GENERATION_MODEL = "llama3.1:8b-instruct-q4_K_M"

# --- Load Data ---
try:
    with open(JOURNAL_FILE, "r", encoding="utf-8") as file:
        text_file = file.read()
except FileNotFoundError:
    print(f"Error: {JOURNAL_FILE} not found. Please create it first.")
    exit()

# --- Semantic Chunking ---
def semantic_chunker(text):
    """Splits by double newlines (\n\n) for structured data."""
    chunks = text.split('\n\n')
    clean_chunks = [c.strip() for c in chunks if c.strip()]
    return clean_chunks

# --- Main Pipeline ---

# 1. Chunking
print("--- 1. Chunking Data ---")
my_chunks = semantic_chunker(text_file) 
print(f"Created {len(my_chunks)} chunks.")

# 2. Metadata Extraction (The Smart Part 🧠)
# We scan each chunk to see which month it belongs to.
MONTHS = ["January", "February", "March", "April", "May", "June", 
          "July", "August", "September", "October", "November", "December"]

metadatas = []
for chunk in my_chunks:
    found_month = "Unknown"
    for m in MONTHS:
        if m in chunk: # e.g. If "January" is in the text
            found_month = m
            break
    metadatas.append({"month": found_month})

# 3. Setup Database
print("--- 2. Building Vector Store ---")
ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    model_name=EMBEDDING_MODEL, 
    url="http://localhost:11434/api/embeddings"
)

client = chromadb.Client()
# Important: reset the collection to ensure clean metadata
try:
    client.delete_collection(name="journal_rag")
except:
    pass 

collection = client.create_collection(name="journal_rag", embedding_function=ollama_ef)

ids = [f"id_{i}" for i in range(len(my_chunks))]

# Add data WITH metadata
collection.add(
    documents=my_chunks, 
    ids=ids,
    metadatas=metadatas # <--- Storing the tags
)

# 4. Chat Loop
convo_history = [
    {'role':'system', 'content':'You are a helpful assistant. Be brief.'}
]

print("\n✅ System Ready! Chat with your journal. (Type 'exit' to stop)")

while True:
    user_input = input("\nYou: ")
    if user_input.lower() == 'exit':
        break

    # 5. Dynamic Filtering Logic
    # If user mentions a month, we tell Chroma to look ONLY at that month.
    filter_dict = None
    for m in MONTHS:
        if m.lower() in user_input.lower():
            filter_dict = {"month": m}
            print(f"[DEBUG] Applied Filter: {filter_dict} (Searching ONLY {m})")
            break
    
    # 6. Retrieval
    # If filter_dict is None, it searches everything.
    # If filter_dict is {"month": "January"}, it searches only January.
    result = collection.query(
        query_texts=[user_input], 
        n_results=10, # Get enough chunks to cover the whole month
        where=filter_dict 
    )
    
    retrieved_chunks = result['documents'][0]
    
    if not retrieved_chunks:
        print("AI: I couldn't find any entries for that specific query.")
        continue

    context_data = "\n\n".join(retrieved_chunks)

    # 7. Generation
    prompt_with_context = f"""
    You are a strict assistant. You must ONLY use the provided Context info to answer.
    If the answer is not in the Context info, you must say "I don't know".
    Do not use your own external knowledge.

    Context info: {context_data}
    User Question: {user_input}
    """

    messages_to_send = convo_history + [{'role': 'user', 'content': prompt_with_context}]    

    response = ollama.chat(
        model=GENERATION_MODEL,
        messages=messages_to_send
    )
    ai_answer = response['message']['content']
    print(f"AI: {ai_answer}")

    convo_history.append({'role': 'user', 'content': user_input})
    convo_history.append({'role': 'assistant', 'content': ai_answer})