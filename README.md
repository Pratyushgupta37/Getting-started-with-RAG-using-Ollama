# Local RAG Personal Journal 🧠📖

A privacy-focused, AI-powered chat interface for your personal journal. 

This project implements a complete **Retrieval-Augmented Generation (RAG)** pipeline from scratch using Python. It allows users to chat with their own text data (e.g., a diary or log) without sending data to the cloud.

## 🚀 Features

*   **100% Local:** Runs entirely on your machine using [Ollama](https://ollama.com). No API keys required.
*   **Vector Search:** Uses **ChromaDB** to index and retrieve relevant journal entries based on meaning, not just keywords.
*   **Context-Aware Memory:** Maintains chat history so you can ask follow-up questions (e.g., "Where did I go?" -> "Was it fun?").
*   **Hallucination Prevention:** Uses strict prompting and grounded context to ensure the AI says "I don't know" rather than making up facts.
*   **Semantic Chunking:** Implements logic to split text by journal entries for higher retrieval accuracy.

## 🛠️ Tech Stack

*   **Python 3.x**
*   **Ollama:** For running the LLM (Llama 3.1) and Embedding Model (Nomic).
*   **ChromaDB:** For vector storage and similarity search.

## 📋 Prerequisites

1.  **Install Python:** Ensure you have Python installed.
2.  **Install Ollama:** Download from [ollama.com](https://ollama.com).
3.  **Pull Models:** Open your terminal and run:
    ```bash
    ollama pull llama3.1:8b-instruct-q4_K_M
    ollama pull nomic-embed-text
    ```

## ⚙️ Installation

1.  Clone this repository:
    ```bash
    git clone https://github.com/Pratyushgupta37/Getting-started-with-RAG-using-Ollama
    cd Getting-started-with-RAG-using-Ollama
    ```

2.  Install required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3.  Create your data file:
    *   Create a file named `journal.txt` in the project folder.
    *   Add your entries separated by double newlines (see `example_journal.txt` for format).

## 🏃‍♂️ Usage

Run the script:

```bash
python main.py