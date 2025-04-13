import ollama
from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL
# generate chunks of text
def chunk_text(text):
    """Splits text into chunks of a specified size with overlap."""
    chunks = []
    for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk = text[i:i + CHUNK_SIZE]
        chunks.append(chunk)
    return chunks

# generate embeddings
def generate_embeddings(text, is_query: bool):
    """Generates embeddings for the given text using the Ollama model."""
    # Use the Ollama model to generate embeddings
    if is_query:
        response = ollama.embeddings(model=EMBEDDING_MODEL, prompt="search_query: " + text)
    else:
        response = ollama.embeddings(model=EMBEDDING_MODEL, prompt="search_document: " + text)
    return response['embedding']
