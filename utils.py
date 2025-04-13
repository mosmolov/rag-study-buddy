import ollama
# generate chunks of text
def chunk_text(text, chunk_size=1024):
    """Splits text into chunks of a specified size."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# generate embeddings
def generate_embeddings(text, is_query: bool):
    """Generates embeddings for the given text using the Ollama model."""
    # Use the Ollama model to generate embeddings
    if is_query:
        response = ollama.embeddings(model='nomic-embed-text', prompt="search_query: " + text)
    else:
        response = ollama.embeddings(model='nomic-embed-text', prompt="search_document: " + text)
    return response['embedding']
