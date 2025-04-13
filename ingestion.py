from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from dotenv import load_dotenv
import ollama
import os
from PyPDF2 import PdfReader
import uuid
load_dotenv()
client = QdrantClient(url=os.getenv("QDRANT_URL"),
api_key=os.getenv("QDRANT_API_KEY"))

def process_pdf(pdf_reader: PdfReader, collection_name="documents", chunk_size=1000):
    """
    Takes a PDF file, converts it to chunks, generates embeddings,
    and stores them in the Qdrant database.

    Args:
        pdf_path (str): Path to the PDF file
        collection_name (str): Name of the collection in Qdrant
        chunk_size (int): Size of text chunks
    """
    # Create collection if it doesn't exist
    try:
        client.get_collection(collection_name)
    except Exception:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )

    # Extract text from PDF
    pdf_text = ""
    for page in pdf_reader.pages:
        pdf_text += page.extract_text() + "\n\n"
    # Check if text extraction was successful
    if not pdf_text.strip():
        raise ValueError("No text found in the PDF file.")
    

    # Split text into chunks
    chunks = chunk_text(pdf_text, chunk_size)

    # Generate embeddings and store in Qdrant
    for i, chunk in enumerate(chunks):
        embedding = generate_embeddings(chunk)
        
        # Store in Qdrant
        client.upsert(
            collection_name=collection_name,
            points=[
                {
                    "id": str(uuid.uuid4()),
                    "vector": embedding,
                    "payload": {
                        "text": chunk,
                        "chunk_index": i
                    }
                }
            ]
        )

    return len(chunks)

# generate chunks of text
def chunk_text(text, chunk_size=1000):
    """Splits text into chunks of a specified size."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# generate embeddings
def generate_embeddings(text):
    """Generates embeddings for the given text using the Ollama model."""
    # Use the Ollama model to generate embeddings
    response = ollama.embeddings(model='nomic-embed-text', prompt=text)
    return response['embedding']

# clear database
def clear_database(collection_name="documents"):
    """Clears the Qdrant database."""
    try:
        client.delete_collection(collection_name)
        print(f"Collection '{collection_name}' deleted successfully.")
    except Exception as e:
        print(f"Error deleting collection: {e}")