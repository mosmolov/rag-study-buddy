from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from dotenv import load_dotenv
from utils import semantically_chunk_text, generate_embeddings
from config import COLLECTION_NAME
import os
from PyPDF2 import PdfReader
import uuid
load_dotenv()
client = QdrantClient(url=os.getenv("QDRANT_URL"),
api_key=os.getenv("QDRANT_API_KEY"))

def process_pdf(pdf_reader: PdfReader, collection_name=COLLECTION_NAME, progress_callback=None):
    """
    Takes a PDF file, converts it to chunks, generates embeddings,
    and stores them in the Qdrant database.

    Args:
        pdf_reader (PdfReader): PDF reader object
        collection_name (str): Name of the collection in Qdrant
        progress_callback (callable, optional): Function to update progress
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
    
    # Show progress during text extraction
    total_pages = len(pdf_reader.pages)
    for i, page in enumerate(pdf_reader.pages):
        pdf_text += page.extract_text() + "\n\n"
        if progress_callback:
            progress_callback(0.5 * (i + 1) / total_pages, "Extracting text")
            
    # Check if text extraction was successful
    if not pdf_text.strip():
        raise ValueError("No text found in the PDF file.")

    # Split text into chunks
    chunks = semantically_chunk_text(pdf_text)
    points = []
    
    # Generate embeddings and store in Qdrant
    for i, chunk in enumerate(chunks):
        embedding = generate_embeddings(chunk, is_query=False)
        
        # Store in Qdrant
        points.append({
            "id": str(uuid.uuid4()),
            "vector": embedding,
            "payload": {
                "text": chunk,
                "chunk_index": i
            }
        })
        
        # Update progress for embedding generation (50% to 100%)
        if progress_callback:
            progress_callback(0.5 + 0.5 * (i + 1) / len(chunks), "Generating embeddings")
    
    if points:
        client.upsert(
            collection_name=collection_name,
            points=points
        )
    return len(points)

# clear database
def clear_database(collection_name="documents"):
    """Clears the Qdrant database."""
    try:
        client.delete_collection(collection_name)
        print(f"Collection '{collection_name}' deleted successfully.")
    except Exception as e:
        print(f"Error deleting collection: {e}")