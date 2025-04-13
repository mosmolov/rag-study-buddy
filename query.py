from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchText, VectorParams, Distance
from dotenv import load_dotenv
from utils import generate_embeddings
import os
from typing import List, Dict
import asyncio 
from ollama import AsyncClient
from config import LLM_MODEL, RETRIEVAL_LIMIT, COLLECTION_NAME
load_dotenv()

client = QdrantClient(url=os.getenv("QDRANT_URL"),
                      api_key=os.getenv("QDRANT_API_KEY"))

def query_collection(query_text: str, collection_name: str=COLLECTION_NAME) -> List[Dict]:
    """
    Query the Qdrant collection for similar text chunks.

    Args:
        collection_name (str): Name of the collection in Qdrant
        query_text (str): Text to query
        limit (int): Number of results to return

    Returns:
        List[Dict]: List of matching documents with their metadata
    """
    # Generate embeddings for the query text
    embedding = generate_embeddings(query_text, is_query=True)

    # Query the collection
    results = client.search(
        collection_name=collection_name,
        query_vector=embedding,
        limit=RETRIEVAL_LIMIT,
    )

    # Extract and return the results
    return [
            {**result.payload, "score": result.score}  # Merging payload with score
            for result in results
        ] if results else []

# query llm using retrieved context
def query_llm_with_context(context: str, query_text: str) -> str:
    """
    Query the LLM with the retrieved context.

    Args:
        context (str): Retrieved context from Qdrant
        query_text (str): User's query

    Returns:
        str: LLM response
    """
    # Combine context and query for the LLM
    prompt = f"""
    Using only the provided retrieved documents, answer the following question. Do not add any external knowledge.
    Context: {context}
    Question: {query_text}

    """
    # query ollama llm
    async def ollama_stream_response(prompt: str) -> str:
        full_response = ""
        async for part in await AsyncClient().chat(
            messages=[{"role": "user", "content": prompt}],
            model=LLM_MODEL,
            stream=True
        ):
            if part.message and part.message.content:
                content = part.message.content
                full_response += content
        
        print()  # Add a newline at the end
        return full_response

    # Use asyncio to run the async function
    response = asyncio.run(ollama_stream_response(prompt))
   
    
    return response