from typing import List
import ollama
from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging
import nltk
import re

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

def semantically_chunk_text(text: str)-> List[str]: 
    """Splits text into semantically meaningful chunks.

    Args:
        text (str): Text to be chunked.
    Returns:
        List[str]: List of semantically meaningful text chunks.
    """
    # split text into sentences based on punctuation
    logging.info("Splitting text into sentences...")
    try:
        text = re.sub(r'\s+', ' ', text).strip()
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab')
        sentences = nltk.sent_tokenize(text, language="english")
        # Log the number of sentences found
        logging.info(f"Number of sentences found: {len(sentences)}")
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        print(f"Number of sentences after filtering: {len(sentences)}")
        embeddings = []
        # generate embeddings for each sentence
        for sentence in sentences:
            embeddings.append(generate_embeddings(sentence, is_query=False))
        # create similarity matrix
        sim_matrix = create_similarity_matrix(np.array(embeddings))
        # create chunks based on similarity
        chunks = []
        used_indices = set()
        # sort sentences based on length
        sentence_indices = sorted(range(len(sentences)), key=lambda i: len(sentences[i]), reverse=True)
        for i in sentence_indices:
            if i in used_indices:
                continue
            curr_chunk = [sentences[i]]
            curr_size = len(sentences[i])
            print(f"Current chunk size: {curr_size}/{CHUNK_SIZE} characters")
            similarity_scores = sorted([(j, sim_matrix[i, j]) 
                                for j in range(len(sentences)) 
                                if j != i and j not in used_indices], key=lambda x: x[1], reverse=True)
            for j, score in similarity_scores:
                if score >= 0.5 and curr_size + len(sentences[j]) <= CHUNK_SIZE:
                    curr_chunk.append(sentences[j])
                    curr_size += len(sentences[j])
                    used_indices.add(j)
                    # Log when we add a sentence to the chunk
                    logging.debug(f"Added sentence {j} to chunk with similarity score {score:.4f}")
                    # Log the current chunk size
                    logging.debug(f"Current chunk size: {curr_size}/{CHUNK_SIZE} characters")
            chunks.append("".join(curr_chunk))
        return chunks
    except Exception as e:
        logging.error(f"Error during semantic chunking: {e}")
        return []

        
def create_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Creates a similarity matrix from the given embeddings.

    Args:
        embeddings (List[List[float]]): List of embeddings.

    Returns:
        np.ndarray: Similarity matrix.
    """
    logging.debug("Creating similarity matrix...")
    return cosine_similarity(embeddings)