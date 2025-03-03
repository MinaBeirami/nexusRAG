# src/data/embedding.py
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Singleton pattern for model caching
_embedding_models = {}


def get_embedding_model(model_name: str) -> SentenceTransformer:
    """Get or create a sentence transformer model"""
    if model_name not in _embedding_models:
        _embedding_models[model_name] = SentenceTransformer(model_name)
    return _embedding_models[model_name]


def generate_embeddings(
    chunks: List[Dict[str, Any]], model_name: str
) -> List[Dict[str, Any]]:
    """Generate embeddings for text chunks"""
    # Get the embedding model
    model = get_embedding_model(model_name)

    # Extract texts
    texts = [chunk["text"] for chunk in chunks]

    # Generate embeddings in batches to avoid memory issues
    batch_size = 32
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch_texts = texts[i : i + batch_size]
        batch_embeddings = model.encode(batch_texts)
        all_embeddings.extend(batch_embeddings)

    # Add embeddings to chunks
    chunks_with_embeddings = chunks.copy()
    for i, chunk in enumerate(chunks_with_embeddings):
        chunk["embedding"] = all_embeddings[i]

    return chunks_with_embeddings
