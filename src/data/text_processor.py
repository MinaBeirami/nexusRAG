import os
import torch
import numpy as np
import networkx as nx


from tqdm import tqdm
from py2neo import Graph, Node, Relationship
from typing import List, Dict, Any, Tuple, Optional, Union
from sentence_transformers import SentenceTransformer

class TextProcessor:
    """Process and chunk text data"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, title: str, source: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks"""
        if not text or len(text.strip()) == 0:
            return []
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            if len(chunk_words) < 50:  # Skip very small chunks
                continue
                
            chunk_text = " ".join(chunk_words)
            chunk_id = f"{source}_{i // (self.chunk_size - self.chunk_overlap)}"
            
            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "title": title,
                "source": source,
                "chunk_index": i // (self.chunk_size - self.chunk_overlap),
                "metadata": metadata
            })
        
        return chunks
    
    def process_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple documents into chunks"""
        all_chunks = []
        
        for doc in tqdm(documents, desc="Chunking documents"):
            chunks = self.chunk_text(
                doc["content"], 
                doc["title"], 
                doc["source"], 
                doc.get("metadata", {})
            )
            all_chunks.extend(chunks)
        
        return all_chunks
    


class EmbeddingGenerator:
    """Generate embeddings for text chunks"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
    
    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings for a list of text chunks"""
        texts = [chunk["text"] for chunk in chunks]
        
        # Generate embeddings in batches to avoid memory issues
        batch_size = 32
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch_texts)
            all_embeddings.extend(batch_embeddings)
        
        # Add embeddings to chunks
        for i, chunk in enumerate(chunks):
            chunk["embedding"] = all_embeddings[i]
        
        return chunks