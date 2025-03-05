# src/rag/engine.py
from typing import List, Dict, Any
import numpy as np

from src.data.data_collector import scrape_urls
from src.data.text_processor import chunk_text
from src.data.embedding import generate_embeddings
from src.database.graph_handler import GraphDatabase
from src.rag.retrieval import retrieve_context
from src.rag.llm import generate_answer


class MinimalRAG:
    """
    Minimal Retrieval Augmented Generation system using a graph database
    Combines data collection, processing, and querying
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "gpt-3.5-turbo",
        db_uri: str = "bolt://localhost:7687",
        db_user: str = "neo4j",
        db_password: str = "password",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        self.embedding_model_name = embedding_model
        self.llm_model = llm_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize database connection
        self.db = GraphDatabase(db_uri, db_user, db_password)

    def ingest_data(self, urls: List[str]) -> None:
        """Ingest data from URLs end-to-end"""
        # 1. Scrape URLs
        documents = scrape_urls(urls)
        print(f"Scraped {len(documents)} documents")

        # 2. Chunk documents
        chunks = chunk_text(documents, self.chunk_size, self.chunk_overlap)
        print(f"Created {len(chunks)} chunks")

        # 3. Generate embeddings
        chunks_with_embeddings = generate_embeddings(chunks, self.embedding_model_name)

        # 4. Add to graph database
        self.db.add_documents_and_chunks(chunks_with_embeddings)
        print("Added to graph database")
