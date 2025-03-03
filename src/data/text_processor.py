# src/data/text_processing.py
from typing import List, Dict, Any
from tqdm import tqdm


def chunk_text(
    documents: List[Dict[str, Any]], chunk_size: int, chunk_overlap: int
) -> List[Dict[str, Any]]:
    """Process documents into chunks with overlap"""
    all_chunks = []

    for doc in tqdm(documents, desc="Chunking documents"):
        # Split text into chunks
        text = doc["content"]
        title = doc["title"]
        source = doc["source"]
        metadata = doc.get("metadata", {})

        if not text or len(text.strip()) == 0:
            continue

        words = text.split()

        for i in range(0, len(words), chunk_size - chunk_overlap):
            chunk_words = words[i : i + chunk_size]
            if len(chunk_words) < 50:  # Skip very small chunks
                continue

            chunk_text = " ".join(chunk_words)
            chunk_id = f"{source}_{i // (chunk_size - chunk_overlap)}"

            all_chunks.append(
                {
                    "id": chunk_id,
                    "text": chunk_text,
                    "title": title,
                    "source": source,
                    "chunk_index": i // (chunk_size - chunk_overlap),
                    "metadata": metadata,
                }
            )

    return all_chunks
