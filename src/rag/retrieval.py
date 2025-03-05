# src/rag/retrieval.py
import numpy as np
from typing import List, Dict, Any

def retrieve_context(db, query_embedding: np.ndarray, top_k: int = 3, expand: bool = True) -> str:
    """
    Retrieve context for a query, optionally expanding to neighboring chunks
    """
    # Find most similar chunks
    similar_chunks = db.find_similar_chunks(query_embedding, top_k=top_k)
    
    # Store unique chunk IDs to avoid duplicates
    retrieved_chunk_ids = set()
    contexts = []
    
    # Process each similar chunk
    for chunk in similar_chunks:
        chunk_id = chunk["id"]
        
        # Add the chunk itself
        if chunk_id not in retrieved_chunk_ids:
            contexts.append(f"[Chunk {chunk['chunk_index']}] {chunk['text']}")
            retrieved_chunk_ids.add(chunk_id)
        
        # Optionally expand to include neighboring chunks
        if expand:
            # Get source and chunk index
            source, index_str = chunk_id.rsplit("_", 1)
            index = int(index_str)
            
            # Get previous chunk if available
            prev_id = f"{source}_{index - 1}"
            cypher_query = """
            MATCH (c:Chunk {id: $id})
            RETURN c.id as id, c.text as text, c.chunk_index as chunk_index
            """
            prev_chunk = db.graph.run(cypher_query, id=prev_id).data()
            
            if prev_chunk and prev_id not in retrieved_chunk_ids:
                prev_data = prev_chunk[0]
                contexts.append(f"[Chunk {prev_data['chunk_index']}] {prev_data['text']}")
                retrieved_chunk_ids.add(prev_id)
            
            # Get next chunk if available
            next_id = f"{source}_{index + 1}"
            next_chunk = db.graph.run(cypher_query, id=next_id).data()
            
            if next_chunk and next_id not in retrieved_chunk_ids:
                next_data = next_chunk[0]
                contexts.append(f"[Chunk {next_data['chunk_index']}] {next_data['text']}")
                retrieved_chunk_ids.add(next_id)
    
    # Combine context chunks
    combined_context = "\n\n".join(contexts)
    return combined_context