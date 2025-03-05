from typing import List, Dict, Any
import numpy as np
from py2neo import Graph, Node, Relationship
from tqdm import tqdm
import os


class GraphDatabase:
    def __init__(self, uri: str, user: str, password: str):
        self.uri = uri
        self.user = user
        self.password = password
        self.graph = None
        self.connect()

    def run_query(self, query, params=None):
        """Run a Cypher query against the graph database"""
        if self.graph:  # or whatever your database attribute is named
            return self.graph.run(query, params)
        return None

    def connect(self):
        """Connect to Neo4j database"""
        try:

            self.graph = Graph(self.uri, auth=(self.user, self.password))
            print("Connected to Neo4j database")

            # Create necessary constraints and indexes
            self.create_constraints()
        except Exception as e:
            print(f"Error connecting to database: {str(e)}")

    def create_constraints(self):
        """Create necessary constraints and indexes in the graph database"""
        if not self.graph:
            print("Database not connected")
            return

        try:
            # Create constraint on Chunk ID (unique)
            try:
                self.graph.run(
                    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE"
                )
            except:
                # For older Neo4j versions
                self.graph.run("CREATE CONSTRAINT ON (c:Chunk) ASSERT c.id IS UNIQUE")

            # Create constraint on Document source (unique)
            try:
                self.graph.run(
                    "CREATE CONSTRAINT document_source IF NOT EXISTS FOR (d:Document) REQUIRE d.source IS UNIQUE"
                )
            except:
                # For older Neo4j versions
                self.graph.run(
                    "CREATE CONSTRAINT ON (d:Document) ASSERT d.source IS UNIQUE"
                )

            # Create indexes for faster searching
            try:
                self.graph.run(
                    "CREATE INDEX chunk_text_index IF NOT EXISTS FOR (c:Chunk) ON (c.text)"
                )
                self.graph.run(
                    "CREATE INDEX document_title_index IF NOT EXISTS FOR (d:Document) ON (d.title)"
                )
            except:
                # For older Neo4j versions
                self.graph.run("CREATE INDEX ON :Chunk(text)")
                self.graph.run("CREATE INDEX ON :Document(title)")

            print("Created database constraints and indexes")

        except Exception as e:
            print(f"Error creating constraints: {str(e)}")

    def add_documents_and_chunks(self, chunks_with_embeddings: List[Dict[str, Any]]):
        """Add document chunks to the graph database"""
        if not self.graph:
            print("Database not connected")
            return

        # Save embeddings to file for faster retrieval
        self.store_embeddings_as_file(chunks_with_embeddings, "embeddings.npz")

        # Process in batches to avoid memory issues
        batch_size = 100

        # Create unique documents dictionary
        documents = {}
        for chunk in chunks_with_embeddings:
            source = chunk["source"]
            if source not in documents:
                documents[source] = {
                    "title": chunk["title"],
                    "source": source,
                    "metadata": chunk.get("metadata", {}),
                }

        # Add documents first
        for doc_source, doc_data in tqdm(documents.items(), desc="Adding documents"):
            doc_node = Node(
                "Document", source=doc_data["source"], title=doc_data["title"]
            )

            # Add metadata properties
            for key, value in doc_data["metadata"].items():
                if isinstance(value, (str, int, float, bool)) and not isinstance(
                    value, (list, dict)
                ):
                    doc_node[key] = value

            # Merge to avoid duplicates
            self.graph.merge(doc_node, "Document", "source")

        # Add chunks in batches
        for i in tqdm(
            range(0, len(chunks_with_embeddings), batch_size),
            desc="Adding chunks to graph",
        ):
            batch = chunks_with_embeddings[i : i + batch_size]

            # Add each chunk and connect to its document
            for chunk in batch:
                chunk_node = Node(
                    "Chunk",
                    id=chunk["id"],
                    text=chunk["text"],
                    chunk_index=chunk["chunk_index"],
                )

                # Merge to avoid duplicates
                self.graph.merge(chunk_node, "Chunk", "id")

                # Get the document node
                cypher_query = "MATCH (d:Document {source: $source}) RETURN d"
                doc_node = self.graph.run(
                    cypher_query, source=chunk["source"]
                ).evaluate()

                if doc_node:
                    # Create relationship between chunk and document
                    part_of = Relationship(chunk_node, "PART_OF", doc_node)
                    self.graph.merge(part_of)

                    # Connect to previous chunk for sequence
                    if chunk["chunk_index"] > 0:
                        prev_chunk_id = f"{chunk['source']}_{chunk['chunk_index'] - 1}"
                        cypher_query = "MATCH (c:Chunk {id: $prev_id}) RETURN c"
                        prev_chunk = self.graph.run(
                            cypher_query, prev_id=prev_chunk_id
                        ).evaluate()

                        if prev_chunk:
                            follows = Relationship(chunk_node, "FOLLOWS", prev_chunk)
                            self.graph.merge(follows)

    def store_embeddings_as_file(self, chunks: List[Dict[str, Any]], file_path: str):
        """Store embeddings separately as a numpy file for faster retrieval"""
        # Make sure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

        embeddings = np.array([chunk["embedding"] for chunk in chunks])
        ids = np.array([chunk["id"] for chunk in chunks])

        np.savez(file_path, embeddings=embeddings, ids=ids)
        print(f"Saved embeddings to {file_path}")

    def find_similar_chunks(self, query_embedding: np.ndarray, top_k: int = 5):
        """Find chunks similar to the query using the stored embeddings"""
        try:
            # Load embeddings from file
            embeddings_file = np.load("embeddings.npz")
            embeddings = embeddings_file["embeddings"]
            ids = embeddings_file["ids"]

            # Calculate similarity scores
            scores = np.dot(embeddings, query_embedding)

            # Get top k results
            top_indices = np.argsort(scores)[-top_k:][::-1]
            top_scores = scores[top_indices]
            top_ids = ids[top_indices]

            # Retrieve the actual chunks from the database
            results = []
            for i, chunk_id in enumerate(top_ids):
                cypher_query = """
                MATCH (c:Chunk {id: $id})
                RETURN c.id as id, c.text as text, c.chunk_index as chunk_index
                """
                result = self.graph.run(cypher_query, id=chunk_id).data()

                if result:
                    result[0]["score"] = float(top_scores[i])
                    results.append(result[0])

            return results
        except Exception as e:
            print(f"Error finding similar chunks: {str(e)}")
            return []
