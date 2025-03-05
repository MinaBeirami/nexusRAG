# NexusRAG: End-to-End Retrieval Augmented Generation with Graph Database

A comprehensive RAG system that uses a graph database to store and retrieve knowledge in a structured, interconnected way. The system features web scraping capabilities for data collection, document processing with semantic chunking, and a Streamlit interface for easy interaction.

## Features

- **Multiple Data Source Options**:
  - Web scraping functionality to collect data from any website
  - Scraping data collection module (Wikipedia format Suported)

- **Advanced Text Processing**:
  - Semantic chunking with configurable size and overlap
  - High-quality embeddings using `Sentence Transformers`
  - Context expansion for improved relevance

- **Graph Database Integration**:
  - Neo4j backend for knowledge storage
  - Semantic relationships between text chunks
  - Document and chunk hierarchies with metadata

- **Streamlined User Interface**:
  - Interactive Streamlit application
  - Ability to ask question or provide URLs for ingesting data
  - Visualization of the knowledge graph structure (TBD)

- **Customizable LLM Integration**:
  - Configurable to work with any OpenAI model
  - Extensible design for other LLM providers

## Installation

### Prerequisites

- Python 3.10
- Neo4j Database (local, docker or cloud instance)
- OpenAI API key (or equivalent)

### Setup

1. Clone this repository:
   ```bash
   git clone git@github.com:MinaBeirami/nexusRAG.git
   cd nexusRAG
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:

   Edit the `.env` file with your API keys and database credentials.

4. Start the Neo4j database (if using a local instance):
    Run it via Docker Desktop,
    OR run the following Command:
   ```bash
   # If using Docker
   docker run --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password -d neo4j
   ```

## Usage

### Starting the Application

```bash
streamlit run app.py
```

This will start the Streamlit server, and you can access the application at `http://localhost:8501`.

### Data Collection

The system provides several ways to collect data:

1. **Web Scraping**: Enter URLs to scrape content from websites
2. **Paste Text**: Directly paste text content into the application
3. **(TODO) Upload Files**: Upload local documents (PDF, DOCX, TXT, CSV)
4. **(TODO) Hugging Face Datasets**: Select and import datasets from Hugging Face

### Building the Knowledge Graph

After collecting data, the system will:

1. Process documents into semantic chunks
2. Generate embeddings for each chunk
3. Store chunks and their relationships in the Neo4j graph database
4. Create semantic relationships between related chunks

### Querying the System

Once the knowledge graph is built, you can:

1. Ask questions in natural language
2. View the retrieved context used to answer the question
3. (TODO)Explore the knowledge graph visually
4. Export answers and sources

## Configuration

The system can be customized through the `src/config/settings.py` file:

- `embedding_model`: Change the embedding model (default: "all-MiniLM-L6-v2")
- `chunk_size`: Adjust the size of text chunks (default: 500)
- `chunk_overlap`: Set the overlap between chunks (default: 50)
- `llm_model`: Select the LLM model (default: "gpt-3.5-turbo")

## Architecture

The system follows a modular architecture:

- `data_collector.py`: Modules for acquiring data from various sources
- `text_processor.py`: Text processing, chunking, and embedding generation
- `rag_engine.py`: Core RAG implementation with LLM integration
- `graph_handler.py`: Neo4j database interaction
- `app.py`: Streamlit user interface


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.