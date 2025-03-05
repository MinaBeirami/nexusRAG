import streamlit as st
import sys
import os
import time

# Add the parent directory to sys.path to import the src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import from src modules
from src.rag.engine import MinimalRAG
from src.config.settings import DATABASE_URI, DATABASE_USER, DATABASE_PASSWORD

# Set page configuration
st.set_page_config(page_title="Minimal RAG System", page_icon="🧠", layout="wide")


# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if "rag" not in st.session_state:
        st.session_state.rag = None

    if "db_connected" not in st.session_state:
        st.session_state.db_connected = False

    if "active_page" not in st.session_state:
        st.session_state.active_page = "Data Collection"


# Main title
st.title("🧠 NexusRAG: RAG App with Graph Database")

# Initialize session state
init_session_state()

# Sidebar for database connection
with st.sidebar:
    st.title("Database Connection")

    with st.form("db_connection_form"):
        db_uri = st.text_input("Neo4j URI", value=DATABASE_URI)
        db_user = st.text_input("Neo4j Username", value=DATABASE_USER)
        db_password = st.text_input(
            "Neo4j Password", value=DATABASE_PASSWORD, type="password"
        )
        connect_button = st.form_submit_button("Connect to Database")

        if connect_button:
            with st.spinner("Connecting to database..."):
                try:
                    st.session_state.rag = MinimalRAG(
                        db_uri=db_uri, db_user=db_user, db_password=db_password
                    )
                    st.session_state.db_connected = True
                    st.sidebar.success("Connected to database!")
                    print(st.session_state.rag.db)
                except Exception as e:
                    st.sidebar.error(f"Error connecting to database: {str(e)}")

    # Navigation
    st.sidebar.title("Navigation")
    pages = ["Data Collection", "Query System", "Knowledge Graph"]
    page_selection = st.sidebar.radio(
        "Go to", pages, index=pages.index(st.session_state.active_page)
    )

    if page_selection != st.session_state.active_page:
        st.session_state.active_page = page_selection
        st.rerun()

# Main content based on active page
if st.session_state.active_page == "Data Collection":
    st.header("Data Collection")

    if not st.session_state.db_connected:
        st.warning("Please connect to the database using the sidebar first.")
    else:
        with st.form("ingest_form"):
            urls_input = st.text_area("Enter URLs (one per line)", height=150)
            ingest_button = st.form_submit_button("Ingest Data")

            if ingest_button:
                if not urls_input.strip():
                    st.error("Please enter at least one URL!")
                else:
                    urls = [
                        url.strip() for url in urls_input.split("\n") if url.strip()
                    ]
                    st.info(
                        f"Starting ingestion of {len(urls)} URLs. This may take a while..."
                    )

                    # Create a progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    try:
                        # We'll set up a progress tracker
                        status_text.text("Scraping URLs...")
                        progress_bar.progress(25)

                        # Start the actual ingestion (this is a blocking operation)
                        st.session_state.rag.ingest_data(urls)

                        # Update progress
                        progress_bar.progress(100)
                        status_text.text("Ingestion complete!")
                        st.success("Data ingestion complete!")
                    except Exception as e:
                        st.error(f"Error during ingestion: {str(e)}")

elif st.session_state.active_page == "Query System":
    st.header("Query System")

    if not st.session_state.db_connected:
        st.warning("Please connect to the database using the sidebar first.")
    else:
        with st.form("query_form"):
            query = st.text_input("Enter your question")
            top_k = st.slider("Number of chunks to retrieve", 1, 10, 3)
            query_button = st.form_submit_button("Submit Query")

            if query_button:
                if not query.strip():
                    st.error("Please enter a query!")
                else:
                    with st.spinner("Processing query..."):
                        try:
                            result = st.session_state.rag.process_query(
                                query, top_k=top_k
                            )

                            st.subheader("Answer")
                            st.write(result["answer"])

                            with st.expander("Show retrieved context"):
                                st.text(result["context"])
                        except Exception as e:
                            st.error(f"Error processing query: {str(e)}")

else:  # Knowledge Graph
    st.header("Knowledge Graph Visualization")

    if not st.session_state.db_connected:
        st.warning("Please connect to the database using the sidebar first.")
    else:
        st.info("Knowledge graph visualization will be implemented in a future update.")

        # Placeholder for the graph visualization
        st.write("A visualization of your knowledge graph would appear here.")

        # Add some basic graph statistics
        if st.button("Get Graph Statistics"):
            try:
                # Query to count nodes and relationships
                doc_count = st.session_state.rag.db.run_query(
                    "MATCH (d:Document) RETURN count(d) as count"
                ).data()[0]["count"]
                chunk_count = st.session_state.rag.db.run_query(
                    "MATCH (c:Chunk) RETURN count(c) as count"
                ).data()[0]["count"]
                rel_count = st.session_state.rag.db.run_query(
                    "MATCH ()-[r]->() RETURN count(r) as count"
                ).data()[0]["count"]

                # Display the statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Documents", doc_count)
                with col2:
                    st.metric("Chunks", chunk_count)
                with col3:
                    st.metric("Relationships", rel_count)
            except Exception as e:
                st.error(f"Error retrieving graph statistics: {str(e)}")
