import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database settings
DATABASE_URI = os.getenv("DATABASE_URI", "http://localhost:7687")
DATABASE_USER = os.getenv("DATABASE_USER", "username")
DATABASE_PASSWORD = os.getenv("DATABASE_PASSWORD", "password")

# Model settings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")

# Processing settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
