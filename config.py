import os

from dotenv import load_dotenv

load_dotenv()

# ---------------------------
# LLM
# ---------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "llama-3.1-8b-instant"
LLM_TEMPERATURE = 0

# ---------------------------
# EMBEDDINGS
# ---------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------
# RAG
# ---------------------------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 3

# ---------------------------
# STORAGE
# ---------------------------
DATA_FOLDER = "data"
CHAT_HISTORY_FILE = "chat_history.json"