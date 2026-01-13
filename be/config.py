import os
from dotenv import load_dotenv

load_dotenv()

# Existing
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY', '')
PORT = int(os.getenv('PORT', 5000))

# Chatbot configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')  # Still needed for embeddings

# Gemini configuration (for chat - FREE)
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')

# FAISS configuration
FAISS_INDEX_PATH = os.getenv('FAISS_INDEX_PATH', 'be/faiss_index')

# Embedding settings (still using OpenAI)
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
MAX_CONTEXT_LENGTH = int(os.getenv('MAX_CONTEXT_LENGTH', 8000))
RAG_TOP_K = int(os.getenv('RAG_TOP_K', 5))
