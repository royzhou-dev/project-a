import os
from dotenv import load_dotenv

load_dotenv()

# Existing
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY', '')
PORT = int(os.getenv('PORT', 5000))

# New chatbot configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY', '')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT', 'us-west1-gcp')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'stock-assistant-rag')

CHAT_MODEL = os.getenv('CHAT_MODEL', 'gpt-4-turbo-preview')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
MAX_CONTEXT_LENGTH = int(os.getenv('MAX_CONTEXT_LENGTH', 8000))
RAG_TOP_K = int(os.getenv('RAG_TOP_K', 5))
