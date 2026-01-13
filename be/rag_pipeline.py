from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import tiktoken
from config import (
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_ENVIRONMENT,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
    RAG_TOP_K
)


class EmbeddingGenerator:
    """Generates embeddings using OpenAI's embedding API"""

    def __init__(self, api_key=None, model=EMBEDDING_MODEL):
        self.client = OpenAI(api_key=api_key or OPENAI_API_KEY)
        self.model = model
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def generate_embedding(self, text):
        """
        Generate embedding vector for text

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector
        """
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

    def chunk_long_text(self, text, max_tokens=8000):
        """
        Split long text into chunks that fit within token limits

        Args:
            text: Text to chunk
            max_tokens: Maximum tokens per chunk

        Returns:
            List of text chunks
        """
        tokens = self.encoding.encode(text)

        if len(tokens) <= max_tokens:
            return [text]

        chunks = []
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)

        return chunks

    def count_tokens(self, text):
        """Count tokens in text"""
        return len(self.encoding.encode(text))


class VectorStore:
    """Manages Pinecone vector database for RAG"""

    def __init__(self, api_key=None, index_name=None):
        self.pc = Pinecone(api_key=api_key or PINECONE_API_KEY)
        self.index_name = index_name or PINECONE_INDEX_NAME
        self.index = None
        self._initialize_index()

    def _initialize_index(self):
        """Initialize or connect to Pinecone index"""
        try:
            # Check if index exists
            existing_indexes = self.pc.list_indexes()

            if self.index_name not in [idx.name for idx in existing_indexes]:
                print(f"Creating new Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=1536,  # text-embedding-3-small dimension
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )

            self.index = self.pc.Index(self.index_name)
            print(f"Connected to Pinecone index: {self.index_name}")

        except Exception as e:
            print(f"Error initializing Pinecone index: {e}")
            raise

    def upsert_document(self, doc_id, embedding, metadata, namespace="news"):
        """
        Store document embedding with metadata in Pinecone

        Args:
            doc_id: Unique document identifier
            embedding: Embedding vector
            metadata: Dictionary of metadata
            namespace: Pinecone namespace for organization

        Returns:
            True if successful, False otherwise
        """
        try:
            self.index.upsert(
                vectors=[{
                    "id": doc_id,
                    "values": embedding,
                    "metadata": metadata
                }],
                namespace=namespace
            )
            return True
        except Exception as e:
            print(f"Error upserting document {doc_id}: {e}")
            return False

    def search(self, query_embedding, ticker=None, doc_type=None, top_k=None, namespace="news"):
        """
        Search for similar documents

        Args:
            query_embedding: Query embedding vector
            ticker: Filter by ticker symbol
            doc_type: Filter by document type
            top_k: Number of results to return
            namespace: Pinecone namespace to search

        Returns:
            List of matching documents with metadata
        """
        try:
            filter_dict = {}
            if ticker:
                filter_dict["ticker"] = ticker
            if doc_type:
                filter_dict["type"] = doc_type

            results = self.index.query(
                vector=query_embedding,
                filter=filter_dict if filter_dict else None,
                top_k=top_k or RAG_TOP_K,
                include_metadata=True,
                namespace=namespace
            )

            return results.matches
        except Exception as e:
            print(f"Error searching Pinecone: {e}")
            return []

    def document_exists(self, doc_id, namespace="news"):
        """
        Check if document already exists in index

        Args:
            doc_id: Document identifier
            namespace: Pinecone namespace

        Returns:
            Boolean indicating if document exists
        """
        try:
            result = self.index.fetch(ids=[doc_id], namespace=namespace)
            return doc_id in result.vectors
        except Exception as e:
            print(f"Error checking document existence: {e}")
            return False

    def delete_by_ticker(self, ticker, namespace="news"):
        """
        Delete all documents for a ticker

        Args:
            ticker: Ticker symbol
            namespace: Pinecone namespace
        """
        try:
            self.index.delete(
                filter={"ticker": ticker},
                namespace=namespace
            )
            print(f"Deleted all documents for {ticker}")
        except Exception as e:
            print(f"Error deleting documents for {ticker}: {e}")


class ContextRetriever:
    """High-level interface for retrieving relevant context"""

    def __init__(self):
        self.embedding_gen = EmbeddingGenerator()
        self.vector_store = VectorStore()

    def retrieve_context(self, query, ticker, doc_type=None, top_k=None):
        """
        Retrieve relevant documents for a query

        Args:
            query: User query text
            ticker: Ticker symbol to filter by
            doc_type: Optional document type filter
            top_k: Number of results to return

        Returns:
            List of relevant documents with metadata
        """
        # Generate query embedding
        query_embedding = self.embedding_gen.generate_embedding(query)

        if not query_embedding:
            return []

        # Search vector store
        matches = self.vector_store.search(
            query_embedding=query_embedding,
            ticker=ticker,
            doc_type=doc_type,
            top_k=top_k
        )

        # Format results
        contexts = []
        for match in matches:
            contexts.append({
                'score': match.score,
                'metadata': match.metadata,
                'id': match.id
            })

        return contexts
