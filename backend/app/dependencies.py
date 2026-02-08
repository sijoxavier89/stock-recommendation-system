"""
FastAPI Dependencies
Manages service initialization and dependency injection.
"""
from functools import lru_cache
from typing import Generator
import os

from app.services.vector_store import VectorStore
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService
from app.services.pdf_processor import PDFProcessor
from app.services.ingestion_pipeline import IngestionPipeline


# Configuration from environment variables
from .core.config import CHROMA_DB_DIR

# Default to backend-local chroma db dir unless overridden
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", str(CHROMA_DB_DIR))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


# Singleton service instances
@lru_cache()
def get_vector_store() -> VectorStore:
    """
    Get or create VectorStore instance.
    Cached as singleton to maintain connection pool.
    """
    return VectorStore(
        persist_directory=CHROMA_DB_PATH,
        collection_name="annual_reports"
    )


@lru_cache()
def get_embedding_service() -> EmbeddingService:
    """
    Get or create EmbeddingService instance.
    Cached to avoid reloading the model.
    """
    return EmbeddingService(
        model_name=EMBEDDING_MODEL,
        device="cpu"  # Use "cuda" if GPU available
    )


@lru_cache()
def get_llm_service() -> LLMService:
    """
    Get or create LLMService instance.
    Cached to maintain client connection.
    """
    api_key = None
    if LLM_PROVIDER == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
    elif LLM_PROVIDER == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
    
    return LLMService(
        provider=LLM_PROVIDER,
        model=LLM_MODEL,
        api_key=api_key,
        base_url=OLLAMA_BASE_URL if LLM_PROVIDER == "ollama" else None
    )


def get_rag_service(
    vector_store: VectorStore = None,
    embedding_service: EmbeddingService = None,
    llm_service: LLMService = None
) -> RAGService:
    """
    Get RAGService instance.
    Creates new instance each time to allow dependency injection in tests.
    
    Args:
        vector_store: Optional override for testing
        embedding_service: Optional override for testing
        llm_service: Optional override for testing
    """
    vs = vector_store or get_vector_store()
    es = embedding_service or get_embedding_service()
    ls = llm_service or get_llm_service()
    
    return RAGService(
        vector_store=vs,
        embedding_service=es,
        llm_service=ls,
        top_k=5
    )


def get_pdf_processor() -> PDFProcessor:
    """
    Get PDFProcessor instance.
    Creates new instance each time (stateless).
    """
    return PDFProcessor()


def get_ingestion_pipeline(
    vector_store: VectorStore = None,
    embedding_service: EmbeddingService = None,
    pdf_processor: PDFProcessor = None
) -> IngestionPipeline:
    """
    Get IngestionPipeline instance.
    
    Args:
        vector_store: Optional override for testing
        embedding_service: Optional override for testing
        pdf_processor: Optional override for testing
    """
    vs = vector_store or get_vector_store()
    es = embedding_service or get_embedding_service()
    pp = pdf_processor or get_pdf_processor()
    
    return IngestionPipeline(
        vector_store=vs,
        embedding_service=es,
        pdf_processor=pp
    )


# Startup event handlers
async def startup_event():
    """
    Initialize services on startup.
    Pre-load models and establish connections.
    """
    print("🚀 Starting up application...")
    
    # Initialize services (loads models into memory)
    print("📊 Loading vector store...")
    vector_store = get_vector_store()
    print(f"   ✓ Vector store ready: {vector_store.collection.count()} chunks indexed")
    
    print("🧠 Loading embedding model...")
    embedding_service = get_embedding_service()
    print(f"   ✓ Embedding model ready: {embedding_service.model_name}")
    
    print("💬 Connecting to LLM...")
    llm_service = get_llm_service()
    print(f"   ✓ LLM ready: {llm_service.provider.value}/{llm_service.model}")
    
    print("✅ Application ready!")


async def shutdown_event():
    """
    Cleanup on shutdown.
    """
    print("👋 Shutting down application...")
    # Add any cleanup logic here if needed
    print("✅ Shutdown complete")


# For testing: dependency override helpers
class DependencyOverrides:
    """
    Helper class to override dependencies in tests.
    
    Usage:
        from app.dependencies import DependencyOverrides
        
        with DependencyOverrides.override_vector_store(mock_vector_store):
            # Test code here
            pass
    """
    
    @staticmethod
    def override_vector_store(vector_store: VectorStore):
        """Override vector store dependency."""
        get_vector_store.cache_clear()
        original = get_vector_store
        
        @lru_cache()
        def _override():
            return vector_store
        
        globals()['get_vector_store'] = _override
        
        try:
            yield
        finally:
            globals()['get_vector_store'] = original
            get_vector_store.cache_clear()
    
    @staticmethod
    def override_llm_service(llm_service: LLMService):
        """Override LLM service dependency."""
        get_llm_service.cache_clear()
        original = get_llm_service
        
        @lru_cache()
        def _override():
            return llm_service
        
        globals()['get_llm_service'] = _override
        
        try:
            yield
        finally:
            globals()['get_llm_service'] = original
            get_llm_service.cache_clear()


# Service health check
def check_services_health() -> dict:
    """
    Check health of all services.
    
    Returns:
        Dict with health status of each service
    """
    health = {
        "vector_store": "unknown",
        "embedding_service": "unknown",
        "llm_service": "unknown",
    }
    
    try:
        vs = get_vector_store()
        count = vs.collection.count()
        health["vector_store"] = "healthy" if count >= 0 else "unhealthy"
    except Exception as e:
        health["vector_store"] = f"unhealthy: {str(e)}"
    
    try:
        es = get_embedding_service()
        # Test embedding generation
        test_emb = es.embed_text("test")
        health["embedding_service"] = "healthy" if len(test_emb) > 0 else "unhealthy"
    except Exception as e:
        health["embedding_service"] = f"unhealthy: {str(e)}"
    
    try:
        ls = get_llm_service()
        health["llm_service"] = f"healthy ({ls.provider.value}/{ls.model})"
    except Exception as e:
        health["llm_service"] = f"unhealthy: {str(e)}"
    
    return health
