"""Services package singletons.

This module creates and exports shared instances used across the application so
routes can import e.g. `from backend.app.services import vector_store`.

Keep instantiation light-weight; heavy models (torch) may be slow to load on import
in development — we import heavy modules lazily and fall back gracefully when
dependencies (sentence-transformers / huggingface) are not available.
"""
from .vector_store import VectorStore
from .pdf_processor import PDFProcessor
from ..core.config import CHROMA_DB_DIR
import logging

logger = logging.getLogger(__name__)

# Instantiate shared services (use defaults; adjust persist_directory as desired)
vector_store = VectorStore(persist_directory=str(CHROMA_DB_DIR), collection_name="annual_reports")

# Try to import and initialize embedding_service lazily. If sentence-transformers or
# huggingface_hub are missing or incompatible, we do not raise here — the app can
# still accept uploads and persist files; embeddings/ingestion will be unavailable
# until the environment is fixed.
try:
    from .embedding_service import EmbeddingService
    try:
        embedding_service = EmbeddingService(model_name="all-MiniLM-L6-v2", device="cpu")
    except Exception as e:
        logger.warning("EmbeddingService initialization failed: %s", e)
        embedding_service = None
except Exception as e:
    logger.warning("Could not import embedding_service: %s", e)
    embedding_service = None

# PDF processor is lightweight
pdf_processor = PDFProcessor()

# High-level ingestion pipeline: only create if embedding_service successfully initialized
if embedding_service is not None:
    try:
        from .ingestion_pipeline import IngestionPipeline

        pipeline = IngestionPipeline(
            vector_store=vector_store,
            embedding_service=embedding_service,
            pdf_processor=pdf_processor,
        )
    except Exception as e:
        logger.warning("Failed to initialize ingestion pipeline: %s", e)
        pipeline = None
else:
    pipeline = None

__all__ = ["vector_store", "embedding_service", "pdf_processor", "pipeline", "VectorStore", "PDFProcessor"]
