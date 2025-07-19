"""
RAG (Retrieval-Augmented Generation) module for job matching and processing.
"""

from .vector_store import JobVectorStore
from .ingest import JobIngestor, ingest_jobs
from .embedder import TextEmbedder

__all__ = [
    'JobVectorStore',
    'JobIngestor',
    'ingest_jobs',
    'TextEmbedder'
]