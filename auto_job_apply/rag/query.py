"""
Module for querying similar job descriptions from the vector store.
"""
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging

from .retriever import VectorRetriever
from .embedder import TextEmbedder

logger = logging.getLogger(__name__)

class JobSearcher:
    def __init__(
        self, 
        retriever: Optional[VectorRetriever] = None, 
        vector_store_path: Optional[Union[str, Path]] = None,
        embedder: Optional[TextEmbedder] = None
    ):
        """
        Initialize the job searcher with a retriever or path to load from.
        
        Args:
            retriever: Optional pre-initialized vector retriever
            vector_store_path: Optional path to load vector store from
            embedder: Optional pre-initialized text embedder (used if loading from path)
        """
        if retriever is not None and vector_store_path is not None:
            raise ValueError("Cannot specify both retriever and vector_store_path")
            
        if vector_store_path is not None:
            self.retriever = VectorRetriever.load(vector_store_path, embedder=embedder)
        else:
            self.retriever = retriever or VectorRetriever(embedder=embedder)
    
    def find_similar_jobs(
        self, 
        query: str, 
        k: int = 5, 
        threshold: float = 0.6,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Find jobs similar to the query.
        
        Args:
            query: Search query
            k: Number of results to return
            threshold: Minimum similarity score (0-1)
            **kwargs: Additional arguments to pass to the retriever
            
        Returns:
            List of similar jobs with metadata and scores
        """
        if not query.strip():
            logger.warning("Empty query provided")
            return []
            
        return self.retriever.similarity_search(
            query=query,
            k=k,
            score_threshold=threshold,
            **kwargs
        )
    
    def format_results(self, results: List[Dict[str, Any]], 
                      max_length: int = 500) -> str:
        """
        Format search results as a string.
        
        Args:
            results: List of job results from find_similar_jobs
            max_length: Maximum length of job description to include
            
        Returns:
            Formatted string with results
        """
        if not results:
            return "No matching jobs found."
            
        formatted = []
        for i, result in enumerate(results, 1):
            score = result.get('score', 0)
            source = result.get('source', 'Unknown source')
            filename = result.get('filename', 'Unknown file')
            text = result.get('text', '')
            
            if len(text) > max_length:
                text = text[:max_length] + '...'
                
            formatted.append(
                f"{i}. File: {filename} | Score: {score:.4f}\n"
                f"Source: {source}\n"
                f"{text}\n"
            )
            
        return "\n".join(formatted)


def search_jobs(
    query: str,
    vector_store_path: Union[str, Path],
    k: int = 5,
    threshold: float = 0.6,
    format_output: bool = True,
    embedder: Optional[TextEmbedder] = None,
    **kwargs
) -> Union[str, List[Dict[str, Any]]]:
    """
    Helper function to search for similar jobs.
    
    Args:
        query: Search query
        vector_store_path: Path to the vector store
        k: Number of results to return
        threshold: Minimum similarity score (0-1)
        format_output: Whether to format the output as a string
        embedder: Optional pre-initialized text embedder
        **kwargs: Additional arguments to pass to the retriever
        
    Returns:
        Formatted search results or raw results based on format_output
    """
    searcher = JobSearcher(
        vector_store_path=vector_store_path,
        embedder=embedder
    )
    results = searcher.find_similar_jobs(
        query=query, 
        k=k, 
        threshold=threshold,
        **kwargs
    )
    
    if format_output:
        return searcher.format_results(results)
    return results
