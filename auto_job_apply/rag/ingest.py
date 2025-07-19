"""
Module for ingesting job descriptions into the vector store.
"""
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import logging
from tqdm import tqdm

from .retriever import VectorRetriever
from .embedder import TextEmbedder

logger = logging.getLogger(__name__)

class JobIngestor:
    def __init__(self, retriever: Optional[VectorRetriever] = None, 
                 embedder: Optional[TextEmbedder] = None):
        """
        Initialize the job ingestor with a vector retriever.
        
        Args:
            retriever: Optional pre-initialized vector retriever
            embedder: Optional pre-initialized text embedder
        """
        if retriever is None:
            self.retriever = VectorRetriever(embedder=embedder)
        else:
            self.retriever = retriever
    
    def ingest_directory(self, dir_path: Union[str, Path], file_ext: str = '.txt') -> int:
        """
        Ingest all files with the given extension from a directory.
        
        Args:
            dir_path: Path to the directory containing job descriptions
            file_ext: File extension to look for (e.g., '.txt')
            
        Returns:
            Number of jobs ingested
        """
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            raise ValueError(f"{dir_path} is not a valid directory")
            
        job_files = list(dir_path.glob(f'*{file_ext}'))
        if not job_files:
            logger.warning(f"No {file_ext} files found in {dir_path}")
            return 0
            
        job_texts = []
        metadata_list = []
        
        for file_path in tqdm(job_files, desc="Processing job descriptions"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:  # Only add non-empty files
                        job_texts.append(content)
                        metadata_list.append({
                            'source': str(file_path),
                            'filename': file_path.name
                        })
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                continue
                
        # Add jobs to retriever
        if job_texts:
            self.retriever.add_documents(job_texts, metadata_list)
            
        return len(job_texts)
    
    def save_retriever(self, path: Optional[str] = None):
        """
        Save the retriever to disk.
        
        Args:
            path: Path to save the retriever
        """
        if path is None:
            raise ValueError("Path must be provided to save the retriever")
        self.retriever.save(path)
        
    def get_retriever(self) -> VectorRetriever:
        """
        Get the underlying vector retriever.
        
        Returns:
            The VectorRetriever instance
        """
        return self.retriever


def ingest_jobs(
    input_dir: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    file_ext: str = '.txt',
    embedder: Optional[TextEmbedder] = None
) -> VectorRetriever:
    """
    Helper function to ingest jobs from a directory.
    
    Args:
        input_dir: Directory containing job description files
        output_path: Optional path to save the vector store
        file_ext: File extension to look for (e.g., '.txt')
        embedder: Optional pre-initialized text embedder
        
    Returns:
        The populated VectorRetriever instance
    """
    ingestor = JobIngestor(embedder=embedder)
    count = ingestor.ingest_directory(input_dir, file_ext=file_ext)
    
    if count > 0 and output_path:
        ingestor.save_retriever(output_path)
        
    return ingestor.get_retriever()
