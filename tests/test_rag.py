#!/usr/bin/env python3
"""
Test script for the refactored RAG pipeline.
"""
import os
import sys
import logging
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from auto_job_apply.rag.embedder import TextEmbedder
from auto_job_apply.rag.ingest import ingest_jobs
from auto_job_apply.rag.query import search_jobs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    jd_dir = data_dir / "sample_jds"
    vector_store_path = data_dir / "vector_store" / "job_retriever.pkl"
    
    # Create vector store directory if it doesn't exist
    vector_store_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize embedder (can be customized)
    embedder = TextEmbedder(model_name='BAAI/bge-small-en-v1.5')
    
    # Check if vector store exists, if not create it
    if not vector_store_path.exists():
        logger.info("Vector store not found. Creating a new one...")
        retriever = ingest_jobs(
            input_dir=jd_dir,
            output_path=vector_store_path,
            file_ext='.txt',
            embedder=embedder
        )
        logger.info(f"Created and saved vector store with {len(retriever.metadata)} jobs")
    else:
        logger.info("Using existing vector store")
    
    # Interactive search loop
    print("\nJob Description Search")
    print("Enter your search query (or 'quit' to exit):")
    
    while True:
        query = input("\nSearch: ").strip()
        
        if query.lower() in ('quit', 'exit', 'q'):
            break
            
        if not query:
            print("Please enter a search query.")
            continue
            
        # Search for similar jobs
        results = search_jobs(
            query=query,
            vector_store_path=vector_store_path,
            embedder=embedder,
            k=3,  # Return top 3 results
            threshold=0.5,  # Minimum similarity score
            format_output=True
        )
        
        print("\nSearch Results:")
        print("=" * 80)
        print(results)
        print("=" * 80)

if __name__ == "__main__":
    main()
