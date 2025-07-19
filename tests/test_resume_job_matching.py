#!/usr/bin/env python3
"""
Resume to Job Matcher

This script processes a resume PDF, extracts text using OCR, and finds the most relevant
job descriptions from the vector store.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from auto_job_apply.rag.embedder import TextEmbedder
from auto_job_apply.rag.retriever import VectorRetriever
from auto_job_apply.rag.ingest import ingest_jobs
from auto_job_apply.ocr.processor import OCRProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_VECTOR_STORE = "data/vector_store/job_descriptions.pkl"
DEFAULT_JD_DIR = "data/sample_jds"
DEFAULT_RESUME_DIR = "data/resumes"
DEFAULT_OUTPUT_DIR = "data/processed_resumes"

def setup_rag_pipeline(vector_store_path: str, jd_dir: str, force_rebuild: bool = False):
    """Set up the RAG pipeline with job descriptions."""
    # Initialize the embedder
    embedder = TextEmbedder()
    
    # Initialize or load the vector store
    if force_rebuild or not os.path.exists(vector_store_path):
        logger.info("Building vector store from job descriptions...")
        retriever = ingest_jobs(
            input_dir=jd_dir,
            output_path=vector_store_path,
            embedder=embedder
        )
    else:
        logger.info("Loading existing vector store...")
        retriever = VectorRetriever.load(vector_store_path, embedder)
    
    return retriever

def process_resume(resume_path: str, output_dir: str) -> str:
    """Process a resume PDF and extract text using OCR."""
    # Initialize OCR processor
    ocr = OCRProcessor(
        output_dir=output_dir,
        lang='eng',  # Language is now required in the config
        config_path=str(Path(__file__).parent.parent / 'auto_job_apply' / 'config' / 'config.yaml')
    )
    
    # Process the resume
    result = ocr.process_file(
        file_path=resume_path,
        save_text=True,
        output_file=str(Path(output_dir) / f"{Path(resume_path).stem}.txt"),
        show_text=False
    )
    
    if not result['success']:
        raise ValueError(f"Failed to process resume: {result.get('error', 'Unknown error')}")
    
    return result['text']

def find_matching_jobs(
    resume_text: str, 
    retriever: VectorRetriever, 
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """Find the top-k most relevant job descriptions for the resume."""
    # Get the most relevant job descriptions
    results = retriever.similarity_search(
        query=resume_text,
        k=top_k,
        score_threshold=0.5  # Minimum similarity score (0-1)
    )
    
    return results

def display_results(resume_path: str, results: List[Dict[str, Any]]):
    """Display the matching job results."""
    print("\n" + "=" * 80)
    print(f"RESUME: {os.path.basename(resume_path)}")
    print("=" * 80)
    
    if not results:
        print("No matching job descriptions found.")
        return
    
    print(f"\nTop {len(results)} Matching Job Descriptions:")
    print("=" * 80)
    
    for i, result in enumerate(results, 1):
        score = result.get('score', 0)
        metadata = result.get('metadata', {})
        content = result.get('text', '')
        
        # Truncate content for display
        preview = (content[:300] + '...') if len(content) > 300 else content
        
        print(f"\n{i}. File: {metadata.get('source', 'N/A')} | Score: {score:.4f}")
        print(f"Source: {metadata.get('source_path', 'N/A')}")
        print("-" * 80)
        print(preview)
        print("-" * 80)

def main():
    parser = argparse.ArgumentParser(description="Match resume to job descriptions.")
    parser.add_argument(
        "--resume", 
        type=str, 
        required=True,
        help="Path to resume PDF file"
    )
    parser.add_argument(
        "--jd-dir", 
        type=str, 
        default=DEFAULT_JD_DIR,
        help=f"Directory containing job descriptions (default: {DEFAULT_JD_DIR})"
    )
    parser.add_argument(
        "--vector-store", 
        type=str, 
        default=DEFAULT_VECTOR_STORE,
        help=f"Path to save/load vector store (default: {DEFAULT_VECTOR_STORE})"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save processed resumes (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--top-k", 
        type=int, 
        default=3,
        help="Number of top matches to return (default: 3)"
    )
    parser.add_argument(
        "--rebuild", 
        action="store_true",
        help="Rebuild the vector store from scratch"
    )
    
    args = parser.parse_args()
    
    try:
        # Set up RAG pipeline
        retriever = setup_rag_pipeline(
            vector_store_path=args.vector_store,
            jd_dir=args.jd_dir,
            force_rebuild=args.rebuild
        )
        
        # Process resume
        logger.info(f"Processing resume: {args.resume}")
        resume_text = process_resume(args.resume, args.output_dir)
        
        # Find matching jobs
        results = find_matching_jobs(resume_text, retriever, args.top_k)
        
        # Display results
        display_results(args.resume, results)
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
