"""
Jobs Router for handling job matching and ingestion.
"""
import logging
import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field
import faiss

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from auto_job_apply.rag.vector_store import JobVectorStore
from auto_job_apply.rag.ingest import ingest_jobs

# Configure logging
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Constants
VECTOR_STORE_PATH = Path("data/vector_store/job_descriptions.pkl")
VECTOR_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)

# Initialize vector store
vector_store = None

def load_or_initialize_vector_store():
    """Load the vector store from disk or initialize with sample data if empty."""
    global vector_store
    
    # Try to load existing vector store first
    if VECTOR_STORE_PATH.exists():
        try:
            vector_store = JobVectorStore.load(VECTOR_STORE_PATH)
            if vector_store.job_metadata:  # If we have jobs loaded
                logger.info(f"Loaded vector store with {len(vector_store.job_metadata)} jobs")
                return
            logger.warning("Vector store exists but is empty")
        except Exception as e:
            logger.error(f"Failed to load vector store: {str(e)}")
    
    # If we get here, either the file doesn't exist, loading failed, or it was empty
    logger.info("Initializing a new vector store with default settings")
    vector_store = JobVectorStore(model_name='BAAI/bge-small-en-v1.5')
    
    # Check if sample job descriptions exist
    sample_jds_dir = Path("data/sample_jds")
    if sample_jds_dir.exists() and any(sample_jds_dir.glob("*.txt")):
        try:
            logger.info("Loading sample job descriptions...")
            jobs = ingest_jobs(str(sample_jds_dir), file_ext=".txt")
            if jobs:
                # Clear any existing data
                vector_store.job_metadata = []
                vector_store.index = None
                
                # Add jobs in batches to avoid memory issues
                batch_size = 5  # Smaller batch size to avoid memory issues
                for i in range(0, len(jobs), batch_size):
                    batch = jobs[i:i+batch_size]
                    vector_store.add_jobs(
                        [job["text"] for job in batch],
                        [{"source": job.get("source", "sample"), "filename": job.get("filename", "")} for job in batch]
                    )
                
                # Save the vector store
                vector_store.save(VECTOR_STORE_PATH)
                logger.info(f"Successfully loaded {len(jobs)} sample job descriptions")
        except Exception as e:
            logger.error(f"Failed to load sample job descriptions: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    else:
        logger.warning(f"No sample job descriptions found in {sample_jds_dir}")

# Initialize the vector store
load_or_initialize_vector_store()

# Models
class JobMatch(BaseModel):
    score: float = Field(..., description="Similarity score (0-1)")
    text: str = Field(..., description="Job description text")
    source: str = Field(..., description="Source of the job description")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class JobMatchRequest(BaseModel):
    resume_text: str = Field(..., description="Text content of the resume")
    top_k: int = Field(5, description="Number of top matches to return")
    score_threshold: float = Field(0.5, description="Minimum similarity score (0-1)")
    include_embeddings: bool = Field(False, description="Whether to include query embeddings in the response")

class JobMatchResponse(BaseModel):
    matches: List[JobMatch] = Field(..., description="List of matching jobs")
    query_embedding: Optional[List[float]] = Field(None, description="Embedding of the query text")

class IngestResponse(BaseModel):
    success: bool
    message: str
    num_jobs: int = 0
    error: Optional[str] = None

@router.post("/match", response_model=JobMatchResponse, summary="Find matching jobs for a resume")
async def match_jobs(request: JobMatchRequest):
    """
    Find job descriptions that match the given resume text.
    """
    if not vector_store:
        raise HTTPException(
            status_code=503,
            detail="Vector store not initialized. Please ingest job descriptions first."
        )
    
    try:
        # Get matches from the vector store
        results = vector_store.get_similar_jobs(
            query=request.resume_text,
            k=request.top_k,
            threshold=request.score_threshold
        )
        
        # Format results
        matches = []
        for result in results:
            matches.append(JobMatch(
                score=result.get("score", 0),
                text=result.get("text", ""),
                source=result.get("source", ""),
                metadata={
                    k: v for k, v in result.items() 
                    if k not in ["score", "text", "source"]
                }
            ))
        
        # Get query embedding if needed
        query_embedding = None
        if request.include_embeddings:
            query_embedding = vector_store.model.encode(
                [request.resume_text], 
                convert_to_tensor=False
            )[0].tolist()
        
        return {
            "matches": matches,
            "query_embedding": query_embedding
        }
    except Exception as e:
        logger.error(f"Error in job matching: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def _ingest_jobs_background(directory: str, file_ext: str):
    """Background task for ingesting job descriptions."""
    global vector_store
    try:
        # Ingest jobs from directory
        jobs = ingest_jobs(directory, file_ext=file_ext)
        
        # Create or update vector store
        if vector_store is None:
            vector_store = JobVectorStore()
        
        # Add jobs to vector store
        vector_store.add_jobs(
            job_texts=[job["text"] for job in jobs],
            metadata_list=jobs
        )
        
        # Save vector store
        vector_store.save(VECTOR_STORE_PATH)
        logger.info(f"Ingested {len(jobs)} job descriptions")
        return {
            "success": True,
            "message": f"Ingested {len(jobs)} job descriptions",
            "num_jobs": len(jobs)
        }
    except Exception as e:
        logger.error(f"Error ingesting jobs: {str(e)}")
        return {
            "success": False,
            "message": f"Error ingesting jobs: {str(e)}",
            "error": str(e)
        }

@router.post("/ingest", response_model=IngestResponse, summary="Ingest job descriptions from a directory")
async def ingest_job_descriptions(
    background_tasks: BackgroundTasks,
    directory: str = Query(..., description="Path to directory containing job descriptions"),
    file_ext: str = Query(".txt", description="File extension to look for")
):
    """
    Ingest job descriptions from a directory.
    This runs as a background task.
    """
    if not Path(directory).exists():
        raise HTTPException(status_code=400, detail=f"Directory not found: {directory}")
    
    # Start background task
    background_tasks.add_task(_ingest_jobs_background, directory, file_ext)
    
    return {
        "success": True,
        "message": f"Started ingesting job descriptions from {directory}",
        "num_jobs": 0
    }
