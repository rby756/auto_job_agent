"""
Jobs Router for handling job matching and ingestion.
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from auto_job_apply.rag.retriever import VectorRetriever
from auto_job_apply.rag.embedder import TextEmbedder
from auto_job_apply.rag.ingest import ingest_jobs

# Configure logging
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Constants
VECTOR_STORE_PATH = Path("data/vector_store/job_descriptions.pkl")
VECTOR_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)

# Initialize components
embedder = TextEmbedder()
retriever = None

# Load retriever if vector store exists
if VECTOR_STORE_PATH.exists():
    try:
        retriever = VectorRetriever.load(VECTOR_STORE_PATH, embedder)
        logger.info(f"Loaded vector store with {len(retriever.metadata) if retriever else 0} jobs")
    except Exception as e:
        logger.error(f"Failed to load vector store: {str(e)}")

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
    if not retriever:
        raise HTTPException(
            status_code=503,
            detail="Vector store not initialized. Please ingest job descriptions first."
        )
    
    try:
        # Get matches from the retriever
        results = retriever.similarity_search(
            query=request.resume_text,
            k=request.top_k,
            score_threshold=request.score_threshold
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
            query_embedding = embedder.embed_texts([request.resume_text])[0].tolist()
        
        return {
            "matches": matches,
            "query_embedding": query_embedding
        }
        
    except Exception as e:
        logger.error(f"Error matching jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error matching jobs: {str(e)}")

def _ingest_jobs_background(directory: str, file_ext: str):
    """Background task for ingesting job descriptions."""
    global retriever
    try:
        retriever = ingest_jobs(
            input_dir=directory,
            output_path=str(VECTOR_STORE_PATH),
            file_ext=file_ext,
            embedder=embedder
        )
        return {
            "success": True,
            "message": f"Successfully ingested jobs from {directory}",
            "num_jobs": len(retriever.metadata) if retriever else 0
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
