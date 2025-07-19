"""
OCR Router for processing resume files.
"""
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from pydantic import BaseModel

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from auto_job_apply.ocr.processor import OCRProcessor

# Configure logging
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Constants
PROCESSED_RESUMES_DIR = Path("data/processed_resumes")
PROCESSED_RESUMES_DIR.mkdir(parents=True, exist_ok=True)

# Models
class OCRResult(BaseModel):
    text: str
    file_path: str
    success: bool
    error: Optional[str] = None

@router.post("/process", response_model=OCRResult, summary="Process a resume file")
async def process_resume(
    file: UploadFile = File(..., description="Resume file to process (PDF or image)"),
    lang: str = Query("eng", description="Language for OCR")
):
    """
    Process a resume file (PDF or image) and extract text using OCR.
    """
    try:
        # Validate file type
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ['.pdf', '.png', '.jpg', '.jpeg']:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Please upload a PDF or image file."
            )
        
        # Save uploaded file temporarily
        temp_file = PROCESSED_RESUMES_DIR / f"temp_{file.filename}"
        with open(temp_file, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Initialize OCR processor
        ocr = OCRProcessor(
            output_dir=PROCESSED_RESUMES_DIR,
            lang=lang
        )
        
        # Process the file
        output_file = PROCESSED_RESUMES_DIR / f"{Path(file.filename).stem}.txt"
        result = ocr.process_file(
            file_path=temp_file,
            save_text=True,
            output_file=output_file,
            show_text=False
        )
        
        # Clean up temporary file
        if temp_file.exists():
            temp_file.unlink()
        
        return {
            "text": result.get("text", ""),
            "file_path": str(output_file),
            "success": result.get("success", False),
            "error": result.get("error")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )
