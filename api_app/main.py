#!/usr/bin/env python3
"""
FastAPI application for the Auto Job Agent API.
"""
import os
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Auto Job Agent API",
    description="API for processing resumes and matching them with job descriptions",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class HealthCheck(BaseModel):
    status: str
    retriever_initialized: bool

# Health check endpoint
@app.get("/health", response_model=HealthCheck, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "retriever_initialized": False  # Will be updated when retriever is initialized
    }

# Import and include routers
from .routers import ocr, jobs

app.include_router(ocr.router, prefix="/api/ocr", tags=["OCR"])
app.include_router(jobs.router, prefix="/api/jobs", tags=["Jobs"])