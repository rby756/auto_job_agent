import os
import sys
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "fixtures"

@pytest.fixture
def mock_embedder():
    """Mock for the TextEmbedder class."""
    with patch('sentence_transformers.SentenceTransformer') as mock:
        mock_instance = Mock()
        mock.return_value = mock_instance
        mock_instance.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_instance.get_sentence_embedding_dimension.return_value = 384  # Default dimension for 'all-MiniLM-L6-v2'
        yield mock_instance

@pytest.fixture
def sample_job_descriptions(tmp_path):
    """Create sample job description files for testing."""
    jobs_dir = tmp_path / "job_descriptions"
    jobs_dir.mkdir()
    
    jobs = [
        {
            "filename": "software_engineer.txt",
            "content": "Software Engineer\n\nWe are looking for a skilled Software Engineer..."
        },
        {
            "filename": "data_scientist.txt",
            "content": "Data Scientist\n\nWe are seeking a Data Scientist with ML experience..."
        }
    ]
    
    job_paths = []
    for job in jobs:
        path = jobs_dir / job["filename"]
        path.write_text(job["content"])
        job_paths.append(str(path))
    
    return job_paths

@pytest.fixture
def mock_vector_store():
    """Mock for the VectorStore class."""
    with patch('auto_job_apply.rag.vector_store.VectorStore') as mock:
        mock_instance = Mock()
        mock.return_value = mock_instance
        yield mock_instance
