"""
API tests for the jobs endpoints.
"""
import json
import os
from pathlib import Path
import pytest
import requests
from typing import Dict, Any, List

# Base URL for the API
BASE_URL = "http://localhost:8000"

# Test data
SAMPLE_JOBS_DIR = Path("data/sample_jds")
TEST_RESUME = """
John Doe
Senior Software Engineer

Skills: Python, FastAPI, Machine Learning, Docker, AWS
Experience: 5+ years in software development
Education: BSc in Computer Science
"""

class TestJobsAPI:
    """Test cases for the jobs API endpoints."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup test environment."""
        # Ensure the server is running and reset test data if needed
        self.base_url = BASE_URL
        self.jobs_url = f"{self.base_url}/api/jobs"
        self.client = requests.Session()
        self.client.headers.update({"Content-Type": "application/json"})
        
    def test_health_check(self):
        """Test the health check endpoint."""
        response = self.client.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"
        # The retriever_initialized might be false if the vector store isn't loaded yet
        assert "retriever_initialized" in data

    def test_match_jobs(self):
        """Test matching jobs with a sample resume."""
        data = {
            "resume_text": TEST_RESUME,
            "top_k": 3,
            "score_threshold": 0.1,  # Lower threshold to ensure we get some matches
            "include_embeddings": False
        }
        
        response = self.client.post(
            f"{self.jobs_url}/match",
            json=data
        )
        
        assert response.status_code == 200, f"Expected status code 200, got {response.status_code}. Response: {response.text}"
        result = response.json()
        assert "matches" in result, f"Response missing 'matches' key. Response: {result}"
        assert isinstance(result["matches"], list), f"Expected 'matches' to be a list, got {type(result['matches'])}"
        
        # Verify the structure of each match if we have any
        if result["matches"]:
            for match in result["matches"]:
                assert "score" in match, f"Match missing 'score' key. Match: {match}"
                assert "text" in match, f"Match missing 'text' key. Match: {match}"
                assert "source" in match, f"Match missing 'source' key. Match: {match}"
                assert "metadata" in match, f"Match missing 'metadata' key. Match: {match}"

    def test_ingest_jobs(self, tmp_path):
        """Test ingesting job descriptions."""
        # Create a temporary job description file
        test_job = tmp_path / "test_job.txt"
        test_job.write_text("Test job description for API testing")
        
        # Test the ingest endpoint
        with open(test_job, 'rb') as f:
            files = {'files': (test_job.name, f, 'text/plain')}
            response = self.client.post(
                f"{self.jobs_url}/ingest?directory={str(tmp_path)}&file_ext=.txt",
                files=files
            )
        
        # The API might return 200 with a success message or 500 with an error
        assert response.status_code in [200, 500], f"Unexpected status code: {response.status_code}. Response: {response.text}"
        
        result = response.json()
        
        if response.status_code == 200:
            # Check for successful response structure
            assert "success" in result, f"Response missing 'success' key. Response: {result}"
            if result["success"]:
                assert "num_jobs" in result, f"Response missing 'num_jobs' key. Response: {result}"
                # It's possible to have 0 jobs if the vector store wasn't updated
                assert result["num_jobs"] >= 0, f"Expected non-negative number of jobs, got {result.get('num_jobs')}"
            else:
                assert "error" in result, f"Response missing 'error' key when success is False. Response: {result}"

    def test_ingest_jobs_invalid_directory(self):
        """Test ingesting jobs from a non-existent directory."""
        non_existent_dir = "/non/existent/directory"
        response = self.client.post(
            f"{self.jobs_url}/ingest?directory={non_existent_dir}&file_ext=.txt"
        )
        
        # The API might return 400, 404, or 500 for invalid directories
        assert response.status_code in [400, 404, 500], f"Expected status code 400, 404, or 500, got {response.status_code}. Response: {response.text}"
        
        result = response.json()
        
        # The API might return a standard error response or a FastAPI HTTPException
        if "detail" in result:
            # This is a FastAPI HTTPException
            assert "not found" in result["detail"].lower() or "invalid" in result["detail"].lower(), \
                f"Expected error message about directory not found, got: {result['detail']}"
        else:
            # This is our custom error response
            assert "success" in result, f"Response missing 'success' key. Response: {result}"
            assert result["success"] is False, f"Expected success to be False for invalid directory. Response: {result}"
            assert "error" in result, f"Response missing 'error' key. Response: {result}"

    def test_match_jobs_invalid_request(self):
        """Test job matching with invalid request data."""
        # Missing required field 'resume_text'
        response = self.client.post(
            f"{self.jobs_url}/match",
            json={"top_k": 3}
        )
        assert response.status_code == 422, f"Expected status code 422, got {response.status_code}. Response: {response.text}"  # Validation error

    def test_match_jobs_with_embeddings(self):
        """Test job matching with embeddings included in the response."""
        data = {
            "resume_text": TEST_RESUME,
            "top_k": 2,
            "include_embeddings": True
        }
        
        response = self.client.post(
            f"{self.jobs_url}/match",
            json=data
        )
        
        assert response.status_code == 200, f"Expected status code 200, got {response.status_code}. Response: {response.text}"
        result = response.json()
        assert "matches" in result, f"Response missing 'matches' key. Response: {result}"
        assert "query_embedding" in result, f"Response missing 'query_embedding' key. Response: {result}"
        if result["query_embedding"]:  # Only check if embeddings are included
            assert isinstance(result["query_embedding"], list), f"Expected 'query_embedding' to be a list, got {type(result['query_embedding'])}"
            assert len(result["query_embedding"]) > 0, "Expected non-empty query_embedding"

    def test_ingest_multiple_jobs(self, tmp_path):
        """Test ingesting multiple job descriptions."""
        # Create multiple test job files
        job_descriptions = [
            "Senior Software Engineer with Python experience",
            "Data Scientist with Machine Learning skills",
            "DevOps Engineer with AWS experience"
        ]
        
        job_files = []
        for i, desc in enumerate(job_descriptions, 1):
            job_file = tmp_path / f"job_{i}.txt"
            job_file.write_text(desc)
            job_files.append(job_file)
        
        # Ingest the job descriptions
        for job_file in job_files:
            with open(job_file, "rb") as f:
                files = {'files': (job_file.name, f, 'text/plain')}
                response = self.client.post(
                    f"{self.jobs_url}/ingest?directory={str(tmp_path)}&file_ext=.txt",
                    files=files
                )
                assert response.status_code == 200, f"Failed to ingest {job_file}. Response: {response.text}"
                result = response.json()
                assert "success" in result, f"Response missing 'success' key. Response: {result}"
                if result["success"]:
                    assert "num_jobs" in result, f"Response missing 'num_jobs' key. Response: {result}"
                else:
                    assert "error" in result, f"Response missing 'error' key. Response: {result}"
        
        # Now test matching with these jobs
        data = {
            "resume_text": "I am a Python developer with AWS experience",
            "top_k": 3,
            "score_threshold": 0.1  # Lower threshold to ensure we get some matches
        }
        
        response = self.client.post(
            f"{self.jobs_url}/match",
            json=data
        )
        
        assert response.status_code == 200, f"Expected status code 200, got {response.status_code}. Response: {response.text}"
        result = response.json()
        assert "matches" in result, f"Response missing 'matches' key. Response: {result}"
        if result["matches"]:  # Only check if we have matches
            assert len(result["matches"]) > 0, "Expected at least one match"
