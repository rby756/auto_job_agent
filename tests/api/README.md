# API Tests

This directory contains integration tests for the FastAPI endpoints in the Auto Job Agent application.

## Prerequisites

1. Make sure the FastAPI server is running:
   ```bash
   uvicorn api_app.main:app --reload
   ```

2. Install the required test dependencies:
   ```bash
   pip install pytest requests
   ```

## Running the Tests

To run all API tests:

```bash
pytest tests/api/ -v
```

To run a specific test file:

```bash
pytest tests/api/test_jobs_api.py -v
```

To run a specific test case:

```bash
pytest tests/api/test_jobs_api.py::TestJobsAPI::test_match_jobs -v
```

## Test Structure

- `test_jobs_api.py`: Contains test cases for the jobs endpoints
  - `test_health_check`: Tests the health check endpoint
  - `test_match_jobs`: Tests job matching functionality
  - `test_ingest_jobs`: Tests job description ingestion
  - `test_match_jobs_invalid_request`: Tests error handling for invalid requests
  - `test_match_jobs_with_embeddings`: Tests job matching with embeddings
  - `test_ingest_multiple_jobs`: Tests ingestion of multiple job descriptions

## Test Data

- The tests use a sample resume text defined in `test_jobs_api.py`
- Temporary files are created and cleaned up automatically during testing
- The test server URL is set to `http://localhost:8000/api` by default

## Configuration

- The `conftest.py` file contains fixtures and setup code
- The server health check will wait up to 30 seconds for the server to be ready
- Test timeouts and other configurations can be adjusted in `conftest.py`
