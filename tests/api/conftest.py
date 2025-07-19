"""
Configuration and fixtures for API tests.
"""
import pytest
import time
import requests
from typing import Generator

# Timeout for waiting for the server to be ready (in seconds)
SERVER_TIMEOUT = 30
SERVER_POLL_INTERVAL = 1

@pytest.fixture(scope="session", autouse=True)
def wait_for_server():
    """Wait for the server to be ready before running tests."""
    base_url = "http://localhost:8000"
    health_url = f"{base_url}/health"
    
    start_time = time.time()
    while True:
        try:
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200 and response.json().get("status") == "ok":
                print("\nTest server is ready!")
                break
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            pass
            
        if time.time() - start_time > SERVER_TIMEOUT:
            raise TimeoutError(
                f"Test server did not become ready within {SERVER_TIMEOUT} seconds. "
                "Make sure the FastAPI server is running on http://localhost:8000"
            )
            
        time.sleep(SERVER_POLL_INTERVAL)

@pytest.fixture(scope="module")
def api_client():
    """Create a test client for the API."""
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    return session
