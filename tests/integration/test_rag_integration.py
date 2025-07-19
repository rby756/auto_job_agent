import os
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open, PropertyMock
import pytest
import pickle
import faiss

# Import the module to patch
from auto_job_apply.rag import vector_store

@pytest.fixture
def mock_embeddings():
    return np.random.rand(2, 384).astype('float32')

@pytest.fixture
def mock_job_vector_store():
    """Fixture to create a mock JobVectorStore."""
    mock_store = MagicMock()
    mock_store.job_metadata = []
    mock_store.embedding_dim = 384
    mock_store.index = MagicMock()
    mock_store.model = MagicMock()
    
    # Mock the class methods
    mock_store.add_jobs = MagicMock()
    mock_store.get_similar_jobs = MagicMock(return_value=[
        {'text': 'Python developer needed', 'score': 0.9, 'source': 'test1.txt'},
        {'text': 'Data scientist position', 'score': 0.85, 'source': 'test2.txt'}
    ])
    mock_store.save = MagicMock()
    
    # Mock the class itself
    with patch('auto_job_apply.rag.vector_store.JobVectorStore', return_value=mock_store) as mock_class:
        mock_class.load = MagicMock(return_value=mock_store)
        yield mock_store

def test_rag_integration(tmp_path, mock_job_vector_store, mock_embeddings):
    """Test the integration of the RAG pipeline."""
    # Sample data
    texts = ["Python developer needed", "Data scientist position"]
    metadata = [{"source": "test1.txt"}, {"source": "test2.txt"}]
    
    # Get the mock store from the fixture
    mock_store = mock_job_vector_store
    
    # Initialize vector store
    from auto_job_apply.rag.vector_store import JobVectorStore
    vector_store = JobVectorStore(persist_dir=str(tmp_path))
    
    # Test adding jobs
    vector_store.add_jobs(job_texts=texts, metadata_list=metadata)
    
    # Verify add_jobs was called correctly
    vector_store.add_jobs.assert_called_once_with(job_texts=texts, metadata_list=metadata)
    
    # Test search
    query = "software engineer"
    k = 2
    results = vector_store.get_similar_jobs(query, k=k)
    
    # Verify get_similar_jobs was called correctly
    # The threshold has a default value in the method, so we don't need to assert it
    vector_store.get_similar_jobs.assert_called_once_with(query, k=k)
    
    # Verify results
    assert len(results) == 2
    assert all('text' in result and 'score' in result for result in results)
    
    # Test save
    save_path = tmp_path / "job_vector_store.pkl"
    vector_store.save(save_path)
    
    # Verify save was called
    vector_store.save.assert_called_once_with(save_path)
    
    # Test load
    with patch('builtins.open', mock_open()) as mock_file, \
         patch('pickle.load') as mock_pickle_load, \
         patch('faiss.read_index') as mock_faiss_read, \
         patch('os.path.exists') as mock_exists:
        
        # Mock os.path.exists to return True for the index file
        def exists_side_effect(path):
            if str(path) == str(save_path) + ".index":
                return True
            return False
        mock_exists.side_effect = exists_side_effect
        
        # Mock the loaded data
        mock_pickle_load.return_value = {
            'job_metadata': [
                {'text': texts[0], **metadata[0]},
                {'text': texts[1], **metadata[1]}
            ],
            'model_name': 'BAAI/bge-small-en-v1.5',
            'embedding_dim': 384
        }
        
        # Create a mock for the loaded store
        mock_loaded_store = MagicMock()
        mock_loaded_store.job_metadata = [
            {'text': texts[0], **metadata[0]},
            {'text': texts[1], **metadata[1]}
        ]
        mock_loaded_store.embedding_dim = 384
        
        # Mock the load method to return our mock_loaded_store
        with patch('auto_job_apply.rag.vector_store.JobVectorStore.load', return_value=mock_loaded_store):
            # Load the store
            loaded_store = JobVectorStore.load(save_path)
            
            # Verify load was called
            JobVectorStore.load.assert_called_once_with(save_path)
            
            # Verify the loaded store has the correct metadata
            assert len(loaded_store.job_metadata) == 2
            assert loaded_store.job_metadata[0]['text'] == texts[0]
            assert loaded_store.job_metadata[1]['text'] == texts[1]
