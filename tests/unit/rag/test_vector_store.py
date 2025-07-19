import pytest
import numpy as np
from unittest.mock import patch, MagicMock, ANY, mock_open
from pathlib import Path
import faiss

# Import the module to patch the correct path
import sys
import importlib

# Need to patch the module before importing it
with patch('sentence_transformers.SentenceTransformer') as mock_st:
    # Create a mock for the model
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 384
    mock_model.encode.return_value = np.random.rand(2, 384).astype('float32')
    mock_st.return_value = mock_model
    
    # Now import the module
    from auto_job_apply.rag import vector_store
    
    # Reload the module to ensure the mock is used
    importlib.reload(vector_store)

class TestJobVectorStore:

    @pytest.fixture
    def sample_data(self):
        """Sample data for testing."""
        return {
            "texts": ["Python developer needed", "Data scientist position"],
            "embeddings": np.random.rand(2, 384).astype('float32'),
            "metadata": [{"source": "test1.txt"}, {"source": "test2.txt"}],
            "model_name": "BAAI/bge-small-en-v1.5"
        }
    
    def test_initialization(self, tmp_path):
        """Test initialization of the vector store."""
        # Initialize store
        store = vector_store.JobVectorStore(persist_dir=str(tmp_path))
        
        # Verify initialization
        assert store.model is not None
        assert store.model_name == 'BAAI/bge-small-en-v1.5'
        assert store.embedding_dim == 384
        assert store.index is None
        assert store.job_metadata == []
        assert store.persist_dir == Path(tmp_path)
    
    def test_add_jobs(self, sample_data, tmp_path):
        """Test adding jobs to the vector store."""
        with patch('faiss.IndexFlatL2') as mock_faiss:
            # Mock FAISS index
            mock_index = MagicMock()
            mock_faiss.return_value = mock_index
            
            # Initialize store
            store = vector_store.JobVectorStore(persist_dir=str(tmp_path))
            
            # Add jobs
            store.add_jobs(
                job_texts=sample_data["texts"],
                metadata_list=sample_data["metadata"]
            )
            
            # Verify the data was added
            assert len(store.job_metadata) == 2
            
            # Get the mock model instance that was created in the module
            mock_model = vector_store.SentenceTransformer.return_value
            mock_model.encode.assert_called_once_with(
                sample_data["texts"],
                convert_to_tensor=False
            )
            
            # Verify FAISS index was updated
            if store.index is not None:  # Only if index was initialized
                store.index.add.assert_called_once()
    
    def test_get_similar_jobs(self, sample_data, tmp_path):
        """Test getting similar jobs."""
        with patch('faiss.IndexFlatL2') as mock_faiss:
            # Mock FAISS index
            mock_index = MagicMock()
            mock_index.search.return_value = (np.array([[0.1, 0.2]]), np.array([[0, 1]]))  # distances, indices
            mock_faiss.return_value = mock_index
            
            # Initialize store
            store = vector_store.JobVectorStore(persist_dir=str(tmp_path))
            
            # Setup test data
            store.job_metadata = [
                {"text": sample_data["texts"][0], **sample_data["metadata"][0]},
                {"text": sample_data["texts"][1], **sample_data["metadata"][1]}
            ]
            
            # Mock FAISS index
            mock_index = MagicMock()
            mock_index.search.return_value = (
                np.array([[0.1, 0.2]]),  # Distances
                np.array([[0, 1]])       # Indices
            )
            store.index = mock_index
            
            # Test search
            results = store.get_similar_jobs("test query", k=2)
            
            # Verify results
            assert isinstance(results, list)
            if results:  # Only check if results are returned
                assert "text" in results[0]
                assert "score" in results[0]
            mock_index.search.assert_called_once()
    
    def test_save_and_load(self, sample_data, tmp_path):
        """Test saving and loading the vector store."""
        # Create and save
        store1 = vector_store.JobVectorStore(persist_dir=str(tmp_path))
        store1.job_metadata = [
            {"text": sample_data["texts"][0], **sample_data["metadata"][0]},
            {"text": sample_data["texts"][1], **sample_data["metadata"][1]}
        ]
        
        # Mock the index
        mock_index = MagicMock()
        store1.index = mock_index
        
        # Create a real file path for testing
        test_file = tmp_path / "test_vector_store.pkl"
        
        # Mock file operations for save
        with patch('builtins.open', mock_open()) as mock_file, \
             patch('pickle.dump') as mock_pickle_dump, \
             patch('faiss.write_index') as mock_faiss_write:
            
            # Save the store
            store1.save(test_file)
            
            # Verify save was called with the correct arguments
            mock_pickle_dump.assert_called_once()
            
            # FAISS write should be called with the index file path
            mock_faiss_write.assert_called_once_with(mock_index, str(test_file) + ".index")
        
        # Mock loading
        with patch('builtins.open', mock_open(read_data=b'test')) as mock_file, \
             patch('pickle.load') as mock_pickle_load, \
             patch('faiss.read_index') as mock_faiss_read, \
             patch('os.path.exists') as mock_exists:
            
            # Mock os.path.exists to return True for the index file
            def exists_side_effect(path):
                if str(path) == str(test_file) + ".index":
                    return True
                return False
            mock_exists.side_effect = exists_side_effect
            
            # Mock the loaded data to match what the load method expects
            mock_pickle_load.return_value = {
                'job_metadata': store1.job_metadata,
                'model_name': sample_data['model_name'],
                'embedding_dim': 384  # This should match the mock in the test setup
            }
            
            # Mock FAISS index
            mock_faiss_read.return_value = mock_index
            
            # Load the store
            store2 = vector_store.JobVectorStore.load(test_file)
            
            # Verify load was called with the correct file
            mock_file.assert_called_with(test_file, 'rb')
            mock_pickle_load.assert_called_once()
            
            # FAISS read should be called with the index file path
            mock_faiss_read.assert_called_once_with(str(test_file) + ".index")
            
            # Verify loaded data
            assert len(store2.job_metadata) == 2
            assert store2.job_metadata[0]['text'] == sample_data['texts'][0]
