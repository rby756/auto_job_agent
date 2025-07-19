import pytest
import numpy as np
from unittest.mock import patch, MagicMock, ANY

class TestTextEmbedder:
    @patch('auto_job_apply.rag.embedder.SentenceTransformer')
    def test_initialization(self, mock_model):
        """Test default initialization with default model."""
        from auto_job_apply.rag.embedder import TextEmbedder
        
        # Setup mock model
        mock_instance = MagicMock()
        mock_instance.get_sentence_embedding_dimension.return_value = 384
        mock_model.return_value = mock_instance
        
        # Create embedder
        embedder = TextEmbedder()
        
        # Verify
        assert embedder is not None
        # Don't compare the entire model object, just check it's set
        assert embedder.model is not None
        assert embedder.embedding_dim == 384
        mock_model.assert_called_once_with('BAAI/bge-small-en-v1.5')
    
    def test_embed_texts(self):
        """Test embedding generation for multiple texts."""
        from auto_job_apply.rag.embedder import TextEmbedder
        
        # Setup test data
        texts = ["test text 1", "test text 2"]
        expected_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        
        # Create a mock model with encode method
        mock_model = MagicMock()
        mock_model.encode.return_value = expected_embeddings
        
        # Create instance and inject mock model
        embedder = TextEmbedder()
        embedder.model = mock_model
        
        # Call the method
        embeddings = embedder.embed_texts(texts)
        
        # Verify results
        assert len(embeddings) == 2
        assert isinstance(embeddings, np.ndarray)
        assert np.array_equal(embeddings, expected_embeddings)
        mock_model.encode.assert_called_once_with(
            texts, 
            convert_to_tensor=False,
            show_progress_bar=True
        )
    
    def test_embed_texts_single_string(self):
        """Test embedding generation for a single text string."""
        from auto_job_apply.rag.embedder import TextEmbedder
        
        # Setup test data
        text = "test text"
        expected_embedding = np.array([0.1, 0.2, 0.3])
        
        # Create a mock model with encode method
        mock_model = MagicMock()
        mock_model.encode.return_value = expected_embedding
        
        # Create instance and inject mock model
        embedder = TextEmbedder()
        embedder.model = mock_model
        
        # Call the method
        embedding = embedder.embed_texts(text)
        
        # Verify results
        assert isinstance(embedding, np.ndarray)
        assert np.array_equal(embedding, expected_embedding)
        mock_model.encode.assert_called_once_with(
            text,
            convert_to_tensor=False,
            show_progress_bar=False
        )
    
    def test_model_name_property(self):
        """Test the model_name property."""
        from auto_job_apply.rag.embedder import TextEmbedder
        
        # Create a mock model
        mock_model = MagicMock()
        mock_model._model_name_or_path = "test/model"
        
        # Create instance and inject mock model
        embedder = TextEmbedder()
        embedder.model = mock_model
        
        # Test property
        assert embedder.model_name == "test/model"
    
    def test_model_name_property_unknown(self):
        """Test the model_name property when _model_name_or_path is not available."""
        from auto_job_apply.rag.embedder import TextEmbedder
        
        # Create a mock model without _model_name_or_path
        mock_model = MagicMock()
        # Explicitly remove the attribute
        delattr(mock_model, '_model_name_or_path')
        
        # Create instance and inject mock model
        embedder = TextEmbedder()
        embedder.model = mock_model
        
        # Test property
        assert embedder.model_name == 'unknown'
