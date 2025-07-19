"""
Module for handling text embeddings using sentence-transformers.
"""
from typing import List, Union, Optional
import numpy as np
from sentence_transformers import SentenceTransformer


class TextEmbedder:
    def __init__(self, model_name: str = 'BAAI/bge-small-en-v1.5'):
        """
        Initialize the text embedder with a sentence transformer model.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def embed_texts(self, texts: Union[str, List[str]], 
                   convert_to_tensor: bool = False) -> np.ndarray:
        """
        Generate embeddings for the input text(s).
        
        Args:
            texts: Single text string or list of text strings
            convert_to_tensor: Whether to return PyTorch tensors
            
        Returns:
            Numpy array of embeddings
        """
        return self.model.encode(
            texts, 
            convert_to_tensor=convert_to_tensor,
            show_progress_bar=len(texts) > 1 if isinstance(texts, list) else False
        )
    
    @property
    def model_name(self) -> str:
        """Get the name of the model being used."""
        return self.model._model_name_or_path if hasattr(self.model, '_model_name_or_path') else 'unknown'
    
    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        return self.embedding_dim